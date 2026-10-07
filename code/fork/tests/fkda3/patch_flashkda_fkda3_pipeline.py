#!/usr/bin/env python3
"""FKDA3 EXPERIMENT (not for production): an opt-in K1->K2 pipeline on top of the fkda3 tree
(tests/fkda3/patch_flashkda_fkda3.py applied first). Built only by `tests/fkda3/build_variant.sh <name> <mod> 2`.

  FKDA3_PIPELINE       compile-time presence (default 1) of an OPT-IN K1->K2 pipeline, engaged per call only when the
                       process environment has FKDA3_PIPELINE=1, no checkpoint is requested, the stream is not
                       capturing, and the full-width K2 grid (H*N blocks) leaves >= 8 SMs free: K2 is launched first on a
                       high-priority side stream and K1 (tile-major grid) on the caller's stream; each K1 block publishes
                       a per-(head, tile) ready flag after its workspace bulk stores fully complete (wait_group 0 +
                       fence.proxy.async + st.release.gpu), K2's load lane acquires it (ld.acquire.gpu +
                       fence.proxy.async) before reading that tile, so K2 consumes K1's output from L2 while K1 runs on the
                       remaining SMs. Same kernels, same arithmetic: results are bit-identical to the sequential path.
                       Flags live after the payload in the workspace (+4 B per head-tile) and are zeroed per call.
                       A K2 wait that exceeds ~2^27 polls traps (CUDA error) instead of hanging.

Measured (tests/fkda3/pipeline_test.py, docs/KDA_FLASHKDA3.md): bit-identical to the sequential path (300-call race
stress), but only -0.15..-0.18 ms per layer (~ -6 ms per 13,824-token chunk): K1 runs ahead of K2 on the free SMs, so
K2's workspace reads still come from DRAM; the gain needs K1 throttled to K2, which is only deadlock-free in a single
co-resident (cooperative) launch. Hazard found on the way: under CUDA lazy loading the first K1 launch while K2 spins
deadlocks (module load waits for the running K2) -- K1 and the delay kernel are force-loaded before K2 here.
Usage: patch_flashkda_fkda3_pipeline.py <fkda3-patched flashkda checkout>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
U = root / "csrc/smxx/utils.cuh"
K1 = root / "csrc/smxx/fwd_kernel1.cuh"
K2 = root / "csrc/smxx/fwd_kernel2.cuh"
LA = root / "csrc/smxx/fwd_launch.cu"
CPP = root / "csrc/flash_kda.cpp"

EDITS = [
    (U, """// [fkda3] fp32 multiply that keeps a subnormal result""",
     """#ifndef FKDA3_PIPELINE
#define FKDA3_PIPELINE 1
#endif

// [fkda3] fp32 multiply that keeps a subnormal result"""),
]
PIPE_EDITS = [
    # ---- workspace: + per-(head, tile) flags after the payload
    (CPP, """    return H * total_tiles * per_tile_bytes;""",
     """#ifndef FKDA3_PIPELINE
#define FKDA3_PIPELINE 1
#endif
#if FKDA3_PIPELINE
    // [fkda3] + the K1->K2 pipeline's per-(head, tile) ready flags (uint32), 128-byte padded
    return H * total_tiles * per_tile_bytes + ((H * total_tiles * 4 + 127) / 128) * 128;
#else
    return H * total_tiles * per_tile_bytes;
#endif"""),
    # ---- K1: signature, tile-major mapping, publish
    (K1, """    cutlass::bfloat16_t* ws_mqk
) {""",
     """    cutlass::bfloat16_t* ws_mqk,
    uint32_t* fk3_flags   // [fkda3] nullptr = sequential launch (unchanged behaviour)
) {"""),
    (K1, """    int global_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;""",
     """    // [fkda3] the pipelined launch uses a tile-major grid (H, tiles) so tiles complete in time order
    int global_tile_idx = fk3_flags ? int(blockIdx.y) : int(blockIdx.x);
    int head_idx = fk3_flags ? int(blockIdx.x) : int(blockIdx.y);"""),
    (K1, """        tma_store_arrive();
        tma_store_wait<0>();
    }
}""",
     """        tma_store_arrive();
        tma_store_wait<0>();
        if (fk3_flags != nullptr) {
            // [fkda3] publish: the bulk stores fully complete (not only their smem reads), ordered before the flag
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(fk3_flags + ws_idx), "r"(1u) : "memory");
        }
    }
}"""),
    # ---- K2: signature + acquire before the tile's workspace reads
    (K2, """    cutlass::bfloat16_t const* ws_mqk
) {""",
     """    cutlass::bfloat16_t const* ws_mqk,
    uint32_t const* fk3_flags   // [fkda3] nullptr = sequential launch (unchanged behaviour)
) {"""),
    (K2, """            int ws_idx = head_idx * total_tiles + tile_base + t;
""",
     """            int ws_idx = head_idx * total_tiles + tile_base + t;
            if (fk3_flags != nullptr) {
                // [fkda3] wait for K1's ready flag of this (head, tile), then order the async-proxy reads after it
                uint32_t const* flag = fk3_flags + ws_idx;
                uint32_t spins = 0;
                while (true) {
                    uint32_t ready;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(ready) : "l"(flag) : "memory");
                    if (ready != 0u) break;
                    __nanosleep(64);
                    if (++spins > (1u << 27)) __trap();
                }
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
"""),
    # ---- launch: helpers
    (LA, """#include "fwd_kernel2.cuh"
""",
     """#include "fwd_kernel2.cuh"

#if FKDA3_PIPELINE
#include <cstdlib>
#include <cstring>
// [fkda3] opt-in K1->K2 pipeline helpers (host)
static bool fkda3_pipeline_requested() {
    const char* e = std::getenv("FKDA3_PIPELINE");
    return e != nullptr && std::strcmp(e, "1") == 0;
}
static bool fkda3_capturing(cudaStream_t s) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(s, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone;
}
struct Fkda3Side { cudaStream_t s = nullptr; cudaEvent_t fork = nullptr; cudaEvent_t join = nullptr; };
// [fkda3] a one-block delay on the caller's stream so K2's blocks (side stream) are resident before K1 floods the SMs
__global__ void fkda3_delay_kernel(unsigned int ns) {
    unsigned long long t0 = clock64();
    unsigned long long cyc = (unsigned long long)ns * 2ull;     // ~2 GHz SM clock; precision does not matter
    while (clock64() - t0 < cyc) { __nanosleep(256); }
}
static unsigned int fkda3_delay_ns() {
    const char* e = std::getenv("FKDA3_PIPELINE_DELAY_NS");
    if (e == nullptr || *e == 0) return 20000u;
    long v = std::strtol(e, nullptr, 10);
    return v < 0 ? 0u : (v > 1000000 ? 1000000u : (unsigned int)v);
}
static Fkda3Side* fkda3_side() {
    static Fkda3Side sides[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return nullptr;
    Fkda3Side& x = sides[dev];
    if (x.s == nullptr) {
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        if (cudaStreamCreateWithPriority(&x.s, cudaStreamNonBlocking, hi) != cudaSuccess ||
            cudaEventCreateWithFlags(&x.fork, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&x.join, cudaEventDisableTiming) != cudaSuccess) {
            x.s = nullptr;
            return nullptr;
        }
    }
    return &x;
}
#endif
"""),
    # ---- launch: decide, zero the flags, fork; K1 becomes a lambda launched before (sequential) or after K2 (pipe)
    (LA, """    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    {
        constexpr int kK1Threads = 128;""",
     """    // [fkda3] the opt-in pipeline (see patch_flashkda_fkda3.py): full-width K2 only, H*N + 8 <= SMs
    uint32_t* fk3_flags = nullptr;
    cudaStream_t k2_stream = stream;
#if FKDA3_PIPELINE
    Fkda3Side* fk3_side = nullptr;
    if (!HasCheckpoint && H * N + 8 <= num_sms && fkda3_pipeline_requested() && !fkda3_capturing(stream)) {
        fk3_side = fkda3_side();
        if (fk3_side != nullptr) {
            cudaFuncAttributes fk3_dattr;
            cudaFuncGetAttributes(&fk3_dattr, fkda3_delay_kernel);   // lazy-loading: load before K2 spins
            fk3_flags = reinterpret_cast<uint32_t*>(ws + n_ht * WS::kPerTile);
            cudaMemsetAsync(fk3_flags, 0, size_t(n_ht) * sizeof(uint32_t), stream);
            cudaEventRecord(fk3_side->fork, stream);
            cudaStreamWaitEvent(fk3_side->s, fk3_side->fork, 0);
            k2_stream = fk3_side->s;
        }
    }
#endif
    const bool fk3_pipe = fk3_flags != nullptr;
    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    auto fk3_launch_k1 = [&](bool fk3_load_only) {
        constexpr int kK1Threads = 128;"""),
    (LA, """        dim3 grid_k1(total_tiles, H);""",
     """        if (fk3_load_only) {
            // [fkda3] force-load K1 now: under CUDA lazy loading a first K1 launch while K2 spins on its flags would
            // wait for the running K2 to finish (module load) -> deadlock (measured: trap after the spin limit)
            cudaFuncAttributes fk3_attr;
            cudaFuncGetAttributes(&fk3_attr, kernel1);
            return;
        }
        dim3 grid_k1 = fk3_pipe ? dim3(H, total_tiles) : dim3(total_tiles, H);"""),
    (LA, """            A_log_ptr, gate_scale,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
        );
    }
#endif
""",
     """            A_log_ptr, gate_scale,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, fk3_flags
        );
    };
    if (!fk3_pipe) fk3_launch_k1(false);
    else fk3_launch_k1(true);     // load K1 (and set its smem attribute) before K2 is launched
#endif
"""),
    (LA, """        if (vsplit_blocks <= 2 * num_sms) {""",
     """        if (!fk3_pipe && vsplit_blocks <= 2 * num_sms) {"""),
    (LA, """                    T_total, H, N, cu_seqlens_ptr, total_tiles,
                    ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk);
                return;""",
     """                    T_total, H, N, cu_seqlens_ptr, total_tiles,
                    ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr);
                return;"""),
    (LA, """        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_initial_state,
            tma_store_final_state,
            tma_store_out,
            out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
            T_total, H, N, cu_seqlens_ptr, total_tiles,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
        );
    }
#endif
}""",
     """        kernel2<<<grid_k2, block_k2, smem_size_k2, k2_stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_initial_state,
            tma_store_final_state,
            tma_store_out,
            out_ptr, checkpoint_state_ptr, checkpoint_offsets_ptr,
            T_total, H, N, cu_seqlens_ptr, total_tiles,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, fk3_flags
        );
    }
#endif
#if FKDA3_PIPELINE
    if (fk3_pipe) {
        // K2 is queued on the side stream first (its H*N blocks land first); K1 then runs on the caller's stream
        // and the caller's stream joins the side stream, so everything after this op is ordered after both.
        const unsigned int fk3_delay = fkda3_delay_ns();
        if (fk3_delay > 0) fkda3_delay_kernel<<<1, 32, 0, stream>>>(fk3_delay);
#if BLOCK_LEVEL_K1 >= 0
        fk3_launch_k1(false);
#endif
        cudaEventRecord(fk3_side->join, fk3_side->s);
        cudaStreamWaitEvent(stream, fk3_side->join, 0);
    }
#endif
}"""),
]
EDITS += PIPE_EDITS


def main():
    texts = {p: p.read_text() for p in (U, K1, K2, LA, CPP)}
    if not any("[fkda3] precision switches" in t for t in texts.values()):
        sys.exit("not an fkda3 tree: apply tests/fkda3/patch_flashkda_fkda3.py first")
    if any("FKDA3_PIPELINE" in t for t in texts.values()):
        sys.exit("already patched (FKDA3_PIPELINE present)")
    for p, old, new in EDITS:
        n = texts[p].count(old)
        if n != 1:
            sys.exit(f"anchor count {n} != 1 in {p.name}: {old.splitlines()[0]!r}")
        texts[p] = texts[p].replace(old, new)
    for p, t in texts.items():
        p.write_text(t)
    print(f"patched {len(EDITS)} anchors (fkda3 pipeline experiment) in {root}")


if __name__ == "__main__":
    main()
