#!/usr/bin/env python3
"""FKDA3: two precision edits on top of the fkda2 tree (FlashKDA 17a037d + tests/fkda2/patch_flashkda_precision.py),
each behind a compile-time switch whose 0 value leaves the fkda2 source path untouched (SASS-identical to the fkda2
build when both are 0 -- checked by tests/fkda3/check_sass.sh).

  FKDA3_DECAY_NOFTZ    K1 decay pass (the FKDA2_FP32_DECAY branch): the fp32 products q*e*scale, k*e, k*e^-1,
                       k*e^-1*gt use mul.rn.f32 WITHOUT .ftz. Under --use_fast_math the fkda2 products flush a
                       subnormal result to 0, which the stock bf16-operator path did not (bf16 keeps subnormals): in a
                       tile whose 16 gates all sit at the lower bound (in-tile log2 decay < ~-112) the last row's
                       decayed q/k lose their small elements (tests/fkda3/decay_sweep.py: tile position 15 error
                       1.1e-2 vs 3.8e-3; the stock build and the Triton chain do not show it). Normal values are
                       rounded identically (ftz only differs on subnormals).
  FKDA3_FP32_STATE_IO  K2: the resident fp32 recurrent state is initialised from the fp32 initial state (not from its
                       bf16 rounding) and the fp32 final state is written from the fp32 resident state (not from a
                       bf16 rounding of it). 17a037d/fkda2 keep the state fp32 only INSIDE a call: every call boundary
                       (each 13,824-token prefill chunk, and the prefill->decode hand-off) rounds the carried state to
                       bf16 (tests/fkda3/state_identity_probe.py: final == bf16(initial) exactly, 1.66e-3 rel; the
                       Triton chain carries it exactly). The fp32 smem buffer shares a union with the pipeline stages,
                       so the init read is ordered before the load warp's first TMA by named barrier 1 (MMA arrive,
                       load warp sync) and the final write after the store warp's last TMA store by named barrier 2
                       (store warp arrive, MMA sync).
Usage: patch_flashkda_fkda3.py <fkda2-patched flashkda checkout>   (refuses a tree without [fkda2] or with [fkda3])
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
U = root / "csrc/smxx/utils.cuh"
K1 = root / "csrc/smxx/fwd_kernel1.cuh"
K2 = root / "csrc/smxx/fwd_kernel2.cuh"

EDITS = [
    (U, """// [fkda2] the activation used for the gate and beta""",
     """// [fkda3] precision switches (tests/fkda3/patch_flashkda_fkda3.py), default ON
#ifndef FKDA3_DECAY_NOFTZ
#define FKDA3_DECAY_NOFTZ 1
#endif
#ifndef FKDA3_FP32_STATE_IO
#define FKDA3_FP32_STATE_IO 1
#endif

// [fkda3] fp32 multiply that keeps a subnormal result (no .ftz even under --use_fast_math)
__device__ __forceinline__ float fkda3_mul(float a, float b) {
#if FKDA3_DECAY_NOFTZ
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
#else
    return a * b;
#endif
}

// [fkda2] the activation used for the gate and beta"""),
    (K1, """                    r_qd(0, v) = BF16(qf * ef * scale);
                    r_kd(0, v) = BF16(kf * ef);""",
     """                    r_qd(0, v) = BF16(fkda3_mul(fkda3_mul(qf, ef), scale));
                    r_kd(0, v) = BF16(fkda3_mul(kf, ef));"""),
    (K1, """                    r_ki(0, v) = BF16(kf * ei);
                    r_kr(0, v) = BF16(kf * ei * reg_gt[pass][tile_idx][v]);""",
     """                    r_ki(0, v) = BF16(fkda3_mul(kf, ei));
                    r_kr(0, v) = BF16(fkda3_mul(fkda3_mul(kf, ei), reg_gt[pass][tile_idx][v]));"""),
    # ---- K2 init: resident fp32 state from the fp32 initial state
    (K2, """                ResidentStateFragment state_bf16;
                copy(
                    resident_load_c,
                    resident_thr_load_c.partition_S(state_block),
                    resident_thr_load_c.retile_D(state_bf16));
                cute::transform(state_bf16, resident_state[bi][m], ToF32{});
            }
        }
""",
     """#if FKDA3_FP32_STATE_IO
                if constexpr (HasStateIn && StateFP32) {
                    // [fkda3] (k, v) coordinates of this thread's C-fragment elements; the fp32 buffer is [VD, D]
                    using FP32StateSmemLayoutI = typename Layouts::FP32StateSmemLayout;
                    Tensor fp32_state = make_tensor(
                        make_smem_ptr(reinterpret_cast<float const*>(shared_storage.state_fp32_buf)),
                        FP32StateSmemLayoutI{});
                    Tensor coord_block = local_tile(
                        make_identity_tensor(make_shape(Int<D>{}, Int<VD>{})),
                        make_shape(Int<16>{}, Int<16>{}),
                        make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
                    Tensor coords = resident_thr_mma.partition_C(coord_block);
                    CUTE_STATIC_ASSERT_V(size(coords) == size(resident_state[bi][m]));
                    #pragma unroll
                    for (int i = 0; i < size(coords); ++i) {
                        resident_state[bi][m](i) = fp32_state(get<1>(coords(i)), get<0>(coords(i)));
                    }
                    continue;
                }
#endif
                ResidentStateFragment state_bf16;
                copy(
                    resident_load_c,
                    resident_thr_load_c.partition_S(state_block),
                    resident_thr_load_c.retile_D(state_bf16));
                cute::transform(state_bf16, resident_state[bi][m], ToF32{});
            }
        }
#if FKDA3_FP32_STATE_IO
        if constexpr (HasStateIn && StateFP32) {
            // [fkda3] the fp32 buffer is now in registers: release the load warp (union with the pipeline stages)
            cutlass::arch::NamedBarrier(kComputeThreads + kWarpSize, 1).arrive();
        }
#endif
"""),
    # ---- K2: the load warp waits for the init read before its first TMA into the union
    (K2, """    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {""",
     """#if FKDA3_FP32_STATE_IO
    if constexpr (HasStateIn && StateFP32) {
        if (warp_role == WarpRole::LOAD_QKG) {
            cutlass::arch::NamedBarrier(kComputeThreads + kWarpSize, 1).arrive_and_wait();
        }
    }
#endif
    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {"""),
    # ---- K2: final fp32 state from the fp32 resident registers (MMA side, after the store warp is done)
    (K2, """            ++load_read;
            ++out_write;
        }
    }

    if (warp_role == WarpRole::STORE && lane_predicate) {""",
     """            ++load_read;
            ++out_write;
        }
#if FKDA3_FP32_STATE_IO
        if constexpr (HasStateOut && StateFP32) {
            // [fkda3] wait until the store warp's last TMA store has read the output stages (union), then write
            // the fp32 resident state into the fp32 buffer the final-state TMA store reads
            cutlass::arch::NamedBarrier(kComputeThreads + kWarpSize, 2).arrive_and_wait();
            using FP32StateSmemLayoutO = typename Layouts::FP32StateSmemLayout;
            Tensor fp32_state = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)), FP32StateSmemLayoutO{});
            #pragma unroll
            for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    Tensor coord_block = local_tile(
                        make_identity_tensor(make_shape(Int<D>{}, Int<VD>{})),
                        make_shape(Int<16>{}, Int<16>{}),
                        make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
                    Tensor coords = resident_thr_mma.partition_C(coord_block);
                    #pragma unroll
                    for (int i = 0; i < size(coords); ++i) {
                        fp32_state(get<1>(coords(i)), get<0>(coords(i))) = resident_state[bi][m](i);
                    }
                }
            }
        }
#endif
    }

    if (warp_role == WarpRole::STORE && lane_predicate) {"""),
    (K2, """            tma_store_arrive();
            tma_store_wait<0>();
        }
    }

    if constexpr (HasStateOut && StateFP32) {""",
     """            tma_store_arrive();
            tma_store_wait<0>();
        }
    }
#if FKDA3_FP32_STATE_IO
    if constexpr (HasStateOut && StateFP32) {
        if (warp_role == WarpRole::STORE) {
            __syncwarp();   // the elected lane has finished its last output TMA store (wait_group.read 0)
            cutlass::arch::NamedBarrier(kComputeThreads + kWarpSize, 2).arrive();
        }
    }
#endif

    if constexpr (HasStateOut && StateFP32) {"""),
    (K2, """        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, VD, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);""",
     """#if !FKDA3_FP32_STATE_IO
        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, VD, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);
#endif"""),
]


def main():
    texts = {p: p.read_text() for p in (U, K1, K2)}
    if not any("[fkda2]" in t for t in texts.values()):
        sys.exit("not an fkda2 tree ([fkda2] marker missing): apply tests/fkda2/patch_flashkda_precision.py first")
    if any("[fkda3]" in t for t in texts.values()):
        sys.exit("already patched ([fkda3] marker present)")
    for p, old, new in EDITS:
        n = texts[p].count(old)
        if n != 1:
            sys.exit(f"anchor count {n} != 1 in {p.name}: {old.splitlines()[0]!r}")
        texts[p] = texts[p].replace(old, new)
    for p, t in texts.items():
        p.write_text(t)
    print(f"patched {len(EDITS)} anchors (fkda3) in {root}")


if __name__ == "__main__":
    main()
