"""P1 (docs/OPTIMIZATION.md): what limits the grouped trellis GEMV at 233-235 GB/s when the measured read ceiling is
247-251 GB/s (tests/bw_ceiling.py)?

A load_inline replica of grouped_kernel<NT=4, W=4> with the exact grid (S_cap, N/64, mats*SK), 128 threads, the
real per-expert pointer tables (3 layers x 288 experts, cold weights) and the real route_prep segment tables
(12 routings per configuration). Variants are template instances; every one is timed in the same process with
alternating rounds (median), together with the shipped ext.grouped:

  PW      k tiles per warp (16: gate/up SK=4 and down SK=1; 8: down SK=2)
  D       register ring depth for the weight words (D=1 = the shipped kernel: load, then consume)
  DX      ring depth for the x (activation) fragments
  PF      L2 prefetch: 0 none; d>0 prefetch.global.L2 of k tile i+d; 99 prefetch the warp's whole range at start;
          98 cp.async.bulk.prefetch.L2 of the warp's whole range (one 512-B run per lane)
  COMP    0: loads + XOR sink; 1: decode_tile + mma16816 exactly as exl3.cu
  RED     0: no reduction (sink); 1: the shipped 4-warp shared-memory reduction (16 KiB); 2: LEAN serial reduction
          (one 4 KiB buffer, ((w0+w1)+w2)+w3, last warp stores Z from registers) - same order as 1
  POL     1: weights loaded with an L2 evict_first cache policy (L1::no_allocate)
  PERS    1: persistent (grid = occupancy x SMs), units in segment-fastest order
runtime: dynamic shared memory (pins blocks/SM), carveout, grid order (0 today = segment fastest, 1 n-block
fastest, 2 expert-major with empty segments last).

For COMP=1 and RED>=1 the produced Z is compared with ext.grouped (torch.equal on every row the segments own).
Usage: tests/gpu_run.sh python3 -u tests/bw_pattern.py [phase ...]   phases: load, compute, pin, order, all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LAYERS, ROUTES = 3, 4
MIB = 2 ** 20

SRC = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

struct PA {
  const half* X0; const half* X1; const int64_t* Tp0; const int64_t* Tp1;
  const int* se; const int* s0; const int* sr; const int* ns;
  float* Z; unsigned* sink; int K, N, P, SK, order, gx, gy, gz;
};

__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int POL>
__device__ __forceinline__ uint32_t ldw(const uint32_t* p, uint64_t pol) {
  uint32_t v;
  if (POL == 1) asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.b32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
  else asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldx(const half* p, bool ok) {
  uint32_t v;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n mov.b32 %0, 0;\n @q ld.global.nc.b32 %0, [%1];\n}\n"
               : "=r"(v) : "l"(p), "r"((int)ok));
  return v;
}

template <int PW, int D, int DX, int PF, int COMP, int RED, int POL, int PERS>
__global__ void __launch_bounds__(128) probe(PA a) {
  constexpr int NT = 4, W = 4;
  extern __shared__ __align__(16) unsigned char dsm_raw[];
  int* rows_sh = reinterpret_cast<int*>(dsm_raw);
  float* red = reinterpret_cast<float*>(dsm_raw + 64);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int NTILES = a.N >> 4;
  const int per_split = (a.K >> 4) / a.SK;
  uint64_t pol = 0;
  if (POL == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const int nseg = a.ns[0];
  const int units = PERS ? nseg * a.gy * a.gz : 0;
  uint32_t xs = 0;
  float fs = 0.f;
  for (int u = blockIdx.x;; u += gridDim.x) {
    int s, y, z;
    if (PERS) {
      if (u >= units) break;
      s = u % nseg; const int r = u / nseg; y = r % a.gy; z = r / a.gy;
    } else {
      const int lin = blockIdx.x;
      if (a.order == 0) { s = lin % a.gx; const int r = lin / a.gx; y = r % a.gy; z = r / a.gy; }
      else if (a.order == 1) { y = lin % a.gy; const int r = lin / a.gy; s = r % a.gx; z = r / a.gx; }
      else { s = lin / (a.gy * a.gz); const int r = lin % (a.gy * a.gz); y = r % a.gy; z = r / a.gy; }
      if (s >= nseg) return;
    }
    const int split = z % a.SK, mat = z / a.SK;
    const int e = a.se[s], r0s = a.s0[s], rn = a.sr[s];
    const half* X = mat ? a.X1 : a.X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>((mat ? a.Tp1 : a.Tp0)[e]);
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < rn ? r0s + (int)threadIdx.x : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * a.K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * a.K + 2 * t;
    const int kt0 = split * per_split + warp * PW;
    const int nt0 = y * NT;
    const size_t kst = (size_t)NTILES * 32;
    const uint32_t* tile = T + ((size_t)kt0 * NTILES + nt0) * 32 + lane;
    if (PF == 99) {
      const uint32_t* pb = T + ((size_t)kt0 * NTILES + nt0) * 32 + (lane & 3) * 32;
#pragma unroll
      for (int k = 0; k < PW; k += 8) {
        const int kk = k + (lane >> 2);
        if (kk < PW) asm volatile("prefetch.global.L2 [%0];" :: "l"(pb + kk * kst));
      }
    } else if (PF == 98) {
      if (lane < PW) {
        const uint32_t* pb = T + ((size_t)(kt0 + lane) * NTILES + nt0) * 32;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], 512;" :: "l"(pb));
      }
    }
    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
    uint32_t w[PW][NT];
    uint32_t xa[PW][4];
#pragma unroll
    for (int i = 0; i < PW; ++i) {
      if (i == 0) {
#pragma unroll
        for (int q = 0; q < D - 1; ++q)
          if (q < PW) {
#pragma unroll
            for (int tt = 0; tt < NT; ++tt) w[q][tt] = ldw<POL>(tile + q * kst + tt * 32, pol);
          }
#pragma unroll
        for (int q = 0; q < DX - 1; ++q)
          if (q < PW) {
            const int k = (kt0 + q) * 16;
            xa[q][0] = ldx(x0 + k, r0 >= 0); xa[q][1] = ldx(x1 + k, r1 >= 0);
            xa[q][2] = ldx(x0 + k + 8, r0 >= 0); xa[q][3] = ldx(x1 + k + 8, r1 >= 0);
          }
      }
      if (i + D - 1 < PW) {
        const int q = i + D - 1;
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) w[q][tt] = ldw<POL>(tile + q * kst + tt * 32, pol);
      }
      if (i + DX - 1 < PW) {
        const int q = i + DX - 1;
        const int k = (kt0 + q) * 16;
        xa[q][0] = ldx(x0 + k, r0 >= 0); xa[q][1] = ldx(x1 + k, r1 >= 0);
        xa[q][2] = ldx(x0 + k + 8, r0 >= 0); xa[q][3] = ldx(x1 + k + 8, r1 >= 0);
      }
      if (PF > 0 && PF < 98) {
        if (lane < NT && i + PF < PW)
          asm volatile("prefetch.global.L2 [%0];" :: "l"(T + ((size_t)(kt0 + i + PF) * NTILES + nt0) * 32 + lane * 32));
      }
      if (COMP) {
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) {
          uint32_t b0[2], b1[2];
          decode_tile(w[i][tt], lane, b0, b1);
          mma16816(acc[tt][0], xa[i], b0);
          mma16816(acc[tt][1], xa[i], b1);
        }
      } else {
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) xs ^= w[i][tt];
        xs ^= xa[i][0] ^ xa[i][1] ^ xa[i][2] ^ xa[i][3];
      }
    }
    if (RED == 0) {
      if (COMP) {
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) fs += acc[i][h][c];
      }
    } else if (RED == 1) {
      float (*rd)[16][NT * 16] = reinterpret_cast<float (*)[16][NT * 16]>(red);
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int col = i * 16 + h * 8 + 2 * t;
          rd[warp][g][col] = acc[i][h][0];
          rd[warp][g][col + 1] = acc[i][h][1];
          rd[warp][g + 8][col] = acc[i][h][2];
          rd[warp][g + 8][col + 1] = acc[i][h][3];
        }
      __syncthreads();
      for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float sum = rd[0][row][col];
#pragma unroll
        for (int ww = 1; ww < W; ++ww) sum += rd[ww][row][col];
        a.Z[(((size_t)mat * a.SK + split) * a.P + r) * a.N + nt0 * 16 + col] = sum;
      }
    } else {
      float (*rb)[NT * 16] = reinterpret_cast<float (*)[NT * 16]>(red);
#pragma unroll
      for (int ww = 0; ww < W - 1; ++ww) {
        if (warp == ww) {
#pragma unroll
          for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int col = i * 16 + h * 8 + 2 * t;
              if (ww == 0) {
                rb[g][col] = acc[i][h][0]; rb[g][col + 1] = acc[i][h][1];
                rb[g + 8][col] = acc[i][h][2]; rb[g + 8][col + 1] = acc[i][h][3];
              } else {
                rb[g][col] += acc[i][h][0]; rb[g][col + 1] += acc[i][h][1];
                rb[g + 8][col] += acc[i][h][2]; rb[g + 8][col + 1] += acc[i][h][3];
              }
            }
        }
        __syncthreads();
      }
      if (warp == W - 1) {
        float* zb = a.Z + ((size_t)mat * a.SK + split) * a.P * a.N + nt0 * 16;
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            if (r0 >= 0)
              *reinterpret_cast<float2*>(zb + (size_t)r0 * a.N + col) =
                  make_float2(rb[g][col] + acc[i][h][0], rb[g][col + 1] + acc[i][h][1]);
            if (r1 >= 0)
              *reinterpret_cast<float2*>(zb + (size_t)r1 * a.N + col) =
                  make_float2(rb[g + 8][col] + acc[i][h][2], rb[g + 8][col + 1] + acc[i][h][3]);
          }
      }
    }
    if (!PERS) break;
    __syncthreads();
  }
  if (xs == 0x9e3779b9u || fs == 1.2345e-30f) a.sink[0] = xs ^ __float_as_uint(fs);
}

// id, PW, D, DX, PF, COMP, RED, POL, PERS
#define VARIANTS(X) \
  X(1, 16, 1, 1, 0, 0, 0, 0, 0) X(2, 16, 2, 2, 0, 0, 0, 0, 0) X(3, 16, 3, 2, 0, 0, 0, 0, 0) \
  X(4, 16, 4, 2, 0, 0, 0, 0, 0) X(5, 16, 8, 2, 0, 0, 0, 0, 0) \
  X(6, 16, 1, 1, 2, 0, 0, 0, 0) X(7, 16, 1, 1, 4, 0, 0, 0, 0) X(8, 16, 1, 1, 8, 0, 0, 0, 0) \
  X(9, 16, 1, 1, 99, 0, 0, 0, 0) X(10, 16, 1, 1, 98, 0, 0, 0, 0) \
  X(11, 16, 1, 1, 0, 0, 0, 1, 0) X(12, 16, 4, 2, 0, 0, 0, 1, 0) X(13, 16, 4, 2, 0, 0, 0, 0, 1) \
  X(14, 16, 4, 4, 0, 0, 0, 0, 0) \
  X(20, 16, 1, 1, 0, 1, 1, 0, 0) X(21, 16, 2, 2, 0, 1, 1, 0, 0) X(22, 16, 3, 2, 0, 1, 1, 0, 0) \
  X(23, 16, 4, 2, 0, 1, 1, 0, 0) X(24, 16, 1, 1, 4, 1, 1, 0, 0) X(25, 16, 1, 1, 99, 1, 1, 0, 0) \
  X(26, 16, 2, 2, 99, 1, 1, 0, 0) \
  X(30, 16, 1, 1, 0, 1, 2, 0, 0) X(31, 16, 2, 2, 0, 1, 2, 0, 0) X(32, 16, 3, 2, 0, 1, 2, 0, 0) \
  X(33, 16, 4, 2, 0, 1, 2, 0, 0) X(34, 16, 2, 2, 99, 1, 2, 0, 0) X(35, 16, 4, 4, 0, 1, 2, 0, 0) \
  X(36, 16, 2, 2, 0, 1, 2, 0, 1) X(37, 16, 2, 2, 0, 1, 2, 1, 0) X(38, 16, 1, 1, 99, 1, 2, 0, 0) \
  X(39, 16, 8, 2, 0, 1, 2, 0, 0) \
  X(40, 8, 1, 1, 0, 0, 0, 0, 0) X(41, 8, 2, 2, 0, 0, 0, 0, 0) X(42, 8, 4, 2, 0, 0, 0, 0, 0) \
  X(43, 8, 1, 1, 99, 0, 0, 0, 0) \
  X(50, 8, 1, 1, 0, 1, 1, 0, 0) X(51, 8, 2, 2, 0, 1, 1, 0, 0) X(52, 8, 2, 2, 0, 1, 2, 0, 0) \
  X(53, 8, 4, 2, 0, 1, 2, 0, 0) X(54, 8, 2, 2, 99, 1, 2, 0, 0) X(55, 8, 1, 1, 99, 1, 2, 0, 0)

typedef void (*KFn)(PA);
static KFn kfn(int vid) {
#define X(id, pw, d, dx, pf, comp, red, pol, pers) if (vid == id) return probe<pw, d, dx, pf, comp, red, pol, pers>;
  VARIANTS(X)
#undef X
  TORCH_CHECK(false, "unknown variant ", vid);
  return nullptr;
}
std::vector<int64_t> info(int64_t vid, int64_t dsm, int64_t carve) {
  KFn f = kfn((int)vid);
  if (dsm > 48 * 1024) cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm);
  cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, (int)carve);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, f);
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, 128, (size_t)dsm);
  return {fa.numRegs, (int64_t)fa.sharedSizeBytes, (int64_t)fa.localSizeBytes, nb};
}
void run(int64_t vid, torch::Tensor X0, torch::Tensor X1, torch::Tensor Tp0, torch::Tensor Tp1, torch::Tensor se,
         torch::Tensor s0, torch::Tensor sr, torch::Tensor ns, torch::Tensor Z, torch::Tensor sink, int64_t K, int64_t N,
         int64_t P, int64_t SK, int64_t S_cap, int64_t mats, int64_t order, int64_t dsm, int64_t pers_blocks) {
  PA a;
  a.X0 = (const half*)X0.data_ptr(); a.X1 = (const half*)X1.data_ptr();
  a.Tp0 = Tp0.data_ptr<int64_t>(); a.Tp1 = Tp1.data_ptr<int64_t>();
  a.se = se.data_ptr<int>(); a.s0 = s0.data_ptr<int>(); a.sr = sr.data_ptr<int>(); a.ns = ns.data_ptr<int>();
  a.Z = Z.data_ptr<float>(); a.sink = (unsigned*)sink.data_ptr();
  a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.order = (int)order;
  a.gx = (int)S_cap; a.gy = (int)(N / 64); a.gz = (int)(mats * SK);
  const unsigned blocks = pers_blocks > 0 ? (unsigned)pers_blocks : (unsigned)(a.gx * a.gy * a.gz);
  kfn((int)vid)<<<blocks, 128, (size_t)dsm, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

CPP = ("std::vector<int64_t> info(int64_t, int64_t, int64_t);\n"
       "void run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
       " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t,"
       " int64_t, int64_t, int64_t, int64_t);")

# id -> (PW, D, DX, PF, COMP, RED, POL, PERS)
V = {}
for line in SRC.split("#define VARIANTS(X)")[1].split("typedef")[0].split("X(")[1:]:
    f = [int(v) for v in line.split(")")[0].split(",")]
    V[f[0]] = tuple(f[1:])

RED_DSM = {0: 64, 1: 64 + 16384, 2: 64 + 4096}


def main():
    phases = set(sys.argv[1:]) or {"all"}
    H.gpu_guard(8.0)
    tf = H.load_tf()
    ext = tf.load_ext()
    from torch.utils.cpp_extension import load_inline

    import os

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
    inc = tf._cuda_include_shim()
    t0 = time.time()
    m = load_inline("bw_pattern_v1", cpp_sources=CPP, cuda_sources=SRC, functions=["info", "run"],
                    extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc], verbose=False)
    print(f"probe module built/loaded in {time.time() - t0:.0f} s", flush=True)
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    SMS = props.multi_processor_count
    ck = H.Checks()

    # weights: 3 layers x 288 experts, gate/up [K/16, N/16, 64] and down [N/16, K/16, 64] int16 each (2 MiB)
    layers = []
    for li in range(LAYERS):
        gg = torch.Generator(device=dev).manual_seed(900 + li)
        tabs = {}
        for name, shp in (("g", (NEXP, K // 16, N // 16, 64)), ("u", (NEXP, K // 16, N // 16, 64)),
                          ("d", (NEXP, N // 16, K // 16, 64))):
            w = torch.randint(-32768, 32767, shp, dtype=torch.int16, generator=gg, device=dev)
            tabs[name] = w
            tabs[name + "p"] = torch.tensor([w.data_ptr() + e * w[0].numel() * 2 for e in range(NEXP)],
                                            dtype=torch.int64, device=dev)
        layers.append(tabs)
    Pmax = 64 * TOPK
    X0 = torch.randn(Pmax, K, device=dev).half()
    X1 = torch.randn(Pmax, K, device=dev).half()
    Xd = torch.randn(Pmax, N, device=dev).half()
    Z = torch.zeros(2 * 4 * Pmax * N, device=dev)
    Zr = torch.zeros_like(Z)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)

    def routing(kind, T):
        g = torch.Generator().manual_seed(4242 + T + (0 if kind == "rand" else 17))
        sets, dist, empt = [], 0.0, 0.0
        for li in range(LAYERS):
            for _ in range(ROUTES):
                ids = H.routing_ids(kind, T, NEXP, TOPK, g, "cpu")
                ec = torch.bincount(ids.flatten(), minlength=NEXP + 1).to(torch.int64).to(dev)
                P = T * TOPK
                S = tf.s_cap(P, NEXP)
                t = {k: torch.empty(n_, dtype=torch.int32, device=dev) for k, n_ in
                     (("pe", P), ("se", S), ("s0", S), ("sr", S), ("ns", 1))}
                ext.route_prep(ec, P, 128, t["pe"], t["se"], t["s0"], t["sr"], t["ns"])
                sets.append((layers[li], P, S, t))
                dist += int(torch.unique(ids).numel())
                empt += S - int(t["ns"])
        return sets, dist / len(sets), empt / len(sets)

    def graph_of(fns):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in fns:
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for f in fns:
                f()
        return gr

    def timed(gr, calls, reps):
        gr.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / (reps * calls)

    SHAPES = {  # name -> (mats, K_arg, N_arg, SK, X0, X1, table keys)
        "gu": (2, K, N, 4, X0, X1, ("gp", "up")),
        "down": (1, N, K, 1, Xd, Xd, ("dp", "dp")),
        "down_sk2": (1, N, K, 2, Xd, Xd, ("dp", "dp")),
    }

    def launcher(vid, shape, sets, order=0, dsm=None, carve=-1, pers=0):
        mats, KK, NN, SK, A0, A1, (k0, k1) = SHAPES[shape]
        pw = V[vid][0]
        assert (KK // 16) // SK == 4 * pw, (vid, shape)
        red = V[vid][5]
        dsm = RED_DSM[red] if dsm is None else dsm
        m.info(vid, dsm, carve)                       # sets the attributes (carveout / max dynamic smem)
        pb = 0
        if V[vid][7]:
            pb = m.info(vid, dsm, carve)[3] * SMS
        return [lambda L=L, P=P, S=S, t=t: m.run(vid, A0, A1, L[k0], L[k1], t["se"], t["s0"], t["sr"], t["ns"], Z,
                                                   sink, KK, NN, P, SK, S, mats, order, dsm, pb)
                for (L, P, S, t) in sets]

    def real_fns(shape, sets):
        mats, KK, NN, SK, A0, A1, (k0, k1) = SHAPES[shape]
        return [lambda L=L, P=P, S=S, t=t: ext.grouped(A0, A1, L[k0], L[k1], t["se"], t["s0"], t["sr"], t["ns"], Zr, mats,
                                                         KK, NN, P, SK, 4, 4, S) for (L, P, S, t) in sets]

    def check_equal(vid, shape, sets, fns):
        """Z of the variant vs ext.grouped on the last set (every row its segments own)."""
        mats, KK, NN, SK = SHAPES[shape][:4]
        L, P, S, t = sets[-1]
        Z.fill_(float("nan"))
        Zr.fill_(float("nan"))
        real_fns(shape, sets)[-1]()
        fns[-1]()
        torch.cuda.synchronize()
        rows = t["pe"] >= 0
        a = Z[: mats * SK * P * NN].view(mats * SK, P, NN)[:, rows]
        b = Zr[: mats * SK * P * NN].view(mats * SK, P, NN)[:, rows]
        return torch.equal(a, b)

    def bench(shape, kind, T, cases, rounds=7):
        """cases: list of (label, vid|None, kwargs). Alternating rounds, median per-call us; GB/s of the
        distinct experts' matrices."""
        sets, dist, empt = routing(kind, T)
        mats = SHAPES[shape][0]
        mb = dist * mats * 2 * MIB
        graphs = {}
        for label, vid, kw in cases:
            if vid is None:
                fns = real_fns(shape, sets)
            else:
                fns = launcher(vid, shape, sets, **kw)
                if V[vid][4] and V[vid][5]:
                    eq = check_equal(vid, shape, sets, fns)
                    ck(eq, f"{label} {shape} {kind} T={T}: Z differs from ext.grouped")
            graphs[label] = graph_of(fns)
        # reps so that one measurement is >= ~15 ms
        t1 = timed(graphs[cases[0][0]], len(sets), 1)
        reps = max(2, int(15000 / (t1 * len(sets))))
        times = {lb: [] for lb in graphs}
        for _ in range(rounds):
            for lb, gr in graphs.items():
                times[lb].append(timed(gr, len(sets), reps))
        base = cases[0][0]
        print(f"--- {shape} {kind} T={T}: distinct {dist:.1f}, empty segments {empt:.1f}, {mb / MIB:.0f} MiB/call, "
              f"{rounds} rounds x {reps} replays", flush=True)
        res = {}
        for lb in graphs:
            v = times[lb]
            med = sorted(v)[len(v) // 2]
            ratio = sorted(a / b for a, b in zip(v, times[base]))[len(v) // 2]
            spread = (max(v) - min(v)) / med
            gbs = mb / (med * 1e-6) / 1e9
            res[lb] = (med, gbs, ratio)
            print(f"   {lb:34s} {med:9.1f} us  {gbs:6.1f} GB/s  ({gbs / 250 * 100:5.1f}% of 250)  ratio {ratio:.3f}"
                  f"  spread {spread * 100:4.1f}%", flush=True)
        del graphs
        return res

    def vinfo(vid, dsm=None, carve=-1):
        dsm = RED_DSM[V[vid][5]] if dsm is None else dsm
        r = m.info(vid, dsm, carve)
        return f"REG {r[0]} static-smem {r[1]} LOCAL {r[2]} blocks/SM {r[3]} (dsm {dsm})"

    print("variant resources:")
    for vid in sorted(V):
        dsm = RED_DSM[V[vid][5]] if V[vid][5] else 64 + 16384   # load-only pinned like the real kernel
        print(f"  v{vid:2d} {V[vid]}: {vinfo(vid, dsm)}")
        ck(m.info(vid, dsm, -1)[2] == 0, f"v{vid} uses local memory")
    PIN5 = 64 + 16384                                  # the real kernel's shared memory -> 5 blocks/SM

    configs = [("rand", 1), ("rand", 8), ("rand", 64), ("corr40", 8), ("corr40", 32)]
    if phases & {"load", "all"}:
        for shape in ("gu", "down"):
            for kind, T in configs:
                if shape == "down" and T == 1:
                    shp, ids = "down_sk2", (40, 41, 42, 43)
                else:
                    shp, ids = shape, (1, 2, 3, 4, 5, 14, 6, 7, 8, 9, 10, 11, 12)
                cases = [("real ext.grouped", None, {})]
                cases += [(f"v{v} L {V[v]}", v, {"dsm": PIN5}) for v in ids]
                if shp != "down_sk2":
                    cases.append((f"v13 L PERS {V[13]}", 13, {"dsm": PIN5}))
                bench(shp, kind, T, cases)
    if phases & {"compute", "all"}:
        for shape in ("gu", "down"):
            for kind, T in configs:
                if shape == "down" and T == 1:
                    shp, ids = "down_sk2", (50, 51, 52, 53, 54, 55)
                else:
                    shp, ids = shape, (20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)
                cases = [("real ext.grouped", None, {})]
                cases += [(f"v{v} C {V[v]}", v, {}) for v in ids]
                bench(shp, kind, T, cases)
    if phases & {"pin", "all"}:
        # blocks/SM pinned with dynamic shared memory, at the default and at a small carveout
        for shape in ("gu", "down"):
            for kind, T in (("rand", 8), ("rand", 64)):
                cases = [("real ext.grouped", None, {})]
                for vid in (1, 4, 31):
                    for nb in (4, 5, 6, 8):
                        dsm = 102400 // nb - 1024 - 16
                        if vid == 31 and dsm < RED_DSM[2]:
                            continue
                        for carve in (-1, 50):
                            occ = m.info(vid, dsm, carve)[3]
                            cases.append((f"v{vid} pin{nb} carve{carve} (occ {occ})", vid,
                                          {"dsm": dsm, "carve": carve}))
                bench(shape, kind, T, cases, rounds=5)
    if phases & {"order", "all"}:
        for shape in ("gu", "down"):
            for kind, T in configs:
                shp = "down_sk2" if (shape == "down" and T == 1) else shape
                v1, vc = (40, 50) if shp == "down_sk2" else (1, 20)
                cases = [("real ext.grouped", None, {})]
                for order in (0, 1, 2):
                    cases.append((f"v{v1} L order{order}", v1, {"dsm": PIN5, "order": order}))
                    cases.append((f"v{vc} C order{order}", vc, {"order": order}))
                bench(shp, kind, T, cases)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
