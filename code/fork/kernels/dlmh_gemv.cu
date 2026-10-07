// GLM53_DEC_DLMH: exact FP8 logits of selected column octets of a Marlin FP8 weight, read in place.
//
// dlmh_gemv_kernel is kernels/fp8_gemv.cu's fp8_gemv_kernel copied VERBATIM except for the [glm53-dlmh] lines
// (tests/dlmh/check_kernel_copy.py proves that textually; tests/dlmh/test_dlmh.py U3b checks the outputs against
// fp8_gemv on the full weight, bit for bit): the weight address of each lane and the scale index of each output column: a virtual 64-column n-tile is made of 8 "octets" of the source weight. In the Marlin FP8
// layout the 8 columns {64 * tile + r + 8 i : i = 0..7} of one n-tile occupy exactly the 4 consecutive lanes 4r..4r+3,
// i.e. one contiguous 128-byte line per k-tile, and keep their (w, h) position inside the mma A fragment; only the
// fragment row (lane / 4) changes, which the tensor core treats identically. Per output column: the same k-tiles per
// warp, the same mma sequence, the same fixed-order split-K reduction, scale and bf16 rounding as fp8_gemv.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dlmhg {


__device__ __forceinline__ void ldw(const uint32_t* p, bool pol_on, uint64_t pol, uint32_t (&v)[8]) {
  if (pol_on) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "l"(p), "l"(pol));
  } else {
    asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                 : "l"(p));
  }
}

__device__ __forceinline__ uint32_t ldx(const __nv_bfloat16* p) {
  uint32_t v;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t bf2_mul(uint32_t a, uint32_t b) {
  uint32_t d;
  // exact: b = 2^120 (x2), a holds e4m3 values * 2^-120 (normal or subnormal bf16); fma.rn.bf16x2 keeps subnormals
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

template <bool EXACT>
__device__ __forceinline__ void cvt4(uint32_t v, uint32_t& lo, uint32_t& hi) {
  // bytes b0..b3 = k offsets {0, 8, 1, 9}: lo = (b0, b2) -> k (0, 1), hi = (b1, b3) -> k (8, 9)
  lo = ((v << 8) & 0x80008000u) | ((v << 4) & 0x07F007F0u);
  hi = (v & 0x80008000u) | ((v >> 4) & 0x07F007F0u);
  if constexpr (EXACT) {
    constexpr uint32_t k2p120 = 0x7B807B80u;  // bf16x2 (2^120, 2^120)
    lo = bf2_mul(lo, k2p120);
    hi = bf2_mul(hi, k2p120);
  }
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// stored position of logical column n in a Marlin-permuted per-channel vector (scale_perm_single, blocks of 32)
__device__ __forceinline__ int perm_col(int n) {
  const int c = n & 31;
  return (n & ~31) + 8 * ((c & 7) >> 1) + (c & 1) + 2 * (c >> 3);
}

// resident blocks per SM the register allocation must allow (64K registers / SM): MB <= 2 -> 3 blocks of 8 warps
// (<= 85 registers), MB <= 4 -> 2 blocks (<= 128); 16-warp blocks: 1 block (<= 128); MB = 8 needs ~230 -> 1 block of 8
template <int WARPS, int MB>
struct MinBlocks {
  static constexpr int value = WARPS == 4 ? (MB <= 2 ? 6 : MB <= 4 ? 4 : 2)
                            : WARPS == 8 ? (MB <= 2 ? 3 : MB <= 4 ? 2 : 1) : 1;
};

template <int WARPS, int MB, int U, int KW, bool EXACT>
__global__ void __launch_bounds__(WARPS * 32, (MinBlocks<WARPS, MB>::value)) dlmh_gemv_kernel(
    const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ X, int64_t lda, __nv_bfloat16* __restrict__ Y, int64_t ldy, int M, int N,
    int ntiles, int ktiles, int pol_on, const int64_t* __restrict__ octs, int src_ntiles) {
  constexpr int THREADS = WARPS * 32;
  constexpr int NT = WARPS / KW;                          // n-tiles per block
  constexpr int NCOL = NT * 64;                           // output columns per block
  constexpr int CW = NCOL < THREADS ? NCOL : THREADS;     // threads along columns in the epilogue
  constexpr int TPC = THREADS / CW;                       // threads per column (split the 8 rows of an m block)
  constexpr int CPT = NCOL / CW;                          // columns per thread
  constexpr int MPT = 8 / TPC;                            // rows per thread per m block
  static_assert(TPC <= 8 && 8 % TPC == 0, "epilogue mapping");
  __shared__ float red[WARPS][64][9];  // [warp][n][m] (+1 pad)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int j = warp % NT, kw = warp / NT;
  const int nt0 = blockIdx.x * NT;
  const int nt = nt0 + j;
  const int first = kw;
  const int nk = (nt < ntiles && first < ktiles) ? (ktiles - first + KW - 1) / KW : 0;

  uint64_t pol = 0;
  if (pol_on) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

  // [glm53-dlmh] the only change vs fp8_gemv_kernel: virtual n-tile nt's 8 column octets (lanes 4o..4o+3 = slot o,
  // one contiguous 128-byte line per k-tile) are read from the source weight: slot 8 * nt + o holds octet
  // octs[8 * nt + o] = 8 * tile + r (the columns {64 * tile + r + 8 i}), at word offset tile * 256 + 32 * r;
  // k-stride = the source weight's row length
  const size_t kstride = (size_t)src_ntiles * 256;
  const int64_t oc = octs[(nt < ntiles ? nt : 0) * 8 + (lane >> 2)];
  const uint32_t* wp = Wq + (size_t)first * kstride + (size_t)((oc >> 3) * 256 + 32 * (oc & 7)) + (lane & 3) * 8;
  const size_t wstep = kstride * KW;

  uint32_t wb[U][8];
#pragma unroll
  for (int u = 0; u < U; ++u)
    if (u < nk) ldw(wp + (size_t)u * wstep, pol_on, pol, wb[u]);

  // epilogue columns of this thread; their scale (and bias) are fetched now, off the critical path
  const int ccol = threadIdx.x % CW, cgrp = threadIdx.x / CW;
  float esc[CPT], ebi[CPT];
#pragma unroll
  for (int q = 0; q < CPT; ++q) {
    const int n = nt0 * 64 + ccol + q * CW;
    esc[q] = 0.f;
    ebi[q] = 0.f;
    if (n < N) {
      // [glm53-dlmh] the scale of the virtual column's SOURCE column (the same bf16 value fp8_gemv reads for it)
      const int64_t oc = octs[n >> 3 & ~7 | (n & 7)];   // octet of slot 8 * (n / 64) + n % 8
      const int src_col = (int)((oc >> 3) * 64 + (oc & 7)) + 8 * ((n & 63) >> 3);
      const int p = perm_col(src_col);
      const float sv = __bfloat162float(scales[p]);
      esc[q] = EXACT ? sv * 0x1p-120f : sv;
      if (bias != nullptr) ebi[q] = __bfloat162float(bias[p]);
    }
  }

  const __nv_bfloat16* xp[MB];
  bool xv[MB];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb) {
    const int m = 8 * mb + (lane >> 2);
    xv[mb] = m < M;
    xp[mb] = X + (int64_t)(xv[mb] ? m : 0) * lda + 2 * (lane & 3) + (int64_t)first * 16;
  }
  constexpr int XSTEP = 16 * KW;

  float acc[4][MB][4];
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[w][mb][c] = 0.f;

  uint32_t xb[MB][2];
#pragma unroll
  for (int mb = 0; mb < MB; ++mb) {
    xb[mb][0] = (xv[mb] && nk > 0) ? ldx(xp[mb]) : 0u;
    xb[mb][1] = (xv[mb] && nk > 0) ? ldx(xp[mb] + 8) : 0u;
  }

  for (int i0 = 0; i0 < nk; i0 += U) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int i = i0 + u;
      if (i < nk) {
        uint32_t xn[MB][2];
        const bool more = i + 1 < nk;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          const __nv_bfloat16* p = xp[mb] + (int64_t)(i + 1) * XSTEP;
          xn[mb][0] = (xv[mb] && more) ? ldx(p) : 0u;
          xn[mb][1] = (xv[mb] && more) ? ldx(p + 8) : 0u;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          uint32_t lo0, hi0, lo1, hi1;
          cvt4<EXACT>(wb[u][2 * w], lo0, hi0);
          cvt4<EXACT>(wb[u][2 * w + 1], lo1, hi1);
#pragma unroll
          for (int mb = 0; mb < MB; ++mb) mma_bf16(acc[w][mb], lo0, lo1, hi0, hi1, xb[mb][0], xb[mb][1]);
        }
        if (i + U < nk) ldw(wp + (size_t)(i + U) * wstep, pol_on, pol, wb[u]);
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          xb[mb][0] = xn[mb][0];
          xb[mb][1] = xn[mb][1];
        }
      }
    }
  }

  // ---- block reduction over the KW warps of each n-tile (fixed order kw = 0..KW-1), per 8-row m block
#pragma unroll
  for (int mb = 0; mb < MB; ++mb) {
    if (8 * mb >= M) break;  // uniform
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const int r = 16 * w + (lane >> 2), c = 2 * (lane & 3);
      red[warp][r][c] = acc[w][mb][0];
      red[warp][r][c + 1] = acc[w][mb][1];
      red[warp][r + 8][c] = acc[w][mb][2];
      red[warp][r + 8][c + 1] = acc[w][mb][3];
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < CPT; ++q) {
      const int nn = ccol + q * CW;
      const int jj = nn >> 6, nl = nn & 63;
      const int n = nt0 * 64 + nn;
#pragma unroll
      for (int pm = 0; pm < MPT; ++pm) {
        const int ml = cgrp + TPC * pm;
        const int m = 8 * mb + ml;
        if (m < M && n < N) {
          float v = 0.f;
#pragma unroll
          for (int kq = 0; kq < KW; ++kq) v += red[kq * NT + jj][nl][ml];
          // Marlin's epilogue order: round the scaled sum to bf16, then add the bias and round again
          __nv_bfloat16 o = __float2bfloat16_rn(v * esc[q]);
          if (bias != nullptr) o = __float2bfloat16_rn(__bfloat162float(o) + ebi[q]);
          Y[(int64_t)m * ldy + n] = o;
        }
      }
    }
    __syncthreads();
  }
}


using KernFn = void (*)(const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int64_t,
                        __nv_bfloat16*, int64_t, int, int, int, int, int, const int64_t*, int);

// the configurations production's TABLE uses for the 77440 x 4096 lm_head shard: MB 1 (8, 4) and MB 2 (16, 4); the
// 4-warp blocks with the same KW (= the same per-column arithmetic: k-tiles kw, kw + KW, ..., mma, fixed-order
// split-K sum) give the small rescoring grids more blocks
KernFn pick(int warps, int mb, int kw) {
  if (warps == 4 && mb == 1 && kw == 4) return dlmh_gemv_kernel<4, 1, 2, 4, true>;
  if (warps == 4 && mb == 2 && kw == 4) return dlmh_gemv_kernel<4, 2, 2, 4, true>;
  if (warps == 8 && mb == 1 && kw == 4) return dlmh_gemv_kernel<8, 1, 2, 4, true>;
  if (warps == 16 && mb == 2 && kw == 4) return dlmh_gemv_kernel<16, 2, 2, 4, true>;
  if (warps == 8 && mb == 2 && kw == 4) return dlmh_gemv_kernel<8, 2, 2, 4, true>;
  return nullptr;
}

}  // namespace dlmhg

// out[M, Nv] = x[M, K] @ (virtual columns of wq)^T; scales = the SOURCE weight's Marlin bf16 scale vector [Npad],
// octs int64 [Nv / 8]: the source octet (8 * tile + r) of each virtual slot.
void dlmh_gemv_out(torch::Tensor out, torch::Tensor x, torch::Tensor wq, torch::Tensor vscales, torch::Tensor octs,
                   int64_t nv, int64_t k, int64_t warps, int64_t kw, bool pol) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "dlmh: x");
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= 16 && x.size(1) == k && k % 16 == 0, "dlmh: M 1..16, K multiple of 16");
  TORCH_CHECK(wq.is_cuda() && wq.scalar_type() == at::kInt && wq.is_contiguous() && wq.dim() == 2 &&
              wq.size(0) == k / 16 && wq.size(1) % 256 == 0, "dlmh: Marlin FP8 weight");
  TORCH_CHECK(nv >= 64 && nv % 64 == 0, "dlmh: virtual N multiple of 64");
  TORCH_CHECK(vscales.is_cuda() && vscales.scalar_type() == at::kBFloat16 && vscales.is_contiguous() &&
              vscales.numel() == wq.size(1) / 4, "dlmh: scales = the source Marlin bf16 [Npad]");
  TORCH_CHECK(octs.is_cuda() && octs.scalar_type() == at::kLong && octs.is_contiguous() &&
              octs.numel() == nv / 8, "dlmh: octs int64 [Nv / 8]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == M &&
              out.size(1) == nv && out.stride(1) == 1, "dlmh: out bf16 [M, Nv]");
  TORCH_CHECK(((uintptr_t)wq.data_ptr() & 31) == 0, "dlmh: weight alignment");
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 3) == 0 && (M == 1 || (x.stride(0) % 2 == 0 && x.stride(0) >= k)), "dlmh: x");
  const int mb = (int)((M + 7) / 8);
  auto fn = dlmhg::pick((int)warps, mb, (int)kw);
  TORCH_CHECK(fn != nullptr, "dlmh: configuration not instantiated");
  const int nt = (int)(warps / kw);
  const int ntiles = (int)(nv / 64), ktiles = (int)(k / 16);
  const int src_ntiles = (int)(wq.size(1) / 256);
  dim3 grid((unsigned)((ntiles + nt - 1) / nt));
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  fn<<<grid, (unsigned)(warps * 32), 0, stream>>>(
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vscales.data_ptr()),
      nullptr, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), (int)M, (int)nv, ntiles, ktiles, pol ? 1 : 0,
      reinterpret_cast<const int64_t*>(octs.data_ptr()), src_ntiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dlmh_gemv_out", &dlmh_gemv_out, "exact FP8 logits of column octets of a Marlin FP8 weight (kernels/dlmh_gemv.cu)");
}
