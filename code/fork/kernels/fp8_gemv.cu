// FP8 (e4m3, per-output-channel scale) weight-only small-M GEMM that reads the Marlin-repacked weight in place.
//
//   Y[M, N] = X[M, K] (bf16) @ (W_fp8[N, K] * scale[N])^T  (+ bias[N]),  fp32 accumulation, bf16 output.
//
// Weight layout (vLLM prepare_fp8_layer_for_marlin -> gptq_marlin_repack(num_bits=8), verified bit-exactly by
// tests/test_fp8_gemv.py::layout): int32 tensor [K/16, 4*Npad]; tile (kt, nt) of 16 (k) x 64 (n) is 256 int32 =
// 1 KiB at int32 offset (kt * Npad/64 + nt) * 256. Inside a tile, int32 j = 8*t + 2*w + h (t = 0..31, w = 0..3,
// h = 0..1) holds 4 fp8 bytes (little endian b0..b3) of column n = 64*nt + 16*w + t/4 + 8*h at rows
// k = 16*kt + 2*(t%4) + {0, 8, 1, 9}[b].  This is exactly an mma.m16n8k16 A-fragment when the weight is the A
// operand (rows = n, cols = k): lane t's 32 contiguous bytes = 4 A fragments (w = 0..3) of a 16(n) x 16(k) block.
// Scales: bf16 [1, Npad], permuted within each 32 columns (marlin_permute_scales, scale_perm_single) and multiplied
// by 2^120 (fp8_fused_exponent_bias_into_scales for bf16). Bias (optional): same permutation (marlin_permute_bias),
// no exponent factor.
//
// Numerics: each e4m3 byte is converted to bf16 exactly (bit move -> value * 2^-120, then * 2^120 in bf16, exact
// incl. e4m3 subnormals), bf16 x bf16 products are exact in the tensor core, accumulation is fp32 (mma), the
// per-channel scale (stored * 2^-120, exact) is applied once to the fp32 sum, which is rounded to bf16; a bias is
// then added and rounded again (bf16(bf16(sum * s) + b): Marlin's epilogue order, bit-compatible with it). Partial sums over K (warps of a block, then split blocks) are added in a fixed order (deterministic).
//
// Work split: block = WARPS (4, 8 or 16) warps = NT n-tiles x KW warps along K (NT * KW = WARPS), grid = ceil(Ntiles/NT)
// blocks, no cross-block reduction (no workspace, no atomics). Warp (j, kw) streams n-tile j over k-tiles kw, kw+KW,
// kw+2KW, ... with U 32-byte loads per lane in flight (weights: ld.global.nc.L1::no_allocate.v8, optionally with an
// L2 evict_first policy; activations: ld.global.nc, one k-tile ahead). The KW partial sums of an n-tile are added
// in smem in the fixed order kw = 0..KW-1, then scaled and written as bf16 (only columns n < N: the padded columns
// of a Marlin-padded weight are never written, so no unpad copy is needed).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace fp8g {


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
__global__ void __launch_bounds__(WARPS * 32, (MinBlocks<WARPS, MB>::value)) fp8_gemv_kernel(
    const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ X, int64_t lda, __nv_bfloat16* __restrict__ Y, int64_t ldy, int M, int N,
    int ntiles, int ktiles, int pol_on) {
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

  const size_t kstride = (size_t)ntiles * 256;
  const uint32_t* wp = Wq + (size_t)first * kstride + (size_t)(nt < ntiles ? nt : 0) * 256 + lane * 8;
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
      const int p = perm_col(n);
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
                        __nv_bfloat16*, int64_t, int, int, int, int, int);

// Instantiated configurations (WARPS, KW, U) for every MB = ceil(M / 8) = 1..8:
//   8-warp blocks:  KW in {1, 2, 4, 8}, U = 2
//   16-warp blocks: KW in {4, 8, 16},  U = 2, MB <= 2 only (MB >= 4 spills under the 128-register cap)
//   4-warp blocks:  KW in {1, 2, 4},   U = 2, MB >= 3 only (finer block granularity when a block's registers
//                   limit the SM to 1-2 blocks of 8 warps)
// The per-shape choice is made in fp8_gemv.py (select_config); anything else is rejected here.
template <int WARPS, int MB>
KernFn pick_kw(int kw) {
  if constexpr (WARPS == 4) {
    if constexpr (MB >= 3) switch (kw) {
      case 1: return fp8_gemv_kernel<4, MB, 2, 1, true>;
      case 2: return fp8_gemv_kernel<4, MB, 2, 2, true>;
      case 4: return fp8_gemv_kernel<4, MB, 2, 4, true>;
    }
  } else if constexpr (WARPS == 8) {
    switch (kw) {
      case 1: return fp8_gemv_kernel<8, MB, 2, 1, true>;
      case 2: return fp8_gemv_kernel<8, MB, 2, 2, true>;
      case 4: return fp8_gemv_kernel<8, MB, 2, 4, true>;
      case 8: return fp8_gemv_kernel<8, MB, 2, 8, true>;
    }
  } else if constexpr (MB <= 2) {
    switch (kw) {
      case 4: return fp8_gemv_kernel<16, MB, 2, 4, true>;
      case 8: return fp8_gemv_kernel<16, MB, 2, 8, true>;
      case 16: return fp8_gemv_kernel<16, MB, 2, 16, true>;
    }
  }
  return nullptr;
}

template <int MB>
KernFn pick_w(int warps, int kw) {
  if (warps == 4) return pick_kw<4, MB>(kw);
  if (warps == 8) return pick_kw<8, MB>(kw);
  if (warps == 16) return pick_kw<16, MB>(kw);
  return nullptr;
}

KernFn pick(int warps, int mb, int u, int kw, bool exact) {
  if (u != 2) return nullptr;
  if (!exact)  // numerics study only: Marlin-style conversion (value * 2^-120, scale folds 2^120) without the rescale
    return (warps == 8 && mb == 1 && kw == 8) ? fp8_gemv_kernel<8, 1, 2, 8, false> : nullptr;
  switch (mb) {
    case 1: return pick_w<1>(warps, kw);
    case 2: return pick_w<2>(warps, kw);
    case 3: return pick_w<3>(warps, kw);
    case 4: return pick_w<4>(warps, kw);
    case 5: return pick_w<5>(warps, kw);
    case 6: return pick_w<6>(warps, kw);
    case 7: return pick_w<7>(warps, kw);
    case 8: return pick_w<8>(warps, kw);
  }
  return nullptr;
}

}  // namespace fp8g

namespace {

// Validates everything the kernel assumes; returns the kernel or raises (c10::Error -> RuntimeError in Python).
fp8g::KernFn check_and_pick(const torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& wq,
                            const torch::Tensor& scales, const c10::optional<torch::Tensor>& bias, int64_t n, int64_t k,
                            int64_t warps, int64_t kw, int64_t u, bool exact) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1,
              "fp8_gemv: x must be a 2-D CUDA bf16 tensor with unit column stride");
  const int64_t M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= 64, "fp8_gemv: M must be 1..64, got ", M);
  TORCH_CHECK(k >= 16 && k % 16 == 0 && x.size(1) == k, "fp8_gemv: K must be a multiple of 16 and match x");
  TORCH_CHECK(wq.is_cuda() && wq.scalar_type() == at::kInt && wq.is_contiguous() && wq.dim() == 2 &&
                  wq.size(0) == k / 16 && wq.size(1) % 256 == 0,
              "fp8_gemv: weight must be the Marlin FP8 int32 [K/16, 4*Npad] tensor");
  const int64_t npad = wq.size(1) / 4;
  TORCH_CHECK(n >= 1 && n <= npad && npad - n < 128, "fp8_gemv: N must be within the Marlin padding of Npad");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kBFloat16 && scales.is_contiguous() &&
                  scales.numel() == npad,
              "fp8_gemv: weight_scale must be the Marlin bf16 [1, Npad] tensor");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 &&
                  out.size(0) == M && out.size(1) == n,
              "fp8_gemv: out must be bf16 [M, N] with unit column stride");
  TORCH_CHECK(x.device() == wq.device() && x.device() == scales.device() && x.device() == out.device(),
              "fp8_gemv: tensors on different devices");
  TORCH_CHECK(((uintptr_t)wq.data_ptr() & 31) == 0, "fp8_gemv: weight must be 32-byte aligned");
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 3) == 0 && (M == 1 || (x.stride(0) % 2 == 0 && x.stride(0) >= k)),
              "fp8_gemv: x rows must be 4-byte aligned");
  TORCH_CHECK(k / 16 <= INT32_MAX && npad / 64 <= INT32_MAX, "fp8_gemv: shape too large");
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() && bias->scalar_type() == at::kBFloat16 &&
                    bias->is_contiguous() && bias->numel() == npad,
                "fp8_gemv: bias must be the Marlin-permuted bf16 [Npad] tensor");
  }
  const int mb = (int)((M + 7) / 8);
  auto fn = fp8g::pick((int)warps, mb, (int)u, (int)kw, exact);
  TORCH_CHECK(fn != nullptr, "fp8_gemv: configuration (warps=", warps, ", kw=", kw, ", u=", u, ", MB=", mb,
              ") is not instantiated");
  return fn;
}

void launch(fp8g::KernFn fn, torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& wq,
            const torch::Tensor& scales, const c10::optional<torch::Tensor>& bias, int64_t n, int64_t k, int64_t warps,
            int64_t kw, bool pol) {
  const int64_t npad = wq.size(1) / 4;
  const int nt = (int)(warps / kw);
  const int ntiles = (int)(npad / 64), ktiles = (int)(k / 16);
  dim3 grid((unsigned)((ntiles + nt - 1) / nt));
  const __nv_bfloat16* bp =
      (bias.has_value() && bias->defined()) ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  fn<<<grid, (unsigned)(warps * 32), 0, stream>>>(
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()), bp,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), (int)x.size(0), (int)n, ntiles, ktiles,
      pol ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Y = fp8_gemv(x[M, K] bf16, Marlin weight, Marlin scale, bias or None, N, K, warps, kw, u, pol) -> new bf16 [M, N]
torch::Tensor fp8_gemv(torch::Tensor x, torch::Tensor wq, torch::Tensor scales, c10::optional<torch::Tensor> bias,
                       int64_t n, int64_t k, int64_t warps, int64_t kw, int64_t u, bool pol) {
  auto out = torch::empty({x.size(0), n}, x.options());
  auto fn = check_and_pick(out, x, wq, scales, bias, n, k, warps, kw, u, true);
  launch(fn, out, x, wq, scales, bias, n, k, warps, kw, pol);
  return out;
}

// same into a caller-provided out (benchmarks); exact = false selects the numerics-study kernel
void fp8_gemv_out(torch::Tensor out, torch::Tensor x, torch::Tensor wq, torch::Tensor scales,
                  c10::optional<torch::Tensor> bias, int64_t n, int64_t k, int64_t warps, int64_t kw, int64_t u,
                  bool pol, bool exact) {
  auto fn = check_and_pick(out, x, wq, scales, bias, n, k, warps, kw, u, exact);
  launch(fn, out, x, wq, scales, bias, n, k, warps, kw, pol);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv", &fp8_gemv, "FP8 Marlin-layout small-M GEMM (kernels/fp8_gemv.cu)");
  m.def("fp8_gemv_out", &fp8_gemv_out, "fp8_gemv into a given output");
  m.attr("VERSION") = 1;
}
