// Large-M (prefill) path for production's FP8 (e4m3, per-output-channel) Marlin-packed linears (fp8_gemv.py,
// GLM53_FP8_LARGE_M):
//
//   Y[M, N] = bf16( (X[M, K] @ W_e4m3[N, K]^T)_fp32 * scale[N] )   (+ bias: bf16(that + bias[n]), Marlin's order)
//
// fp8_gemv.py runs, per chunk of output columns (bounded transient memory):
//   dequant(wq, n0, nc) -> bf16 [nc, K]: rows n0..n0+nc-1 of the e4m3 weight, read in place from the Marlin repack
//     (layout: kernels/fp8_gemv.cu header; tile (kt, nt) of 16 k x 64 n = 256 int32 at (kt * Npad/64 + nt) * 256,
//     int32 j = 8t + 2w + h holds column n = 64nt + 16w + t/4 + 8h at k = 16kt + 2(t%4) + {0, 8, 1, 9}[byte]).
//     Every e4m3 value (subnormals included) is written EXACTLY in bf16 (bit move -> value * 2^-120, then * 2^120 in
//     bf16); the per-channel scale is NOT folded in (that product needs up to 12 significant bits -> would round).
//   then, per chunk of rows: torch.mm(x, w^T, out_dtype=float32) (cuBLAS, fp32 accumulate, fp32 output: every
//     split-K partial is fp32) and scale_cast(): out = bf16(y32 * scale[n]) with one rounding, the value class of
//     Marlin's bf16(sum * s); a bias is added afterwards and rounded again, exactly as Marlin's epilogue does.
// So the only freedom against Marlin is the fp32 summation order.
//
// Why not one cuBLASLt call with the scale in the epilogue: CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
// (a per-row alpha vector) is not offered by any bf16 algorithm of cuBLASLt 13.1 on sm_121 (all 20 algorithm ids
// report pointer-mode mask 3 = host | device scalar; the heuristic returns CUBLAS_STATUS_NOT_SUPPORTED),
// docs/logs/fp8_large_m/probe_cublaslt_alpha_vector.log. A bf16-output GEMM followed by a scale would round twice.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <string>

namespace fp8lm {

__device__ __forceinline__ uint32_t bf2_mul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

// bytes b0..b3 (k offsets {0, 8, 1, 9}) -> lo = bf16x2 (k 0, 1), hi = bf16x2 (k 8, 9), exact e4m3 values
__device__ __forceinline__ void cvt4(uint32_t v, uint32_t& lo, uint32_t& hi) {
  lo = ((v << 8) & 0x80008000u) | ((v << 4) & 0x07F007F0u);
  hi = (v & 0x80008000u) | ((v >> 4) & 0x07F007F0u);
  constexpr uint32_t k2p120 = 0x7B807B80u;  // bf16x2 (2^120, 2^120)
  lo = bf2_mul(lo, k2p120);
  hi = bf2_mul(hi, k2p120);
}

// One thread per output column n (of the chunk) per k-tile: reads the 4 int32 of that column (t%4 = 0..3), writes
// 16 contiguous bf16 (32 bytes, two 16-byte stores). Block = 256 threads = 4 k-tiles x 64 columns (one n-tile).
__global__ void __launch_bounds__(256) dequant_kernel(const uint32_t* __restrict__ wq, __nv_bfloat16* __restrict__ out,
                                                     int64_t ldo, int ntiles_total, int nt0, int n0, int nc,
                                                     int ktiles) {
  const int c = threadIdx.x & 63;          // column within the n-tile
  const int kq = threadIdx.x >> 6;         // k-tile within the block's 4
  const int kt = blockIdx.y * 4 + kq;
  const int nt = nt0 + blockIdx.x;
  if (kt >= ktiles) return;
  const int n = nt * 64 + c;
  const int r = n - n0;
  if (r < 0 || r >= nc) return;
  // column c = 16w + t/4 + 8h  ->  w = c / 16, h = (c / 8) & 1, t/4 = c & 7
  const int w = c >> 4, h = (c >> 3) & 1, tq = c & 7;
  const uint32_t* tile = wq + ((size_t)kt * ntiles_total + nt) * 256;
  uint32_t v[4];
#pragma unroll
  for (int tr = 0; tr < 4; ++tr) v[tr] = __ldg(tile + 8 * (4 * tq + tr) + 2 * w + h);
  // t%4 = tr covers k 2tr, 2tr+1 (lo) and 8+2tr, 9+2tr (hi)
  uint32_t lo[4], hi[4];
#pragma unroll
  for (int tr = 0; tr < 4; ++tr) cvt4(v[tr], lo[tr], hi[tr]);
  uint4* dst = reinterpret_cast<uint4*>(out + (int64_t)r * ldo + (int64_t)kt * 16);
  dst[0] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
  dst[1] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
}


// out[m, n] = bf16(y32[m, n] * alpha[n]) (+ bias: bf16(float(that) + bias[n]), Marlin's epilogue order).
// 8 columns per thread (two float4 loads, one 16-byte store); requires nc % 8 == 0 and 16/32-byte aligned rows.
__global__ void __launch_bounds__(256) scale_cast_kernel(const float* __restrict__ y32, int64_t ld32,
                                                        const float* __restrict__ alpha,
                                                        const __nv_bfloat16* __restrict__ bias,
                                                        __nv_bfloat16* __restrict__ out, int64_t ldo, int M, int nc8) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)M * nc8) return;
  const int m = (int)(i / nc8), c = (int)(i % nc8) * 8;
  const float4* src = reinterpret_cast<const float4*>(y32 + (int64_t)m * ld32 + c);
  const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha + c));
  const float4 a1 = __ldg(reinterpret_cast<const float4*>(alpha + c + 4));
  float4 v0 = __ldcs(src), v1 = __ldcs(src + 1);
  float f[8] = {v0.x * a0.x, v0.y * a0.y, v0.z * a0.z, v0.w * a0.w, v1.x * a1.x, v1.y * a1.y, v1.z * a1.z, v1.w * a1.w};
  __nv_bfloat16 o[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    o[q] = __float2bfloat16_rn(f[q]);
    if (bias != nullptr) o[q] = __float2bfloat16_rn(__bfloat162float(o[q]) + __bfloat162float(bias[c + q]));
  }
  *reinterpret_cast<uint4*>(out + (int64_t)m * ldo + c) = *reinterpret_cast<const uint4*>(o);
}

}  // namespace fp8lm

namespace {

void dequant(torch::Tensor out, torch::Tensor wq, int64_t n0, int64_t nc, int64_t k) {
  TORCH_CHECK(wq.is_cuda() && wq.scalar_type() == at::kInt && wq.is_contiguous() && wq.dim() == 2 &&
                  wq.size(0) * 16 == k && wq.size(1) % 256 == 0,
              "fp8_large_m.dequant: weight must be the Marlin FP8 int32 [K/16, 4*Npad] tensor");
  const int64_t npad = wq.size(1) / 4;
  TORCH_CHECK(n0 >= 0 && nc >= 1 && n0 + nc <= npad, "fp8_large_m.dequant: rows out of range");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == nc &&
                  out.size(1) == k && out.stride(1) == 1 && out.stride(0) % 8 == 0 &&
                  ((uintptr_t)out.data_ptr() & 15) == 0 && out.device() == wq.device(),
              "fp8_large_m.dequant: out must be bf16 [nc, K], unit column stride, 16-byte aligned rows");
  const int ntiles = (int)(npad / 64), ktiles = (int)(k / 16);
  const int nt0 = (int)(n0 / 64), nt1 = (int)((n0 + nc + 63) / 64);
  dim3 grid((unsigned)(nt1 - nt0), (unsigned)((ktiles + 3) / 4));
  const at::cuda::CUDAGuard guard(wq.device());
  fp8lm::dequant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(wq.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0), ntiles, nt0, (int)n0, (int)nc, ktiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void scale_cast(torch::Tensor out, torch::Tensor y32, torch::Tensor alpha, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(y32.is_cuda() && y32.dim() == 2 && y32.scalar_type() == at::kFloat && y32.stride(1) == 1 &&
                  y32.stride(0) % 4 == 0 && ((uintptr_t)y32.data_ptr() & 15) == 0,
              "fp8_large_m.scale_cast: y32 must be fp32 [M, nc], unit column stride, 16-byte aligned rows");
  const int64_t M = y32.size(0), nc = y32.size(1);
  TORCH_CHECK(nc % 8 == 0, "fp8_large_m.scale_cast: nc must be a multiple of 8");
  TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == at::kFloat && alpha.is_contiguous() && alpha.numel() == nc &&
                  ((uintptr_t)alpha.data_ptr() & 15) == 0,
              "fp8_large_m.scale_cast: alpha must be a contiguous 16-byte aligned fp32 [nc]");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 &&
                  out.size(0) == M && out.size(1) == nc && out.stride(0) % 8 == 0 &&
                  ((uintptr_t)out.data_ptr() & 15) == 0,
              "fp8_large_m.scale_cast: out must be bf16 [M, nc] (a column slice is fine), 16-byte aligned rows");
  const __nv_bfloat16* bp = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() && bias->numel() == nc,
                "fp8_large_m.scale_cast: bias must be a contiguous bf16 [nc] in logical column order");
    bp = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
  }
  TORCH_CHECK(y32.device() == out.device() && alpha.device() == out.device(), "fp8_large_m.scale_cast: devices differ");
  const int64_t total = M * (nc / 8);
  if (total == 0) return;
  const at::cuda::CUDAGuard guard(out.device());
  fp8lm::scale_cast_kernel<<<(unsigned)((total + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float*>(y32.data_ptr()), y32.stride(0), reinterpret_cast<const float*>(alpha.data_ptr()),
      bp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), (int)M, (int)(nc / 8));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant", &dequant, "rows n0..n0+nc of the Marlin FP8 weight -> exact bf16 [nc, K] (kernels/fp8_large_m.cu)");
  m.def("scale_cast", &scale_cast, "out = bf16(y32 * alpha) (+ bias, Marlin order)");
  m.attr("VERSION") = 1;
}
