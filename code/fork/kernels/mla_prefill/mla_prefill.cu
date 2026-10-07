// Torch binding of the GLM-5.3 sparse-MLA prefill kernel (mla_prefill_kernel.cuh; docs/MLA_PREFILL.md).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include "mla_prefill_kernel.cuh"

namespace glm53_mla {

// ---------------------------------------------------------------------------------------------------- host
static uint16_t bf16_bits(float f) {
  __nv_bfloat16 b = __float2bfloat16_rn(f);
  return *reinterpret_cast<uint16_t*>(&b);
}

template <typename K>
static void set_smem(K kernel, int bytes, bool& done) {
  if (!done) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess,
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed");
    done = true;
  }
}

static int sm_count() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
  }
  return n;
}

static void launch(const Params& p, bool extra, int variant, int T, cudaStream_t stream) {
  static bool cfg[3][2] = {};
  if (variant == 4) {                       // persistent v3: one CTA per SM walks the tokens
    const int grid = T < sm_count() ? T : sm_count();
    if (extra) {
      set_smem(mla_prefill_v3_kernel<true, true>, V3_SMEM_TOTAL, cfg[2][1]);
      mla_prefill_v3_kernel<true, true><<<grid, V3_THREADS, V3_SMEM_TOTAL, stream>>>(p);
    } else {
      set_smem(mla_prefill_v3_kernel<false, true>, V3_SMEM_TOTAL, cfg[2][0]);
      mla_prefill_v3_kernel<false, true><<<grid, V3_THREADS, V3_SMEM_TOTAL, stream>>>(p);
    }
  } else if (variant == 3) {
    if (extra) {
      set_smem(mla_prefill_v3_kernel<true, false>, V3_SMEM_TOTAL, cfg[1][1]);
      mla_prefill_v3_kernel<true, false><<<T, V3_THREADS, V3_SMEM_TOTAL, stream>>>(p);
    } else {
      set_smem(mla_prefill_v3_kernel<false, false>, V3_SMEM_TOTAL, cfg[1][0]);
      mla_prefill_v3_kernel<false, false><<<T, V3_THREADS, V3_SMEM_TOTAL, stream>>>(p);
    }
  } else {
    if (extra) {
      set_smem(mla_prefill_kernel<true>, SMEM_TOTAL, cfg[0][1]);
      mla_prefill_kernel<true><<<T, THREADS, SMEM_TOTAL, stream>>>(p);
    } else {
      set_smem(mla_prefill_kernel<false>, SMEM_TOTAL, cfg[0][0]);
      mla_prefill_kernel<false><<<T, THREADS, SMEM_TOTAL, stream>>>(p);
    }
  }
}

void run(torch::Tensor q, torch::Tensor kv, torch::Tensor slots, torch::Tensor valid, torch::Tensor out,
         double sm_scale, double kv_scale, int64_t variant) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.dim() == 3 && q.size(1) == NH && q.size(2) == D,
              "q must be [T, 32, 512] bf16");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 &&
                  (reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0,
              "q needs a unit inner stride, 16-byte aligned rows");
  TORCH_CHECK(kv.is_cuda() && kv.element_size() == 1 && kv.size(-1) == D && kv.is_contiguous(),
              "kv must be a contiguous 1-byte [..., 512] tensor");
  TORCH_CHECK(slots.is_cuda() && slots.dtype() == torch::kInt32 && slots.dim() == 2 && slots.stride(1) == 1,
              "slots must be int32 [T, width] with unit inner stride");
  TORCH_CHECK(valid.is_cuda() && valid.dtype() == torch::kInt32 && valid.is_contiguous(), "valid must be int32");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.sizes() == q.sizes() && out.stride(2) == 1 &&
                  out.stride(0) % 4 == 0 && out.stride(1) % 4 == 0 &&
                  (reinterpret_cast<uintptr_t>(out.data_ptr()) & 7) == 0,
              "out must be [T, 32, 512] bf16 with a unit inner stride, 8-byte aligned");
  const int T = (int)q.size(0);
  TORCH_CHECK(slots.size(0) >= T && valid.numel() >= T, "slots/valid shorter than q");
  if (T == 0) return;
  TORCH_CHECK((reinterpret_cast<uintptr_t>(kv.data_ptr()) & 15) == 0, "kv must be 16-byte aligned");
  TORCH_CHECK(slots.stride(0) == slots.size(1), "slots rows must be contiguous");
  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  p.slots = slots.data_ptr<int32_t>();
  p.valid = valid.data_ptr<int32_t>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.q_st = q.stride(0);
  p.q_sh = q.stride(1);
  p.o_st = out.stride(0);
  p.o_sh = out.stride(1);
  p.T = T;
  p.width = (int)slots.size(1);
  p.scale_log2 = (float)(sm_scale * 1.4426950408889634);
  const float s = (float)kv_scale;
  TORCH_CHECK(s > 0.f && std::isfinite(s), "kv_scale must be finite and positive");
  int e2;
  const float m = std::frexp(s, &e2);
  const bool pow2 = (m == 0.5f);
  const float k120 = std::ldexp(1.0f, 120);
  // 2^120 * s must stay a normal bf16 for the one-multiply path; otherwise two multiplies (FA2's own rounding).
  const bool extra = !(pow2 && std::isfinite(k120 * s) && k120 * s >= 1e-30f);
  const uint16_t mb = bf16_bits(extra ? k120 : k120 * s);
  p.mul2 = (uint32_t)mb | ((uint32_t)mb << 16);
  const uint16_t sb = bf16_bits(s);
  p.scale2 = (uint32_t)sb | ((uint32_t)sb << 16);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(variant >= 2 && variant <= 4, "variant must be 2, 3 or 4");
  launch(p, extra, (int)variant, T, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Unit probe: the in-register e4m3 -> bf16 conversion for all 256 codes (x scale), as bf16 bits.
__global__ void conv_probe_kernel(uint16_t* out, uint32_t mul2, uint32_t scale2, int extra) {
  const int c = threadIdx.x;                 // 0..255
  const uint32_t t = (uint32_t)c << 8;       // code in the high byte of the low half
  const uint32_t v = extra ? e4m3_hi_to_bf16x2<true>(t, mul2, scale2) : e4m3_hi_to_bf16x2<false>(t, mul2, scale2);
  out[c] = (uint16_t)(v & 0xffffu);
}

torch::Tensor conv_probe(double kv_scale) {
  auto out = torch::empty({256}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  const float s = (float)kv_scale;
  int e2;
  const bool pow2 = std::frexp(s, &e2) == 0.5f;
  const float k120 = std::ldexp(1.0f, 120);
  const bool extra = !(pow2 && std::isfinite(k120 * s) && k120 * s >= 1e-30f);
  const uint16_t mb = bf16_bits(extra ? k120 : k120 * s);
  const uint16_t sb = bf16_bits(s);
  conv_probe_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()), (uint32_t)mb | ((uint32_t)mb << 16),
      (uint32_t)sb | ((uint32_t)sb << 16), extra ? 1 : 0);
  return out;
}

}  // namespace glm53_mla

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &glm53_mla::run, "GLM-5.3 sparse-MLA prefill (exact bf16, fp8 KV)", pybind11::arg("q"),
        pybind11::arg("kv"), pybind11::arg("slots"), pybind11::arg("valid"), pybind11::arg("out"),
        pybind11::arg("sm_scale"), pybind11::arg("kv_scale"), pybind11::arg("variant") = 4);
  m.def("conv_probe", &glm53_mla::conv_probe);
  m.attr("VERSION") = 4;
  m.attr("SMEM") = glm53_mla::V3_SMEM_TOTAL;   // default variant (4)
}
