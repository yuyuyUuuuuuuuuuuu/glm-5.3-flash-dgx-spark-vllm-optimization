// L1 ceiling probe (docs/MLA_PREFILL.md): FlashInfer 0.6.18's SM120 sparse-MLA multi-group prefill kernel
// (GLM_NSA = 656-byte packed KV, FP8 QK, 2-pass FP8 P.V) at GLM-5.3 production shapes, plus a pack kernel that
// builds the temporary 656-byte layout from the production 512-byte fp8 cache. Benchmark-only extension (JIT via
// torch.utils.cpp_extension.load); the kernel itself is FlashInfer's header, compiled with FlashInfer's own flags.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>
#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

namespace {

constexpr ModelType kMT = ModelType::GLM_NSA;
constexpr int kNH = 32, kTOPK = 2048, kPBS = 64;

void l1_mg(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor topk_len, torch::Tensor out,
           torch::Tensor lse, double sm_scale) {
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.dim() == 3 && q.size(1) == kNH && q.size(2) == 576 && q.is_contiguous());
  TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.size(-1) == 656 && kv.is_contiguous());
  TORCH_CHECK(idx.dtype() == torch::kInt32 && idx.size(1) == kTOPK && idx.is_contiguous());
  TORCH_CHECK(topk_len.dtype() == torch::kInt32 && topk_len.numel() == q.size(0));
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.size(2) == 512 && out.is_contiguous());
  const int T = (int)q.size(0);
  constexpr size_t smem = SmemLayoutMG<kMT, ComputeMode::FP8>::TOTAL;
  auto kernel = sparse_mla_prefill_mg_kernel<kMT, ComputeMode::FP8, kNH, kTOPK, kPBS, 2>;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    configured = true;
  }
  PrefillColdParams cold{(float)sm_scale, T, (size_t)0, (size_t)0, 0, nullptr, topk_len.data_ptr<int>(), nullptr};
  const bf16* Q = reinterpret_cast<const bf16*>(q.data_ptr());
  const uint8_t* KV = kv.data_ptr<uint8_t>();
  const int32_t* I = idx.data_ptr<int32_t>();
  bf16* O = reinterpret_cast<bf16*>(out.data_ptr());
  float* L = lse.data_ptr<float>();
  const float* sink = nullptr;
  cudaLaunchConfig_t config{dim3(T), dim3(BLOCK_THREADS), smem, at::cuda::getCurrentCUDAStream(), nullptr, 0};
  void* args[] = {(void*)&Q, (void*)&KV, (void*)&I, (void*)&O, (void*)&L, (void*)&sink, (void*)&cold};
  TORCH_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args) == cudaSuccess);
}

// packed[s, 0:512] = cache[s, 0:512]; packed[s, 512:528] = 4 x fp32 k_scale; packed[s, 528:656] = 0 (NoPE rope)
__global__ void pack656_kernel(const uint4* __restrict__ cache, uint4* __restrict__ packed, int n, float k_scale) {
  const int s = blockIdx.x * 8 + (threadIdx.x >> 5);   // 8 slots per 256-thread block, one warp per slot
  const int lane = threadIdx.x & 31;
  if (s >= n) return;
  const uint4* src = cache + (size_t)s * 32;             // 512 B = 32 x 16 B
  uint4* dst = packed + (size_t)s * 41;                  // 656 B = 41 x 16 B
  dst[lane] = src[lane];
  if (lane < 9) {
    uint4 v = make_uint4(0, 0, 0, 0);
    if (lane == 0) { uint32_t b = __float_as_uint(k_scale); v = make_uint4(b, b, b, b); }
    dst[32 + lane] = v;
  }
}

void pack656(torch::Tensor cache, torch::Tensor packed, double k_scale) {
  TORCH_CHECK(cache.is_contiguous() && packed.is_contiguous() && cache.size(-1) == 512 && packed.size(-1) == 656);
  const int n = (int)(cache.numel() / 512);
  TORCH_CHECK(packed.numel() / 656 == n);
  pack656_kernel<<<(n + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(cache.data_ptr()), reinterpret_cast<uint4*>(packed.data_ptr()), n, (float)k_scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("l1_mg", &l1_mg);
  m.def("pack656", &pack656);
  m.attr("SMEM") = (int)SmemLayoutMG<kMT, ComputeMode::FP8>::TOTAL;
}
