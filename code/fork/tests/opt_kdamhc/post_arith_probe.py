"""opt-kdamhc: which fp32 operation order reproduces mhc_post_tilelang's residual_cur bit for bit?
Variants (per element, j output stream, k = 0..3): FMA  v = c*d; v = fma(a[k,j], b[k], v)
                                                   MULADD v = c*d; v = v + a[k,j]*b[k] (separately rounded)
                                                   FMA_ALL v = fma(a[0,j], b0, c*d) ... same as FMA
                                                   MULADD_REV the k loop reversed (both kinds)."""
import os, sys
import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mhc_sp"))
from bench_mhc_halving import make  # noqa: E402
from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang  # noqa: E402

SRC = r"""
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <int V>
__global__ void k(const float* a, const __nv_bfloat16* b, const float* c, const __nv_bfloat16* d, __nv_bfloat16* o, int M, int H) {
  int t = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float df = __bfloat162float(d[(size_t)t * H + h]);
    float bf[4];
    for (int q = 0; q < 4; ++q) bf[q] = __bfloat162float(b[((size_t)t * 4 + q) * H + h]);
    for (int j = 0; j < 4; ++j) {
      if (V == 4) {   // LLVM's contraction of TileLang's x = c*d; x = x + a0*b0: fma(c, d, a0*b0), then fma(a_k, b_k, x)
        float v = __fmaf_rn(c[t * 4 + j], df, __fmul_rn(a[t * 16 + j], bf[0]));
        for (int q = 1; q < 4; ++q) v = __fmaf_rn(a[t * 16 + q * 4 + j], bf[q], v);
        o[((size_t)t * 4 + j) * H + h] = __float2bfloat16_rn(v);
        continue;
      }
      if (V == 5) {   // TileLang's source text verbatim (nvcc contracts as it likes)
        float v = c[t * 4 + j] * df;
        for (int q = 0; q < 4; ++q) v = v + a[t * 16 + q * 4 + j] * bf[q];
        o[((size_t)t * 4 + j) * H + h] = __float2bfloat16_rn(v);
        continue;
      }
      float v = __fmul_rn(c[t * 4 + j], df);
      for (int kk = 0; kk < 4; ++kk) {
        int q = (V & 2) ? 3 - kk : kk;
        float av = a[t * 16 + q * 4 + j];
        if (V & 1) v = __fadd_rn(v, __fmul_rn(av, bf[q])); else v = __fmaf_rn(av, bf[q], v);
      }
      o[((size_t)t * 4 + j) * H + h] = __float2bfloat16_rn(v);
    }
  }
}
void run(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d, torch::Tensor o, int64_t v) {
  int M = b.size(0), H = b.size(2);
  auto s = at::cuda::getCurrentCUDAStream();
  auto A = a.data_ptr<float>(); auto C = c.data_ptr<float>();
  auto B = (const __nv_bfloat16*)b.data_ptr(); auto D = (const __nv_bfloat16*)d.data_ptr(); auto O = (__nv_bfloat16*)o.data_ptr();
  if (v == 0) k<0><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
  if (v == 1) k<1><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
  if (v == 2) k<2><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
  if (v == 3) k<3><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
  if (v == 4) k<4><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
  if (v == 5) k<5><<<M, 256, 0, s>>>(A, B, C, D, O, M, H);
}
"""
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
ext = load_inline("opt_post_arith_probe2", cpp_sources="void run(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d, torch::Tensor o, int64_t v);",
                  cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
for M in (512, 3456):
    w = make(M, seed=11)
    ref = torch.empty_like(w["res"])
    mhc_post_tilelang(w["comb"], w["res"], w["post"].view(M, 4), w["x"], ref, 4, 4096)
    for v, nm in ((0, "fma"), (1, "muladd"), (2, "fma_rev"), (3, "muladd_rev"), (4, "fma_cd_first"), (5, "tl_text")):
        o = torch.empty_like(ref)
        ext.run(w["comb"].contiguous(), w["res"].contiguous(), w["post"].view(M, 4).contiguous(), w["x"].contiguous(), o, v)
        torch.cuda.synchronize()
        diff = (o != ref).float().mean().item()
        print(f"M={M} {nm:10s}: bitwise {torch.equal(o, ref)}; {100 * diff:.4f} % elems differ", flush=True)
