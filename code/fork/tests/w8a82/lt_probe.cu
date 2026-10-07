// probe: cuBLASLt fp8 GEMM with OUTER_VEC_32F scales (per-channel weight x per-token activation) on sm_121.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#define CK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x " -> ", (int)s_); } while (0)

static cublasLtHandle_t H = nullptr;

// out[M,N] (row-major bf16) = (a[M,K] fp8 . w[N,K]^T) * sa[M] (x) sb[N]
// column-major view: D^T[N,M] = op(A)=W[N,K] (A stored K x N, lda K, op T) x B=X^T[K,M] (ldb K, op N)
// algo_idx: -1 = best heuristic, i>=0 = the i-th heuristic result; returns the number of heuristic results
int64_t lt_mm(torch::Tensor out, torch::Tensor a, torch::Tensor w, torch::Tensor sa, torch::Tensor sb,
              torch::Tensor ws, int64_t algo_idx, int64_t mode) {
  if (!H) CK(cublasLtCreate(&H));
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  const void* pa = sb.data_ptr();   // cublas "A" = our W -> per-row of op(A) = per channel N
  const void* pb = sa.data_ptr();   // cublas "B" = our X -> per-col of op(B) = per token M
  cublasLtMatmulMatrixScale_t sm = mode == 1 ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                                             : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb)));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsz = ws.numel();
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
  cublasLtMatmulHeuristicResult_t res[16]; int nres = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(H, op, la, lb, lc, lc, pref, 16, res, &nres));
  TORCH_CHECK(nres > 0, "no cublasLt algo");
  int i = algo_idx < 0 ? 0 : (int)std::min<int64_t>(algo_idx, nres - 1);
  float alpha = 1.f, beta = 0.f;
  CK(cublasLtMatmul(H, op, &alpha, w.data_ptr(), la, a.data_ptr(), lb, &beta, out.data_ptr(), lc, out.data_ptr(), lc,
                    &res[i].algo, ws.data_ptr(), wsz, at::cuda::getCurrentCUDAStream()));
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
  return nres;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("lt_mm", &lt_mm); }
