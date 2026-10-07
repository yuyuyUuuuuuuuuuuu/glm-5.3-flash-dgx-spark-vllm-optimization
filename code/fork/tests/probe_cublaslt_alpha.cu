// Experiment (tests/probe_cublaslt_alpha.py): can cuBLASLt apply a per-output-channel alpha vector in the epilogue
// of a bf16 x bf16 -> bf16 GEMM with fp32 compute on this GPU? Lists every algorithm's pointer-mode capability and
// asks the heuristic for each pointer mode.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <string>

std::string algo_caps(bool d_fp32) {
  int ids[256];
  int n = 0;
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  const cudaDataType_t dt = d_fp32 ? CUDA_R_32F : CUDA_R_16BF;
  cublasStatus_t st = cublasLtMatmulAlgoGetIds(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, dt, dt,
                                               256, ids, &n);
  std::string r = "status=" + std::to_string((int)st) + " n=" + std::to_string(n) + " (id:pointer_mode_mask):";
  for (int i = 0; i < n; ++i) {
    cublasLtMatmulAlgo_t a;
    if (cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, dt, dt, ids[i], &a) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    uint32_t pmm = 0;
    size_t sz = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &pmm, sizeof(pmm), &sz);
    r += " " + std::to_string(ids[i]) + ":" + std::to_string(pmm);
  }
  return r;
}

std::string heuristic(int64_t M, int64_t N, int64_t K, bool d_fp32, int64_t pm_i) {
  cublasLtHandle_t h = at::cuda::getCurrentCUDABlasLtHandle();
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t a, b, d;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtPointerMode_t pm = (cublasLtPointerMode_t)pm_i;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));
  cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&d, d_fp32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = at::cuda::getCUDABlasLtWorkspaceSize();
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  cublasLtMatmulHeuristicResult_t res[4];
  int returned = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, desc, a, b, d, d, pref, 4, res, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(d);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(desc);
  return "status=" + std::to_string((int)st) + " returned=" + std::to_string(returned);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("algo_caps", &algo_caps);
  m.def("heuristic", &heuristic);
}
