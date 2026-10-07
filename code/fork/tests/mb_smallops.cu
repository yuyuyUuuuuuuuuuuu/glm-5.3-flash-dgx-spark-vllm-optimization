// nodeC micro-benchmark build of kernels/smallops_kernels.cuh: mhc_fused variants for tests/mb_smallops.py.
// mode 0 = production launcher choice (staged: S=8 NB=4 MC=M / S=4 NB=2 MC=4 double-buffered),
// mode 1..4 = DIRECT (tokens read from global/L2, MC = M) with NB = mode (S from M like production).
#include "../kernels/smallops_kernels.cuh"

template <int S, int NB, int MC>
static void d(const at::Tensor& comb, const at::Tensor& post, const at::Tensor& res_in, const at::Tensor& x_in,
              const at::Tensor& w, at::Tensor& yp, at::Tensor& rp, at::Tensor& res_out, int M, cudaStream_t st) {
    launch_mhc_fused<bf16, S, NB, MC, 0, true>(comb, post, res_in, x_in, w, yp, rp, res_out, M, st);
}

void mhc_fused_var(const at::Tensor& comb, const at::Tensor& post, const at::Tensor& res_in, const at::Tensor& x_in,
                   const at::Tensor& w, at::Tensor& yp, at::Tensor& rp, at::Tensor& res_out, int64_t mode) {
    const int M = (int)x_in.size(0);
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(w.scalar_type() == at::kBFloat16);
#define ARGS comb, post, res_in, x_in, w, yp, rp, res_out, M, st
    if (mode >= 5) {                          // staged, S = 4, MC = 8 single stage (M <= 8 only)
        TORCH_CHECK(M == 8);
        if (mode == 5) launch_mhc_fused<bf16, 4, 2, 8>(ARGS);
        else if (mode == 6) launch_mhc_fused<bf16, 4, 3, 8>(ARGS);
        else launch_mhc_fused<bf16, 4, 1, 8>(ARGS);
    } else if (mode == 0) {
        if (M == 5) launch_mhc_fused<bf16, 8, 4, 5>(ARGS);
        else if (M == 6) launch_mhc_fused<bf16, 8, 4, 6>(ARGS);
        else launch_mhc_fused<bf16, 4, 2, 4>(ARGS);
    } else {
#define DM(NB_) \
        if (M == 5) d<8, NB_, 5>(ARGS); else if (M == 6) d<8, NB_, 6>(ARGS); \
        else if (M == 8) d<4, NB_, 8>(ARGS); else d<4, NB_, 16>(ARGS);
        if (mode == 1) { DM(1) } else if (mode == 2) { DM(2) } else if (mode == 3) { DM(3) } else { DM(4) }
#undef DM
    }
#undef ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mhc_fused_var", &mhc_fused_var); }
