// GLM53_DEC_SMALLOPS: small decode kernels outside the MoE (docs/DEC_SMALLOPS.md).
//
// Every kernel here reproduces the exact floating-point operation sequence of the production op it replaces
// (bit-identical outputs), only the work distribution / memory traffic differs:
//
//   dconv_*          DFlash2 grouped dynamic conv (qwen3_dflash2._grouped_conv): 10 PyTorch eager bf16 elementwise
//                    kernels -> 1. Each PyTorch op rounds to bf16; the kernel rounds after every op the same way.
//   mhc_fused_*      vLLM mhc_fused_tilelang (decode M <= 16: mHC post-mapping + the pre-norm GEMM partials).
//                    The TileLang kernel runs one CTA per (token, 2-3 outputs, K split), so every weight element is
//                    read by M CTAs; here one CTA serves all M tokens for NB outputs of one K split (weights read
//                    once, kept in registers). Per (split, token, output) the partial sum is built by the same
//                    thread <-> h mapping, the same FMA order, the same xor-butterfly and the same 8-warp order.
//                    Optional bf16 weight copy (exact: GLM's hc_*_fn are bf16 values stored as fp32).
//   mhc_pre_*        vLLM mhc_pre_big_fuse_with_norm_tilelang (split sum, sigmoid pre/post mixes, 20-iteration
//                    Sinkhorn, weighted stream sum + fused RMSNorm) with the same arithmetic.
//
// Build: setup.py (AOT, extension glm53_smallops_ext) or JIT from glm53_smallops.load_ext (tests).
// Compiled without fast-math, FMA contraction on (nvcc default) = the flags TileLang uses for these kernels.
#include "smallops_kernels.cuh"
#include "smallops_tc.cuh"

// ------------------------------------------------------------------------------------------------------------------
// bindings

void dconv(const at::Tensor& x, const at::Tensor& delta, const at::Tensor& base, at::Tensor& out, int64_t gs,
           int64_t block_size) {
    TORCH_CHECK(x.is_cuda() && delta.is_cuda() && base.is_cuda() && out.is_cuda(), "dconv: CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && delta.scalar_type() == at::kBFloat16 &&
                    base.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "dconv: bf16 tensors");
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0, "dconv: x [T, H] unit inner stride");
    const int64_t T = x.size(0), H = x.size(1);
    TORCH_CHECK(base.dim() == 2 && base.size(1) == H && base.is_contiguous(), "dconv: base [taps, H] contiguous");
    const int64_t taps = base.size(0);
    TORCH_CHECK(delta.dim() == 3 && delta.size(0) == T && delta.size(1) == taps && delta.size(2) * gs == H &&
                    delta.stride(2) == 1,
                "dconv: delta [T, taps, H/gs] with unit inner stride");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == T && out.size(1) == H && out.is_contiguous(), "dconv: out [T, H]");
    TORCH_CHECK(H % 8 == 0 && block_size >= 1 && gs >= 1, "dconv: H % 8, block_size, gs");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(base.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "dconv: 16-byte aligned x / base / out");
    if (T == 0) return;
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int threads = 128;
    const int64_t n = T * (H / 8);
    const int blocks = (int)((n + threads - 1) / threads);
    auto xp = reinterpret_cast<const bf16*>(x.data_ptr());
    auto dp = reinterpret_cast<const bf16*>(delta.data_ptr());
    auto bp = reinterpret_cast<const bf16*>(base.data_ptr());
    auto op = reinterpret_cast<bf16*>(out.data_ptr());
    switch (taps) {
        case 1: dconv_kernel<1><<<blocks, threads, 0, st>>>(xp, x.stride(0), dp, delta.stride(0), delta.stride(1), bp, op, (int)T, (int)H, (int)gs, (int)block_size); break;
        case 2: dconv_kernel<2><<<blocks, threads, 0, st>>>(xp, x.stride(0), dp, delta.stride(0), delta.stride(1), bp, op, (int)T, (int)H, (int)gs, (int)block_size); break;
        case 3: dconv_kernel<3><<<blocks, threads, 0, st>>>(xp, x.stride(0), dp, delta.stride(0), delta.stride(1), bp, op, (int)T, (int)H, (int)gs, (int)block_size); break;
        case 4: dconv_kernel<4><<<blocks, threads, 0, st>>>(xp, x.stride(0), dp, delta.stride(0), delta.stride(1), bp, op, (int)T, (int)H, (int)gs, (int)block_size); break;
        default: TORCH_CHECK(false, "dconv: taps 1..4, got ", taps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// yp [S, M, 24] fp32, rp [S, M] fp32, res_out [M, 4, 4096] bf16; w [24, 4, 4096] fp32 or bf16 (contiguous).
// S = 8 for M < 8 and 4 for 8 <= M <= 16 (production's split choice; the caller passes the tensors sized so).
void mhc_fused(const at::Tensor& comb, const at::Tensor& post, const at::Tensor& res_in, const at::Tensor& x_in,
               const at::Tensor& w, at::Tensor& yp, at::Tensor& rp, at::Tensor& res_out) {
    TORCH_CHECK(comb.is_cuda() && post.is_cuda() && res_in.is_cuda() && x_in.is_cuda() && w.is_cuda() &&
                    yp.is_cuda() && rp.is_cuda() && res_out.is_cuda(),
                "mhc_fused: CUDA tensors");
    const int64_t M = x_in.size(0);
    TORCH_CHECK(M >= 1 && M <= MHC_MMAX, "mhc_fused: 1 <= M <= 16, got ", M);
    TORCH_CHECK(comb.scalar_type() == at::kFloat && comb.is_contiguous() && comb.numel() == M * 16, "mhc_fused: comb");
    TORCH_CHECK(post.scalar_type() == at::kFloat && post.is_contiguous() && post.numel() == M * 4, "mhc_fused: post");
    TORCH_CHECK(res_in.scalar_type() == at::kBFloat16 && res_in.is_contiguous() && res_in.numel() == M * 4 * HID,
                "mhc_fused: residual_in");
    TORCH_CHECK(x_in.scalar_type() == at::kBFloat16 && x_in.is_contiguous() && x_in.numel() == M * HID, "mhc_fused: x");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x_in.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(res_in.data_ptr()) % 16 == 0,
                "mhc_fused: 16-byte aligned x / residual");
    TORCH_CHECK(res_out.scalar_type() == at::kBFloat16 && res_out.is_contiguous() && res_out.numel() == M * 4 * HID,
                "mhc_fused: residual_out");
    TORCH_CHECK(w.is_contiguous() && w.numel() == (int64_t)NOUT * HC * HID &&
                    (w.scalar_type() == at::kFloat || w.scalar_type() == at::kBFloat16),
                "mhc_fused: weight [24, 16384] fp32 / bf16 contiguous");
    const int64_t S = yp.size(0);
    TORCH_CHECK(yp.scalar_type() == at::kFloat && yp.is_contiguous() && yp.dim() == 3 && yp.size(1) == M &&
                    yp.size(2) == NOUT && (S == 8 || S == 4),
                "mhc_fused: yp [S in {4, 8}, M, 24]");
    TORCH_CHECK(rp.scalar_type() == at::kFloat && rp.is_contiguous() && rp.numel() == S * M, "mhc_fused: rp [S, M]");
    const c10::cuda::CUDAGuard guard(x_in.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const bool wb = w.scalar_type() == at::kBFloat16;
    // S = 8 (M < 8): one stage holding exactly the M tokens (MC = M, compile time: fully unrolled token loop);
    // S = 4: M = 8 one 8-token stage (3 outputs per CTA); 9 <= M <= 16 4-token stages, double-buffered.
    if (S == 8) {
        TORCH_CHECK(M <= 7, "mhc_fused: S = 8 serves M <= 7");
#define MHC_S8(MC_)                                                                                          \
    case MC_:                                                                                                \
        if (wb) launch_mhc_fused<bf16, 8, 4, MC_>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st); \
        else launch_mhc_fused<float, 8, 4, MC_>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st);  \
        break;
        switch (M) { MHC_S8(1) MHC_S8(2) MHC_S8(3) MHC_S8(4) MHC_S8(5) MHC_S8(6) MHC_S8(7) }
#undef MHC_S8
    } else if (M == 8) {
        // M = 8 (K = 7 decode): all 8 tokens in one stage, 3 outputs per CTA (docs/logs/smallops/mb_mhc_m8.log)
        if (wb) launch_mhc_fused<bf16, 4, 3, 8>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st);
        else launch_mhc_fused<float, 4, 3, 8>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st);
    } else {
        if (wb) launch_mhc_fused<bf16, 4, 2, 4>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st);
        else launch_mhc_fused<float, 4, 2, 4>(comb, post, res_in, x_in, w, yp, rp, res_out, (int)M, st);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void l2_prefetch(const at::Tensor& t, at::Tensor& sink, int64_t ctas) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "l2_prefetch: contiguous CUDA tensor");
    TORCH_CHECK(sink.is_cuda() && sink.numel() >= 1 && sink.element_size() == 4, "l2_prefetch: 4-byte sink");
    const int64_t bytes = t.numel() * t.element_size();
    TORCH_CHECK(bytes % 16 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "l2_prefetch: 16-byte units");
    if (bytes == 0) return;
    const c10::cuda::CUDAGuard guard(t.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int64_t n16 = bytes / 16;
    const int64_t need = (n16 + 255) / 256;
    const int grid = (int)std::max<int64_t>(1, std::min<int64_t>(ctas, need));
    l2_prefetch_kernel<<<grid, 256, 0, st>>>(reinterpret_cast<const uint4*>(t.data_ptr()), n16,
                                              reinterpret_cast<unsigned*>(sink.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// y[b, m, n] = bf16(sum_k w[b, n, k] x[b, m, k]) on tensor cores, cuBLAS-wmma bitwise order (smallops_tc.cuh).
// w [B, N, K] with stride(1) == 1 (NCONTIG) or stride(2) == 1 (KCONTIG); x [B, M, K], y [B, M, N] unit inner stride.
// cfg = warps * 100 + stages (KC = 64): warps in {1, 2, 4}, stages in {4, 6} (6 falls back to 4 when the
// shared memory does not fit). Not wired into production (docs/DEC_SMALLOPS.md: cuBLAS is at the read floor).
template <bool NC, int W, int S>
static cudaError_t tc_dispatch_mt(const smallops_tc::TcArgs& a, int batch, cudaStream_t st) {
    if (a.M <= 8) return smallops_tc::tc_launch<NC, W, 64, S, 1>(a, batch, st);
    return smallops_tc::tc_launch<NC, W, 64, S, 2>(a, batch, st);
}
template <bool NC, int W, int S>
static size_t tc_smem_for(const smallops_tc::TcArgs& a) {
    return a.M <= 8 ? smallops_tc::tc_smem_bytes<NC, W, 64, S, 1>(a.K) : smallops_tc::tc_smem_bytes<NC, W, 64, S, 2>(a.K);
}
// Shared memory that one CTA may opt in to (per device; queried once per process, device 0 .. 15).
static size_t smem_optin(int dev) {
    static int cache[16] = {0};
    if (dev < 0 || dev >= 16) return 48 * 1024;
    if (cache[dev] == 0) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || v <= 0) v = 48 * 1024;
        cache[dev] = v;
    }
    return (size_t)cache[dev];
}
// stages = the requested count, or 4 when 6 does not fit the device's shared memory.
template <bool NC>
static cudaError_t tc_dispatch(const smallops_tc::TcArgs& a, int batch, int warps, int stages, cudaStream_t st,
                               size_t optin) {
#define TC_CASE(W_, S_) \
    if (warps == W_ && stages == S_) { \
        if (tc_smem_for<NC, W_, S_>(a) <= optin) return tc_dispatch_mt<NC, W_, S_>(a, batch, st); \
        stages = S_ == 6 ? 4 : 0; \
    }
    TC_CASE(1, 6) TC_CASE(1, 4) TC_CASE(2, 6) TC_CASE(2, 4) TC_CASE(4, 6) TC_CASE(4, 4)
#undef TC_CASE
    return cudaErrorInvalidValue;
}

void tc_gemm(const at::Tensor& w, const at::Tensor& x, at::Tensor& y, int64_t cfg) {
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda(), "tc_gemm: CUDA tensors");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kBFloat16,
                "tc_gemm: bf16");
    TORCH_CHECK(w.dim() == 3 && x.dim() == 3 && y.dim() == 3, "tc_gemm: 3-d views [B, N, K] / [B, M, K] / [B, M, N]");
    const int64_t B = w.size(0), N = w.size(1), K = w.size(2), M = x.size(1);
    TORCH_CHECK(x.size(0) == B && x.size(2) == K && y.size(0) == B && y.size(1) == M && y.size(2) == N,
                "tc_gemm: shapes");
    TORCH_CHECK(M >= 1 && M <= 16, "tc_gemm: 1 <= M <= 16");
    const bool nc = w.stride(1) == 1 && w.size(1) > 1 && w.stride(2) != 1;
    TORCH_CHECK(nc || w.stride(2) == 1, "tc_gemm: weight needs unit stride along N or K");
    TORCH_CHECK(x.stride(2) == 1 && y.stride(2) == 1, "tc_gemm: x / y unit inner stride");
    const int warps = (int)(cfg / 100), stages = (int)(cfg % 100);
    TORCH_CHECK(K % 64 == 0 && N % (16 * warps) == 0, "tc_gemm: K % 64, N % (16 * warps)");
    auto al16 = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    TORCH_CHECK(al16(w.data_ptr()) && al16(x.data_ptr()), "tc_gemm: 16-byte aligned w / x");
    TORCH_CHECK((nc ? w.stride(2) : w.stride(1)) % 8 == 0 && w.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 &&
                    x.stride(0) % 8 == 0,
                "tc_gemm: strides multiple of 8 elements");
    smallops_tc::TcArgs a;
    a.w = reinterpret_cast<const bf16*>(w.data_ptr());
    a.sWb = w.stride(0); a.sWn = w.stride(1); a.sWk = w.stride(2);
    a.x = reinterpret_cast<const bf16*>(x.data_ptr());
    a.sXb = x.stride(0); a.sXm = x.stride(1);
    a.y = reinterpret_cast<bf16*>(y.data_ptr());
    a.sYb = y.stride(0); a.sYm = y.stride(1);
    a.M = (int)M; a.N = (int)N; a.K = (int)K;
    const c10::cuda::CUDAGuard guard(w.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const size_t optin = smem_optin(w.get_device());
    const cudaError_t e = nc ? tc_dispatch<true>(a, (int)B, warps, stages, st, optin)
                             : tc_dispatch<false>(a, (int)B, warps, stages, st, optin);
    TORCH_CHECK(e == cudaSuccess, "tc_gemm: launch failed (cfg ", cfg, "): ", cudaGetErrorString(e));
}

int64_t smallops_version() { return 1; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GLM53_DEC_SMALLOPS decode kernels (docs/DEC_SMALLOPS.md)";
    m.def("dconv", &dconv);
    m.def("mhc_fused", &mhc_fused);
    m.def("l2_prefetch", &l2_prefetch);
    m.def("tc_gemm", &tc_gemm);
    m.def("version", &smallops_version);
}
