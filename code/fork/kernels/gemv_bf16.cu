// Small-M dense GEMMs of GLM-5.3 decode (docs/BF16_GEMV.md).
//
//   Y[M, N] = X[M, K] . W[N, K]^T      X, W bf16 (W = nn.Linear layout, row-major [N, K]), fp32 accumulation
//
// Production runs these through cuBLAS (cutlass_80_wmma 16x16 tiles + cublasLt splitKreduce, or a one-block
// gemmSN for the fp32 head-gate GEMM). At decode M = 1..64, so every call is a weight stream of 0.25-170 MB that
// cuBLAS runs at 7-240 GB/s. Two kernels:
//
// gemm_bf16  tensor cores (mma.sync m16n8k16 bf16 -> fp32). The tokens are the MMA's M side (A, 16 rows per m tile,
//            zero rows past M), 8 weight rows are the MMA's N side (B). A 32-wide k chunk is two MMAs whose k
//            order is permuted so that every lane loads 8 consecutive bf16 (16 bytes) of its weight row and of
//            its activation rows: lane (g = lane/4, t = lane%4) holds physical k 8t..8t+7; MMA j takes 8t+4j..+3
//            as its logical columns {2t, 2t+1, 2t+8, 2t+9}. A and B use the same map, so the dot product is the
//            same sum over k in another order (fp32 accumulation, exact bf16 products).
//            CTA = WN warps, warp w owns 8 weight rows; all warps share the CTA's K range, whose activations are
//            staged in shared memory chunk by chunk (KCH k per chunk). Weights: 16-byte loads straight into
//            registers, one chunk ahead of use (no reuse, optionally L2 evict_first). Grid (N / (8 WN), S): S > 1
//            splits K over CTAs; each writes an fp32 partial, and the last CTA of a column group (atomic ticket)
//            sums the S partials in split order 0..S-1 and writes Y. The order is fixed -> deterministic, and the
//            ticket counter is reset by that CTA -> CUDA-graph replay safe. No second kernel.
//            out_mode 0: bf16 Y, 1: fp32 Y, 2: fp32 Y holding the bf16-rounded value (= cuBLAS bf16 GEMM followed
//            by .to(float32), what production's router does on this GPU).
//
// gemm_f32   IEEE fp32 FMA on CUDA cores for Y = float(X) . float(W)^T with N <= 32 (the indexer head gate, which
//            production computes as torch.mm(x.float(), W.float().t()) on purpose, in fp32). W is given as bf16
//            (the checkpoint values; production's fp32 copy holds exactly these values). Each CTA: one K slice of
//            KS k, lane n owns weight row n, warps own tokens; sequential k within a slice, slices summed in order
//            by the last CTA (same ticket scheme).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ uint4 ld_w16(const void* p, uint64_t pol, int use_pol) {
    uint4 v;
    if (use_pol) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
    } else {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    }
    return v;
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void store2(void* Y, int64_t ldy, int out_mode, int m, int n, float v0, float v1) {
    if (out_mode == 0) {
        __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(Y) + (int64_t)m * ldy + n;
        *reinterpret_cast<__nv_bfloat162*>(y) = __floats2bfloat162_rn(v0, v1);
    } else {
        if (out_mode == 2) {
            v0 = __bfloat162float(__float2bfloat16_rn(v0));
            v1 = __bfloat162float(__float2bfloat16_rn(v1));
        }
        float* y = reinterpret_cast<float*>(Y) + (int64_t)m * ldy + n;
        *reinterpret_cast<float2*>(y) = make_float2(v0, v1);
    }
}

// MT: m tiles of 16 tokens (M <= 16 MT). HALF: M <= 8 (MT == 1), rows 8..15 of the tile are never read.
template <int MT, bool HALF, int WN, int KCH>
__global__ void __launch_bounds__(WN * 32) gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ X, int64_t ldx, const __nv_bfloat16* __restrict__ W, int64_t ldw, int M,
    int N, int K, int Kc, int out_mode, void* __restrict__ Y, int64_t ldy, float* __restrict__ ws,
    int* __restrict__ counters, int use_pol, int full_stage) {
    constexpr int STEPS = KCH / 32;
    constexpr int MROWS = HALF ? 8 : MT * 16;   // staged activation rows
    constexpr int SROW = KCH + 32;              // smem row stride (elements): +64 B -> conflict-free 16 B reads
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __shared__ int s_last;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * WN + warp) * 8;
    const bool active = n0 < N;   // N % 8 == 0 (host-checked); warp-uniform
    const int S = gridDim.y, s = blockIdx.y;
    const int kbeg = s * Kc;   // Kc % KCH == 0, S * Kc == K (host-checked)
    uint64_t pol = 0;
    if (use_pol) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const __nv_bfloat16* wp = W + (int64_t)(active ? n0 + g : 0) * ldw + t * 8;

    float acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[mt][i] = 0.f;

    // stage X[:M, k0 : k0 + KCH] into buffer b (cp.async, zero rows past M)
    auto stage = [&](int k0, int b) {
        constexpr int VPR = KCH / 8;   // 16-byte vectors per row
        __nv_bfloat16* dst = xs + b * (MROWS * SROW);
#pragma unroll 4
        for (int v = threadIdx.x; v < MROWS * VPR; v += WN * 32) {
            const int r = v / VPR, c = (v - r * VPR) * 8;
            if (r < M) cp_async16(dst + r * SROW + c, X + (int64_t)r * ldx + k0 + c);
            else *reinterpret_cast<uint4*>(dst + r * SROW + c) = make_uint4(0u, 0u, 0u, 0u);
        }
        cp_async_commit();
    };

    uint4 wb[STEPS];
    if (active) {
#pragma unroll
        for (int i = 0; i < STEPS; ++i) wb[i] = ld_w16(wp + kbeg + i * 32, pol, use_pol);
    }
    const int nch = Kc / KCH;
    if (full_stage) {   // every chunk of the CTA's K range staged up front: one L2 round trip, no restaging
        for (int c = 0; c < nch; ++c) stage(kbeg + c * KCH, c);
        cp_async_wait<0>();
        __syncthreads();
    } else {
        stage(kbeg, 0);
    }
    for (int c = 0; c < nch; ++c) {
        const int kc = kbeg + c * KCH;
        if (!full_stage) {
            if (c + 1 < nch) {
                stage(kc + KCH, (c + 1) & 1);   // next chunk's activations in flight while this chunk computes
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
        }
        uint4 cur[STEPS];
#pragma unroll
        for (int i = 0; i < STEPS; ++i) cur[i] = wb[i];
        if (active && c + 1 < nch) {   // next chunk's weights in flight too
#pragma unroll
            for (int i = 0; i < STEPS; ++i) wb[i] = ld_w16(wp + kc + KCH + i * 32, pol, use_pol);
        }
        const __nv_bfloat16* xb = xs + (full_stage ? c : (c & 1)) * (MROWS * SROW);
        if (active) {
#pragma unroll
            for (int i = 0; i < STEPS; ++i) {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const __nv_bfloat16* xr = xb + (mt * 16 + g) * SROW + i * 32 + t * 8;
                    const uint4 lo = *reinterpret_cast<const uint4*>(xr);
                    uint4 hi = make_uint4(0u, 0u, 0u, 0u);
                    if (!HALF) hi = *reinterpret_cast<const uint4*>(xr + 8 * SROW);
                    mma_bf16(acc[mt], lo.x, hi.x, lo.y, hi.y, cur[i].x, cur[i].y);
                    mma_bf16(acc[mt], lo.z, hi.z, lo.w, hi.w, cur[i].z, cur[i].w);
                }
            }
        }
        if (!full_stage) __syncthreads();   // buffer (c & 1) is restaged at iteration c + 1
    }

    const int n = n0 + 2 * t;
    if (S == 1) {
        if (active) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int m0 = mt * 16 + g, m1 = m0 + 8;
                if (m0 < M) store2(Y, ldy, out_mode, m0, n, acc[mt][0], acc[mt][1]);
                if (!HALF && m1 < M) store2(Y, ldy, out_mode, m1, n, acc[mt][2], acc[mt][3]);
            }
        }
        return;
    }
    // split K over CTAs: partial -> ws[s][m][n]; the last CTA of this column group sums splits 0..S-1 in order
    if (active) {
        float* wsp = ws + (int64_t)s * M * N;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = mt * 16 + g, m1 = m0 + 8;
            if (m0 < M) __stcg(reinterpret_cast<float2*>(wsp + (int64_t)m0 * N + n), make_float2(acc[mt][0], acc[mt][1]));
            if (!HALF && m1 < M)
                __stcg(reinterpret_cast<float2*>(wsp + (int64_t)m1 * N + n), make_float2(acc[mt][2], acc[mt][3]));
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(counters + blockIdx.x, 1) == S - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // last CTA: the CTA's M x (8 WN) outputs as float4 columns, spread over all threads; for each vector the S
    // partials are summed in split order 0..S-1 (loads issued SB splits x VPT vectors at a time)
    {
        constexpr int VPR = 2 * WN;                        // float4 vectors per output row of this CTA
        constexpr int VPT = (MT * 16 * VPR + WN * 32 - 1) / (WN * 32);   // max vectors per thread
        const int cn0 = blockIdx.x * WN * 8;
        float4 v[VPT];
        int vm[VPT], vn[VPT];
        bool ok[VPT];
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = threadIdx.x + j * WN * 32;
            vm[j] = idx / VPR;
            vn[j] = cn0 + (idx - vm[j] * VPR) * 4;
            ok[j] = vm[j] < M && vn[j] < N;
            v[j] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        const int64_t stride = (int64_t)M * N;
        constexpr int SB = VPT >= 3 ? 4 : 8;               // splits per batch (register budget)
        for (int b = 0; b < S; b += SB) {
            float4 q[VPT][SB];
#pragma unroll
            for (int j = 0; j < VPT; ++j)
#pragma unroll
                for (int u = 0; u < SB; ++u)
                    if (ok[j] && b + u < S)
                        q[j][u] = __ldcg(reinterpret_cast<const float4*>(ws + (int64_t)(b + u) * stride +
                                                                          (int64_t)vm[j] * N + vn[j]));
#pragma unroll
            for (int j = 0; j < VPT; ++j)
#pragma unroll
                for (int u = 0; u < SB; ++u)
                    if (ok[j] && b + u < S) {
                        v[j].x += q[j][u].x;
                        v[j].y += q[j][u].y;
                        v[j].z += q[j][u].z;
                        v[j].w += q[j][u].w;
                    }
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j)
            if (ok[j]) {
                store2(Y, ldy, out_mode, vm[j], vn[j], v[j].x, v[j].y);
                store2(Y, ldy, out_mode, vm[j], vn[j] + 2, v[j].z, v[j].w);
            }
    }
    if (threadIdx.x == 0) counters[blockIdx.x] = 0;
}

// fp32 head gate: N <= 32 rows, lane n owns row n. KS k per CTA, 8 warps; warp w owns tokens w, w + 8, ...
constexpr int F32_KS = 128;
constexpr int F32_WARPS = 8;
constexpr int F32_MMAX = 64;

__global__ void __launch_bounds__(F32_WARPS * 32) gemm_f32_kernel(
    const __nv_bfloat16* __restrict__ X, int64_t ldx, const __nv_bfloat16* __restrict__ W, int64_t ldw, int M,
    int N, int K, float* __restrict__ Y, int64_t ldy, float* __restrict__ ws, int* __restrict__ counters) {
    __shared__ __align__(16) float xs[F32_MMAX][F32_KS];
    __shared__ int s_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int S = gridDim.x, s = blockIdx.x;
    const int kbeg = s * F32_KS;
    // weight row `lane`, this slice: 128 bf16 = 16 x 16 B, in registers
    uint4 w[F32_KS / 8];
    const bool act = lane < N;
    if (act) {
        const uint4* wp = reinterpret_cast<const uint4*>(W + (int64_t)lane * ldw + kbeg);
#pragma unroll
        for (int i = 0; i < F32_KS / 8; ++i) w[i] = __ldg(wp + i);
    }
    // activations of the slice -> fp32 smem (exact)
    for (int v = threadIdx.x; v < M * (F32_KS / 8); v += F32_WARPS * 32) {
        const int r = v / (F32_KS / 8), c = (v - r * (F32_KS / 8)) * 8;
        const uint4 q = __ldg(reinterpret_cast<const uint4*>(X + (int64_t)r * ldx + kbeg + c));
        const uint32_t u[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xs[r][c + 2 * j] = __uint_as_float(u[j] << 16);
            xs[r][c + 2 * j + 1] = __uint_as_float(u[j] & 0xffff0000u);
        }
    }
    __syncthreads();
    for (int m = warp; m < M; m += F32_WARPS) {
        float acc = 0.f;
        const float4* xr = reinterpret_cast<const float4*>(xs[m]);
#pragma unroll
        for (int i = 0; i < F32_KS / 8; ++i) {
            const float4 x0 = xr[2 * i], x1 = xr[2 * i + 1];
            const uint32_t u[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
            acc = fmaf(x0.x, __uint_as_float(u[0] << 16), acc);
            acc = fmaf(x0.y, __uint_as_float(u[0] & 0xffff0000u), acc);
            acc = fmaf(x0.z, __uint_as_float(u[1] << 16), acc);
            acc = fmaf(x0.w, __uint_as_float(u[1] & 0xffff0000u), acc);
            acc = fmaf(x1.x, __uint_as_float(u[2] << 16), acc);
            acc = fmaf(x1.y, __uint_as_float(u[2] & 0xffff0000u), acc);
            acc = fmaf(x1.z, __uint_as_float(u[3] << 16), acc);
            acc = fmaf(x1.w, __uint_as_float(u[3] & 0xffff0000u), acc);
        }
        if (act) {
            if (S == 1) Y[(int64_t)m * ldy + lane] = acc;
            else __stcg(ws + ((int64_t)s * M + m) * N + lane, acc);
        }
    }
    if (S == 1) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(counters, 1) == S - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int o = threadIdx.x; o < M * N; o += F32_WARPS * 32) {
        const int m = o / N, n = o - m * N;
        const float* p0 = ws + (int64_t)m * N + n;
        const int64_t stride = (int64_t)M * N;
        float v = 0.f;
        for (int b = 0; b < S; b += 8) {
            float q[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (b + j < S) q[j] = __ldcg(p0 + (int64_t)(b + j) * stride);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (b + j < S) v += q[j];
        }
        Y[(int64_t)m * ldy + n] = v;
    }
    if (threadIdx.x == 0) counters[0] = 0;
}

// activations of the whole CTA K range are staged up front when they fit in this many bytes of shared memory
constexpr int FULL_STAGE_MAX = 64 * 1024;

template <int MT, bool HALF, int WN, int KCH>
void launch_t(const at::Tensor& x, const at::Tensor& w, int M, int N, int K, int S, int out_mode, at::Tensor& y,
              at::Tensor& ws, at::Tensor& counters, int use_pol, cudaStream_t st) {
    constexpr int MROWS = HALF ? 8 : MT * 16;
    constexpr int SROW = KCH + 32;
    const int nch = (K / S) / KCH;
    const size_t chunk_bytes = (size_t)MROWS * SROW * 2;
    const int full_stage = nch > 1 && (size_t)nch * chunk_bytes <= (size_t)FULL_STAGE_MAX;
    const int nbuf = full_stage ? nch : (nch > 1 ? 2 : 1);
    const size_t smem = (size_t)nbuf * chunk_bytes;
    auto kern = gemm_bf16_kernel<MT, HALF, WN, KCH>;
    constexpr size_t smem_max = 96 * 1024;   // largest request of any launch
    TORCH_CHECK(smem <= 96 * 1024, "gemm_bf16: shared memory ", smem, " > 96 KiB (use a smaller kch)");
    if (smem > 48 * 1024) {
        static bool attr_set = false;   // per instantiation
        if (!attr_set) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                (int)(smem_max < 96 * 1024 ? smem_max : 96 * 1024)));
            attr_set = true;
        }
    }
    const int groups = (N / 8 + WN - 1) / WN;
    dim3 grid(groups, S);
    kern<<<grid, WN * 32, smem, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), w.stride(0), M, N, K, K / S, out_mode, y.data_ptr(),
        y.stride(0), S > 1 ? ws.data_ptr<float>() : nullptr, S > 1 ? counters.data_ptr<int>() : nullptr, use_pol,
        full_stage);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MT, bool HALF, int WN>
void dispatch_kch(int kch, const at::Tensor& x, const at::Tensor& w, int M, int N, int K, int S, int out_mode,
                  at::Tensor& y, at::Tensor& ws, at::Tensor& counters, int use_pol, cudaStream_t st) {
    if (kch == 256) launch_t<MT, HALF, WN, 256>(x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else if (kch == 512) launch_t<MT, HALF, WN, 512>(x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else if (kch == 128) launch_t<MT, HALF, WN, 128>(x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else TORCH_CHECK(false, "gemm_bf16: kch must be 128, 256 or 512, got ", kch);
}

template <int MT, bool HALF>
void dispatch_wn(int wn, int kch, const at::Tensor& x, const at::Tensor& w, int M, int N, int K, int S,
                 int out_mode, at::Tensor& y, at::Tensor& ws, at::Tensor& counters, int use_pol, cudaStream_t st) {
    if (wn == 1) dispatch_kch<MT, HALF, 1>(kch, x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else if (wn == 2) dispatch_kch<MT, HALF, 2>(kch, x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else if (wn == 4) dispatch_kch<MT, HALF, 4>(kch, x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else if (wn == 8) dispatch_kch<MT, HALF, 8>(kch, x, w, M, N, K, S, out_mode, y, ws, counters, use_pol, st);
    else TORCH_CHECK(false, "gemm_bf16: wn must be 1, 2, 4 or 8, got ", wn);
}

}  // namespace

// y = x @ w.T. x [M, K] bf16 (row stride multiple of 8, 16-byte aligned), w [N, K] bf16 contiguous rows,
// y [M, N] bf16 (out_mode 0) or fp32 (1, 2), row stride multiple of 2. S splits of K over CTAs (S == 1: no ws);
// ws fp32 >= S * M * N, counters int32 >= ceil(N / (8 wn)) all zero (left zero after every call).
void gemm_bf16(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, at::Tensor& ws, at::Tensor& counters,
               int64_t S, int64_t wn, int64_t kch, int64_t out_mode, int64_t use_pol) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "gemm_bf16: CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "gemm_bf16: bf16 x, w");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2, "gemm_bf16: 2-D tensors");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N, "gemm_bf16: shape mismatch");
    TORCH_CHECK(M >= 1 && M <= 64, "gemm_bf16: 1 <= M <= 64, got ", M);
    TORCH_CHECK(N % 8 == 0 && K % 32 == 0, "gemm_bf16: N % 8 == 0 and K % 32 == 0");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1, "gemm_bf16: unit inner stride");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 && y.stride(0) % 2 == 0, "gemm_bf16: row strides");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(y.data_ptr()) % 8 == 0,
                "gemm_bf16: alignment");
    TORCH_CHECK(out_mode >= 0 && out_mode <= 2, "gemm_bf16: out_mode 0..2");
    TORCH_CHECK(y.scalar_type() == (out_mode == 0 ? at::kBFloat16 : at::kFloat), "gemm_bf16: y dtype vs out_mode");
    TORCH_CHECK(S >= 1 && K % S == 0 && (K / S) % kch == 0, "gemm_bf16: K / S must be a multiple of kch");
    if (S > 1) {
        const int64_t groups = (N / 8 + wn - 1) / wn;
        TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.is_contiguous() && ws.numel() >= S * M * N,
                    "gemm_bf16: ws too small");
        TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt && counters.is_contiguous() &&
                        counters.numel() >= groups,
                    "gemm_bf16: counters too small");
    }
    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    const int mt = (int)((M + 15) / 16);
    const int m = (int)M, n = (int)N, k = (int)K, s = (int)S, w_ = (int)wn, kc = (int)kch, om = (int)out_mode,
              up = (int)use_pol;
    if (M <= 8) dispatch_wn<1, true>(w_, kc, x, w, m, n, k, s, om, y, ws, counters, up, st);
    else if (mt == 1) dispatch_wn<1, false>(w_, kc, x, w, m, n, k, s, om, y, ws, counters, up, st);
    else if (mt == 2) dispatch_wn<2, false>(w_, kc, x, w, m, n, k, s, om, y, ws, counters, up, st);
    else if (mt == 3) dispatch_wn<3, false>(w_, kc, x, w, m, n, k, s, om, y, ws, counters, up, st);
    else dispatch_wn<4, false>(w_, kc, x, w, m, n, k, s, om, y, ws, counters, up, st);
}

// y = float(x) @ float(w).T in IEEE fp32 (CUDA cores). x [M, K] bf16, w [N, K] bf16 with N <= 32, y [M, N] fp32.
// K % 128 == 0; S = K / 128 CTAs; ws fp32 >= S * M * N, counters int32 >= 1, zero.
void gemm_f32(const at::Tensor& x, const at::Tensor& w, at::Tensor& y, at::Tensor& ws, at::Tensor& counters) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "gemm_f32: CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kFloat,
                "gemm_f32: bf16 x, w; fp32 y");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N, "gemm_f32: shape mismatch");
    TORCH_CHECK(M >= 1 && M <= F32_MMAX && N >= 1 && N <= 32 && K % F32_KS == 0, "gemm_f32: M <= 64, N <= 32, K % 128");
    TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                    w.stride(0) % 8 == 0,
                "gemm_f32: strides");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "gemm_f32: alignment");
    const int64_t S = K / F32_KS;
    if (S > 1) {
        TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.is_contiguous() && ws.numel() >= S * M * N,
                    "gemm_f32: ws too small");
        TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt && counters.numel() >= 1,
                    "gemm_f32: counters");
    }
    const at::cuda::CUDAGuard guard(x.device());
    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    gemm_f32_kernel<<<(unsigned)S, F32_WARPS * 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), w.stride(0), (int)M, (int)N, (int)K,
        y.data_ptr<float>(), y.stride(0), S > 1 ? ws.data_ptr<float>() : nullptr,
        S > 1 ? counters.data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t gemv_version() { return 1; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "small-M dense GEMMs of GLM-5.3 decode (docs/BF16_GEMV.md)";
    m.def("gemm_bf16", &gemm_bf16);
    m.def("gemm_f32", &gemm_f32);
    m.def("version", &gemv_version);
}
