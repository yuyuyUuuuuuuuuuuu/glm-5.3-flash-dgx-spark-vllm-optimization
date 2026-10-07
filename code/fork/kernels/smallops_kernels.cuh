// GLM53_DEC_SMALLOPS device code (kernels + launch templates); included by smallops.cu (the extension) and by the
// nodeC micro-benchmarks. See smallops.cu for what each kernel replaces and why it is bit-identical.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 rbf(float v) { return __float2bfloat16_rn(v); }

// ------------------------------------------------------------------------------------------------------------------
// DFlash2 grouped conv.  out[t, c] for c in group g = c / gs:
//   c_k = bf16(base[k, c] + delta[t, k, g])                       (coefficients = base + delta.unsqueeze(-1))
//   o   = bf16(c_0 * x[t, c])                                     (output = coefficients[:, 0] * blocks)
//   for k in 1..taps-1:
//     s  = t >= k ? x[t - k, c] : 0                               (F.pad(blocks[:-k], ...))
//     p  = bf16(c_k * s); p = bf16(p * (pos(t) >= k))             (coefficients[:, k] * shifted * mask)
//     o  = bf16(o + p)                                            (output += ...)
// pos(t) = t % block_size (production: t & (block_size - 1) for a power of two, the same value for t >= 0).
// One thread = 8 consecutive channels of one token (16-byte loads / stores).
template <int TAPS>
__global__ void __launch_bounds__(128) dconv_kernel(const bf16* __restrict__ x, int64_t x_st,
                                                    const bf16* __restrict__ delta, int64_t d_st, int64_t d_sk,
                                                    const bf16* __restrict__ base, bf16* __restrict__ out,
                                                    int T, int H, int gs, int block_size) {
    const int per_t = H >> 3;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * per_t) return;
    const int t = idx / per_t;
    const int c0 = (idx - t * per_t) << 3;
    const int pos = t % block_size;
    uint4 xv = *reinterpret_cast<const uint4*>(x + (int64_t)t * x_st + c0);
    const bf16* xe = reinterpret_cast<const bf16*>(&xv);
    uint4 bv = *reinterpret_cast<const uint4*>(base + c0);
    const bf16* be = reinterpret_cast<const bf16*>(&bv);
    float o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float d = f32(delta[(int64_t)t * d_st + (c0 + e) / gs]);
        const bf16 ck = rbf(f32(be[e]) + d);
        o[e] = f32(rbf(f32(ck) * f32(xe[e])));
    }
#pragma unroll
    for (int k = 1; k < TAPS; ++k) {
        uint4 sv = make_uint4(0u, 0u, 0u, 0u);
        if (t >= k) sv = *reinterpret_cast<const uint4*>(x + (int64_t)(t - k) * x_st + c0);
        const bf16* se = reinterpret_cast<const bf16*>(&sv);
        uint4 bkv = *reinterpret_cast<const uint4*>(base + (int64_t)k * H + c0);
        const bf16* bke = reinterpret_cast<const bf16*>(&bkv);
        const float m = pos >= k ? 1.0f : 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float d = f32(delta[(int64_t)t * d_st + (int64_t)k * d_sk + (c0 + e) / gs]);
            const bf16 ck = rbf(f32(bke[e]) + d);
            const bf16 p = rbf(f32(ck) * f32(se[e]));
            const bf16 p2 = rbf(__fmul_rn(f32(p), m));
            o[e] = f32(rbf(__fadd_rn(o[e], f32(p2))));
        }
    }
    uint4 ov;
    bf16* oe = reinterpret_cast<bf16*>(&ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = rbf(o[e]);
    *reinterpret_cast<uint4*>(out + (int64_t)t * H + c0) = ov;
}

// ------------------------------------------------------------------------------------------------------------------
// mHC fused post-mapping + pre-norm GEMM partials (replaces mhc_fused_tilelang for M <= 16).
//   new_r[j](h) = fma(cm[3*4+j], r3, fma(cm[2*4+j], r2, fma(cm[1*4+j], r1, fma(pm[j], x, cm[0*4+j] * r0))))
//   residual_out[m, j, h] = bf16(new_r[j])
//   per thread tid of split ks: acc[n] = sum over it, j (in that order) of fma(w[n, j, h], new_r[j], acc[n]),
//                               h = ks * H/S + it * 256 + tid;   sqr = fma(new_r[j], new_r[j], sqr)
//   warp xor-butterfly (16, 8, 4, 2, 1), then v = 0 + warp0 + ... + warp7   -> yp[ks, m, n], rp[ks, m]
// All of this is TileLang's mhc_fused_tilelang_kernel instruction for instruction (checked on its PTX); what differs
// is that one CTA handles NB outputs for all M tokens (weights loaded once into registers).
constexpr int HC = 4;
constexpr int HID = 4096;
constexpr int NOUT = 24;
constexpr int MHC_MMAX = 16;

template <typename WT>
__device__ __forceinline__ float ldw(const WT* p);
template <>
__device__ __forceinline__ float ldw<float>(const float* p) { return __ldg(p); }
template <>
__device__ __forceinline__ float ldw<bf16>(const bf16* p) { return __bfloat162float(__ldg(p)); }

__device__ __forceinline__ float warp_sum(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Stage = the CTA's h slice [ks*HPS, (ks+1)*HPS) of x (q = 0) and residual streams 0..3 (q = 1..4) for tokens
// m0 .. m0+MC-1, as bf16 [MC][5][HPS] in shared memory (cp.async, 16-byte chunks).
template <int HPS, int MC>
__device__ __forceinline__ void stage_tokens(bf16* st, const bf16* __restrict__ x_in, const bf16* __restrict__ res_in,
                                             int m0, int M, int ks, int tid) {
    constexpr int CH = HPS / 8;               // 16-byte chunks per (token, stream)
    for (int i = tid; i < MC * 5 * CH; i += 256) {
        const int m = i / (5 * CH), r = i - m * (5 * CH), q = r / CH, c = r - q * CH;
        if (m0 + m >= M) break;
        const bf16* src = q == 0 ? x_in + (int64_t)(m0 + m) * HID + ks * HPS + c * 8
                                 : res_in + ((int64_t)(m0 + m) * HC + (q - 1)) * HID + ks * HPS + c * 8;
        cp_async16(st + ((m * 5 + q) * HPS + c * 8), src);
    }
}

// One CTA = NB outputs x one K split x all M tokens. Tokens are staged MC at a time (double-buffered when M > MC);
// the first stage and the weights are requested together at kernel start.
// VAR != 0 only in the nodeC micro-benchmark (tests/mb_smallops.cu): 1 = no weight loads, 2 = no token loop,
// 3 = no residual_out stores, 4 = no warp butterflies. Production instantiates VAR = 0.
template <typename WT, int S, int NB, int MC, int VAR = 0, bool DIRECT = false>
__global__ void __launch_bounds__(256) mhc_fused_kernel(const float* __restrict__ comb, const float* __restrict__ post,
                                                         const bf16* __restrict__ res_in, const bf16* __restrict__ x_in,
                                                         const WT* __restrict__ w, float* __restrict__ yp,
                                                         float* __restrict__ rp, bf16* __restrict__ res_out, int M) {
    constexpr int HPS = HID / S;
    constexpr int ITS = HPS / 256;
    constexpr int STAGE = MC * 5 * HPS;       // bf16 elements per stage
    const int g = blockIdx.x;                 // output group: outputs g*NB .. g*NB+NB-1
    const int ks = blockIdx.y;                // K split
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hbase = ks * HPS + tid;

    constexpr int NBUF = (S == 8 || MC >= 8) ? 1 : 2;   // one stage when the launcher guarantees M <= MC
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* stg = reinterpret_cast<bf16*>(smem_raw);                                  // [NBUF][STAGE]
    float* s_pc = reinterpret_cast<float*>(smem_raw + (DIRECT ? 0 : NBUF * STAGE * sizeof(bf16)));  // [16][20]
    float* s_red = s_pc + MHC_MMAX * (HC + HC * HC);                              // [16][8][NB+1]

    const int nchunks = DIRECT ? 1 : (M + MC - 1) / MC;   // DIRECT: MC >= M, tokens read from global (L2)
    if (!DIRECT) {
        stage_tokens<HPS, MC>(stg, x_in, res_in, 0, M, ks, tid);
        cp_async_commit();
    }

    float wr[NB][HC][ITS];
#pragma unroll
    for (int n = 0; n < NB; ++n)
#pragma unroll
        for (int j = 0; j < HC; ++j)
#pragma unroll
            for (int it = 0; it < ITS; ++it)
                wr[n][j][it] = VAR == 1 ? 0.5f : ldw<WT>(w + ((int64_t)(g * NB + n) * HC + j) * HID + hbase + it * 256);

    for (int i = tid; i < M * (HC + HC * HC); i += 256) {
        const int m = i / (HC + HC * HC), e = i - m * (HC + HC * HC);
        s_pc[i] = e < HC ? post[m * HC + e] : comb[m * HC * HC + (e - HC)];
    }

    for (int c = 0; c < nchunks; ++c) {
        if (DIRECT) {
            __syncthreads();                  // s_pc written above
        } else {
            if (c + 1 < nchunks) {
                stage_tokens<HPS, MC>(stg + ((c + 1) % NBUF) * STAGE, x_in, res_in, (c + 1) * MC, M, ks, tid);
                cp_async_commit();
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
        }
        const bf16* st = stg + (c % NBUF) * STAGE;
        const int mend = min(MC, M - c * MC);
        if (VAR != 2) {
            // All MC token slots are computed unconditionally (slots >= mend hold stale stage data; their results
            // are never stored): full unroll = independent work across tokens for the scheduler (ILP), which the
            // latency-bound butterflies need. Per (token, output) the arithmetic is the TileLang kernel's.
            float acc[MC][NB], sqr[MC];
#pragma unroll
            for (int mm = 0; mm < MC; ++mm) {
                const int m = min(c * MC + mm, M - 1);
                const float* pc = s_pc + m * (HC + HC * HC);
                float pm[HC], cm[HC * HC];
#pragma unroll
                for (int j = 0; j < HC; ++j) pm[j] = pc[j];
#pragma unroll
                for (int e = 0; e < HC * HC; ++e) cm[e] = pc[HC + e];
#pragma unroll
                for (int n = 0; n < NB; ++n) acc[mm][n] = 0.0f;
                sqr[mm] = 0.0f;
#pragma unroll
                for (int it = 0; it < ITS; ++it) {
                    const int hl = it * 256 + tid;
                    float xv, rv[HC];
                    if (DIRECT) {
                        const int mg = min(mm, M - 1);
                        xv = f32(x_in[(int64_t)mg * HID + hbase + it * 256]);
#pragma unroll
                        for (int k = 0; k < HC; ++k) rv[k] = f32(res_in[((int64_t)mg * HC + k) * HID + hbase + it * 256]);
                    } else {
                        xv = f32(st[(mm * 5 + 0) * HPS + hl]);
#pragma unroll
                        for (int k = 0; k < HC; ++k) rv[k] = f32(st[(mm * 5 + 1 + k) * HPS + hl]);
                    }
                    float nr[HC];
#pragma unroll
                    for (int j = 0; j < HC; ++j) {
                        // TileLang source: nr = pm*x; nr += cm[0j]*r0; ... nvcc contracts the first add as
                        // fma(pm, x, cm[0j]*r0) (mhc_fused_tilelang PTX: mul.f32 cm*r0; fma.rn pm, x), then fma per k.
                        nr[j] = __fmaf_rn(pm[j], xv, __fmul_rn(cm[0 * HC + j], rv[0]));
#pragma unroll
                        for (int k = 1; k < HC; ++k) nr[j] = __fmaf_rn(cm[k * HC + j], rv[k], nr[j]);
                    }
                    if (g == 0) {
#pragma unroll
                        for (int j = 0; j < HC; ++j) {
                            if (VAR != 3 && mm < mend)
                                res_out[((int64_t)(c * MC + mm) * HC + j) * HID + hbase + it * 256] = rbf(nr[j]);
                            sqr[mm] = __fmaf_rn(nr[j], nr[j], sqr[mm]);
                        }
                    }
#pragma unroll
                    for (int n = 0; n < NB; ++n)
#pragma unroll
                        for (int j = 0; j < HC; ++j) acc[mm][n] = __fmaf_rn(wr[n][j][it], nr[j], acc[mm][n]);
                }
            }
#pragma unroll
            for (int mm = 0; mm < MC; ++mm) {
                if (VAR != 4) {
#pragma unroll
                    for (int n = 0; n < NB; ++n) acc[mm][n] = warp_sum(acc[mm][n]);
                    if (g == 0) sqr[mm] = warp_sum(sqr[mm]);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (int mm = 0; mm < MC; ++mm) {
                    if (mm < mend) {
                        const int m = c * MC + mm;
#pragma unroll
                        for (int n = 0; n < NB; ++n) s_red[(m * 8 + warp) * (NB + 1) + n] = acc[mm][n];
                        if (g == 0) s_red[(m * 8 + warp) * (NB + 1) + NB] = sqr[mm];
                    }
                }
            }
        }
        __syncthreads();   // the stage buffer is refilled next iteration
    }
    const int nres = NB + (g == 0 ? 1 : 0);
    for (int i = tid; i < M * nres; i += 256) {
        const int m = i / nres, n = i - m * nres;
        float v = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) v = __fadd_rn(v, s_red[(m * 8 + q) * (NB + 1) + n]);
        if (n < NB) yp[((int64_t)ks * M + m) * NOUT + g * NB + n] = v;
        else rp[(int64_t)ks * M + m] = v;
    }
}

template <typename WT, int S, int NB, int MC, int VAR = 0, bool DIRECT = false>
void launch_mhc_fused(const at::Tensor& comb, const at::Tensor& post, const at::Tensor& res_in,
                      const at::Tensor& x_in, const at::Tensor& w, at::Tensor& yp, at::Tensor& rp,
                      at::Tensor& res_out, int M, cudaStream_t st) {
    constexpr int HPS = HID / S;
    constexpr size_t smem = (DIRECT ? 0 : ((S == 8 || MC >= 8) ? 1 : 2) * (size_t)MC * 5 * HPS * sizeof(bf16)) +
                            sizeof(float) * (MHC_MMAX * (HC + HC * HC) + MHC_MMAX * 8 * (NB + 1));
    auto kern = mhc_fused_kernel<WT, S, NB, MC, VAR, DIRECT>;
    static bool attr = false;
    if (!attr) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr = true;
    }
    dim3 grid(NOUT / NB, S);
    kern<<<grid, 256, smem, st>>>(
        comb.data_ptr<float>(), post.data_ptr<float>(), reinterpret_cast<const bf16*>(res_in.data_ptr()),
        reinterpret_cast<const bf16*>(x_in.data_ptr()), reinterpret_cast<const WT*>(w.data_ptr()),
        yp.data_ptr<float>(), rp.data_ptr<float>(), reinterpret_cast<bf16*>(res_out.data_ptr()), M);
}

// ------------------------------------------------------------------------------------------------------------------
// L2 prefetch: read `bytes` at p (16-byte aligned) with an L2 evict_last policy and discard the data. Launched on a
// side stream while the main stream waits in a tensor-parallel all-reduce (DRAM otherwise idle), so the next small
// kernel that needs these bytes (mHC weights) finds them in L2. The weight streams of the big decode kernels use
// evict_first, so these lines survive them. The asm volatile loads cannot be elided; nothing is written.
__global__ void __launch_bounds__(256) l2_prefetch_kernel(const uint4* __restrict__ p, int64_t n16,
                                                           unsigned* __restrict__ sink) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        unsigned a, b, c, d;
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i), "l"(pol));
        acc ^= a ^ b ^ c ^ d;
    }
    // keep the loads alive (ptxas drops loads whose values are dead): a store the compiler cannot prove never
    // happens; it hits a private 4-byte scratch word, never read, in the 2^-32 case that it does
    if (acc == 0x9E3779B9u) *sink = acc;
}

}  // namespace
