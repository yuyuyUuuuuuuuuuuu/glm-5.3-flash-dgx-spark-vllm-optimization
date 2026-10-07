// GLM53_DEC_SMALLOPS: small-M BF16 GEMM on tensor cores that reproduces cuBLAS's cutlass_80_wmma_tensorop_bf16
// s161616gemm kernels bit for bit (decode M <= 16), with a multi-stage cp.async weight stream.
//
// y[b, m, n] = bf16( sum_k w[b, n, k] * x[b, m, k] )     (fp32 accumulate; alpha = 1, beta = 0)
//
// Why bitwise: the cuBLAS kernels (16x16_128x1_tn, 32x32_64x2_nn, 32x32_128x2_tn) compute every 16 x 16 output tile
// (16 weight rows x 16 tokens) in one warp as a chain  acc = mma(W[16 rows, k:k+16], X[k:k+16, 16 tokens], acc)  for
// k = 0, 16, 32, ... in order (wmma 16x16x16 = two HMMA.16816 per step, weight rows = the A operand, tokens = B),
// no split-K, then store bf16_rn(1 * acc). This kernel issues the same HMMA.16816 (mma.m16n8k16.row.col, A = weight
// rows, B = tokens) on the same accumulator in the same k order; only how the bytes reach the registers differs
// (a deep cp.async pipeline per CTA instead of one stage per 1-warp CTA). Checked per shape and per M against the
// production call (tests/test_smallops_tc.py; the installer re-checks every served M on the real weight at load).
//
// Weight layouts: KCONTIG (w[b, n, k] k-stride 1: F.linear weights, W_UV) or NCONTIG (n-stride 1: W_UK_T).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace smallops_tc {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp16_zfill(void* s, const void* g, bool valid) {
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(s)), "l"(g), "r"(n));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct TcArgs {
    const bf16* w; int64_t sWb, sWn, sWk;   // exactly one of sWn / sWk is 1
    const bf16* x; int64_t sXb, sXm;        // k stride 1
    bf16* y; int64_t sYb, sYm;              // n stride 1
    int M, N, K;
};

// CTA = WARPS x 16 weight rows of one batch; every warp owns 16 rows and runs the whole K chain for MT x 8 tokens.
// Shared memory per stage (KC = 64 of K): the weight tile [ROWS][KC + 8] (KCONTIG) or [KC][ROWS + 8] (NCONTIG) and
// the token tile [MT * 8][KC + 8] (rows >= M zero-filled); the +8 bf16 pads make every ldmatrix phase conflict-free.
template <bool NCONTIG, int WARPS, int KC, int STAGES, int MT>
__global__ void __launch_bounds__(WARPS * 32) tc_gemm_kernel(const TcArgs a) {
    constexpr int ROWS = WARPS * 16;
    constexpr int NT = WARPS * 32;
    constexpr int WST = NCONTIG ? KC * (ROWS + 8) : ROWS * (KC + 8);   // weight elements per stage
    constexpr int XLD = KC + 8;
    constexpr int XST = MT * 8 * XLD;                                   // token elements per stage
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* sw = reinterpret_cast<bf16*>(smem_raw);
    bf16* sx = sw + STAGES * WST;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b = blockIdx.y;
    const int n0 = blockIdx.x * ROWS;
    const bf16* wb = a.w + (int64_t)b * a.sWb + (int64_t)n0 * (NCONTIG ? 1 : a.sWn);
    const bf16* xb = a.x + (int64_t)b * a.sXb;
    const int nk = a.K / KC;

    auto load_stage = [&](int s, int kc) {
        bf16* dst = sw + s * WST;
        const int k0 = kc * KC;
        if constexpr (!NCONTIG) {
            constexpr int CPR = KC / 8;                    // 16-byte chunks per row
#pragma unroll
            for (int i = tid; i < ROWS * CPR; i += NT) {
                const int r = i / CPR, c = i - r * CPR;
                cp16(dst + r * (KC + 8) + c * 8, wb + (int64_t)r * a.sWn + k0 + c * 8);
            }
        } else {
            constexpr int CPR = ROWS / 8;                  // 16-byte chunks per k row
#pragma unroll
            for (int i = tid; i < KC * CPR; i += NT) {
                const int kk = i / CPR, c = i - kk * CPR;
                cp16(dst + kk * (ROWS + 8) + c * 8, wb + (int64_t)(k0 + kk) * a.sWk + c * 8);
            }
        }
        bf16* xd = sx + s * XST;
        constexpr int XC = KC / 8;
#pragma unroll
        for (int i = tid; i < MT * 8 * XC; i += NT) {
            const int m = i / XC, c = i - m * XC;
            const bool v = m < a.M;
            cp16_zfill(xd + m * XLD + c * 8, xb + (int64_t)(v ? m : 0) * a.sXm + k0 + c * 8, v);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s, s);
        commit();
    }

    float acc[MT][4];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[t][e] = 0.0f;

    for (int kc = 0; kc < nk; ++kc) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nx = kc + STAGES - 1;
            if (nx < nk) load_stage(nx % STAGES, nx);
            commit();
        }
        const bf16* st = sw + (kc % STAGES) * WST;
        const bf16* sxt = sx + (kc % STAGES) * XST;
#pragma unroll
        for (int ks = 0; ks < KC / 16; ++ks) {
            unsigned a0, a1, a2, a3;
            if constexpr (!NCONTIG) {
                const bf16* p = st + (warp * 16 + (lane & 15)) * (KC + 8) + ks * 16 + (lane >> 4) * 8;
                ldsm_x4(a0, a1, a2, a3, p);
            } else {
                const bf16* p = st + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * (ROWS + 8) + warp * 16 +
                                ((lane >> 3) & 1) * 8;
                ldsm_x4_t(a0, a1, a2, a3, p);
            }
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                unsigned b0, b1;
                const bf16* p = sxt + (t * 8 + (lane & 7)) * XLD + ks * 16 + ((lane >> 3) & 1) * 8;
                ldsm_x2(b0, b1, p);
                mma16816(acc[t], a0, a1, a2, a3, b0, b1);
            }
        }
    }
    wait_group<0>();

    // d0, d1: weight row g = lane/4, tokens 2*(lane%4) + {0, 1}; d2, d3: row g + 8.
    const int g = lane >> 2, tq = (lane & 3) * 2;
    bf16* yb = a.y + (int64_t)b * a.sYb + n0 + warp * 16;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int m = t * 8 + tq + (e & 1);
            const int n = g + (e >> 1) * 8;
            if (m < a.M) yb[(int64_t)m * a.sYm + n] = __float2bfloat16_rn(acc[t][e]);
        }
    }
}

template <bool NCONTIG, int WARPS, int KC, int STAGES, int MT>
inline size_t tc_smem_bytes(int K) {
    constexpr int ROWS = WARPS * 16;
    constexpr int WST = NCONTIG ? KC * (ROWS + 8) : ROWS * (KC + 8);
    (void)K;
    return sizeof(bf16) * (size_t)STAGES * (WST + (size_t)MT * 8 * (KC + 8));
}

template <bool NCONTIG, int WARPS, int KC, int STAGES, int MT>
inline cudaError_t tc_launch(const TcArgs& a, int batch, cudaStream_t st) {
    auto kern = tc_gemm_kernel<NCONTIG, WARPS, KC, STAGES, MT>;
    const size_t smem = tc_smem_bytes<NCONTIG, WARPS, KC, STAGES, MT>(a.K);
    static int attr_set = 0;
    if ((size_t)attr_set < smem) {
        cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return e;
        attr_set = (int)smem;
    }
    dim3 grid(a.N / (WARPS * 16), batch);
    // cudaLaunchKernel reports this launch's own status (a <<<>>> launch + cudaGetLastError would also return an
    // unrelated error recorded earlier by someone else, e.g. the torch profiler)
    TcArgs arg = a;
    void* params[] = {&arg};
    return cudaLaunchKernel(reinterpret_cast<const void*>(kern), grid, dim3(WARPS * 32), params, smem, st);
}

}  // namespace smallops_tc
