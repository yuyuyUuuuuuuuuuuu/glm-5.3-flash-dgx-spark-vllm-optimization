// PF3000 TEST D: e4m3 MoE ceiling probe (plan step 4 exit B kill test).
//
// Extends the E4 ceiling probe (tf-exl3-fork-wt-e4/tools/e4/ceil.cu, whose bf16 numbers this
// reproduces first) with the exit-B mainloop: trellis (EXL3) tile decode -> fp16 fragments ->
// cvt.rn.satfinite.e4m3x2.f16x2 -> pack -> mma.sync.aligned.m16n8k32 (e4m3 in, f32 accumulate),
// all operand live in registers, 16 warps/SM, 128-row tiles (MB=8 16-row blocks per warp).
//
// Geometry (E4's layout): one warp owns a 16-column block of the intermediate dim for all 128 rows of
// one expert tile. Per k32 step it decodes the two 16x16 trellis tiles (k 0..15, k 16..31) of each n8
// half and feeds 8 row-blocks, so the decode-to-flop ratio is identical to E3/E4's bf16 k16 mix
// (FLOP-per-decode 8192*MB in both). Routed-expert shapes per rank (TP=2, moe_intermediate 2048):
//   gate/up N = 2048 (2 x 1024), K = 4096; down N = 4096, K = 1024.
//
// Build/run (nodeC, production image, under flock):
//   tests/gpu_run.sh bash -c 'nvcc -O3 -arch=sm_121a -o /tmp/ceil_e4m3 tests/pf3000/ceil_e4m3.cu && /tmp/ceil_e4m3'
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <string>

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma16832(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}
// fp16x2 -> 2 x e4m3 (rn, saturating), returned in the low 16 bits
__device__ __forceinline__ uint16_t cvt_e4m3x2(uint32_t f16x2) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(f16x2));
    return r;
}
// one decoded 16x16 trellis tile (b0[2]+b1[2] fp16 pairs, 4 fp16/lane per n8 half) -> one k32 e4m3
// B fragment {2 uint32} for that n8 half (the two k16 tiles of the same n8 half are concatenated)
__device__ __forceinline__ void frag_e4m3(const uint32_t (&t0)[2], const uint32_t (&t1)[2], uint32_t (&bf)[2]) {
    uint16_t r0 = cvt_e4m3x2(t0[0]), r1 = cvt_e4m3x2(t0[1]);
    uint16_t r2 = cvt_e4m3x2(t1[0]), r3 = cvt_e4m3x2(t1[1]);
    asm("mov.b32 %0, {%1,%2};" : "=r"(bf[0]) : "h"(r0), "h"(r1));
    asm("mov.b32 %0, {%1,%2};" : "=r"(bf[1]) : "h"(r2), "h"(r3));
}

// pure mma m16n8k16 (bf16 path reference peak): ACC independent accumulators per warp
template <int ACC>
__global__ void k_mma(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    float d[ACC][4] = {};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < ACC; ++j) mma16816(d[j], a, b);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < ACC; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1234.5f) out[0] = s;
}

// pure mma m16n8k32 e4m3 peak: ACC independent accumulators per warp
template <int ACC>
__global__ void k_mmaf8(float* out, int iters) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
    uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
    float d[ACC][4] = {};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < ACC; ++j) mma16832(d[j], a, b);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < ACC; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1234.5f) out[0] = s;
}

// bf16 reference: the E4 probe's mix (1 decode -> 2*MB mma m16n8k16)
template <int MB>
__global__ void k_mixbf16(float* out, const uint32_t* wsrc, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w = wsrc[threadIdx.x & 255];
    uint32_t a[MB][4];
#pragma unroll
    for (int m = 0; m < MB; ++m) { a[m][0] = threadIdx.x + m; a[m][1] = threadIdx.x * 3u; a[m][2] = m; a[m][3] = 7u; }
    float d[MB][2][4] = {};
    for (int i = 0; i < iters; ++i) {
        uint32_t b0[2], b1[2];
        decode_tile(w, lane, b0, b1);
        w = w * 1664525u + 1013904223u;
#pragma unroll
        for (int m = 0; m < MB; ++m) { mma16816(d[m][0], a[m], b0); mma16816(d[m][1], a[m], b1); }
    }
    float s = 0;
#pragma unroll
    for (int m = 0; m < MB; ++m) s += d[m][0][0] + d[m][1][3];
    if (s == 1234.5f) out[0] = s;
}

// exit-B mix, B side converted in registers: per iter 2 decodes (k32 of one n8 half) -> 1 e4m3 B
// fragment -> MB mma m16n8k32 (128-row tile = MB 8). FLOP/decode identical to k_mixbf16.
template <int MB>
__global__ void k_mixf8(float* out, const uint32_t* wsrc, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w = wsrc[threadIdx.x & 255];
    uint32_t a[MB][4];
#pragma unroll
    for (int m = 0; m < MB; ++m) { a[m][0] = threadIdx.x + m; a[m][1] = threadIdx.x * 3u; a[m][2] = m; a[m][3] = 7u; }
    float d[MB][4] = {};
    for (int i = 0; i < iters; ++i) {
        uint32_t ta0[2], ta1[2], tb0[2], tb1[2];
        decode_tile(w, lane, ta0, tb0);
        uint32_t w2 = w * 1664525u + 1013904223u;      // the second k16 tile (as the real mainloop streams it)
        decode_tile(w2, lane, ta1, tb1);
        w = w2 * 1664525u + 1013904223u;
        uint32_t bf[2];
        frag_e4m3(ta0, ta1, bf);
#pragma unroll
        for (int m = 0; m < MB; ++m) mma16832(d[m], a[m], bf);
    }
    float s = 0;
#pragma unroll
    for (int m = 0; m < MB; ++m) s += d[m][0] + d[m][3];
    if (s == 1234.5f) out[0] = s;
}

// worst case: BOTH operands converted in registers (A would come from the fp8 gather in the real
// kernel, this is the upper cost bound if A also had to be re-rounded per k step)
template <int MB>
__global__ void k_mixf8ab(float* out, const uint32_t* wsrc, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w = wsrc[threadIdx.x & 255];
    float d[MB][4] = {};
    for (int i = 0; i < iters; ++i) {
        uint32_t ta0[2], ta1[2], tb0[2], tb1[2];
        decode_tile(w, lane, ta0, tb0);
        uint32_t w2 = w * 1664525u + 1013904223u;
        decode_tile(w2, lane, ta1, tb1);
        w = w2 * 1664525u + 1013904223u;
        uint32_t bfa[2], af[MB][4];
        frag_e4m3(tb0, tb1, bfa);                      // the k32 fragment of the OTHER n8 half -> A operand
        af[0][0] = bfa[0]; af[0][1] = bfa[1]; af[0][2] = bfa[0]; af[0][3] = bfa[1];
#pragma unroll
        for (int m = 1; m < MB; ++m) { af[m][0] = af[m - 1][1]; af[m][1] = af[m - 1][2]; af[m][2] = af[m - 1][3]; af[m][3] = af[m][0]; }
        uint32_t bf[2];
        frag_e4m3(ta0, ta1, bf);
#pragma unroll
        for (int m = 0; m < MB; ++m) mma16832(d[m], af[m], bf);
    }
    float s = 0;
#pragma unroll
    for (int m = 0; m < MB; ++m) s += d[m][0] + d[m][3];
    if (s == 1234.5f) out[0] = s;
}


// REVIEW FIX: k_mixf8 decodes two 16x16 trellis tiles (k32 x n16) per iter but consumes only ONE n8
// half (ta*); tb* is dead code the compiler drops, so its FLOP/decode is not the bf16 mix's and the
// real mainloop (which must use both n8 halves) is not modelled. k_mixf8full uses both halves.
// k_mixf8full_smemA additionally reads the A fragments from shared memory each k32 step (a real
// mainloop cannot keep a 128-row x K activation tile in registers; A comes from the smem-staged gather).
template <int MB, bool SMEMA>
__global__ void k_mixf8full(float* out, const uint32_t* wsrc, int iters) {
    __shared__ uint32_t sa[2][MB][4][32];
    const int lane = threadIdx.x & 31;
    uint32_t w = wsrc[threadIdx.x & 255];
    uint32_t a[MB][4];
#pragma unroll
    for (int m = 0; m < MB; ++m) { a[m][0] = threadIdx.x + m; a[m][1] = threadIdx.x * 3u; a[m][2] = m; a[m][3] = 7u; }
    if (SMEMA) {
        for (int i = threadIdx.x; i < 2 * MB * 4 * 32; i += blockDim.x) (&sa[0][0][0][0])[i] = i * 2654435761u;
        __syncthreads();
    }
    float d[MB][2][4] = {};
    for (int i = 0; i < iters; ++i) {
        uint32_t ta0[2], ta1[2], tb0[2], tb1[2];
        decode_tile(w, lane, ta0, tb0);
        uint32_t w2 = w * 1664525u + 1013904223u;
        decode_tile(w2, lane, ta1, tb1);
        w = w2 * 1664525u + 1013904223u;
        uint32_t bfa[2], bfb[2];
        frag_e4m3(ta0, ta1, bfa);
        frag_e4m3(tb0, tb1, bfb);
#pragma unroll
        for (int m = 0; m < MB; ++m) {
            if (SMEMA) {
#pragma unroll
                for (int j = 0; j < 4; ++j) a[m][j] = sa[i & 1][m][j][lane];
            }
            mma16832(d[m][0], a[m], bfa); mma16832(d[m][1], a[m], bfb);
        }
    }
    float s = 0;
#pragma unroll
    for (int m = 0; m < MB; ++m) s += d[m][0][0] + d[m][1][3];
    if (s == 1234.5f) out[0] = s;
}

// decode only, for reference
__global__ void k_dec(float* out, const uint32_t* wsrc, int iters) {
    const int lane = threadIdx.x & 31;
    uint32_t w = wsrc[threadIdx.x & 255];
    uint32_t acc = 0;
    for (int i = 0; i < iters; ++i) {
        uint32_t b0[2], b1[2];
        decode_tile(w, lane, b0, b1);
        w = w * 1664525u + 1013904223u;
        acc ^= b0[0] ^ b0[1] ^ b1[0] ^ b1[1];
    }
    if (acc == 12345u) out[0] = (float)acc;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device %s SMs %d cc %d.%d regsPerSM %d clock(attr) %d kHz L2 %d\n", p.name, p.multiProcessorCount,
           p.major, p.minor, p.regsPerMultiprocessor, [&]{ int c = 0; cudaDeviceGetAttribute(&c, cudaDevAttrClockRate, 0); return c; }(),
           p.l2CacheSize);
    float* out; cudaMalloc(&out, 4);
    uint32_t* ws; cudaMalloc(&ws, 1024); cudaMemset(ws, 0x5a, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int SMS = p.multiProcessorCount;
    const int ITERS = 20000;
    auto bench = [&](const char* name, auto launch, double flop_per_iter_all_warps) {
        launch(); cudaDeviceSynchronize();
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
        }
        printf("%-46s: %7.2f TFLOPS (%.3f ms)\n", name, flop_per_iter_all_warps / best / 1e9, best);
    };
    for (int wpb : {8, 16}) {
        int blocks = SMS * (wpb == 16 ? 1 : 2);
        // mma m16n8k16 (bf16 path reference peak)
        bench((std::string("mma m16n8k16 f16/f32 wpb ") + std::to_string(wpb)).c_str(),
              [&]{ k_mma<8><<<blocks, wpb * 32>>>(out, ITERS); }, 2.0 * 16 * 8 * 16 * 8.0 * ITERS * wpb * blocks);
        // mma m16n8k32 e4m3 peak
        bench((std::string("mma m16n8k32 e4m3/f32 wpb ") + std::to_string(wpb)).c_str(),
              [&]{ k_mmaf8<8><<<blocks, wpb * 32>>>(out, ITERS); }, 2.0 * 16 * 8 * 32 * 8.0 * ITERS * wpb * blocks);
    }
    {   // decode-only rate for reference
        int wpb = 16, blocks = SMS;
        bench("decode only (16w x 1 blk/SM)", [&]{ k_dec<<<blocks, wpb * 32>>>(out, ws, ITERS * 4); }, 0);
        cudaEventRecord(e0); k_dec<<<blocks, wpb * 32>>>(out, ws, ITERS * 4); cudaEventRecord(e1);
        cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("decode rate: %.2f Gtiles/s\n", 4.0 * ITERS * wpb * blocks / ms / 1e6);
    }
#define MIXBF(MB, WPB, BPS) { int blocks = SMS * BPS; \
        bench(#MB "-row x" #WPB "w mixbf16", [&]{ k_mixbf16<MB><<<blocks, WPB * 32>>>(out, ws, ITERS); }, \
              2.0 * 16 * 8 * 16 * 2.0 * MB * ITERS * WPB * blocks); }
#define MIXF8(MB, WPB, BPS) { int blocks = SMS * BPS; \
        bench(#MB "-row x" #WPB "w mixf8 (B cvt)", [&]{ k_mixf8<MB><<<blocks, WPB * 32>>>(out, ws, ITERS); }, \
              2.0 * 16 * 8 * 32 * 1.0 * MB * ITERS * WPB * blocks); }
    MIXBF(4, 8, 2) MIXBF(4, 16, 1) MIXBF(8, 8, 2) MIXBF(8, 16, 1)
    MIXF8(4, 8, 2) MIXF8(4, 16, 1) MIXF8(8, 8, 2) MIXF8(8, 16, 1)
    {   int wpb = 16, blocks = SMS;
        bench("128-row x16w mixf8ab (A+B cvt, worst case)", [&]{ k_mixf8ab<8><<<blocks, wpb * 32>>>(out, ws, ITERS); },
              2.0 * 16 * 8 * 32 * 1.0 * 8 * ITERS * wpb * blocks);
    }
#define MIXF8F(MB, WPB, BPS, SA) { int blocks = SMS * BPS; \
        bench(#MB "-row x" #WPB "w mixf8FULL both n8 " #SA, [&]{ k_mixf8full<MB, SA><<<blocks, WPB * 32>>>(out, ws, ITERS); }, \
              2.0 * 16 * 8 * 32 * 2.0 * MB * ITERS * WPB * blocks); }
    MIXF8F(4, 16, 1, false) MIXF8F(8, 8, 2, false) MIXF8F(8, 16, 1, false) MIXF8F(8, 16, 1, true)
    cudaError_t err = cudaGetLastError();
    printf("status %s\n", cudaGetErrorString(err));
    return 0;
}
