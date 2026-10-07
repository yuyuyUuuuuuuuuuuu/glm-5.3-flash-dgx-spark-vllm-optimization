// GLM-5.3-Flash's EXL3 routed experts on CUDA, as a drop-in for ExLlamaV3's fused decode kernel
// `exllamav3_ext.exl3_moe` (the production vLLM decode path).
//
// Kernels: TensorFold (MIT, Copyright (c) 2026 TensorFold contributors, github.com/ashhart/TensorFold, families/glm5_next/cuda/exl3.cu), adapted here per
// docs/DESIGN.md §A: per-expert weights come from the production int64 pointer tables (zero copy), rows are
// sorted pair indices j (0 <= j < P = token_sorted.numel()), a segment table replaces the member table, the
// routing weight and the accumulation into the fp32 output happen in the down epilogue, and the elementwise
// precision follows exl3_moe (ExLlamaV3, MIT, Copyright (c) 2025 Turboderp) when TF_PARITY=1 (default).
//
// Format: a 16x16 tile is 32 little-endian 32-bit words (= the int16 [.., 64] trellis viewed as uint32);
// lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values are
// exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile).
//
// Per call (all on the current stream, grids depend on shapes only; no host sync, CUDA-graph capturable):
//   route_prep      grid (1)                  block 1024   pair_expert[P], segment table, nseg
//   rot_in          grid (P, K/128, 2)        block 32     xg, xu = fp16(H(fp16(x[t] * suh)) * r)
//   grouped g/u     grid (S_cap, N/64, 2*4)   block 128    Z[mat][split][j][:] = xg/xu[j] @ Wq (fp32)
//   gateup_epilogue grid (P, N/128)           block 32     xd = fp16(H(fp16(silu(g)*u) * suh_d) * r)
//   grouped down    grid (S_cap, K/64, 1)     block 128    Z[0][0][j][:] = xd[j] @ Wq_down
//   down_epilogue   grid (P, K/128)           block 32     out[t] += (H(fp16(Z)) * (r*w)) * svh_d (atomic)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef TF_PARITY
#define TF_PARITY 1
#endif

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
// ExLlamaV3 writes the same constant as 0.088388347648f (xl_exl3_moe_kernel.cuh:88); both are one float.
static_assert(HAD_SCALE == 0.088388347648f, "Hadamard scale must equal ExLlamaV3's r_scale bit for bit");

// ---------------------------------------------------------------------------------------------------------
// Trellis decode + mma (TensorFold, unchanged)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
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

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// ---------------------------------------------------------------------------------------------------------
// Grouped trellis GEMV over segments (A1, A4, A10).
//
// Program (segment s, n block, mat * SK + split): the <= 16 consecutive pair rows of segment s (all of one
// expert e = seg_expert[s]) times W_q of matrix `mat` of expert e over this split's K range, for NT*16
// columns; Z[mat][split][j][n]. The weight comes from the pointer table: Tp_mat[e] is the base of that
// expert's int16 [K/16, N/16, 64] trellis (= uint32 [K/16, N/16, 32]). Warp w of W runs a fixed contiguous
// range of the split's k tiles; warps are added in order 0..W-1. Every (row of the segment, column, split)
// is written, so every Z row an epilogue reads (pair_expert[j] >= 0) was written in this call (A10).
//
// Template knobs (docs/OPTIMIZATION.md); none of them changes a single arithmetic operation or its order, so Z is
// bit-identical across all instances:
//   ORD   which grid dimension carries what. 0: (segment, n block, mat*SK+split) = TensorFold's order, blocks of
//         different experts are adjacent in launch order. 1: (n block, segment, mat*SK+split): the N/(16*NT)
//         blocks that read the same k rows of one matrix are adjacent, so co-resident blocks stream whole
//         8 KiB trellis rows instead of 512-byte pieces of many experts' matrices (DRAM locality).
//         2: (n block, mat*SK+split, segment): all blocks of one segment adjacent, empty segments last.
//   PIPE  1: the k loop is double-buffered (weights and x fragments of tile kt+1 are loaded before tile kt is
//         decoded); needs an even per-warp tile count (host-checked).
//   LEAN  1: the cross-warp reduction goes through one 16 x (NT*16) buffer, warps adding in order 0..W-1
//         (((w0+w1)+w2)+w3, the same order as LEAN 0), the last warp storing Z from registers.
//   POL   1: weight loads carry an L2 evict_first policy (streamed once per call).
template <int POL>
__device__ __forceinline__ uint32_t ld_weight(const uint32_t* p, uint64_t pol) {
    if (POL) {
        uint32_t v;
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.b32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
        return v;
    }
    return __ldg(p);
}

template <int NT, int W, int ORD, int PIPE, int LEAN, int POL>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ Tp0,
    const int64_t* __restrict__ Tp1, const int* __restrict__ seg_expert, const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows, const int* __restrict__ nseg, float* __restrict__ Z, int K, int N, int P,
    int SK) {
    const int s = ORD == 0 ? blockIdx.x : ORD == 1 ? blockIdx.y : blockIdx.z;
    const int nblk = ORD == 0 ? blockIdx.y : blockIdx.x;
    const int zidx = ORD == 2 ? blockIdx.y : blockIdx.z;
    if (s >= nseg[0]) return;                                  // block-uniform
    const int split = zidx % SK, mat = zidx / SK;
    const int e = seg_expert[s], r0s = seg_row0[s], rn = seg_rows[s];
    const half* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>((mat ? Tp1 : Tp0)[e]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    uint64_t pol = 0;
    if (POL) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < rn ? r0s + (int)threadIdx.x : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = nblk * NT;
    const uint32_t* tile = T + ((size_t)kt0 * NTILES + nt0) * 32 + lane;
    const size_t kstride = (size_t)NTILES * 32;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    auto load_x = [&](int kt, uint32_t (&a)[4]) {
        const int k = kt * 16;
        a[0] = load_pair(x0 + k, r0 >= 0);
        a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0);
        a[3] = load_pair(x1 + k + 8, r1 >= 0);
    };
    auto mma_tile = [&](const uint32_t (&words)[NT], const uint32_t (&a)[4]) {
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
    };

    if (PIPE) {
        uint32_t wa[NT], wb[NT], xa[4], xb[4];
#pragma unroll
        for (int i = 0; i < NT; ++i) wa[i] = ld_weight<POL>(tile + i * 32, pol);
        load_x(kt0, xa);
#pragma unroll 1
        for (int kt = kt0; kt < kt0 + per_warp; kt += 2) {
            const uint32_t* tb = tile + kstride;
#pragma unroll
            for (int i = 0; i < NT; ++i) wb[i] = ld_weight<POL>(tb + i * 32, pol);
            load_x(kt + 1, xb);
            mma_tile(wa, xa);
            tile += 2 * kstride;
            if (kt + 2 < kt0 + per_warp) {
#pragma unroll
                for (int i = 0; i < NT; ++i) wa[i] = ld_weight<POL>(tile + i * 32, pol);
                load_x(kt + 2, xa);
            }
            mma_tile(wb, xb);
        }
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = ld_weight<POL>(tile + i * 32, pol);
            uint32_t a[4];
            load_x(kt, a);
            mma_tile(words, a);
            tile += kstride;
        }
    }

    if (LEAN) {
        // warps add into one buffer in order 0..W-1; the last warp adds its own partials and stores Z
        __shared__ float rb[16][NT * 16];
#pragma unroll
        for (int w = 0; w < W - 1; ++w) {
            if (warp == w) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        if (w == 0) {
                            rb[g][col] = acc[i][h][0];
                            rb[g][col + 1] = acc[i][h][1];
                            rb[g + 8][col] = acc[i][h][2];
                            rb[g + 8][col + 1] = acc[i][h][3];
                        } else {
                            rb[g][col] += acc[i][h][0];
                            rb[g][col + 1] += acc[i][h][1];
                            rb[g + 8][col] += acc[i][h][2];
                            rb[g + 8][col + 1] += acc[i][h][3];
                        }
                    }
            }
            __syncthreads();
        }
        if (warp == W - 1) {
            float* zb = Z + ((size_t)mat * SK + split) * P * N + nt0 * 16;
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    if (r0 >= 0)
                        *reinterpret_cast<float2*>(zb + (size_t)r0 * N + col) =
                            make_float2(rb[g][col] + acc[i][h][0], rb[g][col + 1] + acc[i][h][1]);
                    if (r1 >= 0)
                        *reinterpret_cast<float2*>(zb + (size_t)r1 * N + col) =
                            make_float2(rb[g + 8][col] + acc[i][h][2], rb[g + 8][col + 1] + acc[i][h][3]);
                }
        }
        return;
    }

    // warps' partial sums through shared memory, added in warp order
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float sum = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) sum += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = sum;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Hadamard + elementwise stages

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes. Bit-identical to ExLlamaV3's
// shuffle_had_f4x32 after its 4-element Hadamard (same additions in the same order; -a + b == b - a).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

struct H4 { half2 lo, hi; };   // four halves, 8 bytes

__device__ __forceinline__ H4 ld_h4(const half* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    H4 h;
    h.lo = *reinterpret_cast<const half2*>(&r.x);
    h.hi = *reinterpret_cast<const half2*>(&r.y);
    return h;
}

__device__ __forceinline__ void st_h4(half* p, half2 lo, half2 hi) {
    uint2 r;
    r.x = *reinterpret_cast<const uint32_t*>(&lo);
    r.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(p) = r;
}

__device__ __forceinline__ const half* hptr(const int64_t* table, int e) {
    return reinterpret_cast<const half*>(table[e]);
}

// Four hidden-state values as fp16. fp16 input: as stored. bf16 input (K2, the production hidden state before its
// x2d.contiguous().half()): each value converted exactly as that .half() does, fp16(RN(float(bf16))).
__device__ __forceinline__ H4 ld_x4(const half* p) { return ld_h4(p); }
__device__ __forceinline__ H4 ld_x4(const __nv_bfloat16* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&r.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&r.y);
    H4 h;
    h.lo = __floats2half2_rn(__low2float(a), __high2float(a));
    h.hi = __floats2half2_rn(__low2float(b), __high2float(b));
    return h;
}

// A7. Program (pair row j, 128-block of K, matrix): Xmat[j][block] = fp16(H(x[t] * suh_mat[e]) * r), t = token_sorted[j].
template <typename XT>
__global__ void rot_in_kernel(const XT* __restrict__ x, int64_t x_stride, const int64_t* __restrict__ token_sorted,
                              const int* __restrict__ pair_expert, const int64_t* __restrict__ suh_p0,
                              const int64_t* __restrict__ suh_p1, half* __restrict__ out0, half* __restrict__ out1,
                              int K, int64_t B) {
    const int j = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int lane = threadIdx.x;
    const int e = pair_expert[j];
    if (e < 0) return;
    const int64_t t = token_sorted[j];
    if (t < 0 || t >= B) return;                               // REQ-S (A6)
    const int c0 = blk * 128 + 4 * lane;
    const H4 xv = ld_x4(x + t * x_stride + c0);
    const H4 sv = ld_h4(hptr(mat ? suh_p1 : suh_p0, e) + c0);
    float v[4];
#if TF_PARITY
    // exl3_moe: fp16 pre-scale (__hmul2), then fp32 Hadamard (xl_hadamard_inner.cuh:106-118)
    const half2 p01 = __hmul2(xv.lo, sv.lo), p23 = __hmul2(xv.hi, sv.hi);
    v[0] = __low2float(p01); v[1] = __high2float(p01); v[2] = __low2float(p23); v[3] = __high2float(p23);
#else
    v[0] = __low2float(xv.lo) * __low2float(sv.lo);
    v[1] = __high2float(xv.lo) * __high2float(sv.lo);
    v[2] = __low2float(xv.hi) * __low2float(sv.hi);
    v[3] = __high2float(xv.hi) * __high2float(sv.hi);
#endif
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)j * K + c0;
    st_h4(o, __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE), __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE));
}

#if TF_PARITY
__device__ __forceinline__ half2 silu_h2(half2 x) {   // xl_hadamard_inner.cuh:320-328
    const half2 one = __float2half2_rn(1.0f);
    const half2 e = h2exp(__hneg2(x));
    const half2 r = h2rcp(__hadd2(one, e));
    return __hmul2(x, r);
}
#endif

// A8. Program (pair row j, 128-block of N): gate and up outputs summed over the splits in order, rotated,
// scaled by svh; SiLU(g) then the limit (u to [-L, L], g upper only, only if L != 0) exactly as exl3_moe;
// a = g * u * suh_d; Xd[j][block] = fp16(H(a) * r).
// K3b (docs/OPTIMIZATION.md): drop 128-byte L2 lines that no one reads again in this call without writing them back
// (the next writer is stream-ordered after the dropping kernel). After the call their contents are undefined.
__device__ __forceinline__ void l2_discard(const void* p) {
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
}

// dead (discard != 0, forward path only): after its reads, the block drops the Z lines it read (row j, its 128
// columns, every split of both matrices) and its share of row j of xg / xu (read only by the finished gate/up GEMV).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pair_expert,
                                       const int64_t* __restrict__ svh_pg, const int64_t* __restrict__ svh_pu,
                                       const int64_t* __restrict__ suh_pd, half* __restrict__ xd, int P, int N,
                                       int SK, float limit, const half* __restrict__ dead_xg,
                                       const half* __restrict__ dead_xu, int K, int discard) {
    const int j = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pair_expert[j];
    if (e < 0) return;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {                          // fixed order
            sg += Z[((size_t)(0 * SK + s) * P + j) * N + n + i];
            su += Z[((size_t)(1 * SK + s) * P + j) * N + n + i];
        }
#if TF_PARITY
        gv[i] = __half2float(__float2half_rn(sg));              // exl3_moe keeps g0/u0 in fp16 temps
        uv[i] = __half2float(__float2half_rn(su));
#else
        gv[i] = sg;
        uv[i] = su;
#endif
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    if (discard) {                                              // every lane's Z loads are consumed above
        __syncwarp();
        for (int l = lane; l < 2 * SK * 4; l += 32)             // 4 lines of 128 B per (matrix, split) plane
            l2_discard(Z + ((size_t)(l >> 2) * P + j) * N + blk * 128 + (l & 3) * 32);
        const int nb = N >> 7, lines = K >> 6;                  // row j of xg / xu: K * 2 bytes
        for (int l = blk + nb * lane; l < lines; l += nb * 32) {
            l2_discard(dead_xg + (size_t)j * K + l * 64);
            l2_discard(dead_xu + (size_t)j * K + l * 64);
        }
    }
    const H4 sg = ld_h4(hptr(svh_pg, e) + n);
    const H4 su = ld_h4(hptr(svh_pu, e) + n);
    const H4 sd = ld_h4(hptr(suh_pd, e) + n);
    float v[4];
#if TF_PARITY
    half2 g01 = __floats2half2_rn(gv[0] * HAD_SCALE, gv[1] * HAD_SCALE);
    half2 g23 = __floats2half2_rn(gv[2] * HAD_SCALE, gv[3] * HAD_SCALE);
    half2 u01 = __floats2half2_rn(uv[0] * HAD_SCALE, uv[1] * HAD_SCALE);
    half2 u23 = __floats2half2_rn(uv[2] * HAD_SCALE, uv[3] * HAD_SCALE);
    g01 = __hmul2(g01, sg.lo); g23 = __hmul2(g23, sg.hi);       // fp16 post-scale (xl :352-355)
    u01 = __hmul2(u01, su.lo); u23 = __hmul2(u23, su.hi);
    g01 = silu_h2(g01); g23 = silu_h2(g23);
    if (limit != 0.0f) {                                        // xl :375-383
        const half2 lo = __float2half2_rn(-limit), hi = __float2half2_rn(limit);
        u01 = __hmax2(u01, lo); u23 = __hmax2(u23, lo);
        u01 = __hmin2(u01, hi); u23 = __hmin2(u23, hi);
        g01 = __hmin2(g01, hi); g23 = __hmin2(g23, hi);
    }
    g01 = __hmul2(g01, u01); g23 = __hmul2(g23, u23);
    g01 = __hmul2(g01, sd.lo); g23 = __hmul2(g23, sd.hi);
    v[0] = __low2float(g01); v[1] = __high2float(g01); v[2] = __low2float(g23); v[3] = __high2float(g23);
#else
    const float svg[4] = {__low2float(sg.lo), __high2float(sg.lo), __low2float(sg.hi), __high2float(sg.hi)};
    const float svu[4] = {__low2float(su.lo), __high2float(su.lo), __low2float(su.hi), __high2float(su.hi)};
    const float sud[4] = {__low2float(sd.lo), __high2float(sd.lo), __low2float(sd.hi), __high2float(sd.hi)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float g = gv[i] * HAD_SCALE * svg[i];
        float u = uv[i] * HAD_SCALE * svu[i];
        g = g / (1.f + expf(-g));
        if (limit != 0.0f) {
            u = fminf(fmaxf(u, -limit), limit);
            g = fminf(g, limit);
        }
        v[i] = g * u * sud[i];
    }
#endif
    fwht128(v, lane);
    st_h4(xd + (size_t)j * N + n, __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE),
          __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE));
}

// A5. Program (pair row j, 128-block of D): the down projection's output summed over the splits in order,
// rotated, times (r * w_j) and svh_d, added atomically into out[t] (fp32, this rank's partial sum).
// dead (discard != 0, forward path only): after its reads, the block drops the Z lines it read (row j, its 128
// columns, every split) and its share of row j of xd (read only by the finished down GEMV).
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pair_expert,
                                     const int64_t* __restrict__ token_sorted, const half* __restrict__ weight_sorted,
                                     const int64_t* __restrict__ svh_pd, float* __restrict__ out, int P, int D,
                                     int SK, int64_t B, const half* __restrict__ dead_xd, int N, int discard) {
    const int j = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const int e = pair_expert[j];
    if (e < 0) return;
    const int64_t t = token_sorted[j];
    if (t < 0 || t >= B) return;                               // REQ-S (A6)
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + j) * D + n + i];
#if TF_PARITY
        v[i] = __half2float(__float2half_rn(s));                // exl3_moe stores d0 as fp16
#else
        v[i] = s;
#endif
    }
    fwht128(v, lane);
    if (discard) {                                              // every lane's Z loads are consumed above
        __syncwarp();
        for (int l = lane; l < SK * 4; l += 32)
            l2_discard(Z + ((size_t)(l >> 2) * P + j) * D + blk * 128 + (l & 3) * 32);
        const int nb = D >> 7, lines = N >> 6;                  // row j of xd: N * 2 bytes
        for (int l = blk + nb * lane; l < lines; l += nb * 32) l2_discard(dead_xd + (size_t)j * N + l * 64);
    }
    const float rs = 0.088388347648f * __half2float(weight_sorted[j]);   // xl_exl3_moe_kernel.cuh:240
    const H4 sv = ld_h4(hptr(svh_pd, e) + n);
    v[0] *= rs; v[1] *= rs; v[2] *= rs; v[3] *= rs;                    // xl_hadamard_inner.cuh:433-436
    v[0] *= __low2float(sv.lo); v[1] *= __high2float(sv.lo);          // :441-444
    v[2] *= __low2float(sv.hi); v[3] *= __high2float(sv.hi);
    float* o = out + t * (int64_t)D + n;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(reinterpret_cast<float4*>(o), make_float4(v[0], v[1], v[2], v[3]));   // 16 B vector atomic
#else
    atomicAdd(o + 0, v[0]); atomicAdd(o + 1, v[1]); atomicAdd(o + 2, v[2]); atomicAdd(o + 3, v[3]);
#endif
}

// ---------------------------------------------------------------------------------------------------------
// A9 / §B. route_prep: one block of 1024 threads, no atomics, deterministic.

template <typename T>
__device__ __forceinline__ T block_incl_scan(T v, T* tot, T& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const T y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
    }
    if (lane == 31) tot[warp] = v;
    __syncthreads();
    if (warp == 0) {
        T w = lane < nw ? tot[lane] : T(0);
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const T y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < nw) tot[lane] = w;
    }
    __syncthreads();
    const T prefix = warp ? tot[warp - 1] : T(0);
    total = tot[nw - 1];
    __syncthreads();                                           // tot may be reused by the next scan
    return v + prefix;
}

// §B.2: start[e] = sum_{i<e} c[i]; elig[e] = c>0 && c<=R && start+c<=P; ceil(c/16) segments per eligible
// expert; pair_expert[j] = e on eligible spans, -1 elsewhere (sentinel bucket, over-cap, beyond the counts).
__global__ void __launch_bounds__(1024) route_prep_kernel(const int64_t* __restrict__ count, int n, int P, int R,
                                                          int S_cap, int* __restrict__ pair_expert,
                                                          int* __restrict__ seg_expert, int* __restrict__ seg_row0,
                                                          int* __restrict__ seg_rows, int* __restrict__ nseg_out) {
    __shared__ long long tot64[32];
    __shared__ int tot32[32];
    const int tid = threadIdx.x;
    for (int j = tid; j < P; j += blockDim.x) pair_expert[j] = -1;
    __syncthreads();                                           // §B.3 step 3: -1 fill before any "= e"
    long long run_start = 0;                                   // carried across chunks of blockDim experts
    int run_seg = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int e = base + tid;
        const long long c = e < n ? (long long)count[e] : 0LL;
        long long ctot;
        const long long cinc = block_incl_scan<long long>(c, tot64, ctot);
        const long long start = run_start + cinc - c;
        const bool elig = e < n && c > 0 && c <= (long long)R && start >= 0 && start + c <= (long long)P;
        const int ns = elig ? (int)((c + 15) >> 4) : 0;
        int stot;
        const int sinc = block_incl_scan<int>(ns, tot32, stot);
        const int soff = run_seg + sinc - ns;
        if (elig) {
            for (int q = 0; q < ns; ++q) {
                const int s = soff + q;
                if (s < S_cap) {                               // never false (S_cap bound, §B.1); REQ-S
                    seg_expert[s] = e;
                    seg_row0[s] = (int)start + 16 * q;
                    seg_rows[s] = min(16, (int)c - 16 * q);
                }
            }
            for (int i = 0; i < (int)c; ++i) pair_expert[start + i] = e;
        }
        run_start += ctot;
        run_seg += stot;
    }
    if (tid == 0) nseg_out[0] = run_seg < S_cap ? run_seg : S_cap;
}

// K2 (docs/OPTIMIZATION.md): production's decode routing prelude and route_prep in one block. Production
// (apply_exl3_fused_moe, docs/prod_exl3_reference.py:1616-1633) maps ids to local experts (map_topk_to_local), sorts
// the pairs j = t * topk + k by local expert (argsort), gathers token_sorted = j / topk and weight_sorted =
// fp16(weights[j]), and counts expert_count by scatter_add; route_prep then builds the tables of §B.2 from the counts.
// Here: local ids by exactly map_topk_to_local's rules (sentinel n), counts in shared memory, the §B.2 scan and
// segments (route_prep's code and rules), and every pair placed at start[local] + (its rank inside the group). The
// sentinel group is last, as in production. The order inside a group is not specified (production's argsort is not
// stable either): it only permutes rows that are computed independently of each other, and changes nothing but the
// order of down_epilogue's fp32 atomics. P <= blockDim (1024) and n <= 4096 (shared counts), host-checked.
__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename WT>
__global__ void __launch_bounds__(1024) route_ids_kernel(
    const int64_t* __restrict__ ids, int has_map, const int64_t* __restrict__ emap, int64_t n_global,
    const WT* __restrict__ w, int P, int topk, int n, int R, int S_cap, int* __restrict__ pair_expert, int* __restrict__ seg_expert,
    int* __restrict__ seg_row0, int* __restrict__ seg_rows, int* __restrict__ nseg_out,
    int64_t* __restrict__ token_sorted, half* __restrict__ weight_sorted) {
    extern __shared__ int sh_counts[];                         // cnt[n + 1], then cur[n + 1]
    int* cnt = sh_counts;
    int* cur = sh_counts + (n + 1);
    __shared__ int tot32[32];
    const int tid = threadIdx.x;
    for (int e = tid; e <= n; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    int loc = n;
    if (tid < P) {
        const int64_t id = ids[tid];
        if (!has_map) {                                        // map_topk_to_local, expert_map None
            loc = (id < 0 || id >= n) ? n : (int)id;
        } else {                                               // map_topk_to_local with an expert map
            const int64_t hi = n_global > 0 ? n_global - 1 : 0;
            const int64_t safe = id < 0 ? 0 : (id > hi ? hi : id);
            const int64_t mapped = n_global ? emap[safe] : (int64_t)n;   // an empty map: every id non-local
            loc = (id < 0 || id >= n_global || mapped < 0 || mapped >= n) ? n : (int)mapped;
        }
        atomicAdd(&cnt[loc], 1);
    }
    __syncthreads();
    int run_start = 0, run_seg = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int e = base + tid;
        const int c = e < n ? cnt[e] : 0;
        int ctot;
        const int cinc = block_incl_scan<int>(c, tot32, ctot);
        const int start = run_start + cinc - c;
        // route_prep's rule; start + c <= P always holds here (every pair is counted, the sentinel group last)
        const bool elig = e < n && c > 0 && c <= R;
        const int ns = elig ? (c + 15) >> 4 : 0;
        int stot;
        const int sinc = block_incl_scan<int>(ns, tot32, stot);
        const int soff = run_seg + sinc - ns;
        if (e < n) cur[e] = start;
        if (elig) {
            for (int q = 0; q < ns; ++q) {
                const int s = soff + q;
                if (s < S_cap) {                               // never false (S_cap bound, §B.1)
                    seg_expert[s] = e;
                    seg_row0[s] = start + 16 * q;
                    seg_rows[s] = min(16, c - 16 * q);
                }
            }
        }
        run_start += ctot;
        run_seg += stot;
    }
    if (tid == 0) {
        cur[n] = run_start;                                    // the sentinel group follows every real expert
        nseg_out[0] = run_seg < S_cap ? run_seg : S_cap;
    }
    __syncthreads();
    if (tid < P) {
        const int pos = atomicAdd(&cur[loc], 1);
        token_sorted[pos] = tid / topk;
        weight_sorted[pos] = __float2half_rn(to_f32(w[tid]));  // weights.reshape(-1).to(fp16)
        pair_expert[pos] = (loc < n && cnt[loc] <= R) ? loc : -1;
    }
}

// ---------------------------------------------------------------------------------------------------------
// moeglue (docs/DEC_MOEGLUE.md, GLM53_DEC_MOEGLUE): production's whole decode apply_exl3_experts (the int32 -> int64
// copy of the router ids, torch.zeros(out), K2's route_ids + rot_in + grouped(g,u) + gateup_epilogue + grouped(d) +
// down_epilogue, out.to(x.dtype)) in five launches:
//   glue_prep    1-D grid of P * (K / 512) + 1 blocks x 128 threads. Block (pair j, column group kb): the local id of
//                every pair (map_topk_to_local's rules, ids read as int32 or int64), the STABLE sorted position of pair
//                j (pairs grouped by local expert, the sentinel group last, inside a group by pair index), and rows
//                pos of xg / xu for 4 x 128 columns = rot_in's arithmetic exactly; block kb = 0 also writes
//                token_sorted / weight_sorted / pair_expert at pos and inv[j] = pos, and (optionally) prefetches the
//                expert's epilogue vectors into L2. The last block builds the §B.2 segment table (route_ids' code).
//   grouped g/u, gateup_epilogue, grouped d: unchanged (they only see sorted rows).
//   glue_finish  grid (B, K/128) x 32 * topk threads: warp k computes pair t*topk+k's down epilogue exactly as
//                down_epilogue does, the block adds the valid pairs in slot order (fp32, from 0) and stores x.dtype.
// Per pair every value up to the fp32 contribution is bit-identical to K2 (same operations in the same order on the
// same Z rows: the grouped GEMV computes rows independently, so a row's position in the sorted table does not change
// it). The only difference is the order of the fp32 additions of a token's topk contributions: K2 adds them with
// atomics in arrival order (not deterministic run to run), glue_finish in slot order (deterministic). One valid pair
// per token -> bit-identical output.
template <typename IT>
__device__ __forceinline__ int glue_local_id(IT raw, int has_map, const int64_t* __restrict__ emap, int64_t n_global,
                                             int n) {
    const int64_t id = (int64_t)raw;                           // .to(torch.long): exact for int32 ids
    if (!has_map) return (id < 0 || id >= n) ? n : (int)id;  // map_topk_to_local, expert_map None
    const int64_t hi = n_global > 0 ? n_global - 1 : 0;
    const int64_t safe = id < 0 ? 0 : (id > hi ? hi : id);
    const int64_t mapped = n_global ? emap[safe] : (int64_t)n;
    return (id < 0 || id >= n_global || mapped < 0 || mapped >= n) ? n : (int)mapped;
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

constexpr int GLUE_WARPS = 4;                                  // warps (= 128-column blocks) per glue_prep block

template <typename XT, typename IT, typename WT>
__global__ void __launch_bounds__(GLUE_WARPS * 32) glue_prep_kernel(
    const XT* __restrict__ x, int64_t x_stride, const IT* __restrict__ ids, int has_map,
    const int64_t* __restrict__ emap, int64_t n_global, const WT* __restrict__ w, int P, int topk, int n, int R,
    int S_cap, int K, const int64_t* __restrict__ suh_p0, const int64_t* __restrict__ suh_p1,
    const int64_t* __restrict__ pf_g, const int64_t* __restrict__ pf_u, const int64_t* __restrict__ pf_sd,
    const int64_t* __restrict__ pf_vd, int N, int prefetch, half* __restrict__ out0, half* __restrict__ out1,
    int* __restrict__ pair_expert, int* __restrict__ seg_expert, int* __restrict__ seg_row0,
    int* __restrict__ seg_rows, int* __restrict__ nseg_out, int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted, int* __restrict__ inv) {
    extern __shared__ int sh_cnt[];                            // segment block: cnt[n + 1]
    __shared__ int tot32[32];
    __shared__ int red[GLUE_WARPS][3];
    const int KB = K / (128 * GLUE_WARPS);
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (b == P * KB) {                                         // the §B.2 segment table (route_ids' rules)
        for (int e = tid; e <= n; e += blockDim.x) sh_cnt[e] = 0;
        __syncthreads();
        for (int i = tid; i < P; i += blockDim.x) atomicAdd(&sh_cnt[glue_local_id(ids[i], has_map, emap, n_global, n)], 1);
        __syncthreads();
        int run_start = 0, run_seg = 0;
        for (int base = 0; base < n; base += blockDim.x) {
            const int e = base + tid;
            const int c = e < n ? sh_cnt[e] : 0;
            int ctot;
            const int cinc = block_incl_scan<int>(c, tot32, ctot);
            const int start = run_start + cinc - c;
            const bool elig = e < n && c > 0 && c <= R;
            const int ns = elig ? (c + 15) >> 4 : 0;
            int stot;
            const int sinc = block_incl_scan<int>(ns, tot32, stot);
            const int soff = run_seg + sinc - ns;
            if (elig) {
                for (int q = 0; q < ns; ++q) {
                    const int s = soff + q;
                    if (s < S_cap) {                           // never false (S_cap bound, §B.1)
                        seg_expert[s] = e;
                        seg_row0[s] = start + 16 * q;
                        seg_rows[s] = min(16, c - 16 * q);
                    }
                }
            }
            run_start += ctot;
            run_seg += stot;
        }
        if (tid == 0) nseg_out[0] = run_seg < S_cap ? run_seg : S_cap;
        return;
    }
    const int j = b / KB, kb = b - (b / KB) * KB;
    const int me = glue_local_id(ids[j], has_map, emap, n_global, n);
    const int t = j / topk;
    const int c0 = (kb * GLUE_WARPS + warp) * 128 + 4 * lane;
    // this pair's loads first (independent of its sorted position): x[t] and the two suh vectors of expert `me`
    H4 xv, s0, s1;
    if (me < n) {
        xv = ld_x4(x + (int64_t)t * x_stride + c0);
        s0 = ld_h4(hptr(suh_p0, me) + c0);
        s1 = ld_h4(hptr(suh_p1, me) + c0);
    }
    // stable position: pairs of a smaller local id, then same local id and smaller pair index
    int less = 0, eqb = 0, cnt = 0;
    for (int i = tid; i < P; i += blockDim.x) {
        const int l = glue_local_id(ids[i], has_map, emap, n_global, n);
        less += l < me;
        eqb += (l == me) & (i < j);
        cnt += l == me;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        less += __shfl_xor_sync(0xffffffffu, less, o);
        eqb += __shfl_xor_sync(0xffffffffu, eqb, o);
        cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    }
    if (lane == 0) {
        red[warp][0] = less;
        red[warp][1] = eqb;
        red[warp][2] = cnt;
    }
    __syncthreads();
    less = eqb = cnt = 0;
#pragma unroll
    for (int q = 0; q < GLUE_WARPS; ++q) {
        less += red[q][0];
        eqb += red[q][1];
        cnt += red[q][2];
    }
    const int pos = less + eqb;
    const int pe = (me < n && cnt <= R) ? me : -1;             // route_prep / route_ids: over-cap and sentinel -> -1
    if (kb == 0) {
        if (tid == 0) {
            token_sorted[pos] = t;
            weight_sorted[pos] = __float2half_rn(to_f32(w[j]));   // weights.reshape(-1).to(fp16)
            pair_expert[pos] = pe;
            inv[j] = pos;
        }
        if (prefetch && pe >= 0) {                             // the epilogues' vectors of this expert -> L2
            const int lg = N >> 6, ld = K >> 6;               // 128-byte lines: svh_g, svh_u, suh_d (N) + svh_d (K)
            for (int l = tid; l < 3 * lg + ld; l += blockDim.x) {
                const half* p = l < lg ? hptr(pf_g, pe) + l * 64
                              : l < 2 * lg ? hptr(pf_u, pe) + (l - lg) * 64
                              : l < 3 * lg ? hptr(pf_sd, pe) + (l - 2 * lg) * 64
                                           : hptr(pf_vd, pe) + (l - 3 * lg) * 64;
                prefetch_l2(p);
            }
        }
    }
    if (pe < 0) return;                                        // the row is never read (pair_expert -1)
#pragma unroll
    for (int mat = 0; mat < 2; ++mat) {                        // rot_in_kernel's arithmetic, bit for bit
        const H4 sv = mat ? s1 : s0;
        float v[4];
#if TF_PARITY
        const half2 p01 = __hmul2(xv.lo, sv.lo), p23 = __hmul2(xv.hi, sv.hi);
        v[0] = __low2float(p01); v[1] = __high2float(p01); v[2] = __low2float(p23); v[3] = __high2float(p23);
#else
        v[0] = __low2float(xv.lo) * __low2float(sv.lo);
        v[1] = __high2float(xv.lo) * __high2float(sv.lo);
        v[2] = __low2float(xv.hi) * __low2float(sv.hi);
        v[3] = __high2float(xv.hi) * __high2float(sv.hi);
#endif
        fwht128(v, lane);
        half* o = (mat ? out1 : out0) + (size_t)pos * K + c0;
        st_h4(o, __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE),
              __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE));
    }
}

// fp32 -> the output dtype exactly as production's out.to(dtype=x.dtype) on the GPU (c10 BFloat16 / Half from float
// on sm_80+: __float2bfloat16 / __float2half, round to nearest even)
__device__ __forceinline__ void st_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void st_out(half* p, float v) { *p = __float2half(v); }

template <typename OT>
__global__ void glue_finish_kernel(const float* __restrict__ Z, const int* __restrict__ pair_expert,
                                   const int* __restrict__ inv, const half* __restrict__ weight_sorted,
                                   const int64_t* __restrict__ svh_pd, OT* __restrict__ out, int64_t out_stride,
                                   int P, int D, int SK, int topk, const half* __restrict__ dead_xd, int N,
                                   int discard) {
    extern __shared__ float vs[];                              // [topk][128] contributions
    __shared__ int valid_sh[32];
    const int t = blockIdx.x, blk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int j = inv[t * topk + warp];
    const int e = pair_expert[j];
    if (lane == 0) valid_sh[warp] = e >= 0;
    if (e >= 0) {                                              // down_epilogue_kernel's arithmetic for row j, bit for bit
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.f;
            for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + j) * D + n + i];
#if TF_PARITY
            v[i] = __half2float(__float2half_rn(s));
#else
            v[i] = s;
#endif
        }
        fwht128(v, lane);
        if (discard) {                                          // K3b, as down_epilogue: row j's Z lines, its xd share
            __syncwarp();
            for (int l = lane; l < SK * 4; l += 32)
                l2_discard(Z + ((size_t)(l >> 2) * P + j) * D + blk * 128 + (l & 3) * 32);
            const int nb = D >> 7, lines = N >> 6;
            for (int l = blk + nb * lane; l < lines; l += nb * 32) l2_discard(dead_xd + (size_t)j * N + l * 64);
        }
        const float rs = 0.088388347648f * __half2float(weight_sorted[j]);
        const H4 sv = ld_h4(hptr(svh_pd, e) + n);
        v[0] *= rs; v[1] *= rs; v[2] *= rs; v[3] *= rs;
        v[0] *= __low2float(sv.lo); v[1] *= __high2float(sv.lo);
        v[2] *= __low2float(sv.hi); v[3] *= __high2float(sv.hi);
        float* d = vs + warp * 128 + 4 * lane;
        d[0] = v[0]; d[1] = v[1]; d[2] = v[2]; d[3] = v[3];
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 128; c += blockDim.x) {
        float acc = 0.f;                                       // production: torch.zeros(out), then the atomics
        for (int k = 0; k < topk; ++k)
            if (valid_sh[k]) acc += vs[k * 128 + c];
        st_out(out + (int64_t)t * out_stride + blk * 128 + c, acc);
    }
}

// ---------------------------------------------------------------------------------------------------------
// moeglue warm (docs/DEC_MOEGLUE.md, GLM53_DEC_MOEGLUE_WARM): read up to WARM_MAX_REGIONS byte ranges into the L2
// (plain 16 B ld.global.nc loads, the values discarded). Measured on GB10 (tests/probe_l2_warm_kernel.py,
// probe_l2_retain.py): ld.global.nc.L1::no_allocate does NOT leave the lines in the L2 and prefetch.global.L2 barely
// does; plain loads do. Launched on a side stream right after an attention sublayer's o_proj GEMV, it runs while the
// all-reduce and the mHC kernels (DRAM-idle) run on the main stream, so the next MoE sublayer's first weights
// (hc_ffn_fn, router, shared expert) are L2 hits. Regions are read in order (each with a grid stride). Writes
// nothing (the sink store is a liveness guard: it fires only if a thread's XOR of all it read equals a constant).
constexpr int WARM_MAX_REGIONS = 6;
struct WarmRegions {
    const uint4* p[WARM_MAX_REGIONS];
    long long n16[WARM_MAX_REGIONS];                           // region sizes in 16 B units
    int n;
};

// One region after the other (every block walks region k with a grid stride before region k + 1): the region loop
// is unrolled so the parameter struct is only indexed with constants (no local-memory copy of it).
template <int U, bool NA>
__global__ void __launch_bounds__(256) l2_warm_kernel(WarmRegions r, unsigned* __restrict__ sink) {
    unsigned acc = 0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long t0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int k = 0; k < WARM_MAX_REGIONS; ++k) {
        if (k >= r.n) break;
        const uint4* __restrict__ p = r.p[k];
        const long long n16 = r.n16[k];
        for (long long i0 = t0; i0 < n16; i0 += stride * U) {
            unsigned v[U][4];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const long long i = i0 + u * stride;
                v[u][0] = v[u][1] = v[u][2] = v[u][3] = 0u;
                if (i < n16) {
                    if (NA)
                        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(v[u][0]), "=r"(v[u][1]), "=r"(v[u][2]), "=r"(v[u][3]) : "l"(p + i));
                    else
                        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                     : "=r"(v[u][0]), "=r"(v[u][1]), "=r"(v[u][2]), "=r"(v[u][3]) : "l"(p + i));
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) acc ^= v[u][0] ^ v[u][1] ^ v[u][2] ^ v[u][3];
        }
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Host launchers (checked by the callers in exl3.cpp)

int64_t tf_s_cap(int64_t P, int64_t n) { return (P < n ? P : n) + (P + 15) / 16; }

int tf_parity() { return TF_PARITY; }

static inline const int64_t* i64p(const at::Tensor& t) { return t.data_ptr<int64_t>(); }
static inline int* i32p(const at::Tensor& t) { return t.data_ptr<int>(); }
static inline const half* hcp(const at::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }
static inline half* hp(const at::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr()); }

void launch_route_prep(const at::Tensor& count, int64_t n, int64_t P, int64_t R, int64_t S_cap,
                       const at::Tensor& pair_expert, const at::Tensor& seg_expert, const at::Tensor& seg_row0,
                       const at::Tensor& seg_rows, const at::Tensor& nseg) {
    route_prep_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        i64p(count), (int)n, (int)P, (int)R, (int)S_cap, i32p(pair_expert), i32p(seg_expert), i32p(seg_row0),
        i32p(seg_rows), i32p(nseg));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& token_sorted,
                   const at::Tensor& pair_expert, const at::Tensor& suh_p0, const at::Tensor& suh_p1,
                   const at::Tensor& out0, const at::Tensor& out1, int64_t P, int64_t K, int64_t B) {
    dim3 grid((unsigned)P, (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16)
        rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, i64p(token_sorted), i32p(pair_expert),
            i64p(suh_p0), i64p(suh_p1), hp(out0), hp(out1), (int)K, B);
    else
        rot_in_kernel<half><<<grid, 32, 0, stream>>>(hcp(x), x_stride, i64p(token_sorted), i32p(pair_expert),
                                                     i64p(suh_p0), i64p(suh_p1), hp(out0), hp(out1), (int)K, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_route_ids(const at::Tensor& ids, const at::Tensor* emap, const at::Tensor& weights, int64_t P,
                      int64_t topk, int64_t n, int64_t R, int64_t S_cap, const at::Tensor& pair_expert,
                      const at::Tensor& seg_expert, const at::Tensor& seg_row0, const at::Tensor& seg_rows,
                      const at::Tensor& nseg, const at::Tensor& token_sorted, const at::Tensor& weight_sorted) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shm = (size_t)2 * (n + 1) * sizeof(int);
    const int64_t* em = emap ? emap->data_ptr<int64_t>() : nullptr;
    const int64_t ng = emap ? emap->numel() : 0;
#define TF_ROUTE_IDS(WT_)                                                                                   \
    route_ids_kernel<WT_><<<1, 1024, shm, stream>>>(i64p(ids), emap ? 1 : 0, em, ng,                         \
                                                    reinterpret_cast<const WT_*>(weights.data_ptr()), (int)P, \
                                                    (int)topk, (int)n, (int)R, (int)S_cap, i32p(pair_expert),  \
                                                    i32p(seg_expert), i32p(seg_row0), i32p(seg_rows),          \
                                                    i32p(nseg), token_sorted.data_ptr<int64_t>(), hp(weight_sorted))
    if (weights.scalar_type() == at::kFloat) TF_ROUTE_IDS(float);
    else if (weights.scalar_type() == at::kBFloat16) TF_ROUTE_IDS(__nv_bfloat16);
    else TF_ROUTE_IDS(half);
#undef TF_ROUTE_IDS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// moeglue launchers (docs/DEC_MOEGLUE.md); every argument is checked by moe_forward_glue (exl3.cpp) before the
// first launch. ids: int32 or int64; x: bf16 or fp16; w: fp32, bf16 or fp16.
int64_t tf_glue_prep_blocks(int64_t P, int64_t K) { return P * (K / (128 * GLUE_WARPS)) + 1; }

void launch_glue_prep(const at::Tensor& x, const at::Tensor& ids, const at::Tensor* emap, const at::Tensor& w,
                      int64_t P, int64_t topk, int64_t n, int64_t R, int64_t S_cap, int64_t K,
                      const at::Tensor& suh_p0, const at::Tensor& suh_p1, const at::Tensor& pf_g,
                      const at::Tensor& pf_u, const at::Tensor& pf_sd, const at::Tensor& pf_vd, int64_t N,
                      bool prefetch, const at::Tensor& out0, const at::Tensor& out1, const at::Tensor& pair_expert,
                      const at::Tensor& seg_expert, const at::Tensor& seg_row0, const at::Tensor& seg_rows,
                      const at::Tensor& nseg, const at::Tensor& token_sorted, const at::Tensor& weight_sorted,
                      const at::Tensor& inv) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned grid = (unsigned)tf_glue_prep_blocks(P, K);
    const size_t shm = (size_t)(n + 1) * sizeof(int);
    const int64_t* em = emap ? emap->data_ptr<int64_t>() : nullptr;
    const int64_t ng = emap ? emap->numel() : 0;
    const bool xb = x.scalar_type() == at::kBFloat16, i32 = ids.scalar_type() == at::kInt;
#define TF_GLUE_PREP(XT_, IT_, WT_)                                                                              \
    glue_prep_kernel<XT_, IT_, WT_><<<grid, GLUE_WARPS * 32, shm, stream>>>(                                    \
        reinterpret_cast<const XT_*>(x.data_ptr()), x.stride(0), reinterpret_cast<const IT_*>(ids.data_ptr()),   \
        emap ? 1 : 0, em, ng, reinterpret_cast<const WT_*>(w.data_ptr()), (int)P, (int)topk, (int)n, (int)R,     \
        (int)S_cap, (int)K, i64p(suh_p0), i64p(suh_p1), i64p(pf_g), i64p(pf_u), i64p(pf_sd), i64p(pf_vd), (int)N, \
        prefetch ? 1 : 0, hp(out0), hp(out1), i32p(pair_expert), i32p(seg_expert), i32p(seg_row0),               \
        i32p(seg_rows), i32p(nseg), token_sorted.data_ptr<int64_t>(), hp(weight_sorted), i32p(inv))
#define TF_GLUE_PREP_W(XT_, IT_)                                                                                 \
    do {                                                                                                         \
        if (w.scalar_type() == at::kFloat) TF_GLUE_PREP(XT_, IT_, float);                                        \
        else if (w.scalar_type() == at::kBFloat16) TF_GLUE_PREP(XT_, IT_, __nv_bfloat16);                        \
        else TF_GLUE_PREP(XT_, IT_, half);                                                                       \
    } while (0)
    if (xb) {
        if (i32) TF_GLUE_PREP_W(__nv_bfloat16, int32_t);
        else TF_GLUE_PREP_W(__nv_bfloat16, int64_t);
    } else {
        if (i32) TF_GLUE_PREP_W(half, int32_t);
        else TF_GLUE_PREP_W(half, int64_t);
    }
#undef TF_GLUE_PREP_W
#undef TF_GLUE_PREP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static bool discard_on();

void launch_glue_finish(const at::Tensor& Z, const at::Tensor& pair_expert, const at::Tensor& inv,
                        const at::Tensor& weight_sorted, const at::Tensor& svh_pd, const at::Tensor& out, int64_t P,
                        int64_t D, int64_t SK, int64_t topk, int64_t B, const at::Tensor* dead_xd, int64_t N) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)B, (unsigned)(D / 128));
    const size_t shm = (size_t)topk * 128 * sizeof(float);
    const bool dis = dead_xd != nullptr && discard_on();
    const half* dx = dis ? hcp(*dead_xd) : nullptr;
    if (out.scalar_type() == at::kBFloat16)
        glue_finish_kernel<__nv_bfloat16><<<grid, (unsigned)(32 * topk), shm, stream>>>(
            Z.data_ptr<float>(), i32p(pair_expert), i32p(inv), hcp(weight_sorted), i64p(svh_pd),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), (int)P, (int)D, (int)SK, (int)topk, dx,
            (int)N, dis ? 1 : 0);
    else
        glue_finish_kernel<half><<<grid, (unsigned)(32 * topk), shm, stream>>>(
            Z.data_ptr<float>(), i32p(pair_expert), i32p(inv), hcp(weight_sorted), i64p(svh_pd), hp(out),
            out.stride(0), (int)P, (int)D, (int)SK, (int)topk, dx, (int)N, dis ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Grouped-GEMV kernel variants (docs/OPTIMIZATION.md), selected per launch from a process-wide id set by the
// test/bench hook tf_set_variant(): {ORD, PIPE, LEAN, POL}. Id 0 is the shipped configuration; id 1 is the kernel
// as it was on master (ORD 0, no pipelining, 4-buffer reduction) for A/B; 2.. are experiments. Every variant
// produces bit-identical Z (same operations in the same order). Only the production tile (nt = 4, 4 warps) has
// variants; other tiles always run ORD 0.
// sktab: the split-count table moe_forward uses (kernels/exl3.cpp, tf_split_counts); variants with the same sktab
// compute bit-identical results, a different table changes only the fp32 summation tree of the split partials.
struct GroupedVariant { int ord, pipe, lean, pol, sktab, discard = 0; };   // discard: K3b, forward path only
static const GroupedVariant kVariants[] = {
    {1, 0, 0, 1, 6, 1},   // 0 shipped: ORD 1 (K1) + evict_first weights (K3) + split-count table 6 (K6, C2)
                          //   + dead intermediates dropped from L2 (K3b), docs/OPTIMIZATION.md
    {0, 0, 0, 0, 0},   // 1 master
    {1, 0, 0, 0, 0},   // 2 n block fastest
    {2, 0, 0, 0, 0},   // 3 expert-major, empty segments last
    {1, 1, 0, 0, 0},   // 4 ORD1 + double-buffered k loop
    {1, 0, 1, 0, 0},   // 5 ORD1 + LEAN reduction
    {1, 0, 0, 1, 0},   // 6 ORD1 + evict_first weights
    {1, 1, 1, 0, 0},   // 7 ORD1 + PIPE + LEAN
    {0, 1, 0, 0, 0},   // 8 ORD0 + PIPE
    {1, 0, 0, 1, 1},   // 9 shipped kernel, split-count table 1
    {1, 0, 0, 1, 2},   // 10 shipped kernel, split-count table 2
    {1, 0, 0, 1, 3},   // 11 shipped kernel, split-count table 3
    {1, 0, 0, 1, 6, 0},   // 12 shipped without K3b (intermediates not discarded)
    {1, 0, 0, 1, 4, 1},   // 13 shipped, split-count table 4 (gate/up 2 at P > 192)
    {1, 0, 0, 1, 5, 1},   // 14 shipped, split-count table 5 (gate/up 2 at P > 32)
    {1, 0, 0, 1, 1, 1},   // 15 shipped with split-count table 1 (before C2)
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);
static int g_variant = 0;

int tf_num_variants() { return kNumVariants; }
int tf_variant() { return g_variant; }
int tf_sk_table() { return kVariants[g_variant].sktab; }
static bool discard_on() { return kVariants[g_variant].discard != 0; }
int tf_variant_sk_table(int v) { return kVariants[v].sktab; }
static bool variant_always_launchable(const GroupedVariant& v);
void tf_set_variant(int v) {
    TORCH_CHECK(v >= 0 && v < kNumVariants, "tf_exl3_moe: variant must be in [0, ", kNumVariants, ")");
    TORCH_CHECK(variant_always_launchable(kVariants[v]), "tf_exl3_moe: variant ", v,
                " resolves to a grouped-kernel instance that is not compiled");
    g_variant = v;
}

// How the current variant runs one grouped launch of the production tile (nt = 4, 4 warps): the variant's
// {ORD, PIPE, LEAN, POL} after the two shape fallbacks, as the instantiation code ORD*8 + PIPE*4 + LEAN*2 + POL.
//   - PIPE needs an even per-warp tile count ((kdim/16)/SK/W): otherwise PIPE 0.
//   - ORD 1 / 2 put S_cap in gridDim.y / gridDim.z (<= 65535): above that the launch runs ORD 0 (S_cap in
//     gridDim.x). ORD 0 has no LEAN instance, so the fallback also drops LEAN; POL is kept (ORD 0 + POL, code 1,
//     is instantiated for the shipped variant's fallback). Every knob leaves Z bit-identical, so a fallback never
//     changes a result.
// Returns -1 when the resolved combination is not instantiated. moe_forward / moe_forward_ids check both of their
// grouped launches with this before their first kernel (every check before the first launch); tf_set_variant
// refuses a variant that could resolve to a missing instance.
static bool grouped_code_instantiated(int code) {
    switch (code) {
        case 0: case 1: case 4: case 8: case 9: case 10: case 12: case 14: case 16: return true;
        default: return false;
    }
}
static int grouped_code_for(const GroupedVariant& v0, int64_t kdim, int64_t SK, int64_t S_cap) {
    GroupedVariant v = v0;
    if (v.pipe && ((kdim / 16) / SK / 4) % 2 != 0) v.pipe = 0;   // double buffering needs an even tile count
    if (v.ord != 0 && S_cap > 65535) {                            // S_cap must fit gridDim.y / gridDim.z
        v.ord = 0;
        v.lean = 0;
    }
    const int code = v.ord * 8 + v.pipe * 4 + v.lean * 2 + v.pol;
    return grouped_code_instantiated(code) ? code : -1;
}
static bool variant_always_launchable(const GroupedVariant& v) {
    for (int odd = 0; odd < 2; ++odd)                              // kdim/16/SK/4 even (4096/16/4/4) or odd (1*16*SK*4)
        for (int big = 0; big < 2; ++big)
            if (grouped_code_for(v, odd ? 16 * 4 : 4096, odd ? 1 : 4, big ? 65536 : 1) < 0) return false;
    return true;
}
bool tf_grouped_launchable(int64_t kdim, int64_t SK, int64_t S_cap) {
    return grouped_code_for(kVariants[g_variant], kdim, SK, S_cap) >= 0;
}

void launch_grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& Tp0, const at::Tensor& Tp1,
                    const at::Tensor& seg_expert, const at::Tensor& seg_row0, const at::Tensor& seg_rows,
                    const at::Tensor& nseg, const at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                    int64_t SK, int64_t nt, int64_t warps, int64_t S_cap) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const unsigned gs = (unsigned)S_cap, gn = (unsigned)(N / (16 * nt)), gz = (unsigned)(mats * SK);
#define TF_LAUNCH(NT_, W_, ORD_, PIPE_, LEAN_, POL_)                                                            \
    do {                                                                                                     \
        const dim3 grid = ORD_ == 0 ? dim3(gs, gn, gz) : ORD_ == 1 ? dim3(gn, gs, gz) : dim3(gn, gz, gs);    \
        grouped_kernel<NT_, W_, ORD_, PIPE_, LEAN_, POL_><<<grid, W_ * 32, 0, stream>>>(                      \
            hcp(X0), hcp(X1), i64p(Tp0), i64p(Tp1), i32p(seg_expert), i32p(seg_row0), i32p(seg_rows),         \
            i32p(nseg), Z.data_ptr<float>(), (int)K, (int)N, (int)P, (int)SK);                                \
    } while (0)
    if (nt == 4 && warps == 4) {
        switch (grouped_code_for(kVariants[g_variant], K, SK, S_cap)) {
            case 0: TF_LAUNCH(4, 4, 0, 0, 0, 0); break;
            case 1: TF_LAUNCH(4, 4, 0, 0, 0, 1); break;
            case 4: TF_LAUNCH(4, 4, 0, 1, 0, 0); break;
            case 8: TF_LAUNCH(4, 4, 1, 0, 0, 0); break;
            case 9: TF_LAUNCH(4, 4, 1, 0, 0, 1); break;
            case 10: TF_LAUNCH(4, 4, 1, 0, 1, 0); break;
            case 12: TF_LAUNCH(4, 4, 1, 1, 0, 0); break;
            case 14: TF_LAUNCH(4, 4, 1, 1, 1, 0); break;
            case 16: TF_LAUNCH(4, 4, 2, 0, 0, 0); break;
            default: TORCH_CHECK(false, "tf_exl3_moe: variant ", g_variant, " not instantiated");
        }
    } else if (nt == 8 && warps == 4) TF_LAUNCH(8, 4, 0, 0, 0, 0);
    else if (nt == 4 && warps == 8) TF_LAUNCH(4, 8, 0, 0, 0, 0);
    else if (nt == 2 && warps == 4) TF_LAUNCH(2, 4, 0, 0, 0, 0);
    else if (nt == 2 && warps == 8) TF_LAUNCH(2, 8, 0, 0, 0, 0);
    else TORCH_CHECK(false, "unsupported tile setting");
#undef TF_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static bool discard_on();

// dead_xg / dead_xu (optional, forward path): the gate/up GEMV inputs, dropped from L2 when the variant says so
void launch_gateup_epilogue(const at::Tensor& Z, const at::Tensor& pair_expert, const at::Tensor& svh_pg,
                            const at::Tensor& svh_pu, const at::Tensor& suh_pd, const at::Tensor& xd, int64_t P,
                            int64_t N, int64_t SK, double limit, const at::Tensor* dead_xg, const at::Tensor* dead_xu,
                            int64_t K) {
    dim3 grid((unsigned)P, (unsigned)(N / 128));
    const bool dis = dead_xg != nullptr && dead_xu != nullptr && discard_on();
    gateup_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), i32p(pair_expert), i64p(svh_pg), i64p(svh_pu), i64p(suh_pd), hp(xd), (int)P, (int)N,
        (int)SK, (float)limit, dis ? hcp(*dead_xg) : nullptr, dis ? hcp(*dead_xu) : nullptr, (int)K, dis ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_down_epilogue(const at::Tensor& Z, const at::Tensor& pair_expert, const at::Tensor& token_sorted,
                          const at::Tensor& weight_sorted, const at::Tensor& svh_pd, const at::Tensor& out, int64_t P,
                          int64_t D, int64_t SK, int64_t B, const at::Tensor* dead_xd, int64_t N) {
    dim3 grid((unsigned)P, (unsigned)(D / 128));
    const bool dis = dead_xd != nullptr && discard_on();
    down_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), i32p(pair_expert), i64p(token_sorted), hcp(weight_sorted), i64p(svh_pd),
        out.data_ptr<float>(), (int)P, (int)D, (int)SK, B, dis ? hcp(*dead_xd) : nullptr, (int)N, dis ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t tf_warm_max_regions() { return WARM_MAX_REGIONS; }

// ptrs / bytes: 1..WARM_MAX_REGIONS ranges, 16 B aligned, bytes a multiple of 16 (checked by the caller)
void launch_l2_warm(const std::vector<const void*>& ptrs, const std::vector<int64_t>& bytes, int64_t blocks,
                    int64_t unroll, unsigned* sink) {
    WarmRegions r{};
    long long tot = 0;
    r.n = (int)ptrs.size();
    for (int k = 0; k < r.n; ++k) {
        r.p[k] = reinterpret_cast<const uint4*>(ptrs[k]);
        r.n16[k] = bytes[k] / 16;
        tot += r.n16[k];
    }
    if (tot <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool na = unroll < 0;                                // test hook: negative unroll = L1::no_allocate loads
    const int64_t u = unroll < 0 ? -unroll : unroll;
    if (u >= 8) {
        if (na) l2_warm_kernel<8, true><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
        else l2_warm_kernel<8, false><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
    } else if (u >= 4) {
        if (na) l2_warm_kernel<4, true><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
        else l2_warm_kernel<4, false><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
    } else {
        if (na) l2_warm_kernel<1, true><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
        else l2_warm_kernel<1, false><<<(unsigned)blocks, 256, 0, stream>>>(r, sink);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
