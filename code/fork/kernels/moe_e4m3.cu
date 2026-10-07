// MOE_E4M3: routed-MoE prefill of GLM-5.3-Flash's EXL3 K4/mcg experts on e4m3 tensor cores (GLM53_MOE_E4M3,
// docs/MOE_E4M3.md). Numerics contract = the e4m3 quality emulation of branch moefq (glm53_moe_e4m3_emu, the
// "spec"), per (token, expert) row:
//
//   gather:  v = H(float(x16) * float(suh_g)) * r            (fp32; the fp16 x fp16 product is exact in fp32)
//            s = amax(|v|) / 448 (1 if 0) ;  q = e4m3_rn_sat(v / s)          per-row ("per-token") scale
//   gate/up: acc = sum_k q[k] * e4m3_rn(Wq[k, n])            mma.sync m16n8k32 e4m3 x e4m3 -> fp32 accumulate
//            g = (H(acc_g * s) * r) * svh_g ; u = (H(acc_u * s) * r) * svh_u
//            a16 = fp16_rn( (min(g, L) / (1 + exp(-min(g, L)))) * clamp(u, -L, L) )     torch silu * clamp
//   actq:    v = H(float(a16) * float(suh_d)) * r ; s_d = amax / 448 ; q_d = e4m3_rn_sat(v / s_d)
//   down:    out[token] += ((H(acc_d * s_d) * r) * svh_d) * w        fp32 atomics, w = the fp32 router weight
//
// Wq = the trellis decode (kernels/exl3_format_ref.py), rounded to e4m3 with NO scale (the mcg codebook is O(1),
// max |v| 3.949: no saturation). H = 128-point Walsh-Hadamard (fwht128), r = 1/sqrt(128). The spec applies the
// scale before the GEMM ((q * s) @ W) and its Hadamard as an fp32 matmul; this kernel applies s after the exact
// e4m3 GEMM and uses the butterfly: the same real numbers, different fp32 rounding (~1e-7 relative).
//
// k permutation: the mainloop builds the e4m3 B fragment of k32 from two decoded k16 trellis tiles without moving
// data between lanes, so B's mma position p holds the weight of k = perm(p) with perm(p) = 16*(p/16) + 2*q + (i&1)
// + 8*(i>>1), q = (p%16)/4, i = p%4. The A rows are stored with the same permutation inside every 32-byte group (the
// gather / actq kernels write them that way), so the dot products are unchanged.
//
// Building blocks: trellis decode from TensorFold (MIT, kernels/exl3.cu), fwht128 from kernels/exl3.cu, the
// mainloop / persistent item structure from E4 (branch e4fat, kernels/e4_fat.cu). Built WITHOUT --use_fast_math /
// -ftz (the division, the exponential and subnormal e4m3 inputs follow IEEE like torch's own kernels).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;      // 1/sqrt(128) as fp32 (== ExLlamaV3's r_scale)
constexpr float E4M3_MAX = 448.0f;
constexpr int F_HID_G = 4096;              // hidden size (gather16)

// ---------------------------------------------------------------------------------------------------------
// TensorFold trellis decode (kernels/exl3.cu)

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

// This lane's eight values of a 4-bit 16x16 tile (w = tile word `lane`) as the fp16 B fragments (m16n8k16 layout)
// of its two n8 halves: b0[0] = (k 2q, 2q+1), b0[1] = (k 2q+8, 2q+9) of column n = lane/4 (q = lane%4); b1 the
// same for column 8 + lane/4.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

// Lean decode (MS & 8): the same integer function as mcg2 / decode_tile with fewer instructions. The AND mask
// 0x8FFF8FFF and the XOR 0x3B603B60 have equal 16-bit halves, so they commute with the half-word permutation and are
// applied after it as one LOP3 each (the XOR constant held in a register: LOP3 takes one immediate); byte-aligned
// 16-bit fields ((v >> 8) & 0xffff) are one PRMT. Bit-identical outputs.
__device__ __forceinline__ uint32_t lop_andxor(uint32_t a, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, 0x8FFF8FFF, %2, 0x6A;" : "=r"(d) : "r"(a), "r"(c));     // (a & b) ^ c
    return d;
}
__device__ __forceinline__ uint32_t mcg2_lean(uint32_t s0, uint32_t s1, uint32_t cx) {
    const uint32_t x0 = s0 * 0xCBAC1FEDu;
    const uint32_t x1 = s1 * 0xCBAC1FEDu;
    uint32_t lo = lop_andxor(__byte_perm(x0, x1, 0x5410), cx);
    uint32_t hi = lop_andxor(__byte_perm(x0, x1, 0x7632), cx);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void decode_tile_lean(uint32_t w, int lane, uint32_t cx, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2_lean(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu, cx);
    b0[1] = mcg2_lean(s & 0xffffu, w >> 16, cx);
    b1[0] = mcg2_lean((w >> 12) & 0xffffu, __byte_perm(w, 0u, 0x4421), cx);
    b1[1] = mcg2_lean((w >> 4) & 0xffffu, w & 0xffffu, cx);
}

// fp16x2 -> two e4m3 (round to nearest even, saturating), low half -> low byte
__device__ __forceinline__ uint32_t cvt8_h2(uint32_t h2) {
    __half2_raw r;
    r.x = (unsigned short) (h2 & 0xffffu);
    r.y = (unsigned short) (h2 >> 16);
    return (uint32_t) __nv_cvt_halfraw2_to_fp8x2(r, __NV_SATFINITE, __NV_E4M3);
}
// two fp32 -> two e4m3 (rn, saturating), a -> low byte
__device__ __forceinline__ uint32_t cvt8_f2(float a, float b) {
    return (uint32_t) __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// k32 e4m3 B fragment of one n8 half from the k16 tiles t0 (k 0..15) and t1 (k 16..31): byte i of reg 0 = mma
// k 4q + i = actual k perm(4q + i) (see the header).
__device__ __forceinline__ void frag8(const uint32_t (&t0)[2], const uint32_t (&t1)[2], uint32_t (&bf)[2]) {
    bf[0] = cvt8_h2(t0[0]) | (cvt8_h2(t0[1]) << 16);
    bf[1] = cvt8_h2(t1[0]) | (cvt8_h2(t1[1]) << 16);
}

__device__ __forceinline__ void mma16832(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// fp16 x fp16 -> fp32 (GLM53_MOE_E4M3_DOWN=f16: the down projection on production's operand widths)
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 128-point Walsh-Hadamard (kernels/exl3.cu fwht128): lane L holds values 4L..4L+3 (unscaled). The cross-lane
// butterfly (lane & m) ? o - v : v + o is evaluated as fma(+-1, v, o): the same single rounding of the same exact
// sum, one instruction instead of two adds and a select.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float sg = (lane & m) ? -1.0f : 1.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = __fmaf_rn(sg, v[j], o);
        }
    }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// Byte offset inside a row of the e4m3 element at column k (k multiple of 2, pairs stay adjacent): the inverse of
// perm() inside each 32-column group.
__device__ __forceinline__ int perm_pos(int k) {
    const int g = k & ~31, h = k & 16, a = k & 15;
    const int p = a < 8 ? 2 * a : 2 * (a - 8) + 2;
    return g + h + p;
}

// The per-row quantization of the spec: s = amax / 448 (1 if amax == 0); q = e4m3(v / s).
__device__ __forceinline__ float row_scale(float amax) {
    const float s = __fdiv_rn(amax, E4M3_MAX);
    return s > 0.0f ? s : 1.0f;
}

// Store this lane's 4 values (columns 4*lane..4*lane+3 of the 128-column block at `blk`) as e4m3 at their permuted
// positions: lane pairs (m, m^2), m = lane % 8, swap one 16-bit pair so every lane writes one aligned 32-bit word
// (the warp writes the block's 128 bytes in one coalesced store). Byte layout: perm_pos() / the header.
__device__ __forceinline__ void store_q4(uint8_t* blk, int lane, const float (&v)[4], float s) {
    const uint32_t q01 = cvt8_f2(__fdiv_rn(v[0], s), __fdiv_rn(v[1], s));
    const uint32_t q23 = cvt8_f2(__fdiv_rn(v[2], s), __fdiv_rn(v[3], s));
    const int m = lane & 7;
    const bool sel = (m & 2) != 0;
    const uint32_t recv = __shfl_xor_sync(0xffffffffu, sel ? q01 : q23, 2);
    const uint32_t word = sel ? (recv | (q23 << 16)) : (q01 | (recv << 16));
    const int off = (lane >> 3) * 32 + ((m >> 2) & 1) * 16 + (m & 1) * 8 + (m & 2) * 2;
    *reinterpret_cast<uint32_t*>(blk + off) = word;
}

// Four bf16 values as the fp16 the spec sees (x.half(): round to nearest even, inf above 65504, fp16 subnormals),
// then as float: the same bits as ld_h4f of a torch-converted fp16 copy, without that copy's extra pass over x.
__device__ __forceinline__ float4 ld_b4f(const __nv_bfloat16* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&r.x);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&r.y);
    return make_float4(__half2float(__float2half_rn(__low2float(lo))), __half2float(__float2half_rn(__high2float(lo))),
                       __half2float(__float2half_rn(__low2float(hi))), __half2float(__float2half_rn(__high2float(hi))));
}

__device__ __forceinline__ float4 ld_h4f(const half* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const half2 lo = *reinterpret_cast<const half2*>(&r.x), hi = *reinterpret_cast<const half2*>(&r.y);
    return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

// ---------------------------------------------------------------------------------------------------------
// Gather (token-major): a block of topk (<= 8) warps per token, warp k quantizes the pair (t, k):
//   a8[pos[t*topk + k]] = e4m3(H(x[t] * suh[local[t*topk + k]]) * r / s),  asc[row] = s.
// Pairs with local id >= n_exp (non-local / invalid) are skipped. Two passes per row (amax, then quantize); the
// block's warps share the x row in L1.

template <int K, int DBG = 0>
__global__ void __launch_bounds__(256) me_gather_kernel(const half* __restrict__ x, const int* __restrict__ local,
                                                       const int* __restrict__ pos, const int64_t* __restrict__ suh_ptrs,
                                                       uint8_t* __restrict__ a8, float* __restrict__ asc, int T,
                                                       int topk, int n_exp) {
    constexpr int NB = K / 128;
    const int lane = threadIdx.x & 31, k = threadIdx.x >> 5;
    if (k >= topk) return;
    for (int t = blockIdx.x; t < T; t += gridDim.x) {
        const int e = local[t * topk + k];
        if (e < 0 || e >= n_exp) continue;                           // warp-uniform
        const int row = pos[t * topk + k];
        const half* xrow = x + (size_t) t * K + 4 * lane;
        const half* suh = reinterpret_cast<const half*>(suh_ptrs[e]) + 4 * lane;
        float amax = 0.0f;
#pragma unroll 8
        for (int b = 0; b < (DBG & 1 ? 0 : NB); ++b) {
            const float4 xv = ld_h4f(xrow + b * 128), s = ld_h4f(suh + b * 128);
            float v[4] = {xv.x * s.x, xv.y * s.y, xv.z * s.z, xv.w * s.w};
            fwht128(v, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(__fmul_rn(v[i], HAD_SCALE)));
        }
        const float sc = row_scale(warp_max(amax));
        uint8_t* dst = a8 + (size_t) row * K;
#pragma unroll 8
        for (int b = 0; b < NB; ++b) {
            const float4 xv = ld_h4f(xrow + b * 128), s = ld_h4f(suh + b * 128);
            float v[4] = {xv.x * s.x, xv.y * s.y, xv.z * s.z, xv.w * s.w};
            fwht128(v, lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], HAD_SCALE);
            if (DBG & 2) { if (v[0] == 1234.5f) dst[0] = 1; }
            else store_q4(dst + b * 128, lane, v, sc);
        }
        if (lane == 0) asc[row] = sc;
    }
}

// Gather v2 (single pass): a row is split over WPR warps (each holds K / WPR columns of the transformed row in
// registers), the row's amax is combined through shared memory, then the registers are quantized and stored. A block
// of 8 warps = 8 / WPR rows (pairs) of ONE token (they share the x row in L1); topk / (8 / WPR) blocks per token. The
// block also zeroes its slice of out[token] (the fp32 accumulator the down kernel scatters into).
// TOKM (opt-moe, fused TG schedule): one row per TOKEN (row = t, the suh of expert 0 = every expert's, checked at load
// time), 8 / WPR tokens per block; the block zeroes the out rows of its tokens.
template <int K, int WPR, typename XT = half, int TOKM = 0>
__global__ void __launch_bounds__(256) me_gather2_kernel(const XT* __restrict__ x, const int* __restrict__ local,
                                                        const int* __restrict__ pos,
                                                        const int64_t* __restrict__ suh_ptrs, uint8_t* __restrict__ a8,
                                                        float* __restrict__ asc, float* __restrict__ out, int T,
                                                        int topk, int n_exp, int out_cols) {
    constexpr int NBW = K / 128 / WPR;              // 128-column blocks per warp
    constexpr int RPB = 8 / WPR;                    // rows (pairs) per block
    __shared__ float s_max[8];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int q = w % WPR, kr = w / WPR;
    const int bpt = TOKM ? 1 : topk / RPB;
    const int nblocks = TOKM ? (T + RPB - 1) / RPB : T * bpt;
    for (int b = blockIdx.x; b < nblocks; b += gridDim.x) {
        const int t = TOKM ? b * RPB + kr : b / bpt, jb = TOKM ? 0 : b - t * bpt;
        const int k = jb * RPB + kr;
        const int e = TOKM ? 0 : local[t * topk + k];
        const bool valid = TOKM ? t < T : (e >= 0 && e < n_exp);     // warp-uniform
        float v[NBW][4];
        float amax = 0.0f;
        if (valid) {
            const XT* xrow = x + (size_t) t * K + q * (NBW * 128) + 4 * lane;
            const half* suh = reinterpret_cast<const half*>(suh_ptrs[e]) + q * (NBW * 128) + 4 * lane;
#pragma unroll
            for (int i = 0; i < NBW; ++i) {
                float4 xv;
                if constexpr (std::is_same<XT, half>::value) xv = ld_h4f(xrow + i * 128);
                else xv = ld_b4f(xrow + i * 128);
                const float4 sv = ld_h4f(suh + i * 128);
                v[i][0] = xv.x * sv.x; v[i][1] = xv.y * sv.y; v[i][2] = xv.z * sv.z; v[i][3] = xv.w * sv.w;
                fwht128(v[i], lane);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    v[i][c] = __fmul_rn(v[i][c], HAD_SCALE);
                    amax = fmaxf(amax, fabsf(v[i][c]));
                }
            }
            amax = warp_max(amax);
        }
        if (lane == 0) s_max[w] = amax;
        if (TOKM && out != nullptr) {                // zero the out rows of this block's tokens
            const int per4 = out_cols / 4;
            for (int i = threadIdx.x; i < RPB * per4; i += blockDim.x) {
                const int tt = b * RPB + i / per4;
                if (tt < T) reinterpret_cast<float4*>(out + (size_t) tt * out_cols)[i % per4] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        } else if (out != nullptr) {                 // zero this block's slice of out[t]
            const int per = out_cols / bpt;
            float4* o = reinterpret_cast<float4*>(out + (size_t) t * out_cols + jb * per);
            for (int i = threadIdx.x; i < per / 4; i += blockDim.x) o[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();
        if (valid) {
            float m = 0.0f;
#pragma unroll
            for (int i = 0; i < WPR; ++i) m = fmaxf(m, s_max[kr * WPR + i]);
            const float sc = row_scale(m);
            const int row = TOKM ? t : pos[t * topk + k];
            uint8_t* dst = a8 + (size_t) row * K + q * (NBW * 128);
#pragma unroll
            for (int i = 0; i < NBW; ++i) store_q4(dst + i * 128, lane, v[i], sc);
            if (lane == 0 && q == 0) asc[row] = sc;
        }
        __syncthreads();                             // s_max reuse
    }
}

// ---------------------------------------------------------------------------------------------------------
// exp_prod(a) == (float) exp((double) a) BIT FOR BIT, mostly in fp32 (P16's SiLU: exllamav3's fm_gateup epilogue
// computes the sigmoid with the double-precision exp, which costs ~9.5 ms per 13,824-token layer call on GB10's FP64
// units). a = (64 n + j) ln2/64 + r, |r| <= ln2/128: 2^n * T[j] * e^r in float-float (T = 2^(j/64) as hi + lo, e^r =
// 1 + r + r^2/2 + r^3 (1/6 + r/24 + r^2/120), r from a 3-part Cody-Waite reduction), relative error < 2^-46. The
// float-float value (y, yl) has y = RN(y + yl); y is the correctly rounded result unless the true value can lie on the
// other side of a rounding midpoint (|yl| within 2^-42 y of half an ulp), and RN(exp_double(a)) equals the correctly
// rounded result except within ~2^-52 of a midpoint - so every case that is not provably safe (and a outside
// [-87, 88], NaN) takes production's own double statement. Exhaustively checked against it for every float in
// [-87, 88] (tests/moe3/test_fused16.py, ext.exp_check).
__constant__ float2 EXP_T64[64] = {
    {0x1.0000000000000p+0f, 0x0.0p+0f}, {0x1.02c9a40000000p+0f, -0x1.887fa00000000p-28f}, {0x1.059b0e0000000p+0f, -0x1.9d4f520000000p-25f}, {0x1.0874520000000p+0f, -0x1.e2990e0000000p-26f},
    {0x1.0b55860000000p+0f, 0x1.9f31220000000p-25f}, {0x1.0e3ec40000000p+0f, -0x1.a585cc0000000p-25f}, {0x1.11301e0000000p+0f, -0x1.fdb4960000000p-25f}, {0x1.1429aa0000000p+0f, 0x1.d525bc0000000p-25f},
    {0x1.172b840000000p+0f, -0x1.c157420000000p-27f}, {0x1.1a35be0000000p+0f, 0x1.6df96e0000000p-25f}, {0x1.1d48740000000p+0f, -0x1.d2e8ca0000000p-25f}, {0x1.2063b80000000p+0f, 0x1.0c519a0000000p-25f},
    {0x1.2387a60000000p+0f, 0x1.ceac480000000p-25f}, {0x1.26b4560000000p+0f, 0x1.789f380000000p-26f}, {0x1.29e9e00000000p+0f, -0x1.5c04240000000p-25f}, {0x1.2d285a0000000p+0f, 0x1.b900c20000000p-26f},
    {0x1.306fe00000000p+0f, 0x1.4636e20000000p-25f}, {0x1.33c08c0000000p+0f, -0x1.b37d200000000p-25f}, {0x1.371a740000000p+0f, -0x1.18aac60000000p-25f}, {0x1.3a7db40000000p+0f, -0x1.634c020000000p-25f},
    {0x1.3dea640000000p+0f, 0x1.8246840000000p-25f}, {0x1.4160a20000000p+0f, 0x1.f72e2a0000000p-28f}, {0x1.44e0860000000p+0f, 0x1.8624b40000000p-30f}, {0x1.486a2c0000000p+0f, -0x1.47d8660000000p-25f},
    {0x1.4bfdae0000000p+0f, -0x1.593abc0000000p-25f}, {0x1.4f9b280000000p+0f, -0x1.2c5a6c0000000p-25f}, {0x1.5342b60000000p+0f, -0x1.2c56100000000p-25f}, {0x1.56f4740000000p+0f, -0x1.295b040000000p-25f},
    {0x1.5ab07e0000000p+0f, -0x1.5bd5ec0000000p-27f}, {0x1.5e76f20000000p+0f, -0x1.4a5bd60000000p-25f}, {0x1.6247ec0000000p+0f, -0x1.f8b5500000000p-25f}, {0x1.6623880000000p+0f, 0x1.2a91120000000p-27f},
    {0x1.6a09e60000000p+0f, 0x1.9fcef40000000p-26f}, {0x1.6dfb240000000p+0f, -0x1.cd72e80000000p-27f}, {0x1.71f75e0000000p+0f, 0x1.1d8bee0000000p-25f}, {0x1.75feb60000000p+0f, -0x1.37b3060000000p-25f},
    {0x1.7a11480000000p+0f, -0x1.829fd00000000p-25f}, {0x1.7e2f340000000p+0f, -0x1.2616340000000p-25f}, {0x1.82589a0000000p+0f, -0x1.accc7c0000000p-26f}, {0x1.868d9a0000000p+0f, -0x1.2edb440000000p-26f},
    {0x1.8ace540000000p+0f, 0x1.15506e0000000p-27f}, {0x1.8f1aea0000000p+0f, -0x1.baa2320000000p-26f}, {0x1.93737c0000000p+0f, -0x1.e647440000000p-25f}, {0x1.97d82a0000000p+0f, -0x1.0d8d840000000p-31f},
    {0x1.9c49180000000p+0f, 0x1.51f8480000000p-27f}, {0x1.a0c6680000000p+0f, -0x1.2886a60000000p-26f}, {0x1.a5503c0000000p+0f, -0x1.b83b540000000p-25f}, {0x1.a9e6b60000000p+0f, -0x1.50c0480000000p-25f},
    {0x1.ae89fa0000000p+0f, -0x1.a94b140000000p-26f}, {0x1.b33a2c0000000p+0f, -0x1.ec3a820000000p-26f}, {0x1.b7f7700000000p+0f, -0x1.a094380000000p-25f}, {0x1.bcc1ea0000000p+0f, -0x1.f687c60000000p-25f},
    {0x1.c199be0000000p+0f, -0x1.3d56b20000000p-27f}, {0x1.c67f120000000p+0f, 0x1.cafa2a0000000p-25f}, {0x1.cb720e0000000p+0f, -0x1.8837cc0000000p-27f}, {0x1.d072d40000000p+0f, 0x1.40f1300000000p-25f},
    {0x1.d5818e0000000p+0f, -0x1.822dbc0000000p-27f}, {0x1.da9e600000000p+0f, 0x1.ed99420000000p-27f}, {0x1.dfc9740000000p+0f, -0x1.908c940000000p-25f}, {0x1.e502ee0000000p+0f, 0x1.e2cffe0000000p-26f},
    {0x1.ea4afa0000000p+0f, 0x1.52486c0000000p-27f}, {0x1.efa1be0000000p+0f, 0x1.cc2b440000000p-25f}, {0x1.f507660000000p+0f, -0x1.246eb00000000p-26f}, {0x1.fa7c180000000p+0f, 0x1.9e90d80000000p-28f}
};

__device__ __forceinline__ float exp_prod(float a, int* fell = nullptr) {
    if (!(a > -87.0f && a < 88.0f)) return (float) exp((double) a);
    const float k = rintf(a * 0x1.7154760000000p+6f);
    const int ki = (int) k;
    const float r1 = __fmaf_rn(-k, 0x1.62e4300000000p-7f, a);                       // exact (see the header)
    // r = r1 - k * (C_LO + C_LO2) as rh + rl
    const float p = __fmul_rn(k, -0x1.05c6100000000p-35f);
    const float pe = __fmaf_rn(k, -0x1.05c6100000000p-35f, -p);                        // k * C_LO = p + pe exactly
    const float rh = __fsub_rn(r1, p);
    const float bb = __fsub_rn(rh, r1);
    float rl = __fadd_rn(__fsub_rn(r1, __fsub_rn(rh, bb)), __fsub_rn(-p, bb));   // TwoSum(r1, -p) error
    rl = __fsub_rn(rl, __fmaf_rn(k, -0x1.950d880000000p-60f, pe));
    // e^r = 1 + rh + rl + rh^2/2 + rh*rl + tail
    const float hr = __fmul_rn(0.5f, rh);
    const float sq = __fmul_rn(hr, rh);
    const float sqe = __fmaf_rn(hr, rh, -sq);                           // rh^2/2 = sq + sqe exactly
    const float tail = __fmul_rn(__fmul_rn(__fmul_rn(rh, rh), rh),
                                 __fmaf_rn(rh, __fmaf_rn(rh, 0x1.111112p-7f, 0x1.555556p-5f), 0x1.555556p-3f));
    const float mh = __fadd_rn(rh, sq);                                 // |rh| >= |sq|: Fast2Sum
    const float ml = __fsub_rn(sq, __fsub_rn(mh, rh));
    const float L = __fadd_rn(__fadd_rn(__fadd_rn(ml, rl), __fadd_rn(sqe, __fmul_rn(rh, rl))), tail);
    const float eh = __fadd_rn(1.0f, mh);
    const float el = __fadd_rn(__fsub_rn(mh, __fsub_rn(eh, 1.0f)), L);
    const float2 t = EXP_T64[ki & 63];
    const float ph = __fmul_rn(t.x, eh);
    const float pl = __fadd_rn(__fmaf_rn(t.x, eh, -ph), __fmaf_rn(t.x, el, __fmul_rn(t.y, eh)));
    const float y = __fadd_rn(ph, pl);
    const float yl = __fsub_rn(pl, __fsub_rn(y, ph));
    // half an ulp of y on yl's side (a quarter on the lower side of a power of two)
    const int yb = __float_as_int(y);
    float h = __int_as_float((yb & 0x7f800000) - (24 << 23));
    if (yl < 0.0f && (yb & 0x007fffff) == 0) h = __fmul_rn(0.5f, h);
    if (!(__fsub_rn(h, fabsf(yl)) > __fmul_rn(y, 0x1p-42f))) {
        if (fell) *fell = 1;
        return (float) exp((double) a);
    }
    return __int_as_float(yb + ((ki >> 6) << 23));                      // y * 2^n, normal throughout the range
}

// Exhaustive check: every float with bit pattern in [b0, b1) (a sign-homogeneous range): counts
// [0] exp_prod != (float) exp((double) a) (bitwise), [1] in-range fallbacks to the double statement
// (ambiguous rounding).
__global__ void me_exp_check_kernel(uint32_t b0, uint32_t b1, unsigned long long* cnt) {
    unsigned long long bad = 0, fb = 0;
    for (uint64_t b = b0 + (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; b < b1; b += (uint64_t) gridDim.x * blockDim.x) {
        const float a = __uint_as_float((uint32_t) b);
        const float ref = (float) exp((double) a);
        int fell = 0;
        const float v = exp_prod(a, &fell);
        bad += __float_as_uint(ref) != __float_as_uint(v);
        fb += fell;
    }
    if (bad) atomicAdd(cnt, bad);
    if (fb) atomicAdd(cnt + 1, fb);
}

// P16 gather (GLM53_MOE_FUSED16): production's fm_gather_kernel arithmetic, h13[r] = fp16(H(float(x16 * suh)) * r)
// with the fp16 x fp16 product ROUNDED (__hmul2, exllamav3's had_hf_r_128_inner<pre_scale> boundary), one warp per
// row, all 32 Hadamard blocks of the row. Rows are visited in the order perm[i] (rows grouped by token: the warps of a
// block read the same x row from L1/L2 instead of re-reading x from DRAM once per routed expert); rows >= num_rows
// (the live fat-row count) are skipped.
__global__ void __launch_bounds__(256) me_gather16_kernel(const half* __restrict__ x, const int* __restrict__ perm,
                                                         const int64_t* __restrict__ row_token,
                                                         const int* __restrict__ row_expert,
                                                         const int64_t* __restrict__ suh_ptrs, half* __restrict__ h13,
                                                         const int* __restrict__ num_rows_ptr, int rows_cap) {
    const int lane = threadIdx.x & 31;
    const int nrows = *num_rows_ptr;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nw = (gridDim.x * blockDim.x) >> 5;
    for (int i = wid; i < rows_cap; i += nw) {
        const int r = perm != nullptr ? perm[i] : i;
        if (r < 0 || r >= nrows) continue;                   // warp-uniform
        const half* xr = x + (size_t) row_token[r] * F_HID_G + 4 * lane;
        const half* su = reinterpret_cast<const half*>(suh_ptrs[row_expert[r]]) + 4 * lane;
        half* dst = h13 + (size_t) r * F_HID_G + 4 * lane;
#pragma unroll 4
        for (int b = 0; b < F_HID_G / 128; ++b) {
            const uint2 xv = *reinterpret_cast<const uint2*>(xr + b * 128);
            const uint2 sv = *reinterpret_cast<const uint2*>(su + b * 128);
            const half2 p0 = __hmul2(*reinterpret_cast<const half2*>(&xv.x), *reinterpret_cast<const half2*>(&sv.x));
            const half2 p1 = __hmul2(*reinterpret_cast<const half2*>(&xv.y), *reinterpret_cast<const half2*>(&sv.y));
            float v[4] = {__low2float(p0), __high2float(p0), __low2float(p1), __high2float(p1)};
            fwht128(v, lane);
            const half2 o0 = __floats2half2_rn(__fmul_rn(v[0], HAD_SCALE), __fmul_rn(v[1], HAD_SCALE));
            const half2 o1 = __floats2half2_rn(__fmul_rn(v[2], HAD_SCALE), __fmul_rn(v[3], HAD_SCALE));
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&o0);
            ov.y = *reinterpret_cast<const uint32_t*>(&o1);
            *reinterpret_cast<uint2*>(dst + b * 128) = ov;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Down-input quantization: one warp per row of the segments of chunk `chunk` of `nchunks`:
//   a8d[r] = e4m3(H(float(a16[r]) * suh_d[row_expert[r]]) * r / s_d),  dsc[r] = s_d.

// One row (one warp). CG: read a16 through L2 only (rows written by other CTAs of the same kernel).
template <int K, bool CG>
__device__ __forceinline__ void actq_row(const half* a16row, const half* suh_row, uint8_t* dst, float* dsc_r, int lane) {
    constexpr int NB = K / 128;
    const half* suh = suh_row + 4 * lane;
    const half* src = a16row + 4 * lane;
    float v[NB][4];
    float amax = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        float4 a;
        if constexpr (CG) {
            const uint2 r = __ldcg(reinterpret_cast<const uint2*>(src + b * 128));
            const half2 lo = *reinterpret_cast<const half2*>(&r.x), hi = *reinterpret_cast<const half2*>(&r.y);
            a = make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
        } else {
            a = ld_h4f(src + b * 128);
        }
        const float4 s = ld_h4f(suh + b * 128);
        v[b][0] = a.x * s.x; v[b][1] = a.y * s.y; v[b][2] = a.z * s.z; v[b][3] = a.w * s.w;
        fwht128(v[b], lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[b][i] = __fmul_rn(v[b][i], HAD_SCALE);
            amax = fmaxf(amax, fabsf(v[b][i]));
        }
    }
    const float sc = row_scale(warp_max(amax));
#pragma unroll
    for (int b = 0; b < NB; ++b) store_q4(dst + b * 128, lane, v[b], sc);
    if (lane == 0) *dsc_r = sc;
}

template <int K>
__global__ void __launch_bounds__(256) me_actq_kernel(const half* __restrict__ a16, const int* __restrict__ row_expert,
                                                     const int64_t* __restrict__ suh_ptrs, uint8_t* __restrict__ a8,
                                                     float* __restrict__ dsc, const int* __restrict__ seg_row0,
                                                     const int* __restrict__ seg_rows,
                                                     const int* __restrict__ num_segs_ptr, int chunk, int nchunks) {
    const int lane = threadIdx.x & 31;
    const int nsegs = *num_segs_ptr;
    const int s0 = (int) ((long long) nsegs * chunk / nchunks), s1 = (int) ((long long) nsegs * (chunk + 1) / nchunks);
    if (s0 >= s1) return;
    const int r0 = seg_row0[s0], r1 = seg_row0[s1 - 1] + seg_rows[s1 - 1];
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nw = (gridDim.x * blockDim.x) >> 5;
    for (int r = r0 + wid; r < r1; r += nw)
        actq_row<K, false>(a16 + (size_t) r * K, reinterpret_cast<const half*>(suh_ptrs[row_expert[r]]), a8 + (size_t) r * K,
                           dsc + r, lane);
}

// ---------------------------------------------------------------------------------------------------------
// async copies / ldmatrix

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp4ca(void* smem, const void* gmem) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp8ca(void* smem, const void* gmem) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&a)[4], const void* smem) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(sa));
}

// A stage row: KSB*16 bytes = KSB chunks of 16 B. KSB 4: chunk c of row r at c ^ ((r >> 1) & 3); KSB 8: c ^ (r & 7).
// Either way the 8 rows of an ldmatrix phase hit 8 distinct 16-byte bank groups.
template <int KSB>
__device__ __forceinline__ int aswz(int r, int c) {
    if constexpr (KSB == 8) return c ^ (r & 7);
    else return c ^ ((r >> 1) & 3);
}

// ---------------------------------------------------------------------------------------------------------
// Mainloop (E4's structure, e4m3 operands). A CTA has NCB warps; warp w owns 16-column block w of the CTA's output
// columns for all TM = MB*16 rows of the tile: each trellis tile is decoded once per CTA and feeds up to MB row
// blocks. Block cb reads matrix (cb < SPLIT ? W0 : W1) at trellis tile column colbase + cb % SPLIT.
// A (shared by all warps) goes through shared memory: STAGES stages of [TM][KSB*16 B] filled by cp.async, one
// barrier per stage. B is private to its warp (lane l only ever reads word l of its tiles), so it bypasses shared
// memory: each lane loads its KSB trellis words of the NEXT stage into registers while computing this one. Both
// pipelines run continuously across a CTA's items (the next item's first stages load during this item's last steps).

template <int MB, int NCB, int KSB>
struct Cfg {
    static constexpr int TM = MB * 16;
    static constexpr int THREADS = NCB * 32;
    static constexpr int AROW = KSB * 16;                    // A bytes per row per stage
    static constexpr int STAGE_BYTES = TM * AROW;
};

struct Item {
    int row0, rows, mbs;
    const uint32_t* W0;
    const uint32_t* W1;
    int colbase;
    int e;
    int nb;
};

// Part `part` of NP of one stage's A copies.
template <int MB, int NCB, int KSB, int NP>
__device__ __forceinline__ void load_a_part(unsigned char* buf, const uint8_t* __restrict__ a, int K, const Item& it,
                                            int kt, int part) {
    using CF = Cfg<MB, NCB, KSB>;
    const int tid = threadIdx.x;
    const int load_rows = it.mbs * 16;
    for (int c = tid + part * CF::THREADS; c < load_rows * KSB; c += NP * CF::THREADS) {
        const int r = c / KSB, ch = c % KSB;
        const int sr = r < it.rows ? r : it.rows - 1;
        cp16(buf + r * CF::AROW + aswz<KSB>(r, ch) * 16, a + (size_t) (it.row0 + sr) * K + kt * CF::AROW + ch * 16);
    }
}

// This lane's trellis words of stage kt of item it (KSB k16 tiles of its warp's column block).
template <int NCB, int SPLIT, int KSB>
__device__ __forceinline__ void load_b(uint32_t (&b)[KSB], int ntiles, const Item& it, int kt) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t* W = (warp < SPLIT ? it.W0 : it.W1) + ((size_t) kt * KSB * ntiles + it.colbase + warp % SPLIT) * 32 + lane;
#pragma unroll
    for (int ks = 0; ks < KSB; ++ks) b[ks] = __ldg(W + (size_t) ks * ntiles * 32);
}

// DBG (experiment builds only, -DME_DEBUG_VARIANTS): 1 no epilogue (accumulators summed), 2 no copies after the
// first item's prologue, 4 no trellis decode (raw words as fragments), 8 no atomics / stores in the epilogue.
// bc: this lane's B words of the stage about to be computed (carried across items).
template <int MB, int NCB, int SPLIT, int STAGES, int KSB, int DBG = 0>
__device__ __forceinline__ void mainloop(const uint8_t* __restrict__ a, int K, int ntiles, const Item& cur,
                                         const Item& nxt, bool has_nxt, int& g, unsigned char* smem,
                                         uint32_t (&bc)[KSB], float (&acc)[MB][2][4]) {
    using CF = Cfg<MB, NCB, KSB>;
    const int lane = threadIdx.x & 31;
    const int nk = K / CF::AROW;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mb][h][c] = 0.f;
    const int mbs = cur.mbs;
    const int arow = lane & 15, ahi = lane >> 4;
    for (int kt = 0; kt < nk; ++kt, ++g) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int ld = kt + STAGES - 1;
        unsigned char* lbuf = smem + ((g + STAGES - 1) % STAGES) * CF::STAGE_BYTES;
        const bool ld_cur = ld < nk, ld_nxt = !ld_cur && has_nxt;
        const Item& lit = ld_cur ? cur : nxt;
        const int lkt = ld_cur ? ld : ld - nk;
        // B of the next stage (this item's kt + 1, or the next item's stage 0)
        uint32_t bn[KSB];
        if (kt + 1 < nk) load_b<NCB, SPLIT, KSB>(bn, ntiles, cur, kt + 1);
        else if (has_nxt) load_b<NCB, SPLIT, KSB>(bn, ntiles, nxt, 0);
        else {
#pragma unroll
            for (int ks = 0; ks < KSB; ++ks) bn[ks] = 0u;
        }
        const unsigned char* buf = smem + (g % STAGES) * CF::STAGE_BYTES;
#pragma unroll
        for (int j = 0; j < KSB / 2; ++j) {
            uint32_t t0a[2], t0b[2], t1a[2], t1b[2], bf0[2], bf1[2];
            if (DBG & 4) {
                bf0[0] = bc[2 * j]; bf0[1] = bc[2 * j + 1]; bf1[0] = bc[2 * j] ^ bc[2 * j + 1]; bf1[1] = bc[2 * j] + bc[2 * j + 1];
            } else {
                decode_tile(bc[2 * j], lane, t0a, t0b);
                decode_tile(bc[2 * j + 1], lane, t1a, t1b);
                frag8(t0a, t1a, bf0);
                frag8(t0b, t1b, bf1);
            }
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb < mbs) {
                    const int r = mb * 16 + arow, ch = j * 2 + ahi;
                    uint32_t af[4];
                    ldsm_x4(af, buf + r * CF::AROW + aswz<KSB>(r, ch) * 16);
                    mma16832(acc[mb][0], af, bf0);
                    mma16832(acc[mb][1], af, bf1);
                }
            }
            if (!(DBG & 2) && (ld_cur || ld_nxt)) load_a_part<MB, NCB, KSB, KSB / 2>(lbuf, a, K, lit, lkt, j);
        }
        cp_commit();
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) bc[ks] = bn[ks];
    }
}

template <int MB, int NCB, int SPLIT, int STAGES, int KSB>
__device__ __forceinline__ void prologue(const uint8_t* __restrict__ a, int K, int ntiles, const Item& first,
                                         unsigned char* smem, uint32_t (&bc)[KSB]) {
    using CF = Cfg<MB, NCB, KSB>;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        load_a_part<MB, NCB, KSB, 1>(smem + st * CF::STAGE_BYTES, a, K, first, st, 0);
        cp_commit();
    }
    load_b<NCB, SPLIT, KSB>(bc, ntiles, first, 0);
}

// Stage row block mb0 of the accumulators into sc[16][CSTR] (fp32); warp w -> columns w*16 ..
template <int MB, int CSTR>
__device__ __forceinline__ void stage_acc(float* sc, const float (&acc)[MB][2][4], int mb0, int warp, int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = warp * 16 + h * 8 + 2 * t;
        float* d = sc + g * CSTR + col;
        *reinterpret_cast<float2*>(d) = make_float2(acc[mb0][h][0], acc[mb0][h][1]);
        *reinterpret_cast<float2*>(d + 8 * CSTR) = make_float2(acc[mb0][h][2], acc[mb0][h][3]);
    }
}

template <int MB, int CSTR>
__device__ __forceinline__ void stage_acc_rt(float* sc, const float (&acc)[MB][2][4], int mb0, int warp, int lane) {
    static_assert(MB <= 8, "");
    switch (mb0) {
#define ME_STAGE_CASE(i) case i: if constexpr (i < MB) stage_acc<MB, CSTR>(sc, acc, i, warp, lane); break;
        ME_STAGE_CASE(0) ME_STAGE_CASE(1) ME_STAGE_CASE(2) ME_STAGE_CASE(3)
        ME_STAGE_CASE(4) ME_STAGE_CASE(5) ME_STAGE_CASE(6) ME_STAGE_CASE(7)
#undef ME_STAGE_CASE
        default: break;
    }
}

template <int MB>
__device__ __forceinline__ void sum_acc(const float (&acc)[MB][2][4], float* sink) {
    float sum = 0.f;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) sum += acc[mb][h][c];
    if (sum == 1234.5f) sink[0] = sum;
}

template <int MB, int STAGES, int KSB, int NCB = 16>
struct Smem {
    static constexpr int CSTR = NCB * 16 + 8;
    static constexpr int PIPE = STAGES * Cfg<MB, NCB, KSB>::STAGE_BYTES;
    static constexpr int TOTAL = PIPE + 16 * CSTR * 4;
};

// ---------------------------------------------------------------------------------------------------------
// gate/up GEMM + scale + output transforms + SwiGLU -> a16 (fp16, the down input before its transforms).
// CTA = 16 warps: 8 gate blocks + 8 up blocks of one 128-column block of the intermediate dim. Item = (segment,
// 128-column block), column block fastest.

template <int MB, int STAGES, int KSB, int DBG = 0>
__global__ void __launch_bounds__(512, 1) me_gateup_kernel(
    const uint8_t* __restrict__ a8, const float* __restrict__ asc, const int64_t* __restrict__ gate_ptrs,
    const int64_t* __restrict__ up_ptrs, const int64_t* __restrict__ gate_svh, const int64_t* __restrict__ up_svh,
    half* __restrict__ a16, const int* __restrict__ seg_expert, const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows, const int* __restrict__ num_segs_ptr, int K, int N, float limit, int chunk,
    int nchunks) {
    constexpr int NCB = 16;
    using SM = Smem<MB, STAGES, KSB>;
    constexpr int CSTR = SM::CSTR;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N / 128, ntiles = N / 16;
    const int nsegs = *num_segs_ptr;
    const int s0 = (int) ((long long) nsegs * chunk / nchunks), s1 = (int) ((long long) nsegs * (chunk + 1) / nchunks);
    const int items = (s1 - s0) * nblk;
    float* sc = reinterpret_cast<float*>(smem + SM::PIPE);
    auto info = [&](int item) {
        Item it;
        const int seg = s0 + item / nblk;
        it.nb = item - seg * nblk;
        it.e = seg_expert[seg];
        it.row0 = seg_row0[seg];
        it.rows = max(seg_rows[seg], 1);
        it.mbs = (it.rows + 15) >> 4;
        it.W0 = reinterpret_cast<const uint32_t*>(gate_ptrs[it.e]);
        it.W1 = reinterpret_cast<const uint32_t*>(up_ptrs[it.e]);
        it.colbase = it.nb * 8;
        return it;
    };
    int item = blockIdx.x;
    if (item >= items) return;
    Item cur = info(item);
    uint32_t bc[KSB];
    prologue<MB, NCB, 8, STAGES, KSB>(a8, K, ntiles, cur, smem, bc);
    int g = 0;
    for (; item < items; item += gridDim.x) {
        const int nitem = item + gridDim.x;
        const bool has_nxt = nitem < items;
        const Item nxt = has_nxt ? info(nitem) : cur;
        float acc[MB][2][4];
        mainloop<MB, NCB, 8, STAGES, KSB, DBG>(a8, K, ntiles, cur, nxt, has_nxt, g, smem, bc, acc);
        if (DBG & 1) { sum_acc<MB>(acc, reinterpret_cast<float*>(a16)); cur = nxt; continue; }
        const int rows = cur.rows, mbs = cur.mbs, e = cur.e;
        const int n0 = cur.nb * 128 + 4 * lane;
        const float4 svg = ld_h4f(reinterpret_cast<const half*>(gate_svh[e]) + n0);
        const float4 svu = ld_h4f(reinterpret_cast<const half*>(up_svh[e]) + n0);
        const float sg[4] = {svg.x, svg.y, svg.z, svg.w}, su[4] = {svu.x, svu.y, svu.z, svu.w};
        half* outrow = a16 + (size_t) cur.row0 * N + n0;
#pragma unroll 1
        for (int mb0 = 0; mb0 < mbs; ++mb0) {
            stage_acc_rt<MB, CSTR>(sc, acc, mb0, warp, lane);
            __syncthreads();
            const int tr = mb0 * 16 + warp;                         // row within the tile (one row per warp)
            const float s = asc[cur.row0 + (tr < rows ? tr : 0)];
            const float4 gq = *reinterpret_cast<const float4*>(sc + warp * CSTR + 4 * lane);
            const float4 uq = *reinterpret_cast<const float4*>(sc + warp * CSTR + 128 + 4 * lane);
            float gg[4] = {__fmul_rn(gq.x, s), __fmul_rn(gq.y, s), __fmul_rn(gq.z, s), __fmul_rn(gq.w, s)};
            float u[4] = {__fmul_rn(uq.x, s), __fmul_rn(uq.y, s), __fmul_rn(uq.z, s), __fmul_rn(uq.w, s)};
            fwht128(gg, lane);
            fwht128(u, lane);
            float act[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float gv = __fmul_rn(__fmul_rn(gg[i], HAD_SCALE), sg[i]);
                float uv = __fmul_rn(__fmul_rn(u[i], HAD_SCALE), su[i]);
                gv = fminf(gv, limit);
                uv = fminf(fmaxf(uv, -limit), limit);
                act[i] = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);   // torch silu(g) * u
            }
            const half2 h01 = __floats2half2_rn(act[0], act[1]), h23 = __floats2half2_rn(act[2], act[3]);
            uint2 ov;
            ov.x = *reinterpret_cast<const uint32_t*>(&h01);
            ov.y = *reinterpret_cast<const uint32_t*>(&h23);
            if (!(DBG & 8) && tr < rows) *reinterpret_cast<uint2*>(outrow + (size_t) tr * N) = ov;
            __syncthreads();
        }
        cur = nxt;
    }
    cp_wait<0>();
}

// ---------------------------------------------------------------------------------------------------------
// down GEMM + scale + output Hadamard + svh + route weight + fp32 scatter-add. CTA = 16 warps = 256 columns of the
// hidden dim (two 128-column Hadamard blocks). Item = (segment, 256-column block).

template <int MB, int STAGES, int KSB, int DBG = 0>
__global__ void __launch_bounds__(512, 1) me_down_kernel(
    const uint8_t* __restrict__ a8, const float* __restrict__ dsc, const int64_t* __restrict__ down_ptrs,
    const int64_t* __restrict__ down_svh, float* __restrict__ out, const int64_t* __restrict__ row_token,
    const float* __restrict__ row_weight, const int* __restrict__ seg_expert, const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows, const int* __restrict__ num_segs_ptr, int K, int N, int chunk, int nchunks) {
    constexpr int NCB = 16;
    using SM = Smem<MB, STAGES, KSB, NCB>;
    constexpr int CSTR = SM::CSTR;
    constexpr int CW = NCB * 16;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nblk = N / CW, ntiles = N / 16;
    const int nsegs = *num_segs_ptr;
    const int s0 = (int) ((long long) nsegs * chunk / nchunks), s1 = (int) ((long long) nsegs * (chunk + 1) / nchunks);
    const int items = (s1 - s0) * nblk;
    float* sc = reinterpret_cast<float*>(smem + SM::PIPE);
    auto info = [&](int item) {
        Item it;
        const int seg = s0 + item / nblk;
        it.nb = item % nblk;
        it.e = seg_expert[seg];
        it.row0 = seg_row0[seg];
        it.rows = max(seg_rows[seg], 1);
        it.mbs = (it.rows + 15) >> 4;
        it.W0 = it.W1 = reinterpret_cast<const uint32_t*>(down_ptrs[it.e]);
        it.colbase = it.nb * NCB;
        return it;
    };
    int item = blockIdx.x;
    if (item >= items) return;
    Item cur = info(item);
    uint32_t bc[KSB];
    prologue<MB, NCB, NCB, STAGES, KSB>(a8, K, ntiles, cur, smem, bc);
    int g = 0;
    for (; item < items; item += gridDim.x) {
        const int nitem = item + gridDim.x;
        const bool has_nxt = nitem < items;
        const Item nxt = has_nxt ? info(nitem) : cur;
        float acc[MB][2][4];
        mainloop<MB, NCB, NCB, STAGES, KSB, DBG>(a8, K, ntiles, cur, nxt, has_nxt, g, smem, bc, acc);
        if (DBG & 1) { sum_acc<MB>(acc, out); cur = nxt; continue; }
        const int rows = cur.rows, mbs = cur.mbs;
        const half* svh = reinterpret_cast<const half*>(down_svh[cur.e]) + cur.nb * CW + 4 * lane;
        const float4 sv0 = ld_h4f(svh), sv1 = ld_h4f(svh + 128);
        float* outc = out + cur.nb * CW + 4 * lane;
#pragma unroll 1
        for (int mb0 = 0; mb0 < mbs; ++mb0) {
            stage_acc_rt<MB, CSTR>(sc, acc, mb0, warp, lane);
            __syncthreads();
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int pr = warp + rr * NCB;                     // (row, 128-block) pair
                const int r = pr >> 1, hf = pr & 1;
                const int tr = mb0 * 16 + r;
                const bool valid = tr < rows;
                const int frow = cur.row0 + (valid ? tr : 0);
                const float s = dsc[frow];
                const float4 q = *reinterpret_cast<const float4*>(sc + r * CSTR + hf * 128 + 4 * lane);
                float v[4] = {__fmul_rn(q.x, s), __fmul_rn(q.y, s), __fmul_rn(q.z, s), __fmul_rn(q.w, s)};
                fwht128(v, lane);
                const float4 sv = hf ? sv1 : sv0;
                const float w = row_weight[frow];
                float4 o;
                o.x = __fmul_rn(__fmul_rn(__fmul_rn(v[0], HAD_SCALE), sv.x), w);
                o.y = __fmul_rn(__fmul_rn(__fmul_rn(v[1], HAD_SCALE), sv.y), w);
                o.z = __fmul_rn(__fmul_rn(__fmul_rn(v[2], HAD_SCALE), sv.z), w);
                o.w = __fmul_rn(__fmul_rn(__fmul_rn(v[3], HAD_SCALE), sv.w), w);
                if constexpr ((DBG & 16) != 0) {
                    if (valid) *reinterpret_cast<float4*>(outc + row_token[frow] * (int64_t) N + hf * 128) = o;
                } else if constexpr ((DBG & 32) != 0) {
                    if (valid) atomicAdd(reinterpret_cast<float4*>(outc + (row_token[frow] & 511) * (int64_t) N + hf * 128), o);
                } else {
                    if (!(DBG & 8) && valid) atomicAdd(reinterpret_cast<float4*>(outc + row_token[frow] * (int64_t) N + hf * 128), o);
                }
            }
            __syncthreads();
        }
        cur = nxt;
    }
    cp_wait<0>();
}

// ---------------------------------------------------------------------------------------------------------
// Fused persistent kernel: every gate/up item and every down item of the layer in ONE launch, interleaved so that
// each SM alternates the compute-bound gate/up work with the L2/DRAM-bound fp32 scatter-add of the down epilogue.
// Item order (a global ticket counter hands out items in this order): GU(seg 0..L-1), then rounds r = 0.. of
// [8 GU items of seg L+r, 16 DN items of seg r], then the DN items of the last L segs (L = min(lag, nsegs)).
// Dataflow: the CTA that finishes the 8th gate/up item of a segment quantizes that segment's down-input rows (the
// actq arithmetic, a16 read through L2) and then releases ready[seg]; a down item of seg waits for ready[seg].
// Deadlock-free: an item only waits for items with smaller tickets, and every ticket is held by a resident CTA.

struct FusedArgs {
    const uint8_t* a8;
    const float* asc;
    uint8_t* a8d;
    float* dsc;
    half* a16;
    float* out;
    __nv_bfloat16* outb;        // OB != 0: the bf16 accumulator (zeroed by gather2), else unused
    const int64_t* gate_ptrs;
    const int64_t* up_ptrs;
    const int64_t* gate_svh;
    const int64_t* up_svh;
    const int64_t* down_ptrs;
    const int64_t* down_suh;
    const int64_t* down_svh;
    const int64_t* row_token;
    const float* row_weight;
    const int* seg_expert;
    const int* seg_row0;
    const int* seg_rows;
    const int* num_segs;
    int* sync;                  // [0] ticket, [1 .. max_segs] gate/up items done per seg, then ready[max_segs]
    int max_segs;
    int lag;
    float limit;
};

constexpr int F_HID = 4096, F_INT = 1024;

struct Job {
    int kind;                   // 0 gate/up, 1 down, -1 none
    int seg, nb, e, row0, rows, mbs;
    const uint8_t* a;
    int K;                      // A row length in BYTES (e4m3: = k; DN16 down: 2 * k, fp16 rows)
    const uint32_t* wl;         // this lane's trellis word of k16 tile 0 of its warp's column block
    int wstride;                // words between consecutive k16 tiles
    int tps;                    // k16 trellis tiles per A stage (e4m3: KSB; fp16 A: KSB / 2)
    int f16;                    // 1: fp16 A rows x fp16 B on mma m16n8k16 (DN16 down jobs)
};

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

// Fire-and-forget fp32x4 add (the compiler emits ATOM with a discarded result for atomicAdd(float4*) in the fused
// kernel; RED does not wait for a return value). Same arithmetic: fp32 add, round to nearest, subnormals flushed.
__device__ __forceinline__ void red_add4(float* p, const float4& v) {
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z),
                 "f"(v.w) : "memory");
}

// opt-moe OB (bf16 accumulator): the contribution rounded to bf16 (RN) and added in bf16 (RN, no flush) by the L2
// atomic unit. 4 values (8 B) per op (OB 1) or 8 values (16 B) per op (OB 2, two rows exchanged between lane pairs).
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void red_add_bf4(__nv_bfloat16* p, uint32_t w0, uint32_t w1) {
    asm volatile("red.relaxed.gpu.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n" ::"l"(p), "r"(w0), "r"(w1) : "memory");
}
__device__ __forceinline__ void red_add_bf8(__nv_bfloat16* p, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    asm volatile("red.relaxed.gpu.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};\n" ::"l"(p), "r"(w0), "r"(w1),
                 "r"(w2), "r"(w3) : "memory");
}

template <int DN16 = 0, int TG = 0>
__device__ __forceinline__ Job make_job(const FusedArgs& A, int tk, int nsegs) {
    Job j;
    const int L = min(A.lag, nsegs);
    const int head = 8 * L, body = 24 * (nsegs - L);
    int i = tk;
    if (i < head) {
        j.kind = 0; j.seg = i >> 3; j.nb = i & 7;
    } else if ((i -= head) < body) {
        const int r = i / 24, w = i - r * 24;
        if (w < 8) { j.kind = 0; j.seg = L + r; j.nb = w; }
        else { j.kind = 1; j.seg = r; j.nb = w - 8; }
    } else if ((i -= body) < 16 * L) {
        j.kind = 1; j.seg = nsegs - L + (i >> 4); j.nb = i & 15;
    } else {
        j.kind = -1; j.seg = 0; j.nb = 0;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int seg = j.kind >= 0 ? j.seg : 0;
    j.e = A.seg_expert[seg];
    j.row0 = A.seg_row0[seg];
    j.rows = max(A.seg_rows[seg], 1);
    j.mbs = (j.rows + 15) >> 4;
    j.tps = 8; j.f16 = 0;
    if (j.kind == 1 && DN16) {
        // DN16: the down input is the gate/up epilogue's ROTATED fp16 rows (H(fp16(act) * suh_d) * r, in a16)
        j.a = reinterpret_cast<const uint8_t*>(A.a16); j.K = 2 * F_INT; j.wstride = (F_HID / 16) * 32;
        j.tps = 4; j.f16 = 1;
        j.wl = reinterpret_cast<const uint32_t*>(A.down_ptrs[j.e]) + (j.nb * 16 + warp) * 32 + lane;
    } else if (j.kind == 1) {
        j.a = A.a8d; j.K = F_INT; j.wstride = (F_HID / 16) * 32;
        j.wl = reinterpret_cast<const uint32_t*>(A.down_ptrs[j.e]) + (j.nb * 16 + warp) * 32 + lane;
    } else {
        j.a = A.a8; j.K = F_HID; j.wstride = (F_INT / 16) * 32;
        j.wl = reinterpret_cast<const uint32_t*>(warp < 8 ? A.gate_ptrs[j.e] : A.up_ptrs[j.e]) +
               (j.nb * 8 + (warp & 7)) * 32 + lane;
    }
    return j;
}

// TG: the row tokens of the current / next gate/up item of the fused kernel (static shared memory; only kernels that
// reference it allocate it)
__shared__ int g_srt[2][128];

// sb (TG): the g_srt buffer holding this item's row tokens (gate/up jobs: A row r = a8[g_srt[sb][r]]); unused otherwise
template <int MB, int NCB, int KSB, int NP, int TG = 0>
__device__ __forceinline__ void fload_a(unsigned char* buf, const Job& j, int kt, int part, int sb = 0) {
    using CF = Cfg<MB, NCB, KSB>;
    const int tid = threadIdx.x;
    const int load_rows = j.mbs * 16;
    for (int c = tid + part * CF::THREADS; c < load_rows * KSB; c += NP * CF::THREADS) {
        const int r = c / KSB, ch = c % KSB;
        const int sr = r < j.rows ? r : j.rows - 1;
        size_t src = (size_t) (j.row0 + sr);
        if constexpr (TG == 1) {
            if (j.kind == 0) src = (size_t) g_srt[sb][sr];
        } else if constexpr (TG == 2) {     // probe: the tokens are staged and read, the per-pair row is loaded
            if (j.kind == 0 && g_srt[sb][sr] == -1) src = 0;
        }
        cp16(buf + r * CF::AROW + aswz<KSB>(r, ch) * 16, j.a + src * j.K + kt * CF::AROW + ch * 16);
    }
}

// AT (opt-moe2, MS & 128): the A copy addresses of a job computed ONCE (at its first stage load) into a per-thread
// table in shared memory (2 buffers, the current / next job like g_srt; 512 threads x 2 chunks x 8 B each); every later
// stage is one 8-byte shared load + one add per chunk instead of the row / token / swizzle / 64-bit address arithmetic
// of fload_a. With 128 rows x 8 chunks per stage and 512 threads, thread t copies chunks t and t + 512 (rows t / 8 and
// 64 + t / 8, chunk t % 8) of every stage of a job: fixed per job. Same bytes to the same places.
template <int MB, int TG = 0>
__device__ __forceinline__ void at_fill(unsigned long long* tab, const Job& j, int sb) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int c = tid + 512 * p, r = c >> 3;
        const int sr = r < j.rows ? r : j.rows - 1;
        size_t src = (size_t) (j.row0 + sr);
        if constexpr (TG == 1) {
            if (j.kind == 0) src = (size_t) g_srt[sb][sr];
        }
        tab[2 * tid + p] = r < j.mbs * 16 ? (unsigned long long) (j.a + src * j.K + (c & 7) * 16) : 0ull;
    }
}
// part `part` (0..3) of stage kt from the table (parts 2, 3 hold no chunks); so0 = this thread's chunk offset in a stage
__device__ __forceinline__ void at_load(unsigned char* buf, const unsigned long long* tab, int kt, int part, int so0) {
    if (part < 2) {
        const unsigned long long p = tab[2 * threadIdx.x + part];
        if (p != 0ull) cp16(buf + so0 + part * 8192, reinterpret_cast<const unsigned char*>(p) + kt * 128);
    }
}

// BK (opt-moe2, MS & 256): the trellis word loads of a stage with compile-time strides (gate/up tiles 2048 words
// apart, down tiles 8192: immediate offsets, one address computation per stage). e4m3 jobs only (DN16 = 0).
template <int KSB>
__device__ __forceinline__ void fload_b_k(uint32_t (&b)[KSB], const Job& j, int kt) {
    if (j.kind == 0) {
        const uint32_t* w = j.wl + (size_t) kt * KSB * ((F_INT / 16) * 32);
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) b[ks] = __ldg(w + ks * ((F_INT / 16) * 32));
    } else {
        const uint32_t* w = j.wl + (size_t) kt * KSB * ((F_HID / 16) * 32);
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) b[ks] = __ldg(w + ks * ((F_HID / 16) * 32));
    }
}

template <int KSB, int DN16 = 0>
__device__ __forceinline__ void fload_b(uint32_t (&b)[KSB], const Job& j, int kt) {
    if constexpr (DN16) {
        const uint32_t* w = j.wl + (size_t) kt * j.tps * j.wstride;
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) b[ks] = ks < j.tps ? __ldg(w + (size_t) ks * j.wstride) : 0u;
    } else {
        const uint32_t* w = j.wl + (size_t) kt * KSB * j.wstride;
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) b[ks] = __ldg(w + (size_t) ks * j.wstride);
    }
}

template <int MB, int NCB, int STAGES, int KSB, int DN16 = 0, int TG = 0, int AT = 0>
__device__ __forceinline__ void fprologue(const Job& j, int g, unsigned char* smem, uint32_t (&bc)[KSB], int sb = 0,
                                          unsigned long long* at = nullptr, int so0 = 0) {
    using CF = Cfg<MB, NCB, KSB>;
    if constexpr (AT != 0) at_fill<MB, TG>(at + sb * 1024, j, sb);
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if constexpr (AT != 0) {
            at_load(smem + ((g + st) % STAGES) * CF::STAGE_BYTES, at + sb * 1024, st, 0, so0);
            at_load(smem + ((g + st) % STAGES) * CF::STAGE_BYTES, at + sb * 1024, st, 1, so0);
        } else {
            fload_a<MB, NCB, KSB, 1, TG>(smem + ((g + st) % STAGES) * CF::STAGE_BYTES, j, st, 0, sb);
        }
        cp_commit();
    }
    fload_b<KSB, DN16>(bc, j, 0);
}

// pf: the next job's first stages (A and B) may be loaded during this job's last steps.
// PDBG (experiment builds, removal probes; results are garbage): 4 = no trellis decode (raw words as fragments),
// 2 = no A copies after the item's prologue, 1 = no mma, 8 = no trellis (B) loads after the item's prologue,
// 16 = no ldmatrix (A fragments from registers).
template <int MB, int NCB, int STAGES, int KSB, int DN16 = 0, int PDBG = 0, int TG = 0, int LEAN = 0, int AT = 0,
          int BK = 0, int FULL = 0>
__device__ __forceinline__ void fmainloop(const Job& cur, const Job& nxt, bool pf, int& g, unsigned char* smem,
                                          uint32_t (&bc)[KSB], float (&acc)[MB][2][4], int pb = 0, uint32_t cx = 0u,
                                          unsigned long long* at = nullptr, int so0 = 0) {
    static_assert(BK == 0 || DN16 == 0, "BK: e4m3 jobs only");
    using CF = Cfg<MB, NCB, KSB>;
    const int lane = threadIdx.x & 31;
    const int nk = cur.K / CF::AROW;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mb][h][c] = 0.f;
    const int mbs = FULL ? MB : cur.mbs;                         // FULL (caller checked cur.mbs == MB)
    const int arow = lane & 15, ahi = lane >> 4;
    for (int kt = 0; kt < nk; ++kt, ++g) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int ld = kt + STAGES - 1;
        unsigned char* lbuf = smem + ((g + STAGES - 1) % STAGES) * CF::STAGE_BYTES;
        const bool ld_cur = ld < nk, ld_nxt = !ld_cur && pf;
        uint32_t bn[KSB];
        if constexpr ((PDBG & 8) != 0) {
#pragma unroll
            for (int ks = 0; ks < KSB; ++ks) bn[ks] = bc[ks] * 1664525u + (uint32_t) kt;
        } else if (BK) {
            if (kt + 1 < nk) fload_b_k<KSB>(bn, cur, kt + 1);
            else if (pf) fload_b_k<KSB>(bn, nxt, 0);
            else {
#pragma unroll
                for (int ks = 0; ks < KSB; ++ks) bn[ks] = 0u;
            }
        } else if (kt + 1 < nk) fload_b<KSB, DN16>(bn, cur, kt + 1);
        else if (pf) fload_b<KSB, DN16>(bn, nxt, 0);
        else {
#pragma unroll
            for (int ks = 0; ks < KSB; ++ks) bn[ks] = 0u;
        }
        const unsigned char* buf = smem + (g % STAGES) * CF::STAGE_BYTES;
#pragma unroll
        for (int j = 0; j < KSB / 2; ++j) {
            if (DN16 && cur.f16) {
                // fp16 A (16-byte chunk = k8): chunks 2j, 2j+1 = k16 = trellis tile j of this stage; the decoded fp16
                // values ARE the m16n8k16 B fragments (no e4m3 rounding, no k permutation)
                uint32_t b0[2], b1[2];
                if constexpr (LEAN != 0) decode_tile_lean(bc[j], lane, cx, b0, b1);
                else decode_tile(bc[j], lane, b0, b1);
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    if (mb < mbs) {
                        const int r = mb * 16 + arow, ch = j * 2 + ahi;
                        uint32_t af[4];
                        ldsm_x4(af, buf + r * CF::AROW + aswz<KSB>(r, ch) * 16);
                        mma16816(acc[mb][0], af, b0);
                        mma16816(acc[mb][1], af, b1);
                    }
                }
            } else {
            uint32_t t0a[2], t0b[2], t1a[2], t1b[2], bf0[2], bf1[2];
            if constexpr ((PDBG & 4) != 0) {
                bf0[0] = bc[2 * j]; bf0[1] = bc[2 * j + 1]; bf1[0] = bc[2 * j] ^ bc[2 * j + 1]; bf1[1] = bc[2 * j] + bc[2 * j + 1];
            } else if constexpr (LEAN != 0) {
            decode_tile_lean(bc[2 * j], lane, cx, t0a, t0b);
            decode_tile_lean(bc[2 * j + 1], lane, cx, t1a, t1b);
            frag8(t0a, t1a, bf0);
            frag8(t0b, t1b, bf1);
            } else {
            decode_tile(bc[2 * j], lane, t0a, t0b);
            decode_tile(bc[2 * j + 1], lane, t1a, t1b);
            frag8(t0a, t1a, bf0);
            frag8(t0b, t1b, bf1);
            }
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb < mbs) {
                    const int r = mb * 16 + arow, ch = j * 2 + ahi;
                    uint32_t af[4];
                    if constexpr ((PDBG & 16) != 0) {
                        af[0] = bc[mb & 7] + (uint32_t) r; af[1] = af[0] ^ (uint32_t) ch; af[2] = af[0] * 3u; af[3] = af[1] + 5u;
                    } else
                    ldsm_x4(af, buf + r * CF::AROW + aswz<KSB>(r, ch) * 16);
                    if constexpr ((PDBG & 1) != 0) {
                        acc[mb][0][0] += __uint_as_float(af[0] ^ bf0[0]); acc[mb][1][1] += __uint_as_float(af[1] ^ bf1[1]);
                    } else {
                    mma16832(acc[mb][0], af, bf0);
                    mma16832(acc[mb][1], af, bf1);
                    }
                }
            }
            }
            if constexpr (AT != 0) {
                if (ld_cur) at_load(lbuf, at + pb * 1024, ld, j, so0);
                else if (ld_nxt) {
                    if (j == 0 && ld == nk) at_fill<MB, TG>(at + (pb ^ 1) * 1024, nxt, pb ^ 1);   // nxt's first stage
                    at_load(lbuf, at + (pb ^ 1) * 1024, ld - nk, j, so0);
                }
            } else if ((PDBG & 2) == 0 || kt + STAGES - 1 >= nk) {
            if (ld_cur) fload_a<MB, NCB, KSB, KSB / 2, TG>(lbuf, cur, ld, j, pb);
            else if (ld_nxt) fload_a<MB, NCB, KSB, KSB / 2, TG>(lbuf, nxt, ld - nk, j, pb ^ 1);
            }
        }
        cp_commit();
#pragma unroll
        for (int ks = 0; ks < KSB; ++ks) bc[ks] = bn[ks];
    }
}

// ---------------------------------------------------------------------------------------------------------
// smem-B mainloop (opt-moe2 EXPERIMENT, MS & 1, debug builds only - measured SLOWER: +0.7 ms at 13,824, +2.3 ms at
// 4,289; docs/OPT_MOE2.md section 2). A restructured mainloop for the fused kernel, the same arithmetic.
// The shipped mainloop carries this lane's trellis words in registers (bc[8] for the stage being computed, bn[8]
// prefetched for the next one: 16 of the 128 registers) and, with no registers left, ptxas reuses ONE A-fragment
// register quad for all 8 row blocks: ldmatrix(mb) -> 2 mma -> ldmatrix(mb + 1) waits for those mma to read it, a
// serial ldmatrix-latency chain per k32 step, and the trellis decode of step j sits between the mma of steps j - 1
// and j. MS:
//   - B (the trellis words of every warp's KSB k16 tiles) is staged in shared memory with A, by cp.async in the same
//     commit groups (stage = [A: TM x KSB*16 B][B: 16 warps x KSB tiles x 32 lanes x 4 B]); a lane reads its two
//     words of step j with two conflict-free 32-bit loads. No B registers live across stages.
//   - the freed registers double-buffer the A fragment (ldmatrix of mb + 1 is issued before mb's two mma) and hold
//     step j + 1's decoded fragments (decode issued in the middle of step j's mma).
//   - STAGES 3 (3 x 32 KB); the epilogue's fp32 staging (16 x CSTR floats) lives in the stage buffer the mainloop
//     consumed last (free until the next item's iteration 0 loads into it, after its loop-head barrier): one extra
//     barrier per item before the first staging write.
// Every accumulator receives the same mma, with the same fragments, in the same k order: bitwise the same result.
template <int MB, int KSB>
struct MsCfg {
    static constexpr int A_BYTES = MB * 16 * KSB * 16;
    static constexpr int B_BYTES = 16 * KSB * 128;
    static constexpr int STAGE = A_BYTES + B_BYTES;
};

// The trellis words of stage kt of job j for this warp (all its tiles of the stage) -> bbuf (this warp's KSB x 128 B).
template <int KSB, int DN16 = 0>
__device__ __forceinline__ void fload_bs(unsigned char* bbuf, const Job& j, int kt) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tps = DN16 ? j.tps : KSB;
    const uint32_t* w0 = j.wl - lane + (size_t) kt * tps * j.wstride;
    unsigned char* dst = bbuf + warp * KSB * 128;
#pragma unroll
    for (int i = 0; i < KSB / 4; ++i) {
        const int c = lane + 32 * i, ks = c >> 3, q = c & 7;
        if (!DN16 || ks < tps) cp16(dst + ks * 128 + q * 16, w0 + (size_t) ks * j.wstride + q * 4);
    }
}

template <int MB, int STAGES, int KSB, int DN16 = 0, int TG = 0>
__device__ __forceinline__ void fprologue_ms(const Job& j, int g, unsigned char* smem, int sb = 0) {
    using MC = MsCfg<MB, KSB>;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        unsigned char* b = smem + ((g + st) % STAGES) * MC::STAGE;
        fload_bs<KSB, DN16>(b + MC::A_BYTES, j, st);
        fload_a<MB, 16, KSB, 1, TG>(b, j, st, 0, sb);
        cp_commit();
    }
}

template <int LEAN = 0>
__device__ __forceinline__ void decode_pair8(uint32_t w0, uint32_t w1, int lane, uint32_t (&bf0)[2], uint32_t (&bf1)[2],
                                             uint32_t cx = 0u) {
    uint32_t t0a[2], t0b[2], t1a[2], t1b[2];
    if constexpr (LEAN != 0) {
        decode_tile_lean(w0, lane, cx, t0a, t0b);
        decode_tile_lean(w1, lane, cx, t1a, t1b);
    } else {
        decode_tile(w0, lane, t0a, t0b);
        decode_tile(w1, lane, t1a, t1b);
    }
    frag8(t0a, t1a, bf0);
    frag8(t0b, t1b, bf1);
}

template <int MB, int STAGES, int KSB, int DN16 = 0, int TG = 0, int MS = 1>
__device__ __forceinline__ void fmainloop_ms(const Job& cur, const Job& nxt, bool pf, int& g, unsigned char* smem,
                                             float (&acc)[MB][2][4], int pb = 0) {
    uint32_t cx = 0x3B603B60u;                                   // the lean decode's XOR constant, in a register
    if constexpr ((MS & 8) != 0) asm volatile("mov.b32 %0, %1;" : "=r"(cx) : "r"(cx));
    using MC = MsCfg<MB, KSB>;
    constexpr int AROW = KSB * 16;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nk = cur.K / AROW;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mb][h][c] = 0.f;
    const int mbs = cur.mbs;
    const int arow = lane & 15, ahi = lane >> 4;
    const int aoff = arow * AROW;
    for (int kt = 0; kt < nk; ++kt, ++g) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int ld = kt + STAGES - 1;
        unsigned char* lbuf = smem + ((g + STAGES - 1) % STAGES) * MC::STAGE;
        const bool ld_cur = ld < nk, ld_nxt = !ld_cur && pf;
        if (ld_cur) fload_bs<KSB, DN16>(lbuf + MC::A_BYTES, cur, ld);
        else if (ld_nxt) fload_bs<KSB, DN16>(lbuf + MC::A_BYTES, nxt, ld - nk);
        const unsigned char* buf = smem + (g % STAGES) * MC::STAGE;
        const uint32_t* bw = reinterpret_cast<const uint32_t*>(buf + MC::A_BYTES + warp * KSB * 128) + lane;
        if (DN16 && cur.f16) {
            // fp16 A rows x fp16 trellis decode on m16n8k16 (tile j = k16 step j), as fmainloop's DN16 branch
            uint32_t b0[2], b1[2];
            decode_tile(bw[0], lane, b0, b1);
#pragma unroll
            for (int j = 0; j < KSB / 2; ++j) {
                const uint32_t wn = j + 1 < KSB / 2 ? bw[(j + 1) * 32] : 0u;
                const int ch = j * 2 + ahi;
                const unsigned char* ab = buf + aoff + aswz<KSB>(arow, ch) * 16;
                uint32_t af[2][4];
                uint32_t n0[2], n1[2];
                ldsm_x4(af[0], ab);
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    if (mb + 1 < MB && mb + 1 < mbs) ldsm_x4(af[(mb + 1) & 1], ab + (mb + 1) * 16 * AROW);
                    if (mb < mbs) {
                        mma16816(acc[mb][0], af[mb & 1], b0);
                        mma16816(acc[mb][1], af[mb & 1], b1);
                    }
                    if (mb == MB / 2 - 1 && j + 1 < KSB / 2) decode_tile(wn, lane, n0, n1);
                }
                if (j + 1 < KSB / 2) {
                    b0[0] = n0[0]; b0[1] = n0[1]; b1[0] = n1[0]; b1[1] = n1[1];
                }
                if (ld_cur) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, cur, ld, j, pb);
                else if (ld_nxt) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, nxt, ld - nk, j, pb ^ 1);
            }
        } else {
            constexpr bool PIPE_DEC = (MS & 2) == 0, DBUF = (MS & 4) == 0;
            constexpr int LEAN = (MS & 8) != 0;
            uint32_t bf0[2], bf1[2];
            if (PIPE_DEC) decode_pair8<LEAN>(bw[0], bw[32], lane, bf0, bf1, cx);
#pragma unroll
            for (int j = 0; j < KSB / 2; ++j) {
                uint32_t w0n = 0u, w1n = 0u;
                if (!PIPE_DEC) decode_pair8<LEAN>(bw[64 * j], bw[64 * j + 32], lane, bf0, bf1, cx);
                else if (j + 1 < KSB / 2) { w0n = bw[(2 * j + 2) * 32]; w1n = bw[(2 * j + 3) * 32]; }
                const int ch = j * 2 + ahi;
                const unsigned char* ab = buf + aoff + aswz<KSB>(arow, ch) * 16;
                uint32_t af[2][4];
                uint32_t n0[2], n1[2];
                if (DBUF) ldsm_x4(af[0], ab);                        // mb 0 always holds rows (mbs >= 1)
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    if (DBUF) {
                        if (mb + 1 < MB && mb + 1 < mbs) ldsm_x4(af[(mb + 1) & 1], ab + (mb + 1) * 16 * AROW);
                    } else if (mb < mbs) {
                        ldsm_x4(af[0], ab + mb * 16 * AROW);
                    }
                    if (mb < mbs) {
                        mma16832(acc[mb][0], af[DBUF ? (mb & 1) : 0], bf0);
                        mma16832(acc[mb][1], af[DBUF ? (mb & 1) : 0], bf1);
                    }
                    if (PIPE_DEC && mb == MB / 2 - 1 && j + 1 < KSB / 2) decode_pair8<LEAN>(w0n, w1n, lane, n0, n1, cx);
                }
                if (PIPE_DEC && j + 1 < KSB / 2) {
                    bf0[0] = n0[0]; bf0[1] = n0[1]; bf1[0] = n1[0]; bf1[1] = n1[1];
                }
                if (ld_cur) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, cur, ld, j, pb);
                else if (ld_nxt) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, nxt, ld - nk, j, pb ^ 1);
            }
        }
        cp_commit();
    }
}

// RS (opt-moe2 EXPERIMENT, MS & 32, debug builds only - no gain over the lean decode alone): the shipped stage layout (A only in shared memory, 4 stages) but the trellis words are
// streamed per k32 STEP through a 2-slot register ring (prefetch distance 2 steps, ~half a stage) instead of 8 words
// for the stage being computed + 8 prefetched for the next one; the 10 freed registers hold the next step's decoded
// fragments, so step j + 1's decode is issued between step j's mma. The step stream runs across items like the
// shipped bc[] (the next item's first steps are loaded / decoded during this item's last ones when pf). Same
// fragments, same mma order per accumulator.
//   state at a step boundary s: bf = decoded fragments of step s, rw[(s + 1) & 1] = words of step s + 1,
//   rw[s & 1] = words of step s + 2.
template <int KSB, int DN16 = 0>
__device__ __forceinline__ void rs_load(uint32_t (&w)[2], const Job& j, int step) {
    // step: k32 step of job j (e4m3: tiles 2 step, 2 step + 1; fp16 A jobs: tile step); KSB / 2 steps per stage
    if (DN16 && j.f16) {
        w[0] = __ldg(j.wl + (size_t) step * j.wstride);
        w[1] = 0u;
    } else {
        const uint32_t* p = j.wl + (size_t) (2 * step) * j.wstride;
        w[0] = __ldg(p);
        w[1] = __ldg(p + j.wstride);
    }
}
template <int LEAN, int DN16 = 0>
__device__ __forceinline__ void rs_decode(const uint32_t (&w)[2], bool f16, int lane, uint32_t cx, uint32_t (&b)[4]) {
    uint32_t b0[2], b1[2];
    if (DN16 && f16) decode_tile(w[0], lane, b0, b1);
    else decode_pair8<LEAN>(w[0], w[1], lane, b0, b1, cx);
    b[0] = b0[0]; b[1] = b0[1]; b[2] = b1[0]; b[3] = b1[1];
}
// the next (or first) job's step-stream state: bf = step 0 decoded, rw[1] = step 1, rw[0] = step 2
template <int MB, int STAGES, int KSB, int LEAN, int DN16 = 0, int TG = 0, int R = 2>
__device__ __forceinline__ void fprologue_rs(const Job& j, int g, unsigned char* smem, uint32_t (&rw)[R][2],
                                             uint32_t (&bf)[4], uint32_t cx, int sb = 0) {
    using CF = Cfg<MB, 16, KSB>;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        fload_a<MB, 16, KSB, 1, TG>(smem + ((g + st) % STAGES) * CF::STAGE_BYTES, j, st, 0, sb);
        cp_commit();
    }
    uint32_t w0[2];
    rs_load<KSB, DN16>(w0, j, 0);
#pragma unroll
    for (int i = 1; i <= R; ++i) rs_load<KSB, DN16>(rw[i & (R - 1)], j, i);
    rs_decode<LEAN, DN16>(w0, DN16 && j.f16, threadIdx.x & 31, cx, bf);
}

template <int MB, int STAGES, int KSB, int LEAN, int DN16 = 0, int TG = 0, int R = 2>
__device__ __forceinline__ void fmainloop_rs(const Job& cur, const Job& nxt, bool pf, int& g, unsigned char* smem,
                                             uint32_t (&rw)[R][2], uint32_t (&bf)[4], uint32_t cx,
                                             float (&acc)[MB][2][4], int pb = 0) {
    using CF = Cfg<MB, 16, KSB>;
    constexpr int SPS = KSB / 2;                                 // k32 steps per stage
    const int lane = threadIdx.x & 31;
    const int nk = cur.K / CF::AROW;
    const int nsteps = nk * SPS;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mb][h][c] = 0.f;
    const int mbs = cur.mbs;
    const int arow = lane & 15, ahi = lane >> 4;
    const bool f16 = DN16 && cur.f16;
    const bool nf16 = DN16 && nxt.f16;
    for (int kt = 0; kt < nk; ++kt, ++g) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int ld = kt + STAGES - 1;
        unsigned char* lbuf = smem + ((g + STAGES - 1) % STAGES) * CF::STAGE_BYTES;
        const bool ld_cur = ld < nk, ld_nxt = !ld_cur && pf;
        const unsigned char* buf = smem + (g % STAGES) * CF::STAGE_BYTES;
#pragma unroll
        for (int j = 0; j < SPS; ++j) {
            const int st = kt * SPS + j;                         // this step; decode st + 1, load st + 3
            uint32_t nb[4];
            const unsigned char* ab = buf + arow * CF::AROW + aswz<KSB>(arow, j * 2 + ahi) * 16;
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (mb < mbs) {
                    uint32_t af[4];
                    ldsm_x4(af, ab + mb * 16 * CF::AROW);
                    const uint32_t b0[2] = {bf[0], bf[1]}, b1[2] = {bf[2], bf[3]};
                    if (f16) {
                        mma16816(acc[mb][0], af, b0);
                        mma16816(acc[mb][1], af, b1);
                    } else {
                        mma16832(acc[mb][0], af, b0);
                        mma16832(acc[mb][1], af, b1);
                    }
                }
                if (mb == MB / 2 - 1) {
                    // the next step's fragments (this job's step st + 1, or the next job's step 0)
                    const bool in_cur = st + 1 < nsteps;
                    const int sl = (j + 1) & (R - 1);         // R divides the steps per stage
                    rs_decode<LEAN, DN16>(rw[sl], in_cur ? f16 : nf16, lane, cx, nb);
                    // refill that slot with step st + 1 + R (cur, else the next job's step, else 0)
                    const int s3 = st + 1 + R;
                    if (s3 < nsteps) rs_load<KSB, DN16>(rw[sl], cur, s3);
                    else if (pf) rs_load<KSB, DN16>(rw[sl], nxt, s3 - nsteps);
                    else { rw[sl][0] = 0u; rw[sl][1] = 0u; }
                }
            }
            bf[0] = nb[0]; bf[1] = nb[1]; bf[2] = nb[2]; bf[3] = nb[3];
            if (ld_cur) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, cur, ld, j, pb);
            else if (ld_nxt) fload_a<MB, 16, KSB, KSB / 2, TG>(lbuf, nxt, ld - nk, j, pb ^ 1);
        }
        cp_commit();
    }
}

template <int MB, int STAGES, int KSB, int EB>
struct FSmem {
    static constexpr int CSTR = 16 * 16 + 8;
    static constexpr int PIPE = STAGES * Cfg<MB, 16, KSB>::STAGE_BYTES;
    static constexpr int TOTAL = PIPE + EB * 16 * CSTR * 4;
};

// EB: 16-row blocks per epilogue round (fewer barriers; EB * 16 rows of fp32 staging).
// DN16 = 1 (GLM53_MOE_E4M3_DOWN=f16): gate/up as above, but the gate/up epilogue applies the down projection's input
// transform itself (a16 = fp16(act); h = fp16(H(float(a16) * suh_d) * r), stored in place of a16: each 128-column
// gate/up item is exactly one Hadamard block) and the down jobs multiply those fp16 rows by the fp16 trellis decode
// on mma m16n8k16 (production's own down arithmetic: no actq, no e4m3 rounding of the down operands).
// OB (opt-moe): 0 = fp32 accumulator (out, red.v4.f32); 1 = bf16 accumulator (outb, red.v2.bf16x2: 4 values per op);
// 2 = bf16 accumulator, 8 values per op (the warp's two rows of a round exchange halves between lane pairs).
// DBG & 64 (experiment builds): thread 0 of every CTA accumulates clock64 per phase into 64-bit counters placed after
// the sync flags (A.sync + 2 + 2 * max_segs, 8-byte aligned): [0] mainloop of gate/up items, [1] mainloop of down
// items, [2] gate/up epilogue, [3] in-kernel actq (incl. its fences), [4] down epilogue, [5] waits for ready[seg] +
// re-prologue, [6] ticket / peek, [7] total, [8] items gate/up, [9] items down, [10] items that waited.
// MP (opt-moe, debug builds only - measured SLOWER: 31.55 vs 30.44 ms per 13,824-token call): 1 = the epilogue's per-row metadata (row token, router weight, row scale) and per-item vectors (svh /
// suh) are fetched into shared memory when the item starts (cp.async, joined to the mainloop's first copy group; the
// down's row scale dsc, written by another CTA of this launch, through L2 into a register), instead of global loads
// inside every 16-row epilogue round (latency-bound: the down epilogue was ~11 us per item). Same arithmetic.
// TKP (opt-moe, debug builds only - measured SLOWER: 30.93 vs 30.57 ms at 13,824): 1 = ticket prefetch: thread 0 claims the ticket of the item after next while the current item's
// mainloop runs (the atomicAdd's L2 round trip leaves the loop head's critical path). A CTA then holds up to three
// tickets (current < next < next-but-one) and processes them in order; every dependency points to a lower ticket, so
// the lowest outstanding ticket is always runnable (no deadlock). Same arithmetic; item-to-CTA assignment differs.
// TG (opt-moe): 1 = token-shared gate/up input: every expert of the layer has the same w13 suh (checked at load time),
// so the gathered e4m3 row of a (token, expert) pair is the same for all 8 experts of the token; gather_tok writes ONE
// row per token (a8[t], asc[t]) and the gate/up jobs read A row r of an item from a8[row_token[row0 + r]] (the
// epilogue's row scale from asc[row_token[...]]). The same bytes and the same arithmetic as the per-pair rows.
// MS (opt-moe2, docs/OPT_MOE2.md): 0 = the shipped mainloop. GLM53_MOE_E4M3_MAINLOOP=1 = 8 | 128: the shipped mainloop
// with the lean trellis decode (8) and the per-thread A copy address table (128; 16 KB of dynamic shared memory
// behind the staging). Debug-only experiments: 1 = smem-B mainloop (fmainloop_ms; + 2 decode not pipelined, 4 single
// A buffer, 16 epilogue metadata in smem), 32 = RS B-word ring (64 = ring of 4), 256 = compile-time B strides,
// 512 = full-tile specialization. Same arithmetic in every case.
template <int MB, int STAGES, int KSB, int EB, int DBG = 0, int DN16 = 0, int OB = 0, int MP = 0, int TKP = 0,
          int TG = 0, int MS = 0>
__global__ void __launch_bounds__(512, 1) me_fused_kernel(const FusedArgs A) {
    long long pf_t[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    long long pf_c = (DBG & 64) ? clock64() : 0, pf_0 = pf_c;
#define ME_PF(i) do { if constexpr ((DBG & 64) != 0) { __syncthreads(); const long long _n = clock64(); pf_t[i] += _n - pf_c; pf_c = _n; } } while (0)
    constexpr int NCB = 16;
    using SM = FSmem<MB, STAGES, KSB, EB>;
    constexpr int CSTR = SM::CSTR;
    extern __shared__ __align__(16) unsigned char smem[];
    static_assert(!(MP != 0 && TG != 0), "MP and TG are not combined");
    static_assert(MS == 0 || (MP == 0 && TKP == 0 && (DBG & ~64) == 0 && EB == 1), "MS: shipped options only");
    static_assert((MS & 32) == 0 || (MS & ~(32 | 64 | 8)) == 0, "RS: lean decode / ring size only");
    static_assert((MS & 1) != 0 || (MS & 32) != 0 || (MS & ~(8 | 128 | 256 | 512)) == 0, "shipped loop: lean / AT / BK / FULL");
    __shared__ int s_tk, s_flag, s_done;
    static_assert(TG == 0 || MB * 16 <= 128, "g_srt rows");
    int pb = 0;
    __shared__ long long s_tok[MP ? MB * 16 : 1];
    __shared__ float s_w[MP ? MB * 16 : 1], s_rs[MP ? MB * 16 : 1];
    __shared__ __align__(16) half s_sv[MP ? 384 : 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float* sc = reinterpret_cast<float*>(smem + SM::PIPE);      // MS: re-pointed after every mainloop
    int* mti = nullptr;                                         // MS & 16: per-row metadata in shared memory
    float* mtf = nullptr;
    constexpr int MS_META = ((16 * CSTR * 4 + 127) / 128) * 128;
    static_assert(MS == 0 || MS_META + 384 * 4 <= MsCfg<MB, KSB>::STAGE, "metadata fits the free stage buffer");
    int* ticket = A.sync;
    int* gu_cnt = A.sync + 1;
    int* ready = A.sync + 1 + A.max_segs;
    const int nsegs = *A.num_segs;
    auto wait_ready = [&](int seg) {
        if (tid == 0) {
            // bounded: a segment that never becomes ready (a bug) traps after ~30 s instead of hanging the GPU
            long long spins = 0;
            while (ld_acquire(ready + seg) == 0) {
                __nanosleep(256);
                if (++spins > (1LL << 27)) __trap();
            }
        }
        __syncthreads();
    };
    int tk_pre = 0;                                             // TKP: thread 0's prefetched ticket
    if (tid == 0) {
        s_tk = atomicAdd(ticket, 1);
        if constexpr (TKP != 0) tk_pre = atomicAdd(ticket, 1);
    }
    __syncthreads();
    Job cur = make_job<DN16, TG>(A, s_tk, nsegs);
    if (cur.kind < 0) return;
    if constexpr (TG != 0) {
        if (cur.kind == 0) {
            if (tid < MB * 16) g_srt[0][tid] = (int) A.row_token[cur.row0 + min(tid, cur.rows - 1)];
        }
        __syncthreads();
    }
    if (cur.kind == 1) wait_ready(cur.seg);
    uint32_t bc[KSB];
    int g = 0;
    constexpr bool RS = (MS & 32) != 0;                         // RS: step-streamed B ring, shipped stage layout
    constexpr int RS_LEAN = (MS & 8) != 0;
    constexpr int RS_R = (MS & 64) ? 4 : 2;                     // ring slots = prefetch distance in k32 steps
    uint32_t rw[RS_R][2], rbf[4];
    uint32_t rcx = 0x3B603B60u;
    if constexpr (RS_LEAN) asm volatile("mov.b32 %0, %1;" : "=r"(rcx) : "r"(rcx));
    constexpr int AT = (MS & 128) != 0;                         // A copy addresses from a per-thread table
    static_assert(AT == 0 || (MB == 8 && KSB == 8 && NCB == 16 && STAGES >= 2), "AT: 128 x 8 chunks, 512 threads");
    unsigned long long* at = reinterpret_cast<unsigned long long*>(smem + SM::TOTAL);
    const int so0 = (tid >> 3) * 128 + ((tid & 7) ^ ((tid >> 3) & 7)) * 16;   // aswz<8>(r, ch) * 16, r = tid / 8
    if constexpr (RS) fprologue_rs<MB, STAGES, KSB, RS_LEAN, DN16, TG, RS_R>(cur, g, smem, rw, rbf, rcx, 0);
    else if constexpr ((MS & 1) != 0) fprologue_ms<MB, STAGES, KSB, DN16, TG>(cur, g, smem, 0);
    else fprologue<MB, NCB, STAGES, KSB, DN16, TG, AT>(cur, g, smem, bc, 0, at, so0);
    if constexpr ((DBG & 64) != 0) { __syncthreads(); pf_t[5] += clock64() - pf_c; pf_c = clock64(); }
    while (true) {
        __syncthreads();                                        // everyone has read the previous s_tk / s_flag
        if (tid == 0) {
            const int tk = TKP != 0 ? tk_pre : atomicAdd(ticket, 1);
            s_tk = tk;
            const Job peek = make_job<DN16, TG>(A, tk, nsegs);
            s_flag = peek.kind == 0 || (peek.kind == 1 && ld_acquire(ready + peek.seg) != 0);
        }
        __syncthreads();
        const Job nxt = make_job<DN16, TG>(A, s_tk, nsegs);
        if constexpr (TG != 0) {
            // the previous item (buffer pb ^ 1) finished its epilogue before the loop-head barrier. cp.async (the low
            // 32 bits of the int64 token; no stall at the loop head): it joins the copy group cur's mainloop commits
            // in iteration 0, complete and visible from iteration 3's wait + barrier; nxt's first loads come at
            // iteration nk - STAGES + 1 >= 5 (pf) or after cur's mainloop (prologue)
            if (nxt.kind == 0) {
                if (tid < MB * 16)
                    cp4ca(&g_srt[pb ^ 1][tid], reinterpret_cast<const int*>(A.row_token + nxt.row0 + min(tid, nxt.rows - 1)));
            }
        }
        const bool pf = nxt.kind >= 0 && s_flag != 0;
        ME_PF(6);
        float dreg = 0.f;
        if constexpr (MP != 0) {
            // previous item's epilogue is done (loop-head barrier); these copies join the mainloop's first group
            const int rr = min(tid, cur.rows - 1);
            if (cur.kind == 1) {
                if (tid < MB * 16) {
                    cp8ca(&s_tok[tid], A.row_token + cur.row0 + rr);
                    cp4ca(&s_w[tid], A.row_weight + cur.row0 + rr);
                    if (!DN16) dreg = __ldcg(A.dsc + cur.row0 + rr);
                } else if (tid < MB * 16 + 32) {
                    const int c = tid - MB * 16;
                    cp16(s_sv + 8 * c, reinterpret_cast<const half*>(A.down_svh[cur.e]) + cur.nb * 256 + 8 * c);
                }
            } else {
                if (tid < MB * 16) {
                    cp4ca(&s_rs[tid], A.asc + cur.row0 + rr);
                } else if (tid < MB * 16 + 16 * (DN16 ? 3 : 2)) {
                    const int c = tid - MB * 16, v = c >> 4, i = c & 15;
                    const int64_t* tab = v == 0 ? A.gate_svh : (v == 1 ? A.up_svh : A.down_suh);
                    cp16(s_sv + 128 * v + 8 * i, reinterpret_cast<const half*>(tab[cur.e]) + cur.nb * 128 + 8 * i);
                }
            }
        }
        if constexpr (TKP != 0) {
            if (tid == 0 && nxt.kind >= 0) tk_pre = atomicAdd(ticket, 1);   // consumed at the next loop head
        }
        float acc[MB][2][4];
        if constexpr (RS) {
            fmainloop_rs<MB, STAGES, KSB, RS_LEAN, DN16, TG, RS_R>(cur, nxt, pf, g, smem, rw, rbf, rcx, acc, pb);
        } else if constexpr ((MS & 1) != 0) {
            fmainloop_ms<MB, STAGES, KSB, DN16, TG, MS>(cur, nxt, pf, g, smem, acc, pb);
            // staging = the stage buffer consumed last; other warps may still be reading it (ldmatrix) until here
            sc = reinterpret_cast<float*>(smem + ((g + STAGES - 1) % STAGES) * MsCfg<MB, KSB>::STAGE);
            __syncthreads();
            if constexpr ((MS & 16) != 0) {
                // MS & 16: the epilogue's per-row metadata fetched ONCE per item into the free stage buffer (behind the
                // staging), visible after the first round's barrier; the rounds read shared memory instead of
                // issuing a dependent global load each. The same values.
                mti = reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(sc) + MS_META);
                mtf = reinterpret_cast<float*>(mti);
                if (tid < MB * 16) {
                    const int lr = tid < cur.rows ? tid : 0, frow = cur.row0 + lr;
                    if (cur.kind == 1) {
                        mti[tid] = (int) A.row_token[frow];
                        mtf[128 + tid] = A.row_weight[frow];
                        mtf[256 + tid] = DN16 ? 1.0f : __ldcg(A.dsc + frow);
                    } else {
                        mtf[256 + tid] = A.asc[TG == 1 ? g_srt[pb][lr] : frow];
                    }
                }
            }
        } else {
            constexpr int BK = (MS & 256) != 0, FL = (MS & 512) != 0;
            if (FL && cur.mbs == MB)
                fmainloop<MB, NCB, STAGES, KSB, DN16, (DBG >> 8) & 31, TG, RS_LEAN, AT, BK, 1>(cur, nxt, pf, g, smem, bc, acc,
                                                                                      pb, rcx, at, so0);
            else
                fmainloop<MB, NCB, STAGES, KSB, DN16, (DBG >> 8) & 31, TG, RS_LEAN, AT, BK, 0>(cur, nxt, pf, g, smem, bc, acc,
                                                                                      pb, rcx, at, so0);
        }
        // (the mainloop's last iteration waited for every copy group but the newest two and passed a barrier: the
        // metadata group, committed with stage 0 of this item, is complete and visible; nk >= 8 here)
        if constexpr (MP != 0 && !DN16) {
            if (cur.kind == 1 && tid < MB * 16) s_rs[tid] = dreg;     // visible after the epilogue's first barrier
        }
        if constexpr ((DBG & 64) != 0) { ME_PF(cur.kind == 0 ? 0 : 1); pf_t[cur.kind == 0 ? 8 : 9] += 1; }
        const int rows = cur.rows, mbs = cur.mbs, e = cur.e;
        if (cur.kind == 0) {
            // ---- gate/up epilogue (me_gateup_kernel's)
            const int n0 = cur.nb * 128 + 4 * lane;
            const float4 svg = MP ? ld_h4f(s_sv + 4 * lane) : ld_h4f(reinterpret_cast<const half*>(A.gate_svh[e]) + n0);
            const float4 svu = MP ? ld_h4f(s_sv + 128 + 4 * lane) : ld_h4f(reinterpret_cast<const half*>(A.up_svh[e]) + n0);
            const float sg[4] = {svg.x, svg.y, svg.z, svg.w}, su[4] = {svu.x, svu.y, svu.z, svu.w};
            half* outrow = A.a16 + (size_t) cur.row0 * F_INT + n0;
            float sd[4] = {0.f, 0.f, 0.f, 0.f};
            if constexpr (DN16) {
                const float4 s4 = MP ? ld_h4f(s_sv + 256 + 4 * lane) : ld_h4f(reinterpret_cast<const half*>(A.down_suh[e]) + n0);
                sd[0] = s4.x; sd[1] = s4.y; sd[2] = s4.z; sd[3] = s4.w;
            }
#pragma unroll 1
            for (int mb0 = 0; mb0 < mbs; mb0 += EB) {
#pragma unroll
                for (int eb = 0; eb < EB; ++eb) stage_acc_rt<MB, CSTR>(sc + eb * 16 * CSTR, acc, mb0 + eb, warp, lane);
                __syncthreads();
#pragma unroll
              for (int rr = 0; rr < EB; ++rr) {
                const int lr = warp + rr * NCB;                 // row within the round
                const int tr = mb0 * 16 + lr;
                const int arow_i = cur.row0 + (tr < rows ? tr : 0);
                const float s = (MS & 16) ? mtf[256 + (tr < rows ? tr : 0)]
                                : (MP ? s_rs[tr < rows ? tr : 0] : A.asc[TG == 1 ? g_srt[pb][tr < rows ? tr : 0] : arow_i]);
                const float4 gq = *reinterpret_cast<const float4*>(sc + lr * CSTR + 4 * lane);
                const float4 uq = *reinterpret_cast<const float4*>(sc + lr * CSTR + 128 + 4 * lane);
                float gg[4] = {__fmul_rn(gq.x, s), __fmul_rn(gq.y, s), __fmul_rn(gq.z, s), __fmul_rn(gq.w, s)};
                float u[4] = {__fmul_rn(uq.x, s), __fmul_rn(uq.y, s), __fmul_rn(uq.z, s), __fmul_rn(uq.w, s)};
                fwht128(gg, lane);
                fwht128(u, lane);
                float act[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float gv = __fmul_rn(__fmul_rn(gg[i], HAD_SCALE), sg[i]);
                    float uv = __fmul_rn(__fmul_rn(u[i], HAD_SCALE), su[i]);
                    gv = fminf(gv, A.limit);
                    uv = fminf(fmaxf(uv, -A.limit), A.limit);
                    act[i] = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
                }
                half2 h01 = __floats2half2_rn(act[0], act[1]), h23 = __floats2half2_rn(act[2], act[3]);
                if constexpr (DN16) {
                    // the down input transform on this 128-column block: v = H(float(a16) * suh_d) * r -> fp16
                    float v[4] = {__low2float(h01) * sd[0], __high2float(h01) * sd[1], __low2float(h23) * sd[2],
                                  __high2float(h23) * sd[3]};
                    fwht128(v, lane);
                    h01 = __floats2half2_rn(__fmul_rn(v[0], HAD_SCALE), __fmul_rn(v[1], HAD_SCALE));
                    h23 = __floats2half2_rn(__fmul_rn(v[2], HAD_SCALE), __fmul_rn(v[3], HAD_SCALE));
                }
                uint2 ov;
                ov.x = *reinterpret_cast<const uint32_t*>(&h01);
                ov.y = *reinterpret_cast<const uint32_t*>(&h23);
                if (tr < rows) *reinterpret_cast<uint2*>(outrow + (size_t) tr * F_INT) = ov;
              }
                __syncthreads();
            }
            ME_PF(2);
            // ---- completion: the 8th gate/up item of the segment quantizes its down-input rows, then releases it
            __threadfence();
            __syncthreads();
            if (tid == 0) s_done = atomicAdd(gu_cnt + cur.seg, 1) == (F_INT / 128) - 1;
            __syncthreads();
            if (s_done) {
                __threadfence();
                const half* suh = reinterpret_cast<const half*>(A.down_suh[e]);
                for (int r = warp; r < (DN16 ? 0 : cur.rows); r += NCB) {
                    const int row = cur.row0 + r;
                    actq_row<F_INT, true>(A.a16 + (size_t) row * F_INT, suh, A.a8d + (size_t) row * F_INT,
                                          A.dsc + row, lane);
                }
                __threadfence();
                __syncthreads();
                if (tid == 0) st_release(ready + cur.seg, 1);
            }
            ME_PF(3);
        } else {
            // ---- down epilogue (me_down_kernel's), dsc through L2 (written by other CTAs of this launch)
            constexpr int CW = NCB * 16;
            const half* svh = MP ? s_sv + 4 * lane : reinterpret_cast<const half*>(A.down_svh[e]) + cur.nb * CW + 4 * lane;
            const float4 sv0 = ld_h4f(svh), sv1 = ld_h4f(svh + 128);
            float* outc = A.out + cur.nb * CW + 4 * lane;
            if constexpr (OB != 0) {
                // the same fp32 value o per element as below, then rounded to bf16 and added in bf16
#pragma unroll 1
                for (int mb0 = 0; mb0 < mbs; mb0 += EB) {
#pragma unroll
                    for (int eb = 0; eb < EB; ++eb) stage_acc_rt<MB, CSTR>(sc + eb * 16 * CSTR, acc, mb0 + eb, warp, lane);
                    __syncthreads();
#pragma unroll
                    for (int rp = 0; rp < EB; ++rp) {
                        uint32_t wv[2][2];
                        int64_t tok[2];
                        bool vld[2];
                        const int hf = warp & 1;            // pr = warp + rr * NCB, NCB even: hf is the warp's
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const int r = (warp >> 1) + (2 * rp + q) * (NCB / 2);
                            const int tr = mb0 * 16 + r;
                            vld[q] = tr < rows;
                            const int lrow = vld[q] ? tr : 0, frow = cur.row0 + lrow;
                            const float s = DN16 ? 1.0f : ((MS & 16) ? mtf[256 + lrow] : (MP ? s_rs[lrow] : __ldcg(A.dsc + frow)));
                            const float4 qv = *reinterpret_cast<const float4*>(sc + r * CSTR + hf * 128 + 4 * lane);
                            float v[4] = {__fmul_rn(qv.x, s), __fmul_rn(qv.y, s), __fmul_rn(qv.z, s), __fmul_rn(qv.w, s)};
                            fwht128(v, lane);
                            const float4 sv = hf ? sv1 : sv0;
                            const float w = (MS & 16) ? mtf[128 + lrow] : (MP ? s_w[lrow] : A.row_weight[frow]);
                            wv[q][0] = pack_bf2(__fmul_rn(__fmul_rn(__fmul_rn(v[0], HAD_SCALE), sv.x), w),
                                                __fmul_rn(__fmul_rn(__fmul_rn(v[1], HAD_SCALE), sv.y), w));
                            wv[q][1] = pack_bf2(__fmul_rn(__fmul_rn(__fmul_rn(v[2], HAD_SCALE), sv.z), w),
                                                __fmul_rn(__fmul_rn(__fmul_rn(v[3], HAD_SCALE), sv.w), w));
                            tok[q] = (MS & 16) ? (int64_t) mti[lrow] : (MP ? s_tok[lrow] : A.row_token[frow]);
                        }
                        __nv_bfloat16* base = A.outb + cur.nb * CW + hf * 128;
                        if constexpr (OB == 1) {
#pragma unroll
                            for (int q = 0; q < 2; ++q)
                                if (vld[q]) red_add_bf4(base + tok[q] * (int64_t) F_HID + 4 * lane, wv[q][0], wv[q][1]);
                        } else {
                            // even lane 2m: row 0, columns 8m..8m+7; odd lane 2m+1: row 1, the same columns
                            const int odd = lane & 1;
                            const uint32_t s0 = odd ? wv[0][0] : wv[1][0], s1 = odd ? wv[0][1] : wv[1][1];
                            const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 1);
                            const uint32_t r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
                            const int64_t tk = odd ? tok[1] : tok[0];
                            const bool vk = odd ? vld[1] : vld[0];
                            __nv_bfloat16* p = base + tk * (int64_t) F_HID + 8 * (lane >> 1);
                            if constexpr ((DBG & 8) != 0) {
                                if (vk && r0 == 0x12345678u && r1 == wv[1][1]) *p = __float2bfloat16(1.f);   // probe: no reds
                            } else if (vk) {
                                if (odd) red_add_bf8(p, r0, r1, wv[1][0], wv[1][1]);
                                else red_add_bf8(p, wv[0][0], wv[0][1], r0, r1);
                            }
                        }
                    }
                    __syncthreads();
                }
            } else {
#pragma unroll 1
            for (int mb0 = 0; mb0 < mbs; mb0 += EB) {
#pragma unroll
                for (int eb = 0; eb < EB; ++eb) stage_acc_rt<MB, CSTR>(sc + eb * 16 * CSTR, acc, mb0 + eb, warp, lane);
                __syncthreads();
#pragma unroll
                for (int rr = 0; rr < 2 * EB; ++rr) {
                    const int pr = warp + rr * NCB;
                    const int r = pr >> 1, hf = pr & 1;
                    const int tr = mb0 * 16 + r;
                    const bool valid = tr < rows;
                    const int lrow = valid ? tr : 0, frow = cur.row0 + lrow;
                    const float s = DN16 ? 1.0f : ((MS & 16) ? mtf[256 + lrow] : (MP ? s_rs[lrow] : __ldcg(A.dsc + frow)));
                    const float4 q = *reinterpret_cast<const float4*>(sc + r * CSTR + hf * 128 + 4 * lane);
                    float v[4] = {__fmul_rn(q.x, s), __fmul_rn(q.y, s), __fmul_rn(q.z, s), __fmul_rn(q.w, s)};
                    fwht128(v, lane);
                    const float4 sv = hf ? sv1 : sv0;
                    const float w = (MS & 16) ? mtf[128 + lrow] : (MP ? s_w[lrow] : A.row_weight[frow]);
                    float4 o;
                    o.x = __fmul_rn(__fmul_rn(__fmul_rn(v[0], HAD_SCALE), sv.x), w);
                    o.y = __fmul_rn(__fmul_rn(__fmul_rn(v[1], HAD_SCALE), sv.y), w);
                    o.z = __fmul_rn(__fmul_rn(__fmul_rn(v[2], HAD_SCALE), sv.z), w);
                    o.w = __fmul_rn(__fmul_rn(__fmul_rn(v[3], HAD_SCALE), sv.w), w);
                    if constexpr ((DBG & 32) != 0) {
                        if (valid) atomicAdd(reinterpret_cast<float4*>(outc + (A.row_token[frow] & 511) * (int64_t) F_HID + hf * 128), o);
                    } else if constexpr ((DBG & 16) != 0) {
                        if (valid) *reinterpret_cast<float4*>(outc + A.row_token[frow] * (int64_t) F_HID + hf * 128) = o;
                    } else if constexpr ((DBG & 8) != 0) {
                        if (o.x == 1234.5f) outc[0] = o.y;
                    } else {
                        if (valid) red_add4(outc + ((MS & 16) ? (int64_t) mti[lrow] : (MP ? s_tok[lrow] : A.row_token[frow])) * (int64_t) F_HID + hf * 128, o);
                    }
                }
                __syncthreads();
            }
            }   // OB == 0
            ME_PF(4);
        }
        if (nxt.kind < 0) break;
        if (!pf) {
            if (nxt.kind == 1) wait_ready(nxt.seg);
            if constexpr (RS) fprologue_rs<MB, STAGES, KSB, RS_LEAN, DN16, TG, RS_R>(nxt, g, smem, rw, rbf, rcx, pb ^ 1);
            else if constexpr ((MS & 1) != 0) fprologue_ms<MB, STAGES, KSB, DN16, TG>(nxt, g, smem, pb ^ 1);
            else fprologue<MB, NCB, STAGES, KSB, DN16, TG, AT>(nxt, g, smem, bc, pb ^ 1, at, so0);
            if constexpr ((DBG & 64) != 0) pf_t[10] += 1;
            ME_PF(5);
        }
        cur = nxt;
        pb ^= 1;
    }
    cp_wait<0>();
    if constexpr ((DBG & 64) != 0) {
        if (tid == 0) {
            pf_t[7] = clock64() - pf_0;
            unsigned long long* pc = reinterpret_cast<unsigned long long*>(
                reinterpret_cast<uintptr_t>(A.sync + 2 + 2 * A.max_segs + 1) & ~(uintptr_t) 7);
            for (int i = 0; i < 11; ++i) atomicAdd(pc + i, (unsigned long long) pf_t[i]);
        }
    }
#undef ME_PF
}

// ---------------------------------------------------------------------------------------------------------
// P16 gate/up, exllamav3's fm_gateup_kernel ported (docs/ref/mia_exl3-fat-kernel/exl3_fat_moe.cu): 64-row tiles,
// 8 warps, 2 CTAs per SM, warp w = 16 columns of gate AND up (two B streams), K advanced 32 per cp.async stage, A and
// B staged through shared memory, B decoded from shared memory by the owning warp once per k16 and reused across the
// 4 row blocks. Same fragments, same k order, same epilogue statements -> bit-identical h2; the only change is the
// SiLU's exponential: exp_prod (== (float) exp((double) .) bit for bit) instead of the FP64 statement (DBG 512 = the
// literal statement, for A/B).
constexpr int PG_THREADS = 256, PG_WARPS = 8, PG_TK = 32, PG_STAGES = 4, PG_MB = 4;
constexpr int PG_A_STAGE = PG_MB * 16 * PG_TK * 2;               // bytes
constexpr int PG_B_STAGE = 2 * 2 * PG_WARPS * 128;               // bytes: NS x 2 k16 substeps x warps x 128 B
constexpr int PG_PIPE = PG_STAGES * (PG_A_STAGE + PG_B_STAGE);
constexpr int PG_EPI = 16 * 2 * 128 * 4;
constexpr int PG_SMEM = PG_PIPE > PG_EPI ? PG_PIPE : PG_EPI;

__device__ __forceinline__ int pg_swz(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <int NS, int NBS, int DBG = 0>
__device__ __forceinline__ void pg_mainloop(const half* __restrict__ a, int K, int row0, int rows,
                                            const uint32_t* const (&w)[NS], int tiles_n, int nb0,
                                            unsigned char* smem, float (&acc)[NS][PG_MB][2][4]) {
    constexpr int A_CHUNKS = PG_MB * 16 * 4;                 // 16 B chunks per stage
    constexpr int B_CHUNKS = NS * 2 * PG_WARPS * 8;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int k_tiles = K / PG_TK;
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int mb = 0; mb < PG_MB; ++mb)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[s][mb][h][c] = 0.f;
    auto load_stage = [&](int stage, int kt) {
        unsigned char* sa = smem + stage * (PG_A_STAGE + PG_B_STAGE);
        unsigned char* sb = sa + PG_A_STAGE;
#pragma unroll
        for (int i = 0; i < (A_CHUNKS + PG_THREADS - 1) / PG_THREADS; ++i) {
            const int c = i * PG_THREADS + t;
            if (c < A_CHUNKS) {
                const int row = c >> 2, chunk = c & 3;
                const int sr = row < rows ? row : rows - 1;
                cp16(sa + row * (PG_TK * 2) + pg_swz(row, chunk) * 16,
                     a + (size_t) (row0 + sr) * K + kt * PG_TK + chunk * 8);
            }
        }
#pragma unroll
        for (int i = 0; i < (B_CHUNKS + PG_THREADS - 1) / PG_THREADS; ++i) {
            const int c = i * PG_THREADS + t;
            if (c < B_CHUNKS) {
                const int s = c / (2 * PG_WARPS * 8), r = c % (2 * PG_WARPS * 8);
                const int j = r / (PG_WARPS * 8), nb = (r / 8) % PG_WARPS, q = r % 8;
                const uint32_t* ws = w[0];
                if constexpr (NS == 2) ws = s ? w[1] : w[0];
                const uint32_t* src = ws + ((size_t) (kt * 2 + j) * tiles_n + nb0 + s * NBS + nb) * 32 + q * 4;
                cp16(sb + ((s * 2 + j) * PG_WARPS + nb) * 128 + q * 16, src);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < PG_STAGES - 1; ++s) {
        if (s < k_tiles) load_stage(s, s);
        cp_commit();
    }
    const int arow = lane & 15, ahi = lane >> 4;
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<PG_STAGES - 2>();
        __syncthreads();
        const int nk = kt + PG_STAGES - 1;
        if (nk < k_tiles) load_stage(nk % PG_STAGES, nk);
        cp_commit();
        const unsigned char* sa = smem + (kt % PG_STAGES) * (PG_A_STAGE + PG_B_STAGE);
        const uint32_t* sb = reinterpret_cast<const uint32_t*>(sa + PG_A_STAGE);
#pragma unroll
        if constexpr ((DBG & 2048) != 0) {
            // experiment: decode both k16 sub-steps of the stage first, then the ldmatrix / mma
            uint32_t fb[2][NS][2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int s = 0; s < NS; ++s)
                    decode_tile(sb[((s * 2 + j) * PG_WARPS + warp) * 32 + lane], lane, fb[j][s][0], fb[j][s][1]);
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int mb = 0; mb < PG_MB; ++mb) {
                    const int row = mb * 16 + arow, chunk = j * 2 + ahi;
                    uint32_t fa[4];
                    ldsm_x4(fa, sa + row * (PG_TK * 2) + pg_swz(row, chunk) * 16);
#pragma unroll
                    for (int s = 0; s < NS; ++s) {
                        mma16816(acc[s][mb][0], fa, fb[j][s][0]);
                        mma16816(acc[s][mb][1], fa, fb[j][s][1]);
                    }
                }
            continue;
        }
        for (int j = 0; j < 2; ++j) {
            uint32_t fb[NS][2][2];
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                if constexpr ((DBG & 4) != 0) {
                    const uint32_t w = sb[((s * 2 + j) * PG_WARPS + warp) * 32 + lane];
                    fb[s][0][0] = w; fb[s][0][1] = w ^ 0x3c003c00u; fb[s][1][0] = w + 1u; fb[s][1][1] = w ^ 0x00ff00ffu;
                } else {
                    decode_tile(sb[((s * 2 + j) * PG_WARPS + warp) * 32 + lane], lane, fb[s][0], fb[s][1]);
                }
            }
#pragma unroll
            for (int mb = 0; mb < PG_MB; ++mb) {
                const int row = mb * 16 + arow, chunk = j * 2 + ahi;
                uint32_t fa[4];
                ldsm_x4(fa, sa + row * (PG_TK * 2) + pg_swz(row, chunk) * 16);
#pragma unroll
                for (int s = 0; s < NS; ++s) {
                    mma16816(acc[s][mb][0], fa, fb[s][0]);
                    mma16816(acc[s][mb][1], fa, fb[s][1]);
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
}

// fm_stage_acc: one 16-row block of both streams into sc[16][NS * 128] (fp32)
template <int NS>
__device__ __forceinline__ void pg_stage(float* sc, const float (&acc)[NS][PG_MB][2][4], int mb, int warp, int lane) {
    constexpr int W = NS * 128;
    const int r0 = lane >> 2, col = (lane & 3) * 2 + warp * 16;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
        float* d0 = sc + r0 * W + s * 128 + col;
        float* d1 = sc + (r0 + 8) * W + s * 128 + col;
        float a0[4], a1[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            // runtime mb: select (the loop over mb in the caller is not unrolled)
            a0[c] = mb == 0 ? acc[s][0][0][c] : mb == 1 ? acc[s][1][0][c] : mb == 2 ? acc[s][2][0][c] : acc[s][3][0][c];
            a1[c] = mb == 0 ? acc[s][0][1][c] : mb == 1 ? acc[s][1][1][c] : mb == 2 ? acc[s][2][1][c] : acc[s][3][1][c];
        }
        d0[0] = a0[0]; d0[1] = a0[1]; d0[8] = a1[0]; d0[9] = a1[1];
        d1[0] = a0[2]; d1[1] = a0[3]; d1[8] = a1[2]; d1[9] = a1[3];
    }
}

template <int DBG = 0>
__global__ void __launch_bounds__(PG_THREADS, 2) me_pgu_kernel(
    const half* __restrict__ h13, const int64_t* __restrict__ gate_ptrs, const int64_t* __restrict__ up_ptrs,
    const int64_t* __restrict__ gate_svh, const int64_t* __restrict__ up_svh, const int64_t* __restrict__ down_suh,
    half* __restrict__ h2, const int* __restrict__ seg_expert, const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows, const int* __restrict__ num_segs_ptr, float limit, int chunk, int nchunks) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sc = reinterpret_cast<float*>(smem);
    const int ns_all = *num_segs_ptr;
    const int s0 = (int) ((long long) ns_all * chunk / nchunks), num_segs = (int) ((long long) ns_all * (chunk + 1) / nchunks);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles_n = F_INT / 16;
    const int n_base = blockIdx.x * 128;
    for (int seg = s0 + blockIdx.y; seg < num_segs; seg += gridDim.y) {
        const int e = seg_expert[seg], row0 = seg_row0[seg], rows = seg_rows[seg];
        const uint32_t* const w[2] = {reinterpret_cast<const uint32_t*>(gate_ptrs[e]),
                                      reinterpret_cast<const uint32_t*>(up_ptrs[e])};
        const half* svg = reinterpret_cast<const half*>(gate_svh[e]) + n_base + lane * 4;
        const half* svu = reinterpret_cast<const half*>(up_svh[e]) + n_base + lane * 4;
        const half* sdp = reinterpret_cast<const half*>(down_suh[e]) + n_base + lane * 4;
        float acc[2][PG_MB][2][4];
        pg_mainloop<2, 0, DBG>(h13, F_HID, row0, rows, w, tiles_n, n_base / 16, smem, acc);
#pragma unroll 1
        for (int mb = 0; mb < PG_MB; ++mb) {
            const int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            pg_stage<2>(sc, acc, mb, warp, lane);
            __syncthreads();
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int r = warp + rr * PG_WARPS;
                if (r < rows_mb) {
                    const float4 gq = *reinterpret_cast<const float4*>(sc + r * 256 + lane * 4);
                    const float4 uq = *reinterpret_cast<const float4*>(sc + r * 256 + 128 + lane * 4);
                    float g[4] = {gq.x, gq.y, gq.z, gq.w}, u[4] = {uq.x, uq.y, uq.z, uq.w};
                    fwht128(g, lane);
                    fwht128(u, lane);
                    const float4 sg = ld_h4f(svg), su = ld_h4f(svu);
                    const float sgv[4] = {sg.x, sg.y, sg.z, sg.w}, suv[4] = {su.x, su.y, su.z, su.w};
                    float act[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float gv = __fmul_rn(__fmul_rn(g[i], HAD_SCALE), sgv[i]);
                        float uv = __fmul_rn(__fmul_rn(u[i], HAD_SCALE), suv[i]);
                        gv = fminf(gv, limit);
                        uv = fminf(fmaxf(uv, -limit), limit);
                        float ex;
                        if constexpr ((DBG & 512) != 0) ex = (float) exp(-(double) gv);
                        else if constexpr ((DBG & 64) != 0) ex = __expf(-gv);
                        else ex = exp_prod(-gv);
                        act[i] = __fmul_rn(__fmul_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, ex)), gv), uv);
                    }
                    const uint2 r2 = *reinterpret_cast<const uint2*>(sdp);
                    half2 h01 = __hmul2(__floats2half2_rn(act[0], act[1]), *reinterpret_cast<const half2*>(&r2.x));
                    half2 h23 = __hmul2(__floats2half2_rn(act[2], act[3]), *reinterpret_cast<const half2*>(&r2.y));
                    float v[4] = {__low2float(h01), __high2float(h01), __low2float(h23), __high2float(h23)};
                    fwht128(v, lane);
                    h01 = __floats2half2_rn(__fmul_rn(v[0], HAD_SCALE), __fmul_rn(v[1], HAD_SCALE));
                    h23 = __floats2half2_rn(__fmul_rn(v[2], HAD_SCALE), __fmul_rn(v[3], HAD_SCALE));
                    uint2 ov;
                    ov.x = *reinterpret_cast<const uint32_t*>(&h01);
                    ov.y = *reinterpret_cast<const uint32_t*>(&h23);
                    *reinterpret_cast<uint2*>(h2 + (size_t) (row0 + mb * 16 + r) * F_INT + n_base + lane * 4) = ov;
                }
            }
            __syncthreads();
        }
    }
}

// P16 down, exllamav3's fm_down_kernel ported (64-row tiles, 8 warps, 2 CTAs/SM, two adjacent 128-column halves per
// CTA, weighted fp32 red.v4 into out), over the segment chunk [ns * chunk / nchunks, ns * (chunk + 1) / nchunks).
template <int DBG = 0>
__global__ void __launch_bounds__(PG_THREADS, 2) me_pdn_kernel(
    const half* __restrict__ h2, const int64_t* __restrict__ down_ptrs, const int64_t* __restrict__ down_svh,
    float* __restrict__ out, const int64_t* __restrict__ row_token, const float* __restrict__ row_weight,
    const int* __restrict__ seg_expert, const int* __restrict__ seg_row0, const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr, int chunk, int nchunks) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sc = reinterpret_cast<float*>(smem);
    const int ns_all = *num_segs_ptr;
    const int s0 = (int) ((long long) ns_all * chunk / nchunks), num_segs = (int) ((long long) ns_all * (chunk + 1) / nchunks);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_base = blockIdx.x * 256;
    for (int seg = s0 + blockIdx.y; seg < num_segs; seg += gridDim.y) {
        const int e = seg_expert[seg], row0 = seg_row0[seg], rows = seg_rows[seg];
        const uint32_t* const w[2] = {reinterpret_cast<const uint32_t*>(down_ptrs[e]),
                                      reinterpret_cast<const uint32_t*>(down_ptrs[e])};
        float acc[2][PG_MB][2][4];
        pg_mainloop<2, 8, DBG>(h2, F_INT, row0, rows, w, F_HID / 16, n_base / 16, smem, acc);
        const half* svh = reinterpret_cast<const half*>(down_svh[e]) + n_base + lane * 4;
#pragma unroll 1
        for (int mb = 0; mb < PG_MB; ++mb) {
            const int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            pg_stage<2>(sc, acc, mb, warp, lane);
            __syncthreads();
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int r = warp + rr * PG_WARPS;
                if (r < rows_mb) {
                    const int frow = row0 + mb * 16 + r;
                    const float wgt = row_weight[frow];
                    float* dst = out + row_token[frow] * (int64_t) F_HID + n_base + lane * 4;
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const float4 q = *reinterpret_cast<const float4*>(sc + r * 256 + s * 128 + lane * 4);
                        float v[4] = {q.x, q.y, q.z, q.w};
                        fwht128(v, lane);
                        const float4 sv = ld_h4f(svh + s * 128);
                        float4 o;
                        o.x = __fmul_rn(__fmul_rn(__fmul_rn(v[0], HAD_SCALE), sv.x), wgt);
                        o.y = __fmul_rn(__fmul_rn(__fmul_rn(v[1], HAD_SCALE), sv.y), wgt);
                        o.z = __fmul_rn(__fmul_rn(__fmul_rn(v[2], HAD_SCALE), sv.z), wgt);
                        o.w = __fmul_rn(__fmul_rn(__fmul_rn(v[3], HAD_SCALE), sv.w), wgt);
                        if constexpr ((DBG & 8) != 0) { if (o.x == 1234.5f) dst[0] = o.y; }
                        else red_add4(dst + s * 128, o);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// P16 v3 (GLM53_MOE_FUSED16 default): ONE persistent launch, 2 CTAs x 8 warps per SM (exllamav3's CTA shape, so one
// CTA's epilogue overlaps the other's mainloop), 64-row segments (production's build_grouped_fat_tables, tile 64), three
// item kinds handed out by a global ticket counter:
//   G (seg)      gather: h13 rows of the segment = fp16(H(fp16(x * suh)) * r)               (fm_gather_kernel)
//   U (seg, nb)  gate/up of 128 intermediate columns (fm_gateup_kernel's CTA, exp_prod)   waits gathered[seg]
//   D (seg, nb)  down of 256 hidden columns + weighted fp32 red.v4 into out (fm_down's)  waits ready[seg] (8 U done)
// Ticket order: rounds r = 0 .. nsegs + LG + LD - 1 of [G(r), U(r - LG, 0..7), D(r - LG - LD, 0..15)] (absent parts
// skipped), so the DRAM-bound gather and scatter overlap the compute-bound gate/up on every SM. An item only waits
// for items with smaller tickets and every CTA is resident (grid = SMs x occupancy): deadlock-free.
// Per element the arithmetic is fm_gather / fm_gateup / fm_down's (same fragments, same k order, same epilogues).

struct P3Args {
    const half* x;              // xh [T, 4096]
    half* h13;
    half* h2;
    float* out;
    const int64_t* gate_ptrs; const int64_t* up_ptrs; const int64_t* gate_svh; const int64_t* up_svh;
    const int64_t* gate_suh; const int64_t* down_ptrs; const int64_t* down_suh; const int64_t* down_svh;
    const int64_t* row_token;
    const int* row_expert;
    const float* row_weight;
    const int* seg_expert; const int* seg_row0; const int* seg_rows; const int* num_segs;
    int* sync;                  // [0] ticket, [1..ms] gathered, [1+ms..2ms] U done count, [1+2ms..3ms] ready
    int max_segs, lg, ld;
    float limit;
};

struct P3Job { int kind, seg, nb; };

__device__ __forceinline__ P3Job p3_job(int tk, int nsegs, int LG, int LD) {
    // phase sizes: A r<LG: 1 | B LG<=r<LG+LD: 9 | C LG+LD<=r<nsegs: 25 | D nsegs<=r<nsegs+LG: 24 | E ..+LD: 16
    // (the host clamps LG + LD <= nsegs)
    P3Job j{-1, 0, 0};
    int i = tk;
    if (i < LG) { j.kind = 2; j.seg = i; return j; }
    i -= LG;
    if (i < 9 * LD) { const int r = LG + i / 9, w = i % 9;
        if (w == 0) { j.kind = 2; j.seg = r; } else { j.kind = 0; j.seg = r - LG; j.nb = w - 1; } return j; }
    i -= 9 * LD;
    const int nc = nsegs - LG - LD;
    if (i < 25 * nc) { const int r = LG + LD + i / 25, w = i % 25;
        if (w == 0) { j.kind = 2; j.seg = r; } else if (w < 9) { j.kind = 0; j.seg = r - LG; j.nb = w - 1; }
        else { j.kind = 1; j.seg = r - LG - LD; j.nb = w - 9; } return j; }
    i -= 25 * nc;
    if (i < 24 * LG) { const int r = nsegs + i / 24, w = i % 24;
        if (w < 8) { j.kind = 0; j.seg = r - LG; j.nb = w; } else { j.kind = 1; j.seg = r - LG - LD; j.nb = w - 8; }
        return j; }
    i -= 24 * LG;
    if (i < 16 * LD) { const int r = nsegs + LG + i / 16; j.kind = 1; j.seg = r - LG - LD; j.nb = i % 16; return j; }
    return j;
}

__device__ __forceinline__ void p3_wait(const int* flag, int need) {
    if (threadIdx.x == 0) {
        long long spins = 0;
        while (ld_acquire(flag) < need) {
            __nanosleep(128);
            if (++spins > (1LL << 28)) __trap();
        }
    }
    __syncthreads();
}

template <int DBG = 0>
__global__ void __launch_bounds__(PG_THREADS, 2) me_p16b_kernel(const P3Args A) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int s_tk, s_done;
    float* sc = reinterpret_cast<float*>(smem);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nsegs = *A.num_segs;
    const int ms = A.max_segs;
    int* ticket = A.sync;
    int* gathered = A.sync + 1;
    int* ucnt = A.sync + 1 + ms;
    int* ready = A.sync + 1 + 2 * ms;
    if (nsegs <= 0) return;
    const int LG = min(A.lg, max(nsegs / 2, 1));               // LG >= 1, LD >= 0, LG + LD <= nsegs
    const int LD = min(A.ld, nsegs - LG);
    while (true) {
        if (tid == 0) s_tk = atomicAdd(ticket, 1);
        __syncthreads();
        const P3Job jb = p3_job(s_tk, nsegs, LG, LD);
        __syncthreads();                                         // s_tk consumed
        if (jb.kind < 0) break;
        const int seg = jb.seg;
        const int e = A.seg_expert[seg], row0 = A.seg_row0[seg], rows = A.seg_rows[seg];
        if constexpr ((DBG & 128) != 0) { if (jb.kind == 1) continue; }          // experiment: no down items
        if constexpr ((DBG & 1024) != 0) {         // h13 gathered beforehand (gather16): G items only release the flag
            if (jb.kind == 2) { if (tid == 0) st_release(gathered + seg, 1); continue; }
        }
        if constexpr ((DBG & 256) != 0) {                                        // experiment: down items only
            if (jb.kind != 1) {
                if (tid == 0) {
                    if (jb.kind == 2) st_release(gathered + seg, 1);
                    else if (atomicAdd(ucnt + seg, 1) == 7) st_release(ready + seg, 1);
                }
                continue;
            }
        }
        if (jb.kind == 2) {
            // ---- G: fm_gather_kernel's statement for the segment's rows (8 rows per warp)
            const half* su = reinterpret_cast<const half*>(A.gate_suh[e]) + 4 * lane;
            for (int r = warp; r < rows; r += PG_WARPS) {
                const half* xr = A.x + (size_t) A.row_token[row0 + r] * F_HID + 4 * lane;
                half* dst = A.h13 + (size_t) (row0 + r) * F_HID + 4 * lane;
#pragma unroll 4
                for (int b = 0; b < F_HID / 128; ++b) {
                    const uint2 xv = *reinterpret_cast<const uint2*>(xr + b * 128);
                    const uint2 sv = *reinterpret_cast<const uint2*>(su + b * 128);
                    const half2 p0 = __hmul2(*reinterpret_cast<const half2*>(&xv.x), *reinterpret_cast<const half2*>(&sv.x));
                    const half2 p1 = __hmul2(*reinterpret_cast<const half2*>(&xv.y), *reinterpret_cast<const half2*>(&sv.y));
                    float v[4] = {__low2float(p0), __high2float(p0), __low2float(p1), __high2float(p1)};
                    fwht128(v, lane);
                    const half2 o0 = __floats2half2_rn(__fmul_rn(v[0], HAD_SCALE), __fmul_rn(v[1], HAD_SCALE));
                    const half2 o1 = __floats2half2_rn(__fmul_rn(v[2], HAD_SCALE), __fmul_rn(v[3], HAD_SCALE));
                    uint2 ov;
                    ov.x = *reinterpret_cast<const uint32_t*>(&o0);
                    ov.y = *reinterpret_cast<const uint32_t*>(&o1);
                    *reinterpret_cast<uint2*>(dst + b * 128) = ov;
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) st_release(gathered + seg, 1);
            continue;
        }
        if (rows <= 0) {                                         // (cannot happen for seg < num_segs; keep flags live)
            if (jb.kind == 0) {
                if (tid == 0 && atomicAdd(ucnt + seg, 1) == 7) st_release(ready + seg, 1);
            }
            continue;
        }
        if (jb.kind == 0) {
            // ---- U: fm_gateup_kernel's CTA
            p3_wait(gathered + seg, 1);
            const int n_base = jb.nb * 128;
            const uint32_t* const w[2] = {reinterpret_cast<const uint32_t*>(A.gate_ptrs[e]),
                                          reinterpret_cast<const uint32_t*>(A.up_ptrs[e])};
            float acc[2][PG_MB][2][4];
            pg_mainloop<2, 0>(A.h13, F_HID, row0, rows, w, F_INT / 16, n_base / 16, smem, acc);
            const half* svg = reinterpret_cast<const half*>(A.gate_svh[e]) + n_base + lane * 4;
            const half* svu = reinterpret_cast<const half*>(A.up_svh[e]) + n_base + lane * 4;
            const half* sdp = reinterpret_cast<const half*>(A.down_suh[e]) + n_base + lane * 4;
#pragma unroll 1
            for (int mb = 0; mb < PG_MB; ++mb) {
                const int rows_mb = rows - mb * 16;
                if (rows_mb <= 0) break;
                pg_stage<2>(sc, acc, mb, warp, lane);
                __syncthreads();
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int r = warp + rr * PG_WARPS;
                    if (r < rows_mb) {
                        const float4 gq = *reinterpret_cast<const float4*>(sc + r * 256 + lane * 4);
                        const float4 uq = *reinterpret_cast<const float4*>(sc + r * 256 + 128 + lane * 4);
                        float g[4] = {gq.x, gq.y, gq.z, gq.w}, u[4] = {uq.x, uq.y, uq.z, uq.w};
                        fwht128(g, lane);
                        fwht128(u, lane);
                        const float4 sg = ld_h4f(svg), su = ld_h4f(svu);
                        const float sgv[4] = {sg.x, sg.y, sg.z, sg.w}, suv[4] = {su.x, su.y, su.z, su.w};
                        float act[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            float gv = __fmul_rn(__fmul_rn(g[i], HAD_SCALE), sgv[i]);
                            float uv = __fmul_rn(__fmul_rn(u[i], HAD_SCALE), suv[i]);
                            gv = fminf(gv, A.limit);
                            uv = fminf(fmaxf(uv, -A.limit), A.limit);
                            float ex;
                            if constexpr ((DBG & 512) != 0) ex = (float) exp(-(double) gv);
                            else ex = exp_prod(-gv);
                            act[i] = __fmul_rn(__fmul_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, ex)), gv), uv);
                        }
                        const uint2 r2 = *reinterpret_cast<const uint2*>(sdp);
                        half2 h01 = __hmul2(__floats2half2_rn(act[0], act[1]), *reinterpret_cast<const half2*>(&r2.x));
                        half2 h23 = __hmul2(__floats2half2_rn(act[2], act[3]), *reinterpret_cast<const half2*>(&r2.y));
                        float v[4] = {__low2float(h01), __high2float(h01), __low2float(h23), __high2float(h23)};
                        fwht128(v, lane);
                        h01 = __floats2half2_rn(__fmul_rn(v[0], HAD_SCALE), __fmul_rn(v[1], HAD_SCALE));
                        h23 = __floats2half2_rn(__fmul_rn(v[2], HAD_SCALE), __fmul_rn(v[3], HAD_SCALE));
                        uint2 ov;
                        ov.x = *reinterpret_cast<const uint32_t*>(&h01);
                        ov.y = *reinterpret_cast<const uint32_t*>(&h23);
                        *reinterpret_cast<uint2*>(A.h2 + (size_t) (row0 + mb * 16 + r) * F_INT + n_base + lane * 4) = ov;
                    }
                }
                __syncthreads();
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) s_done = atomicAdd(ucnt + seg, 1) == (F_INT / 128) - 1;
            __syncthreads();
            if (s_done) {
                __threadfence();
                if (tid == 0) st_release(ready + seg, 1);
            }
        } else {
            // ---- D: fm_down_kernel's CTA (two adjacent 128-column halves)
            p3_wait(ready + seg, 1);
            const int n_base = jb.nb * 256;
            const uint32_t* const w[2] = {reinterpret_cast<const uint32_t*>(A.down_ptrs[e]),
                                          reinterpret_cast<const uint32_t*>(A.down_ptrs[e])};
            float acc[2][PG_MB][2][4];
            pg_mainloop<2, 8>(A.h2, F_INT, row0, rows, w, F_HID / 16, n_base / 16, smem, acc);
            const half* svh = reinterpret_cast<const half*>(A.down_svh[e]) + n_base + lane * 4;
#pragma unroll 1
            for (int mb = 0; mb < PG_MB; ++mb) {
                const int rows_mb = rows - mb * 16;
                if (rows_mb <= 0) break;
                pg_stage<2>(sc, acc, mb, warp, lane);
                __syncthreads();
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int r = warp + rr * PG_WARPS;
                    if (r < rows_mb) {
                        const int frow = row0 + mb * 16 + r;
                        const float wgt = A.row_weight[frow];
                        float* dst = A.out + A.row_token[frow] * (int64_t) F_HID + n_base + lane * 4;
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const float4 q = *reinterpret_cast<const float4*>(sc + r * 256 + s * 128 + lane * 4);
                            float v[4] = {q.x, q.y, q.z, q.w};
                            fwht128(v, lane);
                            const float4 sv = ld_h4f(svh + s * 128);
                            float4 o;
                            o.x = __fmul_rn(__fmul_rn(__fmul_rn(v[0], HAD_SCALE), sv.x), wgt);
                            o.y = __fmul_rn(__fmul_rn(__fmul_rn(v[1], HAD_SCALE), sv.y), wgt);
                            o.z = __fmul_rn(__fmul_rn(__fmul_rn(v[2], HAD_SCALE), sv.z), wgt);
                            o.w = __fmul_rn(__fmul_rn(__fmul_rn(v[3], HAD_SCALE), sv.w), wgt);
                            if constexpr ((DBG & 8) != 0) { if (o.x == 1234.5f) dst[0] = o.y; }
                            else red_add4(dst + s * 128, o);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Segment tables for `tile`-row tiles over ALL experts with rows (E4's seg_tables_kernel with cap = 0): experts back
// to back in expert order, tile i of expert e = rows row_off[e] + i*tile .. + min(tile, cnt[e] - i*tile). Also the
// row count. One block, n_exp <= 1024.

__global__ void __launch_bounds__(1024) me_seg_tables_kernel(const int64_t* __restrict__ counts, int n_exp, int tile,
                                                            int max_segs, int* __restrict__ seg_expert,
                                                            int* __restrict__ seg_row0, int* __restrict__ seg_rows,
                                                            int* __restrict__ num_segs, int* __restrict__ num_rows) {
    __shared__ int s_rows[1024], s_tiles[1024];
    const int e = threadIdx.x;
    const int c = e < n_exp ? (int) counts[e] : 0;
    const int nt = (c + tile - 1) / tile;
    s_rows[e] = c;
    s_tiles[e] = nt;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int r = e >= off ? s_rows[e - off] : 0, t = e >= off ? s_tiles[e - off] : 0;
        __syncthreads();
        s_rows[e] += r;
        s_tiles[e] += t;
        __syncthreads();
    }
    const int row_off = s_rows[e] - c, tile_off = s_tiles[e] - nt;
    for (int i = 0; i < nt && tile_off + i < max_segs; ++i) {
        seg_expert[tile_off + i] = e;
        seg_row0[tile_off + i] = row_off + i * tile;
        seg_rows[tile_off + i] = min(tile, c - i * tile);
    }
    if (e == 1023) {
        num_segs[0] = min(s_tiles[1023], max_segs);
        num_rows[0] = s_rows[1023];
    }
}

// ---------------------------------------------------------------------------------------------------------
// host side

void check_ptrs(const at::Tensor& t, const char* name, int64_t n) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kLong && t.dim() == 1 && t.size(0) >= n,
                name, " must be a contiguous int64 CUDA vector of >= n_exp device pointers");
}
void check_i32(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt && t.dim() == 1, name,
                " must be a contiguous int32 CUDA vector");
}
void check_u8(const at::Tensor& t, const char* name, int64_t cols) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kByte && t.dim() == 2 && t.size(1) == cols,
                name, " must be a contiguous uint8 [rows, ", cols, "] CUDA tensor");
}

int g_num_sms = 0;
int num_sms() {
    if (!g_num_sms) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return g_num_sms;
}

template <typename F>
int grid_for(F kernel, int threads, int smem, int64_t max_items) {
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem);
    if (per_sm < 1) per_sm = 1;
    int64_t g = (int64_t) num_sms() * per_sm;
    if (g > max_items) g = max_items;
    return (int) (g < 1 ? 1 : g);
}

template <typename F>
void set_smem(F kernel, int smem) {
    // opt-moe2-rev: a refused attribute (e.g. more dynamic shared memory than the device gives one block) is a
    // non-sticky error that the runtime ALSO records as the thread's last error; left there, the next
    // C10_CUDA_KERNEL_LAUNCH_CHECK of an unrelated kernel (the production fallback path after the load self-test
    // caught this exception) raises it. Consume it before throwing so the Python-level fallback really falls back.
    const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        TORCH_CHECK(false, "MOE_E4M3: cannot set dynamic shared memory to ", smem, " (", cudaGetErrorString(err), ")");
    }
}

constexpr int GU_MB = 8, GU_STAGES = 4, GU_KSB = 8, DN_MB = 8, DN_STAGES = 4, DN_KSB = 8, F_EB = 1, MS_STAGES = 3;

}  // namespace

int64_t me_tile_rows() { return GU_MB * 16; }

void me_gather(at::Tensor x, at::Tensor local, at::Tensor pos, at::Tensor suh_ptrs, at::Tensor a8, at::Tensor asc,
               int64_t topk, int64_t n_exp, int64_t variant) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf && x.dim() == 2, "x: half [T, K]");
    const int T = (int) x.size(0), K = (int) x.size(1);
    TORCH_CHECK(K == 4096, "MOE_E4M3 gather: K must be 4096 (got ", K, ")");
    check_i32(local, "local");
    check_i32(pos, "pos");
    TORCH_CHECK(local.size(0) >= (int64_t) T * topk && pos.size(0) >= (int64_t) T * topk, "pair tables too short");
    check_ptrs(suh_ptrs, "suh_ptrs", n_exp);
    check_u8(a8, "a8", K);
    TORCH_CHECK(asc.is_cuda() && asc.is_contiguous() && asc.scalar_type() == at::kFloat && asc.size(0) >= a8.size(0),
                "asc: float [rows_cap]");
    if (T == 0) return;
    TORCH_CHECK(topk >= 1 && topk <= 8, "MOE_E4M3 gather: 1 <= topk <= 8");
    int blocks = T;
    const int cap = num_sms() * 24;
    if (blocks > cap) blocks = cap;
    auto launch = [&](auto kern) {
        kern<<<blocks, 32 * (int) topk, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(x.data_ptr()), local.data_ptr<int>(), pos.data_ptr<int>(),
            suh_ptrs.data_ptr<int64_t>(), a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), T, (int) topk, (int) n_exp);
    };
    switch (variant) {
        case 0: launch(me_gather_kernel<4096, 0>); break;
#ifdef ME_DEBUG_VARIANTS
        case 101: launch(me_gather_kernel<4096, 1>); break;
        case 102: launch(me_gather_kernel<4096, 2>); break;
        case 103: launch(me_gather_kernel<4096, 3>); break;
#endif
        default: TORCH_CHECK(false, "MOE_E4M3: unknown gather variant ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Single-pass gather (me_gather2_kernel, 4 warps per row); out (float [T, out_cols]) is zeroed when defined.
void me_gather2(at::Tensor x, at::Tensor local, at::Tensor pos, at::Tensor suh_ptrs, at::Tensor a8, at::Tensor asc,
                c10::optional<at::Tensor> out, int64_t topk, int64_t n_exp) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                x.dim() == 2, "x: half or bfloat16 [T, K] (bf16 is rounded to fp16 in the kernel, as x.half())");
    const int T = (int) x.size(0), K = (int) x.size(1);
    TORCH_CHECK(K == 4096, "MOE_E4M3 gather: K must be 4096 (got ", K, ")");
    TORCH_CHECK(topk == 2 || topk == 4 || topk == 8, "MOE_E4M3 gather2: topk must be 2, 4 or 8");
    check_i32(local, "local");
    check_i32(pos, "pos");
    TORCH_CHECK(local.size(0) >= (int64_t) T * topk && pos.size(0) >= (int64_t) T * topk, "pair tables too short");
    check_ptrs(suh_ptrs, "suh_ptrs", n_exp);
    check_u8(a8, "a8", K);
    TORCH_CHECK(asc.is_cuda() && asc.is_contiguous() && asc.scalar_type() == at::kFloat && asc.size(0) >= a8.size(0),
                "asc: float [rows_cap]");
    float* op = nullptr;
    int oc = 0;
    if (out.has_value() && out->defined()) {
        TORCH_CHECK(out->is_cuda() && out->is_contiguous() &&
                    (out->scalar_type() == at::kFloat || out->scalar_type() == at::kBFloat16) && out->dim() == 2 &&
                    out->size(0) == T, "out: float or bfloat16 [T, cols]");
        // zeroing is by bytes: a bf16 [T, cols] accumulator is zeroed as float [T, cols / 2]
        oc = (int) (out->scalar_type() == at::kFloat ? out->size(1) : out->size(1) / 2);
        TORCH_CHECK(oc % (4 * (topk / 2)) == 0, "out cols");
        op = reinterpret_cast<float*>(out->data_ptr());
    }
    if (T == 0) return;
    int64_t blocks = (int64_t) T * (topk / 2);
    const int cap = num_sms() * 16;
    if (blocks > cap) blocks = cap;
    if (x.scalar_type() == at::kBFloat16)
        me_gather2_kernel<4096, 4, __nv_bfloat16><<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), local.data_ptr<int>(), pos.data_ptr<int>(),
            suh_ptrs.data_ptr<int64_t>(), a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), op, T, (int) topk, (int) n_exp, oc);
    else
        me_gather2_kernel<4096, 4><<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(x.data_ptr()), local.data_ptr<int>(), pos.data_ptr<int>(),
            suh_ptrs.data_ptr<int64_t>(), a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), op, T, (int) topk, (int) n_exp, oc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// opt-moe TG: one e4m3 row per TOKEN (a8[t], asc[t]) with the suh of expert 0 (every expert's, checked by the
// caller); out (float or bf16 [T, cols]) zeroed when defined.
void me_gather_tok(at::Tensor x, at::Tensor suh_ptrs, at::Tensor a8, at::Tensor asc, c10::optional<at::Tensor> out) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                x.dim() == 2, "x: half or bfloat16 [T, K]");
    const int T = (int) x.size(0), K = (int) x.size(1);
    TORCH_CHECK(K == 4096, "MOE_E4M3 gather_tok: K must be 4096 (got ", K, ")");
    TORCH_CHECK(suh_ptrs.is_cuda() && suh_ptrs.scalar_type() == at::kLong && suh_ptrs.numel() >= 1, "suh_ptrs");
    check_u8(a8, "a8", K);
    TORCH_CHECK(a8.size(0) >= T, "a8 rows < T");
    TORCH_CHECK(asc.is_cuda() && asc.is_contiguous() && asc.scalar_type() == at::kFloat && asc.size(0) >= T,
                "asc: float [>= T]");
    float* op = nullptr;
    int oc = 0;
    if (out.has_value() && out->defined()) {
        TORCH_CHECK(out->is_cuda() && out->is_contiguous() &&
                    (out->scalar_type() == at::kFloat || out->scalar_type() == at::kBFloat16) && out->dim() == 2 &&
                    out->size(0) == T, "out: float or bfloat16 [T, cols]");
        oc = (int) (out->scalar_type() == at::kFloat ? out->size(1) : out->size(1) / 2);
        TORCH_CHECK(oc % 4 == 0, "out cols");
        op = reinterpret_cast<float*>(out->data_ptr());
    }
    if (T == 0) return;
    int64_t blocks = ((int64_t) T + 1) / 2;
    const int cap = num_sms() * 16;
    if (blocks > cap) blocks = cap;
    const int* dummy = nullptr;
    if (x.scalar_type() == at::kBFloat16)
        me_gather2_kernel<4096, 4, __nv_bfloat16, 1><<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), dummy, dummy, suh_ptrs.data_ptr<int64_t>(),
            a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), op, T, 1, 1, oc);
    else
        me_gather2_kernel<4096, 4, half, 1><<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(x.data_ptr()), dummy, dummy, suh_ptrs.data_ptr<int64_t>(),
            a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), op, T, 1, 1, oc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void me_actq(at::Tensor a16, at::Tensor row_expert, at::Tensor suh_ptrs, at::Tensor a8, at::Tensor dsc,
             at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor num_segs, int64_t chunk, int64_t nchunks) {
    const at::cuda::OptionalCUDAGuard guard(a16.device());
    TORCH_CHECK(a16.is_cuda() && a16.is_contiguous() && a16.scalar_type() == at::kHalf && a16.dim() == 2,
                "a16: half [rows_cap, N]");
    const int K = (int) a16.size(1);
    TORCH_CHECK(K == 1024, "MOE_E4M3 actq: intermediate must be 1024 (got ", K, ")");
    check_i32(row_expert, "row_expert");
    check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows"); check_i32(num_segs, "num_segs");
    check_ptrs(suh_ptrs, "suh_ptrs", 1);
    check_u8(a8, "a8d", K);
    TORCH_CHECK(a8.size(0) >= a16.size(0) && row_expert.size(0) >= a16.size(0), "row capacity");
    TORCH_CHECK(dsc.is_cuda() && dsc.is_contiguous() && dsc.scalar_type() == at::kFloat && dsc.size(0) >= a16.size(0),
                "dsc: float [rows_cap]");
    TORCH_CHECK(nchunks >= 1 && chunk >= 0 && chunk < nchunks, "chunk");
    int64_t blocks = (a16.size(0) / nchunks + 7) / 8;
    const int cap = num_sms() * 16;
    if (blocks > cap) blocks = cap;
    if (blocks < 1) blocks = 1;
    me_actq_kernel<1024><<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a16.data_ptr()), row_expert.data_ptr<int>(), suh_ptrs.data_ptr<int64_t>(),
        a8.data_ptr<uint8_t>(), dsc.data_ptr<float>(), seg_row0.data_ptr<int>(), seg_rows.data_ptr<int>(),
        num_segs.data_ptr<int>(), (int) chunk, (int) nchunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

template <int ST, int KSB, int DBG>
void launch_gateup(const at::Tensor& a8, const at::Tensor& asc, const at::Tensor& gp, const at::Tensor& up,
                   const at::Tensor& gs, const at::Tensor& us, at::Tensor& a16, const at::Tensor& se,
                   const at::Tensor& sr0, const at::Tensor& sr, const at::Tensor& ns, int K, int N, float limit,
                   int chunk, int nchunks, int grid) {
    constexpr int smem = Smem<GU_MB, ST, KSB>::TOTAL;
    auto kern = me_gateup_kernel<GU_MB, ST, KSB, DBG>;
    static bool attr = false;
    if (!attr) { set_smem(kern, smem); attr = true; }
    const int64_t max_items = se.size(0) * (int64_t) (N / 128);
    int g = grid_for(kern, 512, smem, max_items);
    if (grid > 0 && grid < g) g = grid;
    kern<<<g, 512, smem, at::cuda::getCurrentCUDAStream()>>>(
        a8.data_ptr<uint8_t>(), asc.data_ptr<float>(), gp.data_ptr<int64_t>(), up.data_ptr<int64_t>(),
        gs.data_ptr<int64_t>(), us.data_ptr<int64_t>(), reinterpret_cast<half*>(a16.data_ptr()), se.data_ptr<int>(),
        sr0.data_ptr<int>(), sr.data_ptr<int>(), ns.data_ptr<int>(), K, N, limit, chunk, nchunks);
}

template <int ST, int KSB, int DBG>
void launch_down(const at::Tensor& a8, const at::Tensor& dsc, const at::Tensor& dp, const at::Tensor& dsv,
                 at::Tensor& out, const at::Tensor& rt, const at::Tensor& rw, const at::Tensor& se,
                 const at::Tensor& sr0, const at::Tensor& sr, const at::Tensor& ns, int K, int N, int chunk,
                 int nchunks, int grid) {
    constexpr int smem = Smem<DN_MB, ST, KSB>::TOTAL;
    auto kern = me_down_kernel<DN_MB, ST, KSB, DBG>;
    static bool attr = false;
    if (!attr) { set_smem(kern, smem); attr = true; }
    const int64_t max_items = se.size(0) * (int64_t) (N / 256);
    int g = grid_for(kern, 512, smem, max_items);
    if (grid > 0 && grid < g) g = grid;
    kern<<<g, 512, smem, at::cuda::getCurrentCUDAStream()>>>(
        a8.data_ptr<uint8_t>(), dsc.data_ptr<float>(), dp.data_ptr<int64_t>(), dsv.data_ptr<int64_t>(),
        out.data_ptr<float>(), rt.data_ptr<int64_t>(), rw.data_ptr<float>(), se.data_ptr<int>(), sr0.data_ptr<int>(),
        sr.data_ptr<int>(), ns.data_ptr<int>(), K, N, chunk, nchunks);
}

}  // namespace

// variant 0 = the shipped configuration; others exist only in experiment builds (-DME_DEBUG_VARIANTS):
// 2 / 3 = 5 / 3 pipeline stages, 100 + DBG = parts removed (mainloop()).
void me_gateup(at::Tensor a8, at::Tensor asc, at::Tensor gate_ptrs, at::Tensor up_ptrs, at::Tensor gate_svh,
               at::Tensor up_svh, at::Tensor a16, at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows,
               at::Tensor num_segs, double limit, int64_t variant, int64_t chunk, int64_t nchunks, int64_t grid) {
    const at::cuda::OptionalCUDAGuard guard(a8.device());
    TORCH_CHECK(nchunks >= 1 && chunk >= 0 && chunk < nchunks, "chunk");
    const int K = (int) a8.size(1);
    check_u8(a8, "a8", K);
    TORCH_CHECK(a16.is_cuda() && a16.is_contiguous() && a16.scalar_type() == at::kHalf && a16.dim() == 2, "a16");
    const int N = (int) a16.size(1);
    TORCH_CHECK(K % 128 == 0 && K >= 128 * 4 && N % 128 == 0, "MOE_E4M3 gate/up: K % 128, N % 128");
    TORCH_CHECK(a8.size(0) == a16.size(0) && asc.size(0) >= a8.size(0), "row capacity");
    const int64_t n = gate_ptrs.size(0);
    check_ptrs(gate_ptrs, "gate_ptrs", 1);
    for (auto* p : {&up_ptrs, &gate_svh, &up_svh}) check_ptrs(*p, "pointer table", n);
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
#define ME_GU(ST, KSB, DBG) launch_gateup<ST, KSB, DBG>(a8, asc, gate_ptrs, up_ptrs, gate_svh, up_svh, a16, seg_expert, \
                                              seg_row0, seg_rows, num_segs, K, N, (float) limit, \
                                                   (int) chunk, (int) nchunks, (int) grid)
    switch (variant) {
        case 0: ME_GU(GU_STAGES, GU_KSB, 0); break;
#ifdef ME_DEBUG_VARIANTS
        case 2: ME_GU(5, 4, 0); break;
        case 3: ME_GU(3, 8, 0); break;
        case 4: ME_GU(4, 4, 0); break;
        case 101: ME_GU(GU_STAGES, GU_KSB, 1); break;
        case 103: ME_GU(GU_STAGES, GU_KSB, 3); break;
        case 105: ME_GU(GU_STAGES, GU_KSB, 5); break;
        case 107: ME_GU(GU_STAGES, GU_KSB, 7); break;
        case 108: ME_GU(GU_STAGES, GU_KSB, 8); break;
#endif
        default: TORCH_CHECK(false, "MOE_E4M3: unknown gate/up variant ", variant);
    }
#undef ME_GU
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void me_down(at::Tensor a8, at::Tensor dsc, at::Tensor down_ptrs, at::Tensor down_svh, at::Tensor out,
             at::Tensor row_token, at::Tensor row_weight, at::Tensor seg_expert, at::Tensor seg_row0,
             at::Tensor seg_rows, at::Tensor num_segs, int64_t variant, int64_t chunk, int64_t nchunks, int64_t grid) {
    const at::cuda::OptionalCUDAGuard guard(a8.device());
    TORCH_CHECK(nchunks >= 1 && chunk >= 0 && chunk < nchunks, "chunk");
    const int K = (int) a8.size(1);
    check_u8(a8, "a8d", K);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.dim() == 2, "out");
    const int N = (int) out.size(1);
    TORCH_CHECK(K % 128 == 0 && K >= 128 * 4 && N % 256 == 0, "MOE_E4M3 down: K % 128, N % 256");
    TORCH_CHECK(row_token.is_cuda() && row_token.scalar_type() == at::kLong && row_token.is_contiguous() &&
                row_token.size(0) >= a8.size(0), "row_token must be int64[rows_cap]");
    TORCH_CHECK(row_weight.is_cuda() && row_weight.scalar_type() == at::kFloat && row_weight.is_contiguous() &&
                row_weight.size(0) >= a8.size(0), "row_weight must be float[rows_cap]");
    TORCH_CHECK(dsc.size(0) >= a8.size(0), "dsc capacity");
    check_ptrs(down_ptrs, "down_ptrs", 1);
    check_ptrs(down_svh, "down_svh", down_ptrs.size(0));
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
#define ME_DN(ST, KSB, DBG) launch_down<ST, KSB, DBG>(a8, dsc, down_ptrs, down_svh, out, row_token, row_weight, seg_expert, \
                                            seg_row0, seg_rows, num_segs, K, N, (int) chunk, \
                                                 (int) nchunks, (int) grid)
    switch (variant) {
        case 0: ME_DN(DN_STAGES, DN_KSB, 0); break;
#ifdef ME_DEBUG_VARIANTS
        case 2: ME_DN(5, 4, 0); break;
        case 3: ME_DN(3, 8, 0); break;
        case 4: ME_DN(4, 4, 0); break;
        case 101: ME_DN(DN_STAGES, DN_KSB, 1); break;
        case 103: ME_DN(DN_STAGES, DN_KSB, 3); break;
        case 105: ME_DN(DN_STAGES, DN_KSB, 5); break;
        case 107: ME_DN(DN_STAGES, DN_KSB, 7); break;
        case 108: ME_DN(DN_STAGES, DN_KSB, 8); break;
        case 116: ME_DN(DN_STAGES, DN_KSB, 16); break;
        case 132: ME_DN(DN_STAGES, DN_KSB, 32); break;
#endif
        default: TORCH_CHECK(false, "MOE_E4M3: unknown down variant ", variant);
    }
#undef ME_DN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused gate/up + actq + down (me_fused_kernel). sync: int32 [1 + 2 * max_segs] ZEROED by the caller per launch.
void me_fused(at::Tensor a8, at::Tensor asc, at::Tensor a8d, at::Tensor dsc, at::Tensor a16, at::Tensor out,
              at::Tensor gate_ptrs, at::Tensor up_ptrs, at::Tensor gate_svh, at::Tensor up_svh, at::Tensor down_ptrs,
              at::Tensor down_suh, at::Tensor down_svh, at::Tensor row_token, at::Tensor row_weight,
              at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor num_segs, at::Tensor sync,
              double limit, int64_t lag, int64_t grid, int64_t variant) {
    const at::cuda::OptionalCUDAGuard guard(a8.device());
    check_u8(a8, "a8", F_HID);
    check_u8(a8d, "a8d", F_INT);
    TORCH_CHECK(a16.is_cuda() && a16.is_contiguous() && a16.scalar_type() == at::kHalf && a16.dim() == 2 &&
                a16.size(1) == F_INT, "a16: half [rows_cap, 1024]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16) && out.dim() == 2 &&
                out.size(1) == F_HID, "out: float or bfloat16 [T, 4096]");
    const bool ob = out.scalar_type() == at::kBFloat16;     // opt-moe: bf16 accumulator (variants 0 / 1 / 16)
    const int64_t rows_cap = a8.size(0);
    TORCH_CHECK(a8d.size(0) >= rows_cap && a16.size(0) >= rows_cap && asc.size(0) >= rows_cap &&
                dsc.size(0) >= rows_cap, "row capacity");
    TORCH_CHECK(asc.scalar_type() == at::kFloat && dsc.scalar_type() == at::kFloat, "scales: float");
    TORCH_CHECK(row_token.scalar_type() == at::kLong && row_token.size(0) >= rows_cap, "row_token: int64[rows_cap]");
    TORCH_CHECK(row_weight.scalar_type() == at::kFloat && row_weight.size(0) >= rows_cap, "row_weight: float[rows_cap]");
    const int64_t n = gate_ptrs.size(0);
    check_ptrs(gate_ptrs, "gate_ptrs", 1);
    for (auto* p : {&up_ptrs, &gate_svh, &up_svh, &down_ptrs, &down_suh, &down_svh}) check_ptrs(*p, "pointer table", n);
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
    check_i32(sync, "sync");
    const int64_t max_segs = seg_expert.size(0);
    TORCH_CHECK(sync.size(0) >= 1 + 2 * max_segs, "sync must hold 1 + 2 * max_segs ints");
    TORCH_CHECK(lag >= 1, "lag >= 1");
    FusedArgs A;
    A.a8 = a8.data_ptr<uint8_t>(); A.asc = asc.data_ptr<float>(); A.a8d = a8d.data_ptr<uint8_t>();
    A.dsc = dsc.data_ptr<float>(); A.a16 = reinterpret_cast<half*>(a16.data_ptr()); A.out = ob ? nullptr : out.data_ptr<float>();
    A.outb = ob ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
    A.gate_ptrs = gate_ptrs.data_ptr<int64_t>(); A.up_ptrs = up_ptrs.data_ptr<int64_t>();
    A.gate_svh = gate_svh.data_ptr<int64_t>(); A.up_svh = up_svh.data_ptr<int64_t>();
    A.down_ptrs = down_ptrs.data_ptr<int64_t>(); A.down_suh = down_suh.data_ptr<int64_t>();
    A.down_svh = down_svh.data_ptr<int64_t>(); A.row_token = row_token.data_ptr<int64_t>();
    A.row_weight = row_weight.data_ptr<float>(); A.seg_expert = seg_expert.data_ptr<int>();
    A.seg_row0 = seg_row0.data_ptr<int>(); A.seg_rows = seg_rows.data_ptr<int>(); A.num_segs = num_segs.data_ptr<int>();
    A.sync = sync.data_ptr<int>(); A.max_segs = (int) max_segs; A.lag = (int) lag; A.limit = (float) limit;
    [[maybe_unused]] constexpr int MS_ST = MsCfg<GU_MB, GU_KSB>::STAGE, MS_SM = MS_STAGES * MS_ST;
    // MAINLOOP=1: the shipped layout + the A address table (2 x 512 threads x 2 x 8 B behind the staging)
    // ME_AT_SM_EXTRA (test builds only, default 0): inflate the MAINLOOP launch's dynamic shared memory request past
    // the device limit to exercise the load self-test's fallback (tests/optmoe2rev/fallback_ms.py); host-side only
#ifndef ME_AT_SM_EXTRA
#define ME_AT_SM_EXTRA 0
#endif
    constexpr int MS_SHIP = 8 | 128, AT_SM = FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL + 16384 + ME_AT_SM_EXTRA;
    auto launch = [&](auto kern, int smem) {
        set_smem(kern, smem);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, 512, smem);
        TORCH_CHECK(per_sm >= 1, "MOE_E4M3 fused: kernel does not fit on an SM");
        // persistent: every CTA must be resident (an item may wait for an item held by another CTA)
        int g = num_sms() * per_sm;
        if (grid > 0 && grid < g) g = (int) grid;
        kern<<<g, 512, smem, at::cuda::getCurrentCUDAStream()>>>(A);
    };
#define ME_F(EB, DBG) launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, EB, DBG>, FSmem<GU_MB, GU_STAGES, GU_KSB, EB>::TOTAL)
#define ME_FS(ST, KSB) launch(me_fused_kernel<GU_MB, ST, KSB, 1, 0>, FSmem<GU_MB, ST, KSB, 1>::TOTAL)
    if (ob) {
        // bf16 accumulator: variant 0 = 8 values per op (OB 2), 1 = 4 values per op (OB 1), 16 = DN16 + OB 2
        constexpr int SM_ = FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL;
        switch (variant) {
            case 0: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2>, SM_); break;
            case 1: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 1>, SM_); break;
            case 16: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2>, SM_); break;
            case 2048: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 0, 1>, SM_); break;   // TG
            case 2064: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2, 0, 0, 1>, SM_); break;   // TG + DN16
            // opt-moe2 GLM53_MOE_E4M3_MAINLOOP=1 (+ 8192): the shipped mainloop with the lean trellis decode and the A
            // copy address table (MS 136), same arithmetic
            case 8192: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 0, 0, MS_SHIP>, AT_SM); break;
            case 8208: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2, 0, 0, 0, MS_SHIP>, AT_SM); break;
            case 10240: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 0, 1, MS_SHIP>, AT_SM); break;
            case 10256: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2, 0, 0, 1, MS_SHIP>, AT_SM); break;
#ifdef ME_DEBUG_VARIANTS
            // opt-moe2 experiments (TG, bf16 accumulator; docs/OPT_MOE2.md): code = 2048 + (MS << 15) for the 4-stage
            // shipped layout; smem-B mainloop (MS & 1, 3 x 32 KB stages) as listed
            case 264192: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 8>, SM_); break;            // lean only
            case 4196352: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 128>, AT_SM); break;       // AT only
            case 12847104: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 392>, AT_SM); break;      // lean+AT+BK
            case 21235712: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 648>, AT_SM); break;      // lean+AT+FULL
            case 29624320: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 904>, AT_SM); break;      // lean+AT+BK+FULL
            case 4460544: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 64, 0, 2, 0, 0, 1, 136>, AT_SM); break;      // MAINLOOP=1, profiled (DBG 64)
            case 1050624: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 32>, SM_); break;          // RS (B word ring)
            case 1312768: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 40>, SM_); break;          // RS + lean
            case 2361344: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 104>, SM_); break;         // RS ring 4 + lean
            case 59392: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 1>, MS_SM); break;           // smem-B MS
            case 67584: launch(me_fused_kernel<GU_MB, 2, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 1>, 2 * MS_ST); break;              // smem-B, 2 stages
            case 92160: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 3>, MS_SM); break;           // smem-B, decode not pipelined
            case 124928: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 5>, MS_SM); break;          // smem-B, single A buffer
            case 157696: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 7>, MS_SM); break;          // smem-B only
            case 190464: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 9>, MS_SM); break;          // smem-B + lean
            case 321536: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 17>, MS_SM); break;         // smem-B + metadata
            case 452608: launch(me_fused_kernel<GU_MB, MS_STAGES, GU_KSB, 1, 0, 0, 2, 0, 0, 1, 25>, MS_SM); break;         // smem-B + lean + metadata
#endif
#ifdef ME_DEBUG_VARIANTS
            // TKP (ticket prefetch): measured slower (30.93 vs 30.57 ms at 13,824; 13.99 vs 13.55 at 4,289)
            case 512: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 1>, SM_); break;
            case 528: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2, 0, 1>, SM_); break;
            // MP = 1 (metadata prefetch into shared memory): measured slower (31.55 vs 30.44 ms at 13,824), debug only
            case 256: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 1>, SM_); break;
            case 272: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 2, 1>, SM_); break;
            case 164: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 64, 0, 2>, SM_); break;
            case 420: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 64, 0, 2, 1>, SM_); break;
            case 1001: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x100, 0, 2>, SM_); break;   // no mma
            case 1002: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x200, 0, 2>, SM_); break;   // no A copies
            case 1004: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x400, 0, 2>, SM_); break;   // no decode
            case 1005: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x500, 0, 2>, SM_); break;   // no decode, no mma
            case 1006: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x600, 0, 2>, SM_); break;   // no decode, no copies
            case 1100: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 8, 0, 2>, SM_); break;   // no down reds
            case 1102: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, 2, 0, 0, 2>, FSmem<GU_MB, GU_STAGES, GU_KSB, 2>::TOTAL); break;  // EB 2
            case 1103: launch(me_fused_kernel<GU_MB, 3, GU_KSB, 2, 0, 0, 2>, FSmem<GU_MB, 3, GU_KSB, 2>::TOTAL); break;  // EB 2, 3 stages
            case 4097: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 0, 3>, SM_); break;  // TG staging only
            case 4096: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 2, 0, 0, 2>, SM_); break;  // TG probe
            case 3164: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 64, 0, 2, 0, 0, 1>, SM_); break;  // TG, profiled
            case 1164: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x1F00 | 64, 0, 2>, SM_); break;  // skeleton, profiled
            case 1016: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x1000, 0, 2>, SM_); break;  // no ldsm
            case 1017: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x1100, 0, 2>, SM_); break;  // no ldsm, no mma
            case 1020: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x1400, 0, 2>, SM_); break;  // no ldsm, no decode
            case 1015: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0xF00, 0, 2>, SM_); break;   // no mma/decode/A/B
            case 1031: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x1F00, 0, 2>, SM_); break;  // skeleton
            case 1008: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0x800, 0, 2>, SM_); break;   // no B loads
            case 1010: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0xA00, 0, 2>, SM_); break;   // no A, no B loads
            case 1012: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0xC00, 0, 2>, SM_); break;   // no B loads, no decode
#endif
            default: TORCH_CHECK(false, "MOE_E4M3: unknown fused variant ", variant, " for a bf16 accumulator");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    switch (variant) {
        case 0: ME_F(F_EB, 0); break;
        case 16: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 2048: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 0, 0, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 2064: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 0, 0, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 8192: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 0, 0, 0, 0, MS_SHIP>, AT_SM); break;
        case 8208: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 0, 0, 0, 0, MS_SHIP>, AT_SM); break;
        case 10240: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 0, 0, 0, 1, MS_SHIP>, AT_SM); break;
        case 10256: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 0, 0, 0, 1, MS_SHIP>, AT_SM); break;
#ifdef ME_DEBUG_VARIANTS
        case 512: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 0, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 528: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 0, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 256: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 0, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 272: launch(me_fused_kernel<GU_MB, GU_STAGES, GU_KSB, F_EB, 0, 1, 0, 1>, FSmem<GU_MB, GU_STAGES, GU_KSB, F_EB>::TOTAL); break;
        case 2: ME_F(2, 0); break;
        case 3: ME_FS(3, 8); break;
        case 4: ME_FS(6, 4); break;
        case 5: ME_FS(5, 8); break;
        case 108: ME_F(F_EB, 8); break;
        case 116: ME_F(F_EB, 16); break;
        case 132: ME_F(F_EB, 32); break;
        case 164: ME_F(F_EB, 64); break;
#endif
        default: TORCH_CHECK(false, "MOE_E4M3: unknown fused variant ", variant);
    }
#undef ME_F
#undef ME_FS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// P16 (GLM53_MOE_FUSED16): gather16 = production's fm_gather_kernel arithmetic, rows visited in perm order.
void me_gather16(at::Tensor x, c10::optional<at::Tensor> perm, at::Tensor row_token, at::Tensor row_expert,
                 at::Tensor suh_ptrs, at::Tensor h13, at::Tensor num_rows) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(1) == F_HID,
                "x: half [T, 4096]");
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous() && h13.scalar_type() == at::kHalf && h13.dim() == 2 &&
                h13.size(1) == F_HID, "h13: half [rows_cap, 4096]");
    const int64_t rows_cap = h13.size(0);
    TORCH_CHECK(row_token.is_cuda() && row_token.scalar_type() == at::kLong && row_token.is_contiguous() &&
                row_token.size(0) >= rows_cap, "row_token: int64 [rows_cap]");
    check_i32(row_expert, "row_expert");
    TORCH_CHECK(row_expert.size(0) >= rows_cap, "row_expert: int32 [rows_cap]");
    check_i32(num_rows, "num_rows");
    check_ptrs(suh_ptrs, "suh_ptrs", 1);
    const int* pp = nullptr;
    if (perm.has_value() && perm->defined()) {
        check_i32(*perm, "perm");
        TORCH_CHECK(perm->size(0) >= rows_cap, "perm: int32 [rows_cap]");
        pp = perm->data_ptr<int>();
    }
    if (rows_cap == 0) return;
    int64_t blocks = (rows_cap + 7) / 8;
    const int cap = num_sms() * 16;
    if (blocks > cap) blocks = cap;
    me_gather16_kernel<<<(int) blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr()), pp, row_token.data_ptr<int64_t>(), row_expert.data_ptr<int>(),
        suh_ptrs.data_ptr<int64_t>(), reinterpret_cast<half*>(h13.data_ptr()), num_rows.data_ptr<int>(), (int) rows_cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void me_exp_check(int64_t b0, int64_t b1, at::Tensor cnt) {
    TORCH_CHECK(cnt.is_cuda() && cnt.scalar_type() == at::kLong && cnt.numel() >= 2, "cnt: int64[2] on the GPU");
    me_exp_check_kernel<<<num_sms() * 16, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (uint32_t) b0, (uint32_t) b1, reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void me_pgu(at::Tensor h13, at::Tensor gate_ptrs, at::Tensor up_ptrs, at::Tensor gate_svh, at::Tensor up_svh,
            at::Tensor down_suh, at::Tensor h2, at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows,
            at::Tensor num_segs, double limit, int64_t variant, int64_t chunk, int64_t nchunks) {
    const at::cuda::OptionalCUDAGuard guard(h13.device());
    TORCH_CHECK(nchunks >= 1 && chunk >= 0 && chunk < nchunks, "chunk");
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous() && h13.scalar_type() == at::kHalf && h13.size(1) == F_HID, "h13");
    TORCH_CHECK(h2.is_cuda() && h2.is_contiguous() && h2.scalar_type() == at::kHalf && h2.size(1) == F_INT, "h2");
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
    const int64_t n = gate_ptrs.size(0);
    check_ptrs(gate_ptrs, "gate_ptrs", 1);
    for (auto* p : {&up_ptrs, &gate_svh, &up_svh, &down_suh}) check_ptrs(*p, "pointer table", n);
    int64_t gy = seg_expert.size(0);
    if (gy > 512) gy = 512;
    if (gy < 1) gy = 1;
    dim3 grid(F_INT / 128, (unsigned) gy);
    auto launch = [&](auto kern) {
        set_smem(kern, PG_SMEM);
        kern<<<grid, PG_THREADS, PG_SMEM, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(h13.data_ptr()), gate_ptrs.data_ptr<int64_t>(), up_ptrs.data_ptr<int64_t>(),
            gate_svh.data_ptr<int64_t>(), up_svh.data_ptr<int64_t>(), down_suh.data_ptr<int64_t>(),
            reinterpret_cast<half*>(h2.data_ptr()), seg_expert.data_ptr<int>(), seg_row0.data_ptr<int>(),
            seg_rows.data_ptr<int>(), num_segs.data_ptr<int>(), (float) limit, (int) chunk, (int) nchunks);
    };
    switch (variant) {
        case 0: launch(me_pgu_kernel<0>); break;
        case 512: launch(me_pgu_kernel<512>); break;
#ifdef ME_DEBUG_VARIANTS
        case 4: launch(me_pgu_kernel<4>); break;
        case 64: launch(me_pgu_kernel<64>); break;
        case 68: launch(me_pgu_kernel<68>); break;
        case 2048: launch(me_pgu_kernel<2048>); break;
#endif
        default: TORCH_CHECK(false, "MOE P16: unknown pgu variant ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void me_pdn(at::Tensor h2, at::Tensor down_ptrs, at::Tensor down_svh, at::Tensor out, at::Tensor row_token,
            at::Tensor row_weight, at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor num_segs,
            int64_t variant, int64_t chunk, int64_t nchunks) {
    const at::cuda::OptionalCUDAGuard guard(h2.device());
    TORCH_CHECK(nchunks >= 1 && chunk >= 0 && chunk < nchunks, "chunk");
    TORCH_CHECK(h2.is_cuda() && h2.is_contiguous() && h2.scalar_type() == at::kHalf && h2.size(1) == F_INT, "h2");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.size(1) == F_HID, "out");
    TORCH_CHECK(row_token.scalar_type() == at::kLong && row_token.size(0) >= h2.size(0), "row_token");
    TORCH_CHECK(row_weight.scalar_type() == at::kFloat && row_weight.size(0) >= h2.size(0), "row_weight: float");
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
    check_ptrs(down_ptrs, "down_ptrs", 1);
    check_ptrs(down_svh, "down_svh", down_ptrs.size(0));
    int64_t gy = seg_expert.size(0) / nchunks + 1;
    if (gy > 512) gy = 512;
    dim3 grid(F_HID / 256, (unsigned) gy);
    auto launch = [&](auto kern) {
        set_smem(kern, PG_SMEM);
        kern<<<grid, PG_THREADS, PG_SMEM, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(h2.data_ptr()), down_ptrs.data_ptr<int64_t>(), down_svh.data_ptr<int64_t>(),
            out.data_ptr<float>(), row_token.data_ptr<int64_t>(), row_weight.data_ptr<float>(), seg_expert.data_ptr<int>(),
            seg_row0.data_ptr<int>(), seg_rows.data_ptr<int>(), num_segs.data_ptr<int>(), (int) chunk, (int) nchunks);
    };
    switch (variant) {
        case 0: launch(me_pdn_kernel<0>); break;
#ifdef ME_DEBUG_VARIANTS
        case 8: launch(me_pdn_kernel<8>); break;
#endif
        default: TORCH_CHECK(false, "MOE P16: unknown pdn variant ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// P16 v3: gather + gate/up + down in one persistent launch (me_p16b_kernel). Tables: production's
// build_grouped_fat_tables with tile 64. sync: int32 [1 + 3 * max_segs] zeroed by the caller. out is accumulated into.
void me_p16b(at::Tensor xh, at::Tensor h13, at::Tensor h2, at::Tensor out, at::Tensor gate_ptrs, at::Tensor up_ptrs,
             at::Tensor gate_svh, at::Tensor up_svh, at::Tensor gate_suh, at::Tensor down_ptrs, at::Tensor down_suh,
             at::Tensor down_svh, at::Tensor row_token, at::Tensor row_expert, at::Tensor row_weight,
             at::Tensor seg_expert, at::Tensor seg_row0, at::Tensor seg_rows, at::Tensor num_segs, at::Tensor sync,
             double limit, int64_t lg, int64_t ld, int64_t grid, int64_t variant) {
    const at::cuda::OptionalCUDAGuard guard(h13.device());
    TORCH_CHECK(xh.is_cuda() && xh.is_contiguous() && xh.scalar_type() == at::kHalf && xh.dim() == 2 &&
                xh.size(1) == F_HID, "xh: half [T, 4096]");
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous() && h13.scalar_type() == at::kHalf && h13.dim() == 2 &&
                h13.size(1) == F_HID, "h13: half [rows_cap, 4096]");
    TORCH_CHECK(h2.is_cuda() && h2.is_contiguous() && h2.scalar_type() == at::kHalf && h2.dim() == 2 &&
                h2.size(1) == F_INT, "h2: half [rows_cap, 1024]");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                out.size(1) == F_HID && out.size(0) == xh.size(0), "out: float [T, 4096]");
    const int64_t rows_cap = h13.size(0);
    TORCH_CHECK(h2.size(0) >= rows_cap, "row capacity");
    TORCH_CHECK(row_token.scalar_type() == at::kLong && row_token.is_contiguous() && row_token.size(0) >= rows_cap,
                "row_token: int64[rows_cap]");
    check_i32(row_expert, "row_expert");
    TORCH_CHECK(row_weight.scalar_type() == at::kFloat && row_weight.is_contiguous() && row_weight.size(0) >= rows_cap,
                "row_weight: float[rows_cap]");
    const int64_t n = gate_ptrs.size(0);
    check_ptrs(gate_ptrs, "gate_ptrs", 1);
    for (auto* p : {&up_ptrs, &gate_svh, &up_svh, &gate_suh, &down_ptrs, &down_suh, &down_svh})
        check_ptrs(*p, "pointer table", n);
    check_i32(seg_expert, "seg_expert"); check_i32(seg_row0, "seg_row0"); check_i32(seg_rows, "seg_rows");
    check_i32(num_segs, "num_segs");
    check_i32(sync, "sync");
    const int64_t max_segs = seg_expert.size(0);
    TORCH_CHECK(sync.size(0) >= 1 + 3 * max_segs, "sync must hold 1 + 3 * max_segs ints");
    TORCH_CHECK(lg >= 1 && ld >= 1, "lead / lag >= 1");
    P3Args A;
    A.x = reinterpret_cast<const half*>(xh.data_ptr()); A.h13 = reinterpret_cast<half*>(h13.data_ptr());
    A.h2 = reinterpret_cast<half*>(h2.data_ptr()); A.out = out.data_ptr<float>();
    A.gate_ptrs = gate_ptrs.data_ptr<int64_t>(); A.up_ptrs = up_ptrs.data_ptr<int64_t>();
    A.gate_svh = gate_svh.data_ptr<int64_t>(); A.up_svh = up_svh.data_ptr<int64_t>();
    A.gate_suh = gate_suh.data_ptr<int64_t>(); A.down_ptrs = down_ptrs.data_ptr<int64_t>();
    A.down_suh = down_suh.data_ptr<int64_t>(); A.down_svh = down_svh.data_ptr<int64_t>();
    A.row_token = row_token.data_ptr<int64_t>(); A.row_expert = row_expert.data_ptr<int>();
    A.row_weight = row_weight.data_ptr<float>(); A.seg_expert = seg_expert.data_ptr<int>();
    A.seg_row0 = seg_row0.data_ptr<int>(); A.seg_rows = seg_rows.data_ptr<int>(); A.num_segs = num_segs.data_ptr<int>();
    A.sync = sync.data_ptr<int>(); A.max_segs = (int) max_segs; A.limit = (float) limit;
    // the lead / lag are clamped in the kernel against the live segment count (LG >= 1, LG + LD <= nsegs)
    A.lg = (int) lg; A.ld = (int) ld;
    auto launch = [&](auto kern) {
        set_smem(kern, PG_SMEM);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, PG_THREADS, PG_SMEM);
        TORCH_CHECK(per_sm >= 1, "MOE P16b: kernel does not fit on an SM");
        int g = num_sms() * per_sm;
        if (grid > 0 && grid < g) g = (int) grid;
        kern<<<g, PG_THREADS, PG_SMEM, at::cuda::getCurrentCUDAStream()>>>(A);
    };
    switch (variant) {
        case 0: launch(me_p16b_kernel<0>); break;
#ifdef ME_DEBUG_VARIANTS
        case 8: launch(me_p16b_kernel<8>); break;
        case 512: launch(me_p16b_kernel<512>); break;
        case 128: launch(me_p16b_kernel<128>); break;
        case 256: launch(me_p16b_kernel<256>); break;
        case 264: launch(me_p16b_kernel<264>); break;
        case 1024: launch(me_p16b_kernel<1024>); break;
#endif
        default: TORCH_CHECK(false, "MOE P16b: unknown variant ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// (seg_expert, seg_row0, seg_rows, num_segs, num_rows), max_segs = rows_cap / tile + n_exp entries
std::vector<at::Tensor> me_seg_tables(at::Tensor counts, int64_t tile, int64_t rows_cap) {
    const at::cuda::OptionalCUDAGuard guard(counts.device());
    TORCH_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == at::kLong && counts.dim() == 1,
                "counts must be a contiguous int64 CUDA vector");
    const int64_t n_exp = counts.size(0);
    TORCH_CHECK(n_exp >= 1 && n_exp <= 1024 && tile >= 16 && rows_cap >= 0, "seg tables: 1 <= n_exp <= 1024");
    const int64_t max_segs = (rows_cap + tile - 1) / tile + n_exp;
    auto opt = torch::dtype(torch::kInt).device(counts.device());
    auto se = torch::zeros({max_segs}, opt), sr0 = torch::zeros({max_segs}, opt), sr = torch::zeros({max_segs}, opt);
    auto ns = torch::empty({1}, opt), nr = torch::empty({1}, opt);
    me_seg_tables_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        counts.data_ptr<int64_t>(), (int) n_exp, (int) tile, (int) max_segs, se.data_ptr<int>(), sr0.data_ptr<int>(),
        sr.data_ptr<int>(), ns.data_ptr<int>(), nr.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {se, sr0, sr, ns, nr};
}

int64_t me_version() { return 1; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather", &me_gather, "token-major gather + input transform + per-row e4m3 quantization");
    m.def("gather2", &me_gather2, "single-pass gather (4 warps per row) + optional zeroing of out");
    m.def("gather_tok", &me_gather_tok, "opt-moe TG: one gathered e4m3 row per token (shared w13 suh) + optional zeroing");
    m.def("gateup", &me_gateup, "e4m3 gate/up GEMM + transforms + SwiGLU -> fp16");
    m.def("actq", &me_actq, "down-input transform + per-row e4m3 quantization");
    m.def("down", &me_down, "e4m3 down GEMM + transforms + weighted fp32 scatter-add");
    m.def("fused", &me_fused, "gate/up + actq + down in one persistent launch (interleaved items, dataflow flags)");
    m.def("exp_check", &me_exp_check, "P16: exp_prod vs (float) exp((double) a) over a float bit-pattern range");
    m.def("gather16", &me_gather16, "P16: production's fat-row gather (fp16 rows, rounded fp16 input scale)");
    m.def("pgu", &me_pgu, "P16: exllamav3 fm_gateup ported, exp_prod (64-row tiles from build_grouped_fat_tables)");
    m.def("pdn", &me_pdn, "P16: exllamav3 fm_down ported (64-row tables), segment chunks");
    m.def("p16b", &me_p16b, "P16: gather + gate/up + down, one persistent launch, 2 CTAs/SM, 64-row segments");
    m.def("seg_tables", &me_seg_tables, "128-row segment tables over all experts with rows");
    m.def("tile_rows", &me_tile_rows);
    m.def("version", &me_version);
}
