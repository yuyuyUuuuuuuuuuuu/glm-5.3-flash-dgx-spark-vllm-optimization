// GLM-5.3 sparse-MLA prefill attention kernel, exact variant (docs/MLA_PREFILL.md), sm_121a (GB10).
//
// One CTA per query token, all 32 heads of the rank (one KV gather serves every head), 64-key tiles, 2 KV stages.
// Operands and rounding follow production's FlashInfer FA2 path (mla.cuh, BatchMLAPagedAttentionKernel with fp8 KV):
//   K = V = bf16(fp8 e4m3) [* bf16(ckv_scale), rounded, when the scale is not 1], Q bf16, QK mma bf16 -> fp32,
//   p = ex2.approx(s * scale_log2 - m * scale_log2), P rounded to bf16 for the PV mma (bf16 -> fp32 accumulate),
//   row sum d from the fp32 p, out = o * (1 / d) rounded to bf16.
// What differs from FA2 is only fp32 summation order and the running-max schedule (64-key tiles instead of 16-key
// tiles, so the max used to offset p can differ, which changes which bf16 value p rounds to but not the error bound).
//
// Warp layout: 8 math warps = 2 head groups (wg, 16 heads) x 4 dim quarters (wq, 128 dims) + 1 IO warp.
//   QK: warp (wg, wq) computes the partial S[16 heads x 64 keys] over its 128 dims; Q stays in registers.
//       The 4 partials of a head group are summed through shared memory in a fixed order, so all 4 warps hold
//       bitwise the same S and each runs the softmax for the whole row itself (no cross-warp max exchange).
//   PV: warp (wg, wq) accumulates O[16 heads x its 128 dims] over the 64 keys; P comes straight from registers.
//   fp8 -> bf16 conversion happens in registers (no bf16 copy of the tile in shared memory).
//   IO warp: cp.async.bulk of 64 x 512-byte KV rows per tile (row stride 528 in smem), mbarrier full/empty pipeline.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#ifndef GLM53_MLA_DBG
#define GLM53_MLA_DBG 0   // timing ablations only (tests/probe_mla_ablation.py); 0 in every real build
#endif

namespace glm53_mla {
constexpr int DBG = GLM53_MLA_DBG;

constexpr int NH = 32;            // heads per rank
constexpr int D = 512;            // kv_lora_rank (NoPE: qk_rope_head_dim = 0)
constexpr int BK = 64;            // keys per tile
constexpr int KV_STRIDE = 528;    // smem bytes per key row (16-byte pad: conflict-free ldmatrix.trans)
constexpr int NSTAGE = 2;
constexpr int N_MATH_WARPS = 8;
constexpr int THREADS = (N_MATH_WARPS + 1) * 32;
constexpr int SMEM_KV = NSTAGE * BK * KV_STRIDE;                // 67584
constexpr int SMEM_SX = 2 * 4 * 8 * 32 * 16;                     // [wg][wq][tile j][lane] float4 = 32768
constexpr int SMEM_BAR = 4 * 8;
constexpr int SMEM_TOTAL = SMEM_KV + SMEM_SX + SMEM_BAR;       // 100384
static_assert(SMEM_TOTAL <= 101376, "smem over the 99 KB per-block limit of sm_121");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(smem_u32(b)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  uint32_t done = 0;
  const uint32_t a = smem_u32(b);
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(a), "r"(parity)
        : "memory");
  }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
#ifndef GLM53_MLA_L2HINT
#define GLM53_MLA_L2HINT 0   // 1: KV evict_last, Q/O evict_first (measured slower at 13.8k, equal at 100k)
#endif
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void bulk_g2s_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint4 ldg_stream(const void* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void stg_stream(void* p, uint2 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1,%2}, %3;\n" ::"l"(p), "r"(v.x), "r"(v.y), "l"(pol) : "memory");
}
template <int ID>
__device__ __forceinline__ void named_sync() {
  asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "n"(128) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ void sts128(void* p, float a, float b, float c, float d) {
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"r"(smem_u32(p)), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Two e4m3 codes sitting in the HIGH byte of each 16-bit half of t -> bf16x2 (exact).
// bf16 bits = sign | (e4m3 & 0x7f) << 4 is the value * 2^-120 (normals and subnormals alike, zero stays zero);
// the bf16 multiply by 2^120 restores it exactly. MUL2 is 2^120 (x the kv scale when that is a power of two).
template <bool EXTRA>
__device__ __forceinline__ uint32_t e4m3_hi_to_bf16x2(uint32_t t, uint32_t mul2, uint32_t scale2) {
  if constexpr (DBG & 1) return t;
  uint32_t u = (t >> 4) & 0x07F007F0u;
  uint32_t v = (t & 0x80008000u) | u;
  if constexpr (DBG & 128) v ^= mul2;   // ablation: integer op instead of the bf16 multiply
  else v = bf16x2_mul(v, mul2);
  if constexpr (EXTRA) v = bf16x2_mul(v, scale2);   // FA2: __hmul2(bf16(k), bf16(ckv_scale))
  return v;
}

struct Params {
  const __nv_bfloat16* q;      // [T, 32, 512], element strides q_st (token) / q_sh (head), unit inner stride
  const uint8_t* kv;           // [num_slots, 512] e4m3
  const int32_t* slots;        // [T, width] compacted slot ids, -1 past the valid count
  const int32_t* valid;        // [T]
  __nv_bfloat16* out;          // [T, 32, 512], strides o_st / o_sh
  long long q_st, q_sh, o_st, o_sh;
  int T;
  int width;
  float scale_log2;            // sm_scale * log2(e)
  uint32_t mul2;               // bf16x2 multiplier after the bit trick
  uint32_t scale2;             // bf16x2 ckv_scale (EXTRA only)
};

template <bool EXTRA>
__global__ void __launch_bounds__(THREADS, 1) mla_prefill_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* kvs = smem;                                                  // [NSTAGE][BK][KV_STRIDE]
  float4* sx = reinterpret_cast<float4*>(smem + SMEM_KV);               // [2][4][8][32]
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + SMEM_KV + SMEM_SX);
  uint64_t* full = bars;                                                // [2]
  uint64_t* empty = bars + 2;                                           // [2]

  const int tok = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    mbar_init(&full[0], 1);
    mbar_init(&full[1], 1);
    mbar_init(&empty[0], N_MATH_WARPS);
    mbar_init(&empty[1], N_MATH_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  int valid = __ldg(p.valid + tok);
  valid = valid < 0 ? 0 : (valid > p.width ? p.width : valid);
  const int ntiles = (valid + BK - 1) / BK;

  if (warp == N_MATH_WARPS) {
    if constexpr (DBG & 4) return;
    // ---------------------------------------------------------------- IO warp
    const int32_t* idx_row = p.slots + (size_t)tok * p.width;
    for (int ti = 0; ti < ntiles; ti++) {
      const int st = ti & 1;
      if (ti >= NSTAGE) mbar_wait(&empty[st], ((ti >> 1) - 1) & 1);
      if (lane == 0) mbar_expect_tx(&full[st], BK * D);
      __syncwarp();
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int k = lane + 32 * h;
        const int pos = ti * BK + k;
        int idx = pos < p.width ? __ldg(idx_row + pos) : 0;
        idx = idx < 0 ? 0 : idx;
        bulk_g2s(kvs + st * BK * KV_STRIDE + k * KV_STRIDE, p.kv + (size_t)idx * D, D, &full[st]);
      }
    }
    return;
  }

  // ------------------------------------------------------------------ math warps
  const int wg = warp >> 2, wq = warp & 3;
  const int gid = lane >> 2, tid = lane & 3;
  const int dbase = wq * 128;                 // this warp's dim quarter
  // Q A-fragments: row r in {gid, gid+8} of head group wg; k16 step kk covers dims dbase + 32*t + 4*kk + [0,4) over
  // t = 0..3 (thread tid owns t = tid). qr[r][w] = bf16 pair (dims dbase + 32*tid + 2w, +1), w = 0..15.
  uint32_t qr[2][16];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int h = wg * 16 + gid + 8 * r;
    const uint4* src = reinterpret_cast<const uint4*>(p.q + tok * p.q_st + h * p.q_sh + dbase + 32 * tid);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      uint4 v = __ldg(src + c);
      qr[r][4 * c + 0] = v.x;
      qr[r][4 * c + 1] = v.y;
      qr[r][4 * c + 2] = v.z;
      qr[r][4 * c + 3] = v.w;
    }
  }

  float o[16][4];
#pragma unroll
  for (int t = 0; t < 16; t++) o[t][0] = o[t][1] = o[t][2] = o[t][3] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY};
  float l_run[2] = {0.f, 0.f};
  const float c = p.scale_log2;
  float4* sx_mine = sx + ((wg * 4 + wq) * 8) * 32 + lane;
  const float4* sx_grp = sx + (wg * 4) * 8 * 32 + lane;

  for (int ti = 0; ti < ntiles; ti++) {
    const int st = ti & 1;
    if constexpr (!(DBG & 4)) mbar_wait(&full[st], (ti >> 1) & 1);
    const uint8_t* kv = kvs + st * BK * KV_STRIDE;

    // ---------------- QK partial over this warp's 128 dims: s[j] = n8 tile j (keys 8j + gid as B column)
    float s[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint8_t* rowp = kv + (8 * j + gid) * KV_STRIDE + dbase + 32 * tid;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint4 kb = lds128(rowp + 16 * h);
        const uint32_t w4[4] = {kb.x, kb.y, kb.z, kb.w};
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
          const int kk = 4 * h + q4;            // k16 step: dims dbase + 32*tid + 4*kk + [0,4)
          const uint32_t x = w4[q4];            // bytes: dims +0,+1,+2,+3
          // b0 = (dim +0, dim +1), b1 = (dim +2, dim +3): place bytes into the high byte of each half.
          const uint32_t t0 = __byte_perm(x, 0u, 0x1404);  // [0, b0, 0, b1]
          const uint32_t t1 = __byte_perm(x, 0u, 0x3424);  // [0, b2, 0, b3]
          const uint32_t b0 = e4m3_hi_to_bf16x2<EXTRA>(t0, p.mul2, p.scale2);
          const uint32_t b1 = e4m3_hi_to_bf16x2<EXTRA>(t1, p.mul2, p.scale2);
          if constexpr (DBG & 8) { s[j][0] += __uint_as_float(b0 ^ b1); }
          else mma_bf16(s[j], qr[0][2 * kk], qr[1][2 * kk], qr[0][2 * kk + 1], qr[1][2 * kk + 1], b0, b1);
        }
      }
    }

    // ---------------- sum the 4 dim-quarter partials of this head group in a fixed order (bitwise equal in all 4)
#pragma unroll
    for (int j = 0; j < 8; j++) sts128(sx_mine + j * 32, s[j][0], s[j][1], s[j][2], s[j][3]);
    if constexpr (!(DBG & 2)) { if (wg == 0) named_sync<1>(); else named_sync<2>(); }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc[4];
#pragma unroll
      for (int w = 0; w < 4; w++) {
        float4 v;
        if (w == wq) {
          v = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        } else {
          const uint4 u = lds128(sx_grp + (w * 8 + j) * 32);
          v = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w));
        }
        if (w == 0) {
          acc[0] = v.x; acc[1] = v.y; acc[2] = v.z; acc[3] = v.w;
        } else {
          acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
        }
      }
      s[j][0] = acc[0]; s[j][1] = acc[1]; s[j][2] = acc[2]; s[j][3] = acc[3];
    }
    if constexpr (!(DBG & 2)) { if (wg == 0) named_sync<3>(); else named_sync<4>(); }

    // ---------------- mask keys past the row's valid count (compacted prefix)
    const int kbase = ti * BK + 2 * tid;
    if (ti * BK + BK > valid) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        if (kbase + 8 * j >= valid) { s[j][0] = -INFINITY; s[j][2] = -INFINITY; }
        if (kbase + 8 * j + 1 >= valid) { s[j][1] = -INFINITY; s[j][3] = -INFINITY; }
      }
    }

    // ---------------- online softmax (rows gid: elements 0,1; gid + 8: elements 2,3)
    float mx[2] = {m_run[0], m_run[1]};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
      mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
    }
    float alpha[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      alpha[r] = ex2(m_run[r] * c - mx[r] * c);
      m_run[r] = mx[r];
      l_run[r] *= alpha[r];
    }
#pragma unroll
    for (int t = 0; t < 16; t++) {
      o[t][0] *= alpha[0]; o[t][1] *= alpha[0];
      o[t][2] *= alpha[1]; o[t][3] *= alpha[1];
    }
    const float mc0 = mx[0] * c, mc1 = mx[1] * c;
    uint32_t pa[4][4];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float p0 = ex2(s[j][0] * c - mc0), p1 = ex2(s[j][1] * c - mc0);
      const float p2 = ex2(s[j][2] * c - mc1), p3 = ex2(s[j][3] * c - mc1);
      l_run[0] += p0 + p1;
      l_run[1] += p2 + p3;
      // A fragment of k16 step q = j/2: tile 2q -> a0 (row gid), a1 (row gid+8); tile 2q+1 -> a2, a3
      pa[j >> 1][(j & 1) * 2 + 0] = pack_bf16(p0, p1);
      pa[j >> 1][(j & 1) * 2 + 1] = pack_bf16(p2, p3);
    }

    // ---------------- PV over this warp's 128 dims: 4 dim blocks of 32 x 4 k16 steps
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int key = 16 * q + (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
      for (int b = 0; b < 4; b++) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_trans(r0, r1, r2, r3, kv + key * KV_STRIDE + dbase + 32 * b + 16 * (lane >> 4));
        // r = [V[k0][2g], V[k0][2g+1], V[k0+1][2g], V[k0+1][2g+1]] (k0 = 16q + 2tid (+8 for r1/r3), dims +16 r2/r3)
        const uint32_t e0 = e4m3_hi_to_bf16x2<EXTRA>(r0 << 8, p.mul2, p.scale2);
        const uint32_t o0 = e4m3_hi_to_bf16x2<EXTRA>(r0, p.mul2, p.scale2);
        const uint32_t e1 = e4m3_hi_to_bf16x2<EXTRA>(r1 << 8, p.mul2, p.scale2);
        const uint32_t o1 = e4m3_hi_to_bf16x2<EXTRA>(r1, p.mul2, p.scale2);
        const uint32_t e2 = e4m3_hi_to_bf16x2<EXTRA>(r2 << 8, p.mul2, p.scale2);
        const uint32_t o2 = e4m3_hi_to_bf16x2<EXTRA>(r2, p.mul2, p.scale2);
        const uint32_t e3 = e4m3_hi_to_bf16x2<EXTRA>(r3 << 8, p.mul2, p.scale2);
        const uint32_t o3 = e4m3_hi_to_bf16x2<EXTRA>(r3, p.mul2, p.scale2);
        if constexpr (DBG & 16) {
          o[4 * b][0] += __uint_as_float(e0 ^ e1 ^ o0 ^ o1 ^ e2 ^ e3 ^ o2 ^ o3 ^ pa[q][0]);
          continue;
        }
        mma_bf16(o[4 * b + 0], pa[q][0], pa[q][1], pa[q][2], pa[q][3], e0, e1);   // dims 32b + 2g
        mma_bf16(o[4 * b + 1], pa[q][0], pa[q][1], pa[q][2], pa[q][3], o0, o1);   // dims 32b + 2g + 1
        mma_bf16(o[4 * b + 2], pa[q][0], pa[q][1], pa[q][2], pa[q][3], e2, e3);   // dims 32b + 16 + 2g
        mma_bf16(o[4 * b + 3], pa[q][0], pa[q][1], pa[q][2], pa[q][3], o2, o3);   // dims 32b + 16 + 2g + 1
      }
    }
    __syncwarp();
    if constexpr (!(DBG & 4)) { if (lane == 0) mbar_arrive(&empty[st]); }
  }

  // ------------------------------------------------------------------ epilogue
#pragma unroll
  for (int r = 0; r < 2; r++) {
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 1);
    l_run[r] += __shfl_xor_sync(0xffffffffu, l_run[r], 2);
  }
  const float inv0 = l_run[0] > 0.f ? 1.f / l_run[0] : 0.f;
  const float inv1 = l_run[1] > 0.f ? 1.f / l_run[1] : 0.f;
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int h = wg * 16 + gid + 8 * r;
    __nv_bfloat16* dst = p.out + tok * p.o_st + h * p.o_sh + dbase;
    const float inv = r ? inv1 : inv0;
#pragma unroll
    for (int b = 0; b < 4; b++) {
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const float* E = o[4 * b + 2 * hh];
        const float* O = o[4 * b + 2 * hh + 1];
        // dims 32b + 16hh + 4tid + {0: E[2r], 1: O[2r], 2: E[2r+1], 3: O[2r+1]}
        uint2 v;
        v.x = pack_bf16(E[2 * r] * inv, O[2 * r] * inv);
        v.y = pack_bf16(E[2 * r + 1] * inv, O[2 * r + 1] * inv);
        *reinterpret_cast<uint2*>(dst + 32 * b + 16 * hh + 4 * tid) = v;
      }
    }
  }
}


// ================================================================================================================
// v3 layout: every K/V fp8 value is converted to bf16 once per CTA and feeds both 16-head groups (m32 per warp).
//   Each 64-key tile is processed as two 32-key sub-tiles.
//   QK : warp w = (wq = w & 3, kh = w >> 2) computes the partial S[32 heads x 16 keys (half kh of the sub-tile)]
//        over dims [128 wq, +128); Q (32 heads x its 128 dims) stays in registers (64 regs).
//   S  : the 4 dim-quarter partials are summed in a fixed order by the softmax owner: warp w owns heads
//        [4w, 4w+4) x 32 keys (4 keys per lane), keeps the running max / row sum of those heads, writes P (bf16)
//        and the rescale factor alpha to shared memory.
//   PV : warp w accumulates O[32 heads x dims [64 w, +64)] (64 regs) over the sub-tile's 32 keys.
//   Two CTA-wide barriers per sub-tile (partials written; P written).
// Numerics are those of v2 (FA2 operand roundings); the running max advances per 32 keys.
constexpr int V3_SUB = 32;
constexpr int V3_THREADS = 12 * 32;                               // 8 math warps + 4 IO warps (1 per SMSP)
constexpr int V3_NST = 4;                                         // KV ring: 4 stages x 32 keys (same bytes as 2 x 64)
constexpr int V3_SMEM_SX = 2 * 4 * 32 * 16 * 4;                  // [kh][wq][head][16 keys] fp32 = 16384
constexpr int V3_SMEM_P = 32 * 32 * 2;                           // [head][32 keys] bf16 = 2048
constexpr int V3_SMEM_TOTAL = SMEM_KV + V3_SMEM_SX + 2 * V3_SMEM_P + 3 * 32 * 4 + 2 * 4 * 8 + 8;   // 88520
static_assert(V3_SMEM_TOTAL <= 101376, "v3 smem over the 99 KB per-block limit of sm_121");

__device__ __forceinline__ void named_sync_all() {                // the 8 math warps
  asm volatile("bar.sync %0, %1;\n" ::"n"(1), "n"(256) : "memory");
}
__device__ __forceinline__ void sts64(void* p, uint32_t a, uint32_t b) {
  asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" ::"r"(smem_u32(p)), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}

// PERSIST: gridDim.x <= #SMs CTAs walk the tokens tok = blockIdx.x + k * gridDim.x; the KV stage ring and the
// sub-tile mbarrier phases continue across tokens, so the IO warps gather the next token's first tiles while the
// math warps finish the current one, and the next token's Q is loaded during the last sub-tile's softmax / PV.
template <bool EXTRA, bool PERSIST>
__global__ void __launch_bounds__(V3_THREADS, 1) mla_prefill_v3_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* kvs = smem;                                                  // [NSTAGE][BK][KV_STRIDE]
  float* sx = reinterpret_cast<float*>(smem + SMEM_KV);                 // [2][4][32][16]
  uint8_t* pbuf = smem + SMEM_KV + V3_SMEM_SX;                          // [2][32][64 B], 16-B chunks swizzled
  float* alpha_s = reinterpret_cast<float*>(pbuf + 2 * V3_SMEM_P);      // [2][32]
  float* lsum_s = alpha_s + 64;                                         // [32]
  uint64_t* bars = reinterpret_cast<uint64_t*>(lsum_s + 32);
  uint64_t* full = bars;                                                // [V3_NST] 32-key KV stages
  uint64_t* empty = bars + V3_NST;                                      // [V3_NST]
  uint64_t* sx_full = bars + 2 * V3_NST;                                // 8 math-warp arrivals per sub-tile

  const int tok0 = blockIdx.x;
  const int tstep = PERSIST ? (int)gridDim.x : p.T;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < V3_NST; i++) {
      mbar_init(&full[i], 1);
      mbar_init(&empty[i], N_MATH_WARPS);
    }
    mbar_init(sx_full, N_MATH_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  auto row_valid = [&](int tok) {
    int v = __ldg(p.valid + tok);
    v = v < 0 ? 0 : (v > p.width ? p.width : v);
    return (DBG & 32) ? 0 : v;
  };

  if (warp >= N_MATH_WARPS) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    if constexpr (DBG & 4) return;
    // IO warps: threads 0..63 each copy one 512-byte KV row per tile
    const int io = threadIdx.x - N_MATH_WARPS * 32;
    const uint64_t pol_kv = l2_policy_evict_last();
    int G = 0;                                           // sub-tiles issued by this CTA so far (stage ring position)
    for (int tok = tok0; tok < p.T; tok += tstep) {
    const int nsub = (row_valid(tok) + V3_SUB - 1) / V3_SUB;
    const int32_t* idx_row = p.slots + (size_t)tok * p.width;
    for (int ti = 0; ti < nsub; ti++, G++) {
      const int st = G & (V3_NST - 1);
      if (G >= V3_NST) mbar_wait(&empty[st], ((G / V3_NST) - 1) & 1);
      if (io == 0) mbar_expect_tx(&full[st], V3_SUB * D);
      if (io < V3_SUB) {
        const int pos = ti * V3_SUB + io;
        int idx = pos < p.width ? __ldg(idx_row + pos) : 0;
        idx = idx < 0 ? 0 : idx;
        if constexpr (GLM53_MLA_L2HINT)
          bulk_g2s_hint(kvs + st * V3_SUB * KV_STRIDE + io * KV_STRIDE, p.kv + (size_t)idx * D, D, &full[st], pol_kv);
        else
          bulk_g2s(kvs + st * V3_SUB * KV_STRIDE + io * KV_STRIDE, p.kv + (size_t)idx * D, D, &full[st]);
      }
    }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));

  const int gid = lane >> 2, tid = lane & 3;
  const int wq = warp & 3, kh = warp >> 2;
  const int qdim = 128 * wq + 32 * tid;                  // QK: this thread's 32 dims
  const int pdim = 64 * warp;                            // PV: this warp's 64 dims
  const int sh = 4 * warp + (lane >> 3);                 // softmax: head
  const int skg = lane & 7;                              // softmax: keys 4*skg .. +3 of the sub-tile

  // Q registers: qr[g][r][w] = bf16 pair (dims qdim + 2w, +1) of head 16 g + gid + 8 r
  const uint64_t pol_stream = l2_policy_evict_first();
  uint32_t qr[2][2][16];
  auto load_q = [&](int tok) {
#pragma unroll
    for (int g = 0; g < 2; g++) {
#pragma unroll
      for (int r = 0; r < 2; r++) {
        const int h = 16 * g + gid + 8 * r;
        const uint4* src = reinterpret_cast<const uint4*>(p.q + tok * p.q_st + h * p.q_sh + qdim);
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++) {
          const uint4 v = GLM53_MLA_L2HINT ? ldg_stream(src + c4, pol_stream) : __ldg(src + c4);
          qr[g][r][4 * c4 + 0] = v.x;
          qr[g][r][4 * c4 + 1] = v.y;
          qr[g][r][4 * c4 + 2] = v.z;
          qr[g][r][4 * c4 + 3] = v.w;
        }
      }
    }
  };
  load_q(tok0);
  float o[2][8][4];
  int S0 = 0;                                            // sub-tiles consumed before this token (ring position)
  for (int tok = tok0; tok < p.T; tok += tstep) {
  named_sync_all();                                      // lsum_s / P / alpha of the previous token are read
  const int valid = row_valid(tok);
#pragma unroll
  for (int g = 0; g < 2; g++)
#pragma unroll
    for (int t = 0; t < 8; t++) o[g][t][0] = o[g][t][1] = o[g][t][2] = o[g][t][3] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;                  // softmax owner state for head sh (8 lanes share it)
  const float c = p.scale_log2;
  float* sx_w = sx + (size_t)((kh * 4 + wq) * 32) * 16;

  // ---------------------------------------------------------------- QK partial of sub-tile n -> sx, arrive sx_full
  auto qk_sub = [&](int n) {
    const uint8_t* kv = kvs + ((S0 + n) & (V3_NST - 1)) * V3_SUB * KV_STRIDE;
    float s[2][2][4];
#pragma unroll
    for (int g = 0; g < 2; g++)
#pragma unroll
      for (int j = 0; j < 2; j++) s[g][j][0] = s[g][j][1] = s[g][j][2] = s[g][j][3] = 0.f;
    uint32_t kw[2][8];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const uint8_t* rowp = kv + (16 * kh + 8 * j + gid) * KV_STRIDE + qdim;
      const uint4 a = lds128(rowp), b = lds128(rowp + 16);
      kw[j][0] = a.x; kw[j][1] = a.y; kw[j][2] = a.z; kw[j][3] = a.w;
      kw[j][4] = b.x; kw[j][5] = b.y; kw[j][6] = b.z; kw[j][7] = b.w;
    }
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const uint32_t x = kw[j][kk];
        const uint32_t b0 = e4m3_hi_to_bf16x2<EXTRA>(__byte_perm(x, 0u, 0x1404), p.mul2, p.scale2);
        const uint32_t b1 = e4m3_hi_to_bf16x2<EXTRA>(__byte_perm(x, 0u, 0x3424), p.mul2, p.scale2);
#pragma unroll
        for (int g = 0; g < 2; g++) {
          if constexpr (DBG & 8) { s[g][j][0] += __uint_as_float(b0 ^ b1); }
          else mma_bf16(s[g][j], qr[g][0][2 * kk], qr[g][1][2 * kk], qr[g][0][2 * kk + 1], qr[g][1][2 * kk + 1],
                        b0, b1);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < 2; g++)
#pragma unroll
      for (int j = 0; j < 2; j++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
          const int h = 16 * g + gid + 8 * r;
          const int slot = (4 * j + tid) ^ (((h >> 1) & 1) << 2);
          sts64(sx_w + h * 16 + 2 * slot, __float_as_uint(s[g][j][2 * r]), __float_as_uint(s[g][j][2 * r + 1]));
        }
    __syncwarp();
    if (lane == 0) mbar_arrive(sx_full);
  };

  // ---------------------------------------------------------------- softmax owner: head sh, keys 4 skg .. +3 of n
  auto softmax_sub = [&](int n) {
    if constexpr (!(DBG & 2)) mbar_wait(sx_full, (S0 + n) & 1);
    if constexpr (DBG & 64) {
      uint8_t* pb = pbuf + (n & 1) * V3_SMEM_P;
      sts64(pb + sh * 64 + (skg >> 1) * 16 + (skg & 1) * 8, 0x3f803f80u, 0x3f803f80u);
      if (skg == 0) alpha_s[(n & 1) * 32 + sh] = 1.f;
      return;
    }
    const int khr = skg >> 2;
    const int slot = (2 * (skg & 3)) ^ (((sh >> 1) & 1) << 2);
    float v[4];
#pragma unroll
    for (int w4 = 0; w4 < 4; w4++) {
      const uint4 u = lds128(sx + ((khr * 4 + w4) * 32 + sh) * 16 + 2 * slot);
      if (w4 == 0) {
        v[0] = __uint_as_float(u.x); v[1] = __uint_as_float(u.y);
        v[2] = __uint_as_float(u.z); v[3] = __uint_as_float(u.w);
      } else {
        v[0] += __uint_as_float(u.x); v[1] += __uint_as_float(u.y);
        v[2] += __uint_as_float(u.z); v[3] += __uint_as_float(u.w);
      }
    }
    const int kpos = V3_SUB * n + 4 * skg;
#pragma unroll
    for (int i = 0; i < 4; i++)
      if (kpos + i >= valid) v[i] = -INFINITY;
    float mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    const float m_new = fmaxf(m_run, mx);
    const float alpha = ex2(m_run * c - m_new * c);
    const float mc = m_new * c;
    const float p0 = ex2(v[0] * c - mc), p1 = ex2(v[1] * c - mc);
    const float p2 = ex2(v[2] * c - mc), p3 = ex2(v[3] * c - mc);
    l_run = l_run * alpha + ((p0 + p1) + (p2 + p3));
    m_run = m_new;
    uint8_t* pb = pbuf + (n & 1) * V3_SMEM_P;
    const int chunk = (skg >> 1) ^ ((sh >> 1) & 3);
    sts64(pb + sh * 64 + chunk * 16 + (skg & 1) * 8, pack_bf16(p0, p1), pack_bf16(p2, p3));
    if (skg == 0) alpha_s[(n & 1) * 32 + sh] = alpha;
  };

  // ---------------------------------------------------------------- PV of sub-tile n: O[32 heads x pdim .. +64]
  auto pv_sub = [&](int n) {
    {
      const float* al_s = alpha_s + (n & 1) * 32;
      float al[2][2];
#pragma unroll
      for (int g = 0; g < 2; g++) {
        al[g][0] = al_s[16 * g + gid];
        al[g][1] = al_s[16 * g + gid + 8];
      }
      const bool resc = (al[0][0] != 1.f) | (al[0][1] != 1.f) | (al[1][0] != 1.f) | (al[1][1] != 1.f);
      if (__any_sync(0xffffffffu, resc)) {
#pragma unroll
        for (int g = 0; g < 2; g++)
#pragma unroll
          for (int t = 0; t < 8; t++) {
            o[g][t][0] *= al[g][0]; o[g][t][1] *= al[g][0];
            o[g][t][2] *= al[g][1]; o[g][t][3] *= al[g][1];
          }
      }
    }
    const uint8_t* kv = kvs + ((S0 + n) & (V3_NST - 1)) * V3_SUB * KV_STRIDE;
    const uint8_t* pb = pbuf + (n & 1) * V3_SMEM_P;
#pragma unroll
    for (int q = 0; q < 2; q++) {
      uint32_t pa[2][4];
#pragma unroll
      for (int g = 0; g < 2; g++) {
        const int row = 16 * g + (lane & 7) + 8 * ((lane >> 3) & 1);
        const int chunk = (2 * q + (lane >> 4)) ^ ((row >> 1) & 3);
        ldsm_x4(pa[g][0], pa[g][1], pa[g][2], pa[g][3], pb + row * 64 + chunk * 16);
      }
      const int key = 16 * q + (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
      for (int b = 0; b < 2; b++) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_trans(r0, r1, r2, r3, kv + key * KV_STRIDE + pdim + 32 * b + 16 * (lane >> 4));
        if constexpr (DBG & 256) {     // opt-dense ablation: e4m3-PV cost model (no V conversion, half the PV MMAs)
          if (q == 0) {
#pragma unroll
            for (int g = 0; g < 2; g++) {
              mma_bf16(o[g][4 * b + 0], pa[g][0], pa[g][1], pa[g][2], pa[g][3], r0, r1);
              mma_bf16(o[g][4 * b + 1], pa[g][0], pa[g][1], pa[g][2], pa[g][3], r2, r3);
              mma_bf16(o[g][4 * b + 2], pa[g][0], pa[g][1], pa[g][2], pa[g][3], r1, r0);
              mma_bf16(o[g][4 * b + 3], pa[g][0], pa[g][1], pa[g][2], pa[g][3], r3, r2);
            }
          }
          continue;
        }
        const uint32_t e0 = e4m3_hi_to_bf16x2<EXTRA>(r0 << 8, p.mul2, p.scale2);
        const uint32_t o0 = e4m3_hi_to_bf16x2<EXTRA>(r0, p.mul2, p.scale2);
        const uint32_t e1 = e4m3_hi_to_bf16x2<EXTRA>(r1 << 8, p.mul2, p.scale2);
        const uint32_t o1 = e4m3_hi_to_bf16x2<EXTRA>(r1, p.mul2, p.scale2);
        const uint32_t e2 = e4m3_hi_to_bf16x2<EXTRA>(r2 << 8, p.mul2, p.scale2);
        const uint32_t o2 = e4m3_hi_to_bf16x2<EXTRA>(r2, p.mul2, p.scale2);
        const uint32_t e3 = e4m3_hi_to_bf16x2<EXTRA>(r3 << 8, p.mul2, p.scale2);
        const uint32_t o3 = e4m3_hi_to_bf16x2<EXTRA>(r3, p.mul2, p.scale2);
#pragma unroll
        for (int g = 0; g < 2; g++) {
          if constexpr (DBG & 16) {
            o[g][4 * b][0] += __uint_as_float(e0 ^ e1 ^ o0 ^ o1 ^ e2 ^ e3 ^ o2 ^ o3 ^ pa[g][0]);
            continue;
          }
          mma_bf16(o[g][4 * b + 0], pa[g][0], pa[g][1], pa[g][2], pa[g][3], e0, e1);
          mma_bf16(o[g][4 * b + 1], pa[g][0], pa[g][1], pa[g][2], pa[g][3], o0, o1);
          mma_bf16(o[g][4 * b + 2], pa[g][0], pa[g][1], pa[g][2], pa[g][3], e2, e3);
          mma_bf16(o[g][4 * b + 3], pa[g][0], pa[g][1], pa[g][2], pa[g][3], o2, o3);
        }
      }
    }
  };
  auto release_sub = [&](int n) {
    __syncwarp();
    if constexpr (!(DBG & 4)) { if (lane == 0) mbar_arrive(&empty[(S0 + n) & (V3_NST - 1)]); }
  };
  auto wait_sub = [&](int n) {
    if constexpr (!(DBG & 4)) mbar_wait(&full[(S0 + n) & (V3_NST - 1)], ((S0 + n) / V3_NST) & 1);
  };

  // Software pipeline over 32-key sub-tiles: QK(n) runs before PV(n - 1), so the wait for every warp's partial S
  // of sub-tile n overlaps the PV MMAs of sub-tile n - 1.
  const int nsub_total = (valid + V3_SUB - 1) / V3_SUB;
  if (nsub_total > 0) {
    wait_sub(0);
    qk_sub(0);
    softmax_sub(0);
    if constexpr (!(DBG & 2)) named_sync_all();
#pragma unroll 1
    for (int n = 1; n < nsub_total; n++) {
      wait_sub(n);
      qk_sub(n);
      pv_sub(n - 1);
      release_sub(n - 1);
      softmax_sub(n);
      if constexpr (!(DBG & 2)) named_sync_all();
    }
  }
  if (PERSIST && tok + tstep < p.T) load_q(tok + tstep);   // Q of this token is dead after its last QK
  if (nsub_total > 0) {
    pv_sub(nsub_total - 1);
    release_sub(nsub_total - 1);
  }

  // ------------------------------------------------------------------ epilogue
  l_run += __shfl_xor_sync(0xffffffffu, l_run, 1);
  l_run += __shfl_xor_sync(0xffffffffu, l_run, 2);
  l_run += __shfl_xor_sync(0xffffffffu, l_run, 4);
  if (skg == 0) lsum_s[sh] = l_run;
  named_sync_all();
#pragma unroll
  for (int g = 0; g < 2; g++) {
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int h = 16 * g + gid + 8 * r;
      const float l = lsum_s[h];
      const float inv = l > 0.f ? 1.f / l : 0.f;
      __nv_bfloat16* dst = p.out + tok * p.o_st + h * p.o_sh + pdim;
#pragma unroll
      for (int b = 0; b < 2; b++) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const float* E = o[g][4 * b + 2 * hh];
          const float* O = o[g][4 * b + 2 * hh + 1];
          uint2 v;
          v.x = pack_bf16(E[2 * r] * inv, O[2 * r] * inv);
          v.y = pack_bf16(E[2 * r + 1] * inv, O[2 * r + 1] * inv);
          if constexpr (GLM53_MLA_L2HINT) stg_stream(dst + 32 * b + 16 * hh + 4 * tid, v, pol_stream);
          else *reinterpret_cast<uint2*>(dst + 32 * b + 16 * hh + 4 * tid) = v;
        }
      }
    }
  }
  S0 += nsub_total;
  }
}

}  // namespace glm53_mla
