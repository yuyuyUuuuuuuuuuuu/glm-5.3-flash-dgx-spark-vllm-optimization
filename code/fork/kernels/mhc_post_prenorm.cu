// opt-kdamhc: fused mHC post-mapping + prenorm GEMM for PREFILL-sized token counts (one pass over the residual).
//
// Production's prefill branch of mhc_fused_post_pre_tilelang (num_tokens > 16) runs
//   mhc_post_tilelang   : residual_cur[t,j,h] = bf16( post[t,j]*x[t,h] + sum_k comb[t,k,j]*residual[t,k,h] )
//   tf32_hc_prenorm_gemm: out[t,n] = sum_{j,h} residual_cur[t,j,h] * fn[n, j*H+h] (tf32 MMA), sqrsum[t] = sum r^2
// i.e. it writes residual_cur (16 KB per token at H=4096, 4 streams) and immediately reads it back for the GEMM.
// This kernel computes the dot products while the new residual is still in registers/shared memory, so the GEMM's
// re-read disappears (DRAM traffic per call: residual + x read, residual_cur written, as mhc_post alone).
//
// Arithmetic:
//   residual_cur: bitwise mhc_post_tilelang's (v = fma(post, x, comb[0][j]*r0); v = fma(comb[k][j], rk, v) k = 1..3; bf16 RNE).
//   dot products, ROUND_A = false (default, "decode-consistent"): fp32 new_r (before the bf16 rounding) x fp32 fn with
//     fp32 FMA accumulation - the arithmetic class of the DECODE branch (mhc_fused_tilelang: acc += fn * new_r in fp32),
//     sqrsum from the fp32 new_r likewise. ROUND_A = true: the bf16-rounded residual_cur values (the prefill branch's
//     operand) with fp32 fn (no tf32 truncation). Accumulation order differs from both production kernels (fp32-order).
//
// Tiling: a CTA owns BM tokens (all 24 outputs, the full K = 4*H), 256 threads. Per h-chunk of HT positions:
//   phase 1 (all threads): new residual for BM x 4 x HT -> global bf16 + shared A[BM][4*HT] (fp32);
//                          fn chunk -> shared B[24][4*HT] (fp32, from L2);
//   phase 2 (all threads): thread (tokens q, q+BM/2; output triple; k segment) accumulates a 2x3 micro-tile with float4 loads.
// Final: the two k halves are summed in shared memory; out[0, t, n] and sqrsum[0, t] written.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace mpp {

constexpr int NOUT = 24;
constexpr int HC = 4;
constexpr int NTHR = 256;

template <int BM, int HT, bool ROUND_A, bool PF>
__global__ void __launch_bounds__(NTHR) post_prenorm_kernel(
    const float* __restrict__ comb,        // [M, 4, 4]  comb[t][k][j]
    const __nv_bfloat16* __restrict__ res, // [M, 4, H]
    const float* __restrict__ post,        // [M, 4]
    const __nv_bfloat16* __restrict__ x,   // [M, H]
    const float* __restrict__ fn,          // [24, 4*H]
    __nv_bfloat16* __restrict__ res_out,   // [M, 4, H]
    float* __restrict__ out,               // [M, 24]
    float* __restrict__ sqrsum,            // [M]
    int M, int H) {
  constexpr int KT = HC * HT;              // k values per chunk
  constexpr int KP = KT + 4;               // padded row (bank spread, keeps float4 alignment)
  constexpr int TPR = HT / 2;              // threads per token row in phase 1 (2 h per thread)
  constexpr int TOK_PER_PASS = NTHR / TPR;
  static_assert(BM % TOK_PER_PASS == 0, "BM must be a multiple of the phase-1 token rows per pass");
  constexpr int PASSES = BM / TOK_PER_PASS;
  static_assert(BM == 64 || BM == 32 || BM == 16, "micro-tile mapping assumes BM 16, 32 or 64");
  extern __shared__ float smem[];
  float* As = smem;                        // [BM][KP]
  float* Bs = smem + BM * KP;              // [NOUT][KP]
  __shared__ float s_post[BM][HC];
  __shared__ float s_comb[BM][HC][HC];
  __shared__ float s_red[NTHR / 2][6];
  __shared__ float s_sq[BM];

  const int tid = threadIdx.x;
  const int t0 = blockIdx.x * BM;
  for (int i = tid; i < BM * HC; i += NTHR) {
    int t = i / HC, j = i % HC;
    s_post[t][j] = (t0 + t < M) ? post[(size_t)(t0 + t) * HC + j] : 0.f;
  }
  for (int i = tid; i < BM * HC * HC; i += NTHR) {
    int t = i / (HC * HC), r = i % (HC * HC);
    s_comb[t][r / HC][r % HC] = (t0 + t < M) ? comb[(size_t)(t0 + t) * HC * HC + r] : 0.f;
  }
  if (tid < BM) s_sq[tid] = 0.f;
  // phase-1 mapping
  const int p_lane = tid % TPR;            // which 2-h pair of the chunk
  const int p_row = tid / TPR;             // token row within a pass
  // phase-2 mapping: BM/2 token pairs x 8 output triples x k halves
  constexpr int TPAIRS = BM / 2;
  const int q_t = tid % TPAIRS;
  const int q_n = (tid / TPAIRS) % 8;
  const int q_k = tid / (TPAIRS * 8);      // 0/1 for BM = 32 (2 halves), 0..3 for BM = 16
  constexpr int KSPLIT = NTHR / (TPAIRS * 8);
  constexpr int KSEG = KT / KSPLIT;
  float acc[2][3];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 3; ++b) acc[a][b] = 0.f;
  float sq[PASSES];
#pragma unroll
  for (int p = 0; p < PASSES; ++p) sq[p] = 0.f;
  __syncthreads();

  const int nchunks = H / HT;
  // PF (prefetch): chunk c+1's fn slice and residual/x values are loaded into registers right after chunk c's
  // phase 1, so their DRAM/L2 latency overlaps chunk c's phase 2 (FMA from shared memory) instead of sitting in
  // series with it. The arithmetic and every store are the non-PF kernel's (bitwise the same outputs).
  constexpr int FPT = (NOUT * KT / 4 + NTHR - 1) / NTHR;   // fn float4 per thread per chunk
  float4 pf_fn[PF ? FPT : 1];
  __nv_bfloat162 pf_r[PF ? PASSES : 1][HC];
  __nv_bfloat162 pf_x[PF ? PASSES : 1];
  auto prefetch = [&](int cc) {
    const int hh0 = cc * HT;
#pragma unroll
    for (int q = 0; q < FPT; ++q) {
      int i = tid + q * NTHR;
      if (i < NOUT * KT / 4) {
        int n = i / (KT / 4), r = (i % (KT / 4)) * 4;
        int j = r / HT, hh = r % HT;
        pf_fn[q] = *reinterpret_cast<const float4*>(fn + (size_t)n * HC * H + (size_t)j * H + hh0 + hh);
      }
    }
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      const int t = t0 + p * TOK_PER_PASS + p_row;
      const int h = hh0 + 2 * p_lane;
      if (t < M) {
        pf_x[p] = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)t * H + h);
#pragma unroll
        for (int k = 0; k < HC; ++k)
          pf_r[p][k] = *reinterpret_cast<const __nv_bfloat162*>(res + ((size_t)t * HC + k) * H + h);
      }
    }
  };
  if (PF) prefetch(0);
  for (int c = 0; c < nchunks; ++c) {
    const int h0 = c * HT;
    // ---- phase 1a: fn chunk -> Bs[n][j*HT + hh]
    if (PF) {
#pragma unroll
      for (int q = 0; q < FPT; ++q) {
        int i = tid + q * NTHR;
        if (i < NOUT * KT / 4) *reinterpret_cast<float4*>(Bs + (i / (KT / 4)) * KP + (i % (KT / 4)) * 4) = pf_fn[q];
      }
    } else {
      for (int i = tid; i < NOUT * KT / 4; i += NTHR) {
        int n = i / (KT / 4), r = (i % (KT / 4)) * 4;
        int j = r / HT, hh = r % HT;
        float4 v = *reinterpret_cast<const float4*>(fn + (size_t)n * HC * H + (size_t)j * H + h0 + hh);
        *reinterpret_cast<float4*>(Bs + n * KP + r) = v;
      }
    }
    // ---- phase 1b: new residual
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
      const int tl = p * TOK_PER_PASS + p_row;
      const int t = t0 + tl;
      const int h = h0 + 2 * p_lane;
      float nr[HC][2];
      if (t < M) {
        __nv_bfloat162 xv = PF ? pf_x[PF ? p : 0] : *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)t * H + h);
        float xf0 = __bfloat162float(xv.x), xf1 = __bfloat162float(xv.y);
        float rf[HC][2];
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          __nv_bfloat162 rv = PF ? pf_r[PF ? p : 0][k]
                                 : *reinterpret_cast<const __nv_bfloat162*>(res + ((size_t)t * HC + k) * H + h);
          rf[k][0] = __bfloat162float(rv.x); rf[k][1] = __bfloat162float(rv.y);
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float pj = s_post[tl][j];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            // mhc_post_tilelang's text is x = c*d; x = x + a0*b0; ... and LLVM contracts the first add as
            // fma(c, d, a0*b0) (the left multiply), the rest as fma(a_k, b_k, x): bitwise (post_arith_probe.py)
            float v = __fmaf_rn(pj, e ? xf1 : xf0, __fmul_rn(s_comb[tl][0][j], rf[0][e]));
#pragma unroll
            for (int k = 1; k < HC; ++k) v = __fmaf_rn(s_comb[tl][k][j], rf[k][e], v);
            nr[j][e] = v;
          }
          __nv_bfloat162 o = __floats2bfloat162_rn(nr[j][0], nr[j][1]);
          *reinterpret_cast<__nv_bfloat162*>(res_out + ((size_t)t * HC + j) * H + h) = o;
          if (ROUND_A) { nr[j][0] = __bfloat162float(o.x); nr[j][1] = __bfloat162float(o.y); }
          sq[p] = __fmaf_rn(nr[j][0], nr[j][0], sq[p]);
          sq[p] = __fmaf_rn(nr[j][1], nr[j][1], sq[p]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < HC; ++j) nr[j][0] = nr[j][1] = 0.f;
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
        *reinterpret_cast<float2*>(As + tl * KP + j * HT + 2 * p_lane) = make_float2(nr[j][0], nr[j][1]);
    }
    if (PF && c + 1 < nchunks) prefetch(c + 1);   // in flight during the barrier + phase 2
    __syncthreads();
    // ---- phase 2: 2 tokens x 3 outputs on this thread's k segment
    {
      const float* a0 = As + q_t * KP + q_k * KSEG;            // tokens q_t and q_t + BM/2 (bank spread: KP = 4 mod 32)
      const float* a1 = a0 + TPAIRS * KP;
      const float* b0 = Bs + (3 * q_n) * KP + q_k * KSEG;
      const float* b1 = b0 + KP;
      const float* b2 = b1 + KP;
#pragma unroll 4
      for (int kk = 0; kk < KSEG; kk += 4) {
        float4 va0 = *reinterpret_cast<const float4*>(a0 + kk);
        float4 va1 = *reinterpret_cast<const float4*>(a1 + kk);
        float4 vb0 = *reinterpret_cast<const float4*>(b0 + kk);
        float4 vb1 = *reinterpret_cast<const float4*>(b1 + kk);
        float4 vb2 = *reinterpret_cast<const float4*>(b2 + kk);
#define MPP_FMA4(ACC, A, B) ACC = __fmaf_rn(A.x, B.x, ACC); ACC = __fmaf_rn(A.y, B.y, ACC); \
                             ACC = __fmaf_rn(A.z, B.z, ACC); ACC = __fmaf_rn(A.w, B.w, ACC);
        MPP_FMA4(acc[0][0], va0, vb0) MPP_FMA4(acc[0][1], va0, vb1) MPP_FMA4(acc[0][2], va0, vb2)
        MPP_FMA4(acc[1][0], va1, vb0) MPP_FMA4(acc[1][1], va1, vb1) MPP_FMA4(acc[1][2], va1, vb2)
#undef MPP_FMA4
      }
    }
    __syncthreads();
  }
  // ---- sqrsum: the TPR lanes of a token row (contiguous threads) reduce, then one atomic-free write per pass
#pragma unroll
  for (int p = 0; p < PASSES; ++p) {
    float v = sq[p];
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, TPR < 32 ? TPR : 32);
    if (TPR > 32) {
      // not used for HT <= 64
    }
    if (p_lane == 0) s_sq[p * TOK_PER_PASS + p_row] = v;
  }
  // ---- reduce the k segments of the GEMM
  for (int s = KSPLIT - 1; s > 0; --s) {
    if (q_k == s) {
      int slot = tid - s * (TPAIRS * 8);
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 3; ++b) s_red[slot][a * 3 + b] = acc[a][b];
    }
    __syncthreads();
    if (q_k == s - 1) {
      int slot = tid - (s - 1) * (TPAIRS * 8);
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 3; ++b) acc[a][b] += s_red[slot][a * 3 + b];
    }
    __syncthreads();
  }
  if (q_k == 0) {
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      int t = t0 + q_t + a * TPAIRS;
      if (t < M) {
#pragma unroll
        for (int b = 0; b < 3; ++b) out[(size_t)t * NOUT + 3 * q_n + b] = acc[a][b];
      }
    }
  }
  if (tid < BM && t0 + tid < M) sqrsum[t0 + tid] = s_sq[tid];
}

template <int BM, int HT, bool RA, bool PF = false>
void launch(const torch::Tensor& comb, const torch::Tensor& res, const torch::Tensor& post, const torch::Tensor& x,
            const torch::Tensor& fn, torch::Tensor& res_out, torch::Tensor& out, torch::Tensor& sqrsum) {
  const int M = res.size(0), H = res.size(2);
  constexpr int KP = HC * HT + 4;
  size_t smem = (size_t)(BM + NOUT) * KP * sizeof(float);
  auto kern = post_prenorm_kernel<BM, HT, RA, PF>;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  dim3 grid((M + BM - 1) / BM);
  kern<<<grid, NTHR, smem, at::cuda::getCurrentCUDAStream()>>>(
      comb.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(res.data_ptr()), post.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), fn.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()), out.data_ptr<float>(), sqrsum.data_ptr<float>(), M, H);
}

}  // namespace mpp

// cfg: 0 = BM32/HT32, 1 = BM16/HT32, 2 = BM32/HT64(1 CTA/SM), 3 = BM16/HT64, 4 = BM64/HT32, 5 = BM64/HT16,
//      6 = BM32/HT16 (BM 64: no k split, a quarter of the fn L2 re-reads of BM 16); 7..10 = 1, 0, 3, 6 with the
//      next chunk's loads prefetched into registers (same outputs bit for bit); round_a: 0/1
void post_prenorm(torch::Tensor comb, torch::Tensor res, torch::Tensor post, torch::Tensor x, torch::Tensor fn,
                  torch::Tensor res_out, torch::Tensor out, torch::Tensor sqrsum, int64_t cfg, int64_t round_a) {
  TORCH_CHECK(res.is_contiguous() && x.is_contiguous() && comb.is_contiguous() && post.is_contiguous() &&
              fn.is_contiguous() && res_out.is_contiguous() && out.is_contiguous() && sqrsum.is_contiguous());
  TORCH_CHECK(res.dim() == 3 && res.size(1) == 4 && res.size(2) % 64 == 0);
  TORCH_CHECK(fn.size(0) == 24 && fn.size(1) == 4 * res.size(2));
  TORCH_CHECK(out.numel() == res.size(0) * 24 && sqrsum.numel() == res.size(0));
#define MPP_L(BM, HT) round_a ? mpp::launch<BM, HT, true>(comb, res, post, x, fn, res_out, out, sqrsum) \
                              : mpp::launch<BM, HT, false>(comb, res, post, x, fn, res_out, out, sqrsum)
#define MPP_LP(BM, HT) round_a ? mpp::launch<BM, HT, true, true>(comb, res, post, x, fn, res_out, out, sqrsum) \
                               : mpp::launch<BM, HT, false, true>(comb, res, post, x, fn, res_out, out, sqrsum)
  switch (cfg) {
    case 0: MPP_L(32, 32); break;
    case 1: MPP_L(16, 32); break;
    case 2: MPP_L(32, 64); break;
    case 3: MPP_L(16, 64); break;
    case 4: MPP_L(64, 32); break;
    case 5: MPP_L(64, 16); break;
    case 6: MPP_L(32, 16); break;
    case 7: MPP_LP(16, 32); break;   // = cfg 1 with register prefetch of the next chunk (bitwise cfg 1's outputs)
    case 8: MPP_LP(32, 32); break;   // = cfg 0 + prefetch
    case 9: MPP_LP(16, 64); break;   // = cfg 3 + prefetch
    case 10: MPP_LP(32, 16); break;  // = cfg 6 + prefetch
    default: TORCH_CHECK(false, "cfg");
  }
#undef MPP_L
#undef MPP_LP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("post_prenorm", &post_prenorm); }
