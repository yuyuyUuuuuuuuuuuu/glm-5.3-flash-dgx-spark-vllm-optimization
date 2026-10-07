// PDL (programmatic dependent launch) on GB10 inside CUDA graphs: what a kernel boundary costs and what PDL saves.
// nvcc -O3 -arch=sm_121a -o tests/decode6/mb_pdl tests/decode6/mb_pdl.cu
// Run inside the production image: tests/gpu_run.sh tests/decode6/mb_pdl
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" :::); }

// tiny kernel: every CTA reads a few values of the previous output and writes its own (a latency-bound "small op")
template <bool WAIT, bool TRIG>
__global__ void tiny(const float* __restrict__ in, float* __restrict__ out, int n, int spin) {
  if (TRIG) gdc_launch();
  if (WAIT) gdc_wait();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = in[i % n];
  for (int s = 0; s < spin; ++s) v = v * 0.999f + 0.001f;
  if (i < n) out[i] = v + 1.0f;
}

// streaming GEMV: y[m][n] = sum_k x[m][k] * W[n][k], W bf16 [N][K], M <= 8, one warp per output row n (grid-stride)
// PF: issue the first PFV 16-byte weight loads per lane before griddepcontrol.wait (weights do not depend on the
// predecessor), then wait, then read x.
template <int M, bool WAIT, int PFV>
__global__ void __launch_bounds__(256) gemv(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x,
                                            float* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarps = (gridDim.x * blockDim.x) >> 5;
  const int KV = K / 8;  // 16-byte vectors per row
  uint4 pf[PFV > 0 ? PFV : 1];
  int n0 = warp;
  if (PFV > 0 && n0 < N) {
#pragma unroll
    for (int p = 0; p < PFV; ++p) {
      int kv = lane + 32 * p;
      if (kv < KV) pf[p] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)n0 * K) + kv);
    }
  }
  if (WAIT) gdc_wait();
  for (int n = n0; n < N; n += nwarps) {
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int kv = lane; kv < KV; kv += 32) {
      uint4 w;
      int p = (kv - lane) / 32;
      if (PFV > 0 && n == n0 && p < PFV) w = pf[p < PFV ? p : 0];
      else w = __ldg(reinterpret_cast<const uint4*>(W + (size_t)n * K) + kv);
      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float wf = __bfloat162float(wb[j]);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] += wf * x[m * K + kv * 8 + j];
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[m];
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) y[m * N + n] = v;
    }
  }
}

template <typename Kern, typename... Args>
void launch(bool pdl, dim3 g, dim3 b, cudaStream_t s, Kern k, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = 0; cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl ? at : nullptr; cfg.numAttrs = pdl ? 1 : 0;
  CK(cudaLaunchKernelEx(&cfg, k, args...));
}

template <typename F>
float time_graph(cudaStream_t s, F body, int reps = 20) {
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  body();
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> v;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
  return v[v.size() / 2];
}

int main() {
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int n = 48 * 128;
  float *b0, *b1; CK(cudaMalloc(&b0, n * 4 * 2)); CK(cudaMalloc(&b1, n * 4 * 2));
  CK(cudaMemset(b0, 0, n * 8)); CK(cudaMemset(b1, 0, n * 8));
  const int CH = 400;
  printf("== chain of %d tiny kernels (48 CTAs x 128 thr), per kernel us\n", CH);
  for (int spin : {0, 200, 1000}) {
    float t_base = time_graph(s, [&] { for (int i = 0; i < CH; ++i) launch(false, 48, 128, s, tiny<false, false>, (i & 1) ? b1 : b0, (i & 1) ? b0 : b1, n, spin); });
    float t_pdl = time_graph(s, [&] { for (int i = 0; i < CH; ++i) launch(i > 0, 48, 128, s, tiny<true, false>, (i & 1) ? b1 : b0, (i & 1) ? b0 : b1, n, spin); });
    float t_pdlt = time_graph(s, [&] { for (int i = 0; i < CH; ++i) launch(i > 0, 48, 128, s, tiny<true, true>, (i & 1) ? b1 : b0, (i & 1) ? b0 : b1, n, spin); });
    printf("spin %4d: base %.3f  pdl(wait) %.3f  pdl(wait+early trigger) %.3f   us/kernel\n", spin,
           t_base * 1000 / CH, t_pdl * 1000 / CH, t_pdlt * 1000 / CH);
  }
  // grid sizes like production small ops
  for (int ctas : {5, 40, 480}) {
    float t_base = time_graph(s, [&] { for (int i = 0; i < CH; ++i) launch(false, ctas, 128, s, tiny<false, false>, (i & 1) ? b1 : b0, (i & 1) ? b0 : b1, n, 0); });
    float t_pdl = time_graph(s, [&] { for (int i = 0; i < CH; ++i) launch(i > 0, ctas, 128, s, tiny<true, true>, (i & 1) ? b1 : b0, (i & 1) ? b0 : b1, n, 0); });
    printf("grid %3d spin 0: base %.3f  pdl(wait+trigger) %.3f us/kernel\n", ctas, t_base * 1000 / CH, t_pdl * 1000 / CH);
  }

  // small op -> GEMV, 30 distinct weights (cold, 4096 x K bf16), x M=5
  const int K = 2048, N = 2048, L = 30;  // 8 MiB per weight
  std::vector<__nv_bfloat16*> Ws(L);
  for (auto& w : Ws) { CK(cudaMalloc(&w, (size_t)N * K * 2)); CK(cudaMemset(w, 0, (size_t)N * K * 2)); }
  float *x, *y; CK(cudaMalloc(&x, 8 * K * 4)); CK(cudaMalloc(&y, 8 * N * 4)); CK(cudaMemset(x, 0, 8 * K * 4));
  printf("== %d x [small op (48 CTAs, spin S) -> GEMV 8 MiB bf16 M=5 (192 CTAs x 256)], us per pair\n", L);
  for (int spin : {0, 300, 1500}) {
    auto body = [&](bool pdl, bool trig, bool pf) {
      for (int l = 0; l < L; ++l) {
        if (trig) launch(pdl && l > 0, 48, 128, s, tiny<true, true>, y, x, 5 * K < n ? 5 * K : n, spin);
        else launch(pdl && l > 0, 48, 128, s, tiny<true, false>, y, x, 5 * K < n ? 5 * K : n, spin);
        if (pf) launch(pdl, 192, 256, s, gemv<5, true, 4>, (const __nv_bfloat16*)Ws[l], (const float*)x, y, N, K);
        else launch(pdl, 192, 256, s, gemv<5, true, 0>, (const __nv_bfloat16*)Ws[l], (const float*)x, y, N, K);
      }
    };
    float t0 = time_graph(s, [&] { body(false, false, false); });
    float t1 = time_graph(s, [&] { body(true, false, false); });
    float t2 = time_graph(s, [&] { body(true, false, true); });
    float t3 = time_graph(s, [&] { body(true, true, true); });
    float tg = time_graph(s, [&] { for (int l = 0; l < L; ++l) launch(false, 192, 256, s, gemv<5, false, 0>, (const __nv_bfloat16*)Ws[l], (const float*)x, y, N, K); });
    float ts = time_graph(s, [&] { for (int l = 0; l < L; ++l) launch(false, 48, 128, s, tiny<false, false>, y, x, 5 * K < n ? 5 * K : n, spin); });
    printf("spin %4d: base %.2f | pdl %.2f | pdl+wprefetch %.2f | pdl+wprefetch+early trigger %.2f || gemv alone %.2f, small alone %.2f\n",
           spin, t0 * 1000 / L, t1 * 1000 / L, t2 * 1000 / L, t3 * 1000 / L, tg * 1000 / L, ts * 1000 / L);
  }
  printf("done\n");
  return 0;
}
