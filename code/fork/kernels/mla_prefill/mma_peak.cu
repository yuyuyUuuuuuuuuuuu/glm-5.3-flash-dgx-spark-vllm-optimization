// Tensor-core issue-rate probe on GB10 (sm_121a): back-to-back mma.sync with independent accumulators.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template <int KIND, int NACC>
__global__ void mma_loop(float* out, int iters) {
  uint32_t a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  float c[NACC][4];
#pragma unroll
  for (int i = 0; i < NACC; i++) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < NACC; i++) {
      if constexpr (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(*reinterpret_cast<uint32_t*>(&c[i][0])), "+r"(*reinterpret_cast<uint32_t*>(&c[i][1]))
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NACC; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
  if (s == 123.456f) out[threadIdx.x] = s;
}

double run(int kind, int blocks, int threads, int iters) {
  auto out = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto st = at::cuda::getCurrentCUDAStream();
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  auto launch = [&]() {
    if (kind == 0) mma_loop<0, 8><<<blocks, threads, 0, st>>>(out.data_ptr<float>(), iters);
    else if (kind == 1) mma_loop<1, 8><<<blocks, threads, 0, st>>>(out.data_ptr<float>(), iters);
    else mma_loop<2, 8><<<blocks, threads, 0, st>>>(out.data_ptr<float>(), iters);
  };
  launch();
  cudaEventRecord(a, st);
  launch();
  cudaEventRecord(b, st);
  cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  double flop_per_mma = (kind == 1) ? 2.0 * 16 * 8 * 32 : 2.0 * 16 * 8 * 16;
  double total = flop_per_mma * 8 * iters * (double)blocks * (threads / 32);
  return total / (ms * 1e-3) / 1e12;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
