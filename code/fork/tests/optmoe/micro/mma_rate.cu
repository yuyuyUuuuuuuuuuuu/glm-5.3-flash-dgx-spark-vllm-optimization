// opt-moe microbenchmark: mma.sync m16n8k32 e4m3 (f32 acc) and m16n8k16 f16 (f32 acc / f16 acc) throughput on this
// GPU, 48 CTAs x 512 threads (1 CTA/SM, the fused kernel's shape), 16 independent accumulator chains per warp (MB=8 x 2).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
template <int KIND>
__global__ void __launch_bounds__(512, 1) k(float* out, int iters, uint32_t seed) {
    float acc[8][2][4];
    for (int i = 0; i < 8; ++i) for (int h = 0; h < 2; ++h) for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
    uint32_t a[4] = {seed ^ threadIdx.x, seed * 3u, seed + 7u, seed ^ 0x3c3c3c3cu}, b0[2] = {seed * 5u, seed * 9u}, b1[2] = {seed * 11u, seed ^ 0x24242424u};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if constexpr (KIND == 0) {
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][0][0]), "+f"(acc[i][0][1]), "+f"(acc[i][0][2]), "+f"(acc[i][0][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0[0]), "r"(b0[1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][1][0]), "+f"(acc[i][1][1]), "+f"(acc[i][1][2]), "+f"(acc[i][1][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b1[0]), "r"(b1[1]));
            } else if constexpr (KIND == 1) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][0][0]), "+f"(acc[i][0][1]), "+f"(acc[i][0][2]), "+f"(acc[i][0][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0[0]), "r"(b0[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][1][0]), "+f"(acc[i][1][1]), "+f"(acc[i][1][2]), "+f"(acc[i][1][3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b1[0]), "r"(b1[1]));
            } else {
                // e4m3 with f16 accumulator (m16n8k32 .f16.e4m3.e4m3.f16): 2 regs per acc
                uint32_t* d0 = reinterpret_cast<uint32_t*>(&acc[i][0][0]);
                uint32_t* d1 = reinterpret_cast<uint32_t*>(&acc[i][1][0]);
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                    : "+r"(d0[0]), "+r"(d0[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0[0]), "r"(b0[1]));
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                    : "+r"(d1[0]), "+r"(d1[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b1[0]), "r"(b1[1]));
            }
        }
    }
    float s = 0.f;
    for (int i = 0; i < 8; ++i) for (int h = 0; h < 2; ++h) for (int c = 0; c < 4; ++c) s += acc[i][h][c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
template <int KIND>
void run(const char* name, int kdim) {
    float* out; cudaMalloc(&out, 48 * 512 * 4);
    const int iters = 4000;
    k<KIND><<<48, 512>>>(out, 10, 1u);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    k<KIND><<<48, 512>>>(out, iters, 2u);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    const double macs = 48.0 * 16 * iters * 16 * (16.0 * 8 * kdim);
    const double cyc = ms * 1e-3 * clk * 1e3;
    printf("%s: %.3f ms, %.1f TFLOPS, %.0f MAC/clk/SM at the attribute clock %d MHz\n", name, ms, 2 * macs / (ms * 1e-3) / 1e12, macs / cyc / 48, clk / 1000);
    cudaFree(out);
}
int main() {
    run<0>("m16n8k32 e4m3 f32acc", 32);
    run<1>("m16n8k16 f16  f32acc", 16);
    run<2>("m16n8k32 e4m3 f16acc", 32);
    return 0;
}
