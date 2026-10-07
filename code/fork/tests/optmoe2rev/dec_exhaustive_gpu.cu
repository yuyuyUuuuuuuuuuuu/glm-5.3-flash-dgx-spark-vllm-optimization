// opt-moe2-rev: exhaustive GPU identity of the lean trellis decode. The device functions below were extracted verbatim
// from kernels/moe_e4m3.cu (b5ddd55) with sed. Build/run: tests/gpu_run.sh bash -c "nvcc -O3 -arch=sm_121a -o /w/scratch_rev/d tests/optmoe2rev/dec_exhaustive_gpu.cu && /w/scratch_rev/d"
// Result (nodeC, 2026-10-03): mcg2 0 / 2^32, decode_tile 0 / 2 x 2^32, negative control 1,378,025,472 / 2^32 mismatches.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
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
// exhaustive: (a) mcg2(s0,s1) == mcg2_lean(s0,s1,cx) for all 16-bit s0,s1 (2^32, incl. hadd2);
// (b) decode_tile(w) == decode_tile_lean(w) for all 2^32 w (p = the previous lane's w, as in the kernel)
__global__ void k_mcg(unsigned long long* bad, uint32_t base) {
    uint32_t cx = 0x3B603B60u;
    asm volatile("mov.b32 %0, %1;" : "=r"(cx) : "r"(cx));
    uint64_t i = (uint64_t) base + (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s0 = (uint32_t)(i >> 16) & 0xffffu, s1 = (uint32_t) i & 0xffffu;
    if (mcg2(s0, s1) != mcg2_lean(s0, s1, cx)) atomicAdd(bad, 1ull);
}
__global__ void k_tile(unsigned long long* bad, uint32_t base, uint32_t mix) {
    uint32_t cx = 0x3B603B60u;
    asm volatile("mov.b32 %0, %1;" : "=r"(cx) : "r"(cx));
    const int lane = threadIdx.x & 31;
    uint32_t w = base + blockIdx.x * blockDim.x + threadIdx.x;
    w ^= mix;     // mix != 0 permutes which neighbour each w sees as p
    uint32_t a0[2], a1[2], b0[2], b1[2];
    decode_tile(w, lane, a0, a1);
    decode_tile_lean(w, lane, cx, b0, b1);
    if (a0[0] != b0[0] || a0[1] != b0[1] || a1[0] != b1[0] || a1[1] != b1[1]) atomicAdd(bad, 1ull);
}
__global__ void k_neg(unsigned long long* bad, unsigned long long* cnt, uint32_t base) {
    uint64_t i = (uint64_t) base + (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t s0 = (uint32_t)(i >> 16) & 0xffffu, s1 = (uint32_t) i & 0xffffu;
    uint32_t x0 = s0 * 0xCBAC1FEDu, x1 = s1 * 0xCBAC1FEDu;
    uint32_t lo = (__byte_perm(x0, x1, 0x5410) & 0x8FFF8FFEu) ^ 0x3B603B60u, hi = (__byte_perm(x0, x1, 0x7632) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    if (mcg2(s0, s1) != *reinterpret_cast<uint32_t*>(&r)) atomicAdd(bad, 1ull);
    atomicAdd(cnt, 1ull);
}
int main() {
    unsigned long long* d; cudaMalloc(&d, 16); cudaMemset(d, 0, 16);
    const uint32_t chunk = 1u << 30;
    for (int c = 0; c < 4; ++c) k_mcg<<<chunk / 256, 256>>>(d, (uint32_t) c * chunk);
    for (int c = 0; c < 4; ++c) k_tile<<<chunk / 256, 256>>>(d + 1, (uint32_t) c * chunk, 0u);
    for (int c = 0; c < 4; ++c) k_tile<<<chunk / 256, 256>>>(d + 1, (uint32_t) c * chunk, 0x5A5A1234u);
    unsigned long long* n; cudaMalloc(&n, 16); cudaMemset(n, 0, 16);
    for (int c = 0; c < 4; ++c) k_neg<<<chunk / 256, 256>>>(n, n + 1, (uint32_t) c * chunk);
    unsigned long long hn[2]; cudaMemcpy(hn, n, 16, cudaMemcpyDeviceToHost);
    printf("negative control: mismatches %llu of %llu threads\n", hn[0], hn[1]);
    unsigned long long h[2]; cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    printf("err=%s  mcg2 mismatches (2^32 s0,s1 pairs, full incl hadd2) = %llu ; decode_tile mismatches (2x 2^32 w) = %llu\n",
           cudaGetErrorString(e), h[0], h[1]);
    return (h[0] || h[1] || e) ? 1 : 0;
}
