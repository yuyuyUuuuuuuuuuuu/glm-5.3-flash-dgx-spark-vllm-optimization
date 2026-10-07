#include <cstdio>
#include <cstdint>
// positive control for racecheck on sm_121: (1) plain smem WAR/RAW race without a barrier, (2) cp.async write read by
// another thread with wait_group but WITHOUT __syncthreads
__global__ void k(float* o, const float* g) {
    __shared__ float s[256];
    __shared__ __align__(16) float t[256 * 4];
    s[threadIdx.x] = threadIdx.x;
    o[threadIdx.x] = s[(threadIdx.x + 1) & 255];      // race (1)
    uint32_t sa = (uint32_t) __cvta_generic_to_shared(&t[threadIdx.x * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(g + threadIdx.x * 4));
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    o[256 + threadIdx.x] = t[((threadIdx.x + 1) & 255) * 4];   // race (2)
}
int main() { float *o, *g; cudaMalloc(&o, 4096 * 4); cudaMalloc(&g, 4096 * 4); cudaMemset(g, 0, 4096 * 4);
  k<<<1, 256>>>(o, g); printf("done %s\n", cudaGetErrorString(cudaDeviceSynchronize())); }
