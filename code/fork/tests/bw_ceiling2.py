"""GB10 の読み出し帯域をいくつもの方式で測り直す(仕様 273 GB/s。前回は 16B ldg の 1 方式で 250 GB/s)。
方式: (a) 16B ldg, (b) ld.global.nc.L1::no_allocate.L2::evict_first 16B, (c) cp.async.bulk (TMA 1D) → shared,
(d) cudaMemcpy D2D (読み+書きで 2 倍のバイト), (e) torch sum。各構成で複数回の最大/中央値。"""
import sys, statistics, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tf_exl3_moe as T
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
struct u8x { unsigned a[8]; };
// 32-byte loads (sm_121 accepts .v8.b32); EF variant adds the L2 evict_first hint
template<bool EF> __device__ __forceinline__ u8x ld32(const uint4* p) {
  u8x v;
  if (EF) asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    : "=r"(v.a[0]),"=r"(v.a[1]),"=r"(v.a[2]),"=r"(v.a[3]),"=r"(v.a[4]),"=r"(v.a[5]),"=r"(v.a[6]),"=r"(v.a[7]) : "l"(p));
  else asm volatile("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    : "=r"(v.a[0]),"=r"(v.a[1]),"=r"(v.a[2]),"=r"(v.a[3]),"=r"(v.a[4]),"=r"(v.a[5]),"=r"(v.a[6]),"=r"(v.a[7]) : "l"(p));
  return v; }
template<int U, bool EF>
__global__ void rd32(const uint4* __restrict__ p, size_t n2, unsigned* out) {   // n2 = number of 32-byte words
  unsigned acc = 0;
  size_t i = (size_t)blockIdx.x * blockDim.x * U + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x * U;
  for (; i + (size_t)(U-1)*blockDim.x < n2; i += stride) {
    u8x v[U];
#pragma unroll
    for (int u=0; u<U; ++u) v[u] = ld32<EF>(p + 2*(i + (size_t)u*blockDim.x));
#pragma unroll
    for (int u=0; u<U; ++u) for (int k=0;k<8;++k) acc ^= v[u].a[k];
  }
  if (acc == 0x12345678u) out[0] = 1;
}
template<int U, bool EF>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  uint4 acc = make_uint4(0,0,0,0);
  size_t i = (size_t)blockIdx.x * blockDim.x * U + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x * U;
  for (; i + (size_t)(U-1)*blockDim.x < n; i += stride) {
    uint4 v[U];
#pragma unroll
    for (int u=0; u<U; ++u) v[u] = __ldg(p + i + (size_t)u*blockDim.x);
#pragma unroll
    for (int u=0; u<U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}
// TMA 1D bulk copy global -> shared, STAGES-deep ring of CH-byte chunks per block, mbarrier completion
template<int CH, int STAGES>
__global__ void bulk(const char* __restrict__ p, size_t nbytes, unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  __shared__ __align__(8) uint64_t bar[STAGES];
  const size_t nch = nbytes / CH;
  if (threadIdx.x == 0) for (int s=0;s<STAGES;++s) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&bar[s])));
  __syncthreads();
  unsigned acc = 0; int phase[STAGES]; for (int s=0;s<STAGES;++s) phase[s]=0;
  size_t c = blockIdx.x; int issued = 0;
  auto issue = [&](size_t chunk, int s) {
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar[s]);
    unsigned d = (unsigned)__cvta_generic_to_shared(sm + (size_t)s*CH);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"(CH));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(d), "l"(p + chunk*CH), "r"(CH), "r"(b) : "memory");
  };
  if (threadIdx.x == 0) for (int s=0; s<STAGES && c + (size_t)s*gridDim.x < nch; ++s) { issue(c + (size_t)s*gridDim.x, s); ++issued; }
  __syncthreads();
  int s = 0;
  for (size_t k = c; k < nch; k += gridDim.x) {
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar[s]); unsigned done = 0;
    while (!done) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0,1,0,P; }" : "=r"(done) : "r"(b), "r"(phase[s]));
    phase[s] ^= 1;
    acc ^= ((const unsigned*)(sm + (size_t)s*CH))[threadIdx.x];
    __syncthreads();
    size_t nk = k + (size_t)STAGES*gridDim.x;
    if (threadIdx.x == 0 && nk < nch) issue(nk, s);
    s = (s + 1) % STAGES;
  }
  if (acc == 0x12345678u) out[0] = 1;
}
void run(torch::Tensor buf, int blocks, int threads, int unroll, bool ef, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream(); size_t n = buf.numel()*buf.element_size()/16; auto p=(const uint4*)buf.data_ptr(); auto o=(unsigned*)out.data_ptr();
  // mode: ef=false -> 16B __ldg ; ef=true -> 32B loads (unroll<0 => 32B with evict_first, unroll>0 => 32B plain)
  if (!ef) { if (unroll==4) rd<4,false><<<blocks,threads,0,st>>>(p,n,o); else if (unroll==8) rd<8,false><<<blocks,threads,0,st>>>(p,n,o); else rd<16,false><<<blocks,threads,0,st>>>(p,n,o); }
  else { size_t n2 = n/2; int u = unroll < 0 ? -unroll : unroll; bool h = unroll < 0;
    if (h) { if (u==2) rd32<2,true><<<blocks,threads,0,st>>>(p,n2,o); else if (u==4) rd32<4,true><<<blocks,threads,0,st>>>(p,n2,o); else rd32<8,true><<<blocks,threads,0,st>>>(p,n2,o); }
    else   { if (u==2) rd32<2,false><<<blocks,threads,0,st>>>(p,n2,o); else if (u==4) rd32<4,false><<<blocks,threads,0,st>>>(p,n2,o); else rd32<8,false><<<blocks,threads,0,st>>>(p,n2,o); } }
}
void runbulk(torch::Tensor buf, int blocks, int ch, int stages, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream(); size_t nb = buf.numel()*buf.element_size(); auto p=(const char*)buf.data_ptr(); auto o=(unsigned*)out.data_ptr();
  size_t sh = (size_t)ch*stages;
#define L(C,S) { auto k = bulk<C,S>; cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh); k<<<blocks,128,sh,st>>>(p,nb,o); auto e_ = cudaGetLastError(); TORCH_CHECK(e_ == cudaSuccess, cudaGetErrorString(e_)); }
  if (ch==16384 && stages==4) L(16384,4) else if (ch==16384 && stages==8) L(16384,8) else if (ch==32768 && stages==4) L(32768,4)
  else if (ch==65536 && stages==2) L(65536,2) else if (ch==8192 && stages==8) L(8192,8) else L(32768,2)
}
'''
inc = T._cuda_include_shim()
m = load_inline("bwceil2", cpp_sources="void run(torch::Tensor,int,int,int,bool,torch::Tensor); void runbulk(torch::Tensor,int,int,int,torch::Tensor);",
                cuda_sources=src, functions=["run", "runbulk"], extra_cuda_cflags=["-O3", "-arch=sm_121a", *inc], extra_cflags=[*inc], verbose=False)
dev = "cuda"; sms = torch.cuda.get_device_properties(0).multi_processor_count
out = torch.zeros(1, dtype=torch.int32, device=dev)
def ev(fn, it=6):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / 1e3
GB = 4; buf = torch.randint(0, 2**31-1, (GB*2**30//4,), dtype=torch.int32, device=dev); NB = buf.numel()*4
def best(fn, bytes_, reps=5):
    v = [bytes_ / ev(fn) / 1e9 for _ in range(reps)]; return max(v), statistics.median(v)
res = {}
for ef, uns, label in ((False, (4, 8, 16), "(a) __ldg 16B"), (True, (2, 4, 8), "(b1) ld.nc.v8.b32 32B"), (True, (-2, -4, -8), "(b2) ld.nc.v8.b32 32B + L2::evict_first")):
    b0 = (0, 0, None)
    for thr in (256, 512, 1024):
        for mult in (2, 4, 8, 16, 32):
            for un in uns:
                mx, md = best(lambda: m.run(buf, sms*mult, thr, un, ef, out), NB, 3)
                if mx > b0[0]: b0 = (mx, md, (thr, mult, un))
    mx, md = best(lambda t=b0[2], ef=ef: m.run(buf, sms*t[1], t[0], t[2], ef, out), NB, 9)
    res[label] = (mx, md, b0[2]); print(f"  {label:36s} max {mx:6.1f} GB/s median {md:6.1f} ({mx/273*100:5.1f}% of spec) cfg={b0[2]}", flush=True)
bb = (0, 0, None)
for ch, stg in ((8192, 8), (16384, 4), (16384, 8), (32768, 2), (32768, 4), (65536, 2)):
    for mult in (1, 2, 4):
        try:
            mx, md = best(lambda: m.runbulk(buf, sms*mult, ch, stg, out), NB, 3)
            torch.cuda.synchronize()
        except Exception as e:
            print("bulk", ch, stg, "failed:", str(e)[:80]); continue
        if mx > bb[0]: bb = (mx, md, (ch, stg, mult))
if bb[2]:
    mx, md = best(lambda t=bb[2]: m.runbulk(buf, sms*t[2], t[0], t[1], out), NB, 9); res["(c) cp.async.bulk (TMA 1D)"] = (mx, md, bb[2])
dst = torch.empty(2*2**30//4, dtype=torch.int32, device=dev); srcb = buf[: dst.numel()]
mx, md = best(lambda: dst.copy_(srcb), 2 * dst.numel()*4, 9); res["(d) D2D copy (read+write counted)"] = (mx, md, None)
mx, md = best(lambda: buf.sum(), NB, 9); res["(e) torch int32 sum"] = (mx, md, None)
print(f"GB10 SMs={sms}; buffer {GB} GiB (cold, >> 24 MiB L2); spec 273 GB/s")
for k, (mx, md, cfg) in res.items():
    print(f"  {k:36s} max {mx:6.1f} GB/s  median {md:6.1f} GB/s  ({mx/273*100:5.1f}% of spec)  cfg={cfg}")
