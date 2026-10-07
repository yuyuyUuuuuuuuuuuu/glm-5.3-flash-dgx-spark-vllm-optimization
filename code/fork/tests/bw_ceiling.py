"""GB10 の実効読み出し帯域の上限を測る(最適化の天井を決めるため)。
(a) 大きな連続バッファを 16B ロードで流し読み  (b) decode と同じ形: 2 MiB の塊を K 個、プールから冷えた状態で読む。"""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tf_exl3_moe as T
from torch.utils.cpp_extension import load_inline

src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
template<int U>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  uint4 acc = make_uint4(0,0,0,0);
  size_t i = (size_t)blockIdx.x * blockDim.x * U + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x * U;
  for (; i + (U-1)*blockDim.x < n; i += stride) {
    uint4 v[U];
    #pragma unroll
    for (int u=0; u<U; ++u) v[u] = __ldg(p + i + (size_t)u*blockDim.x);
    #pragma unroll
    for (int u=0; u<U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}
// chunked: chunk c (of nc) at ptrs[c], each `words` uint4; block b handles a slice of all chunks
__global__ void rdc(const long* __restrict__ ptrs, int nc, size_t words, unsigned* out) {
  uint4 acc = make_uint4(0,0,0,0);
  size_t tot = (size_t)nc * words;
  for (size_t i = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x * 4) {
    uint4 v[4];
    #pragma unroll
    for (int u=0; u<4; ++u) { size_t j = i + (size_t)u*blockDim.x; if (j < tot) { const uint4* q = (const uint4*)ptrs[j / words]; v[u] = __ldg(q + (j % words)); } else v[u] = make_uint4(0,0,0,0); }
    #pragma unroll
    for (int u=0; u<4; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}
void run(torch::Tensor buf, int blocks, int threads, int unroll, torch::Tensor out) {
  auto s = at::cuda::getCurrentCUDAStream();
  size_t n = buf.numel() * buf.element_size() / 16;
  const uint4* p = (const uint4*)buf.data_ptr();
  if (unroll == 1) rd<1><<<blocks,threads,0,s>>>(p,n,(unsigned*)out.data_ptr());
  else if (unroll == 4) rd<4><<<blocks,threads,0,s>>>(p,n,(unsigned*)out.data_ptr());
  else rd<8><<<blocks,threads,0,s>>>(p,n,(unsigned*)out.data_ptr());
}
void runc(torch::Tensor ptrs, int nc, int64_t words, int blocks, int threads, torch::Tensor out) {
  auto s = at::cuda::getCurrentCUDAStream();
  rdc<<<blocks,threads,0,s>>>((const long*)ptrs.data_ptr(), nc, (size_t)words, (unsigned*)out.data_ptr());
}
'''
inc = T._cuda_include_shim()
m = load_inline("bwceil", cpp_sources="void run(torch::Tensor,int,int,int,torch::Tensor); void runc(torch::Tensor,int,int64_t,int,int,torch::Tensor);",
                cuda_sources=src, functions=["run", "runc"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc], verbose=False)
dev = "cuda"
props = torch.cuda.get_device_properties(0)
sms = props.multi_processor_count
print(f"GPU {props.name} SMs={sms} L2={getattr(props,'L2_cache_size',0)/2**20:.0f} MiB")
out = torch.zeros(1, dtype=torch.int32, device=dev)
def ev(fn, it):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / 1e3
GB = 4
buf = torch.randint(0, 2**31 - 1, (GB * 2**30 // 4,), dtype=torch.int32, device=dev)
print(f"(a) streaming read of {GB} GiB (cold each pass, >> L2)")
best = 0
for thr in (256, 512, 1024):
    for mult in (1, 2, 4, 8, 16):
        for un in (1, 4, 8):
            t = ev(lambda: m.run(buf, sms * mult, thr, un, out), 5)
            bw = buf.numel() * 4 / t / 1e9; best = max(best, bw)
            if mult in (4, 16) or bw >= best: pass
    print(f"   threads {thr}: best so far {best:.1f} GB/s")
print(f"(a) CEILING streaming: {best:.1f} GB/s")
# (b) decode shape: nc chunks of 2 MiB chosen from a pool (cold): T=1 -> 48 chunks, T=8 -> ~351 chunks
pool = torch.randint(0, 2**31 - 1, (3 * 2**30 // 4,), dtype=torch.int32, device=dev)   # 3 GiB pool
chunk_words = 2 * 2**20 // 16
nchunks_pool = pool.numel() * 4 // (2 * 2**20)
base = pool.data_ptr()
for nc in (24, 48, 176, 351, 1446):
    g = torch.Generator().manual_seed(nc)
    sets = [torch.tensor([base + int(i) * 2 * 2**20 for i in torch.randperm(nchunks_pool, generator=g)[:nc]], dtype=torch.int64, device=dev) for _ in range(8)]
    bestc = 0
    for mult in (2, 4, 8, 16):
        k = [0]
        def f():
            m.runc(sets[k[0] % 8], nc, chunk_words, sms * mult, 512, out); k[0] += 1
        t = ev(f, 16)
        bestc = max(bestc, nc * 2 * 2**20 / t / 1e9)
    print(f"(b) {nc:5d} x 2 MiB chunks ({nc*2} MiB): best {bestc:.1f} GB/s  -> floor {nc*2*2**20/bestc/1e9*1e6:8.1f} us")
