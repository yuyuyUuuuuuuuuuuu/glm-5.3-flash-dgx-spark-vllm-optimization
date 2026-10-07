"""TLB / page-granularity sensitivity of GPU read bandwidth on GB10 (investigating 'bandwidth decays with uptime').
Reads NCH randomly chosen chunks of size C from an 8 GiB pool (one block per chunk, 16-byte loads) for
C in {4 KiB, 64 KiB, 2 MiB} plus a sequential sweep. If the slow state is a translation (TLB/page-size) problem,
small random chunks degrade much more than sequential/2 MiB on a slow node than on a fresh node."""
import sys, statistics, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tf_exl3_moe as T
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void rd(const uint4* __restrict__ base, const long* __restrict__ offs, int words, unsigned* out) {
  const uint4* p = base + offs[blockIdx.x];
  uint4 a = make_uint4(0,0,0,0);
  for (int i = threadIdx.x; i < words; i += blockDim.x) { uint4 v = __ldg(p + i); a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1;
}
void run(torch::Tensor pool, torch::Tensor offs, int64_t words, torch::Tensor out) {
  int thr = words >= 256 ? 256 : (int)words;
  rd<<<(unsigned)offs.numel(), thr, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)pool.data_ptr(), (const long*)offs.data_ptr(), (int)words, (unsigned*)out.data_ptr());
}
'''
inc = T._cuda_include_shim()
m = load_inline("bwtlb", cpp_sources="void run(torch::Tensor,torch::Tensor,int64_t,torch::Tensor);", cuda_sources=src,
                functions=["run"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc], verbose=False)
dev = "cuda"; POOL = 8 * 2**30
pool = torch.randint(0, 2**31 - 1, (POOL // 4,), dtype=torch.int32, device=dev); out = torch.zeros(1, dtype=torch.int32, device=dev)
def ev(fn, it=5):
    fn(); torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it / 1e3
res = {}
g = torch.Generator().manual_seed(0)
for label, C, total in (("seq 2MiB (sequential chunks)", 2 * 2**20, 1 * 2**30), ("rand 2MiB", 2 * 2**20, 1 * 2**30),
                        ("rand 64KiB", 64 * 2**10, 1 * 2**30), ("rand 4KiB", 4 * 2**10, 512 * 2**20)):
    n = total // C; words = C // 16; slots = POOL // C
    if label.startswith("seq"): idx = torch.arange(n)
    else: idx = torch.randperm(slots, generator=g)[:n]
    offs = (idx * words).to(torch.int64).to(dev)
    bws = [total / ev(lambda: m.run(pool, offs, words, out)) / 1e9 for _ in range(5)]
    res[label] = (max(bws), statistics.median(bws))
    print(f"{label:30s} max {max(bws):6.1f} GB/s  median {statistics.median(bws):6.1f}", flush=True)
