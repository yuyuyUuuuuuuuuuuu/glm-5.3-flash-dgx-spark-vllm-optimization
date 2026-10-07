"""Sustained-load read bandwidth vs time and SoC temperature (is the 'fresh node is faster' effect thermal?).
Streams a 4 GiB buffer continuously for DURATION seconds; every 10 s prints GB/s and the thermal-zone temperatures."""
import sys, time, glob, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tf_exl3_moe as T
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  uint4 a = make_uint4(0,0,0,0); size_t i = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x, st = (size_t)gridDim.x * blockDim.x * 4;
  for (; i + 3 * blockDim.x < n; i += st) {
#pragma unroll
    for (int u = 0; u < 4; ++u) { uint4 v = __ldg(p + i + (size_t)u * blockDim.x); a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; } }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1; }
void run(torch::Tensor b, torch::Tensor o, int64_t blocks) {
  rd<<<(unsigned)blocks, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)b.data_ptr(), b.numel() * 4 / 16, (unsigned*)o.data_ptr()); }
'''
inc = T._cuda_include_shim()
m = load_inline("bwsoak", cpp_sources="void run(torch::Tensor,torch::Tensor,int64_t);", cuda_sources=src, functions=["run"],
                extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc], verbose=False)
DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 300
b = torch.randint(0, 2**31 - 1, (4 * 2**30 // 4,), dtype=torch.int32, device="cuda"); o = torch.zeros(1, dtype=torch.int32, device="cuda")
blocks = torch.cuda.get_device_properties(0).multi_processor_count * 32
def temps():
    out = []
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try: out.append(f"{open(z + '/type').read().strip()}={int(open(z + '/temp').read()) / 1000:.0f}C")
        except Exception: pass
    return " ".join(out) or "no thermal zones visible"
t0 = time.time(); nxt = t0 + 10; n = 0; tb = time.time()
while time.time() - t0 < DUR:
    m.run(b, o, blocks); n += 1
    if n % 8 == 0:
        torch.cuda.synchronize()
        if time.time() >= nxt:
            bw = n * b.numel() * 4 / (time.time() - tb) / 1e9
            print(f"t={time.time() - t0:5.0f}s  {bw:6.1f} GB/s  {temps()}", flush=True); n = 0; tb = time.time(); nxt += 10
