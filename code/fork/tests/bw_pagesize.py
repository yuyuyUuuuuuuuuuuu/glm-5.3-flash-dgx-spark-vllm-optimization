"""Is GB10 GPU read bandwidth sensitive to the physical page size / contiguity of the memory it reads?
(A) pageable host memory read directly by the GPU (ATS / host page tables): same buffer backed by 4 KiB pages
    (MADV_NOHUGEPAGE) vs 2 MiB THP (MADV_HUGEPAGE), THP backing verified from /proc/self/smaps.
(B) cudaMalloc: allocate CHUNKS x 1 GiB one after another and measure each chunk on its own (a fragmented free list
    would show up as slow / bimodal chunks).  Usage: bw_pagesize.py [host_GiB=4] [chunks=24]"""
import sys, mmap, ctypes, statistics, re, torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tf_exl3_moe as T
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out) {
  uint4 a = make_uint4(0,0,0,0); size_t i = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x, st = (size_t)gridDim.x * blockDim.x * 4;
  for (; i + 3 * blockDim.x < n; i += st) {
#pragma unroll
    for (int u = 0; u < 4; ++u) { uint4 v = __ldg(p + i + (size_t)u * blockDim.x); a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; } }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) out[0] = 1; }
void run_ptr(int64_t ptr, int64_t nbytes, torch::Tensor o, int64_t blocks) {
  rd<<<(unsigned)blocks, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)ptr, (size_t)nbytes / 16, (unsigned*)o.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK(); }
std::vector<int64_t> attrs() {
  int a[5] = {0,0,0,0,0};
  cudaDeviceGetAttribute(&a[0], cudaDevAttrPageableMemoryAccess, 0);
  cudaDeviceGetAttribute(&a[1], cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  cudaDeviceGetAttribute(&a[2], cudaDevAttrIntegrated, 0);
  cudaDeviceGetAttribute(&a[3], cudaDevAttrConcurrentManagedAccess, 0);
  cudaDeviceGetAttribute(&a[4], cudaDevAttrDirectManagedMemAccessFromHost, 0);
  return {a[0], a[1], a[2], a[3], a[4]}; }
'''
inc = T._cuda_include_shim()
m = load_inline("bwpagesize", cpp_sources="void run_ptr(int64_t,int64_t,torch::Tensor,int64_t); std::vector<int64_t> attrs();",
                cuda_sources=src, functions=["run_ptr", "attrs"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc], verbose=False)
HOST_GIB = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
BALLOON_GIB = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # (C) no-root defrag: touch this much MADV_HUGEPAGE memory, then free it
o = torch.zeros(1, dtype=torch.int32, device="cuda")
blocks = torch.cuda.get_device_properties(0).multi_processor_count * 32
a = m.attrs()
print(f"attrs: pageableMemoryAccess={a[0]} usesHostPageTables={a[1]} integrated={a[2]} concurrentManaged={a[3]} directManagedFromHost={a[4]}", flush=True)

def bw(ptr, nbytes, reps=7):
    m.run_ptr(ptr, nbytes, o, blocks); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); m.run_ptr(ptr, nbytes, o, blocks); e.record()
        torch.cuda.synchronize(); out.append(nbytes / (s.elapsed_time(e) / 1e3) / 1e9)
    return max(out), statistics.median(out)

def smaps_for(addr):
    cur = None; res = {}
    for line in open("/proc/self/smaps"):
        mm_ = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
        if mm_:
            lo, hi = int(mm_.group(1), 16), int(mm_.group(2), 16); cur = lo <= addr < hi
            continue
        if cur and ":" in line:
            k, v = line.split(":", 1); res[k.strip()] = v.strip()
    return res

def buddy():
    for line in open("/proc/buddyinfo"):
        if "Normal" in line: return line.split()[4:]

print("buddyinfo Normal (order 0..13) before:", " ".join(buddy()), flush=True)
# ---- (A) pageable host memory
if a[0]:
    H = HOST_GIB * 2**30; ALIGN = 2 * 2**20
    for label, adv in (("host 4KiB pages (MADV_NOHUGEPAGE)", mmap.MADV_NOHUGEPAGE), ("host 2MiB THP (MADV_HUGEPAGE)", mmap.MADV_HUGEPAGE)):
        mm_ = mmap.mmap(-1, H + ALIGN, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mm_)); off = (-base) % ALIGN
        mm_.madvise(adv, off - (off % mmap.PAGESIZE), H)
        arr = np.frombuffer(mm_, dtype=np.uint8, count=H, offset=off); arr[:] = 7   # first touch on the CPU
        sm = smaps_for(base + off)
        mx, med = bw(base + off, H)
        print(f"{label:38s} max {mx:6.1f} GB/s median {med:6.1f}   smaps AnonHugePages={sm.get('AnonHugePages')} KernelPageSize={sm.get('KernelPageSize')} Rss={sm.get('Rss')}", flush=True)
        mx2, med2 = bw(base + off, H)
        print(f"{'  (second pass)':38s} max {mx2:6.1f} GB/s median {med2:6.1f}", flush=True)
        del arr; mm_.close()
else:
    print("pageable memory access not supported; skipping (A)")
# ---- (C) balloon: THP faults with defrag=madvise do direct reclaim + compaction; freeing lets the buddies merge
if BALLOON_GIB:
    import time
    t0 = time.time(); Bn = BALLOON_GIB * 2**30
    mm_ = mmap.mmap(-1, Bn + 2 * 2**20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    mm_.madvise(mmap.MADV_HUGEPAGE)
    arr = np.frombuffer(mm_, dtype=np.uint8); step = 2 * 2**20
    for i in range(0, Bn, 1 << 30): arr[i:i + (1 << 30):4096] = 1          # touch every 4 KiB page, 1 GiB at a time
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm_)); sm = smaps_for(base)
    print(f"balloon {BALLOON_GIB} GiB touched in {time.time()-t0:.1f}s AnonHugePages={sm.get('AnonHugePages')}", flush=True)
    print("buddyinfo Normal with balloon held:", " ".join(buddy()), flush=True)
    del arr; mm_.close()
    print("buddyinfo Normal after balloon freed:", " ".join(buddy()), flush=True)
# ---- (B) cudaMalloc chunks
ch = []; res = []
for i in range(CHUNKS):
    t = torch.empty(2**30, dtype=torch.uint8, device="cuda"); t.fill_(3); ch.append(t)
    mx, med = bw(t.data_ptr(), t.numel())
    res.append(mx)
    print(f"cudaMalloc chunk {i:2d}  max {mx:6.1f} GB/s median {med:6.1f}   buddy o9..13={' '.join(buddy()[9:])}", flush=True)
big = torch.cat([c.view(1, -1) for c in ch[:4]]) if False else None
print(f"cudaMalloc chunks: min {min(res):.1f} median {statistics.median(res):.1f} max {max(res):.1f} GB/s", flush=True)
sm = smaps_for(ch[0].data_ptr())
print("smaps for a cudaMalloc chunk:", {k: sm.get(k) for k in ("Size", "Rss", "KernelPageSize", "MMUPageSize", "AnonHugePages")} if sm else "(not mapped in the CPU address space)")
print("buddyinfo Normal after:", " ".join(buddy()), flush=True)
