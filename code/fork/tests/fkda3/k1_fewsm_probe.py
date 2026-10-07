#!/usr/bin/env python3
"""FKDA3 probe: how fast is K1 (prepare) when only 16 of 48 SMs are free (what a K1->K2 pipeline would give it while
K2's 32 blocks sit on 32 SMs)? An occupier kernel (1 block/SM via 99 KB dynamic smem, spinning on the clock, no memory
traffic) holds 32 SMs while the K1-only build runs."""
import statistics, sys, torch
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, "/fkda/builds/k1only"); import _fk3a_C  # noqa
src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void occ(long long cycles, int* flag) {
  extern __shared__ char s[];
  long long t0 = clock64();
  while (clock64() - t0 < cycles) { if (threadIdx.x == 0) s[0] = 1; }
}
void occupy(int64_t blocks, int64_t cycles) {
  cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
  cudaFuncSetAttribute(occ, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
  occ<<<blocks, 128, 99 * 1024, st>>>(cycles, nullptr);
}
"""
m = load_inline("occ_ext", cpp_sources="void occupy(int64_t blocks, int64_t cycles);", cuda_sources=src,
                functions=["occupy"], verbose=False, extra_cuda_cflags=["-arch=sm_121a"])
D, H, T = 128, 32, 13824
g = torch.Generator(device="cpu").manual_seed(0)
rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
q, k, v, gg, b = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D, sc=.5), rn(1, T, H)
s0 = torch.zeros(1, H, D, D, device="cuda"); fs = torch.empty_like(s0)
A = torch.zeros(H, device="cuda"); dtb = torch.full((H, D), -6.0, device="cuda")
cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
ws = torch.empty(int(torch.ops._fk3a_C.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
out = torch.empty_like(q)
k1 = lambda: torch.ops._fk3a_C.fwd(q, k, v, gg, b, D ** -.5, out, ws, A, dtb, -5.0, s0, fs, cu, None, None)
side = torch.cuda.Stream()
def timed(occ_blocks):
    ts = []
    for _ in range(8):
        torch.cuda.synchronize()
        if occ_blocks:
            with torch.cuda.stream(side):
                m.occupy(occ_blocks, 40_000_000)       # ~20+ ms at ~2 GHz
            torch.cuda._sleep(2_000_000)              # let the occupier land first
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); k1(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
for blocks in (0, 16, 24, 32, 40):
    print(f"K1 T={T} with {blocks} SMs occupied ({48 - blocks} free): {timed(blocks):.3f} ms", flush=True)
