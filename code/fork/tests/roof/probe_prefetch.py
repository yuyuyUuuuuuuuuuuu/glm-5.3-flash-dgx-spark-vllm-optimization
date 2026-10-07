"""Can the DRAM-idle window before an FP8 GEMV (latency-bound small kernels: AR, mHC, norms, KDA recurrent) be used to
prefetch the first P MiB of its weight into L2 from a side stream, and does the GEMV then get faster by ~P / BW?
Graph per rep (4 reps x 3 weight sets, cold): X = GEMV on another weight; fork: side stream prefetches P MiB of Y's
weight (mode: bulk = cp.async.bulk.prefetch.L2, line = prefetch.global.L2, ld = real loads); main stream: spin T us
(no DRAM traffic) [+ optional dirty writes]; join; Y = GEMV. Reported: us per rep vs P = 0 (paired rounds).
Usage: probe_prefetch.py T_us mode P1,P2,... [dirty_MiB] [N K]"""
import sys, statistics, os
from pathlib import Path
R = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "tests"))
import torch
import fp8_gemv as G
import tf_exl3_moe as T
from fp8_bench_common import random_marlin_layers
from torch.utils.cpp_extension import load_inline

src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void spin_k(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}
__global__ void pf_k(const char* p, long long nbytes, int mode, unsigned* sink) {
  const long long chunk = 65536;
  if (mode == 0) {
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * chunk; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * chunk) {
      long long sz = nbytes - o < chunk ? nbytes - o : chunk;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"((unsigned)sz) : "memory");
    }
  } else if (mode == 1) {
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 128; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * 128)
      asm volatile("prefetch.global.L2 [%0];" :: "l"(p + o));
  } else {
    unsigned a = 0;
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 16; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * 16) {
      uint4 v = __ldcg((const uint4*)(p + o)); a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x9e3779b9u) sink[0] = a;
  }
}
void spin(int64_t cycles, int64_t blocks) {
  spin_k<<<(unsigned)blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(cycles);
}
void pf(torch::Tensor w, int64_t nbytes, int64_t mode, int64_t ctas, int64_t threads, torch::Tensor sink) {
  pf_k<<<(unsigned)ctas, (unsigned)threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const char*)w.data_ptr(), nbytes, (int)mode, (unsigned*)sink.data_ptr());
}
'''
inc = T._cuda_include_shim()
X = load_inline("roofpf", cpp_sources="void spin(int64_t,int64_t); void pf(torch::Tensor,int64_t,int64_t,int64_t,int64_t,torch::Tensor);",
                cuda_sources=src, functions=["spin", "pf"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc])
E = G.ext()
Tus = float(sys.argv[1]); mode = {"bulk": 0, "line": 1, "ld": 2}[sys.argv[2]]
Ps = [float(v) for v in sys.argv[3].split(",")]
dirty_mib = int(sys.argv[4]) if len(sys.argv) > 4 else 0
n, k = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (12576, 4096)
cfgY = G.select_config(-(-n // 64) * 64 if n % 64 else n, k, 5) or (8, 8, 2, True)
if n == 12576: cfgY = (8, 4, 2, False)
props = torch.cuda.get_device_properties(0)
clk = int(os.environ.get("ROOF_CLK_KHZ", "0")) or 2_000_000   # spin calibration below
M = 5
PCTA = int(os.environ.get("PF_CTAS", "4")); PTHR = int(os.environ.get("PF_THREADS", "32"))
NS = 3
Ys = random_marlin_layers(n, k, NS, seed=7)
Xs = random_marlin_layers(4096, 4096, NS, seed=8)
x = torch.randn(M, k, device="cuda").to(torch.bfloat16)
x4 = torch.randn(M, 4096, device="cuda").to(torch.bfloat16)
yY = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
yX = torch.empty(M, 4096, dtype=torch.bfloat16, device="cuda")
sink = torch.zeros(4, dtype=torch.int32, device="cuda")
dirty = torch.empty(max(dirty_mib, 1) << 20, dtype=torch.uint8, device="cuda")
side = torch.cuda.Stream()
# calibrate spin: cycles per us
a, b = torch.cuda.Event(True), torch.cuda.Event(True)
X.spin(1000, 48); torch.cuda.synchronize()
a.record(); X.spin(2_000_000, 48); b.record(); torch.cuda.synchronize()
cyc_per_us = 2_000_000 / (a.elapsed_time(b) * 1000)
cycles = int(Tus * cyc_per_us)
print(f"spin calibration {cyc_per_us:.0f} cycles/us; Y = [{n}x{k}] cfg {cfgY}; X = 4096x4096; T = {Tus} us; mode "
      f"{sys.argv[2]}; dirty {dirty_mib} MiB", flush=True)
# pf kernel alone
for P in Ps:
    if P <= 0: continue
    nb = int(P * 2**20)
    ts = []
    for _ in range(5):
        a.record(); X.pf(Ys[0].weight, nb, mode, PCTA, PTHR, sink); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    print(f"  prefetch kernel alone P={P} MiB: {statistics.median(ts):.1f} us (4 CTAs x 32 thr)", flush=True)


def seq(i, P):
    E.fp8_gemv_out(yX, x4, Xs[i].weight, Xs[i].weight_scale.view(-1), None, 4096, 4096, 16, 8, 2, True, True)
    if P > 0:
        ev = torch.cuda.Event(); ev.record()
        side.wait_event(ev)
        with torch.cuda.stream(side):
            X.pf(Ys[i].weight, int(P * 2**20), mode, PCTA, PTHR, sink)
        done = torch.cuda.Event(); done.record(side)
    if dirty_mib:
        dirty.fill_(i + 1)
    X.spin(cycles, 48)
    if P > 0:
        torch.cuda.current_stream().wait_event(done)
    E.fp8_gemv_out(yY, x, Ys[i].weight, Ys[i].weight_scale.view(-1), None, n, k, *cfgY, True)


def capture(P, reps=4):
    for i in range(NS): seq(i, P)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for i in range(NS): seq(i, P)
    g.replay(); torch.cuda.synchronize()
    return g, reps * NS


graphs = {P: capture(P) for P in [0.0] + [p for p in Ps if p > 0]}
res = {P: [] for P in graphs}
for r in range(11):
    for P in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
        g, cnt = graphs[P]
        a.record()
        for _ in range(3): g.replay()
        b.record(); torch.cuda.synchronize()
        res[P].append(a.elapsed_time(b) * 1000 / (3 * cnt))
base = statistics.median(res[0.0])
bw = 240e3  # bytes/us
print(f"per rep (X gemv + spin {Tus} us + Y gemv): P=0 {base:.1f} us")
for P in graphs:
    if P == 0: continue
    m = statistics.median(res[P]); rr = [p / q for p, q in zip(res[P], res[0.0])]
    print(f"  P={P:5.1f} MiB: {m:8.1f} us  saving {base - m:6.1f} us (ideal {P * 2**20 / bw:5.1f})  ratio median "
          f"{statistics.median(rr):.4f} [{min(rr):.4f}, {max(rr):.4f}]", flush=True)
