"""Node1 simulation of production's KDA+MoE decode layer (docs/DEC_FP8ROOF.md) with and without the L2 prefetch
triggers of fp8_roof.py, all real FP8 GEMV kernels at production shapes and configs, cold weights (4 layers cycled):
  in_proj (12608x4096, (8,4,2,F))                                   -- T1 fork: prefetch P1 MiB of o_proj(L)
  window A: f_b, g_b GEMVs; recurrent stand-in (read 2 MiB state, write D MiB per-token states); spin SA us
  o_proj (4096x4096, (16,8,2,T))                                    -- T2 fork: prefetch shared gate_up + down (L)
  window B: spin SB us (all-reduce + mHC), router stand-in (2.4 MB read)
  shared gate_up (2048x4096) + down (4096x1024) on an aux stream || MoE stand-in: evict_first streaming of ~150 MB
  window C: spin SC us (epilogue + all-reduce + mHC)                -- T3 fork (MoE hook): P3 MiB of in_proj(L+1)
Paired rounds of graphs {off, T1, T2, T3, all}; reported: us per simulated layer and the saving vs off.
Usage: sim_prefetch.py [D_MiB=10] [P1=12] [P3=12] [ctas=16] [pol=0] [SA SB SC = 20 40 40]"""
import sys, statistics
from pathlib import Path
R = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "tests"))
import torch
import fp8_gemv as G
import fp8_roof as FR
import tf_exl3_moe as T
from fp8_bench_common import random_marlin_layers
from torch.utils.cpp_extension import load_inline

a = [float(v) for v in sys.argv[1:]]
D, P1, P3, CTAS, POL = (int(a[0]) if len(a) > 0 else 10, a[1] if len(a) > 1 else 12, a[2] if len(a) > 2 else 12,
                        int(a[3]) if len(a) > 3 else 16, int(a[4]) if len(a) > 4 else 0)
SA, SB, SC = (a[5], a[6], a[7]) if len(a) > 7 else (20.0, 40.0, 40.0)
import os
LATE = os.environ.get("ROOF_JOIN", "late") == "late"
inc = T._cuda_include_shim()
S = load_inline("roofspin", cpp_sources="void spin(int64_t,int64_t);", cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void spin_k(long long c) { long long t0 = clock64(); while (clock64() - t0 < c) {} }
void spin(int64_t c, int64_t b) { spin_k<<<(unsigned)b, 128, 0, at::cuda::getCurrentCUDAStream()>>>(c); }
''', functions=["spin"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc])
E = G.ext()
PX = FR.ext()
dev = "cuda"; M = 5; NL = 4
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
S.spin(1000, 48); torch.cuda.synchronize()
ev0.record(); S.spin(2_000_000, 48); ev1.record(); torch.cuda.synchronize()
cyc = 2_000_000 / (ev1.elapsed_time(ev0) * -1000) if False else 2_000_000 / (ev0.elapsed_time(ev1) * 1000)
inp = random_marlin_layers(12576, 4096, 1)[0]
in_w = [inp.weight] + [inp.weight.clone() for _ in range(NL - 1)]
fb = random_marlin_layers(4096, 128, 2, seed=1)
op = random_marlin_layers(4096, 4096, NL, seed=2)
sg = random_marlin_layers(2048, 4096, NL, seed=4)
sd = random_marlin_layers(4096, 1024, NL, seed=5)
mo = random_marlin_layers(12288, 4096, 3, seed=3)
rt = torch.randn(288 * 4096, device=dev).to(torch.bfloat16)     # router stand-in 2.4 MB
x = torch.randn(M, 4096, device=dev).to(torch.bfloat16)
xa = torch.randn(M, 128, device=dev).to(torch.bfloat16)
x1 = torch.randn(M, 1024, device=dev).to(torch.bfloat16)
y_in = torch.empty(M, 12576, dtype=torch.bfloat16, device=dev)
y4 = torch.empty(M, 4096, dtype=torch.bfloat16, device=dev)
y4b = torch.empty(M, 4096, dtype=torch.bfloat16, device=dev)
y2 = torch.empty(M, 2048, dtype=torch.bfloat16, device=dev)
ym = torch.empty(M, 12288, dtype=torch.bfloat16, device=dev)
rsum = torch.empty((), dtype=torch.float32, device=dev)
h0 = [torch.randn(1, 2 << 18, device=dev) for _ in range(NL)]
nst = max(1, D // 2)
st = [torch.empty(nst, 2 << 18, device=dev) for _ in range(NL)]
side = torch.cuda.Stream(); aux = torch.cuda.Stream()


def g(y, xx, l, n, k, cfg):
    E.fp8_gemv_out(y, xx, l.weight, l.weight_scale.view(-1), None, n, k, *cfg, True)


def fork(tensors_bytes):
    e = torch.cuda.Event(); e.record()
    side.wait_event(e)
    with torch.cuda.stream(side):
        for t, nb in tensors_bytes:
            PX.l2_prefetch(t, 0, min(nb, t.numel() * t.element_size()) // 16 * 16, CTAS, POL)
    d = torch.cuda.Event(); d.record(side)
    return d


def layer(l, trig):
    ln = (l + 1) % NL
    d1 = d2 = None
    g(y_in, x, type("W", (), {"weight": in_w[l], "weight_scale": inp.weight_scale})(), 12576, 4096, (8, 4, 2, False))
    if "T1" in trig:
        d1 = fork([(op[l].weight, int(P1 * 2**20))])
    for f in fb:
        g(y4b, xa, f, 4096, 128, (8, 8, 2, True))
    if D:
        st[l].copy_(h0[l].expand(nst, -1))
    S.spin(int(SA * cyc), 48)
    if d1 is not None and not LATE:
        torch.cuda.current_stream().wait_event(d1)
    g(y4, x, op[l], 4096, 4096, (16, 8, 2, True))
    if d1 is not None and LATE:
        torch.cuda.current_stream().wait_event(d1)
    if "T2" in trig:
        d2 = fork([(sg[l].weight, 1 << 40), (sd[l].weight, 1 << 40)])
    S.spin(int(SB * cyc), 48)
    torch.sum(rt, dim=0, out=rsum)
    e = torch.cuda.Event(); e.record(); aux.wait_event(e)
    with torch.cuda.stream(aux):
        if d2 is not None and not LATE:
            aux.wait_event(d2)
        g(y2, x, sg[l], 2048, 4096, (16, 16, 2, True))
        if d2 is not None and LATE:
            aux.wait_event(d2)
        g(y4b, x1, sd[l], 4096, 1024, (16, 8, 2, True))
    for m in mo:
        g(ym, x, m, 12288, 4096, (16, 4, 2, True))
    ea = torch.cuda.Event(); ea.record(aux); torch.cuda.current_stream().wait_event(ea)
    for d in PEND:          # the previous layer's T3 prefetch (in_proj(l) was launched above)
        torch.cuda.current_stream().wait_event(d)
    PEND.clear()
    if "T3" in trig:
        d3 = fork([(in_w[ln], int(P3 * 2**20))])
        if LATE:
            PEND.append(d3)
        else:
            S.spin(int(SC * cyc), 48)
            torch.cuda.current_stream().wait_event(d3)
            return
    S.spin(int(SC * cyc), 48)


PEND = []


def run(trig):
    for l in range(NL):
        layer(l, trig)
    for d in PEND:
        torch.cuda.current_stream().wait_event(d)
    PEND.clear()


def capture(trig, reps=3):
    run(trig); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            run(trig)
    gr.replay(); torch.cuda.synchronize()
    return gr, reps * NL


TR = {"off": (), "T1": ("T1",), "T2": ("T2",), "T3": ("T3",), "all": ("T1", "T2", "T3")}
graphs = {k: capture(v) for k, v in TR.items()}
res = {k: [] for k in graphs}
for r in range(13):
    for key in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
        gr, n = graphs[key]
        ev0.record()
        for _ in range(3): gr.replay()
        ev1.record(); torch.cuda.synchronize()
        res[key].append(ev0.elapsed_time(ev1) * 1000 / (3 * n))
base = statistics.median(res["off"])
print(f"join {'late' if LATE else 'early'}; D={D} MiB, P1={P1} P3={P3} MiB, prefetch CTAs {CTAS} pol {POL}, windows SA/SB/SC {SA}/{SB}/{SC} us (+ kernels)")
for key in graphs:
    m = statistics.median(res[key]); rr = [p / q for p, q in zip(res[key], res["off"])]
    print(f"  {key:4s}: {m:8.1f} us/layer  saving {base - m:6.1f}  ratio med {statistics.median(rr):.4f} "
          f"[{min(rr):.4f}, {max(rr):.4f}]  spread {(max(res[key]) - min(res[key])) / m * 100:.1f}%", flush=True)
