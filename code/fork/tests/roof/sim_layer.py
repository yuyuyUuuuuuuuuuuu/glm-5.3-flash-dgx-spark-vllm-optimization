"""Node1 simulation of production's KDA+MoE decode layer sequence, to measure the NET effect of the L2 eviction
policy of the KDA in_proj FP8 GEMV (the per-layer bimodality of production's in_proj is reproduced by dirty L2 lines:
tests/roof/probe_copies.py). Per simulated layer (4 layers of distinct weights cycled, captured in one CUDA graph):
  in_proj  (12608 x 4096 Marlin FP8, the config under test)
  f_b, g_b (4096 x 128)
  state    torch copy: read the 2 MiB initial state, write D MiB of per-token states (normal stores = dirty L2 lines,
           as fused_recurrent_gated_delta_rule's spec-decode per-token ht stores)
  o_proj   (4096 x 4096, production config (16, 8, 2, evict_first))
  moe      evict_first streaming read of ~150 MB (3 x 12288 x 4096 FP8 GEMV, pol on) standing in for the TF MoE kernel
           (which loads weights with L2::evict_first)
Configs A / B for in_proj are replayed in alternating rounds; reported: median us per simulated layer, and the in_proj
share (separately captured graphs of the same sequence with in_proj removed give the rest).
Usage: sim_layer.py D_MiB rounds cfgA cfgB  (cfg = W,KW,U,pol)"""
import sys, statistics
from pathlib import Path
R = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "tests"))
import torch
import fp8_gemv as G
from fp8_bench_common import random_marlin_layers

D = int(sys.argv[1]); rounds = int(sys.argv[2])
cfgs = []
for c in sys.argv[3:5]:
    w, kw, u, p = (int(v) for v in c.split(","))
    cfgs.append((w, kw, u, bool(p)))
E = G.ext()
M = int(__import__("os").environ.get("SIM_M", "5"))
NL = 4
dev = "cuda"
inp = random_marlin_layers(12576, 4096, 1)[0]
inw = [inp.weight] + [inp.weight.clone() for _ in range(NL - 1)]
insc = inp.weight_scale.view(-1)
fb = random_marlin_layers(4096, 128, 2, seed=1)
op = random_marlin_layers(4096, 4096, NL, seed=2)
mo = random_marlin_layers(12288, 4096, 3, seed=3)
x = torch.randn(M, 4096, device=dev).to(torch.bfloat16)
xa = torch.randn(M, 128, device=dev).to(torch.bfloat16)
y_in = torch.empty(M, 12576, dtype=torch.bfloat16, device=dev)
y_s = torch.empty(M, 4096, dtype=torch.bfloat16, device=dev)
y_m = torch.empty(M, 12288, dtype=torch.bfloat16, device=dev)
h0 = [torch.randn(1, 2 << 18, device=dev) for _ in range(NL)]          # 2 MiB fp32 initial state per layer
nst = max(1, D // 2)
st = [torch.empty(nst, 2 << 18, device=dev) for _ in range(NL)]       # D MiB of per-token states per layer


def layer(l, cfg, with_in=True):
    if with_in:
        E.fp8_gemv_out(y_in, x, inw[l], insc, None, 12576, 4096, *cfg, True)
    for f in fb:
        E.fp8_gemv_out(y_s, xa, f.weight, f.weight_scale.view(-1), None, 4096, 128, 8, 8, 2, True, True)
    if D:
        st[l].copy_(h0[l].expand(nst, -1))
    E.fp8_gemv_out(y_s, x, op[l].weight, op[l].weight_scale.view(-1), None, 4096, 4096, 16, 8, 2, True, True)
    for m in mo:
        E.fp8_gemv_out(y_m, x, m.weight, m.weight_scale.view(-1), None, 12288, 4096, 16, 4, 2, True, True)


def capture(cfg, with_in=True, reps=3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for l in range(NL):
            layer(l, cfg, with_in)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for l in range(NL):
                layer(l, cfg, with_in)
    g.replay(); torch.cuda.synchronize()
    return g, reps * NL


graphs = {("A", True): capture(cfgs[0]), ("B", True): capture(cfgs[1]), ("rest", False): capture(cfgs[0], False)}
res = {k: [] for k in graphs}
for r in range(rounds):
    keys = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
    for key in keys:
        g, n = graphs[key]
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(3):
            g.replay()
        b.record(); torch.cuda.synchronize()
        res[key].append(a.elapsed_time(b) * 1000 / (3 * n))
med = {k: statistics.median(v) for k, v in res.items()}
sp = {k: (max(v) - min(v)) / statistics.median(v) * 100 for k, v in res.items()}
rest = med[("rest", False)]
print(f"D={D} MiB dirty per layer, M={M}, {rounds} rounds: us per simulated layer (spread %)")
for key, cfg in ((("A", True), cfgs[0]), (("B", True), cfgs[1])):
    print(f"  in_proj {cfg}: layer {med[key]:8.1f} ({sp[key]:.1f}%)  -> in_proj share {med[key] - rest:7.1f} us")
print(f"  rest (no in_proj): {rest:8.1f} ({sp[('rest', False)]:.1f}%)")
pr = [b / a for a, b in zip(res[("A", True)], res[("B", True)])]
print(f"  B/A layer ratio per round: median {statistics.median(pr):.4f} min {min(pr):.4f} max {max(pr):.4f}")
