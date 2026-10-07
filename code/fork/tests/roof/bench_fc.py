"""Paired CUDA-graph benchmark of kernels/fp8_gemv.cu configs vs production's Marlin on arbitrary (N, K) shapes
(tests/bench_fp8_gemv.py style: cold weights = a pool of independent layers > 4x L2, rotated; every round replays
Marlin and every candidate; median over rounds). Used for the drafter fc (4096 x 20480, GLM53_DRAFT_FP8=layers,fc),
which production serves with Marlin (not in fp8_gemv.TABLE).
Usage: bench_fc.py N K M1,M2,.. rounds [cfg;cfg;...]   (default candidates: every instantiated MB 1-2 config)"""
import sys, statistics
from pathlib import Path
R = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "tests"))
import torch
import fp8_gemv as G
from fp8_bench_common import random_marlin_layers, marlin

n, k = int(sys.argv[1]), int(sys.argv[2])
Ms = [int(v) for v in sys.argv[3].split(",")]
rounds = int(sys.argv[4])
if len(sys.argv) > 5:
    cands = [tuple(int(x) for x in c.split(",")) for c in sys.argv[5].split(";")]
    cands = [(w, kw, u, bool(p)) for w, kw, u, p in cands]
else:
    cands = [(8, kw, 2, p) for kw in (1, 2, 4, 8) for p in (False, True)] + [(16, kw, 2, p) for kw in (4, 8, 16) for p in (False, True)]
E = G.ext()
L2 = 24 << 20
nbytes = n * k
copies = max(2, -(-4 * L2 // nbytes))
layers = random_marlin_layers(n, k, copies)
reps = max(2 * copies, min(32, int(2e8 // nbytes)))


def capture(fns):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:2]:
            f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fns[i % len(fns)]()
    g.replay(); torch.cuda.synchronize()
    return g


print(f"N {n} K {k}: {copies} cold copies, {reps} calls per graph, {rounds} rounds; us per call (GB/s of fp8 bytes)")
for M in Ms:
    x = torch.randn(M, k, device="cuda").to(torch.bfloat16)
    o = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
    graphs = {"marlin": capture([lambda l=l: marlin(l, x, n, k) for l in layers])}
    for c in cands:
        try:
            E.fp8_gemv_out(o, x, layers[0].weight, layers[0].weight_scale.view(-1), None, n, k, *c, True)
        except RuntimeError:
            continue
        graphs[c] = capture([lambda l=l, c=c: E.fp8_gemv_out(o, x, l.weight, l.weight_scale.view(-1), None, n, k, *c, True)
                             for l in layers])
    res = {key: [] for key in graphs}
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for r in range(rounds):
        for key in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
            ev0.record(); graphs[key].replay(); graphs[key].replay(); ev1.record(); torch.cuda.synchronize()
            res[key].append(ev0.elapsed_time(ev1) * 1000 / (2 * reps))
    med = {key: statistics.median(v) for key, v in res.items()}
    tm = med["marlin"]
    ranked = sorted((key for key in med if key != "marlin"), key=med.get)
    best = ranked[0]
    ratios = [a / b for a, b in zip(res["marlin"], res[best])]
    print(f"M {M:3d}: Marlin {tm:8.1f} ({nbytes / tm / 1e3:5.1f})  best {best} {med[best]:8.1f} ({nbytes / med[best] / 1e3:5.1f})"
          f"  Marlin/best per round: med {statistics.median(ratios):.3f} min {min(ratios):.3f}  | "
          + " ".join(f"{c[0]},{c[1]},{int(c[3])}:{med[c]:.1f}" for c in ranked[:6]), flush=True)
    del graphs
