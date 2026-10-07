"""Paired benchmark: kernels/fp8_gemv.cu vs production's Marlin FP8 (apply_fp8_marlin_linear) on every production
FP8-Marlin decode shape, cold weights (a pool of independent layers > 4x L2, rotated), CUDA-graph replay, interleaved
rounds (every round replays Marlin and every candidate once; median over rounds).

  --tune:  all candidate configs per (shape, M); prints the best per (shape, MB) as a Python table
  default: the shipped selection (fp8_gemv.select_config) only; prints the report table
Usage: bench_fp8_gemv.py [--tune] [--m 1,5,8,16] [--shapes i,j] [--rounds 7]
"""
import argparse, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import fp8_gemv as G
from fp8_bench_common import PROD_SHAPES, PROD_R7_US, PROD_R7_UNPAD_COPY_US, UNIQUE_SHAPES, random_marlin_layers, marlin

ap = argparse.ArgumentParser()
ap.add_argument("--tune", action="store_true")
ap.add_argument("--m", default="1,2,4,5,6,7,8,10,12,15,16,18,20,24,25,30,32,40,48,56,64")
ap.add_argument("--shapes", default="")
ap.add_argument("--rounds", type=int, default=7)
a = ap.parse_args()
dev = "cuda"
E = G.ext()
L2 = 24 << 20
Ms = [int(v) for v in a.m.split(",")]
sel = [int(v) for v in a.shapes.split(",")] if a.shapes else list(range(len(UNIQUE_SHAPES)))


def candidates(mb, n, k):
    if not a.tune:
        c = G.select_config(n, k, 8 * mb)
        return [c] if c is not None else []
    c = [(8, kw, 2, True) for kw in (1, 2, 4, 8)]
    if mb == 1:
        c += [(8, kw, 2, False) for kw in (2, 4, 8)]
    if mb <= 2:
        c += [(16, kw, 2, True) for kw in (4, 8, 16)]
    if mb >= 3:
        c += [(4, kw, 2, True) for kw in (1, 2, 4)]
    return c


def capture(fn_list, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list[:3]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn_list[i % len(fn_list)]()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_graph(g, replays):
    x, y = torch.cuda.Event(True), torch.cuda.Event(True)
    x.record()
    for _ in range(replays):
        g.replay()
    y.record()
    torch.cuda.synchronize()
    return x.elapsed_time(y) * 1000


best = {}
meas = {}   # (N, K, M) -> (marlin us, new us or None)
print(f"GPU {torch.cuda.get_device_name()} rounds={a.rounds} (median us per call; GB/s = (N*K fp8 + N*2 scale bytes) / t)")
print(f"{'N':>6s} {'K':>5s} {'M':>3s} | {'marlin us':>9s} {'GB/s':>6s} | {'new us':>8s} {'GB/s':>6s} | {'x':>6s} | config (W,KW,U,pol)  spread%")
for si in sel:
    n, k = UNIQUE_SHAPES[si]
    nbytes = n * k
    copies = max(2, -(-4 * L2 // nbytes))
    layers = random_marlin_layers(n, k, copies)
    npad = layers[0].weight.shape[1] // 4
    reps = max(2 * copies, min(64, int(3e8 // nbytes)))
    for M in Ms:
        mb = (M + 7) // 8
        x = torch.randn(M, k, device=dev).to(torch.bfloat16)
        outs = [torch.empty(M, n, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        graphs = {"marlin": capture([lambda l=l: marlin(l, x, n, k) for l in layers], reps)}
        for cfg in candidates(mb, npad, k):
            w, kw, u, pol = cfg
            graphs[cfg] = capture([lambda l=l, o=outs[i % 2]: E.fp8_gemv_out(o, x, l.weight, l.weight_scale.view(-1),
                                                                               None, n, k, w, kw, u, pol, True)
                                   for i, l in enumerate(layers)], reps)
        res = {key: [] for key in graphs}
        for r in range(a.rounds):
            for key, g in graphs.items():
                res[key].append(time_graph(g, 3) / (3 * reps))
        med = {key: statistics.median(v) for key, v in res.items()}
        spread = {key: (max(v) - min(v)) / statistics.median(v) * 100 for key, v in res.items()}
        tm = med.pop("marlin")
        if med:
            cfg = min(med, key=med.get)
            t = med[cfg]
            best.setdefault((npad, k), {})[mb] = best.get((npad, k), {}).get(mb, []) + [(M, cfg, t, tm)]
            meas[(n, k, M)] = (tm, t)
            print(f"{n:6d} {k:5d} {M:3d} | {tm:9.2f} {(nbytes + 2*n)/tm/1e3:6.1f} | {t:8.2f} {(nbytes + 2*n)/t/1e3:6.1f} | "
                  f"{tm/t:6.3f} | {cfg}  {spread[cfg]:.1f}/{spread.get('marlin', 0):.1f}"
                  + ("  [" + " ".join(f"{c[0]},{c[1]},{c[2]},{int(c[3])}:{v:.1f}" for c, v in sorted(med.items(), key=lambda z: z[1])[:5]) + "]" if a.tune else ""),
                  flush=True)
        else:
            meas[(n, k, M)] = (tm, None)
            print(f"{n:6d} {k:5d} {M:3d} | {tm:9.2f} {(nbytes + 2*n)/tm/1e3:6.1f} | {'Marlin (not selected)':>24s}", flush=True)
        del graphs
    del layers
    torch.cuda.empty_cache()
if a.tune:
    print("\n# best config per (Npad, K) and MB (M values measured, their best configs)")
    for key, d in best.items():
        print(key, {mb: [(M, c, round(tm / t, 3)) for M, c, t, tm in v] for mb, v in sorted(d.items())})


def predict(label, m_target, m_draft, m_head):
    """Per decode step (one rank): nodeC saving sum(calls * (marlin - new)) and production-scaled saving
    sum(calls * prod_us * (1 - new / marlin)) (+ the KDA in_proj unpad copy the new kernel does not need)."""
    s1 = s2 = base = 0.0
    rows = []
    for name, (n, k, calls) in PROD_SHAPES.items():
        M = m_head if name == "draft.lmhead" else m_draft if name.startswith("draft.") else m_target
        if (n, k, M) not in meas:
            return None
        tm, t = meas[(n, k, M)]
        base += calls * PROD_R7_US[name]
        if t is None:
            rows.append(f"{name}:Marlin")
            continue
        d1 = calls * (tm - t)
        d2 = calls * PROD_R7_US[name] * (1 - t / tm)
        if name == "kda.in_proj":
            d2 += calls * PROD_R7_UNPAD_COPY_US
        s1 += d1
        s2 += d2
        rows.append(f"{name}:{d2/1000:.3f}")
    print(f"{label}: nodeC saving {s1/1000:.3f} ms/step; production-scaled {s2/1000:.3f} ms/step = "
          f"{s2/base*100:.1f}% of the {base/1000:.2f} ms of FP8-Marlin calls per step (R7 profile)")
    print("   per linear (ms/step, production-scaled): " + " ".join(rows))


if not a.tune and not a.shapes:
    print()
    for label, mt, md, mh in (("batch 1, K=7 (M=8, drafter 8, lm_head 7)", 8, 8, 7),
                              ("batch 1, K=5 (M=6, drafter 8, lm_head 7)", 6, 8, 7),
                              ("batch 1, K=4 (M=5, drafter 8, lm_head 7)", 5, 8, 7),
                              ("batch 2, K=7 (M=16, drafter 16, lm_head 15)", 16, 16, 15),
                              ("batch 4, K=7 (M=32, drafter 32, lm_head 30)", 32, 32, 30),
                              ("batch 8, K=7 (M=64, drafter 64, lm_head 56)", 64, 64, 56)):
        predict(label, mt, md, mh)
