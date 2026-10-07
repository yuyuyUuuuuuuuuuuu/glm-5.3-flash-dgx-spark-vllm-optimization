# copied from decode4-b-spec tests/vtrim/bench_dead_rows.py: live-row MoE layer cost per row count (decode5 breakdown)
"""Decode MoE layer cost of DEAD verify rows (routed ids = -1, the TF sentinel) vs LIVE rows (docs/SPEC_VTRIM.md).

Production-shaped decode MoE layer (tests/moeglue_rig.py: router GEMV, grouped_topk, production's apply_exl3_experts
with the K2 TF apply hook, shared expert on the aux stream), 42 layers per graph (one decode step's MoE), corr40
routing at T = 8 rows. Modes:
  full<T>   : T live rows (T = 8 - d), the rows' own x / ids
  dead<d>   : 8 rows, the last d rows' routed ids set to -1 before the apply (what GLM53_SPEC_VTRIM does)
Paired, interleaved rounds (MR.ab_rounds). Output: per-layer-call median us per mode and the dead-row saving per row.
Usage: tests/gpu_run.sh python3 -u tests/vtrim/bench_dead_rows.py [--dead 0,1,2,3,4] [--rounds 101]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

import harness as H  # noqa: E402
import moeglue_rig as MR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead", default="0,1,2,3,4,5")
    ap.add_argument("--rounds", type=int, default=101)
    ap.add_argument("--layers", type=int, default=42)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--kind", default="corr40")
    a = ap.parse_args()
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    H.load_tf()
    import integrate

    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    print(f"K2 apply hook {'on' if rep.get('apply_hook') else 'off'}", flush=True)
    layers = MR.make_layers(prod, dev, 3, a.layers)
    rig = MR.Rig(prod, layers, dev)
    T = a.T
    sets = MR.make_sets(a.kind, T, a.layers, 1, dev, seed=4242 + T)
    graphs, outs = {}, {}
    for d in [int(x) for x in a.dead.split(",")]:
        live = T - d
        dead_sets = []
        full_sets = []
        for (li, x, ids, w) in sets:
            idd = ids.clone()
            if d:
                idd[live:] = -1
            # NOTE: run ONE d per process: two B=8 graphs over the same x tensor interfere in this rig (the second
            # capture's routing is replayed by both; cloning x per graph produced NaNs), so compare dead<d> with
            # full<8-d> inside one process and full8 from a --dead 0 process.
            dead_sets.append((li, x, idd.contiguous(), w))
            full_sets.append((li, x[:live].contiguous(), ids[:live].contiguous(), w[:live].contiguous()))
        for name, ss in ((f"dead{d}", dead_sets), (f"full{live}", full_sets)):
            if name in graphs:
                continue
            fns = [lambda s=s: rig.call(s[0], s[1], s[2], s[3]) for s in ss]
            graphs[name], outs[name] = MR.capture(fns, dev)
    # numerics: the live rows of dead<d> must equal full<T-d> (routes of live rows are unchanged)
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    for d in [int(x) for x in a.dead.split(",")]:
        live = T - d
        same = all(torch.equal(o[:live], o0) for o, o0 in zip(outs[f"dead{d}"], outs[f"full{live}"]))
        worst = max(((o[:live].float() - o0.float()).norm() / o0.float().norm().clamp_min(1e-30)).item()
                    for o, o0 in zip(outs[f"dead{d}"], outs[f"full{live}"]))
        print(f"check dead{d} live rows vs full{live}: bitwise {same} rel_l2 max {worst:.2e}", flush=True)
    res = MR.ab_rounds(graphs, a.layers, a.rounds, 1)
    base = f"full{T}" if f"full{T}" in res else None
    print(f"{a.kind} T={T}, {a.layers} layers/graph, per layer call (median us over {a.rounds} rounds):", flush=True)
    for name in sorted(graphs, key=lambda n: (n[:4], int(n[4:]))):
        print(f"  {name:8s} {MR.med(res[name]):8.1f} us  (p10-p90 {MR.pct(res[name], .1):.1f}-{MR.pct(res[name], .9):.1f})",
              flush=True)
    if base is None:
        for d in [int(x) for x in a.dead.split(",")]:
            print(f"  d={d}: dead - full = {MR.med(res[f'dead{d}']) - MR.med(res[f'full{T - d}']):+.1f} us/layer", flush=True)
        H.report_peak(8.0)
        return
    f8 = MR.med(res[base])
    for d in [int(x) for x in a.dead.split(",")]:
        if d == 0:
            continue
        dm, fl = MR.med(res[f"dead{d}"]), MR.med(res[f"full{T - d}"])
        print(f"  d={d}: dead saves {(f8 - dm) / d:.1f} us/row/layer (full rows save {(f8 - fl) / d:.1f}); "
              f"x{a.layers} layers: {(f8 - dm) / d * a.layers / 1000:.2f} ms/row/step (full {(f8 - fl) / d * a.layers / 1000:.2f})",
              flush=True)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
