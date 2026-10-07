"""Review check (decode4 B): do DEAD rows change the LIVE rows of the production decode MoE layer at the SAME row count?

bench_dead_rows.py compares dead<d> (8 rows, d dead) with full<8-d> (8-d rows): a different T, so a different kernel
schedule, and its rel_l2 jumps from 3e-5 (d<=2) to 9e-4 (d>=4). In production a trimmed step keeps T = K+1 rows, so
the question is whether the live rows of dead<d> equal the live rows of full8 (the same 8 rows, nothing dead) within
the A/A spread of full8 itself. Eager calls (no graph; the graph replays the same kernels), production's
apply_exl3_experts through the TF K2 apply hook, corr40 routing, 42 layers.
Usage: tests/r16/gpu.sh python3 -u tests/vtrim/check_dead_vs_full8.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

import harness as H  # noqa: E402
import moeglue_rig as MR  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    H.load_tf()
    import integrate

    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    print(f"K2 apply hook {'on' if rep.get('apply_hook') else 'off'}", flush=True)
    L = 42
    layers = MR.make_layers(prod, dev, 3, L)
    rig = MR.Rig(prod, layers, dev)
    T = 8
    sets = MR.make_sets("corr40", T, L, 1, dev, seed=4242 + T)
    full = [rig.call(li, x, ids, w).clone() for (li, x, ids, w) in sets]
    torch.cuda.synchronize()
    aa = []
    for rep_i in range(3):
        again = [rig.call(li, x, ids, w).clone() for (li, x, ids, w) in sets]
        torch.cuda.synchronize()
        aa.append(max(rel(o, f) for o, f in zip(again, full)))
    print(f"A/A full8 vs full8 (3 repeats): rel_l2 max per repeat {['%.2e' % v for v in aa]}", flush=True)
    worst_ok = True
    for d in range(1, 7):
        live = T - d
        outs = []
        for (li, x, ids, w) in sets:
            idd = ids.clone()
            idd[live:] = -1
            outs.append(rig.call(li, x, idd.contiguous(), w).clone())
        torch.cuda.synchronize()
        r = max(rel(o[:live], f[:live]) for o, f in zip(outs, full))
        fin = all(torch.isfinite(o).all().item() for o in outs)
        same = all(torch.equal(o[:live], f[:live]) for o, f in zip(outs, full))
        ok = fin and r <= max(10 * max(aa), 1e-4)
        worst_ok &= ok
        print(f"dead{d} live rows vs full8 live rows: bitwise {same} rel_l2 max {r:.2e} finite {fin} -> "
              f"{'PASS' if ok else 'FAIL'} (bound max(10 x A/A, 1e-4))", flush=True)
    print("ALL PASSED" if worst_ok else "SOME FAILED", flush=True)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
