#!/usr/bin/env python3
"""Summarize tests/w8a8layers/dvp_driver.py output (dvp.jsonl of one or more runs): per arm, the PAIRED excess over the
off arm on the same trajectories and positions (dvp_arm - dvp_off per position), a bootstrap 90 % interval over
positions (resampled within each trajectory), the share of the 'all' arm's excess, and the prefill-side KL pf.
Usage: summarize.py <run dir> [<run dir> ...] [--ref off] [--all all] [--trim 0]
"""
import argparse
import json
import random
import statistics as S
import sys
from collections import OrderedDict, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--ref", default="off")
ap.add_argument("--all", default="all")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--markdown", action="store_true")
A = ap.parse_args()


def load(run):
    arms = OrderedDict()
    for line in open(f"{run}/dvp.jsonl"):
        r = json.loads(line)
        arms.setdefault(r["arm"], {"expr": r["expr"], "traj": {}})["traj"][r["traj"]] = r
    return arms


def boot_ci(diffs_by_traj, n=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    tot = sum(len(v) for v in diffs_by_traj.values())
    acc = np.zeros(n)
    for k in sorted(diffs_by_traj):
        d = np.asarray(diffs_by_traj[k])
        acc += d[rng.integers(0, len(d), size=(n, len(d)))].sum(1)
    m = np.sort(acc / tot)
    return float(m[int(0.05 * n)]), float(m[int(0.95 * n)])


out = []
for run in A.runs:
    arms = load(run)
    if A.ref not in arms:
        sys.exit(f"{run}: no {A.ref} arm")
    ref = arms[A.ref]["traj"]
    allx = None
    if A.all in arms:
        allx = S.mean(v for t in arms[A.all]["traj"].values() for v in t["dvp"]) - \
            S.mean(v for t in ref.values() for v in t["dvp"])
    for lab, a in arms.items():
        d = defaultdict(list)
        pf, dvp, ev = [], [], 0
        for tn, t in a["traj"].items():
            r = ref[tn]
            for x, y in zip(t["dvp"], r["dvp"]):
                d[tn].append(x - y)
            pf += t["pf"]
            dvp += t["dvp"]
            ev += sum(1 for v in t["dvp"] if v > 0.05)
        diffs = [v for k in d for v in d[k]]
        ex = S.mean(diffs)
        lo, hi = boot_ci(d, A.boot) if lab != A.ref else (0.0, 0.0)
        pertraj = {k: S.mean(v) for k, v in sorted(d.items())}
        out.append({"run": run.rstrip("/").rsplit("/", 1)[-1], "arm": lab, "expr": a["expr"], "dvp": S.mean(dvp),
                    "excess": ex, "ci90": (lo, hi), "share_of_all": (ex / allx) if allx else None,
                    "pf": S.mean(pf), "events": ev, "per_traj": pertraj})

if A.markdown:
    print("| run | arm | expr | dvp | excess vs off | 90% CI | share of all | pf (KL B_off||B) | events>0.05 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        sh = f"{100 * r['share_of_all']:.0f} %" if r["share_of_all"] is not None else ""
        print(f"| {r['run']} | {r['arm']} | `{r['expr']}` | {r['dvp']:.5f} | {r['excess']:+.5f} | "
              f"[{r['ci90'][0]:+.5f}, {r['ci90'][1]:+.5f}] | {sh} | {r['pf']:.5f} | {r['events']} |")
else:
    for r in out:
        sh = f"{100 * r['share_of_all']:5.0f}%" if r["share_of_all"] is not None else "      "
        pt = " ".join(f"{k}:{v:+.5f}" for k, v in r["per_traj"].items())
        print(f"{r['run']:10s} {r['arm']:22s} dvp {r['dvp']:.5f} ex {r['excess']:+.5f} "
              f"[{r['ci90'][0]:+.5f},{r['ci90'][1]:+.5f}] {sh} pf {r['pf']:.5f} ev {r['events']:3d} | {pt}")
json.dump(out, open(f"{A.runs[-1]}/summary.json", "w"), indent=1)
