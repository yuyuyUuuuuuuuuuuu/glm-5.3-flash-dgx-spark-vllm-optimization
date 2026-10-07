#!/usr/bin/env python3
"""[opt-decodekit-rev] consistency_kl.py grouped by the greedy TRAJECTORY (host only).

The handoff mini's dvp KL is KL(decode_j || fresh prefill_j) along the run's OWN greedy continuation. Runs whose
continuation diverged (a near-tie flip early on: gen[4] 39408 vs 46779 on prompts 6000,3001) are measured on different
token sequences, so their KL means are not comparable. This prints, per run: the trajectory id (hash of the 96 generated
tokens), the mean, and the number of "events" (positions with top-5 KL > 0.02; the per-position top-5 KL saturates near
0.1, so the mean is ~ events x 0.08 / 96), and the event positions, so that identical-event (deterministic) runs are not
counted as independent samples.
Usage: python3 tests/mla_exactlens/consistency_kl_traj.py <run dir>/result.json ..."""
import hashlib, json, math, statistics, sys


def kl_top5(pa, pb):  # == tests/handoff/compare.py kl_top5
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}; lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values())); zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)


rows = []
for p in sys.argv[1:]:
    r = json.load(open(p))
    ks = [kl_top5(r["logprobs"][j], pb) for j, pb in enumerate(r["B"])]
    traj = hashlib.sha1(json.dumps(r["gen"][:len(ks)]).encode()).hexdigest()[:6]
    ev = tuple(j for j, v in enumerate(ks) if v > 0.02)
    rows.append((traj, r["label"], statistics.mean(ks), ev))
for traj, label, m, ev in sorted(rows):
    print(f"traj {traj}  {label:>10s}  KL mean {m:.5f}  events {len(ev):2d}  {list(ev)}")
