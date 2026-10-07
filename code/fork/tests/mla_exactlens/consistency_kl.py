#!/usr/bin/env python3
"""Summarize the handoff-mini decode-vs-prefill consistency of several runs (host only, no GPU).
Usage: python3 tests/mla_exactlens/consistency_kl.py <run dir>/result.json ...
KL(A_j || B_j) over the top-5 logprob union (tests/handoff/compare.py kl_top5): A = the decode step's logprobs,
B_j = a fresh prefill of prompt + A's first j tokens."""
import json, math, statistics, sys, os


def kl_top5(pa, pb):  # == tests/handoff/compare.py kl_top5 (copied: that module needs torch)
    keys = set(pa) | set(pb)
    if not keys:
        return float("nan")
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}
    lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values()))
    zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)

for p in sys.argv[1:]:
    r = json.load(open(p))
    ks, agree = [], 0
    for j, pb in enumerate(r.get("B", [])):
        pa = r["logprobs"][j]
        ks.append(kl_top5(pa, pb))
        agree += max(pa, key=pa.get) == max(pb, key=pb.get)
    ks_s = sorted(ks)
    n = len(ks)
    print(f"{r['label']:>10s}: n={n} KL mean {statistics.mean(ks):.5f} p50 {ks_s[n // 2]:.5f} "
          f"p95 {ks_s[int(n * 0.95)]:.5f} max {ks_s[-1]:.4f} | top-1 agree {agree}/{n} | gen[:12] {r['gen'][:12]}")
