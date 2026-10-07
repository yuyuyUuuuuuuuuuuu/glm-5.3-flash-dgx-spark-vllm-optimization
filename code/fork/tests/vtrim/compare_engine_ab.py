#!/usr/bin/env python3
"""Compare the greedy outputs of tests/vtrim/run_engine_ab.sh labels (result.json per label): per request the first
position where two labels' generated ids differ (len = identical), plus every [glm53-spec-vtrim] line of each label.
Usage: compare_engine_ab.py <out dir> [labels...]"""
import json, os, re, sys

out = sys.argv[1]
labels = sys.argv[2:] or ["off_a", "off_b", "shadow30", "on30", "on100"]
R = {}
for l in labels:
    p = os.path.join(out, l, "result.json")
    if os.path.exists(p):
        R[l] = [r["gen"] for r in json.load(open(p))["requests"]]
    else:
        print(f"{l}: no result.json")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


ref = labels[0]
for l in R:
    if l == ref or ref not in R:
        continue
    fd = [first_diff(a, b) for a, b in zip(R[ref], R[l])]
    n = [len(a) for a in R[ref]]
    print(f"{ref} vs {l}: first differing position per request {fd} of {n} -> "
          f"{'IDENTICAL' if fd == n else 'differs'}")
for l in labels:
    for f in ("container.log", "harness.log"):
        p = os.path.join(out, l, f)
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            if "glm53-spec-vtrim" in line or "glm53_spec_vtrim" in line:
                print(f"  [{l}] " + re.sub(r"^.*?(\[glm53-spec-vtrim\]|glm53_spec_vtrim)", r"\1", line.rstrip())[:260])
        break
