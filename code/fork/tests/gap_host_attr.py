"""Attribute the GPU-idle intervals inside a decode step to the host code that was running.

Counterpart to tests/hostloop_trace_ana.py (which reports the per-step totals and the collective
positions): here every idle gap >10 us between the step's GPU kernels/memcpys is paired with the
innermost host cpu_op / cuda_runtime event that covers the gap midpoint, so the gap says WHICH
host code produced it. "<HOST IDLE>" = no host op covered the midpoint: pure Python between ops.

Usage: python3 tests/gap_host_attr.py <torch.profiler trace (.json / .json.gz)> [--step-step]
Steps are delimited like dec_gaps.py in the profile's own tools: one marlin_gemm kernel per step.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics as S


def short(n: str) -> str:
    n = n.replace("void ", "").replace("(anonymous namespace)::", "")
    m = re.match(r"([\w:]+)", n)
    return (m.group(1) if m else n)[:56]


def load(path: str):
    with (gzip.open(path) if path.endswith(".gz") else open(path)) as f:
        d = json.load(f)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                  and e.get("ph") == "X"), key=lambda e: e["ts"])
    host = sorted((e for e in ev if e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver")
                   and e.get("ph") == "X"), key=lambda e: e["ts"])
    return gpu, host, d


def innermost(host, host_ts, t):
    i = bisect.bisect_right(host_ts, t) - 1
    while i >= 0:
        e = host[i]
        if e["ts"] <= t <= e["ts"] + e["dur"]:
            return e
        i -= 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--min-gap", type=float, default=10.0, help="us")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    gpu, host, d = load(a.trace)
    host_ts = [e["ts"] for e in host]
    mar = [i for i, e in enumerate(gpu) if "marlin::Marlin" in e["name"]]
    steps = list(zip(mar[:-1], mar[1:]))
    n = len(steps)
    step_ms = S.median((gpu[b]["ts"] - gpu[a]["ts"]) / 1e3 for a, b in steps)
    gap = collections.defaultdict(lambda: [0, 0.0])          # host op -> [count, us]
    chains = collections.defaultdict(lambda: [0, 0.0])       # last GPU op before the gap -> host op
    tot = 0.0
    for a_i, b_i in steps:
        end = gpu[a_i]["ts"] + gpu[a_i]["dur"]
        prev = gpu[a_i]["name"]
        for e in gpu[a_i + 1:b_i + 1]:
            g = e["ts"] - end
            if g > a.min_gap:
                tot += g
                cov = innermost(host, host_ts, e["ts"] - g / 2)
                key = short(cov["name"]) if cov else "<HOST IDLE: python between ops>"
                gap[key][0] += 1
                gap[key][1] += g
                chains[(short(prev), key)][0] += 1
                chains[(short(prev), key)][1] += g
            if e["ts"] + e["dur"] > end:
                end = e["ts"] + e["dur"]
                prev = e["name"]
    print(f"{a.trace}: {n} decode steps, step median {step_ms:.2f} ms; GPU idle >{a.min_gap} us per step "
          f"{tot / n / 1e3:.3f} ms ({100 * tot / (n * step_ms * 1e3):.2f} % of the step)")
    print("\nby host code that was running at the gap midpoint:")
    print("%-56s %8s %10s %10s" % ("host op", "gaps/st", "us/gap", "ms/step"))
    for key, (c, t) in sorted(gap.items(), key=lambda x: -x[1][1])[:a.top]:
        print("%-56s %8.1f %10.1f %10.3f" % (key, c / n, t / c, t / n / 1e3))
    print("\ntop idle chains (previous GPU kernel -> host code):")
    for (p, key), (c, t) in sorted(chains.items(), key=lambda x: -x[1][1])[:a.top]:
        if t / n < 5:
            continue
        print("%7.1f us/step  x%4.1f  %-22s %s" % (t / n, c / n, p, key))


if __name__ == "__main__":
    main()
