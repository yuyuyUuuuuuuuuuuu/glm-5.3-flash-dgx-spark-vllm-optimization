"""[hostgap review] tests/gap_host_attr.py credits a GPU-idle gap to the host op covering its midpoint. For the gaps it
credits to cudaMemcpyAsync (the planner's blocking seq_lens.cpu() D2H), print which GPU op ENDS the gap: the host is
blocked in that copy, so the GPU is not waiting for the host there but for something inside its own queue.
Usage: python3 tests/review_gap_memcpy_next.py <trace.json.gz>   (steps = marlin kernel to marlin kernel, gaps > 10 us)
"""
import bisect
import collections
import gzip
import json
import sys

d = json.load(gzip.open(sys.argv[1]) if sys.argv[1].endswith(".gz") else open(sys.argv[1]))
ev = d["traceEvents"]
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"),
             key=lambda e: e["ts"])
host = sorted((e for e in ev if e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver") and e.get("ph") == "X"),
              key=lambda e: e["ts"])
hts = [e["ts"] for e in host]


def inner(t):
    i = bisect.bisect_right(hts, t) - 1
    while i >= 0:
        e = host[i]
        if e["ts"] <= t <= e["ts"] + e["dur"]:
            return e
        i -= 1
    return None


mar = [i for i, e in enumerate(gpu) if "marlin::Marlin" in e["name"]]
steps = list(zip(mar[:-1], mar[1:]))
n = len(steps)
agg = collections.defaultdict(lambda: [0, 0.0])
tot = 0.0
for a, b in steps:
    end = gpu[a]["ts"] + gpu[a]["dur"]
    for e in gpu[a + 1:b + 1]:
        g = e["ts"] - end
        if g > 10:
            c = inner(e["ts"] - g / 2)
            if c and c["name"] == "cudaMemcpyAsync":
                agg[e["name"][:64]][0] += 1
                agg[e["name"][:64]][1] += g
                tot += g
        end = max(end, e["ts"] + e["dur"])
print(f"{sys.argv[1]}: {n} steps; gaps credited to cudaMemcpyAsync: {tot / n / 1e3:.3f} ms/step, ended by:")
for k, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print("  %-64s gaps/step %.2f  us/gap %6.1f  ms/step %.3f" % (k, c / n, t / c, t / n / 1e3))
