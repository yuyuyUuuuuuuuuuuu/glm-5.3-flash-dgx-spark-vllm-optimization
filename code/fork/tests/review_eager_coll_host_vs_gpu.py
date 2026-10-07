"""[hostgap review] Is an eager collective's HOST return a shared clock? No: vLLM's pynccl.all_reduce only enqueues
ncclAllReduce on the current stream. For every eager vllm::all_reduce / vllm::all_gather cpu_op of a torch.profiler
trace, pair it with the NCCL kernel it launched (correlation id of the last cuda_runtime call inside the op) and print
the host op duration, kernel start - host return, kernel duration and kernel END - host return.
Usage: python3 tests/review_eager_coll_host_vs_gpu.py <trace.json.gz>
"""
import gzip
import json
import statistics as S
import sys

t = json.load(gzip.open(sys.argv[1]) if sys.argv[1].endswith(".gz") else open(sys.argv[1]))
ev = t["traceEvents"]
ops = sorted([e for e in ev if e.get("cat") == "cpu_op" and e.get("name") in ("vllm::all_reduce", "vllm::all_gather")
              and e.get("ph") == "X"], key=lambda e: e["ts"])
rt = sorted([e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"],
            key=lambda e: e["ts"])
kern = {e["args"].get("correlation"): e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"
        and "nccl" in e["name"].lower()}
rows = []
for o in ops:
    a, b = o["ts"], o["ts"] + o["dur"]
    ls = [r for r in rt if a <= r["ts"] <= b and r["args"].get("correlation") in kern]
    if not ls:
        continue
    k = kern[ls[-1]["args"]["correlation"]]
    rows.append((o["name"], o["dur"], k["ts"] - b, k["dur"], k["ts"] + k["dur"] - b))
print(f"{sys.argv[1]}: {len(rows)} eager collectives paired with their NCCL kernel (us)")
for name in ("vllm::all_reduce", "vllm::all_gather"):
    r = [x for x in rows if x[0] == name]
    if not r:
        continue
    print(f"{name:18s} n={len(r)}  host op med {S.median(x[1] for x in r):.1f} | kernel start - host return med "
          f"{S.median(x[2] for x in r):.1f} | kernel dur med {S.median(x[3] for x in r):.1f} | kernel END - host "
          f"return med {S.median(x[4] for x in r):.1f} (min {min(x[4] for x in r):.1f}, max {max(x[4] for x in r):.1f})")
