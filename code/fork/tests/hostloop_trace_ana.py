"""[dec-hostloop] one-rank decode host-loop analysis of a torch.profiler trace (glm53_runtime SIGUSR2 output):
  * host tail: from the return of the blocking per-step D2H copy (production _kv_lens_host: seq_lens.cpu(), a pageable
    cudaMemcpyAsync that waits for the drafter) to the launch of the eager target-embedding all-reduce, with the GPU
    work issued in that window (with GLM53_DEC_HOSTLOOP there is no such blocking copy: the section reports none);
  * collectives by position in the step (a step starts at the eager all-reduce): mean time inside the kernel beyond
    the transfer floor (waiting for the other rank) and mean GPU idle right before the kernel.
Usage: python3 tests/hostloop_trace_ana.py trace.json.gz [--floor-us 20.6] [--top 8]
"""
import argparse
import collections
import gzip
import json
import statistics as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--floor-us", type=float, default=20.6)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    t = json.load(gzip.open(a.trace) if a.trace.endswith(".gz") else open(a.trace))
    ev = t["traceEvents"]
    rts = sorted((e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("ph") == "X"),
                 key=lambda e: e["ts"])
    cpu = sorted((e for e in ev if e.get("cat") == "cpu_op" and e.get("ph") == "X"), key=lambda e: e["ts"])
    gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"),
                 key=lambda e: e["ts"])
    if not gpu:
        raise SystemExit("no GPU events in this trace (CPU-only session: see GLM53_DEC_PROF_DIAG)")
    corr = {e["args"]["correlation"]: e for e in rts if e.get("args", {}).get("correlation") is not None}
    eager_ar = []
    for g in gpu:
        if "AllReduce" in g["name"]:
            r = corr.get(g["args"].get("correlation"))
            if r is not None and r["name"] != "cudaGraphLaunch":
                eager_ar.append((g, r))
    blocking = [e for e in rts if e["name"] == "cudaMemcpyAsync" and e["dur"] > 1000]
    rows = []
    for b in blocking:
        t0 = b["ts"] + b["dur"]
        nxt = [x for x in eager_ar if x[1]["ts"] > t0]
        if not nxt or nxt[0][1]["ts"] - t0 > 10000:
            continue
        g, r = nxt[0]
        t1 = r["ts"]
        gw = [x for x in gpu if t0 <= x["ts"] < g["ts"]]
        la = collections.Counter(x["name"] for x in rts if t0 <= x["ts"] < t1)
        top, end = [], 0.0
        for x in cpu:
            if t0 <= x["ts"] < t1 and x["ts"] >= end:
                top.append(x)
                end = x["ts"] + x["dur"]
        rows.append(dict(block=b["dur"], tail=t1 - t0, gpu_ops=len(gw), gpu_busy=sum(x["dur"] for x in gw),
                         memcpy=la.get("cudaMemcpyAsync", 0),
                         launches=sum(v for k, v in la.items() if "Launch" in k and k != "cudaGraphLaunch"),
                         aten=sum(x["dur"] for x in top)))
    print(f"{a.trace}: host {t.get('host_name')}")
    if rows:
        m = lambda k: S.median(r[k] for r in rows)  # noqa: E731
        print("host tail after the blocking D2H (%d steps, medians): blocked %.1f ms, tail %.0f us, GPU ops issued %d "
              "(%d memcpy, %d kernel launches, %.0f us GPU busy), top-level aten %.0f us" % (
                  len(rows), m("block") / 1e3, m("tail"), m("gpu_ops"), m("memcpy"), m("launches"), m("gpu_busy"),
                  m("aten")))
    else:
        print("no blocking per-step D2H copy followed by the eager all-reduce (GLM53_DEC_HOSTLOOP active?)")
    steps, cur, prev_end = [], None, None
    for e in gpu:
        gap = e["ts"] - prev_end if prev_end is not None else 0.0
        prev_end = max(prev_end or 0.0, e["ts"] + e["dur"])
        if "nccl" not in e["name"]:
            continue
        r = corr.get(e["args"].get("correlation"))
        eager = r is not None and r["name"] != "cudaGraphLaunch"
        if eager and "AllReduce" in e["name"]:
            cur = []
            steps.append(cur)
        if cur is not None:
            cur.append((e["dur"], gap, "AG" if "AllGather" in e["name"] else "AR", eager))
    lens = [len(s) for s in steps if len(s) >= 20]
    if not lens:
        return
    n = S.mode(lens)
    steps = [s for s in steps if len(s) == n]
    ex = [S.mean(s[i][0] - a.floor_us for s in steps) for i in range(n)]
    gp = [S.mean(max(0.0, s[i][1]) for s in steps) for i in range(n)]
    print("collectives: %d steps x %d; per step mean %.2f ms inside collectives, %.2f ms beyond the %.1f us floor, "
          "%.2f ms GPU idle right before them" % (len(steps), n, S.mean(sum(x[0] for x in s) for s in steps) / 1e3,
                                                  sum(ex) / 1e3, a.floor_us, sum(gp) / 1e3))
    for i in sorted(range(n), key=lambda i: -(ex[i] + gp[i]))[: a.top]:
        print("  pos %3d %s %-5s mean beyond floor %6.0f us, idle before %6.0f us" % (
            i, steps[0][i][2], "eager" if steps[0][i][3] else "graph", ex[i], gp[i]))


if __name__ == "__main__":
    main()
