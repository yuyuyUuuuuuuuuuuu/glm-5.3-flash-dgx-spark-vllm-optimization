"""[dec-hostloop] rank skew from two torch.profiler traces of the same decode steps (rank 0 = head, rank 1 = worker),
e.g. the two files the glm53_runtime SIGUSR2 profiler writes when both worker processes are signalled together.

Clock alignment without trusting the nodes' clocks: a ring all-reduce / all-gather finishes on both ranks at (almost)
the same instant, so offset = median over matched collectives of (end_r1 - end_r0). Collectives are matched by their
order inside a step (a step starts at the eager AllReduce = target embedding all-reduce; production decode has 104
collectives per step: 1 eager AR, 101 in-graph ARs, 1 eager + 1 in-graph AllGather).
After alignment, for every collective position:
  arrive_diff = start_r1 - start_r0   (> 0: rank 1 arrives later, rank 0 waits that long inside the kernel)
  wait_r0 / wait_r1 = duration - min(duration_r0, duration_r1) (time spent waiting for the other rank)
and per rank the GPU-idle time before each collective (gap to the previous kernel on that rank).
Usage: python3 tests/skew_from_traces.py rank0.json.gz rank1.json.gz [--top 12]
"""
import argparse
import gzip
import json
import statistics as S


def load(path):
    with (gzip.open(path) if path.endswith(".gz") else open(path)) as f:
        t = json.load(f)
    ev = t["traceEvents"]
    rt = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                rt[c] = e["name"]
    gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"),
                 key=lambda e: e["ts"])
    rt_ev = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
             and e.get("ph") == "X" and e.get("args", {}).get("correlation") is not None}
    blocking = sorted(e["ts"] + e["dur"] for e in ev if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
                      and e["name"] == "cudaMemcpyAsync" and e["dur"] > 1000)
    if not gpu:
        raise SystemExit(f"{path}: no GPU events (CUPTI recorded nothing: see GLM53_DEC_PROF_DIAG)")
    steps, cur, prev_end = [], None, None
    for e in gpu:
        gap = e["ts"] - prev_end if prev_end is not None else 0.0
        prev_end = max(prev_end or 0.0, e["ts"] + e["dur"])
        if "nccl" not in e["name"]:
            continue
        eager = rt.get(e["args"].get("correlation")) != "cudaGraphLaunch"
        if eager and "AllReduce" in e["name"]:
            cur = []
            steps.append(cur)
        if cur is not None:
            launch = rt_ev.get(e["args"].get("correlation"))
            host = {}
            if eager and launch is not None:
                import bisect
                j = bisect.bisect_right(blocking, launch["ts"]) - 1
                host = dict(launch=launch["ts"], copy_ret=blocking[j] if j >= 0 and
                            launch["ts"] - blocking[j] < 10000 else None)
            cur.append(dict(ts=e["ts"], end=e["ts"] + e["dur"], dur=e["dur"], gap=gap, eager=eager,
                            kind=("AG" if "AllGather" in e["name"] else "AR"), **host))
    long_steps = [len(s) for s in steps if len(s) >= 20]      # decode steps (prefill / warm-up windows differ)
    n = S.mode(long_steps) if long_steps else 0
    return [s for s in steps if len(s) == n], t.get("host_name"), t.get("distributedInfo", {}).get("rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r0")
    ap.add_argument("r1")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    s0, h0, k0 = load(a.r0)
    s1, h1, k1 = load(a.r1)
    if not s0 or not s1 or len(s0[0]) != len(s1[0]):
        raise SystemExit(f"collective structure differs: {len(s0[0]) if s0 else 0} vs {len(s1[0]) if s1 else 0}")
    n = len(s0[0])
    # match steps by the sequence of step periods (adaptive K and acceptance make it distinctive), not by the clocks
    p0 = [b[0]["end"] - a[0]["end"] for a, b in zip(s0, s0[1:])]
    p1 = [b[0]["end"] - a[0]["end"] for a, b in zip(s1, s1[1:])]
    best = None
    for sh in range(-len(p1) + 1, len(p0)):
        idx = [i for i in range(len(p0)) if 0 <= i - sh < len(p1)]
        if len(idx) < min(8, len(p0), len(p1)):
            continue
        cost = S.median(abs(p0[i] - p1[i - sh]) for i in idx)
        if best is None or cost < best[0]:
            best = (cost, sh)
    if best is None:
        raise SystemExit("no overlapping decode steps")
    sh = best[1]
    pairs = [(s0[i], s1[i - sh]) for i in range(len(s0)) if 0 <= i - sh < len(s1)]
    off = S.median(c1["end"] - c0["end"] for p0, p1 in pairs for c0, c1 in zip(p0, p1))
    resid = [abs((c1["end"] - off) - c0["end"]) for p0, p1 in pairs for c0, c1 in zip(p0, p1)]
    print(f"rank0 {h0} (dist rank {k0}), rank1 {h1} (dist rank {k1}): {len(pairs)} matched steps x {n} collectives; "
          f"clock offset r1-r0 {off / 1e3:.3f} ms (end-time residual med {S.median(resid):.1f} us, "
          f"p90 {sorted(resid)[int(.9 * len(resid))]:.1f} us; step-period match cost {best[0]:.1f} us)")
    rows = []
    for i in range(n):
        ad = [(p1[i]["ts"] - off) - p0[i]["ts"] for p0, p1 in pairs]
        d0 = [p0[i]["dur"] for p0, p1 in pairs]
        d1 = [p1[i]["dur"] for p0, p1 in pairs]
        w0 = [x - min(x, y) for x, y in zip(d0, d1)]
        w1 = [y - min(x, y) for x, y in zip(d0, d1)]
        g0 = [p0[i]["gap"] for p0, p1 in pairs]
        g1 = [p1[i]["gap"] for p0, p1 in pairs]
        rows.append(dict(i=i, kind=pairs[0][0][i]["kind"], eager=pairs[0][0][i]["eager"], ad=S.mean(ad),
                         ad_med=S.median(ad), w0=S.mean(w0), w1=S.mean(w1), g0=S.mean(g0), g1=S.mean(g1)))
    tw0, tw1 = sum(r["w0"] for r in rows), sum(r["w1"] for r in rows)
    print(f"per step (means): rank0 waits {tw0 / 1e3:.3f} ms, rank1 waits {tw1 / 1e3:.3f} ms inside collectives; "
          f"GPU idle right before collectives: rank0 {sum(r['g0'] for r in rows) / 1e3:.3f} ms, "
          f"rank1 {sum(r['g1'] for r in rows) / 1e3:.3f} ms")
    hd = [(p1[0]["copy_ret"] - off - p0[0]["copy_ret"], p1[0]["launch"] - off - p0[0]["launch"],
           p0[0]["launch"] - p0[0]["copy_ret"], p1[0]["launch"] - p1[0]["copy_ret"]) for p0, p1 in pairs
          if p0[0].get("copy_ret") and p1[0].get("copy_ret")]
    if hd:
        print("host loop at the target start (%d steps, medians): blocking D2H returns r1-r0 %.0f us; eager embedding "
              "all-reduce launched r1-r0 %.0f us; host tail (return -> launch) r0 %.0f us, r1 %.0f us" % (
                  len(hd), S.median(x[0] for x in hd), S.median(x[1] for x in hd), S.median(x[2] for x in hd),
                  S.median(x[3] for x in hd)))
    else:
        print("host loop: no blocking per-step D2H before the eager all-reduce on both ranks (GLM53_DEC_HOSTLOOP?)")
    print("top positions by total wait (pos kind eager | arrive r1-r0 mean/med us | wait r0 / r1 us | idle-before "
          "r0 / r1 us):")
    for r in sorted(rows, key=lambda r: -(r["w0"] + r["w1"]))[: a.top]:
        print("  %3d %s %-5s | %8.1f %8.1f | %7.1f %7.1f | %7.1f %7.1f" % (
            r["i"], r["kind"], "eager" if r["eager"] else "graph", r["ad"], r["ad_med"], r["w0"], r["w1"], r["g0"],
            r["g1"]))


if __name__ == "__main__":
    main()
