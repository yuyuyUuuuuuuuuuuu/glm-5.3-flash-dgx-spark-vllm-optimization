"""Pair two ranks' GLM53_DEC_TRACE files step by step: GPU-exact arrival skew and waits at the eager collectives,
and each rank's host windows (docs/DEC_HOSTGAP.md §3).

Usage: python3 tests/align_dectrace.py <rank0 trace> <rank1 trace> [--csv FILE] [--steps N]
                                       [--offset-us X | --wallclock]

Steps are matched by index: both ranks run the same schedule, so execute_model call k is the same step on both
(an unmatched count > a few percent means the two files are not the same serving window).

1. Rank wait at the eager collectives (EXACT, no clock needed; needs the `gar`/`gag` GPU records, i.e.
   GLM53_DEC_TRACE_GPU on). A collective finishes on both ranks together, and its GPU duration on a rank = its wait
   for the other rank + the transfer. Per step and position k (the k-th eager all-reduce / all-gather of the step):
       arrival skew r1-r0 = dur_r0 - dur_r1   (> 0: rank 1's GPU reached the collective later, rank 0 waited)
       wait_r            = dur_r - min(dur_r0, dur_r1)
   Production decode: AR position 0 = the target embedding all-reduce (the step's start), AG position 0 = the
   target logits all-gather. The ~102 collectives inside the CUDA graphs are not visible to Python: use a profiler
   pair and tests/skew_from_traces.py for those.

2. Host windows per rank (each on its own clock, valid): sched->prep (input build), prep->fg_b (host work before the
   target's FULL-graph launch; with GLM53_DEC_HOSTLOOP it overlaps the drafter, i.e. it is no longer GPU idle),
   fg_b->fg_e (graph launch call), prep->pw_b / pw_b->pw_e (the same for a PIECEWISE step: breakable-cudagraph replay),
   tgt_b->tgt_e (eager forward), sam_b->sam_e, dr_b->dr_e, and each
   eager collective's HOST call (ar_b->ar_e: the NCCL enqueue, NOT the wait).

3. Cross-rank HOST skew (sched, prep, fg_b, ...) needs a common clock, which no host event provides: an eager
   collective's host call returns after the NCCL enqueue, not when the other rank arrives (p6h: the eager all-reduce's
   host call returns a median 127 us before its kernel ends, the eager all-gather's 58 ms before its kernel starts;
   tests/review_eager_coll_host_vs_gpu.py). The first version of this script took its offset from those returns; on
   tests/test_dectrace.py A1's physically modelled pair it reported the host skew with the wrong sign and no
   all-reduce wait (docs/logs/hostgap/review/old_aligner_on_physical_pair.log). So cross-rank host skew is printed only
   with --offset-us X (an offset r1-r0 obtained elsewhere) or --wallclock (from the files' `#! clock`
   perf_counter/time_ns pairs: as good as the nodes' NTP sync, typically tens of us to ms: a coarse reference, never
   the verdict).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys


def load(path: str):
    steps: dict[int, dict[str, list[int]]] = {}
    gpu: dict[int, dict[str, list[int]]] = {}
    hdr, rank, stopped, clocks = {}, None, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#!"):
                if line.startswith("#!glm53-dectrace "):
                    try:
                        hdr = json.loads(line[len("#!glm53-dectrace "):])
                    except json.JSONDecodeError:
                        hdr = {}
                elif line.startswith("#! rank"):
                    rank = int(line.split()[2])
                elif line.startswith("#! clock"):
                    try:
                        _, _, pc, wall = line.split()
                        clocks.append((int(pc), int(wall)))
                    except ValueError:
                        pass
                elif line.startswith("#! stopped"):
                    stopped = line
                continue
            parts = line.split(" ")
            try:
                if len(parts) == 3:
                    steps.setdefault(int(parts[0]), {}).setdefault(parts[2], []).append(int(parts[1]))
                elif len(parts) == 4:          # GPU record: n t_host_begin gar|gag dur_ns
                    gpu.setdefault(int(parts[0]), {}).setdefault(parts[2], []).append(int(parts[3]))
            except ValueError:
                continue
    if "boot_pc_ns" in hdr and "boot_epoch_ns" in hdr:
        clocks.insert(0, (int(hdr["boot_pc_ns"]), int(hdr["boot_epoch_ns"])))
    return {"steps": steps, "gpu": gpu, "hdr": hdr, "rank": rank, "stopped": stopped, "clocks": clocks,
            "path": path}


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def wall_offset(A, B):
    """r1-r0 offset of the two perf_counter clocks from their (perf_counter, time_ns) pairs (NTP-limited)."""
    if not A["clocks"] or not B["clocks"]:
        return None
    a = med([w - p for p, w in A["clocks"]])      # rank0: wall = pc + a
    b = med([w - p for p, w in B["clocks"]])      # rank1: wall = pc + b
    return a - b                                  # pc1 - pc0 at the same wall instant


def gpu_positions(g0: dict, g1: dict):
    """(kind, k, d0, d1) for every eager collective position present on both ranks in one step."""
    out = []
    for ev, kind in (("gar", "AR"), ("gag", "AG")):
        a, b = g0.get(ev, []), g1.get(ev, [])
        for k in range(min(len(a), len(b))):
            out.append((kind, k, a[k], b[k]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("r0")
    ap.add_argument("r1")
    ap.add_argument("--csv", help="per-step CSV (matched steps)")
    ap.add_argument("--steps", type=int, default=0, help="only the first N matched steps")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--offset-us", type=float, help="clock offset r1-r0 (us) obtained elsewhere -> cross-rank host skew")
    g.add_argument("--wallclock", action="store_true", help="cross-rank host skew from the NTP wall-clock pairs")
    a = ap.parse_args()
    A, B = load(a.r0), load(a.r1)
    s0, s1 = A["steps"], B["steps"]
    k0 = {n for n, ev in s0.items() if "sched" in ev}
    k1 = {n for n, ev in s1.items() if "sched" in ev}
    common = sorted(k0 & k1)
    if len(common) < 2:
        print(f"only {len(common)} common steps (rank{A['rank']} {len(k0)}, rank{B['rank']} {len(k1)}); "
              f"not the same window?", file=sys.stderr)
        return 2
    if a.steps:
        common = common[:a.steps]
    nomatch = len(k0 ^ k1)
    print(f"rank{A['rank']} {A['hdr'].get('host')} pid {A['hdr'].get('pid')} ({len(k0)} steps), "
          f"rank{B['rank']} {B['hdr'].get('host')} pid {B['hdr'].get('pid')} ({len(k1)} steps); "
          f"{len(common)} matched, {nomatch} unmatched")
    if A["rank"] is not None and A["rank"] == B["rank"]:
        print(f"WARNING: both files say rank {A['rank']}: the same rank twice?")
    if nomatch:
        print(f"NOTE: {nomatch} steps exist on one rank only (different execute_model counts)")

    rows = [{"step": n} for n in common]

    # ---- 1. GPU-exact rank wait at the eager collectives
    pos: dict[tuple, dict[str, list[float]]] = {}
    ngpu = 0
    for row in rows:
        n = row["step"]
        for kind, k, d0, d1 in gpu_positions(A["gpu"].get(n, {}), B["gpu"].get(n, {})):
            ngpu += 1
            p = pos.setdefault((kind, k), {"skew": [], "w0": [], "w1": [], "d0": [], "d1": []})
            m = min(d0, d1)
            p["skew"].append((d0 - d1) / 1e3)
            p["w0"].append((d0 - m) / 1e3)
            p["w1"].append((d1 - m) / 1e3)
            p["d0"].append(d0 / 1e3)
            p["d1"].append(d1 / 1e3)
            row[f"{kind}{k}_dur_r0_us"], row[f"{kind}{k}_dur_r1_us"] = d0 / 1e3, d1 / 1e3
            row[f"{kind}{k}_skew_us"] = (d0 - d1) / 1e3
    print("\n1. rank wait at the eager collectives (GPU durations, exact; skew > 0 = rank 1 later, rank 0 waits):")
    if not pos:
        print("   no GPU records on both ranks (GLM53_DEC_TRACE_GPU off, TP=1, or no eager collective in the window)")
    else:
        print("   %-5s %5s %9s %9s %9s %9s %9s %9s %9s" % ("pos", "n", "skew med", "skew p10", "skew p90",
                                                        "wait r0", "wait r1", "dur r0", "dur r1"))
        for (kind, k), p in sorted(pos.items()):
            print("   %-5s %5d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f   (us; waits = means, durs = medians)" % (
                f"{kind}{k}", len(p["skew"]), med(p["skew"]), pct(p["skew"], 0.1), pct(p["skew"], 0.9),
                mean(p["w0"]), mean(p["w1"]), med(p["d0"]), med(p["d1"])))
        tw0 = sum(mean(p["w0"]) for p in pos.values())
        tw1 = sum(mean(p["w1"]) for p in pos.values())
        print(f"   per step (sum of the position means): rank0 waits {tw0:.1f} us, rank1 waits {tw1:.1f} us "
              f"at the eager collectives")

    # ---- 2. host windows per rank
    WINDOWS = (("sched", "prep", "input build"), ("prep", "fg_b", "prep->FULL graph launch"),
               ("fg_b", "fg_e", "FULL graph launch call"), ("prep", "pw_b", "prep->PIECEWISE run"),
               ("pw_b", "pw_e", "PIECEWISE run (target)"), ("tgt_b", "tgt_e", "target fwd (eager)"),
               ("sam_b", "sam_e", "sample_tokens"), ("dr_b", "dr_e", "drafter propose"),
               ("sched", "exec_e", "execute_model"), ("ar_b", "ar_e", "eager AR host call (enqueue)"),
               ("ag_b", "ag_e", "eager AG host call (enqueue)"))
    cols = {nm: [] for _, _, nm in WINDOWS}
    for row in rows:
        n = row["step"]
        e0, e1 = s0[n], s1[n]
        for b0, b1, nm in WINDOWS:
            u0, t0 = (e0.get(b0) or [None])[0], (e0.get(b1) or [None])[0]
            u1, t1 = (e1.get(b0) or [None])[0], (e1.get(b1) or [None])[0]
            if None in (u0, t0, u1, t1) or t0 < u0 or t1 < u1:
                continue
            cols[nm].append(((t0 - u0) / 1e3, (t1 - u1) / 1e3))
            row[nm + "_r0_us"], row[nm + "_r1_us"] = (t0 - u0) / 1e3, (t1 - u1) / 1e3
    print("\n2. host windows per rank (each on its own clock; medians, us):")
    print("   %-30s %6s %9s %9s %9s" % ("window", "n", "rank0", "rank1", "r1-r0"))
    for _, _, nm in WINDOWS:
        v = cols[nm]
        if v:
            m0, m1 = med(x[0] for x in v), med(x[1] for x in v)
            print("   %-30s %6d %9.1f %9.1f %+9.1f" % (nm, len(v), m0, m1, m1 - m0))

    # ---- 3. cross-rank host skew: only with a clock from outside
    off, off_src = None, None
    if a.offset_us is not None:
        off, off_src = a.offset_us * 1e3, "--offset-us (given)"
    elif a.wallclock:
        off = wall_offset(A, B)
        off_src = "wall clock (#! clock pairs; NTP accuracy, coarse)" if off is not None else None
    sk = {}
    if off is None:
        print("\n3. cross-rank host skew: not computed. No host event is a shared clock (an eager collective's host "
              "call returns after the NCCL enqueue, not when the other rank arrives); pass --offset-us or "
              "--wallclock. The rank wait itself is section 1.")
    else:
        print(f"\n3. cross-rank host skew r1-r0 with offset {off / 1e6:+.3f} ms from {off_src} (medians, us):")
        for k in ("sched", "prep", "fg_b", "pw_b", "tgt_b", "sam_b", "dr_b", "dr_e", "ar_b"):
            v = []
            for row in rows:
                n = row["step"]
                x0, x1 = (s0[n].get(k) or [None])[0], (s1[n].get(k) or [None])[0]
                if None not in (x0, x1):
                    v.append(((x1 - off) - x0) / 1e3)
                    row["hostskew_" + k + "_us"] = v[-1]
            if v:
                sk[k] = med(v)
                print("   %-6s n %5d  med %9.1f  p10 %9.1f  p90 %9.1f" % (k, len(v), med(v), pct(v, 0.1), pct(v, 0.9)))

    def clean(x):
        return None if isinstance(x, float) and math.isnan(x) else x

    summary = {
        "matched": len(rows), "unmatched": nomatch, "gpu_pairs": ngpu,
        "gpu_positions": {f"{kind}{k}": {"n": len(p["skew"]), "skew_med_us": clean(med(p["skew"])),
                                          "wait_r0_mean_us": clean(mean(p["w0"])),
                                          "wait_r1_mean_us": clean(mean(p["w1"]))}
                          for (kind, k), p in sorted(pos.items())},
        "windows_us": {nm: [clean(med(x[0] for x in cols[nm])), clean(med(x[1] for x in cols[nm]))]
                       for _, _, nm in WINDOWS if cols[nm]},
        "host_offset_ms": None if off is None else off / 1e6, "host_offset_source": off_src,
        "host_skew_us": sk,
    }
    print("SUMMARY " + json.dumps(summary))
    if A["stopped"] or B["stopped"]:
        print(f"stopped: r{A['rank']} {A['stopped']}; r{B['rank']} {B['stopped']}")
    if a.csv:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"per-step rows -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
