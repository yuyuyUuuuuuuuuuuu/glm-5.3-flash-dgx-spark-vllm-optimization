"""Speed of production's prefill MoE apply with GLM53_PREFILL_FUSED_CAP (docs/PREFILL_CAP.md).

Per layer call of production's apply_exl3_fused_moe (through the TF K2 hook and the prefill-cap wrapper, as production
calls it), GLM-5.3 rank shapes (288 experts, hidden 4096, intermediate 1024 per rank, top-8), production prefill env
(EXL3_FAT_GROUPED=1, fused cap 256). T in BENCH_T (default 1791, 4608, 13824: production's chunks are 13824 + a
remainder; 1791 is the remainder of the profiled 15.6k prompt), routing kinds in BENCH_KINDS (tests/prefill_cap_common
routing()). For every (kind, T): each thin cap n in BENCH_N (0 = production's current path) is timed over 2 layers
(different weights and routing each) with CUDA events, ROUNDS rounds with the n order rotated every round; the median
over rounds is reported. Then one profiled call per n in BENCH_PROF_N gives the kernel breakdown
(thin exl3_moe_kernel / E3 fm_gather + fm_gateup + fm_down / everything else = routing prelude, zeroing, casts).
"""
from __future__ import annotations

import os
import statistics

import torch

import prefill_cap_common as C
import harness as H


def env_list(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v.strip()]


def kernel_breakdown(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    agg = {"thin": 0.0, "e3": 0.0, "other": 0.0}
    for e in prof.events():
        if not str(getattr(e, "device_type", "")).endswith("CUDA"):
            continue
        us = e.device_time if hasattr(e, "device_time") else e.cuda_time
        name = e.name
        if "exl3_moe_kernel" in name or "glm53_exl3_moe_fast_kernel" in name:
            agg["thin"] += us
        elif "fm_gather" in name or "fm_gateup" in name or "fm_down" in name:
            agg["e3"] += us
        else:
            agg["other"] += us
    return {k: v / 1000.0 for k, v in agg.items()}


def main():
    H.gpu_guard(8.0)
    prod, xl, tf, layers = C.build(2)
    import glm53_prefill_cap as PC

    rep = PC.install(prodmod=prod, n=32)
    assert rep["installed"], rep
    dev = torch.device("cuda", 0)
    Ts = env_list("BENCH_T", "1791,4608,13824")
    kinds = os.environ.get("BENCH_KINDS", "real,collapsed,zipf").split(",")
    ns = env_list("BENCH_N", "0,16,32,48,64,96,128")
    prof_ns = env_list("BENCH_PROF_N", "0,16,32,64")
    rounds = int(os.environ.get("BENCH_ROUNDS", "5"))
    reps = int(os.environ.get("BENCH_REPS", "2"))
    summary = []
    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, C.K, generator=g).to(torch.bfloat16).to(dev)
            calls = []
            for li, L in enumerate(layers):
                seed = 1000 * li + T + (0 if kind == "real" else 50000 if kind == "collapsed" else 90000)
                ids = C.routing(kind, T, seed, dev)
                calls.append((ids, C.weights_for(T, seed, dev), L))
            h = C.histogram(calls[0][0])
            print(f"[{kind} T={T}] layer0 routing: {C.fmt_hist(h)}; rows on experts > n: "
                  + " ".join(f"{n}:{v / (T * C.TOPK):.2f}" for n, v in h["rows_gt"].items()), flush=True)

            def run(n):
                PC.set_cap(n)
                for _ in range(reps):
                    for ids, w, L in calls:
                        C.apply(prod, x, ids, w, L)

            for n in ns:                                      # warm-up (temps alloc, scratch growth, autotune)
                run(n)
            torch.cuda.synchronize()
            sw0 = PC.STATS["swapped"]
            times = {n: [] for n in ns}
            for r in range(rounds):
                order = ns[r % len(ns):] + ns[:r % len(ns)]
                for n in order:
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    torch.cuda.synchronize()
                    s.record()
                    run(n)
                    e.record()
                    torch.cuda.synchronize()
                    times[n].append(s.elapsed_time(e) / (reps * len(calls)))
            swapped = PC.STATS["swapped"] - sw0
            exp_swapped = rounds * reps * len(calls) * sum(1 for n in ns if 0 < n < 256)
            assert swapped == exp_swapped, f"swapped {swapped} != {exp_swapped}"
            base = statistics.median(times[0]) if 0 in times else None
            line = []
            for n in ns:
                med = statistics.median(times[n])
                spread = (max(times[n]) - min(times[n])) / med
                sp = f" x{base / med:.3f}" if base else ""
                line.append(f"n={n}: {med:.2f} ms (spread {spread * 100:.1f}%){sp}")
                summary.append((kind, T, n, med, base))
            print(f"[{kind} T={T}] per layer call: " + " | ".join(line), flush=True)
            for n in prof_ns:
                PC.set_cap(n)
                ids, w, L = calls[0]
                kb = kernel_breakdown(lambda: C.apply(prod, x, ids, w, L))
                print(f"[{kind} T={T}] n={n} kernels (layer0, 1 call): thin {kb['thin']:.2f} ms, E3 {kb['e3']:.2f} ms,"
                      f" other {kb['other']:.2f} ms, total {sum(kb.values()):.2f} ms", flush=True)
            del x, calls
            torch.cuda.empty_cache()
    print("SUMMARY kind T n ms_per_layer_call speedup_vs_n0")
    for kind, T, n, med, base in summary:
        print(f"SUMMARY {kind} {T} {n} {med:.3f} {base / med if base else float('nan'):.3f}")
    d = prod.exl3_fat_diag()
    print(f"E3 grouped scratch bytes {d['grouped_scratch_bytes']} (MNBT {os.environ.get('MAX_NUM_BATCHED_TOKENS')} x top-k"
          f" 8 x (hidden + intermediate) x 2 B = {16384 * 8 * (C.K + C.N) * 2}); prefill-cap temps bytes "
          f"{PC.STATS['temps_bytes']} over {PC.STATS['temps_allocs']} sets")
    H.report_peak()


if __name__ == "__main__":
    H.run_main(main)
