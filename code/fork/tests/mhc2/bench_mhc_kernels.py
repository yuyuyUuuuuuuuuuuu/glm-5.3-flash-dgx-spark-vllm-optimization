"""mhc2 B1: kernel-level breakdown of the production mHC prefill chain at the SP shard sizes (nodeC, one GB10).

Per T: one production mhc_fused_post_pre_tilelang call (model.py hc_fused_post_pre's exact arguments) profiled with
torch.profiler (CUDA kernel durations, median over reps) + the eager wall time per call (production prefill is eager),
plus the quickwins qw_post_mean (aux/final layers). Bytes moved are computed from the shapes to give each kernel's
achieved DRAM bandwidth.

T: 13824 (TP, full chunk), 6912 (SP shard of a full chunk), 3456 (SP sub-chunk of a 2-way pipelined shard),
4289 (TP tail chunk), 2145/2146 (SP shard of the padded tail), 1073 (sub-chunk of the padded tail).

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc2/bench_mhc_kernels.py
"""
from __future__ import annotations

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mhc_sp"))
from bench_mhc_halving import make, fused_call, post_mean_call  # noqa: E402

REPS = int(os.environ.get("BENCH_REPS", "20"))
TS = [int(t) for t in os.environ.get("BENCH_TS", "13824,6912,3456,4289,2146,1073").split(",")]
H, N = 4096, 4


def kernel_table(fn, reps):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    per = {}
    for ev in prof.events():
        if ev.device_type is not None and str(ev.device_type).endswith("CUDA") and ev.device_time > 0:
            per.setdefault(ev.name, []).append(ev.device_time)   # us
    return {k: (statistics.median(v), len(v) / reps) for k, v in per.items()}


def wall(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1000 / reps)
    return statistics.median(out)


def short(n):
    for k in ("mhc_post", "pre_big_fuse", "hc_prenorm", "post_mean", "fill", "copy", "Memset"):
        if k in n:
            return k
    return n[:40]


def main():
    print(f"gpu {torch.cuda.get_device_name(0)}")
    # bytes per token (bf16 = 2): post reads residual 4H + x H, writes 4H; prenorm GEMM reads 4H (+ writes 25 fp32);
    # big_fuse reads 4H (+ the 24+1 fp32 gemm outs), writes H; post_mean reads 4H + H, writes H
    bpt = {"mhc_post": 9 * H * 2, "hc_prenorm": 4 * H * 2 + 25 * 4, "pre_big_fuse": 5 * H * 2 + 25 * 4,
           "post_mean": 6 * H * 2}
    rows = []
    for T in TS:
        w = make(T)
        f = lambda: fused_call(w, w)
        m = lambda: post_mean_call(w, w)
        kt = kernel_table(f, REPS)
        km = kernel_table(m, REPS)
        wf, wm = wall(f, REPS), wall(m, REPS)
        print(f"\nT={T}: fused eager wall {wf:8.1f} us/call, post_mean eager wall {wm:7.1f} us/call")
        tot = 0.0
        for name, (us, cnt) in sorted(kt.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
            s = short(name)
            bw = (bpt[s] * T / (us * 1e-6) / 1e9) if s in bpt else float("nan")
            tot += us * cnt
            print(f"   {s:<14s} {us:8.1f} us x{cnt:.0f}  {bw:6.1f} GB/s   {name[:90]}")
        print(f"   kernel sum {tot:8.1f} us  (launch/host gap {wf - tot:6.1f} us)")
        for name, (us, cnt) in km.items():
            s = short(name)
            bw = (bpt[s] * T / (us * 1e-6) / 1e9) if s in bpt else float("nan")
            print(f"   [mean] {s:<14s} {us:8.1f} us x{cnt:.0f}  {bw:6.1f} GB/s")
        rows.append((T, wf, tot, wm))
        del w
        torch.cuda.empty_cache()
    print("\nper-13,824-token-chunk projections (90 fused + 6 post_mean per forward):")
    d = {T: (wf, tot, wm) for T, wf, tot, wm in rows}
    for lbl, T, mult in (("TP full chunk", 13824, 1), ("SP shard", 6912, 1), ("SP 2-way sub-chunks", 3456, 2)):
        if T in d:
            wf, tot, wm = d[T]
            print(f"   {lbl:<22s} T={T:6d} x{mult}: eager {mult * (90 * wf + 6 * wm) / 1000:7.1f} ms, "
                  f"kernels {mult * 90 * tot / 1000:7.1f} ms (+ means)")


if __name__ == "__main__":
    main()
