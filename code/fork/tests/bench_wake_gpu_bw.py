"""[dec-hostloop review] Does GLM53_DEC_HOSTLOOP_WAKE (SCHED_IDLE tickers on the GPU-interrupt CPUs + pinned host-callback
thread) cost GPU throughput on GB10 (shared SoC power budget)? Paired ABBA: A = no wake, B = wake=<cpus> tick 100 us.
Workload: one CUDA graph of bandwidth-bound reductions over a bf16 buffer (~decode-step length), host thread waits in
Event.synchronize like the worker. Reports per-replay GPU ms (median, p10, p90) and effective GB/s.
Run: tests/hostloop_gpu.sh python3 tests/bench_wake_gpu_bw.py auto [rounds] [reps] [GiB] [control]
"""
import statistics
import sys
import time

sys.path.insert(0, "/w")
import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402

cpus = sys.argv[1] if len(sys.argv) > 1 else "auto"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
gib = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
control = len(sys.argv) > 5 and sys.argv[5] == "control"      # A/A: arm B does not start wake (noise floor)
n = int(gib * (1 << 30) / 2)
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
out = torch.empty(64, device="cuda", dtype=torch.float32)
chunks = x.view(8, -1)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        for i in range(8):
            torch.sum(chunks[i], dim=0, dtype=torch.float32, out=out[i])
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(3):
        for i in range(8):
            torch.sum(chunks[i], dim=0, dtype=torch.float32, out=out[i])
bytes_per = 3 * x.numel() * 2


def one():
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


for _ in range(5):
    one()
res = {"A": [], "B": []}
for rnd in range(rounds):
    for arm in (("A", "B") if rnd % 2 == 0 else ("B", "A")):
        if arm == "B" and not control:
            H._WAKE.__init__()
            H._WAKE.raw = cpus
            H._WAKE.tick_us = 100
            H._WAKE.pin = True
            H._wake_step()                  # start helper + probe
            t0 = time.time()
            while not H._WAKE.cb_tid and time.time() - t0 < 5:
                one()
            H._wake_step()                  # pin
            assert not H._WAKE.failed and H._WAKE.procs, "wake failed"
        time.sleep(0.2)
        for _ in range(reps):
            res[arm].append(one())
        if arm == "B" and not control:
            H._wake_stop()
            import os
            os.sched_setaffinity(H._WAKE.cb_tid, set(range(os.cpu_count())))
            time.sleep(0.2)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


print(f"GPU graph: 24 bf16 reductions, {bytes_per / 1e9:.2f} GB read per replay; wake cpus {cpus}; "
      f"{rounds} rounds x {reps} replays per arm (ABBA)")
for arm, name in (("A", "no wake"), ("B", "no wake (control)" if control else "WAKE tick 100 us")):
    v = res[arm]
    m = statistics.median(v)
    print("  %-18s ms med %.3f p10 %.3f p90 %.3f  -> %.1f GB/s (n=%d)" % (name, m, q(v, .1), q(v, .9),
                                                                          bytes_per / m / 1e6, len(v)))
d = statistics.median(res["B"]) - statistics.median(res["A"])
print("  B - A median: %+.3f ms (%+.2f %%)" % (d, 100 * d / statistics.median(res["A"])))
