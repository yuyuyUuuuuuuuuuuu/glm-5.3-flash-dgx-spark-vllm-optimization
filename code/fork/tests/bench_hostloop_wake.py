"""[dec-hostloop] paired A/B of GLM53_DEC_HOSTLOOP_WAKE on the cold CUDA-graph host-node latency (what the first NCCL
collective of every graph waits for): A = production (no spinner, callback thread unpinned), B = wake with the given
CPU list, interleaved blocks (ABBA), graph = sleep 2 ms -> host node -> sleep 0.1 ms.
Run: tests/hostloop_gpu.sh python3 tests/bench_hostloop_wake.py 12,0 [rounds] [ticks_us, e.g. 100,0]
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import os
import statistics
import sys
import time

sys.path.insert(0, "/w")
import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402

cpus = sys.argv[1] if len(sys.argv) > 1 else "12,0"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ticks = [x for x in (sys.argv[3] if len(sys.argv) > 3 else "100,0").split(",")]   # B arms: tick us (0 = spin),
#                                                                         suffix n = callback thread not pinned
torch.cuda.init()
torch.zeros(1, device="cuda")
rt = H._cudart()
rt.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
cb = ctypes.CDLL(_cb_lib())
slot = (ctypes.c_long * 4)()
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
torch.cuda._sleep(20_000_000)
e.record()
e.synchronize()
cyc = 20_000_000 / s.elapsed_time(e)


def build(host):
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        torch.cuda._sleep(1000)
    torch.cuda.current_stream().wait_stream(st)
    with torch.cuda.graph(g):
        torch.cuda._sleep(int(2.0 * cyc))
        if host:
            assert rt.cudaLaunchHostFunc(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                                         ctypes.cast(cb.glm53_cb, ctypes.c_void_p),
                                         ctypes.cast(slot, ctypes.c_void_p)) == 0
        torch.cuda._sleep(int(0.1 * cyc))
    return g


gh, gp = build(True), build(False)


def timed(g):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


timed(gh)
cb_tid = int(slot[2])
allc = set(range(os.cpu_count()))
arms = ["A"] + [f"B{t}" for t in ticks]
res = {a: [] for a in arms}
order = []
for rnd in range(rounds):
    order += arms if rnd % 2 == 0 else arms[::-1]
for arm in order:
    if arm != "A":
        H._WAKE.__init__()
        H._WAKE.raw = cpus
        H._WAKE.tick_us = int(arm[1:].rstrip("n"))
        H._WAKE.pin = not arm.endswith("n")
        H._wake_step()
        t0 = time.time()
        while not H._WAKE.cb_tid and time.time() - t0 < 5:
            timed(gp)
        H._wake_step()
        assert H._WAKE.pinned_tid == (cb_tid if H._WAKE.pin else 0), (H._WAKE.pinned_tid, cb_tid)
    time.sleep(0.05)
    for _ in range(16):
        res[arm].append((timed(gh) - timed(gp)) * 1e3)
    if arm != "A":
        H._wake_stop()
        os.sched_setaffinity(cb_tid, allc)
        time.sleep(0.05)
print("cold graph host-node latency, %d rounds x 16 per arm, wake cpus %s:" % (rounds, cpus))
for arm in arms:
    name = "production" if arm == "A" else ("WAKE spin" if arm[1:].rstrip("n") == "0" else
                                            f"WAKE tick {arm[1:].rstrip('n')} us") + (" no pin" if arm.endswith("n") else "")
    v = sorted(res[arm])
    print("  %-22s med %6.1f us  p10 %6.1f  p90 %6.1f  p99 %6.1f  mean %6.1f  (n=%d)" % (
        name, statistics.median(v), v[len(v) // 10], v[9 * len(v) // 10], v[min(len(v) - 1, 99 * len(v) // 100)],
        statistics.mean(v), len(v)))
