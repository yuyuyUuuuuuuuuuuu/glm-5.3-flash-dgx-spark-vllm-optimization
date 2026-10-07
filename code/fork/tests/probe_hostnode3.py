"""[dec-hostloop] GB10 probe 3: NCCL-in-CUDA-graph structure (hostStreamPlanCallback host node as a ROOT of the graph,
first collective kernel depends on it) and whether a no-op host node on a side branch shortly before the graph starts
("pre-warm") hides the cold wake-up of the driver's host-callback thread.
Stream: [drafter: sleep Q-W] -> fork(side: no-op host node) -> [sleep W] -> graph{ root host node H ; sleep P -> wait(H)
-> 'AR' 10 us -> sleep 1 ms }. Delay = total - same without H. Rounds interleaved.
Run: tests/hostloop_gpu.sh python3 tests/probe_hostnode3.py
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import sys
import statistics

import torch


def lib(name):
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if name in p:
            return p
    raise SystemExit(name)


torch.cuda.init()
torch.zeros(1, device="cuda")
cudart = ctypes.CDLL(lib("libcudart.so"))
cb = ctypes.CDLL(_cb_lib())
cb_ptr = ctypes.cast(cb.glm53_cb, ctypes.c_void_p)
cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
slot = (ctypes.c_long * 2)()
slot2 = (ctypes.c_long * 2)()


def host(stream, s=slot):
    assert cudart.cudaLaunchHostFunc(ctypes.c_void_p(stream.cuda_stream), cb_ptr, ctypes.cast(s, ctypes.c_void_p)) == 0


def calib():
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(20_000_000)
    e.record()
    e.synchronize()
    return 20_000_000 / s.elapsed_time(e)


cyc = calib()
us = lambda x: int(x * 1e-3 * cyc)  # noqa: E731


def build(P, with_host):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cuda._sleep(1000)
    torch.cuda.current_stream().wait_stream(s)
    side = torch.cuda.Stream()
    with torch.cuda.graph(g):
        main = torch.cuda.current_stream()
        side.wait_stream(main)                  # root: joins at the graph origin (nothing captured before it)
        with torch.cuda.stream(side):
            if with_host:
                host(side)
            torch.cuda._sleep(10)
        torch.cuda._sleep(us(P))                # first layer's compute before the first collective
        main.wait_stream(side)                  # the collective kernel depends on the host task
        torch.cuda._sleep(us(10))               # 'AR'
        torch.cuda._sleep(us(1000))
    return g


side_warm = torch.cuda.Stream()


def run(g, Q, W):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    main = torch.cuda.current_stream()
    torch.cuda._sleep(us(Q - (W or 0)))
    if W:
        side_warm.wait_stream(main)
        host(side_warm, slot2)                  # no-op host node, nobody waits for it
    torch.cuda._sleep(us(W or 0))
    g.replay()
    e.record()
    e.synchronize()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


Q = 3000
for P in (150, 400):
    gh, gp = build(P, True), build(P, False)
    for W in (0, 100, 300, 600, 1000, 2000):
        d = []
        for _ in range(30):
            d.append((run(gh, Q, W) - run(gp, Q, W)) * 1e3)
        d.sort()
        print("first collective after %4d us of compute, pre-warm %4s us before the graph: delay us med %6.1f "
              "p10 %6.1f p90 %6.1f" % (P, W or "none", statistics.median(d), d[len(d) // 10], d[9 * len(d) // 10]))
print("callbacks", slot[1], slot2[1])
