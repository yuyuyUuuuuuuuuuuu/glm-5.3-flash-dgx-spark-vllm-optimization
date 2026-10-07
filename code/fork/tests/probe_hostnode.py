"""[dec-hostloop] GB10 probe: latency of a CUDA-graph HOST NODE (what NCCL captures before a network collective,
hostStreamPlanCallback) as a function of what the launching host thread is doing while the graph runs.

Graph: sleep A -> host node (C function, no GIL) -> sleep B. Overhead = (end - start) - (graph without the host node),
events recorded outside the graph. Host thread states while the graph runs (production decode analogues):
  event    : end_event.synchronize()                         (async-output thread / GLM53_DEC_HOSTLOOP snapshot wait)
  pageable : small_gpu_tensor.cpu() queued after the graph   (production _kv_lens_host: cam.seq_lens[:n].cpu())
  stream   : current_stream().synchronize()
  pybusy   : 8 ms Python busy loop (GIL held, no CUDA call), then event sync   (host doing plan()/builders)
  sleep    : time.sleep(8 ms), then event sync
Run: tests/hostloop_gpu.sh python3 tests/probe_hostnode.py
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import statistics
import sys
import time

import torch

dev = torch.device("cuda")
torch.cuda.init()


def find_cudart():
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if "libcudart.so" in p:
            return p
    raise SystemExit("libcudart not mapped")


cudart = ctypes.CDLL(find_cudart())
cb_lib = ctypes.CDLL(_cb_lib())
cb_ptr = ctypes.cast(cb_lib.glm53_cb, ctypes.c_void_p)
cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
cudart.cudaLaunchHostFunc.restype = ctypes.c_int
slot = (ctypes.c_long * 2)()


def calib():
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(20_000_000)
    e.record()
    e.synchronize()
    return 20_000_000 / s.elapsed_time(e)


cyc = calib()
A_MS, B_MS = 2.0, 2.0


def build(with_host: bool):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cuda._sleep(1000)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        torch.cuda._sleep(int(A_MS * cyc))
        if with_host:
            rc = cudart.cudaLaunchHostFunc(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), cb_ptr,
                                           ctypes.cast(slot, ctypes.c_void_p))
            assert rc == 0, rc
        torch.cuda._sleep(int(B_MS * cyc))
    return g


g_host, g_plain = build(True), build(False)
small = torch.arange(8, dtype=torch.int32, device=dev)


def run(g, state):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    if state == "event":
        e.synchronize()
    elif state == "pageable":
        small.cpu()
    elif state == "stream":
        torch.cuda.current_stream().synchronize()
    elif state == "pybusy":
        t_end = time.perf_counter() + 0.008
        while time.perf_counter() < t_end:
            pass
        e.synchronize()
    elif state == "sleep":
        time.sleep(0.008)
        e.synchronize()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


states = ("event", "pageable", "stream", "pybusy", "sleep")
res = {(h, st): [] for h in ("host", "plain") for st in states}
for rnd in range(40):                                   # interleaved
    for st in states:
        res[("plain", st)].append(run(g_plain, st))
        res[("host", st)].append(run(g_host, st))
print("torch", torch.__version__, "graph = sleep %.1f ms -> host node -> sleep %.1f ms; callbacks run: %d" % (
    A_MS, B_MS, slot[1]))
for st in states:
    h, p = res[("host", st)], res[("plain", st)]
    d = [a - b for a, b in zip(h, p)]
    print("%-9s host-node overhead us: med %7.1f p10 %7.1f p90 %7.1f max %7.1f | plain graph ms med %.3f" % (
        st, 1e3 * statistics.median(d), 1e3 * sorted(d)[len(d) // 10], 1e3 * sorted(d)[9 * len(d) // 10],
        1e3 * max(d), statistics.median(p)))
