"""[dec-hostloop] GB10 probe 2: CUDA-graph host-node latency vs (a) the context's schedule flag and (b) the time
since the previous host node (is the driver's callback thread 'hot' when host nodes come in quick succession?).
Graph k: [sleep S] -> host -> [sleep S] -> host -> ... (N host nodes) vs the same without host nodes.
Run: tests/hostloop_gpu.sh python3 tests/probe_hostnode2.py <flag>   flag in auto|spin|yield|blocking
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import statistics
import sys

flag = sys.argv[1] if len(sys.argv) > 1 else "auto"


def find_lib(name):
    import torch  # noqa: F401  (maps libcudart)
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if name in p:
            return p
    raise SystemExit(name + " not mapped")


import torch  # noqa: E402
cudart = ctypes.CDLL(find_lib("libcudart.so"))
FLAGS = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}
rc = cudart.cudaSetDeviceFlags(ctypes.c_uint(FLAGS[flag]))
torch.cuda.init()
torch.zeros(1, device="cuda")
got = ctypes.c_uint(0)
cudart.cudaGetDeviceFlags(ctypes.byref(got))
cb = ctypes.CDLL(_cb_lib())
cb_ptr = ctypes.cast(cb.glm53_cb, ctypes.c_void_p)
cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
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


def build(spacing_us, n, host):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cuda._sleep(1000)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(n):
            torch.cuda._sleep(int(spacing_us * 1e-3 * cyc))
            if host:
                assert cudart.cudaLaunchHostFunc(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), cb_ptr,
                                                 ctypes.cast(slot, ctypes.c_void_p)) == 0
        torch.cuda._sleep(int(spacing_us * 1e-3 * cyc))
    return g


def timed(g):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


print("device flags requested %s (rc %d) -> cudaGetDeviceFlags 0x%x" % (flag, rc, got.value))
for spacing in (20, 100, 300, 1000, 3000):
    n = 8
    gh, gp = build(spacing, n, True), build(spacing, n, False)
    d = []
    for _ in range(25):
        d.append((timed(gh) - timed(gp)) * 1e3 / n)
    d.sort()
    print("spacing %5d us, %d host nodes: overhead per host node us med %7.1f p10 %7.1f p90 %7.1f" % (
        spacing, n, statistics.median(d), d[len(d) // 10], d[9 * len(d) // 10]))
