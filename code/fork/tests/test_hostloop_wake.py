"""[dec-hostloop] GLM53_DEC_HOSTLOOP_WAKE on nodeC (production image):
  W1 the no-op host function glm53_hostloop launches finds the SAME driver thread that executes CUDA-graph host nodes
     (what NCCL captures), and that thread gets pinned to the first listed CPU;
  W2 tickers: one helper process with one SCHED_IDLE thread pinned per listed CPU; it dies when the process that
     started it dies (PDEATHSIG);
  W3 cold graph host-node latency (sleep 2 ms -> host node -> sleep 0.1 ms) before / with / after wake, interleaved
     blocks (report medians);
  W4 a spinner that dies switches wake off (logged), nothing raises.
Run: tests/hostloop_gpu.sh python3 tests/test_hostloop_wake.py 12,0
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "/w")
import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402

FAILS = []
cpus = sys.argv[1] if len(sys.argv) > 1 else "12,0"


def check(c, m):
    if not c:
        FAILS.append(m)
        print("FAIL:", m, flush=True)


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
                                         ctypes.cast(cb.glm53_cb, ctypes.c_void_p), ctypes.cast(slot, ctypes.c_void_p)) == 0
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


def block(n=16):
    return [(timed(gh) - timed(gp)) * 1e3 for _ in range(n)]


lat = {"before": [], "wake": [], "after": []}
lat["before"] += block()
H._WAKE.raw = cpus
H._wake_step()                                # start: spinners + probe host function
for _ in range(3):
    timed(gh)
t0 = time.time()
while not H._WAKE.cb_tid and time.time() - t0 < 5:
    time.sleep(0.01)
H._wake_step()                                # pin
W = H._WAKE
print("W1 probe host-function thread", W.cb_tid, "graph host-node thread", slot[2], "pinned", W.pinned_tid,
      "affinity", sorted(os.sched_getaffinity(W.cb_tid)) if W.cb_tid else None)
check(W.cb_tid and W.cb_tid == slot[2], "W1: probe thread != graph host-node thread")
check(W.pinned_tid == W.cb_tid and os.sched_getaffinity(W.cb_tid) == {W.cpus[0]}, "W1: callback thread not pinned")
def ticker_threads():
    out = {}
    for p in W.procs:
        for tid in os.listdir(f"/proc/{p.pid}/task"):
            tid = int(tid)
            if tid != p.pid:
                out[tid] = (os.sched_getscheduler(tid), sorted(os.sched_getaffinity(tid)))
    return out


t0 = time.time()                              # the tickers set their own policy/affinity once Python is up
while time.time() - t0 < 5 and not (len(ticker_threads()) == len(W.cpus)
                                    and all(v[0] == os.SCHED_IDLE for v in ticker_threads().values())):
    time.sleep(0.05)
pol = ticker_threads()
print("W2 ticker process(es)", [p.pid for p in W.procs], "threads", pol, "SCHED_IDLE =", os.SCHED_IDLE)
check(len(W.procs) == 1 and len(pol) == len(W.cpus) and all(v[0] == os.SCHED_IDLE for v in pol.values())
      and sorted(v[1][0] for v in pol.values()) == sorted(W.cpus), "W2: ticker policy/affinity")
for rnd in range(3):
    lat["wake"] += block()
    cpu_seen = slot[3]
print("W3 graph host node ran on cpu", cpu_seen)
check(cpu_seen == W.cpus[0], "W3: host node did not run on the pinned cpu")
# W4: kill a spinner -> next step switches wake off
W.procs[0].kill()
W.procs[0].wait()
H._wake_step()
check(W.failed and not W.procs, "W4: dead spinner not detected")
aff_after = os.sched_getaffinity(W.cb_tid)
print("W4 callback thread affinity after wake switched off:", sorted(aff_after))
check(len(aff_after) > 1 and not W.pinned_tid, "W4: callback thread left pinned after wake switched off")
lat["after"] += block()
lat["before"] += block()
for k, v in lat.items():
    v = sorted(v)
    print("W3 %-6s cold host-node latency us: med %6.1f p10 %6.1f p90 %6.1f max %6.1f (n=%d)" % (
        k, statistics.median(v), v[len(v) // 10], v[9 * len(v) // 10], v[-1], len(v)))
check(statistics.median(lat["wake"]) < statistics.median(lat["before"]), "W3: wake did not reduce the latency")
# W2b: PDEATHSIG - a parent that exits takes its spinner with it
child = subprocess.run([sys.executable, "-c", "import subprocess,sys,os,time;sys.path.insert(0,'/w');"
                        "import glm53_hostloop as H;p=subprocess.Popen([sys.executable,'-S','-c',H._TICK_CODE,"
                        "str(os.getpid()),'100','3,4']);print(p.pid,flush=True);time.sleep(0.5)"],
                       capture_output=True, text=True)
spid = int(child.stdout.split()[0])
time.sleep(0.5)
alive = os.path.exists(f"/proc/{spid}") and "Z" not in open(f"/proc/{spid}/stat").read().split()[2]
print("W2b spinner", spid, "alive after its parent exited:", alive)
check(not alive, "W2b: spinner survived its parent")
print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(1 if FAILS else 0)
