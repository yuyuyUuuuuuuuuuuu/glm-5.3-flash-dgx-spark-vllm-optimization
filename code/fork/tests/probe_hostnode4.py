"""[dec-hostloop] GB10 probe 4: is the cold CUDA host-node latency (~0.5 ms, what NCCL pays at the first collective
of every CUDA graph on the network path) the CPU's deep idle-state exit (GB10 cpuidle LPI-3: exit latency 433 us)?
Graph: [sleep 2 ms] -> host node -> [sleep 0.1 ms], latency = total - same graph without the host node.
Variants (interleaved per round):
  base    : nothing else running
  allspin : a SCHED_IDLE busy-spinner process pinned on every CPU (no core can enter an idle state)
  pin1    : one SCHED_IDLE spinner on CPU K and the driver's host-callback thread pinned to CPU K
  irq     : SCHED_IDLE spinners only on the CPUs the nvidia MSI-X vectors are delivered to (effective affinity)
  irqpin  : as irq, and the host-callback thread pinned to the busiest nvidia vector's CPU
Run: tests/hostloop_gpu.sh python3 tests/probe_hostnode4.py <cpus of the nvidia MSI-X vectors, busiest first, e.g. 12,0>
"""
import ctypes
import sys as _sys
_sys.path.insert(0, "/w/tests")
from hostnode.build import ensure as _cb_lib  # noqa: E402  (builds tests/hostnode/cb.so if needed)
import sys
import multiprocessing as mp
import os
import statistics
import time

import torch


def spinner(cpu, stop):
    os.sched_setaffinity(0, {cpu})
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except Exception:  # noqa: BLE001
        os.nice(19)
    while not stop.is_set():
        for _ in range(100000):
            pass


def lib(name):
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if name in p:
            return p
    raise SystemExit(name)


def main():
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    cudart = ctypes.CDLL(lib("libcudart.so"))
    cb = ctypes.CDLL(_cb_lib())
    cb_ptr = ctypes.cast(cb.glm53_cb, ctypes.c_void_p)
    cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
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
                assert cudart.cudaLaunchHostFunc(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), cb_ptr,
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
    cb_tid, cb_cpu0 = int(slot[2]), int(slot[3])
    ncpu = os.cpu_count()
    K = ncpu - 1
    irq_cpus = []
    for line in open("/proc/interrupts"):
        if line.rstrip().endswith("nvidia"):
            irq = line.split(":")[0].strip()
            cnt = sum(int(x) for x in line.split()[1:1 + ncpu] if x.isdigit())
            try:
                cpu = int(open(f"/proc/irq/{irq}/effective_affinity_list").read().split(",")[0].split("-")[0])
            except OSError:
                continue
            if cnt > 0:
                irq_cpus.append((cnt, cpu))
    import sys
    if len(sys.argv) > 1:            # the container cannot read /proc/irq/*/effective_affinity_list: pass them in
        irq_cpus = [(len(sys.argv) - i, int(c)) for i, c in enumerate(sys.argv[1].split(","))]
    irq_cpus.sort(reverse=True)
    icpus = sorted({c for _, c in irq_cpus})
    top = irq_cpus[0][1] if irq_cpus else K
    print("nvidia vectors with interrupts (count, cpu):", irq_cpus)
    variants = ("base", "allspin", "pin1", "irq", "irqpin")
    res = {v: [] for v in variants}
    cpus_seen = {v: set() for v in variants}
    stop = mp.Event()
    for rnd in range(10):
        for var in variants:
            procs = []
            stop.clear()
            if var == "allspin":
                procs = [mp.Process(target=spinner, args=(c, stop), daemon=True) for c in range(ncpu)]
            elif var == "pin1":
                procs = [mp.Process(target=spinner, args=(K, stop), daemon=True)]
            elif var in ("irq", "irqpin"):
                procs = [mp.Process(target=spinner, args=(c, stop), daemon=True) for c in icpus]
            for p in procs:
                p.start()
            try:
                os.sched_setaffinity(cb_tid, {K} if var == "pin1" else ({top} if var == "irqpin" else set(range(ncpu))))
            except OSError as exc:
                print("setaffinity", exc)
            time.sleep(0.05)
            for _ in range(8):
                res[var].append((timed(gh) - timed(gp)) * 1e3)
                cpus_seen[var].add(int(slot[3]))
            stop.set()
            for p in procs:
                p.join()
    print("host-callback thread tid %d (first ran on cpu %d); %d cpus" % (cb_tid, cb_cpu0, ncpu))
    for var, d in res.items():
        d = sorted(d)
        print("%-8s host-node latency us: med %7.1f p10 %7.1f p90 %7.1f max %7.1f (n=%d) callback cpus %s" % (
            var, statistics.median(d), d[len(d) // 10], d[9 * len(d) // 10], d[-1], len(d), sorted(cpus_seen[var])))


if __name__ == "__main__":
    main()
