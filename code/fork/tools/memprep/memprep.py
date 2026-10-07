#!/usr/bin/env python3
"""Pre-start memory preparation for a GB10 serving node (run while the GPU workload is STOPPED).

Why: on GB10 every cudaMalloc is backed by 64 KiB chunks taken one by one from the Linux buddy allocator
(open-gpu-kernel-modules 580: nv_alloc_system_pages, order = get_order(64 KiB), GFP_KERNEL, unmovable). The buddy
allocator serves the smallest free block first, so scattered small free blocks (left by page cache churn over uptime)
end up under the model weights. Measured on nodeC: chunks carved from >= 8 MiB free blocks read at 260-265 GB/s,
chunks from scattered small blocks at 231-249 GB/s. A fresh boot has only large blocks; this script restores that
state without a reboot:

  1. root path (if `sudo -n /usr/local/sbin/glm53-memprep prep` is allowed): sync, drop_caches=3, compact_memory=1;
  2. otherwise a no-root balloon: map anonymous MADV_HUGEPAGE memory 1 GiB at a time and touch it. Faulting THPs
     makes the kernel reclaim page cache and compact (defrag=madvise), then the balloon is freed so the buddies
     merge. It stops early when MemAvailable would drop under --floor-gib or swap-out starts, and marks itself
     oom_score_adj=1000 so it is the first thing the OOM killer takes.

Prints /proc/buddyinfo before/after and the GiB free in blocks of >= 2 MiB and >= 8 MiB.
Usage: memprep.py [--floor-gib 4] [--max-gib N] [--no-sudo] [--dry-run]"""
import argparse, ctypes, mmap, os, subprocess, sys, time

PAGE = 4096


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) * 1024
    return d


def vmstat(keys=("pswpout", "compact_stall", "compact_success", "compact_fail", "pgmigrate_success")):
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in keys:
            d[k] = int(v)
    return d


def buddy():
    """Normal-zone free block counts per order (DMA zone added in)."""
    tot = None
    for line in open("/proc/buddyinfo"):
        parts = line.split()
        counts = [int(x) for x in parts[4:]]
        tot = counts if tot is None else [a + b for a, b in zip(tot, counts)]
    return tot


def summarize(b):
    gib = lambda o0: sum(c * (PAGE << o) for o, c in enumerate(b) if o >= o0) / 2**30
    return f"free GiB total {gib(0):.1f} | >=2MiB {gib(9):.1f} | >=8MiB {gib(11):.1f} | >=32MiB {gib(13):.1f}"


def balloon(floor, max_gib, dry):
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    maps, t0 = [], time.time()
    sw0 = vmstat()["pswpout"]
    reason = "max reached"
    while len(maps) < max_gib:
        mi = meminfo()
        if mi["MemAvailable"] - 2**30 < floor:
            reason = f"MemAvailable {mi['MemAvailable']/2**30:.1f} GiB near floor"
            break
        if vmstat()["pswpout"] - sw0 > 25600:                     # > 100 MiB swapped out: stop, do not push others out
            reason = "swap-out started"
            break
        if dry:
            maps.append(None)
            continue
        m = mmap.mmap(-1, 2**30 + 2 * 2**20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        m.madvise(mmap.MADV_HUGEPAGE)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
        ctypes.memset(addr, 1, 2**30)                              # fault every page
        maps.append((m, addr))
    held = len(maps)
    thp = meminfo().get("AnonHugePages", 0) / 2**30
    for m in maps:
        if m is not None:
            ctypes.c_char.from_buffer(m[0])  # keep ref semantics simple
    del_ok = 0
    while maps:
        m = maps.pop()
        if m is not None:
            try:
                m[0].close()
            except BufferError:
                pass
            del_ok += 1
    return held, thp, time.time() - t0, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-gib", type=float, default=4.0)
    ap.add_argument("--max-gib", type=int, default=10**6)
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-free-gib", type=float, default=0.0,
                    help="first wait (up to --wait-timeout s) until MemAvailable >= this: the stopped container's memory is still being released")
    ap.add_argument("--wait-timeout", type=float, default=90.0)
    a = ap.parse_args()
    host = os.uname().nodename
    if a.wait_free_gib:
        t0 = time.time()
        while meminfo()["MemAvailable"] < a.wait_free_gib * 2**30 and time.time() - t0 < a.wait_timeout:
            time.sleep(1)
        print(f"[memprep {host}] waited {time.time()-t0:.0f}s for MemAvailable >= {a.wait_free_gib:.0f} GiB "
              f"(now {meminfo()['MemAvailable']/2**30:.1f})", flush=True)
    b0, m0, v0 = buddy(), meminfo(), vmstat()
    print(f"[memprep {host}] before: MemAvailable {m0['MemAvailable']/2**30:.1f} GiB MemFree {m0['MemFree']/2**30:.1f} "
          f"Cached {m0['Cached']/2**30:.1f} | {summarize(b0)}", flush=True)
    print(f"[memprep {host}] buddyinfo before: {' '.join(map(str, b0))}", flush=True)
    how = None
    if not a.no_sudo and os.path.exists("/usr/local/sbin/glm53-memprep"):
        r = subprocess.run(["sudo", "-n", "/usr/local/sbin/glm53-memprep", "prep"], capture_output=True, text=True)
        if r.returncode == 0:
            how = "root: drop_caches + compact_memory"
        else:
            print(f"[memprep {host}] sudo path not available ({(r.stderr or r.stdout).strip()[:120]}); using balloon")
    if how is None:
        held, thp, dt, reason = balloon(a.floor_gib * 2**30, a.max_gib, a.dry_run)
        how = f"balloon {held} GiB (THP {thp:.1f} GiB) in {dt:.0f}s, stopped: {reason}"
    time.sleep(1)
    b1, m1, v1 = buddy(), meminfo(), vmstat()
    dv = {k: v1[k] - v0.get(k, 0) for k in v1}
    print(f"[memprep {host}] method: {how}", flush=True)
    print(f"[memprep {host}] after:  MemAvailable {m1['MemAvailable']/2**30:.1f} GiB MemFree {m1['MemFree']/2**30:.1f} "
          f"Cached {m1['Cached']/2**30:.1f} | {summarize(b1)} | vmstat delta {dv}", flush=True)
    print(f"[memprep {host}] buddyinfo after:  {' '.join(map(str, b1))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
