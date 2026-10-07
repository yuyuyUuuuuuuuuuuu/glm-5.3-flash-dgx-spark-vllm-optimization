"""[dec-hostloop] GB10 probe: does host code stall behind queued GPU work?

Measures, with ~20 ms of GPU work queued on the current stream:
  * pageable H2D  dev.copy_(cpu, non_blocking=True)      (flashinfer plan() indptr copies)
  * pinned   H2D  dev.copy_(pinned, non_blocking=True)
  * pageable D2H  gpu.cpu()                              (the production sync in _kv_lens_host)
  * pinned   D2H  pinned.copy_(gpu, non_blocking=True) + event   (the hostloop snapshot)
  * event.synchronize() on an event recorded BEFORE the queued work (must return at once)
"""
import time
import torch

dev = torch.device("cuda")
torch.cuda.init()
x = torch.randn(4096, 4096, device=dev)


def busy(ms: float):
    # queue roughly `ms` of GPU work (matmuls) without a host sync
    n = max(1, int(ms / 0.9))
    for _ in range(n):
        torch.mm(x, x)


busy(5); torch.cuda.synchronize()
t = time.perf_counter(); busy(20); torch.cuda.synchronize(); print("calibrate busy(20): %.1f ms" % ((time.perf_counter() - t) * 1e3))

cpu_pageable = torch.arange(7169, dtype=torch.int32)
cpu_pinned = torch.arange(7169, dtype=torch.int32).pin_memory()
dst = torch.empty(7169, dtype=torch.int32, device=dev)
small = torch.arange(8, dtype=torch.int32, device=dev)
snap = torch.empty(8, dtype=torch.int32).pin_memory()


def probe(name, fn, reps=5):
    res = []
    for _ in range(reps):
        torch.cuda.synchronize()
        ev_before = torch.cuda.Event(); ev_before.record()
        busy(20)
        t0 = time.perf_counter()
        fn(ev_before)
        t1 = time.perf_counter()
        q = torch.cuda.current_stream().query()
        torch.cuda.synchronize()
        res.append(((t1 - t0) * 1e3, q))
    ts = sorted(r[0] for r in res)
    print("%-44s host ms: min %.3f med %.3f max %.3f   stream idle after call: %s" % (
        name, ts[0], ts[len(ts) // 2], ts[-1], [r[1] for r in res]))


probe("pageable H2D copy_(non_blocking=True)", lambda e: dst.copy_(cpu_pageable, non_blocking=True))
probe("pinned   H2D copy_(non_blocking=True)", lambda e: dst.copy_(cpu_pinned, non_blocking=True))
probe("pageable D2H .cpu() (production sync)", lambda e: small.cpu())
probe("pinned   D2H copy_(non_blocking) only", lambda e: snap.copy_(small, non_blocking=True))
probe("event recorded before work .synchronize()", lambda e: e.synchronize())
