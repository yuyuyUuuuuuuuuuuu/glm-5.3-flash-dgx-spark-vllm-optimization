#!/usr/bin/env python3
"""FKDA3 probe: is FlashKDA's K2 (serial recurrence, 32 of 48 SMs at H=32) latency- or bandwidth-bound, i.e. would
overlapping K1 (prepare) with K2 pay? Two builds of the fkda2 sources: _fk3a_C launches only K1 (-DBLOCK_LEVEL_K2=-1),
_fk3b_C only K2 (-DBLOCK_LEVEL_K1=-1). Times K1 alone, K2 alone (on a workspace K1 filled), and K2(problem A) on
stream 1 concurrently with K1(problem B) on stream 2."""
import statistics, sys, torch
sys.path.insert(0, "/fkda/builds/k1only"); sys.path.insert(0, "/fkda/builds/k2only")
import _fk3a_C, _fk3b_C  # noqa
D, T, H = 128, int(sys.argv[1]) if len(sys.argv) > 1 else 13824, 32
g = torch.Generator(device="cpu").manual_seed(0)
def prob():
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
    p = dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, sc=0.5), b=rn(1, T, H),
             s0=(torch.randn(1, H, D, D, generator=g) * 0.3).cuda(), A=(torch.randn(H, generator=g) * .2).cuda(),
             dtb=(torch.rand(H, D, generator=g) * 8 - 10).cuda(), cu=torch.tensor([0, T], dtype=torch.int32, device="cuda"))
    p["ws"] = torch.empty(int(torch.ops._fk3a_C.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
    p["out"] = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    p["fs"] = torch.empty(1, H, D, D, device="cuda")
    return p
A, B = prob(), prob()
def run(ns, p):
    getattr(torch.ops, ns).fwd(p["q"], p["k"], p["v"], p["g"], p["b"], D ** -.5, p["out"], p["ws"], p["A"], p["dtb"], -5.0,
                               p["s0"], p["fs"], p["cu"], None, None)
k1 = lambda p: run("_fk3a_C", p)
k2 = lambda p: run("_fk3b_C", p)
k1(A); k1(B); torch.cuda.synchronize()
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
def med(fn, n=15):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
def conc():
    cur = torch.cuda.current_stream()
    s1.wait_stream(cur); s2.wait_stream(cur)
    with torch.cuda.stream(s1): k2(A)
    with torch.cuda.stream(s2): k1(B)
    cur.wait_stream(s1); cur.wait_stream(s2)
def seq():
    k2(A); k1(B)
r = dict(T=T, k1=med(lambda: k1(B)), k2=med(lambda: k2(A)), seq=med(seq), conc_k2first=med(conc))
print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    conc(); torch.cuda.synchronize()
ev = sorted([(e.time_range.start, e.time_range.end, e.name[:40]) for e in p.events() if e.device_type.name == "CUDA"])
t0 = ev[0][0]
for s, e, n in ev:
    print(f"  {n:40s} start {s - t0:8.0f} us end {e - t0:8.0f} us")
