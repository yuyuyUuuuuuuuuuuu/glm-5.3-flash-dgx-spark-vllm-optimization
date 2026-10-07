#!/usr/bin/env python3
"""FKDA3 probe: K2 time vs local head count H at T=13,824 (H<=24 takes the V-split path: 2 blocks per head, VD=64)."""
import statistics, sys, torch
sys.path.insert(0, "/fkda/builds/k1only"); sys.path.insert(0, "/fkda/builds/k2only")
import _fk3a_C, _fk3b_C  # noqa
D, T = 128, int(sys.argv[1]) if len(sys.argv) > 1 else 13824
for H in [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8,16,24,25,32,48").split(",")]:
    g = torch.Generator(device="cpu").manual_seed(0)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
    p = dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, sc=0.5), b=rn(1, T, H),
             s0=(torch.randn(1, H, D, D, generator=g) * 0.3).cuda(), A=(torch.randn(H, generator=g) * .2).cuda(),
             dtb=(torch.rand(H, D, generator=g) * 8 - 10).cuda(), cu=torch.tensor([0, T], dtype=torch.int32, device="cuda"))
    p["ws"] = torch.empty(int(torch.ops._fk3a_C.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
    p["out"] = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda"); p["fs"] = torch.empty(1, H, D, D, device="cuda")
    run = lambda ns: getattr(torch.ops, ns).fwd(p["q"], p["k"], p["v"], p["g"], p["b"], D ** -.5, p["out"], p["ws"], p["A"],
                                                 p["dtb"], -5.0, p["s0"], p["fs"], p["cu"], None, None)
    run("_fk3a_C"); torch.cuda.synchronize()
    ts = []
    for _ in range(12):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); run("_fk3b_C"); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        run("_fk3b_C"); torch.cuda.synchronize()
    nm = [e.name for e in pr.events() if "recurrence" in e.name][0]
    m = statistics.median(ts[2:])
    print(f"H={H:3d} K2 {m:.3f} ms  {m*1e3/(T//16):.2f} us/tile  per-head-equiv {m/H*32:.3f} ms@32  "
          f"path={'vsplit' if 'Li64EE' in nm else 'full'}", flush=True)
    del p
