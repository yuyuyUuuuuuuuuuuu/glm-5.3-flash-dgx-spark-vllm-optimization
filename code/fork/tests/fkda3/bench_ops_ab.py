#!/usr/bin/env python3
"""FKDA3: interleaved op-level speed A/B of FlashKDA builds loaded side by side (distinct op namespaces), production
shapes (H=32/rank, D=128, one sequence, fp32 initial state, row-strided beta). Rounds alternate A,B,A,B...; reports
the median per build per shape and the per-13,824-token-chunk projection over 34 KDA layers."""
import argparse, statistics, sys, torch
ap = argparse.ArgumentParser()
ap.add_argument("--builds", default="fkda2=/fkda/builds/fix_n:_fk3f_C,fkda3=/fkda/builds/f3_all:_fk3n_C")
ap.add_argument("--lengths", default="13824,4608,1791")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
B = {}
for it in a.builds.split(","):
    n, r = it.split("="); d, m = r.split(":"); sys.path.insert(0, d); __import__(m); B[n] = getattr(torch.ops, m)
D, H = 128, 32
for T in [int(x) for x in a.lengths.split(",")]:
    for init in ("zeros", "state"):
        g = torch.Generator(device="cpu").manual_seed(3)
        rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
        q, k, v, g1 = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D, sc=0.5)
        beta = rn(1, T, 3 * H)[:, :, H:2 * H]
        s0 = torch.zeros(1, H, D, D, device="cuda") if init == "zeros" else (torch.randn(1, H, D, D, generator=g) * .3).cuda()
        A = (torch.randn(H, generator=g) * .2).cuda(); dtb = (torch.rand(H, D, generator=g) * 8 - 10).cuda()
        cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        ws = torch.empty(int(next(iter(B.values())).get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda"); fs = torch.empty(1, H, D, D, device="cuda")
        ts = {n: [] for n in B}
        for r in range(a.rounds):
            for n, op in (B.items() if r % 2 == 0 else reversed(list(B.items()))):
                fn = lambda: op.fwd(q, k, v, g1, beta, D ** -.5, out, ws, A, dtb, -5.0, s0, fs, cu, None, None)
                for _ in range(3): fn()
                torch.cuda.synchronize()
                for _ in range(a.iters):
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record(); fn(); e.record(); torch.cuda.synchronize(); ts[n].append(s.elapsed_time(e))
        med = {n: statistics.median(x) for n, x in ts.items()}
        base = next(iter(med.values()))
        print(f"T={T:5d} {init:5s}: " + " | ".join(f"{n} {m:.3f} ms ({(m - base) * 34:+.1f} ms/chunk)" for n, m in med.items()), flush=True)
