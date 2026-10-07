"""Cost of glm53_vtrim_stats.records() per verify step at production's vocab (V = 154880), 1 / 4 / 8 requests x 8 rows."""
import sys, statistics as st
import torch
sys.path.insert(0, "/w")
import glm53_vtrim_stats as S
dev = "cuda"; V = 154880; STEPS = 7
dl = torch.full((8, STEPS, V), float("-inf"), device=dev)
for r in range(8):
    for s in range(STEPS):
        dl[r, s, torch.randperm(V, device=dev)[:16]] = torch.randn(16, device=dev)
temp = torch.ones(8, device=dev)
for R in (1, 4, 8):
    NL = R * 8
    toks = torch.randint(0, V, (NL,), device=dev)
    cu = torch.arange(0, NL + 1, 8, device=dev, dtype=torch.int32)
    rs = torch.arange(R, device=dev, dtype=torch.int32).repeat_interleave(8)
    lp = torch.arange(8, device=dev, dtype=torch.int32).repeat(R)
    ns = torch.full((R,), 4, device=dev, dtype=torch.int32)
    for _ in range(3): S.records(dl, toks, cu, rs, lp, temp, ns, R)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); e0.record(); S.records(dl, toks, cu, rs, lp, temp, ns, R); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    print(f"R={R} ({NL} rows): records() median {st.median(ts):.3f} ms GPU+launch", flush=True)
