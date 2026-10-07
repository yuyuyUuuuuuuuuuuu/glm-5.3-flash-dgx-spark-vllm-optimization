"""glm53_vtrim_stats.records() vs a CPU reference loop (random batches: 1..8 requests, 1..8 drafts each, temperature 0
for some, DFlash2-like draft logits with 16 finite candidates per position, random acceptance)."""
import math
import random
import sys

import torch

sys.path.insert(0, "/w")
import glm53_vtrim_stats as S  # noqa: E402

dev = "cuda"
V, STEPS, MAXR = 4096, 7, 8
random.seed(3)
g = torch.Generator(device="cpu").manual_seed(5)
bad = 0
cases = 0
for case in range(200):
    R = random.randint(1, 8)
    ns = [random.randint(0, 7) for _ in range(R)]               # drafts per request (0 = no verify rows)
    dl = torch.full((MAXR, STEPS, V), float("-inf"))
    for r in range(MAXR):
        for s_ in range(STEPS):
            ids = torch.randperm(V, generator=g)[:16]
            dl[r, s_, ids] = torch.randn(16, generator=g) * 2
    states = random.sample(range(MAXR), R)
    temp = torch.tensor([random.choice([0.0, 0.7, 1.0, 1.0]) for _ in range(MAXR)])
    rows_tok, req_rows, lpos, cu = [], [], [], [0]
    for r in range(R):
        st = states[r]
        rows_tok.append(random.randint(0, V - 1))                  # row 0 input (last accepted token)
        for i in range(ns[r]):
            cand = (dl[st, i] > float("-inf")).nonzero().flatten()
            rows_tok.append(int(cand[random.randint(0, len(cand) - 1)]))
        for i in range(ns[r] + 1):
            req_rows.append(st)
            lpos.append(i)
        cu.append(cu[-1] + ns[r] + 1)
    nsamp = torch.tensor([random.randint(1, n + 1) for n in ns])
    got = S.records(dl.to(dev), torch.tensor(rows_tok, device=dev), torch.tensor(cu, device=dev, dtype=torch.int32),
                    torch.tensor(req_rows, device=dev, dtype=torch.int32),
                    torch.tensor(lpos, device=dev, dtype=torch.int32), temp.to(dev), nsamp.to(dev), R).cpu()
    ref = torch.zeros(R, S.REC)
    for r in range(R):
        st = states[r]
        if temp[st] <= 0 or ns[r] == 0:
            continue
        n = min(ns[r], S.NPOS)
        ref[r, 0] = n
        ref[r, 1] = min(int(nsamp[r]) - 1, n)
        for i in range(n):
            q = torch.softmax(dl[st, i].double() / float(temp[st]), -1)
            d = rows_tok[cu[r] + i + 1]
            ref[r, 2 + i] = float(q.max())
            ref[r, 2 + S.NPOS + i] = float(q[d])
    ok = torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    bad += not ok
    if not ok and bad < 4:
        print("mismatch", R, ns, (got - ref).abs().max())
    cases += 1
print(("ok   " if bad == 0 else "FAIL ") + f"records == CPU reference ({cases} batches)  bad {bad}")
# ring + write
import os
import numpy as np
S.ST.cap = 1000
S.ST.path = "/tmp/vtrim_test.npy"
S.ST.ring = None
S.ST.n = 0
for _ in range(30):
    S._append(torch.tensor([[3, 1, .9, .8, .7, 0, 0, 0, 0, .5, .4, .3, 0, 0, 0, 0]] * 50, device=dev, dtype=torch.float32))
S._write()
arr = np.load(S.ST.path)
okw = arr.shape == (1000, S.REC) and abs(arr[:, 2].mean() - 0.9) < 1e-6
print(("ok   " if okw else "FAIL ") + f"ring wraps at cap and the file holds {arr.shape}")
print("RESULT:", "ALL OK" if bad == 0 and okw else "FAIL")
sys.exit(0 if bad == 0 and okw else 1)
