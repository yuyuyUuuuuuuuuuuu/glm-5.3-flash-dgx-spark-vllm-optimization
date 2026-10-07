#!/usr/bin/env python3
"""FKDA3: error at tile position 15 vs per-token decay strength s (log2-decay per token = -7.21*s, in-tile total
-115*s). `decay_sweep.py <build dir|triton>`; same inputs per s for every build (fixed seeds)."""
import math, os, sys, torch
sys.path.insert(0, "/w/tests/fkda3")
from adv_accuracy import Gen, ref64, rel, run_fk, run_tr
b = sys.argv[1]
if b != "triton":
    sys.path.insert(0, b); mod = [f[:-len(".abi3.so")] for f in os.listdir(b) if f.endswith(".abi3.so")][0]; __import__(mod)
H, D, T = 8, 128, 512
for s in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
    c = Gen(808, None).case([T], H=H, init="rand", regime="mixed")
    c["A_log"] = torch.zeros(H, device="cuda")                    # A = 1
    c["dt_bias"] = torch.full((H, D), math.log(s / (1 - s)), device="cuda")
    c["g"] = (torch.randn(1, T, H, D, generator=torch.Generator().manual_seed(1)) * 0.01).cuda().bfloat16()
    ref, fin = ref64(c, "VK")
    o, h = run_tr(c) if b == "triton" else run_fk(mod, c)
    pos = torch.arange(T, device="cuda") % 16
    e = o.double() - ref
    p15 = float((e[pos == 15] ** 2).sum().sqrt() / (ref[pos == 15] ** 2).sum().sqrt())
    p0_14 = float((e[pos != 15] ** 2).sum().sqrt() / (ref[pos != 15] ** 2).sum().sqrt())
    print(f"{os.path.basename(b):8s} s={s:5.3f} in-tile log2 total {-115.4*s:7.1f}: pos15 {p15:.2e} pos0-14 {p0_14:.2e} state {rel(h, fin):.2e}", flush=True)
