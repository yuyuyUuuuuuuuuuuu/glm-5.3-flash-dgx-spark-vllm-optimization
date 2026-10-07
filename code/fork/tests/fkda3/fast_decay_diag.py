#!/usr/bin/env python3
"""FKDA3 Part A follow-up: the strong-decay ("fast") gate regime, where adv_accuracy.py found the fkda2 build's
worst-token error ABOVE the stock build's and the Triton chain's. One process per build (all builds register the
same op namespace): `fast_decay_diag.py <build dir> <out.json>`; inputs and the fp64 reference are cached in
/fkda/acc/fast_cache.pt by the first run so every build sees identical bytes.
Reports out rel-RMS, worst token, and the error split by position inside the 16-token tile and by the per-token
decay strength (how far the in-tile cumulative log2-decay has fallen)."""
import math
import os
import sys

import torch

sys.path.insert(0, "/w/tests/fkda3")
from adv_accuracy import Gen, ref64, rel, D, LB  # noqa: E402

bdir, outp = sys.argv[1], sys.argv[2]
regime = sys.argv[3] if len(sys.argv) > 3 else "fast"
T = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
cache = f"/fkda/acc/diag_cache_{regime}_{T}.pt"
if os.path.exists(cache):
    c, ref, fin = torch.load(cache)
    c = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in c.items()}
    ref, fin = ref.cuda(), fin.cuda()
else:
    c = Gen(707, None).case([T], init="rand", regime=regime)
    ref, fin = ref64(c, "VK")
    torch.save(({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in c.items()}, ref.cpu(), fin.cpu()), cache)
out = {}
if bdir == "triton":
    from adv_accuracy import run_tr
    o, h = run_tr(c)
else:
    sys.path.insert(0, bdir)
    mod = [f[:-len(".abi3.so")] for f in os.listdir(bdir) if f.endswith(".abi3.so")][0]
    __import__(mod)
    from adv_accuracy import run_fk
    o, h = run_fk(mod, c)
torch.cuda.synchronize()
H = c["q"].shape[2]
e = (o.double() - ref)                                   # [T,H,D]
r = ref.norm(dim=-1).clamp_min(1e-30)                    # per (t,h)
te = e.norm(dim=-1) / r                                  # per (t,h) rel err
out["out"] = rel(o, ref)
out["state"] = rel(h, fin)
out["tok_max"] = float((e.norm(dim=(1, 2)) / ref.norm(dim=(1, 2))).max())
out["th_p999"] = float(te.flatten().quantile(0.999))
out["th_max"] = float(te.max())
pos = torch.arange(T, device="cuda") % 16
out["by_tile_pos"] = [float((e[pos == p] ** 2).sum().sqrt() / (ref[pos == p] ** 2).sum().sqrt()) for p in range(16)]
# per-head error vs the head's mean gate
A = torch.exp(c["A_log"].double())
gate = LB * torch.sigmoid(A.view(1, H, 1) * (c["g"][0].double() + c["dt_bias"].double().view(1, H, D)))
out["by_head"] = [[float(gate[:, hh].mean()), float((e[:, hh] ** 2).sum().sqrt() / (ref[:, hh] ** 2).sum().sqrt())]
                  for hh in range(H)]
out["ref_rms"] = float(ref.pow(2).mean().sqrt())
print(f"{bdir:45s} out {out['out']:.3e} state {out['state']:.3e} tok_max {out['tok_max']:.3e} (t,h) p99.9 "
      f"{out['th_p999']:.3e} max {out['th_max']:.3e}")
print("   by tile position:", " ".join(f"{x:.2e}" for x in out["by_tile_pos"]))
import json
json.dump(out, open(outp, "w"))
