"""Shared setup for tests/test_prefill_cap.py and tests/bench_prefill_cap.py (docs/PREFILL_CAP.md).

Production's prefill env (docs/PRODUCTION_RESULT.md R11/R12): EXL3_FUSED_MOE=1, EXL3_FAT_KERNEL=1, EXL3_FAT_GROUPED=1,
EXL3_TEMP_ROWS_FUSED=256, MAX_NUM_BATCHED_TOKENS=16384 (sizes E3's grouped scratch: MNBT x EXL3_FAT_GROUPED_TOPK(8)
rows). Set before the production module is imported. Layers are built by production's own
process_weights_after_loading (harness.make_layer) with shared gate/up SUH, so they resolve to the "grouped" tier
exactly as the real checkpoint does (docs/PRODUCTION_RESULT.md: effective_tier=grouped).
"""
from __future__ import annotations

import os

for _k, _v in (("EXL3_FUSED_MOE", "1"), ("EXL3_FAT_KERNEL", "1"), ("EXL3_FAT_GROUPED", "1"),
               ("EXL3_TEMP_ROWS_FUSED", "256"), ("MAX_NUM_BATCHED_TOKENS", "16384")):
    os.environ.setdefault(_k, _v)
os.environ.pop("GLM53_PREFILL_FUSED_CAP", None)      # tests install explicitly

import torch  # noqa: E402

import harness as H  # noqa: E402

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0                                          # production: swiglu_limit or SWIGLU_LIMIT_DEFAULT (10.0)
BUCKETS = (16, 32, 64, 128, 256, 512, 1024)


def routing(kind: str, T: int, seed: int, dev) -> torch.Tensor:
    """Top-8 of 288 per token, Gumbel-top-k over fixed per-call expert popularity logits.
    real:      popularity logits sigma * N(0,1), sigma 1.1 (lognormal / Zipf-like skew; at T=13824 mean 384 rows,
               ~5 % of experts <= 32 rows, ~1/2 in 33..400, a heavy head up to several thousand rows)
    zipf:      logits -1.3 * log(rank + 8) over a random rank permutation (heavier head, flatter tail)
    collapsed: sigma 3.0 (random-token prompts: most rows on a few dozen experts, ~half the experts ~empty)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "real":
        logits = 1.1 * torch.randn(NEXP, generator=g)
    elif kind == "zipf":
        ranks = torch.randperm(NEXP, generator=g).double()
        logits = (-1.3 * torch.log(ranks + 8.0)).float()
    elif kind == "collapsed":
        logits = 3.0 * torch.randn(NEXP, generator=g)
    else:
        raise ValueError(kind)
    gd = torch.Generator(device=dev).manual_seed(seed + 1)
    u = torch.rand(T, NEXP, generator=gd, device=dev).clamp_(1e-12, 1 - 1e-7)
    score = logits.to(dev)[None, :] - torch.log(-torch.log(u))
    return score.topk(TOPK, dim=1).indices.to(torch.long).contiguous()


def weights_for(T: int, seed: int, dev) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 7)
    return H.random_weights(T, TOPK, g, dev)


def histogram(ids: torch.Tensor) -> dict:
    c = torch.bincount(ids.reshape(-1), minlength=NEXP).cpu()
    edges = (0,) + BUCKETS
    b = [int(((c > lo) & (c <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    b.append(int((c > BUCKETS[-1]).sum()))
    b[0] += int((c == 0).sum())
    return {"counts": c, "buckets": b, "max": int(c.max()), "zero": int((c == 0).sum()),
            "rows_gt": {n: int(c[c > n].sum()) for n in (16, 32, 48, 64, 96, 128, 256)}}


def fmt_hist(h: dict) -> str:
    names = ["<=16", "17-32", "33-64", "65-128", "129-256", "257-512", "513-1024", ">1024"]
    return " ".join(f"{a}:{b}" for a, b in zip(names, h["buckets"])) + f" (max {h['max']}, empty {h['zero']})"


def build(n_layers: int, seed0: int = 900, tf_install: bool = True):
    """(prod, xl, tf or None, layers). TF (integrate.install) goes in before the layers are built, as in production
    (its build hook registers each layer), so the prefill calls walk production's full stack: K2 apply hook ->
    production apply -> TF dispatcher (delegates B > R to production's exl3_moe) + E3."""
    xl = H.load_xl()
    prod = H.load_prod()
    tf = None
    if tf_install:
        tf = H.load_tf()
        import integrate

        rep = integrate.install(prodmod=prod, ext=xl, force=True)
        assert rep["installed"] and rep.get("apply_hook"), f"TF install: {rep}"
    return prod, xl, tf, make_layers(prod, n_layers, seed0)


def make_layers(prod, n_layers: int, seed0: int = 900) -> list:
    dev = torch.device("cuda", 0)
    layers = []
    for i in range(n_layers):
        W = H.Weights(NEXP, K, N, dev, seed=seed0 + i, shared_suh=True)
        L = H.make_layer(prod, W)
        assert L._exl3_fat_effective_tier == "grouped", L._exl3_fat_tier_reason
        L._test_weights = W
        layers.append(L)
    return layers


def apply(prod, x, ids, w, layer):
    """Production's entry exactly as apply_exl3_experts calls it (module-global name -> whatever hooks are in)."""
    emap = prod.pin_exl3_expert_map(layer, x.device)
    return prod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, emap, LIMIT)
