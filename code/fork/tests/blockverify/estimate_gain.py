#!/usr/bin/env python3
"""Expected accepted tokens per step, block vs standard verification, on synthetic LLM-like draft/target chains
calibrated to production's per-position acceptance (host CPU, numpy only; no GPU, no production access).

Basis (stated in docs/BLOCK_VERIFY.md): production cannot be sampled here, so the target/draft pairs are a model.
For each chain the target p_i (temperature 1.0, top_p 0.95, sort-based nucleus) and the DFlash2-like draft q_i
(the target's pre-nucleus logits + Gaussian noise, restricted to its own top-16 candidates) are drawn fresh at every
position (a random LM sampled along the drafted path), the draft X_i ~ q_i, and the expected accepted length given
the drafts is computed exactly (Rao-Blackwellized, no acceptance coin is sampled):
  standard: E[tau | X] = sum_k prod_{i<=k} min(1, p_i(X_i)/q_i(X_i))
  block   : P_i = min(P_{i-1} p_i(X_i)/q_i(X_i), 1), h_i = r_i / (r_i + 1 - P_i) with
            r_i = sum_x max(P_i p_{i+1}(x) - q_{i+1}(x), 0) (i < n), h_n = P_n  (Sun et al. 2024, Alg. 2;
            the image's _rejection_kernel), tau = max{i : u_i <= h_i}, so P(tau >= k | X) = 1 - prod_{i>=k} (1 - h_i)
The per-position noise sigma_k is calibrated so that the standard conditional acceptance E[sum min(p, q)] matches
production's vLLM SpecDecoding per-position rates (0.70, 0.48, 0.34, 0.21 unconditional -> 0.70, 0.686, 0.708,
0.618 conditional; positions 5-7 are rarely verified under adaptive K {4,5,7}: 0.60 assumed). Several entropy
mixtures are run because the block gain depends on the spread of p/q ratios, not on the mean acceptance alone.
Output: tokens/step (= 1 + E[tau]) for n = 4, 5, 7 per family, and the relative gain, with standard errors.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

V = 4096           # vocab of the synthetic LM (large enough for a realistic tail; the nucleus decides)
TOPK = 16          # DFlash2 selector_top_k
TOP_P = 0.95
COND_ACC = [0.70, 0.686, 0.708, 0.618, 0.60, 0.60, 0.60]
BULK_SHIFT = 7.0
B = int(sys.argv[1]) if len(sys.argv) > 1 else 6000      # chains per family
rng_master = np.random.default_rng(20260929)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nucleus(logits, top_p):
    """Sort-based top-p (the image's PyTorch path): mask tokens whose ascending cumulative prob <= 1 - top_p."""
    order = np.argsort(logits, axis=1)
    srt = np.take_along_axis(logits, order, axis=1)
    cum = np.cumsum(softmax(srt), axis=1)
    msk = cum <= 1 - top_p
    msk[:, -1] = False
    out = logits.copy()
    rows = np.arange(logits.shape[0])[:, None]
    out[rows, order] = np.where(msk, -np.inf, srt)
    return out


def target_logits(rng, n, family, kind=None):
    """Pre-nucleus target logits. Families differ in how entropy is spread across positions."""
    z = rng.standard_normal((n, V)) - BULK_SHIFT     # the bulk: ~4096 * e^-6.5 = 6 units of mass
    if family == "mixed":            # 45 % near-deterministic, 35 % moderate, 20 % flat ("runs": sticky per chain)
        if kind is None:
            kind = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
        gap = np.where(kind == 0, rng.uniform(9, 14, n), np.where(kind == 1, rng.uniform(5, 9, n), rng.uniform(2, 5, n)))
        head = np.where(kind == 0, 1, np.where(kind == 1, rng.integers(2, 6, n), rng.integers(4, 24, n)))
    elif family == "peaky":        # mostly confident, a few open choices (code-like)
        kind = rng.choice(2, size=n, p=[0.7, 0.3])
        gap = np.where(kind == 0, rng.uniform(10, 16, n), rng.uniform(4, 8, n))
        head = np.where(kind == 0, 1, rng.integers(2, 8, n))
    elif family == "flat":         # prose-like open distributions
        gap = rng.uniform(3, 8, n)
        head = rng.integers(3, 30, n)
    else:
        raise ValueError(family)
    head = np.minimum(head, 32)
    # a head of `head` tokens at `gap` (logit units above 0), spaced like a Zipf tail (duplicates ~ negligible)
    j = np.arange(32)[None, :]
    idx = rng.integers(0, V, size=(n, 32))
    val = np.where(j < head[:, None], gap[:, None] - 1.2 * np.log1p(j), -np.inf)
    rows = np.arange(n)[:, None]
    z[rows, idx] = np.maximum(z[rows, idx], val)
    return z


def draft_probs(rng, z, sigma, beta=1.0):
    """DFlash2-like draft: noisy copy of the target's logits (beta > 1: over-confident draft), restricted to its own
    top-16 candidates."""
    d = beta * z + sigma * rng.standard_normal(z.shape)
    kth = np.partition(d, -TOPK, axis=1)[:, -TOPK][:, None]
    d = np.where(d >= kth, d, -np.inf)
    return softmax(d)


def overlap(p, q):
    return np.minimum(p, q).sum(axis=1)


def base_family(family):
    return {"runs": "mixed", "mixed-sharp": "mixed"}.get(family, family)


def beta_of(family):
    return 1.6 if family == "mixed-sharp" else 1.0


def calibrate(family, target, rng_seed, n=1500):
    lo, hi = 0.0, 8.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(rng_seed)
        z = target_logits(rng, n, base_family(family))
        p = softmax(nucleus(z, TOP_P))
        a = overlap(p, draft_probs(rng, z, mid, beta_of(family))).mean()
        lo, hi = (mid, hi) if a > target else (lo, mid)
    return 0.5 * (lo + hi)


def run_family(family, sigmas, n_max=7):
    rng = np.random.default_rng(int(rng_master.integers(1 << 31)))
    P, Q, X = [], [], []
    kind = rng.choice(3, size=B, p=[0.45, 0.35, 0.20]) if family in ("mixed", "runs", "mixed-sharp") else None
    for k in range(n_max + 1):     # position n_max + 1 only feeds h at the last verified row
        if kind is not None and k > 0:   # "runs": keep the entropy class with prob 0.75 (easy / hard spans)
            redraw = rng.random(B) > (0.75 if family == "runs" else 0.0)
            kind = np.where(redraw, rng.choice(3, size=B, p=[0.45, 0.35, 0.20]), kind)
        z = target_logits(rng, B, base_family(family), kind)
        p = softmax(nucleus(z, TOP_P))
        q = draft_probs(rng, z, sigmas[min(k, n_max - 1)], beta_of(family))
        cum = np.cumsum(q, axis=1)
        x = (cum < rng.random((B, 1))).sum(axis=1).clip(max=V - 1)
        P.append(p); Q.append(q); X.append(x)
    rows = np.arange(B)
    ratio = [P[k][rows, X[k]] / Q[k][rows, X[k]] for k in range(n_max + 1)]
    acc_pos = [overlap(P[k], Q[k]).mean() for k in range(n_max)]
    res = {"sigma": [round(s, 3) for s in sigmas], "cond_acceptance": [round(float(a), 4) for a in acc_pos]}
    for n in (4, 5, 7):
        # standard
        run = np.ones(B)
        e_std = np.zeros(B)
        for k in range(n):
            run = run * np.minimum(1.0, ratio[k])
            e_std += run
        # block
        Pk = np.ones(B)
        h = []
        for i in range(n):
            Pk = np.minimum(Pk * ratio[i], 1.0)
            if i < n - 1:
                r = np.maximum(Pk[:, None] * P[i + 1] - Q[i + 1], 0.0).sum(axis=1)
                den = r + 1.0 - Pk
                h.append(np.where(den > 0, r / np.where(den > 0, den, 1.0), 1.0))
            else:
                h.append(Pk.copy())
        H = np.stack(h, axis=1)                       # [B, n]
        tail = np.cumprod((1.0 - H)[:, ::-1], axis=1)[:, ::-1]   # prod_{i>=k} (1 - h_i)
        e_blk = (1.0 - tail).sum(axis=1)
        d = e_blk - e_std
        res[f"n{n}"] = dict(
            std_tokens_per_step=float(1 + e_std.mean()), block_tokens_per_step=float(1 + e_blk.mean()),
            gain_pct=float(100 * d.mean() / (1 + e_std.mean())),
            gain_pct_se=float(100 * d.std(ddof=1) / math.sqrt(B) / (1 + e_std.mean())),
            std_uncond_per_pos=[round(float(np.mean(np.prod([np.minimum(1, ratio[j]) for j in range(k + 1)], axis=0))), 4)
                                for k in range(n)])
    return res


out = {}
for fam in ("mixed", "runs", "mixed-sharp", "peaky", "flat"):
    sig = [calibrate(fam, COND_ACC[k], 100 + k) for k in range(7)]
    r = run_family(fam, sig)
    out[fam] = r
    print(f"== family {fam}: sigma per position {r['sigma']}; standard conditional acceptance {r['cond_acceptance']}")
    for n in (4, 5, 7):
        x = r[f"n{n}"]
        print(f"   n={n}: tokens/step standard {x['std_tokens_per_step']:.4f} block {x['block_tokens_per_step']:.4f} "
              f"gain {x['gain_pct']:+.2f} % (SE {x['gain_pct_se']:.2f}); standard unconditional per position "
              f"{x['std_uncond_per_pos']}")
print(json.dumps(out))
