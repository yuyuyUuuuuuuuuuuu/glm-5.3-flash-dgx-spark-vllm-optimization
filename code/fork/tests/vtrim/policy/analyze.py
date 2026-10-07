#!/usr/bin/env python3
"""Reproduces every number of docs/SPEC_VTRIM.md section 1-2 from the production traces in docs/logs/spec_vtrim/traces
(no GPU, no production access): adaptive-K lag inference, the per-step cost model, the uncensored acceptance
L(p) at every position of 13 temp-0 references, the CPU K-policy sweep (position replay), the target-confidence
proxy for confidence trimming and the oracle bounds. Usage: python3 tests/vtrim/policy/analyze.py"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sim import simulate_trace                     # noqa: E402
from policy_eval import EMA, Fixed, ms, WS, C0, SLOPE  # noqa: E402
from posreplay import load_probe                   # noqa: E402
T = os.path.join(HERE, "../../../docs/logs/spec_vtrim/traces")
MOE = {1: 319, 2: 452, 3: 585, 4: 693, 5: 784, 6: 850, 7: 906, 8: 956}   # us/layer at T rows (bench_dead_rows.log)

print("== 1. adaptive-K decision lag (production trace, 20 runs): K_t from acceptances <= t-1-lag")
ada = json.load(open(f"{T}/ada_a.json"))
for lag in (0, 1, 2, 3, 4):
    viol = tot = 0
    for r in ada["runs"]:
        a = [x[1] - 1 for x in r["chunks"][1:]]
        Ks = simulate_trace(a, lag); viol += sum(x > k for k, x in zip(Ks, a)); tot += len(a)
    print(f"  lag {lag}: steps whose accepted count exceeds the simulated K: {viol}/{tot}")
print("== 2. step time vs K (inferred K, lag 2), per workload: median ms")
rows = []
for r in ada["runs"]:
    c = r["chunks"]; a = [x[1] - 1 for x in c[1:]]; dt = np.diff([x[0] for x in c]) * 1000
    for i, (k, y) in enumerate(zip(simulate_trace(a, 2), dt)):
        rows.append((r["workload"], k, y, i))
for w in WS:
    print("  %-10s" % w, {k: (sum(1 for x in rows if x[0] == w and x[1] == k),
                             round(float(np.median([x[2] for x in rows if x[0] == w and x[1] == k])), 1))
                         for k in (4, 5, 7) if any(x[0] == w and x[1] == k for x in rows)})
print(f"  linear fit used below: ms = C0[w] + {SLOPE} * K, C0 = {C0}")
PA = load_probe(f"{T}/probe_a.json"); PB = load_probe(f"{T}/probe_b.json")
texts = {w: [] for w in WS}
tlp = json.load(open(f"{T}/tlp.json"))
for src in (PA, PB):
    for k, v in src.items():
        texts[k.split("_")[0]].append((k, v["L"], np.exp(np.array([x if x is not None else -9 for x in tlp[k]["lp_top1"]]))))
print("== 3. uncensored acceptance L(p) of a fresh K=7 block at every position (13 references, 200 tokens each)")
for w in WS:
    for k, L, _ in texts[w]:
        print(f"  {k:16s} E[L] {L[L >= 0].mean():.2f}  P(L>=i) {[round(float((L >= i).mean()), 2) for i in range(1, 8)]}")


def replay(w, mk, lag=2, trim=None, dead_save=False):
    toks = t = 0.0
    for k, L, c in texts[w]:
        pol = mk(); p = 0; Ks = []; As = []
        while p < 199:
            K = pol.choose(); l = max(int(L[p]), 0)
            n = K
            if trim == "oracle":
                n = min(l, K)
            elif isinstance(trim, float):
                S = 1.0; n = 0
                for i in range(1, K + 1):
                    S *= c[p + i] if p + i < len(c) else 0.0
                    if S >= trim: n = i
                    else: break
            a = min(l, n); Ks.append(K); As.append(a)
            j = len(As) - 1 - lag
            if j >= 0: pol.observe(Ks[j], As[j])
            t += ms(w, K) - (42 * (MOE[K + 1] - MOE[n + 1]) / 1000 if dead_save else 0); p += a + 1
        toks += p
    return toks / t * 1000


print("== 4. CPU policies, position replay (tok/s; % vs production's EMA {4,5,7})")
base = {w: replay(w, lambda: EMA()) for w in WS}
def row(name, f):
    v = {w: f(w) for w in WS}
    print("  %-26s" % name + "  ".join("%s %5.1f (%+5.1f%%)" % (w[:4], v[w], 100 * (v[w] / base[w] - 1)) for w in WS))
row("EMA {4,5,7} (production)", lambda w: base[w])
for K in range(2, 8): row(f"fixed K={K}", lambda w, K=K: replay(w, lambda: Fixed(K)))
for ks in [(3, 4, 5, 7), (2, 4, 7), (4, 7), (4, 5, 6, 7), (5, 7), (3, 5, 7)]:
    row(f"EMA {ks}", lambda w, ks=ks: replay(w, lambda: EMA(kset=ks)))
for al, mg in [(0.15, 1.0), (0.4, 0.5), (0.25, 0.5), (0.25, 1.5)]:
    row(f"EMA a={al} m={mg}", lambda w, al=al, mg=mg: replay(w, lambda: EMA(alpha=al, margin=mg)))
row("EMA lag 0 (no async lag)", lambda w: replay(w, lambda: EMA(), lag=0))
print("== 5. per-step trimming inside the EMA's K, dead rows cost no routed-expert bytes (MoE table above)")
row("oracle n* = L", lambda w: replay(w, lambda: EMA(), trim="oracle", dead_save=True))
for tau in (0.3, 0.5, 0.7):
    row(f"target-conf proxy tau {tau}", lambda w, tau=tau: replay(w, lambda: EMA(), trim=tau, dead_save=True))
