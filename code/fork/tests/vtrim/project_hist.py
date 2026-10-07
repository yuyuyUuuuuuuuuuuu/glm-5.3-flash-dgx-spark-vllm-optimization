#!/usr/bin/env python3
"""Project GLM53_SPEC_VTRIM=on from a SHADOW run's histogram (docs/SPEC_VTRIM.md section 5).

Input: the JSON the shadow module writes on rank 0 next to the vLLM cache (<cache root>/glm53_spec_vtrim_hist.json,
production: nodeA ~/.cache/vllm-glm53-flash/glm53_spec_vtrim_hist.json): hist[K][min(n*,K)][a] request-steps, with a =
the accepted drafts of the UNTRIMMED step. For every request-step, trimming would verify n = min(n*, K) drafts, accept
min(a, n) and skip the routed experts of K - n rows. Step time model: production's measured ms(K) (C0 + 4.46 K, the
trace fit of section 2) minus 42 MoE layers x (MoE(K+1 rows) - MoE(n+1 rows)) from the nodeC dead-row bench.
Every cut accepted draft counts as lost (no recovery by the next block). Calibrated against the exact position
replay of tests/vtrim/policy/analyze.py (target-confidence rule, taus 0.3-0.7, 13 references): the implied recovery is
~0 (-0.3..+0.4) because adaptive-K's EMA then sees lower acceptance and lowers K; the projection matched the replay
within 0.5 point except prose at tau 0.7 (projected +1.9 %, replay -0.8 %): treat it as an upper estimate at high tau.
Usage: project_hist.py <hist.json> [--before <earlier snapshot>] [--c0 53.7]
"""
import argparse
import json

MOE = {1: 319, 2: 452, 3: 585, 4: 693, 5: 784, 6: 850, 7: 906, 8: 956}
SLOPE = 4.46


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist")
    ap.add_argument("--before", default="", help="an earlier snapshot of the same file: project the difference only")
    ap.add_argument("--c0", type=float, default=53.7, help="per-step fixed ms (prose 53.7, ja 50.1, coding 57.1)")
    a = ap.parse_args()
    d = json.load(open(a.hist))
    if a.before:
        b = json.load(open(a.before))
        sub = lambda x, y: [sub(u, v) for u, v in zip(x, y)] if isinstance(x, list) else x - y
        d["hist"] = sub(d["hist"], b["hist"])
        if "hist_grid" in d and "hist_grid" in b:
            d["hist_grid"] = sub(d["hist_grid"], b["hist_grid"])
    print(f"{a.hist}: mode {d.get('mode')} tau {d.get('tau')} min {d.get('min')}")
    project(d["hist"], a.c0, f"tau {d.get('tau')} (the run's own)")
    for g, hg in zip(d.get("grid", []), d.get("hist_grid", [])):
        project(hg, a.c0, f"tau {g} (shadow grid)")


def project(h, c0, label):
    steps = tok0 = t0 = 0.0
    lost = dead = 0.0
    saved_ms = 0.0
    for K in range(8):
        for n in range(8):
            for acc in range(8):
                c = h[K][n][acc]
                if not c:
                    continue
                steps += c
                ms = c0 + SLOPE * K
                tok0 += c * (acc + 1)
                t0 += c * ms
                nn = min(n, K)
                lost += c * max(0, acc - nn)
                dead += c * (K - nn)
                saved_ms += c * 42 * (MOE[K + 1] - MOE[nn + 1]) / 1000
    if not steps:
        print("empty histogram"); return
    t1 = t0 - saved_ms
    base = tok0 / t0 * 1000
    cons = (tok0 - lost) / t1 * 1000
    print(f"  [{label}] {int(steps)} request-steps")
    print(f"  untrimmed: {tok0 / steps:.3f} tokens/step, {t0 / steps:.1f} ms/step -> {base:.1f} tok/s (model)")
    print(f"  trimmed  : dead rows {dead / steps:.2f}/step, MoE saved {saved_ms / steps:.2f} ms/step, accepted drafts cut "
          f"{lost / steps:.3f}/step ({100 * lost / max(tok0 - steps, 1):.1f}% of accepted)")
    print(f"  projected tok/s {cons:.1f} ({100 * (cons / base - 1):+.1f}%)")


if __name__ == "__main__":
    main()
