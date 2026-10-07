#!/usr/bin/env python3
"""Markdown table of docs/logs/blockverify/exact_*.json (tests/blockverify/test_block_exactness.py results)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/logs/blockverify")
print("| config / n | variant | N sequences | seq chi2/dof, p | TV | step2 chi2/dof, p | law: mean(τ − E[τ ∣ X]), z | tokens/step |")
print("|---|---|---|---|---|---|---|---|")
for cfg in ("prod", "stress", "textbook", "vocab"):
    f = root / f"exact_{cfg}.json"
    if not f.exists():
        continue
    for key, res in json.loads(f.read_text()).items():
        for v in ("std-prod", "blk-prod", "blk-fix"):
            r = res.get(v)
            if not r:
                continue
            print(f"| {key} | {v} | {r['N']} | {r['seq_chi2']:.0f}/{r['seq_dof']}, {r['seq_p']:.3g} | {r['seq_tv']:.4f} | "
                  f"{r['step2_chi2']:.0f}/{r['step2_dof']}, {r['step2_p']:.3g} | {r['rb_mean_tau_minus_expected']:+.5f}, "
                  f"{r['rb_z']:+.2f} | {r['tokens_per_step']:.4f} |")
print()
print("| config / n | E[tau] standard (same drafts) | E[tau] block (same drafts) | tokens/step gain % (SE) | measured tokens/step std -> blk-fix |")
print("|---|---|---|---|---|")
for cfg in ("prod", "stress", "textbook", "vocab"):
    f = root / f"exact_{cfg}.json"
    if not f.exists():
        continue
    for key, res in json.loads(f.read_text()).items():
        a, b = res.get("std-prod"), res.get("blk-fix")
        if not a or not b:
            continue
        print(f"| {key} | {b['rb_e_std']:.4f} | {b['rb_e_blk']:.4f} | {b['rb_gain_pct']:+.2f} ({b['rb_gain_pct_se']:.2f}) | "
              f"{a['tokens_per_step']:.4f} -> {b['tokens_per_step']:.4f} ({100 * (b['tokens_per_step'] / a['tokens_per_step'] - 1):+.2f} %) |")
