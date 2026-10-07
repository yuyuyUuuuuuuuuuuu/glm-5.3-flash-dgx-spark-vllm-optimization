#!/usr/bin/env python3
"""[w8a8] Summarise the handoff A/B runs of the flag (GLM53_DENSE_W8A8) on vs off (docs/DENSE_W8A8.md 6).

Same shape as tests/poolfix_ab_summary.py (the kpooldown A/B), with the w8a8 differences this feature calls for:
the tokens of each request per run (greedy identity within an arm and across its engines), the decode ms/step, the
consistency probe KL (W8A8 prefill vs the untouched Marlin decode path), and the shadow verdict. The shadow's
feature pass (P vs F) is trivially IDENTICAL for this knob (the harness cannot force GLM53_DENSE_W8A8 off for P);
the control P vs P IDENTICAL is the meaningful part - docs/DENSE_W8A8.md 6.

Run INSIDE the production image (needs torch for records.pt):
  GPU_RUN_RO=/tmp/w8a8-ab:/tmp/w8a8-ab tests/gpu_run.sh python3 tests/w8a8_ab_summary.py /tmp/w8a8-ab

Per labelled run dir (<dir>/<label>/): the tokens of each request and of every decode step (result.json + records.pt),
the decode ms/step (records.pt per-step wall clock "t" over the steps that are NOT prefills, i.e. spec-verify steps,
paired per run), the consistency probe (compare.consistency: KL(decode A_j || fresh prefill B_j), mean/max), and the
shadow verdict from the run log. Printed as a table suitable for docs/DEPLOY_R16L.md §3.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def consistency(res) -> list[float]:
    """tests/handoff/compare.py:consistency — KL(A_j || B_j) from the top-5 logprobs of the decoded token and of the
    fresh prefill of prompt + the first j tokens."""
    if "B" not in res:
        return []
    out = []
    for j, pb in enumerate(res["B"]):
        pa = res["logprobs"][j] if j < len(res["logprobs"]) else {}
        keys = set(pa) | set(pb)
        kl = 0.0
        for k in keys:
            pa_k, pb_k = pa.get(k, -20.0), pb.get(k, -20.0)
            import math
            kl += math.exp(pa_k) * (pa_k - pb_k)
        out.append(kl)
    return out


def main() -> int:
    root = sys.argv[1]
    labels = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    print(f"{'run':8s} {'flag':5s} {'gen A0':>7s} {'gen A1':>7s} {'dec steps':>9s} {'ms/step':>8s} {'ms/step2':>8s} "
          f"{'KL mean':>8s} {'KL max':>8s}  capture/shadow")
    import torch
    for lab in labels:
        d = os.path.join(root, lab)
        try:
            res = json.load(open(os.path.join(d, "result.json")))
        except FileNotFoundError:
            continue
        env = open(os.path.join(d, "env.txt")).read()
        flag = "1" if "GLM53_DENSE_W8A8=1" in env else ("0" if "GLM53_DENSE_W8A8=0" in env else "unset")
        prom = re.search(r"--prompts ([0-9,]+)", open(os.path.join(d, "container.log")).read())
        extra = f" prompts={prom.group(1)}" if prom else ""
        recs = torch.load(os.path.join(d, "records.pt"), map_location="cpu", weights_only=False)["records"]
        dec = [r for r in recs if not r.get("prefill")]
        # per-request wall clock over its decode steps (records carry the step's start time "t")
        steps = sorted(dec, key=lambda r: r["step"])
        spans = []
        for r0, r1 in zip(steps, steps[1:]):
            dt = (r1["t"] - r0["t"]) * 1000.0
            spans.append(dt)
        spans.sort()
        med = spans[len(spans) // 2] if spans else float("nan")
        mean = sum(spans) / len(spans) if spans else float("nan")
        kl = consistency(res)
        log = open(os.path.join(d, "container.log")).read()
        capture = "FULL" if "Capturing CUDA graphs (FULL): 100%" in log else "?"
        shadow = re.search(r"control \(P vs P\) (IDENTICAL|DIFFERENT); feature \(P vs F\) (IDENTICAL|DIFFERENT)", log)
        gen = [len(r["gen"]) for r in res["requests"]]
        print(f"{lab:8s} {flag:5s}{extra:18s} {gen[0]:>7d} {gen[1] if len(gen) > 1 else 0:>7d} {len(steps):>9d} "
              f"{mean:>8.1f} {med:>8.1f} {(sum(kl) / len(kl) if kl else float('nan')):>8.4f} "
              f"{(max(kl) if kl else float('nan')):>8.4f}  {capture}, "
              f"shadow {shadow.group(1)}/{shadow.group(2) if shadow else '?'}")
        # the per-request spans, for the record
        print(f"     per-step ms: min {min(spans):.1f} p50 {med:.1f} p90 {spans[int(0.9 * (len(spans) - 1))]:.1f} "
              f"max {max(spans):.1f} (n={len(spans)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
