#!/usr/bin/env python3
"""FKDA2: per-position KL (quality_probe.py's metric: KL(A||B) over the union of the two top-20 sets) between the
prompt_logprobs.json files of tests/fkda2/e2e_kl.py runs. Host-side, stdlib only.
Usage: compare_e2e.py <e2e dir> A:B [A:B ...]   (labels = run directories)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tools/prodcheck"))
import quality_probe as QP  # noqa: E402


def kls(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    out = []
    for pa, pb in zip(A["prefill"], B["prefill"]):
        for ea, eb in zip(pa or [], pb or []):
            if ea and eb:
                out.append(QP.kl(ea, eb))
    return sorted(out)


def main():
    d = sys.argv[1]
    for pair in sys.argv[2:]:
        a, b = pair.split(":")
        ks = kls(os.path.join(d, a, "prompt_logprobs.json"), os.path.join(d, b, "prompt_logprobs.json"))
        n = len(ks)
        print(f"KL({a:>8s} || {b:<8s}) mean {sum(ks) / n:.5f} p95 {ks[int(0.95 * n)]:.5f} max {ks[-1]:.4f} "
              f"(n={n}, zero {sum(1 for x in ks if x == 0)})")


if __name__ == "__main__":
    main()
