#!/usr/bin/env python3
"""r16z8 engine smoke: bench_check-style output sanity of handoff-rig runs (tests/r16z7/run_kit.sh result.json) and the
proof / forbidden lines of their container logs.

Per run and request: generated tokens (must be the requested count), every logprob finite, doubled-token runs (adjacent equal ids)
and the longest repeated 4-gram count (degeneration heuristics, compared with the OFF arm), the first token where the
run differs from the reference run and the largest |logprob difference| of the chosen tokens before it. The reference
pair OFF vs OFF2 is the rig's run-to-run band (A/A); ON vs OFF must stay inside it.
Usage: engine_sanity.py <runs dir> <ref label> <label>... [--proof-labels a,b] [--need 'fixed string' ...]
       [--never 'fixed string' ...]   (--need / --never are checked in the container.log of every --proof-labels run)
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path


def load(d: Path, label: str) -> dict:
    return json.load(open(d / label / "result.json"))


def lp_of(step: dict, tok: int) -> float | None:
    v = step.get(str(tok))
    return None if v is None else float(v)


def main() -> int:
    args = sys.argv[1:]
    need, never, labels, plabs = [], [], [], []
    i = 0
    while i < len(args):
        if args[i] == "--proof-labels":
            plabs = args[i + 1].split(","); i += 2
        elif args[i] == "--need":
            need.append(args[i + 1]); i += 2
        elif args[i] == "--never":
            never.append(args[i + 1]); i += 2
        else:
            labels.append(args[i]); i += 1
    d, ref, rest = Path(labels[0]), labels[1], labels[2:]
    R = load(d, ref)
    fail = 0
    gen_n = int(R["meta"]["args"]["gen"])
    for lab in [ref] + rest:
        X = load(d, lab)
        for k, (rq, rr) in enumerate(zip(X["requests"], R["requests"])):
            g, gr = rq["gen"], rr["gen"]
            lps = rq["logprobs"]
            finite = all(math.isfinite(float(v)) for st in lps for v in st.values())
            dbl = sum(1 for a, b in zip(g, g[1:]) if a == b)
            ng = Counter(tuple(g[j:j + 4]) for j in range(len(g) - 3))
            rep4 = max(ng.values()) if ng else 0
            fd = next((j for j, (a, b) in enumerate(zip(g, gr)) if a != b), None)
            upto = fd if fd is not None else min(len(g), len(gr))
            dl = [abs(lp_of(lps[j], g[j]) - lp_of(rr["logprobs"][j], gr[j])) for j in range(upto)
                  if lp_of(lps[j], g[j]) is not None and lp_of(rr["logprobs"][j], gr[j]) is not None]
            ok = len(g) == gen_n and finite
            fail |= not ok
            print(f"[{lab} vs {ref}] request {k}: {len(g)}/{gen_n} tokens, logprobs finite {finite}, "
                  f"doubled-token runs {dbl}, max repeated 4-gram {rep4}, identical {fd is None}, first diff at {fd}, "
                  f"max |dlogprob| before it {max(dl) if dl else 0:.4g} (n {len(dl)}){'' if ok else '  <-- FAIL'}")
        if lab in plabs:
            log = (d / lab / "container.log").read_text(errors="replace")
            for s in need:
                n = log.count(s)
                print(f"[{lab}] need  ({n}) {s}{'' if n else '  <-- FAIL'}")
                fail |= n == 0
            for s in never:
                n = log.count(s)
                print(f"[{lab}] never ({n}) {s}{'' if n == 0 else '  <-- FAIL'}")
                fail |= n != 0
    print("ENGINE-SANITY:", "FAIL" if fail else "OK")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
