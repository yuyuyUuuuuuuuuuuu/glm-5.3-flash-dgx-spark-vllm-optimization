"""opt-decode-rev: token / logprob comparison of handoff result.json files (A vs B per request).
Usage: compare_handoff_results.py dirA dirB [dirC ...]  (each holds result.json; compares every dir to the first)"""
import json
import sys

base = json.load(open(f"{sys.argv[1]}/result.json"))["requests"]
for d in sys.argv[2:]:
    other = json.load(open(f"{d}/result.json"))["requests"]
    for i, (x, y) in enumerate(zip(base, other)):
        gx, gy = x["gen"], y["gen"]
        first = next((j for j, (p, q) in enumerate(zip(gx, gy)) if p != q), None)
        lx, ly = x.get("logprobs") or [], y.get("logprobs") or []
        n_upto = first if first is not None else min(len(lx), len(ly))
        d_upto = []
        for p, q in list(zip(lx, ly))[:n_upto]:      # per step: top-k logprob dicts {token: logprob}
            common = set(p) & set(q)
            d_upto.append(max((abs(p[t] - q[t]) for t in common), default=0.0))
        print(f"{sys.argv[1]} vs {d} request {i}: {len(gx)}/{len(gy)} tokens, identical {gx == gy}, first diff at "
              f"{first}; max |dlogprob| before it {max(d_upto) if d_upto else 0:.4g} (n {len(d_upto)})")
