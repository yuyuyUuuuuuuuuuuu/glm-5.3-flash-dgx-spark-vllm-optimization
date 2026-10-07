#!/usr/bin/env python3
"""Target's own per-token probabilities along each probed reference (one prefill per text, prompt_logprobs=2):
p_top1(j) and p(ref_j) at every reference position j. Read-only, concurrency 1."""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from trace_stream import BASE, MODEL, headers, metrics
from probe_positions import wait_idle

def main():
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src)); res = json.load(open(out)) if os.path.exists(out) else {}
    for w, rec in d.items():
        if w in res: continue
        wait_idle()
        ids = rec["prompt_ids"] + rec["ref_ids"]
        body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 2}
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=headers(), method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            o = json.load(r)
        pl = o.get("prompt_logprobs") or o["choices"][0].get("prompt_logprobs")
        n0 = len(rec["prompt_ids"])
        top1, pref = [], []
        for j, tok in enumerate(rec["ref_ids"]):
            ent = pl[n0 + j] or {}
            vals = {int(k): v["logprob"] if isinstance(v, dict) else v for k, v in ent.items()}
            top1.append(max(vals.values()) if vals else None)
            pref.append(vals.get(tok))
        res[w] = {"lp_top1": top1, "lp_ref": pref}
        print(w, "ok", sum(1 for x in pref if x is not None), flush=True)
        json.dump(res, open(out, "w"))

main()
