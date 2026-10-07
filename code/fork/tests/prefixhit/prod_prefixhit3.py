#!/usr/bin/env python3
"""prefixhit prod3: hit-vs-fresh at an UNCERTAIN position (nodeA, GLM dir, under the prodbench lock, concurrency 1).

No prompt_logprobs (37k-position prompt logprobs are far too heavy for production). For each suffix s: seq = text
T[:C + s - q] + an open question (q tokens), so the first generated token is genuinely uncertain. seq runs as hit
(salt of a warm request over T[:C+1000], expected hit C = 36864), fresh / fresh2 (unique salts: noise floor), hit2.
Greedy, max_tokens PH_GEN (8), top-20 logprobs per generated token (positions compared while the tokens agree).
The key is read from ./.env into this process only. Waits for vllm:num_requests_running < 2 before every request.
"""
import glob
import json
import math
import os
import sys
import time
import urllib.request

BASE = "http://localhost:8888"
OUT = sys.argv[1]
C = int(os.environ.get("PH_C", "36864"))
SPAN = int(os.environ.get("PH_SPAN", "700"))
GEN = int(os.environ.get("PH_GEN", "8"))
SFX = [int(x) for x in os.environ.get("PH_SUFFIXES", "20,64,600").split(",")]
KEY = next(l.split("=", 1)[1].strip().strip("\"'") for l in open(".env") if l.startswith("VLLM_API_KEY="))


def http(path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def wait_idle():
    t0 = time.time()
    while True:
        r = 0.0
        for line in http("/metrics", timeout=10).splitlines():
            if line.startswith("vllm:num_requests_running"):
                r = float(line.split()[-1])
        if r < 2:
            return r
        if time.time() - t0 > 1200:
            raise SystemExit("production busy for 20 min; stopping")
        time.sleep(5)


model = json.loads(http("/v1/models"))["data"][0]["id"]
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3*/[a-m]*.py"))
text = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
need = C + 1200
T = json.loads(http("/tokenize", {"model": model, "prompt": text[: need * 6], "add_special_tokens": False}))["tokens"][:need]
assert len(T) == need
stamp = str(int(time.time()))
WARM = "phW" + stamp
res = {"model": model, "C": C, "span": SPAN, "runs": {}}


def req(seq, salt, tag, prompt_lp=None):
    r0 = wait_idle()
    t0 = time.time()
    body = {"model": model, "prompt": seq, "max_tokens": GEN, "temperature": 0.0, "logprobs": 20, "cache_salt": salt}
    out = json.loads(http("/v1/completions", body))
    ch = out["choices"][0]
    u = out.get("usage", {})
    rec = {"tag": tag, "len": len(seq), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "lp": ch["logprobs"]["top_logprobs"] if ch.get("logprobs") else [], "tokens": ch["logprobs"].get("tokens") if ch.get("logprobs") else None, "secs": round(time.time() - t0, 2),
           "running_before": r0}
    plp = ch.get("prompt_logprobs") or out.get("prompt_logprobs")
    if plp:
        # vLLM: list (one per prompt position, None for position 0) of {token_id: {logprob, rank, decoded_token}}
        rec["plp"] = [None if d is None else {k: (v["logprob"] if isinstance(v, dict) else v) for k, v in d.items()}
                      for d in plp[C:C + SPAN]]
    print(f"{tag:10s} len {len(seq)} cached {rec['cached']} secs {rec['secs']} plp {len(rec.get('plp') or [])}", flush=True)
    return rec



Q = json.loads(http("/tokenize", {"model": model, "prompt": "\n\n# Q: write one random English word, any word.\n# A:",
                                   "add_special_tokens": False}))["tokens"]
res["q_tokens"] = len(Q)
res["runs"]["warm"] = req(T[: C + 1000], WARM, "warm")
res["runs"]["groups"] = []
for s in SFX:
    seq = T[: C + s - len(Q)] + Q              # filler from the text, then an open question at the end
    g = {"suffix": s, "hit": req(seq, WARM, f"hit+{s}"), "fresh": req(seq, f"phF{stamp}-{s}-1", f"fresh+{s}"),
         "fresh2": req(seq, f"phF{stamp}-{s}-2", f"fresh2+{s}"), "hit2": req(seq, WARM, f"hit2+{s}")}
    res["runs"]["groups"].append(g)
    json.dump(res, open(OUT, "w"))
json.dump(res, open(OUT, "w"))
print("wrote", OUT)
