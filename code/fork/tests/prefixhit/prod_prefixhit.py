#!/usr/bin/env python3
"""prefixhit: production check of prefix-hit vs fresh prefill consistency (runs ON nodeA, concurrency 1).

Usage (from nodeC):  ssh nodeA 'cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && flock ~/tf-exl3-assets-prodbench.lock \
                         python3 ~/tf-exl3-assets/prefixhit/prod_prefixhit.py OUT.json'
The API key is read from ./.env into this process only (never printed). Before every request the script waits until
vllm:num_requests_running < 2 (gives up after 20 min).

Plan: T = real text tokens (server /tokenize). Warm W = T[:c1 + 1000] once (prefill chunk ends land on the 4608
grid, so the checkpoints c0 = 27648 / c1 = 36864 are materialised). Then for every suffix s and checkpoint c:
seq = T[:c + s], run as hit (salt 'phA'), hit2 (salt 'phA'), fresh (unique salt), fresh2 (unique salt);
max_tokens 1, temperature 0, top-20 logprobs. usage.prompt_tokens_details.cached_tokens is recorded per request.
"""
import glob
import json
import os
import sys
import time
import urllib.request

BASE = "http://localhost:8888"
OUT = sys.argv[1]
SUFFIXES = [int(x) for x in os.environ.get("PH_SUFFIXES", "4,16,64,600").split(",")]
CKPTS = [int(x) for x in os.environ.get("PH_CKPTS", "36864,27648").split(",")]
KEY = None
for line in open(".env"):
    if line.startswith("VLLM_API_KEY="):
        KEY = line.split("=", 1)[1].strip().strip("\"'")
assert KEY, "no VLLM_API_KEY in ./.env"


def http(path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def running():
    for line in http("/metrics", timeout=10).splitlines():
        if line.startswith("vllm:num_requests_running"):
            return float(line.split()[-1])
    return 0.0


def wait_idle():
    t0 = time.time()
    while True:
        r = running()
        if r < 2:
            return r
        if time.time() - t0 > 1200:
            raise SystemExit("production busy for 20 min; stopping")
        time.sleep(5)


model = json.loads(http("/v1/models"))["data"][0]["id"]
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3*/[a-m]*.py"))
text = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
need = max(CKPTS) + max(SUFFIXES) + 1200
toks = json.loads(http("/tokenize", {"model": model, "prompt": text[: need * 6], "add_special_tokens": False}))["tokens"]
assert len(toks) >= need, (len(toks), need)
T = toks[:need]
res = {"model": model, "n_text_tokens": len(T), "suffixes": SUFFIXES, "ckpts": CKPTS, "runs": []}


def one(seq, salt, tag):
    r0 = wait_idle()
    t0 = time.time()
    body = {"model": model, "prompt": seq, "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
            "cache_salt": salt}
    out = json.loads(http("/v1/completions", body))
    ch = out["choices"][0]
    lp = ch["logprobs"]["top_logprobs"][0] if ch.get("logprobs") else {}
    u = out.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    rec = {"tag": tag, "len": len(seq), "salt": salt, "cached": cached, "prompt_tokens": u.get("prompt_tokens"),
           "lp": lp, "text": ch.get("text"), "secs": round(time.time() - t0, 2), "running_before": r0}
    print(f"{tag:8s} len {len(seq)} cached {cached} secs {rec['secs']} top {max(lp, key=lp.get) if lp else None!r}",
          flush=True)
    return rec


stamp = str(int(time.time()))
res["runs"].append(one(T[: max(CKPTS) + 1000], "phA" + stamp, "warm"))
for c in CKPTS:
    for s in SUFFIXES:
        seq = T[: c + s]
        grp = {"ckpt": c, "suffix": s}
        grp["hit"] = one(seq, "phA" + stamp, "hit")
        grp["fresh"] = one(seq, f"phF{stamp}-{c}-{s}-1", "fresh")
        grp["hit2"] = one(seq, "phA" + stamp, "hit2")
        grp["fresh2"] = one(seq, f"phF{stamp}-{c}-{s}-2", "fresh2")
        res["runs"].append(grp)
        json.dump(res, open(OUT, "w"))
json.dump(res, open(OUT, "w"))
print("wrote", OUT)
