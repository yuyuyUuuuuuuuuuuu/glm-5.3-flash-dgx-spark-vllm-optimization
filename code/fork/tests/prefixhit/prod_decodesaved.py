#!/usr/bin/env python3
"""adv check: prefix hit on a DECODE-saved mamba checkpoint (spec decode + lazy KDA crossing a 4608 boundary) on
production, vs fresh prefills of the same tokens. Concurrency 1, waits for an idle server before each request.
Key read from the GLM .env into this process only (never printed)."""
import json, os, sys, time, glob, urllib.request

BASE = "http://localhost:8888"
OUT = sys.argv[1]
ENVF = os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")
KEY = next(l.split("=", 1)[1].strip().strip("\"'") for l in open(ENVF) if l.startswith("VLLM_API_KEY="))
B = int(os.environ.get("ADV_B", "36864"))      # checkpoint the decode crosses
LEAD = int(os.environ.get("ADV_LEAD", "120"))  # prompt ends LEAD tokens before B
NGEN = int(os.environ.get("ADV_NGEN", "400"))
SFX = [int(x) for x in os.environ.get("ADV_SFX", "8,16,17,24,48,120,250").split(",")]
GEN = int(os.environ.get("ADV_GEN", "12"))


def http(path, body=None, timeout=1200):
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
        if r == 0 or (r < 2 and time.time() - t0 > 90):
            return r
        if time.time() - t0 > 1200:
            raise SystemExit("production busy for 20 min; stopping")
        time.sleep(3)


model = json.loads(http("/v1/models"))["data"][0]["id"]
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3*/[n-z]*.py"))
text = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
tok = lambda s: json.loads(http("/tokenize", {"model": model, "prompt": s, "add_special_tokens": False}))["tokens"]
Q = tok("\n\n---\nIgnore the reference text above. Write a long, original fantasy story about a lighthouse keeper "
        "and a talking fox who argue about the sea. Begin now.\n\nThe")
P = B - LEAD
T = tok(text[: (P + 200) * 6])[: P - len(Q)]
assert len(T) == P - len(Q), len(T)
prompt = T + Q
stamp = str(int(time.time()))
SW = "advW" + stamp
res = {"model": model, "B": B, "lead": LEAD, "q": len(Q), "runs": {}}


def comp(seq, salt, max_tokens, tag):
    r0 = wait_idle()
    t0 = time.time()
    body = {"model": model, "prompt": seq, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 20,
            "cache_salt": salt, "return_tokens_as_token_ids": True, "ignore_eos": True}
    out = json.loads(http("/v1/completions", body))
    ch = out["choices"][0]
    u = out.get("usage", {})
    lp = ch["logprobs"]
    rec = {"tag": tag, "len": len(seq), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "ids": [int(t.split(":")[1]) for t in lp["tokens"]],
           "lp": [{k.split(":")[1]: v for k, v in d.items()} for d in lp["top_logprobs"]],
           "secs": round(time.time() - t0, 2), "running_before": r0}
    print(f"{tag:12s} len {len(seq)} cached {rec['cached']} secs {rec['secs']} run_before {r0}", flush=True)
    return rec


w = comp(prompt, SW, NGEN, "warm")
res["runs"]["warm"] = w
G = prompt + w["ids"]
res["groups"] = []
for s in SFX:
    if B + s > len(G) - 1:
        continue
    seq = G[: B + s]
    g = {"suffix": s, "decode_next": G[B + s]}
    for arm, salt in (("hit", SW), ("fresh", f"advF{stamp}-{s}-1"), ("fresh2", f"advF{stamp}-{s}-2"), ("hit2", SW)):
        g[arm] = comp(seq, salt, GEN, f"{arm}+{s}")
    res["groups"].append(g)
    json.dump(res, open(OUT, "w"))
json.dump(res, open(OUT, "w"))
print("wrote", OUT)
