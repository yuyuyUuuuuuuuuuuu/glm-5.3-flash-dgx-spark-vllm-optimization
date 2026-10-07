#!/usr/bin/env python3
"""Quality probe against the local GLM server (run on nodeA). Two measurements, saved as JSON:
  decode : 20 fixed prompts, greedy (temperature 0), max_tokens 96, logprobs=5 per generated token
  prefill: 6 fixed texts, prompt_logprobs=20 (top-20 per position), max_tokens 1
Compare two saved runs with --compare A.json B.json:
  decode : identical completions, first-divergence token index, mean/max |dlogprob| of the chosen token on the shared prefix
  prefill: per-position KL(A||B) over the union of both top-20 sets (missing mass -> floor), mean / p95 / max
Auth: API_KEY / VLLM_API_KEY env. Uses only the stdlib."""
import json, math, os, sys, time, urllib.request

URL = os.environ.get("GLM_URL", "http://127.0.0.1:8888/v1/completions")
MODEL = os.environ.get("GLM_MODEL", "GLM-5.3-Flash-EXL3")
KEY = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY") or ""

DECODE = [
    "The three laws of thermodynamics can be summarized as follows:",
    "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
    "日本の四季について、それぞれの特徴を説明すると、",
    "Write a haiku about autumn rain:\n",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\n",
    "The capital of Australia is",
    "In 1905, Albert Einstein published four papers that",
    "機械学習における過学習とは、",
    "import numpy as np\n\ndef softmax(x):\n",
    "Once upon a time, in a village at the edge of a great forest,",
    "The difference between TCP and UDP is that",
    "東京から大阪へ新幹線で移動する場合、",
    "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n",
    "A good README file should contain",
    "The derivative of sin(x) * x^2 with respect to x is",
    "量子コンピュータが従来のコンピュータと異なる点は、",
    "<!DOCTYPE html>\n<html>\n<head>\n",
    "The main causes of the French Revolution were",
    "Q: What is 17 * 23?\nA:",
    "明日の天気予報によると、",
]
PREFILL = [
    "The history of mathematics spans thousands of years, from ancient counting systems to modern abstract algebra. " * 6,
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n" * 5,
    "人工知能の研究は、推論、学習、知覚、言語理解など、人間の知的な能力を計算機で実現することを目指している。" * 6,
    "In a distributed system, consensus algorithms such as Paxos and Raft allow a set of nodes to agree on a value. " * 6,
    "The patient presented with a two-day history of fever, cough, and shortness of breath, and was admitted for observation. " * 5,
    "東京都は日本の首都であり、政治、経済、文化の中心地として多くの人々が暮らしている。交通網も発達している。" * 6,
]


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def run(out):
    res = {"t": time.strftime("%F %T"), "decode": [], "prefill": []}
    for p in DECODE:
        r = post({"model": MODEL, "prompt": p, "max_tokens": 96, "temperature": 0, "logprobs": 5, "seed": 0})
        c = r["choices"][0]; lp = c.get("logprobs") or {}
        res["decode"].append({"prompt": p, "text": c["text"], "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs")})
    for p in PREFILL:
        r = post({"model": MODEL, "prompt": p, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20})
        pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
        res["prefill"].append(pl)
    json.dump(res, open(out, "w"), ensure_ascii=False)
    print(f"wrote {out}: {len(res['decode'])} decode, {len(res['prefill'])} prefill texts")


def kl(a, b, floor=1e-9):
    """a, b: {token: logprob} top-k dicts. KL(A||B) over union support; unseen mass -> floor."""
    keys = set(a) | set(b)
    pa = {k: math.exp(a[k]) if k in a else floor for k in keys}
    pb = {k: math.exp(b[k]) if k in b else floor for k in keys}
    za, zb = sum(pa.values()), sum(pb.values())
    return sum((pa[k] / za) * math.log((pa[k] / za) / (pb[k] / zb)) for k in keys)


def as_dict(entry):
    if entry is None: return None
    out = {}
    for k, v in entry.items():
        out[k] = v["logprob"] if isinstance(v, dict) else v
    return out


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    same = 0; div = []; dl = []
    for x, y in zip(A["decode"], B["decode"]):
        if x["text"] == y["text"]: same += 1
        ta, tb = x["tokens"] or [], y["tokens"] or []
        n = next((i for i, (u, v) in enumerate(zip(ta, tb)) if u != v), min(len(ta), len(tb)))
        div.append(n)
        for i in range(n):
            if x["token_logprobs"][i] is not None and y["token_logprobs"][i] is not None:
                dl.append(abs(x["token_logprobs"][i] - y["token_logprobs"][i]))
    print(f"decode: identical {same}/{len(A['decode'])}; first divergence token idx median {sorted(div)[len(div)//2]} min {min(div)}; "
          f"|dlogprob| on shared prefix mean {sum(dl)/max(1,len(dl)):.2e} max {max(dl) if dl else 0:.2e} (n={len(dl)})")
    ks = []
    for pa, pb in zip(A["prefill"], B["prefill"]):
        for ea, eb in zip(pa or [], pb or []):
            da, db = as_dict(ea), as_dict(eb)
            if da and db: ks.append(kl(da, db))
    ks.sort()
    if ks:
        print(f"prefill KL(A||B) nats/token over top-20 union: mean {sum(ks)/len(ks):.4f} p95 {ks[int(0.95*len(ks))]:.4f} max {ks[-1]:.4f} (n={len(ks)})")


if __name__ == "__main__":
    if sys.argv[1] == "--compare": compare(sys.argv[2], sys.argv[3])
    else: run(sys.argv[1])
