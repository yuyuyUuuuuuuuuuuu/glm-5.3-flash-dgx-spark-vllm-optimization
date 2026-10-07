#!/usr/bin/env python3
"""Long-context quality probe (prefill path): 3 fixed real-text prompts of ~14k tokens with prompt_logprobs=5, compared per position.
  run:      quality_long.py <out.json>
  compare:  quality_long.py --compare A.json B.json   -> KL over the top-5 union (missing mass floor), top-1 agreement, only positions >= 4608
Fixed texts (license prose + Python stdlib, same offsets every run). Reads VLLM_API_KEY from the launcher .env (never printed)."""
import glob, json, math, os, random, sys, urllib.request
URL = "http://127.0.0.1:8888/v1/completions"
def key():
    for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
        if line.startswith("VLLM_API_KEY="): return line.split("=", 1)[1].strip().strip("\"'")
def texts():
    files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
    t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
    out = []
    for i in range(3):
        s = random.Random("qlong-%d" % i).randrange(0, len(t) - 70000)
        out.append(t[s:s + 60000])
    return out
def run(path):
    k = key(); res = []
    for i, tx in enumerate(texts()):
        body = {"model": "GLM-5.3-Flash-EXL3", "prompt": tx, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5}
        r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + k})
        d = json.load(urllib.request.urlopen(r, timeout=1200))
        pl = d["choices"][0].get("prompt_logprobs") or []
        res.append([{t: v["logprob"] for t, v in (p or {}).items()} for p in pl])
        print("text %d: %d positions" % (i, len(pl)), flush=True)
    json.dump(res, open(path, "w"))
def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b)); kls = []; agree = 0; n = 0
    for ta, tb in zip(A, B):
        for pa, pb in list(zip(ta, tb))[4608:]:
            if not pa or not pb: continue
            keys = set(pa) | set(pb); floor = min(min(pa.values()), min(pb.values())) - 1.0
            la = {x: pa.get(x, floor) for x in keys}; lb = {x: pb.get(x, floor) for x in keys}
            za = math.log(sum(math.exp(v) for v in la.values())); zb = math.log(sum(math.exp(v) for v in lb.values()))
            kls.append(sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys))
            agree += max(pa, key=pa.get) == max(pb, key=pb.get); n += 1
    kls.sort()
    print("long-context (positions >= 4608, n=%d): KL mean %.4f p95 %.4f max %.3f | top-1 agree %.2f%%" % (n, sum(kls) / len(kls), kls[int(.95 * len(kls))], kls[-1], 100.0 * agree / n))
if __name__ == "__main__":
    compare(sys.argv[2], sys.argv[3]) if sys.argv[1] == "--compare" else run(sys.argv[1])
