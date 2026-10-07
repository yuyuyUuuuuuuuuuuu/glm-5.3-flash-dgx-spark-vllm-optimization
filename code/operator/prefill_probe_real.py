#!/usr/bin/env python3
"""Prefill probe on REAL text (English license prose + Python stdlib source; random-word prompts collapse the MoE routing).
Unique salt first (no prefix-cache hits), different corpus offset per run, max_tokens=1.
Usage: prefill_probe_real.py <label> [target_tokens=6000,16000] [runs=3]"""
import glob, json, os, random, sys, time, urllib.request
L = sys.argv[1]; sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "6000,16000").split(",")]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
key = ""
for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
    if line.startswith("VLLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip("\"'")
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
corpus = []
for f in files:
    try:
        corpus.append(f"\n\n### {os.path.basename(f)}\n" + open(f, errors="ignore").read())
    except OSError:
        pass
text = "".join(corpus)
for n in sizes:
    res = []
    for r in range(runs):
        salt = f"{L}-{n}-{r}-{time.time()}-{random.random()}"
        start = random.Random(f"fixed-{n}-{r}").randrange(0, len(text) - n * 6)   # same texts across configs
        body_text = text[start:start + int(n * 4.4)]
        msg = f"[{salt}] Read the following files and list the main topics.\n{body_text}"
        body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": msg}], "max_tokens": 1, "temperature": 0}
        req = urllib.request.Request("http://localhost:8888/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=900) as fh:
            d = json.load(fh)
        res.append((d["usage"]["prompt_tokens"], time.time() - t))
    tps = sorted(p / dt for p, dt in res)
    print(f"[{L}] real-text prefill ~{res[0][0]} tok: {', '.join(f'{dt:.2f}s' for _, dt in res)} -> median {tps[len(tps)//2]:.0f} tok/s", flush=True)
