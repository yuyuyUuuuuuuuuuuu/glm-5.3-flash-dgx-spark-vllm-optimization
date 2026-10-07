#!/usr/bin/env python3
"""Prefill throughput probe: unique random prompts (no prefix-cache hits), max_tokens=1, non-streaming.
Usage: prefill_probe.py <label> [sizes_words=6000,24000] [runs=3]   (reads VLLM_API_KEY from the launcher .env; never prints it)"""
import json, os, random, sys, time, urllib.request
L = sys.argv[1]; sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "6000,24000").split(",")]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
key = ""
for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
    if line.startswith("VLLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip("\"'")
words = ("time person year way day thing man world life hand part child eye woman place work week case point government "
         "company number group problem fact river mountain signal kernel memory bandwidth tensor matrix vector cache "
         "ocean forest garden window market doctor engine planet silver copper orange purple quiet rapid gentle").split()
out = {}
for n in sizes:
    res = []
    for r in range(runs):
        rng = random.Random(f"{L}-{n}-{r}-{time.time()}")
        text = " ".join(rng.choice(words) for _ in range(n))
        body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": f"[{rng.random()}] Summarize: {text}"}],
                "max_tokens": 1, "temperature": 0}
        req = urllib.request.Request("http://localhost:8888/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=900) as f:
            d = json.load(f)
        dt = time.time() - t
        pt = d["usage"]["prompt_tokens"]
        res.append((pt, dt))
    tps = sorted(pt / dt for pt, dt in res)
    out[n] = res
    print(f"[{L}] prefill ~{res[0][0]} tok: {', '.join(f'{dt:.2f}s' for _, dt in res)} -> median {tps[len(tps)//2]:.0f} tok/s", flush=True)
