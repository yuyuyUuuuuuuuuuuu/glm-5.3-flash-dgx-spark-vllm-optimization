#!/usr/bin/env python3
"""Stream a decode and record chunk arrival gaps (stalls). Usage: stall_probe.py <label> <target_tokens> [max_tokens=1500]"""
import glob, json, os, re, sys, time, urllib.request
L, n = sys.argv[1], int(sys.argv[2]); mt = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
key = ""
for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
    if line.startswith("VLLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip("\"'")
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
text = "".join(f"\n\n### {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
while len(text) < n * 5: text += text
msg = f"[stall-{L}-{time.time()}] Read the following files. Then write a long, detailed essay in English about the history and design ideas of software licenses and of the Python standard library, citing the files.\n" + text[: int(n * 4.4)]
s = urllib.request.urlopen("http://localhost:8888/metrics", timeout=10).read().decode()
if re.search(r'^vllm:num_requests_(running|waiting)\{[^}]*\} [1-9]', s, re.M): print("busy, abort"); sys.exit(3)
body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": msg}], "max_tokens": mt, "temperature": 0, "stream": True}
req = urllib.request.Request("http://localhost:8888/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
ts = []
with urllib.request.urlopen(req, timeout=1800) as fh:
    for raw in fh:
        line = raw.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            d = json.loads(line[5:]); ch = d.get("choices") or []
            if ch and any(ch[0].get("delta", {}).get(k) for k in ("content", "reasoning_content", "reasoning")): ts.append(time.time())
g = sorted((b - a) * 1000 for a, b in zip(ts, ts[1:]))
tot = (ts[-1] - ts[0]) * 1000
big = [x for x in g if x > 200]
print(f"[{L}] chunks {len(ts)} decode {tot/1000:.1f}s | gap p50 {g[len(g)//2]:.0f} p90 {g[9*len(g)//10]:.0f} p99 {g[99*len(g)//100]:.0f} max {g[-1]:.0f} ms | gaps>200ms: {len(big)} sum {sum(big)/1000:.1f}s", flush=True)
# where do big gaps sit (chunk index)?
gi = [(i, (b - a) * 1000) for i, (a, b) in enumerate(zip(ts, ts[1:])) if (b - a) * 1000 > 200]
print(f"[{L}] big gaps at chunk idx: {[(i, round(x)) for i, x in gi][:20]}")
