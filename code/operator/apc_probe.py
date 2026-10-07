#!/usr/bin/env python3
"""Prefix-cache probe: turn 1 (prompt P), exact repeat of P, and a chat-style turn 2 (P + answer + new question).
Reports cached_tokens, TTFT and the engine's prefix_cache_hits/queries counter deltas. Aborts when the engine is busy."""
import glob, json, os, re, sys, time, urllib.request
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
key = ""
for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
    if line.startswith("VLLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip("\"'")
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
text = "".join(f"\n\n### {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
body_text = text[200000: 200000 + int(n * 4.4)]
salt = f"apc-{n}-{time.time()}"
def metrics():
    s = urllib.request.urlopen("http://localhost:8888/metrics", timeout=10).read().decode()
    g = lambda name: sum(float(m.group(1)) for m in re.finditer(r"^" + name + r"\{[^}]*\} ([0-9.e+]+)$", s, re.M))
    return {k: g("vllm:" + k) for k in ("prefix_cache_hits_total", "prefix_cache_queries_total", "num_requests_running", "num_requests_waiting")}
def ask(tag, msgs, mt=64):
    m0 = metrics()
    if m0["num_requests_running"] or m0["num_requests_waiting"]:
        print("busy, abort"); sys.exit(3)
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": msgs, "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request("http://localhost:8888/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t; m1 = metrics(); u = d["usage"]
    print(f"{tag}: prompt {u['prompt_tokens']} cached {(u.get('prompt_tokens_details') or {}).get('cached_tokens')} "
          f"e2e {dt:.1f}s hits+{m1['prefix_cache_hits_total'] - m0['prefix_cache_hits_total']:.0f} "
          f"queries+{m1['prefix_cache_queries_total'] - m0['prefix_cache_queries_total']:.0f}", flush=True)
    return d["choices"][0]["message"].get("content") or ""
m1 = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"[{salt}] Read these files.\n{body_text}\nWhich licenses are mentioned? Answer briefly."}]
a = ask("turn1", m1)
ask("repeat", m1)
ask("turn2", m1 + [{"role": "assistant", "content": a}, {"role": "user", "content": "Now name one Python module from the files."}])
