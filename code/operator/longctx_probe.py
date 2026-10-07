#!/usr/bin/env python3
"""Decode speed vs context on real text. Same prompt twice: run 1 pays the prefill, run 2 hits the prefix cache;
decode ms/step from the engine's spec-decode draft counter over the streamed decode window.
Usage: longctx_probe.py <label> <target_tokens> [max_tokens=400] [prof=head_pid,worker_pid]"""
import glob, json, os, re, subprocess, sys, time, urllib.request
L, n = sys.argv[1], int(sys.argv[2]); mt = int(sys.argv[3]) if len(sys.argv) > 3 else 400
prof = sys.argv[4].split(",") if len(sys.argv) > 4 else None
key = ""
for line in open(os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env")):
    if line.startswith("VLLM_API_KEY="):
        key = line.split("=", 1)[1].strip().strip("\"'")
files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
text = "".join(f"\n\n### {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
while len(text) < n * 5:
    text += text
body_text = text[: int(n * 4.4)]
msg = f"[longctx-{n}] Read the following files. Then write a long, detailed essay in English about the history and design ideas of software licenses and of the Python standard library, citing the files.\n{body_text}"
M = "http://localhost:8888/metrics"
def metrics():
    s = urllib.request.urlopen(M, timeout=10).read().decode()
    g = lambda name: sum(float(m.group(1)) for m in re.finditer(r"^" + name + r"\{[^}]*\} ([0-9.e+]+)$", s, re.M))
    return {"drafts": g("vllm:spec_decode_num_drafts_total"), "acc": g("vllm:spec_decode_num_accepted_tokens_total"),
            "run": g("vllm:num_requests_running"), "wait": g("vllm:num_requests_waiting")}
def run(tag, arm=False):
    m0 = metrics()
    if m0["run"] or m0["wait"]:
        print(f"[{L}] busy (running={m0['run']} waiting={m0['wait']}), abort"); sys.exit(3)
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": msg}], "max_tokens": mt,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://localhost:8888/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time(); t1 = None; usage = None; md = None; armed = False
    with urllib.request.urlopen(req, timeout=1800) as fh:
        for raw in fh:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content") or ch[0].get("delta", {}).get("reasoning")):
                if t1 is None:
                    t1 = time.time(); md = metrics()
                elif arm and not armed and time.time() - t1 > 5 and prof:
                    subprocess.Popen(["docker", "exec", "glm53-exl3-head", "kill", "-USR2", prof[0]])
                    subprocess.Popen(["ssh", "-o", "BatchMode=yes", os.environ["WORKER_SSH"], f"docker exec glm53-exl3-worker kill -USR2 {prof[1]}"])
                    armed = True
    t2 = time.time(); me = metrics()
    steps = me["drafts"] - md["drafts"]; acc = me["acc"] - md["acc"]
    print(f"[{L}] {tag}: prompt {usage['prompt_tokens']} cached {((usage.get('prompt_tokens_details') or {}).get('cached_tokens'))} "
          f"completion {usage['completion_tokens']} ttft {t1 - t0:.2f}s decode {t2 - t1:.2f}s steps {steps:.0f} "
          f"-> {1000 * (t2 - t1) / max(steps, 1):.1f} ms/step, {1 + acc / max(steps, 1):.2f} tok/step, "
          f"{usage['completion_tokens'] / (t2 - t1):.1f} tok/s", flush=True)
if os.environ.get("ONCE"):
    run("once", arm=True)
else:
    run("cold")
    run("warm", arm=True)
