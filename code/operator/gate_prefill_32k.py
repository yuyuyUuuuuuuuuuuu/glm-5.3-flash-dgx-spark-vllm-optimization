#!/usr/bin/env python3
"""32k cold-prefill yardstick: the public kindling GX10 `gate/prefill.py` size sweep, unchanged in method (gibberish
prompt with a per-run seed, one request at a time, temperature 0, prefill = prompt_tokens / TTFT), plus a Bearer key
and the number of other requests running on the server before and after each probe (a row only counts when both say
running=0 waiting=0).

usage: gate_prefill_32k.py <sizes, e.g. 32k or 8k,32k> [seed]
env:   KINDLING_GATE   path to gate/prefill.py of https://github.com/kindlingai/glm-5.3-flash-gx10 (commit 45b438be
                       was used; that repository is not redistributed here)
       GLM_URL         OpenAI-compatible base URL of the server (default http://127.0.0.1:8888/v1)
       GLM_MODEL       served model name (default GLM-5.3-Flash-EXL3)
       VLLM_API_KEY    the server's API key (required; never printed)
The production A/B ran six seeds (71..76) per arm and took the median (tools/arm_env_ab.sh)."""
import importlib.util
import os
import re
import sys
import time
import urllib.request

gate = os.environ.get("KINDLING_GATE")
if not gate or not os.path.isfile(gate):
    sys.exit("set KINDLING_GATE=<checkout of kindlingai/glm-5.3-flash-gx10>/gate/prefill.py")
KEY = os.environ.get("VLLM_API_KEY")
if not KEY:
    sys.exit("set VLLM_API_KEY (the server's key)")
base = os.environ.get("GLM_URL", "http://127.0.0.1:8888/v1").rstrip("/")
model = os.environ.get("GLM_MODEL", "GLM-5.3-Flash-EXL3")
metrics = base[:-3] + "/metrics" if base.endswith("/v1") else base + "/metrics"

spec = importlib.util.spec_from_file_location("g", gate)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
_orig = urllib.request.Request


def Req(url, data=None, headers=None, **kw):
    h = dict(headers or {})
    h["Authorization"] = "Bearer " + KEY
    return _orig(url, data=data, headers=h, **kw)


g.urllib.request.Request = Req


def running():
    t = urllib.request.urlopen(Req(metrics), timeout=10).read().decode()
    m = re.search(r'^vllm:num_requests_running\{[^}]*\} ([0-9.]+)', t, re.M)
    w = re.search(r'^vllm:num_requests_waiting\{[^}]*\} ([0-9.]+)', t, re.M)
    return f"running={m.group(1) if m else '?'} waiting={w.group(1) if w else '?'}"


sizes = [g.parse_size(s) for s in sys.argv[1].split(",")]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else int(time.time())
# calibrate tokens/word the way the upstream script does: one short probe
_, _, u, _, e = g.chat(base, model, g.words(2000, seed), 1)
tpw = (u["prompt_tokens"] / 2000) if u else 2.0
print(f"tokens/word {tpw:.3f}  seed {seed}  {running()}", flush=True)
for i, s in enumerate(sizes):
    print(f"before {s}: {running()}", flush=True)
    g.size_sweep(base, model, [s], seed + 1000 * (i + 1), 32, tpw, None)
    print(f"after {s}: {running()}", flush=True)
