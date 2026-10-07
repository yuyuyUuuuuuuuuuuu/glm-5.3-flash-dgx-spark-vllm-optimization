#!/usr/bin/env python3
"""Offline test of tools/prodcheck/kpool_decode_consistency.py against a local mock of the image's OpenAI server
(response shapes from vllm/entrypoints/openai/completion/protocol.py at 487ecf187: choices[0].prompt_token_ids /
token_ids with return_token_ids, logprobs.top_logprobs keyed 'token_id:<id>' with return_tokens_as_token_ids,
prompt_logprobs as [None, {"<id>": {"logprob", "rank", "decoded_token"}}, ...]; /metrics in Prometheus text).
Checks: request fields, busy refusal, position alignment (decode j <-> prefill prompt_len + j), KL/top-1 math on a
known perturbation, --compare. Host python, no GPU, no network beyond 127.0.0.1.
"""
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tools/prodcheck/kpool_decode_consistency.py"
V = 50
STATE = {"running": 0, "waiting": 0, "shift": 0.0, "reqs": []}
fails = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def dist(ctx_len, tok, shift):
    """Deterministic 'model': logits depend on (position, previous token); `shift` perturbs the decode side."""
    rng = random.Random(ctx_len * 7919 + tok)
    lg = [rng.gauss(0, 2) for _ in range(V)]
    lg[(ctx_len * 31 + tok) % V] += 3.0 + shift
    z = math.log(sum(math.exp(x) for x in lg))
    return [x - z for x in lg]


def top5(lp):
    return sorted(range(V), key=lambda i: -lp[i])[:5]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        assert self.path == "/metrics"
        body = (f'vllm:num_requests_running{{engine="0",model_name="m"}} {STATE["running"]}.0\n'
                f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {STATE["waiting"]}.0\n').encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["reqs"].append((req, self.headers.get("Authorization")))
        if isinstance(req["prompt"], str):  # A: decode
            p = [ord(c) % V for c in req["prompt"][:300]]
            g, tops = [], []
            for j in range(req["max_tokens"]):
                ctx = p + g
                lp = dist(len(ctx), ctx[-1], STATE["shift"] if j >= 10 else 0.0)
                tops.append({f"token_id:{i}": lp[i] for i in top5(lp)})
                g.append(top5(lp)[0])
            ch = {"index": 0, "text": "x", "prompt_token_ids": p, "token_ids": g,
                  "logprobs": {"tokens": [f"token_id:{t}" for t in g], "token_logprobs": [0.0] * len(g),
                               "top_logprobs": tops, "text_offset": [0] * len(g)}}
        else:  # B: prefill
            ids = req["prompt"]
            pl = [None]
            for i in range(1, len(ids)):
                lp = dist(i, ids[i - 1], 0.0)
                pl.append({str(t): {"logprob": lp[t], "rank": r + 1, "decoded_token": "x"}
                           for r, t in enumerate(top5(lp))})
            ch = {"index": 0, "text": "y", "prompt_logprobs": pl, "prompt_token_ids": ids}
        body = json.dumps({"choices": [ch], "model": "m", "usage": {}}).encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
tmp = Path(tempfile.mkdtemp(prefix="kpool-probe-"))
(tmp / ".env").write_text("OTHER=1\nVLLM_API_KEY='mock-secret'\n")
env = dict(os.environ, KPOOL_PROBE_URL=f"http://127.0.0.1:{srv.server_port}", KPOOL_PROBE_ENV=str(tmp / ".env"))


def probe(*args):
    return subprocess.run([sys.executable, str(PROBE), *args], env=env, capture_output=True, text=True)


STATE["running"] = 1
r = probe(str(tmp / "busy.json"), "--gen", "40")
check(r.returncode != 0 and "server busy" in (r.stdout + r.stderr) and not STATE["reqs"], "busy server refused, no request sent")
STATE["running"] = 0

r = probe(str(tmp / "same.json"), "--gen", "40")
check(r.returncode == 0, f"run (no perturbation) rc={r.returncode} {r.stderr.strip()[-200:]}")
check("mock-secret" not in r.stdout + r.stderr, "API key never printed")
a_reqs = [q for q, _ in STATE["reqs"] if isinstance(q["prompt"], str)]
b_reqs = [q for q, _ in STATE["reqs"] if not isinstance(q["prompt"], str)]
check(len(a_reqs) == 3 and len(b_reqs) == 3, f"3 decode + 3 prefill requests ({len(a_reqs)}, {len(b_reqs)})")
check(all(q["temperature"] == 0 and q["ignore_eos"] and q["logprobs"] == 5 and q["return_tokens_as_token_ids"]
          and q["return_token_ids"] and q["max_tokens"] == 40 for q in a_reqs), "decode request fields")
check(all(q["max_tokens"] == 1 and q["prompt_logprobs"] == 5 and isinstance(q["prompt"][0], int) for q in b_reqs),
      "prefill request fields (token-id prompt)")
salts = [q["cache_salt"] for q in a_reqs + b_reqs]
check(len(set(salts)) == 6, "every request has its own cache_salt")
check(all(h == "Bearer mock-secret" for _, h in STATE["reqs"]), "auth header from .env")
check("KL mean 0.00000" in r.stdout and "top-1 agree 100.00%" in r.stdout, "identical model -> KL 0, top-1 100%")

STATE["shift"] = -2.5
r2 = probe(str(tmp / "shift.json"), "--gen", "40")
lines = [l for l in r2.stdout.splitlines() if "gen pos" in l]
check(r2.returncode == 0 and "KL mean 0.00000" not in r2.stdout.split("all:")[1], "perturbed decode side -> KL > 0")
# positions < 10 are unperturbed in the mock; they fall in bucket [0, 128) together with 10..39, so check alignment
# directly: recompute from the saved json that position j of decode matches prefill exactly for j < 10.
d = json.load(open(tmp / "shift.json"))
exact = all(d["texts"][0]["decode"][j] == d["texts"][0]["prefill"][j] for j in range(10))
check(exact, "decode position j aligned with prefill prompt_len + j (unperturbed prefix identical)")
r3 = probe("--compare", str(tmp / "shift.json"), str(tmp / "same.json"))
check(r3.returncode == 0 and "-> after 0.00000 (-100.0%)" in r3.stdout, "--compare prints before -> after")
print(r3.stdout.strip().splitlines()[-1])
print("== " + ("ALL OK" if not fails else f"{len(fails)} FAILED"))
sys.exit(1 if fails else 0)
