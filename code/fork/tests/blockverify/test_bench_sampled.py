#!/usr/bin/env python3
"""tools/blockverify/bench_sampled.py against a fake vLLM server on 127.0.0.1 (host only, stdlib; no production).

The fake speaks the three endpoints the bench uses: /health, /metrics (vllm:num_requests_running / _waiting gauges and
the vllm:spec_decode_* counters with labels and _created twins, as vLLM exports them) and streaming
/v1/chat/completions. Each request is decoded in speculative steps (5 draft tokens, per-position acceptance
0.70/0.69/0.71/0.62/0.30 from a seed-keyed RNG), updating the counters step by step, so the ground truth of every
request (drafts, accepted) is known.
  T1 busy server (running=1 throughout): exit 1, not a single chat request sent
  T2 idle server: exit 0; per-request accepted/step, draft tokens/step and the pooled summary equal the fake's truth;
     the bearer key reaches the server and never appears in the output
  T3 a second client during one request (gauge running=2 + its own counter increments): that attempt is marked
     contaminated, retried, and the summary still equals the truth of the clean attempts
  T4 counters moved by an invisible client (no gauge change): caught by the counter-consistency check, retried
  T5 --compare of two runs prints the per-prompt deltas
Usage: python3 tests/blockverify/test_bench_sampled.py   (exit 1 on any failure)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "tools/blockverify/bench_sampled.py"
KEY = "test-key-" + "x" * 8
ACC = [0.70, 0.69, 0.71, 0.62, 0.30]
FAIL: list[str] = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


class Fake:
    def __init__(self):
        self.lock = threading.Lock()
        self.c = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": [0] * 7}
        self.running = 0
        self.waiting = 0
        self.always_busy = False
        self.chat_requests = 0
        self.auth_ok = 0
        self.truth = []                     # (seed, drafts, accepted, draft_tokens, tokens)
        self.inject_second_client = set()   # seeds whose first attempt sees another client
        self.inject_invisible = set()       # seeds whose first attempt gets foreign counter increments
        self.seen = {}

    def metrics(self):
        with self.lock:
            lab = 'engine="0",model_name="GLM-5.3-Flash-EXL3"'
            run = self.running + (1 if self.always_busy else 0)
            lines = [f"vllm:num_requests_running{{{lab}}} {run}.0", f"vllm:num_requests_waiting{{{lab}}} {self.waiting}.0",
                     f"vllm:spec_decode_num_drafts_total{{{lab}}} {self.c['drafts']}.0",
                     f"vllm:spec_decode_num_drafts_created{{{lab}}} 1.7e9",
                     f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {self.c['draft_tokens']}.0",
                     f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {self.c['accepted']}.0"]
            lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{lab},position="{i}"}} {v}.0'
                      for i, v in enumerate(self.c["pos"])]
            return "# HELP x\n" + "\n".join(lines) + "\n"

    def step(self, rng, k=5):
        a = 0
        while a < k and rng.random() < ACC[a]:
            a += 1
        with self.lock:
            self.c["drafts"] += 1
            self.c["draft_tokens"] += k
            self.c["accepted"] += a
            for i in range(a):
                self.c["pos"][i] += 1
        return a


FAKE = Fake()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path == "/metrics":
            body = FAKE.metrics().encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n))
        with FAKE.lock:
            FAKE.chat_requests += 1
            FAKE.auth_ok += int(self.headers.get("Authorization") == f"Bearer {KEY}")
            FAKE.running += 1
        seed, max_tokens = int(body["seed"]), int(body["max_tokens"])
        attempt = FAKE.seen.get(seed, 0)
        FAKE.seen[seed] = attempt + 1
        rng = random.Random(seed)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        tokens = 1
        emit({"choices": [{"delta": {"content": "t"}, "finish_reason": None}]})
        drafts = acc = dtok = 0
        steps = 0
        while tokens < max_tokens:
            a = FAKE.step(rng)
            drafts += 1; acc += a; dtok += 5; steps += 1
            tokens = min(max_tokens, tokens + a + 1)
            emit({"choices": [{"delta": {"content": "t" * (a + 1)}, "finish_reason": None}]})
            time.sleep(0.002)
            if steps == 3 and attempt == 0 and seed in FAKE.inject_second_client:
                with FAKE.lock:
                    FAKE.running += 1
                FAKE.step(random.Random(1))
                time.sleep(1.2)                      # visible to the 0.5 s poller
                with FAKE.lock:
                    FAKE.running -= 1
            if steps == 3 and attempt == 0 and seed in FAKE.inject_invisible:
                for _ in range(20):
                    FAKE.step(random.Random(2))
        emit({"choices": [{"delta": {}, "finish_reason": "length"}]})
        emit({"choices": [], "usage": {"completion_tokens": tokens, "prompt_tokens": 20}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        clean = not (attempt == 0 and (seed in FAKE.inject_second_client or seed in FAKE.inject_invisible))
        if clean:
            FAKE.truth.append((body["messages"][0]["content"][:10], seed, drafts, acc, dtok, tokens))
        with FAKE.lock:
            FAKE.running -= 1


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
tmp = Path(tempfile.mkdtemp(prefix="bs_"))
FAST = ["--idle-polls", "2", "--idle-interval", "0.05", "--max-wait", "3", "--poll", "0.2", "--warmup-tokens", "4"]


def bench(label, out, *extra):
    env = dict(os.environ, API_KEY=KEY)
    p = subprocess.run([sys.executable, str(BENCH), "--port", str(port), "--label", label, "--out", str(out), *FAST, *extra],
                       env=env, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


# T1
FAKE.always_busy = True
rc, txt = bench("busy", tmp / "busy.json", "--seeds", "1", "--prompts", "prose")
check(rc == 1 and FAKE.chat_requests == 0 and not (tmp / "busy.json").exists(),
      f"T1 busy server: exit {rc}, {FAKE.chat_requests} chat requests sent, no output")
FAKE.always_busy = False

# T2
FAKE.truth.clear()
rc, txt = bench("standard", tmp / "a.json", "--seeds", "3", "--prompts", "prose,coding", "--max-tokens", "64")
rec = json.loads((tmp / "a.json").read_text()) if (tmp / "a.json").exists() else {}
runs = [r for r in rec.get("runs", []) if not r.get("contaminated")]
truth = {(t[1], t[0]): t for t in FAKE.truth}
ok = len(runs) == 6 and rc == 0
for r in runs:
    t = [x for (s, _p), x in truth.items() if s == r["seed"] and x[5] == r["completion_tokens"]]
    ok &= bool(t) and abs(r["accepted_tokens_per_step"] - (1 + t[0][3] / t[0][2])) < 1e-9 and \
        abs(r["draft_tokens_per_step"] - t[0][4] / t[0][2]) < 1e-9
check(ok, f"T2 idle server: exit {rc}, {len(runs)} clean requests, per-request accepted/step and draft tokens/step == the fake's truth")
allt = [t for t in FAKE.truth if t[5] == 64]
pooled = 1 + sum(t[3] for t in allt[-6:]) / sum(t[2] for t in allt[-6:])
check(abs(rec["summary"]["all"]["accepted_tokens_per_step"] - pooled) < 1e-9 and rec["summary"]["all"]["ms_per_step_mean_se"][0] > 0
      and rec["summary"]["all"]["tok_s_mean_se"][0] > 0, f"T2 pooled summary accepted/step {rec['summary']['all']['accepted_tokens_per_step']:.4f} "
      f"== truth {pooled:.4f}; tok/s and ms/step reported")
check(FAKE.auth_ok == FAKE.chat_requests and KEY not in txt and KEY not in (tmp / "a.json").read_text(),
      "T2 bearer key sent on every chat request and never printed or saved")

# T3 + T4
FAKE.truth.clear()
FAKE.inject_second_client = {20260929 + 1}
FAKE.inject_invisible = {20260929 + 2}
FAKE.seen.clear()
rc, txt = bench("block", tmp / "b.json", "--seeds", "3", "--prompts", "prose", "--max-tokens", "64")
rec = json.loads((tmp / "b.json").read_text())
bad = [r for r in rec["runs"] if r.get("contaminated")]
why = " / ".join(r["why"] for r in bad)
check(rc == 0 and len(bad) == 2 and any("other requests seen" in r["why"] for r in bad)
      and any("counter delta not explained" in r["why"] for r in bad),
      f"T3/T4 a second client and invisible counter moves are both caught ({why})")
clean = [r for r in rec["runs"] if not r.get("contaminated")]
check(len(clean) == 3 and sorted(r["seed"] for r in clean) == [20260929, 20260930, 20260931],
      "T3/T4 the contaminated attempts are retried and every seed ends with one clean request")
tt = [t for t in FAKE.truth if t[5] == 64]          # not the 4-token warmup
pooled = 1 + sum(t[3] for t in tt) / sum(t[2] for t in tt)
check(len(tt) == 3 and abs(rec["summary"]["all"]["accepted_tokens_per_step"] - pooled) < 1e-9,
      "T3/T4 summary == truth of the clean attempts only")

# T5
p = subprocess.run([sys.executable, str(BENCH), "--compare", str(tmp / "a.json"), str(tmp / "b.json")], capture_output=True, text=True)
check(p.returncode == 0 and "prose" in p.stdout and "accepted/step" in p.stdout and "%" in p.stdout,
      "T5 --compare prints per-prompt deltas:\n" + "\n".join("     " + l for l in p.stdout.splitlines()))
srv.shutdown()
print("ALL PASSED" if not FAIL else f"FAILED: {len(FAIL)}")
sys.exit(1 if FAIL else 0)
