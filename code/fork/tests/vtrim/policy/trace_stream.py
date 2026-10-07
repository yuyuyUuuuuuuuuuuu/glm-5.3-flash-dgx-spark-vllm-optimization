#!/usr/bin/env python3
"""Per-step decode trace from the live server (read-only, concurrency 1).

Each SSE chunk of a speculative-decoding stream carries the tokens one engine step emitted
(1 + accepted drafts); with continuous_usage_stats the chunk also carries the cumulative
completion_tokens, so the per-chunk token delta is the per-step emitted count and the
inter-chunk gap is the step time (serving cycle). Optionally pins the request to the full
draft length (K=7) through a no-op structured-output regex (production's adaptive-K keeps
structured-output requests at full length), which yields the *uncensored* accepted length
L_t = number of the 7 drafts accepted.

Same prompts / sampling as tests/bench_decode.py (temp 0, top_p 1, thinking off, 200 tokens).
Never prints the API key (read from VLLM_API_KEY env only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8888")
MODEL = "GLM-5.3-Flash-EXL3"
PROMPTS = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough.",
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "coding": "Write a Python function named clamp_range that takes a list of ints and "
    "returns a new list with each value clamped to [0, 50]. Include a short docstring.",
    "ja": "ハッシュマップの仕組みを、衝突の扱い、リサイズ、計算量を含めて、日本語で順を追って詳しく説明してください。",
}
SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-zA-Z0-9_]+)\{([^}]*)\}\s+(\S+)$")


def headers():
    key = os.environ.get("VLLM_API_KEY") or os.environ.get("API_KEY")
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def metrics() -> dict:
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:num_requests_running{"):
                out["running"] = float(line.split()[-1])
            m = SPEC_RE.match(line)
            if not m:
                continue
            name, labels, val = m.groups()
            pos = re.search(r'position="(\d+)"', labels)
            key = name + (f":{pos.group(1)}" if pos else "")
            out[key] = out.get(key, 0.0) + float(val)
    return out


def one(prompt: str, max_tokens: int, pin7: bool, regex: str, extra: dict | None = None) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if pin7:
        body["structured_outputs"] = {"regex": regex}
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers(), method="POST")
    t0 = time.perf_counter()
    chunks = []
    last_ct = 0
    text = []
    finish = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        buf = b""
        while True:
            piece = resp.read1(4096)
            if not piece:
                break
            tnow = time.perf_counter()
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or {}
                ct = int(usage.get("completion_tokens") or last_ct)
                content = ""
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    content += d.get("content") or d.get("reasoning") or d.get("reasoning_content") or ""
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if ct != last_ct or content:
                    chunks.append([round(tnow - t0, 6), ct - last_ct, content])
                    text.append(content)
                    last_ct = ct
    return {"chunks": chunks, "text": "".join(text), "completion_tokens": last_ct,
            "finish": finish, "wall_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="structured,prose,coding,ja")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--pin7", action="store_true")
    ap.add_argument("--regex", default="[\\s\\S]*")
    ap.add_argument("--max-running", type=int, default=0,
                    help="skip a run when more than this many other requests are running")
    ap.add_argument("--extra", default="", help="JSON merged into the request body")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = json.loads(a.extra) if a.extra else None
    res = {"extra": extra, "pin7": a.pin7, "regex": a.regex if a.pin7 else None, "runs": []}
    for rep in range(a.runs):
        for w in a.workloads.split(","):
            m0 = metrics()
            if m0.get("running", 0) > a.max_running:
                print(f"[trace] skip {w} rep{rep}: running={m0.get('running')}", flush=True)
                res["runs"].append({"workload": w, "rep": rep, "skipped": m0.get("running")})
                continue
            r = one(PROMPTS[w], a.max_tokens, a.pin7, a.regex, extra)
            m1 = metrics()
            sd = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1 if k.startswith("vllm:spec")}
            r.update({"workload": w, "rep": rep, "spec_delta": sd,
                      "running_before": m0.get("running"), "running_after": m1.get("running")})
            res["runs"].append(r)
            steps = [c[1] for c in r["chunks"]]
            print(f"[trace] {w} rep{rep} pin7={a.pin7} chunks={len(steps)} tokens={r['completion_tokens']} "
                  f"drafts_metric={sd.get('vllm:spec_decode_num_drafts_total')} "
                  f"acc_metric={sd.get('vllm:spec_decode_num_accepted_tokens_total')}", flush=True)
            with open(a.out, "w") as fh:
                json.dump(res, fh, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
