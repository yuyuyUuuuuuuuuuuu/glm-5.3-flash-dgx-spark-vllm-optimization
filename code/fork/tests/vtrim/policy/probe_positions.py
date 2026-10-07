#!/usr/bin/env python3
"""Uncensored DFlash2 acceptance length L(p) at every position p of a temp-0 reference output.

For each workload: render the chat prompt with /tokenize (thinking off), generate a 200-token temp-0
reference with /v1/completions (prompt ids, return_token_ids), then for every p send
prompt_ids + ref[:p] with max_tokens=10. The prefill emits ref[p] (checked) and the first verify step
runs at full length (adaptive-K keeps a request at K=7 for its first MIN_STEPS steps) and emits 1+L
tokens, L = accepted drafts of a fresh block anchored at ref[p] (drafts cover ref[p+1..p+7]).
A policy simulator can then replay any per-step K along the reference exactly: a = min(L(p), K).

Read-only, concurrency 1, waits while other requests run. API key only from VLLM_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from trace_stream import BASE, MODEL, PROMPTS, headers, metrics  # noqa: E402


def post(path, body, stream=False, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers(), method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize_chat(prompt: str) -> list[int]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    with post("/tokenize", body) as r:
        return json.load(r)["tokens"]


def completion_stream(prompt_ids: list[int], max_tokens: int) -> dict:
    body = {"model": MODEL, "prompt": prompt_ids, "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "return_token_ids": True}
    chunks = []
    t0 = time.perf_counter()
    with post("/v1/completions", body, timeout=300) as resp:
        buf = b""
        while True:
            piece = resp.read1(4096)
            if not piece:
                break
            tn = time.perf_counter()
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
                for ch in obj.get("choices") or []:
                    ids = ch.get("token_ids") or []
                    if ids:
                        chunks.append([round(tn - t0, 6), list(ids)])
    return {"chunks": chunks}


def wait_idle(max_wait=600):
    t0 = time.time()
    while True:
        m = metrics()
        if m.get("running", 0) < 1:
            return True
        if time.time() - t0 > max_wait:
            return False
        time.sleep(2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="structured,prose,coding,ja")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prompts-json", default="", help="file with {name: prompt} added to PROMPTS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.prompts_json:
        PROMPTS.update(json.load(open(a.prompts_json)))
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
    for w in a.workloads.split(","):
        wait_idle()
        pids = tokenize_chat(PROMPTS[w])
        body = {"model": MODEL, "prompt": pids, "temperature": 0, "top_p": 1, "max_tokens": a.n,
                "return_token_ids": True}
        with post("/v1/completions", body) as r:
            ref = json.load(r)["choices"][0]
        ref_ids = ref["token_ids"]
        rec = {"prompt_ids": pids, "ref_ids": ref_ids, "ref_text": ref["text"], "probes": {}}
        print(f"[probe] {w}: prompt {len(pids)} tok, ref {len(ref_ids)} tok", flush=True)
        for p in range(0, len(ref_ids) - 1, a.stride):
            if not wait_idle():
                print("[probe] busy too long, stop", flush=True)
                break
            r = completion_stream(pids + ref_ids[:p], 10)
            ch = r["chunks"]
            first = ch[0][1] if ch else []
            verify = ch[1][1] if len(ch) > 1 else []
            # L = accepted drafts in the first verify step; match flags vs the reference
            emitted = first + verify
            L = len(verify) - 1 if verify else None
            match = sum(1 for i, t in enumerate(emitted) if p + i < len(ref_ids) and ref_ids[p + i] == t
                        and all(ref_ids[p + j] == emitted[j] for j in range(i)))
            rec["probes"][str(p)] = {"first": first, "verify": verify, "L": L, "ref_match": match,
                                     "dt": [c[0] for c in ch[:3]]}
            if p % 25 == 0:
                print(f"[probe] {w} p={p} L={L} match={match}/{len(emitted)}", flush=True)
        res[w] = rec
        with open(a.out, "w") as fh:
            json.dump(res, fh, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
