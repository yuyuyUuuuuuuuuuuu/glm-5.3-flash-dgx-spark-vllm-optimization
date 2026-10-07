#!/usr/bin/env python3
"""A stdlib mock of the vLLM OpenAI completions endpoint, for testing tools/klh/klh.py without production.

It serves a toy LM whose next-token distribution is a deterministic function of the last 4 context tokens, with
the knobs a production A/B has:
  sys_arm    an arm's systematic perturbation of the PREFILL path (seeded by arm id + context; like e4m3/W8A8)
  sys_dec    the decode path's structural difference to prefill (seeded by context; same for every arm)
  noise_pf   per-request random noise of the prefill path (production's run-to-run non-determinism)
  noise_dec  per-request random noise of the decode path
The API surface mimics what klh/quality_long/kpool_decode_consistency use:
  POST /v1/completions {prompt: [ids], max_tokens, temperature 0, logprobs, prompt_logprobs, return_token_ids,
       return_tokens_as_token_ids, ignore_eos, cache_salt, ...}
       -> choices[0] {text, token_ids, prompt_token_ids, prompt_logprobs: [None, {"id": {logprob, rank,
          decoded_token}}...], logprobs: {tokens, token_logprobs, top_logprobs: [{"token_id:ID": lp}], text_offset}}
       the first generated token comes from the prefill path (last prompt position), the rest from the decode path.
  GET /metrics (vllm:num_requests_running / _waiting / request_success_total / prompt_tokens_total /
       prompt_tokens_cached_total, with label sets + # lines like vLLM's), GET /health.
Test hooks (POST /mock/config with JSON): phantom_at=[n,...] (during the n-th completion a phantom request is
"running" and finishes with it), busy=true (always one running request), cached_at=[n,...] (report prefix-cache hits
during the n-th completion), arm, sys_arm, sys_dec, noise_pf, noise_dec, delay. GET /mock/log returns every
completion request seen (prompt length, max_tokens, generated ids, nonce).
Exact distributions for a context are available in-process via Model.dist(ctx, path, nonce) (tests import this module).
"""
import argparse
import hashlib
import json
import math
import random
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _seed(*parts):
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "little")


class Model:
    def __init__(self, V=256, seed=7, peak=7.0, arm=0, sys_arm=0.0, sys_dec=0.0, noise_pf=0.0, noise_dec=0.0):
        self.V, self.seed, self.peak = V, seed, peak
        self.arm, self.sys_arm, self.sys_dec, self.noise_pf, self.noise_dec = arm, sys_arm, sys_dec, noise_pf, noise_dec

    def logits(self, ctx, path, nonce):
        tail = tuple(ctx[-4:])
        pos = len(ctx)
        rng = random.Random(_seed(self.seed, tail))
        lg = [rng.gauss(0.0, 1.5) for _ in range(self.V)]
        a = (tail[-1] if tail else 0)
        lg[(a * 31 + 7) % self.V] += self.peak
        # a frequent near tie (makes greedy paths sensitive to small perturbations, like real text)
        lg[(a * 17 + 3) % self.V] += self.peak - (0.3 if a % 5 == 0 else 2.5)
        if self.sys_arm and path == "prefill":
            r = random.Random(_seed("arm", self.arm, tail))
            lg = [x + self.sys_arm * r.gauss(0.0, 1.0) for x in lg]
        if self.sys_dec and path == "decode":
            r = random.Random(_seed("dec", tail))
            lg = [x + self.sys_dec * r.gauss(0.0, 1.0) for x in lg]
        nz = self.noise_pf if path == "prefill" else self.noise_dec
        if nz and nonce is not None:
            r = random.Random(_seed("noise", nonce, pos, path))
            lg = [x + nz * r.gauss(0.0, 1.0) for x in lg]
        return lg

    def dist(self, ctx, path, nonce=None):
        lg = self.logits(ctx, path, nonce)
        m = max(lg)
        z = m + math.log(sum(math.exp(x - m) for x in lg))
        return [x - z for x in lg]


def topk(lp, k):
    order = sorted(range(len(lp)), key=lambda i: (-lp[i], i))
    return order[:k]


class State:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.running = 0
        self.success = 0
        self.prompt_tokens = 0
        self.cached = 0
        self.ncomp = 0
        self.phantom_at = set()
        self.cached_at = set()
        self.busy = False
        self.delay = 0.0
        self.log = []


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, ctype="application/json"):
            data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, "", "text/plain")
            if self.path == "/mock/log":
                with st.lock:
                    return self._send(200, st.log)
            if self.path == "/metrics":
                with st.lock:
                    run = st.running + (1 if st.busy else 0)
                    lines = [
                        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
                        "# TYPE vllm:num_requests_running gauge",
                        'vllm:num_requests_running{engine="0",model_name="m"} %.1f' % run,
                        'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0',
                        'vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 0.0',
                        "# TYPE vllm:request_success_total counter",
                        'vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} %.1f' % st.success,
                        'vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 0.0',
                        'vllm:request_success_created{engine="0",finished_reason="length",model_name="m"} 1.7e9',
                        'vllm:prompt_tokens_total{engine="0",model_name="m"} %.1f' % st.prompt_tokens,
                        'vllm:prompt_tokens_cached_total{engine="0",model_name="m"} %.1f' % st.cached,
                    ]
                return self._send(200, "\n".join(lines) + "\n", "text/plain")
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/mock/config":
                with st.lock:
                    m = st.model
                    for k in ("arm", "sys_arm", "sys_dec", "noise_pf", "noise_dec"):
                        if k in body:
                            setattr(m, k, body[k])
                    if "phantom_at" in body:
                        st.phantom_at = set(body["phantom_at"])
                    if "cached_at" in body:
                        st.cached_at = set(body["cached_at"])
                    if "busy" in body:
                        st.busy = bool(body["busy"])
                    if "delay" in body:
                        st.delay = float(body["delay"])
                    if body.get("reset_log"):
                        st.log = []
                        st.ncomp = 0
                return self._send(200, {"ok": True})
            if self.path != "/v1/completions":
                return self._send(404, {"error": "not found"})
            return self.complete(body)

        def complete(self, b):
            prompt = b.get("prompt")
            if not isinstance(prompt, list) or not prompt or not all(isinstance(x, int) for x in prompt):
                return self._send(400, {"error": "mock: prompt must be a non-empty list of token ids"})
            if b.get("temperature", 1.0) != 0:
                return self._send(400, {"error": "mock: only temperature 0"})
            with st.lock:
                st.ncomp += 1
                idx = st.ncomp
                st.running += 1
                phantom = idx in st.phantom_at
                if phantom:
                    st.running += 1
                nonce = random.getrandbits(48)
            m = st.model
            gen = None
            try:
                if st.delay:
                    time.sleep(st.delay)
                k_pl = b.get("prompt_logprobs")
                k_lp = b.get("logprobs")
                pl = None
                if k_pl is not None:
                    pl = [None]
                    for j in range(1, len(prompt)):
                        lp = m.dist(prompt[:j], "prefill", nonce)
                        # vLLM's layout: the actual token first (its rank = #logprobs >= its own), then the top-k
                        # with positional ranks; a token in both keeps the first position, the top-k rank
                        a = prompt[j]
                        row = {str(a): {"logprob": lp[a], "rank": sum(1 for x in lp if x >= lp[a]),
                                        "decoded_token": "<%d>" % a}}
                        for r, t in enumerate(topk(lp, k_pl)):
                            row[str(t)] = {"logprob": lp[t], "rank": r + 1, "decoded_token": "<%d>" % t}
                        pl.append(row)
                ctx = list(prompt)
                gen, tl, tops = [], [], []
                for i in range(int(b.get("max_tokens", 16))):
                    lp = m.dist(ctx, "prefill" if i == 0 else "decode", nonce)
                    t = max(range(len(lp)), key=lambda x: (lp[x], -x))
                    gen.append(t)
                    tl.append(lp[t])
                    if k_lp is not None:
                        key = (lambda x: "token_id:%d" % x) if b.get("return_tokens_as_token_ids") else (lambda x: "<%d>" % x)
                        d = {key(t): lp[t]}                       # the sampled token first, then the top-k
                        for x in topk(lp, k_lp):
                            d[key(x)] = lp[x]
                        tops.append(d)
                    ctx.append(t)
                c = {"index": 0, "text": "".join("<%d>" % t for t in gen), "finish_reason": "length",
                     "stop_reason": None, "prompt_logprobs": pl}
                if k_lp is not None:
                    key = (lambda x: "token_id:%d" % x) if b.get("return_tokens_as_token_ids") else (lambda x: "<%d>" % x)
                    c["logprobs"] = {"tokens": [key(t) for t in gen], "token_logprobs": tl, "top_logprobs": tops,
                                     "text_offset": list(range(len(gen)))}
                else:
                    c["logprobs"] = None
                if b.get("return_token_ids"):
                    c["token_ids"] = gen
                    c["prompt_token_ids"] = list(prompt)
                resp = {"id": "cmpl-mock", "object": "text_completion", "model": b.get("model"), "choices": [c],
                        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(gen)}}
            finally:
                with st.lock:
                    st.running -= 1
                    st.success += 1
                    st.prompt_tokens += len(prompt)
                    if idx in st.cached_at:
                        st.cached += 4608
                    if phantom:
                        st.running -= 1
                        st.success += 1
                    st.log.append({"n": idx, "prompt_len": len(prompt), "max_tokens": b.get("max_tokens"),
                                   "gen": gen, "nonce": nonce,
                                   "salt": b.get("cache_salt"), "prompt_logprobs": k_pl, "phantom": phantom})
            return self._send(200, resp)
    return H


def serve(port=0, **kw):
    st = State(Model(**kw))
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18888)
    ap.add_argument("--V", type=int, default=256)
    a = ap.parse_args()
    srv, st = serve(a.port, V=a.V)
    print("mock vLLM on 127.0.0.1:%d" % srv.server_address[1], flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
