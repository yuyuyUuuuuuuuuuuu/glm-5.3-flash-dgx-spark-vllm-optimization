#!/usr/bin/env python3
"""Sampled-decoding bench for the GLM server (run on nodeA by the operator; read-only, stdlib only).

Why: tests/bench_decode.py of the launcher runs temperature 0, where block verification == standard by construction.
This bench drives the live OpenAI API with production's sampling (temperature 1.0, top_p 0.95; repetition_penalty
left to the server default = the launcher's 1.05 override unless --repetition-penalty is given) and FIXED seeds, one
request at a time, and reports per prompt:
  accepted tokens / step  = 1 + d(num_accepted_tokens) / d(num_drafts)   (vllm:spec_decode_* counters of /metrics)
  per-position acceptance = d(num_accepted_tokens_per_pos{position=i}) / d(num_drafts)
  draft tokens / step     = d(num_draft_tokens) / d(num_drafts)          (adaptive K's verified length)
  tok/s                   = (completion_tokens - 1) / (end - first token)  (streaming, as bench_decode.py)
  ms/step                 = decode seconds * 1000 / d(num_drafts)
Idle gating (the counters are server-global, and the bench must not compete with users):
  * before the bench: /metrics must show vllm:num_requests_running == 0 and vllm:num_requests_waiting == 0 on 3
    consecutive polls 5 s apart (as tools/wait_idle.sh), else exit 1 without sending anything (--max-wait, 900 s)
  * before every request: the same gauges must be 0 (else wait, up to --max-wait)
  * during every request: a poller reads the gauges every 0.5 s; running > 1 or waiting > 0 marks the request
    "contaminated" (another client arrived), as does a counter delta that our request alone cannot explain
    (drafts + accepted must equal completion_tokens - 1 up to the last step's overshoot: -3..8 tolerated). Contaminated requests
    are dropped from the summary and retried (--retries, default 2 per request).
Seeds: request i of prompt P uses seed = --seed-base + i (same set for every run, so A and B draw the same seeds).
Auth: API_KEY / VLLM_API_KEY env (never printed). /metrics and /health are read without a key.
Usage:
  bench_sampled.py --label standard --out /tmp/bs_standard.json [--seeds 8] [--max-tokens 256] [--port 8888]
  bench_sampled.py --compare /tmp/bs_standard.json /tmp/bs_block.json
Exit: 0 ok, 1 server busy (nothing sent) or every request contaminated, 2 unreachable / not healthy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = {
    # the three regimes of the launcher's tests/bench_decode.py
    "prose": ("Write a detailed step-by-step explanation of how a hash map works, including collision handling, "
              "resizing, and time complexity. Be thorough."),
    "coding": ("Write a Python function named clamp_range that takes a list of ints and returns a new list with each "
               "value clamped to [0, 50]. Include a short docstring."),
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
}
SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-zA-Z0-9_]+?)(?:\{([^}]*)\})?\s+(\S+)$")
GAUGE_RE = re.compile(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})?\s+(\S+)$")


class Server:
    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def _key_headers(self) -> dict[str, str]:
        key = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def health(self) -> int:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0

    def metrics(self) -> tuple[dict[str, float], dict[str, float]]:
        """(spec counters, request gauges). Raises if the gauges are missing (not treated as idle)."""
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            raw = r.read().decode("utf-8", "replace")
        spec: dict[str, float] = {}
        gauges: dict[str, float] = {}
        for line in raw.splitlines():
            m = GAUGE_RE.match(line)
            if m:
                gauges[m.group(1)] = gauges.get(m.group(1), 0.0) + float(m.group(2))
                continue
            m = SPEC_RE.match(line)
            if not m or m.group(1).endswith("_created"):
                continue
            name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
            if "per_pos" in name:
                pos = re.search(r'position="(\d+)"', labels)
                if pos:
                    k = f"pos:{pos.group(1)}"
                    spec[k] = spec.get(k, 0.0) + val
            else:
                spec[name] = spec.get(name, 0.0) + val
        if "vllm:num_requests_running" not in gauges or "vllm:num_requests_waiting" not in gauges:
            raise RuntimeError("/metrics has no vllm:num_requests_running / num_requests_waiting gauge")
        return spec, gauges

    def stream(self, prompt: str, seed: int, max_tokens: int, temperature: float, top_p: float,
               repetition_penalty: float | None) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                "top_p": top_p, "seed": seed, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **self._key_headers()}, method="POST")
        t0 = time.perf_counter()
        first = None
        usage = None
        finish = None
        n_chunks = 0
        with urllib.request.urlopen(req, timeout=900) as resp:
            buf = b""
            while True:
                piece = resp.read1(256)
                if not piece:
                    break
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
                    if obj.get("usage"):
                        usage = obj["usage"]
                    ch = obj.get("choices") or []
                    if not ch:
                        continue
                    d = ch[0].get("delta") or {}
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        n_chunks += 1
                        if first is None:
                            first = time.perf_counter()
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
        t1 = time.perf_counter()
        ct = int((usage or {}).get("completion_tokens") or 0)
        return {"ttft_s": None if first is None else first - t0, "decode_s": None if first is None else t1 - first,
                "wall_s": t1 - t0, "completion_tokens": ct, "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
                "finish_reason": finish, "chunks": n_chunks}


def idle_now(srv: Server) -> bool:
    _, g = srv.metrics()
    return g["vllm:num_requests_running"] == 0 and g["vllm:num_requests_waiting"] == 0


def wait_idle(srv: Server, polls: int, interval: float, max_wait: float) -> bool:
    t0, ok = time.time(), 0
    while True:
        _, g = srv.metrics()
        run, wai = g["vllm:num_requests_running"], g["vllm:num_requests_waiting"]
        ok = ok + 1 if run == 0 and wai == 0 else 0
        print(f"[bench] idle gate {time.strftime('%T')}: running={run:g} waiting={wai:g} idle_polls={ok}/{polls}", flush=True)
        if ok >= polls:
            return True
        if time.time() - t0 >= max_wait:
            return False
        time.sleep(interval)


class Poller(threading.Thread):
    """Samples the request gauges while our request runs; any second client marks the request contaminated."""

    def __init__(self, srv: Server, period: float):
        super().__init__(daemon=True)
        self.srv, self.period = srv, period
        self.stop = threading.Event()
        self.max_running = 0.0
        self.max_waiting = 0.0
        self.errors = 0

    def run(self):
        while not self.stop.is_set():
            try:
                _, g = self.srv.metrics()
                self.max_running = max(self.max_running, g["vllm:num_requests_running"])
                self.max_waiting = max(self.max_waiting, g["vllm:num_requests_waiting"])
            except Exception:
                self.errors += 1
            self.stop.wait(self.period)


def spec_delta(a: dict, b: dict, k_max: int = 7) -> dict:
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    return {"drafts": int(drafts), "draft_tokens": int(d.get("vllm:spec_decode_num_draft_tokens_total", 0.0)),
            "accepted": int(d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)),
            "accepted_per_pos": [int(d.get(f"pos:{i}", 0.0)) for i in range(k_max)]}


def one_request(srv: Server, args, name: str, prompt: str, seed: int) -> dict:
    t_wait = time.time()
    while not idle_now(srv):
        if time.time() - t_wait > args.max_wait:
            return {"prompt": name, "seed": seed, "contaminated": True, "why": "server busy before the request"}
        time.sleep(2.0)
    before, _ = srv.metrics()
    poll = Poller(srv, args.poll)
    poll.start()
    try:
        r = srv.stream(prompt, seed, args.max_tokens, args.temperature, args.top_p, args.repetition_penalty)
    finally:
        poll.stop.set()
        poll.join()
    time.sleep(0.3)   # the scheduler updates the counters after the last step
    after, _ = srv.metrics()
    s = spec_delta(before, after)
    r.update(prompt=name, seed=seed, spec=s, max_running=poll.max_running, max_waiting=poll.max_waiting)
    explained = s["drafts"] + s["accepted"] - max(r["completion_tokens"] - 1, 0)
    why = []
    if poll.max_running > 1 or poll.max_waiting > 0:
        why.append(f"other requests seen (running<={poll.max_running:g}, waiting<={poll.max_waiting:g})")
    if not -3 <= explained <= 8:
        why.append(f"counter delta not explained by this request (drafts+accepted-(tokens-1)={explained})")
    if s["drafts"] <= 0 or not r["decode_s"]:
        why.append("no speculative steps / no decode time")
    r["contaminated"] = bool(why)
    r["why"] = "; ".join(why)
    if not why:
        r["accepted_tokens_per_step"] = 1.0 + s["accepted"] / s["drafts"]
        r["draft_tokens_per_step"] = s["draft_tokens"] / s["drafts"]
        r["tok_s"] = (r["completion_tokens"] - 1) / r["decode_s"] if r["completion_tokens"] > 1 else None
        r["ms_per_step"] = 1000.0 * r["decode_s"] / s["drafts"]
    return r


def mean_se(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else None
    return m, se, len(xs)


def summarize(runs: list[dict]) -> dict:
    out = {}
    for name in list(PROMPTS) + ["all"]:
        rs = [r for r in runs if not r.get("contaminated") and (name == "all" or r["prompt"] == name)]
        if not rs:
            continue
        drafts = sum(r["spec"]["drafts"] for r in rs)
        acc = sum(r["spec"]["accepted"] for r in rs)
        dtok = sum(r["spec"]["draft_tokens"] for r in rs)
        per_pos = [sum(r["spec"]["accepted_per_pos"][i] for r in rs) / drafts for i in range(7)]
        a, ase, n = mean_se([r["accepted_tokens_per_step"] for r in rs])
        t, tse, _ = mean_se([r["tok_s"] for r in rs])
        ms, mse, _ = mean_se([r["ms_per_step"] for r in rs])
        out[name] = {"requests": n, "steps": drafts,
                     "accepted_tokens_per_step": 1.0 + acc / drafts,        # pooled over steps
                     "accepted_tokens_per_step_mean_se": [a, ase],         # per-request mean, SE across seeds
                     "draft_tokens_per_step": dtok / drafts,
                     "acceptance_per_position": [round(x, 4) for x in per_pos],
                     "tok_s_mean_se": [t, tse], "ms_per_step_mean_se": [ms, mse],
                     "completion_tokens": sum(r["completion_tokens"] for r in rs)}
    return out


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def compare(a_path: str, b_path: str) -> int:
    A, B = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    print(f"A = {A['label']} ({A['ts']}), B = {B['label']} ({B['ts']}); sampling {A['sampling']} vs {B['sampling']}")
    if A["sampling"] != B["sampling"] or A["seeds"] != B["seeds"]:
        print("WARNING: A and B used different sampling parameters or seeds")
    for name in list(PROMPTS) + ["all"]:
        sa, sb = A["summary"].get(name), B["summary"].get(name)
        if not sa or not sb:
            continue
        line = [f"{name:10s}"]
        for key, label in (("accepted_tokens_per_step_mean_se", "accepted/step"), ("tok_s_mean_se", "tok/s"),
                           ("ms_per_step_mean_se", "ms/step")):
            (ma, ea), (mb, eb) = sa[key], sb[key]
            if ma is None or mb is None:
                continue
            se = math.sqrt((ea or 0) ** 2 + (eb or 0) ** 2)
            line.append(f"{label} {fmt(ma)} -> {fmt(mb)} ({100 * (mb / ma - 1):+.2f} %, diff {mb - ma:+.3f} +/- {1.96 * se:.3f})")
        line.append(f"draft tok/step {fmt(sa['draft_tokens_per_step'], 2)} -> {fmt(sb['draft_tokens_per_step'], 2)}")
        print(" | ".join(line))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", help="what the server runs, e.g. standard | block (recorded, not detected)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GLM_PORT", "8888")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", "GLM-5.3-Flash-EXL3"))
    ap.add_argument("--prompts", default="prose,coding,structured")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=20260929)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="default: not sent (the server's default, i.e. the launcher's 1.05 generation override)")
    ap.add_argument("--warmup-tokens", type=int, default=32)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--idle-polls", type=int, default=3)
    ap.add_argument("--idle-interval", type=float, default=5.0)
    ap.add_argument("--max-wait", type=float, default=900.0)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.label or not args.out:
        ap.error("--label and --out are required (or --compare A B)")
    names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for p in names:
        if p not in PROMPTS:
            ap.error(f"unknown prompt {p!r} (known: {', '.join(PROMPTS)})")
    srv = Server(f"http://{args.host}:{args.port}", args.model)
    if srv.health() != 200:
        print("[bench] /health is not 200 - not running", flush=True)
        return 2
    try:
        if not wait_idle(srv, args.idle_polls, args.idle_interval, args.max_wait):
            print(f"[bench] server still busy after {args.max_wait:g}s - nothing sent", flush=True)
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[bench] /metrics unusable: {exc}", flush=True)
        return 2
    rec = {"label": args.label, "ts": time.strftime("%F %T"), "port": args.port, "model": args.model,
           "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty or "server default", "max_tokens": args.max_tokens,
                        "thinking": False},
           "seeds": [args.seed_base + i for i in range(args.seeds)], "prompts": names, "runs": []}
    print(f"[bench] warmup ({args.warmup_tokens} tokens)", flush=True)
    srv.stream(PROMPTS[names[0]], args.seed_base - 1, args.warmup_tokens, args.temperature, args.top_p,
               args.repetition_penalty)
    for name in names:
        for i in range(args.seeds):
            seed = args.seed_base + i
            for attempt in range(args.retries + 1):
                r = one_request(srv, args, name, PROMPTS[name], seed)
                r["attempt"] = attempt
                rec["runs"].append(r)
                if not r.get("contaminated"):
                    print(f"[bench] {name:10s} seed {seed}: {r['completion_tokens']} tokens, accepted/step "
                          f"{r['accepted_tokens_per_step']:.3f}, draft tok/step {r['draft_tokens_per_step']:.2f}, "
                          f"tok/s {fmt(r['tok_s'], 2)}, ms/step {r['ms_per_step']:.1f}", flush=True)
                    break
                print(f"[bench] {name:10s} seed {seed}: contaminated ({r['why']}) - "
                      f"{'retry' if attempt < args.retries else 'dropped'}", flush=True)
    rec["summary"] = summarize(rec["runs"])
    rec["contaminated_requests"] = sum(1 for r in rec["runs"] if r.get("contaminated"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    for name, s in rec["summary"].items():
        a, ase = s["accepted_tokens_per_step_mean_se"]
        t, tse = s["tok_s_mean_se"]
        m, mse = s["ms_per_step_mean_se"]
        print(f"[bench] {args.label} {name:10s} n={s['requests']} steps={s['steps']} accepted/step {s['accepted_tokens_per_step']:.3f} "
              f"(per-request {fmt(a)} +/- {fmt(ase)}) draft tok/step {s['draft_tokens_per_step']:.2f} tok/s {fmt(t, 2)} "
              f"+/- {fmt(tse, 2)} ms/step {fmt(m, 1)} +/- {fmt(mse, 1)} per-pos {s['acceptance_per_position']}", flush=True)
    print(f"[bench] wrote {args.out} ({rec['contaminated_requests']} contaminated requests dropped)", flush=True)
    return 0 if "all" in rec["summary"] else 1


if __name__ == "__main__":
    sys.exit(main())
