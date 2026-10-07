#!/usr/bin/env python3
"""klh - KL harness for the production GLM server (nodeA, operator-run, idle production only). stdlib only
(numpy is used when importable, never required). Reads VLLM_API_KEY from the launcher .env; the key is never printed.

Why (docs/KL_HARNESS.md): production prefill is run-to-run non-deterministic at the request level (two identical
requests in one boot: per-position KL 0.005 before the e4m3 MoE, 0.0139 now), so
  * quality_long's "KL vs R15" (0.0141-0.0153) is mostly that noise; its per-run spread is +-0.00035 (3 texts,
    28,275 positions; the noise is ~iid per position, per-position sd 0.07) - resolution scales with 1/sqrt(positions);
  * the free-running decode-vs-prefill probe compares DIFFERENT texts per run (base runs diverge from each other
    after 0..124 tokens), hence 0.0118-0.0224 for one config.
klh therefore measures on FIXED token sequences only, repeats each request (within-run A/A = the noise floor of that
boot), and compares runs position by position with block-bootstrap intervals.

Measurements (one `run` = one output file <out>/<label>.klh.gz):
  pf   teacher-forced prefill (prompt_logprobs) of each fixture text, --reps times (default 2), interleaved
       (rep 0 of every text, then rep 1). Gives per-position top-K dicts + the logprob of the actual next token.
  dec  teacher-forced decode-vs-prefill on fixed trajectories T (greedy continuations made ONCE by `make-traj`
       on production and frozen; or the real continuation for fixtures marked real):
       B = one prefill of prompt+T with prompt_logprobs (per rep);
       A = the DECODE path on the same contexts, restart-on-divergence: request prompt+T[:k] with temperature 0,
           compare the generated tokens with T; every position up to and including the first mismatch was decoded
           on the forced context P+T[:pos]; the next request starts after the mismatch. Each segment's first row is
           computed by prefill (marked, excluded from the decode metric). Every position of T gets exactly one A row.
       dvptf = mean KL(A_j || mean_r B_j) over decode-computed rows: the same positions for every arm.
Every request runs only when the server is idle (running+waiting == 0) and is discarded + retried when anything
else ran during it (1 Hz /metrics poll: running+waiting <= 1; request_success_total delta == 1 after; prompt cache
hits == 0). Unique cache_salt per request; prompt_logprobs requests skip prefix-cache reads anyway.

usage:
  klh.py make-traj [--fixtures F] [--traj OUT] [--force]       (once, on production base; idle server)
  klh.py run <label> [--fixtures F] [--traj T] [--reps 2] [--dec-reps 1] [--only pf|dec] [--quick] [--dec-gen N]
  klh.py summary <run.klh.gz>
  klh.py compare <ref> <base> [<arm> ...] [--r15 QL_R15a.json] [--json out.json]
        <ref> = a fixed reference run (a .klh.gz, or 'none' = base is the reference); arms are judged vs base.
  klh.py import-ql <QL_x.json> <out.klh.gz> [--fixtures F]      (quality_long output -> klh pf records ql0..ql2)
Env: KLH_URL (http://127.0.0.1:8888), KLH_ENV (launcher .env), KLH_NO_AUTH=1, KLH_MODEL, KLH_OUT (~/tf-exl3-deploy/klh),
     KLH_IDLE_WAIT (900 s), KLH_TRIES (4), KLH_POLL (1 s), KLH_BUSY_SLEEP (5 s).
Exit codes: 0 ok, 2 usage/fixture error, 3 server busy (could not get a clean measurement), 4 response format error.
"""
import argparse
import array
import base64
import gzip
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

try:  # optional acceleration, never required
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("KLH_URL", "http://127.0.0.1:8888").rstrip("/")
ENVF = os.path.expanduser(os.environ.get("KLH_ENV", "~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env"))
MODEL = os.environ.get("KLH_MODEL", "GLM-5.3-Flash-EXL3")
OUT = os.path.expanduser(os.environ.get("KLH_OUT", "~/tf-exl3-deploy/klh"))
DEF_FIX = os.path.join(HERE, "fixtures", "klh_v1.json.gz")
IDLE_WAIT = float(os.environ.get("KLH_IDLE_WAIT", "900"))
TRIES = int(os.environ.get("KLH_TRIES", "4"))
POLL = float(os.environ.get("KLH_POLL", "1.0"))          # /metrics poll interval (idle check, during a request)
BUSY_SLEEP = float(os.environ.get("KLH_BUSY_SLEEP", "5"))  # wait between idle checks while the server is busy
FORMAT = 1


class Busy(Exception):
    pass


class FormatError(Exception):
    pass


def log(msg):
    print("[klh %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ----------------------------------------------------------------------------------------------- server access
def api_key():
    if os.environ.get("KLH_NO_AUTH") == "1":
        return None
    try:
        for line in open(ENVF):
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    raise SystemExit("klh: no VLLM_API_KEY in %s (set KLH_ENV, or KLH_NO_AUTH=1 for an unauthenticated server)" % ENVF)


def http(path, body=None, key=None, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if body is not None else raw.decode()


METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success_total",
           "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total")


def metrics():
    """sum over label sets of the metrics klh uses; a metric the server does not expose is None."""
    out = {m: None for m in METRICS}
    for line in http("/metrics", timeout=30).splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            try:
                v = float(line.rsplit(" ", 1)[1])
            except ValueError:
                continue
            out[name] = (out[name] or 0.0) + v
    return out


def _load(m):
    return (m["vllm:num_requests_running"] or 0.0) + (m["vllm:num_requests_waiting"] or 0.0)


class Guard:
    """Runs one completion request on an idle server and proves nothing else ran during it."""

    def __init__(self, key, idle_wait=IDLE_WAIT, tries=TRIES, poll=POLL, force=False):
        self.key, self.idle_wait, self.tries, self.poll, self.force = key, idle_wait, tries, poll, force
        self.discarded = 0
        self.requests = 0
        self.last_clean = 0.0      # end time of the last clean request (back-to-back requests need one idle poll)

    def wait_idle(self):
        t0 = time.time()
        clean = 0
        need = 1 if time.time() - self.last_clean < 5.0 else 2
        while True:
            if _load(metrics()) == 0:
                clean += 1
                if clean >= need:
                    return
            else:
                clean = 0
            if time.time() - t0 > self.idle_wait:
                raise Busy("server not idle for %.0f s" % self.idle_wait)
            time.sleep(self.poll if clean else max(self.poll, BUSY_SLEEP))

    def call(self, body):
        last = "?"
        for attempt in range(self.tries):
            if not self.force:
                self.wait_idle()
            m0 = metrics()
            peak = [0.0]
            stop = threading.Event()

            def poller():
                while not stop.wait(self.poll):
                    try:
                        peak[0] = max(peak[0], _load(metrics()))
                    except Exception:
                        pass
            th = threading.Thread(target=poller, daemon=True)
            th.start()
            try:
                resp = http("/v1/completions", body, self.key)
            finally:
                stop.set()
                th.join()
            self.requests += 1
            if self.force:
                return resp
            # the success counter can lag the response by an output-processor step: wait up to 3 s for our +1
            m1 = metrics()
            for _ in range(15):
                if m0["vllm:request_success_total"] is None or \
                        m1["vllm:request_success_total"] - m0["vllm:request_success_total"] >= 1:
                    break
                time.sleep(0.2)
                m1 = metrics()
            why = []
            if peak[0] > 1:
                why.append("running+waiting reached %.0f during the request" % peak[0])
            if _load(m1) != 0:
                why.append("running+waiting %.0f after the request" % _load(m1))
            if m0["vllm:request_success_total"] is not None:
                d = m1["vllm:request_success_total"] - m0["vllm:request_success_total"]
                if d != 1:
                    why.append("request_success_total moved by %.0f" % d)
            if m0["vllm:prompt_tokens_cached_total"] is not None and m1["vllm:prompt_tokens_cached_total"] is not None:
                d = m1["vllm:prompt_tokens_cached_total"] - m0["vllm:prompt_tokens_cached_total"]
                if d != 0:
                    why.append("prefix-cache hits %.0f tokens" % d)
            if not why:
                self.last_clean = time.time()
                return resp
            last = "; ".join(why)
            self.discarded += 1
            log("request discarded (attempt %d/%d): %s" % (attempt + 1, self.tries, last))
        raise Busy("no clean measurement after %d attempts (%s)" % (self.tries, last))


# ----------------------------------------------------------------------------------------------- record packing
def _b64(arr):
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _unb64(s, typecode):
    a = array.array(typecode)
    a.frombytes(base64.b64decode(s))
    return a


def tid(t):
    return int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else int(t)


def pack_rows(rows, k):
    """rows: vLLM logprob dicts ({token(str|int): logprob | {'logprob', 'rank', ...}}) or None. The server returns
    the actual/sampled token FIRST, then the top-k in rank order (a token in both keeps the first position and the
    top-k rank), so a row has <= k+1 entries. -> arrays of width k+1 in the server's order: ids int32 (-1 = empty),
    lps float32, rk int32 (the 'rank' field; 0 = not reported, e.g. decode top_logprobs)."""
    n = len(rows)
    W = k + 1
    ids = array.array("i", [-1]) * (n * W)
    lps = array.array("f", [0.0]) * (n * W)
    rk = array.array("i", [0]) * (n * W)
    for i, row in enumerate(rows):
        if not row:
            continue
        base = i * W
        for j, (t, v) in enumerate(row.items()):
            if j >= W:
                break
            ids[base + j] = tid(t)
            if isinstance(v, dict):
                lps[base + j] = float(v["logprob"])
                rk[base + j] = int(v.get("rank") or 0)
            else:
                lps[base + j] = float(v)
    return ids, lps, rk


def row_dict(rec, i, topk_only=True):
    """position i of a packed record -> {token: logprob} in the server's order. topk_only drops an entry whose
    reported rank is > k (the actual token outside the top-k: present on one side only, it would bias KL)."""
    k = rec["k"]
    W = k + 1
    base = i * W
    ids, lps, rk = rec["ids"], rec["lps"], rec.get("rk")
    d = {}
    for j in range(W):
        t = ids[base + j]
        if t < 0:
            continue
        if topk_only and rk is not None and rk[base + j] > k:
            continue
        d[t] = lps[base + j]
    return d


def row_dict_topn(rec, i, topn):
    """vLLM's dict for prompt_logprobs=topn reconstructed from a top-k record (k >= topn): the first (actual) entry,
    then the entries of rank <= topn in the server's order - quality_long's dicts, incl. their insertion order."""
    k = rec["k"]
    W = k + 1
    base = i * W
    ids, lps, rk = rec["ids"], rec["lps"], rec.get("rk")
    d = {}
    for j in range(W):
        t = ids[base + j]
        if t < 0:
            continue
        r = rk[base + j] if rk is not None else 0
        if j == 0 or r == 0 or r <= topn:
            if t not in d:
                d[t] = lps[base + j]
    return d


def lp_of(rec, i, tok):
    W = rec["k"] + 1
    base = i * W
    ids = rec["ids"]
    for j in range(W):
        if ids[base + j] == tok:
            return rec["lps"][base + j]
    return None


class Writer:
    def __init__(self, path, meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tmp = path + ".part"
        self.path = path
        self.f = gzip.open(self.tmp, "wt")
        self.f.write(json.dumps(dict(meta, kind="meta")) + "\n")

    def rec(self, **r):
        out = {}
        for key, v in r.items():
            out[key] = {"b64": _b64(v), "t": v.typecode} if isinstance(v, array.array) else v
        self.f.write(json.dumps(out) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
        os.replace(self.tmp, self.path)


def read_run(path):
    meta, recs = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            for key, v in list(r.items()):
                if isinstance(v, dict) and "b64" in v:
                    r[key] = _unb64(v["b64"], v["t"])
            if r.get("kind") == "meta":
                meta = r
            else:
                recs.append(r)
    if meta is None:
        raise FormatError("%s: no meta line" % path)
    return meta, recs


# ----------------------------------------------------------------------------------------------- fixtures
def load_fixtures(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        fx = json.load(f)
    if fx.get("version") != 1:
        raise SystemExit("klh: %s: unknown fixture version" % path)
    raw = json.dumps({"pf": fx["pf"], "dec": fx["dec"]}, sort_keys=True).encode()
    fx["sha"] = hashlib.sha256(raw).hexdigest()[:16]
    return fx


def load_traj(path, fx):
    if not path or not os.path.exists(path):
        return None
    t = json.load(open(path))
    if t.get("fixtures_sha") != fx["sha"]:
        raise SystemExit("klh: %s was made for fixtures %s, these are %s" % (path, t.get("fixtures_sha"), fx["sha"]))
    return t


DEC_GEN = None     # --dec-gen: score only the first N trajectory positions of every dec fixture


def traj_of(d, traj):
    g = d["gen"] if DEC_GEN is None else min(d["gen"], DEC_GEN)
    if d.get("traj") == "real":
        return d["real"][:g]
    if traj is None or d["name"] not in traj["traj"]:
        return None
    return traj["traj"][d["name"]][:g]


# ----------------------------------------------------------------------------------------------- requests
def _choice(resp):
    try:
        return resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise FormatError("no choices[0] in the response")


def prompt_rows(resp, n):
    c = _choice(resp)
    pl = c.get("prompt_logprobs")
    if pl is None:
        pl = resp.get("prompt_logprobs")
    if pl is None or len(pl) != n:
        raise FormatError("prompt_logprobs has %s rows for %d prompt tokens" % (None if pl is None else len(pl), n))
    return pl


def salt(label, *parts):
    return "klh-%s-%s-%d" % (label, "-".join(str(p) for p in parts), time.time_ns())


def do_prefill(g, label, ids, k, tag):
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
            "cache_salt": salt(label, *tag)}
    t0 = time.time()
    rows = prompt_rows(g.call(body), len(ids))
    return rows, time.time() - t0


def do_decode_tf(g, label, prompt, T, k, tag, cap0=64, capmin=16, capmax=256):
    """Teacher-forced decode by restart-on-divergence. Returns per position j of T: (row dict, first-row flag,
    segment id); plus counters."""
    G = len(T)
    rows = [None] * G
    first = array.array("b", [0]) * G
    seg = array.array("i", [0]) * G
    k0 = 0
    s = 0
    ema = float(cap0) / 2
    gen_tokens = 0
    while k0 < G:
        cap = max(capmin, min(capmax, int(2 * ema) + 1))
        n = min(cap, G - k0)
        body = {"model": MODEL, "prompt": prompt + T[:k0], "max_tokens": n, "temperature": 0, "ignore_eos": True,
                "logprobs": k, "return_token_ids": True, "return_tokens_as_token_ids": True,
                "cache_salt": salt(label, *(tag + (s,)))}
        c = _choice(g.call(body))
        gids = c.get("token_ids")
        lp = c.get("logprobs") or {}
        tops = lp.get("top_logprobs")
        if gids is None or tops is None or len(tops) != len(gids) or not gids:
            raise FormatError("decode response: token_ids %s / top_logprobs %s" %
                              (None if gids is None else len(gids), None if tops is None else len(tops)))
        gen_tokens += len(gids)
        m = next((i for i, t in enumerate(gids) if t != T[k0 + i]), None)
        last = len(gids) - 1 if m is None else m
        for i in range(last + 1):
            rows[k0 + i] = (tops[i], gids[i])
            first[k0 + i] = 1 if i == 0 else 0
            seg[k0 + i] = s
        run = last + 1
        ema = 0.7 * ema + 0.3 * (run if m is not None else max(run, ema))
        k0 += run
        s += 1
    return rows, first, seg, s, gen_tokens


# ----------------------------------------------------------------------------------------------- make-traj / run
def cmd_make_traj(a):
    fx = load_fixtures(a.fixtures)
    if os.path.exists(a.traj) and not a.force:
        raise SystemExit("klh: %s exists (trajectories are made ONCE and frozen; --force to remake)" % a.traj)
    g = Guard(api_key(), force=a.force_busy)
    out = {"version": 1, "fixtures_sha": fx["sha"], "made": time.strftime("%Y-%m-%d %H:%M:%S"),
           "label": a.label, "traj": {}}
    for d in fx["dec"]:
        if d.get("traj") == "real":
            continue
        body = {"model": MODEL, "prompt": d["prompt"], "max_tokens": d["gen"], "temperature": 0, "ignore_eos": True,
                "return_token_ids": True, "cache_salt": salt("traj", d["name"])}
        c = _choice(g.call(body))
        t = c.get("token_ids")
        if not t or len(t) != d["gen"]:
            raise FormatError("make-traj %s: %s tokens for max_tokens %d" % (d["name"], None if t is None else len(t),
                                                                              d["gen"]))
        out["traj"][d["name"]] = t
        log("traj %s: prompt %d, generated %d" % (d["name"], len(d["prompt"]), len(t)))
    os.makedirs(os.path.dirname(os.path.abspath(a.traj)), exist_ok=True)
    json.dump(out, open(a.traj + ".part", "w"))
    os.replace(a.traj + ".part", a.traj)
    log("wrote %s (fixtures %s)" % (a.traj, fx["sha"]))


def cmd_run(a):
    global DEC_GEN
    DEC_GEN = a.dec_gen
    fx = load_fixtures(a.fixtures)
    traj = load_traj(a.traj, fx)
    pf = [d for d in fx["pf"] if not a.quick or d.get("quick")]
    dec = [d for d in fx["dec"] if not a.quick or d.get("quick")]
    if a.preset in ("gate", "gate-pf"):
        # every pf text (ql0 stays for long5 continuity), 3 reps; the dvp guard fixtures only (gate-pf: none)
        a.reps = max(a.reps, 3)
        dec = [d for d in fx["dec"] if d["name"] in GATE_DEC] if a.preset == "gate" else []
    if a.only == "pf":
        dec = []
    if a.only == "dec":
        pf = []
    missing = [d["name"] for d in dec if traj_of(d, traj) is None]
    if missing:
        raise SystemExit("klh: no trajectory for %s - run `klh.py make-traj` once on the base server (or --only pf)"
                         % ",".join(missing))
    path = os.path.join(a.out, a.label + ".klh.gz")
    if os.path.exists(path) and not a.overwrite:
        raise SystemExit("klh: %s exists (--overwrite)" % path)
    key = api_key()
    g = Guard(key, force=a.force_busy)
    meta = {"format": FORMAT, "label": a.label, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "url": BASE,
            "fixtures": os.path.basename(a.fixtures), "fixtures_path": os.path.abspath(a.fixtures),
            "fixtures_sha": fx["sha"], "reps": a.reps,
            "dec_reps": a.dec_reps, "kpf": a.kpf, "kdec": a.kdec,
            "traj_made": traj.get("made") if traj else None, "dec_gen": a.dec_gen, "pf": [d["name"] for d in pf],
            "dec": [d["name"] for d in dec], "quick": a.quick}
    w = Writer(path, meta)
    t0 = time.time()
    for r in range(a.reps):
        for d in pf:
            rows, dt = do_prefill(g, a.label, d["ids"], a.kpf, ("pf", d["name"], r))
            ids, lps, rk = pack_rows(rows[1:], a.kpf)        # record row i = prompt position i + 1
            w.rec(kind="pf", fx=d["name"], rep=r, n=len(d["ids"]), k=a.kpf, ids=ids, lps=lps, rk=rk,
                  secs=round(dt, 2))
            log("pf %s rep %d: %d positions, %.1f s" % (d["name"], r, len(d["ids"]), dt))
        for d in dec:
            T = traj_of(d, traj)
            full = d["prompt"] + T
            rows, dt = do_prefill(g, a.label, full, a.kdec, ("decB", d["name"], r))
            P = len(d["prompt"])
            sub = rows[P:P + len(T)]
            ids, lps, rk = pack_rows(sub, a.kdec)
            w.rec(kind="decB", fx=d["name"], rep=r, n=len(T), k=a.kdec, ids=ids, lps=lps, rk=rk, secs=round(dt, 2))
            log("decB %s rep %d: %d positions, %.1f s" % (d["name"], r, len(T), dt))
    for r in range(a.dec_reps):
        for d in dec:
            T = traj_of(d, traj)
            t1 = time.time()
            rows, first, seg, nseg, gen = do_decode_tf(g, a.label, d["prompt"], T, a.kdec, ("decA", d["name"], r))
            ids, lps, rk = pack_rows([x[0] for x in rows], a.kdec)
            rec = {"k": a.kdec, "ids": ids, "lps": lps}
            # the logprob of T's token at each position (NaN when outside the top-k and not sampled)
            tl = array.array("f", [float("nan")]) * len(T)
            for j in range(len(T)):
                v = lp_of(rec, j, T[j])
                if v is not None:
                    tl[j] = v
            w.rec(kind="decA", fx=d["name"], rep=r, n=len(T), k=a.kdec, ids=ids, lps=lps, rk=rk, first=first, seg=seg,
                  tlp=tl, segments=nseg, gen_tokens=gen, secs=round(time.time() - t1, 2))
            log("decA %s rep %d: %d positions, %d segments (%d decode rows), %d tokens generated, %.1f s" %
                (d["name"], r, len(T), nseg, len(T) - sum(first), gen, time.time() - t1))
    w.close()
    log("wrote %s in %.0f s (%d requests, %d discarded)" % (path, time.time() - t0, g.requests, g.discarded))
    summary(path, a.fixtures)


# ----------------------------------------------------------------------------------------------- metrics
def kl_dict(pa, pb):
    """quality_long.py's KL(A||B) over the union of both dicts, missing mass at min(both) - 1 (verbatim)."""
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}
    lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values()))
    zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)


def mix_dicts(ds):
    """the mean DISTRIBUTION of several top-k dicts (missing entries at each dict's floor) as a log dict."""
    if len(ds) == 1:
        return ds[0]
    keys = set()
    for d in ds:
        keys |= set(d)
    out = {}
    floors = [min(d.values()) - 1.0 for d in ds]
    for x in keys:
        out[x] = math.log(sum(math.exp(d.get(x, f)) for d, f in zip(ds, floors)) / len(ds))
    return out


def argmax(d):
    """the first maximal key in insertion order (= quality_long's max(p, key=p.get), ties included)."""
    return max(d, key=d.get)


def block_ci(units, nboot=2000, seed=1, q=(0.05, 0.95)):
    """units: list of (sum, count) blocks. -> (mean, lo, hi, se) by bootstrap over blocks."""
    units = [u for u in units if u[1] > 0]
    if not units:
        return (float("nan"),) * 4
    S = sum(u[0] for u in units)
    N = sum(u[1] for u in units)
    m = S / N
    if len(units) < 6:          # too few independent units for an interval (e.g. NLL over 3 legacy texts)
        return m, float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    nb = len(units)
    vals = []
    if _np is not None:
        s = _np.array([u[0] for u in units])
        c = _np.array([u[1] for u in units])
        idx = _np.random.default_rng(seed).integers(0, nb, size=(nboot, nb))
        vals = sorted((s[idx].sum(1) / c[idx].sum(1)).tolist())
    else:
        for _ in range(nboot):
            ss = cc = 0.0
            for _ in range(nb):
                u = units[rng.randrange(nb)]
                ss += u[0]
                cc += u[1]
            vals.append(ss / cc)
        vals.sort()
    mu = sum(vals) / len(vals)
    se = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
    return m, vals[int(q[0] * nboot)], vals[min(nboot - 1, int(q[1] * nboot))], se


# block sizes of the bootstrap (positions). Calibrated on 7 production base runs of 2026-10-03 (A/A pairs): the KL
# noise is ~iid per position (block-1024 SE 0.00050 vs observed A/A sd 0.00057); NLL has a run/text-level component
# (block SE 0.0011-0.0014 vs observed 0.0023), so NLL is bootstrapped over whole (fixture, rep) sequences.
BLOCK_KL = 1024
BLOCK_NLL = None      # whole sequence
BLOCK_DEC = 128


def blocks_of(values, size):
    """values: list of float|None (None = not counted) -> [(sum, count)] per block of `size` positions (size None =
    one block per sequence; smaller blocks for short sequences so that a bootstrap has >= ~16 blocks)."""
    if size is None:
        ch = [v for v in values if v is not None]
        return [(sum(ch), len(ch))]
    size = max(1, min(size, len(values) // 16))
    out = []
    for i in range(0, len(values), size):
        ch = [v for v in values[i:i + size] if v is not None]
        out.append((sum(ch), len(ch)))
    return out


class Run:
    def __init__(self, path, fx=None):
        self.path = path
        self.meta, recs = read_run(path)
        self.label = self.meta.get("label", os.path.basename(path))
        self.pf = {}    # name -> {rep: (ids, lps, k, n)}
        self.decB = {}
        self.decA = {}
        for r in recs:
            tgt = {"pf": self.pf, "decB": self.decB, "decA": self.decA}[r["kind"]]
            tgt.setdefault(r["fx"], {})[r["rep"]] = r
        self.fx = fx

    def pf_dict(self, name, rep, i, topn=None):
        r = self.pf[name][rep]
        if topn is not None:
            return row_dict_topn(r, i, topn)
        return row_dict(r, i)


def fixture_index(fx):
    return {d["name"]: d for d in fx["pf"]}, {d["name"]: d for d in fx["dec"]}


# GLM-5.3 role tokens (production tokenizer.json): a chat-format fixture is scored only where the MODEL speaks.
ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_OBSERVATION = 154826, 154827, 154828, 154829
ROLE_TOKENS = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_OBSERVATION)
SCORE_ALL = False     # --score-all: every position from min_pos (the klh_v1 P0 behaviour)
_POS_CACHE = {}


def assistant_mask(ids):
    """True for prompt positions the assistant generated: after <|assistant|> up to and including the role token
    that ends its turn (the model emits it as its stop). System prompts, user turns and <tool_response> bodies are
    loss-masked in chat training; production (P0, 2026-10-03) predicts them erratically - per-position A/A KL up to
    18, confident and different top-1 across identical requests (tool0 tool_response A/A 1.64 vs 0.067 elsewhere)."""
    m = [False] * len(ids)
    inside = False
    for j, t in enumerate(ids):
        if t in ROLE_TOKENS:
            if inside:
                m[j] = True          # the turn-ending role token is predicted by the assistant
            inside = (t == ROLE_ASSISTANT)
            continue
        m[j] = inside
    return m


def pf_positions(d, n):
    """record rows scored for fixture d (row i = prompt position i + 1): from min_pos, and for chat-format fixtures
    (group "tool") only the assistant's own tokens."""
    key = (d["name"], n, SCORE_ALL)
    if key in _POS_CACHE:
        return _POS_CACHE[key]
    lo = max(1, int(d.get("min_pos", 1)))
    rows = list(range(lo - 1, n - 1))
    if not SCORE_ALL and d.get("group") == "tool":
        m = assistant_mask(d["ids"])
        rows = [i for i in rows if m[i + 1]]
    _POS_CACHE[key] = rows
    return rows


def summary(path, fxpath=None, quiet=False):
    meta, _ = read_run(path)
    if not fxpath:
        fxpath = meta.get("fixtures_path")
        if not fxpath or not os.path.exists(fxpath):
            fxpath = os.path.join(HERE, "fixtures", meta.get("fixtures", "klh_v1.json.gz"))
    fx = load_fixtures(fxpath)
    if fx["sha"] != meta.get("fixtures_sha"):
        raise SystemExit("klh: %s was measured with fixtures %s, not %s" % (path, meta.get("fixtures_sha"), fx["sha"]))
    R = Run(path, fx)
    pfi, deci = fixture_index(fx)
    res = {"label": R.label, "groups": {}}
    # pf: NLL / top-1 of the actual token and the within-run A/A (rep 0 vs rep 1)
    tot = {"nll": [0.0, 0], "acc": [0.0, 0], "aa": [0.0, 0]}
    aa_units = []
    for name, reps in R.pf.items():
        d = pfi[name]
        grp = d.get("group", "other")
        G = res["groups"].setdefault(grp, {"nll": [0.0, 0], "acc": [0.0, 0], "aa": [0.0, 0]})
        ids = d["ids"]
        for rep, rec in reps.items():
            for i in pf_positions(d, len(ids)):
                a = ids[i + 1]
                v = lp_of(rec, i, a)
                if v is None:
                    continue
                top = row_dict(rec, i)
                hit = bool(top) and v >= max(top.values())        # the actual token is (one of) the most likely
                for acc in (G, tot):
                    acc["nll"][0] -= v
                    acc["nll"][1] += 1
                    acc["acc"][0] += 1.0 if hit else 0.0
                    acc["acc"][1] += 1
        if 0 in reps and 1 in reps:
            vals = []
            for i in pf_positions(d, len(ids)):
                pa, pb = R.pf_dict(name, 0, i), R.pf_dict(name, 1, i)
                if pa and pb:
                    v = kl_dict(pa, pb)
                    vals.append(v)
                    for acc in (G, tot):
                        acc["aa"][0] += v
                        acc["aa"][1] += 1
                else:
                    vals.append(None)
            aa_units += blocks_of(vals, BLOCK_KL)
    aa_m, aa_lo, aa_hi, aa_se = block_ci(aa_units)
    res["pf"] = {"nll": tot["nll"][0] / max(1, tot["nll"][1]), "acc": tot["acc"][0] / max(1, tot["acc"][1]),
                 "n": tot["nll"][1], "aa": aa_m, "aa_ci": (aa_lo, aa_hi)}
    # dec: teacher-forced dvp, B averaged over reps
    dv_units = []
    nd = nf = segs = gen = 0
    for name, reps in R.decA.items():
        if name not in R.decB:
            continue
        A = reps[0]
        Bs = R.decB[name]
        ok = dec_scored(R, name)
        vals = []
        for j in range(A["n"]):
            if A["first"][j]:
                vals.append(None)
                nf += 1
                continue
            if not ok[j]:
                vals.append(None)
                continue
            pa = row_dict(A, j)
            pb = mix_dicts([row_dict(b, j) for b in Bs.values()])
            vals.append(kl_dict(pa, pb) if pa and pb else None)
            nd += 1
        segs += A.get("segments", 0)
        gen += A.get("gen_tokens", 0)
        dv_units += blocks_of(vals, BLOCK_DEC)
    dm, dlo, dhi, dse = block_ci(dv_units)
    res["dec"] = {"dvptf": dm, "dvptf_ci": (dlo, dhi), "se": dse, "decode_rows": nd, "prefill_rows": nf,
                  "segments": segs, "gen_tokens": gen}
    if not quiet:
        print("== klh summary %s (%s, fixtures %s, reps %s)" % (R.label, meta.get("when"), meta.get("fixtures_sha"),
                                                              meta.get("reps")))
        p = res["pf"]
        print("  pf : n=%d  NLL %.5f  top-1(actual) %.2f%%  within-run A/A KL %.5f [%.5f, %.5f]" %
              (p["n"], p["nll"], 100 * p["acc"], p["aa"], p["aa_ci"][0], p["aa_ci"][1]))
        for grp, G in sorted(res["groups"].items()):
            print("       %-8s n=%-7d NLL %.5f  top-1 %.2f%%  A/A %.5f" % (
                grp, G["nll"][1], G["nll"][0] / max(1, G["nll"][1]), 100 * G["acc"][0] / max(1, G["acc"][1]),
                G["aa"][0] / G["aa"][1] if G["aa"][1] else float("nan")))
        q = res["dec"]
        if q["decode_rows"]:
            print("  dec: dvptf %.5f [%.5f, %.5f] (se %.5f) over %d decode rows (%d segment-start prefill rows, "
                  "%d segments, %d tokens generated)" % (q["dvptf"], q["dvptf_ci"][0], q["dvptf_ci"][1], q["se"],
                                                         q["decode_rows"], q["prefill_rows"], q["segments"],
                                                         q["gen_tokens"]))
        print("KLH %s nll=%.5f top1=%.4f aa=%.5f dvptf=%.5f dvptf_se=%.5f n_pf=%d n_dec=%d" % (
            R.label, res["pf"]["nll"], res["pf"]["acc"], res["pf"]["aa"], q["dvptf"], q["se"], res["pf"]["n"],
            q["decode_rows"]))
    return res


def legacy_ql(path):
    """quality_long output: list (3 texts) of lists of {token_id(str): logprob} -> per text list of dicts."""
    d = json.load(open(path))
    return [[{int(t): v for t, v in (p or {}).items()} for p in text] for text in d]


def cmd_import_ql(a):
    fx = load_fixtures(a.fixtures)
    pfi, _ = fixture_index(fx)
    L = legacy_ql(a.ql)
    meta = {"format": FORMAT, "label": a.label or os.path.basename(a.ql), "when": "imported", "fixtures":
            os.path.basename(a.fixtures), "fixtures_sha": fx["sha"], "reps": 1, "dec_reps": 0, "kpf": 5, "kdec": 0,
            "pf": ["ql%d" % i for i in range(len(L))], "dec": [], "imported_from": os.path.basename(a.ql)}
    w = Writer(a.out, meta)
    for ti, text in enumerate(L):
        name = "ql%d" % ti
        d = pfi[name]
        if len(text) != len(d["ids"]):
            raise SystemExit("klh: %s text %d has %d rows, fixture %s has %d tokens" % (a.ql, ti, len(text), name,
                                                                                       len(d["ids"])))
        rows = text[1:]
        ids, lps, rk = pack_rows(rows, 5)                 # legacy rows carry no rank: rk = 0 (all kept)
        w.rec(kind="pf", fx=name, rep=0, n=len(d["ids"]), k=5, ids=ids, lps=lps, rk=rk, secs=0)
    w.close()
    log("wrote %s" % a.out)


# ----------------------------------------------------------------------------------------------- compare
def kl_vs_ref_values(ref, X, name, d, topn=None):
    """per pf position of fixture `name`: KL(ref || X) with both sides averaged over their reps (rep-mean of the
    per-rep KLs against the rep-mixture of ref); topn reconstructs vLLM's top-n dicts (long5 continuity)."""
    ids = d["ids"]
    vals = []
    rr = sorted(ref.pf[name])
    xr = sorted(X.pf[name])
    for i in pf_positions(d, len(ids)):
        pref = [ref.pf_dict(name, r, i, topn) for r in rr]
        pref = [p for p in pref if p]
        if not pref:
            vals.append(None)
            continue
        pr = pref[0] if len(pref) == 1 else mix_dicts(pref)
        ks = []
        for r in xr:
            px = X.pf_dict(name, r, i, topn)
            if px:
                ks.append(kl_dict(pr, px))
        vals.append(sum(ks) / len(ks) if ks else None)
    return vals


def nll_values(X, name, d):
    ids = d["ids"]
    vals = []
    for i in pf_positions(d, len(ids)):
        a = ids[i + 1]
        v = [lp_of(rec, i, a) for rec in X.pf[name].values()]
        v = [-x for x in v if x is not None]
        vals.append(sum(v) / len(v) if v else None)
    return vals


def dec_scored(X, name):
    """positions j of a dec fixture that count: all, except for chat-format fixtures (group "tool") where only the
    assistant's own tokens of prompt+T count (P0: once the model emitted <|observation|> it wrote a fake tool
    response - an untrained span, B-side A/A 0.10). T is read back from the decB record (the server's actual-token
    entry is each row's first)."""
    fx = X.fx
    d = {x["name"]: x for x in fx["dec"]}[name] if fx else None
    A = X.decA[name][0]
    if SCORE_ALL or d is None or d.get("group") != "tool":
        return [True] * A["n"]
    B = X.decB[name][min(X.decB[name])]
    W = B["k"] + 1
    T = [B["ids"][j * W] for j in range(B["n"])]
    m = assistant_mask(d["prompt"] + T)
    P = len(d["prompt"])
    return [m[P + j] for j in range(A["n"])]


def dvp_values(X, name):
    A = X.decA[name][0]
    Bs = X.decB[name]
    ok = dec_scored(X, name)
    vals = []
    for j in range(A["n"]):
        if A["first"][j] or not ok[j]:
            vals.append(None)
            continue
        pa = row_dict(A, j)
        pb = mix_dicts([row_dict(b, j) for b in Bs.values()])
        vals.append(kl_dict(pa, pb) if pa and pb else None)
    return vals


def paired(va, vb, size):
    """paired difference b - a on positions where both exist -> blocks."""
    diff = [(y - x) if (x is not None and y is not None) else None for x, y in zip(va, vb)]
    return blocks_of(diff, size)


# ---- the gate statistic (P0 analysis, docs/KL_HARNESS.md section 7)
# ql0 is excluded: its per-request KL vs the reference moves as a whole (sd 0.0024 = 20 % of its level, every
# 1024-block shifted alike; the seven 10-03 base runs: per-run sd 0.0018 vs 0.0003-0.0005 for ql1/ql2) - it alone
# made the 12-text interval 2.6x wider. d-lp1 (real continuation) and d-tool are excluded from the dvp guard
# (per-position sd 0.13 / 0.20 vs 0.012-0.063).
GATE_PF = ("ql1", "ql2", "lp3", "lp4", "lp5", "ja-lit0", "ja-lit1", "ja-man", "tool0", "tool1", "code-c")
GATE_DEC = ("d-lp0", "d-ja", "d-jaman", "d-code")


def per_request_kl(ref, X, name, d):
    """one number per request (rep) of X: mean over the scored positions of KL(mix(ref reps) || X rep)."""
    rows = pf_positions(d, len(d["ids"]))
    rr = sorted(ref.pf[name])
    mix = []
    for i in rows:
        pr = [q for q in (ref.pf_dict(name, r, i) for r in rr) if q]
        mix.append(mix_dicts(pr) if pr else None)
    out = []
    for r in sorted(X.pf[name]):
        v = [kl_dict(p, X.pf_dict(name, r, i)) for p, i in zip(mix, rows) if p and X.pf_dict(name, r, i)]
        out.append(sum(v) / len(v))
    return out, len(rows)


def gate_delta(ref, base, arm, names, cache=None):
    """Delta pfKL (arm - base) over the gate texts, position-weighted, with a REQUEST-LEVEL standard error: per text
    the between-request variance of the per-request KLs (pooled over base and arm, >= 2 reps each) / reps. The
    request is the noise unit (a request's whole text shifts); a position bootstrap under-states it for some texts."""
    cache = {} if cache is None else cache
    if not names:
        return None
    tot_n = 0
    parts = []
    for name in names:
        d = {x["name"]: x for x in ref.fx["pf"]}[name]
        res = []
        for X in (base, arm):
            key = (X.path, name)
            if key not in cache:
                cache[key] = per_request_kl(ref, X, name, d)
            res.append(cache[key])
        (kb, n), (ka, _) = res
        if len(kb) < 2 or len(ka) < 2:
            return None
        mb, ma = sum(kb) / len(kb), sum(ka) / len(ka)
        vb = sum((x - mb) ** 2 for x in kb) / (len(kb) - 1)
        va = sum((x - ma) ** 2 for x in ka) / (len(ka) - 1)
        vp = ((len(kb) - 1) * vb + (len(ka) - 1) * va) / (len(kb) + len(ka) - 2)
        parts.append((name, n, ma - mb, vp * (1.0 / len(kb) + 1.0 / len(ka)), mb))
        tot_n += n
    delta = sum(n * d for _, n, d, _, _ in parts) / tot_n
    var = sum((n / tot_n) ** 2 * v for _, n, _, v, _ in parts)
    level = sum(n * m for _, n, _, _, m in parts) / tot_n
    df = sum(len(cache[(base.path, nm)][0]) + len(cache[(arm.path, nm)][0]) - 2 for nm, *_ in parts)
    # 90 % two-sided: Student t with the pooled df (small reps -> wider than 1.645)
    tq = {2: 2.92, 4: 2.13, 6: 1.94, 8: 1.86, 10: 1.81, 12: 1.78, 16: 1.75, 22: 1.72, 33: 1.69, 44: 1.68}
    t = next((v for k, v in sorted(tq.items()) if df <= k), 1.645)
    se = math.sqrt(var)
    return {"delta": delta, "se": se, "lo": delta - t * se, "hi": delta + t * se, "level": level, "df": df,
            "per_text": {nm: (dl, math.sqrt(v)) for nm, _, dl, v, _ in parts}}


def legacy_long5(r15, X, fx):
    """quality_long --compare r15 X on ql0..ql2 rep 0: KL mean, top-1 agree, positions >= 4608 (verbatim rule)."""
    pfi, _ = fixture_index(fx)
    kls = []
    agree = n = 0
    per_rep = {}
    for ti, text in enumerate(r15):
        name = "ql%d" % ti
        if name not in X.pf:
            return None
        d = pfi[name]
        for rep in sorted(X.pf[name]):
            acc = per_rep.setdefault(rep, [[], 0, 0])
            for pos in range(4608, len(text)):
                pa = text[pos]
                if not pa:
                    continue
                pb = X.pf_dict(name, rep, pos - 1, 5)
                if not pb:
                    continue
                acc[0].append(kl_dict(pa, pb))
                acc[1] += argmax(pa) == argmax(pb)
                acc[2] += 1
    out = {}
    for rep, (k, a, c) in per_rep.items():
        out[rep] = (sum(k) / len(k), 100.0 * a / c, c)
    return out


def cmd_compare(a):
    fx = load_fixtures(a.fixtures)
    pfi, deci = fixture_index(fx)
    runs = [Run(p, fx) for p in a.runs]
    for r in runs:
        if r.meta.get("fixtures_sha") != fx["sha"]:
            raise SystemExit("klh: %s: fixtures %s != %s" % (r.path, r.meta.get("fixtures_sha"), fx["sha"]))
    ref = None if a.ref == "none" else Run(a.ref, fx)
    base = runs[0]
    R15 = legacy_ql(a.r15) if a.r15 else None
    out = {"ref": a.ref, "runs": {}}
    cache = {}

    def vals(kind, X, name):
        key = (kind, X.path, name)
        if key not in cache:
            if kind == "kl":
                cache[key] = kl_vs_ref_values(ref if ref is not None else base, X, name, pfi[name])
            elif kind == "nll":
                cache[key] = nll_values(X, name, pfi[name])
            else:
                cache[key] = dvp_values(X, name)
        return cache[key]

    common_pf = [n for n in base.pf if all(n in X.pf for X in runs) and (ref is None or n in ref.pf)]
    common_dec = [n for n in base.decA if all(n in X.decA and n in X.decB for X in runs)]
    for X in runs:
        o = {}
        for kind, names, bs in (("kl", common_pf, BLOCK_KL), ("nll", common_pf, BLOCK_NLL), ("dvp", common_dec, BLOCK_DEC)):
            units = []
            for n in names:
                if ref is None and kind == "kl" and X is base:
                    units = []
                    break
                units += blocks_of(vals(kind, X, n), bs)
            m, lo, hi, se = block_ci(units)
            o[kind] = {"mean": m, "ci": (lo, hi), "se": se, "n": sum(u[1] for u in units)}
        if R15 is not None:
            o["long5"] = legacy_long5(R15, X, fx)
        out["runs"][X.label] = o
        if X is not base:
            dd = {}
            for kind, names, bs in (("kl", common_pf, BLOCK_KL), ("nll", common_pf, BLOCK_NLL),
                                    ("dvp", common_dec, BLOCK_DEC)):
                if ref is None and kind == "kl":
                    continue
                g = gate_delta(ref, base, X, names) if (kind == "kl" and ref is not None) else None
                if g is not None:      # request-level interval (the request is the noise unit; >= 2 reps each)
                    dd[kind] = {"delta": g["delta"], "ci": (g["lo"], g["hi"]), "se": g["se"], "n": len(names),
                                "method": "request-level t"}
                    continue
                units = []
                for n in names:
                    units += paired(vals(kind, base, n), vals(kind, X, n), bs)
                m, lo, hi, se = block_ci(units)
                dd[kind] = {"delta": m, "ci": (lo, hi), "se": se, "n": sum(u[1] for u in units),
                            "method": "block bootstrap"}
            o["vs_base"] = dd
    # print
    print("== klh compare (ref %s, base %s; 90%% block-bootstrap intervals)" % (a.ref, base.label))
    for X in runs:
        o = out["runs"][X.label]
        s = "  %-28s" % X.label
        if o["kl"]["n"]:
            s += " pfKL %.5f [%.5f,%.5f]" % (o["kl"]["mean"], o["kl"]["ci"][0], o["kl"]["ci"][1])
        s += " NLL %.5f" % o["nll"]["mean"] if o["nll"]["n"] else ""
        if o["dvp"]["n"]:
            s += " dvptf %.5f [%.5f,%.5f]" % (o["dvp"]["mean"], o["dvp"]["ci"][0], o["dvp"]["ci"][1])
        if o.get("long5"):
            s += " long5(R15) " + " ".join("r%d %.4f/%.2f%%" % (r, v[0], v[1]) for r, v in sorted(o["long5"].items()))
        print(s)
        if "vs_base" in o:
            parts = []
            for kind, nm in (("kl", "pfKL"), ("nll", "NLL"), ("dvp", "dvptf")):
                if kind in o["vs_base"] and o["vs_base"][kind]["n"]:
                    v = o["vs_base"][kind]
                    parts.append("%s %+.5f [%+.5f,%+.5f]" % (nm, v["delta"], v["ci"][0], v["ci"][1]))
            print("  %-28s   vs base (every text incl. ql0 - informational, judge with KLH-GATE): %s" % (
                "", "  ".join(parts)))
    for X in runs:
        o = out["runs"][X.label]
        l5 = o.get("long5") or {}
        l5m = sum(v[0] for v in l5.values()) / len(l5) if l5 else float("nan")
        l5t = sum(v[1] for v in l5.values()) / len(l5) if l5 else float("nan")
        line = "KLH-CMP %s pfkl=%.5f nll=%.5f dvptf=%.5f long5=%.5f long5_top1=%.2f" % (
            X.label, o["kl"]["mean"], o["nll"]["mean"], o["dvp"]["mean"], l5m, l5t)
        if "vs_base" in o:
            for kind in ("kl", "nll", "dvp"):
                if kind in o["vs_base"]:
                    v = o["vs_base"][kind]
                    line += " d_%s=%+.5f d_%s_lo=%+.5f d_%s_hi=%+.5f" % (kind, v["delta"], kind, v["ci"][0], kind,
                                                                         v["ci"][1])
        print(line)
    # the gate (section 7): request-level Delta pfKL over GATE_PF, dvp guard over GATE_DEC
    if ref is not None and len(runs) > 1 and any(n in common_pf for n in GATE_PF):
        gpf = [n for n in GATE_PF if n in common_pf]
        gdec = [n for n in GATE_DEC if n in common_dec]
        gc = {}
        print("== klh gate (Delta vs base %s; pfKL over %d texts with a request-level 90%% t-interval; dvp guard over "
              "%s)" % (base.label, len(gpf), ",".join(gdec) or "-"))
        for X in runs[1:]:
            g = gate_delta(ref, base, X, gpf, gc)
            units = []
            for n in gdec:
                units += paired(vals("dvp", base, n), vals("dvp", X, n), BLOCK_DEC)
            dm, dlo, dhi, dse = block_ci(units) if units else (float("nan"),) * 4
            out["runs"][X.label]["gate"] = {"pfkl": g, "dvp": {"delta": dm, "ci": (dlo, dhi)}}
            if g is None:
                print("  %-28s gate needs >= 2 reps per text in both runs" % X.label)
                continue
            print("  %-28s gate pfKL %+.5f [%+.5f, %+.5f] (se %.5f, df %d, level %.5f)  dvp guard %+.5f [%+.5f, %+.5f]"
                  % (X.label, g["delta"], g["lo"], g["hi"], g["se"], g["df"], g["level"], dm, dlo, dhi))
            print("KLH-GATE %s d_pfkl=%+.5f lo=%+.5f hi=%+.5f se=%.5f df=%d d_dvp=%+.5f dvp_lo=%+.5f dvp_hi=%+.5f" % (
                X.label, g["delta"], g["lo"], g["hi"], g["se"], g["df"], dm, dlo, dhi))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("make-traj")
    p.add_argument("--fixtures", default=DEF_FIX)
    p.add_argument("--traj", default=os.path.join(OUT, "traj_v1.json"))
    p.add_argument("--label", default="base")
    p.add_argument("--force", action="store_true", help="remake an existing trajectory file")
    p.add_argument("--force-busy", action="store_true", help="skip the idle/contamination guard (tests only)")
    p = sp.add_parser("run")
    p.add_argument("label")
    p.add_argument("--fixtures", default=DEF_FIX)
    p.add_argument("--traj", default=os.path.join(OUT, "traj_v1.json"))
    p.add_argument("--out", default=OUT)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--dec-reps", type=int, default=1)
    p.add_argument("--kpf", type=int, default=10)
    p.add_argument("--kdec", type=int, default=20)
    p.add_argument("--only", choices=("pf", "dec"))
    p.add_argument("--quick", action="store_true", help="only the fixtures marked quick")
    p.add_argument("--dec-gen", type=int, default=None, help="score only the first N positions of each trajectory")
    p.add_argument("--preset", choices=("gate", "gate-pf"), default=None,
                   help="gate: 12 pf texts x 3 reps + the 4 dvp-guard fixtures (~8 min); gate-pf: pf only (~4 min)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force-busy", action="store_true", help="skip the idle/contamination guard (tests only)")
    p = sp.add_parser("summary")
    p.add_argument("run")
    p.add_argument("--fixtures", default=None)
    p = sp.add_parser("compare")
    p.add_argument("--score-all", action="store_true", help="score every position (klh_v1 P0 behaviour)")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--fixtures", default=DEF_FIX)
    p.add_argument("--r15", default=None, help="quality_long JSON of R15 (QL_R15a.json) for the legacy long5 number")
    p.add_argument("--json", default=None)
    p = sp.add_parser("import-ql")
    p.add_argument("ql")
    p.add_argument("out")
    p.add_argument("--fixtures", default=DEF_FIX)
    p.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "make-traj":
            cmd_make_traj(a)
        elif a.cmd == "run":
            cmd_run(a)
        elif a.cmd == "summary":
            summary(a.run, a.fixtures)
        elif a.cmd == "compare":
            global SCORE_ALL
            SCORE_ALL = a.score_all
            cmd_compare(a)
        elif a.cmd == "import-ql":
            cmd_import_ql(a)
    except Busy as e:
        log("ABORT (busy): %s" % e)
        return 3
    except FormatError as e:
        log("ABORT (response format): %s" % e)
        return 4
    except urllib.error.URLError as e:
        log("ABORT (server): %s" % e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
