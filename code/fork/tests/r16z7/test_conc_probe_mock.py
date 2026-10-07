#!/usr/bin/env python3
"""CPU test of tools/prodcheck/conc_quality_probe.py against a mock OpenAI-compatible server (no GPU, no network).

The mock's next-token distribution is a pure function of the token-id prefix (the "teacher-forced prefill" truth).
Decode requests that overlap another decode request get a perturbed distribution (MOCK_CORRUPT > 0: the stock tail
ring's cross-request mixing), solo decodes and every max_tokens=1 request are exact. Checks:
  T1 corrupt=0: conc and solo KL ~ 0, top-1 100 %, positions/alignment right (any off-by-one shows as KL > 0)
  T2 corrupt=0.6: conc KL >> solo KL (~0) and the summary's conc/solo ratio is large
  T3 busy server (/metrics running=1): rc 3 without --force, no completion request sent
  T4 auth: the key from --env is sent as Bearer and never printed; absent .env = no header
  T5 --compare prints a CONCQ-CMP line with the conc ratio; text mode (corpus prompts) runs end to end
  T6 --deadline 0: partial summary, rc 0
  T7 (r16z7 review) decode logprobs carry a per-position constant offset (a different normaliser of the same
     distribution, as one production position showed): raw dlp > 0.1 but dlpc ~ 0 and KL ~ 0 - the primary dlpc
     must not read an offset as damage
  T8 (r16z7 review) --wait-idle: a server busy for ~3 s is waited for (rc 0); --wait-idle 0 keeps T3's refusal
"""
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "..", "tools", "prodcheck", "conc_quality_probe.py")
V = 50
STATE = {"inflight_dec": 0, "lock": threading.Lock(), "corrupt": 0.0, "busy": 0, "auth": [], "n_compl": 0, "shift": 0.0,
         "noise": 0.0}


NOISE_RND = random.Random(7)


def dist(prefix):
    h = hashlib.sha256(json.dumps(prefix[-64:] + [len(prefix)]).encode()).digest()
    rnd = random.Random(h)
    logits = [rnd.gauss(0, 2.0) for _ in range(V)]
    z = math.log(sum(math.exp(x) for x in logits))
    return [x - z for x in logits]


def top(lp, k):
    idx = sorted(range(V), key=lambda i: -lp[i])[:k + 1]
    return {f"token_id:{i}": lp[i] for i in idx}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/metrics":
            self._send(f'vllm:num_requests_running{{model_name="m"}} {STATE["busy"]}.0\nvllm:num_requests_waiting{{model_name="m"}} 0.0\n', "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["auth"].append(self.headers.get("Authorization"))
        STATE["n_compl"] += 1
        p = body["prompt"]
        if isinstance(p, str):
            p = [ord(c) % V for c in p]
        n, k = body["max_tokens"], body.get("logprobs") or 0
        dec = n > 1
        if body.get("prompt_logprobs"):     # teacher-forced: exact distribution per prompt position (vLLM format)
            kk = body["prompt_logprobs"]
            pl = [None] + [{str(int(t.split(":")[1])): {"logprob": v, "rank": 1} for t, v in top(dist(p[:i]), kk).items()}
                           for i in range(1, len(p))]
            self._send({"choices": [{"prompt_token_ids": p, "token_ids": [0], "prompt_logprobs": pl,
                                     "logprobs": None}], "usage": {"prompt_tokens": len(p)}})
            return
        with STATE["lock"]:
            if dec:
                STATE["inflight_dec"] += 1
        ids, tops = [], []
        rnd = random.Random(len(p))
        for j in range(n):
            lp = dist(p + ids)
            time.sleep(0.002 if dec else 0)
            with STATE["lock"]:
                conc = STATE["inflight_dec"] > 1
            if dec and STATE["noise"] > 0:     # decode numerics floor (every decode, solo included)
                lp = [x + NOISE_RND.gauss(0, STATE["noise"]) for x in lp]
                z = math.log(sum(math.exp(x) for x in lp)); lp = [x - z for x in lp]
            if dec and conc and STATE["corrupt"] > 0:
                lp = [x + rnd.gauss(0, STATE["corrupt"]) for x in lp]
                z = math.log(sum(math.exp(x) for x in lp)); lp = [x - z for x in lp]
            t = max(range(V), key=lambda i: lp[i])
            ids.append(t)
            if dec and STATE["shift"]:          # reported logprobs only: same distribution, another normaliser
                tops.append(top([x + STATE["shift"] * (1 + j % 3) for x in lp], k))
            else:
                tops.append(top(lp, k))
        if dec:
            time.sleep(0.05)
            with STATE["lock"]:
                STATE["inflight_dec"] -= 1
        self._send({"choices": [{"prompt_token_ids": p, "token_ids": ids, "logprobs": {"top_logprobs": tops}}],
                    "usage": {"prompt_tokens": len(p), "prompt_tokens_details": {"cached_tokens": 0}}})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    td = tempfile.mkdtemp(prefix="concq-test-")
    rnd = random.Random(1)
    prompts = [[rnd.randrange(V) for _ in range(300 + 50 * i)] for i in range(5)]
    pf = os.path.join(td, "p.json"); json.dump(prompts, open(pf, "w"))
    envf = os.path.join(td, "env"); open(envf, "w").write("FOO=1\nVLLM_API_KEY=sekrit-value-xyz\n")
    fails = 0

    def probe(*args, env=None):
        if "--compare" not in args and "--mode" not in args and "--ab" not in args:
            args = ("--mode", "sampled", *args)
        r = subprocess.run([sys.executable, PROBE, "--url", url, "--model", "m", *args], capture_output=True, text=True,
                           env={**os.environ, **(env or {})})
        return r.returncode, r.stdout + r.stderr

    def ck(ok, msg):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + msg); fails += 0 if ok else 1

    def line(out):
        return [l for l in out.splitlines() if l.startswith("CONCQ ")][-1]

    def num(l, key):
        import re
        return float(re.search(key + r"=([0-9.na]+)", l).group(1))

    STATE["corrupt"] = 0.0
    rc, out = probe("--out", os.path.join(td, "t1.json"), "--prompt-ids-file", pf, "--gen", "64", "--positions", "8",
                    "--env", envf)
    l1 = line(out) if rc == 0 else ""
    ck(rc == 0 and num(l1, "conc dlpc") < 1e-9 and num(l1, "solo dlpc") < 1e-9 and num(l1, "conc kl_mean") < 1e-9
       and "n=64/8" in l1 and "conc kl_mean=0.00000 top1=100.0%" in l1 and "solo kl_mean=0.00000 top1=100.0%" in l1,
       f"T1 exact server: conc/solo dlp + KL 0, top-1 100 %, 64 (2 rounds x 4 x 8) + 8 positions (rc {rc}): {l1[:150]}")
    ck(set(STATE["auth"]) == {"Bearer sekrit-value-xyz"} and "sekrit" not in out, "T4 Bearer from --env sent, never printed")
    STATE["corrupt"] = 0.6
    rc, out = probe("--out", os.path.join(td, "t2.json"), "--prompt-ids-file", pf, "--gen", "64", "--positions", "8",
                    "--env", envf)
    l2 = line(out) if rc == 0 else ""
    ck(rc == 0 and num(l2, "conc dlpc") > 0.1 and num(l2, "solo dlpc") < 1e-9 and num(l2, "conc kl_mean") > 0.05,
       f"T2 corrupting concurrent decodes: conc dlpc/KL >> solo = 0 (rc {rc}): {l2[:150]}")
    STATE["busy"] = 1; n0 = STATE["n_compl"]
    rc, out = probe("--out", os.path.join(td, "t3.json"), "--prompt-ids-file", pf, "--gen", "8", "--positions", "2",
                    "--wait-idle", "0")
    ck(rc == 3 and "server busy" in out and STATE["n_compl"] == n0, f"T3 busy server -> rc {rc}, no request sent")
    threading.Timer(3.0, lambda: STATE.update(busy=0)).start()
    t8 = time.time()
    rc, out = probe("--out", os.path.join(td, "t8.json"), "--prompt-ids-file", pf, "--gen", "8", "--positions", "2",
                    "--wait-idle", "30")
    ck(rc == 0 and 2.5 < time.time() - t8 < 30 and line(out).startswith("CONCQ "),
       f"T8 --wait-idle 30: busy for 3 s is waited for, then runs (rc {rc}, {time.time() - t8:.1f}s)")
    STATE["busy"] = 0; STATE["auth"].clear()
    rc, out = probe("--compare", os.path.join(td, "t2.json"), os.path.join(td, "t1.json"))
    ck(rc == 0 and "CONCQ-CMP t2.json -> t1.json: conc dlpc" in out, "T5 --compare: CONCQ-CMP line")
    STATE["corrupt"] = 0.0
    rc, out = probe("--out", os.path.join(td, "t5.json"), "--ctx", "400,500", "--streams", "2", "--solo-ctx", "450",
                    "--gen", "24", "--positions", "4", "--env", os.path.join(td, "missing-env"))
    l5 = line(out) if rc == 0 else ""
    ck(rc == 0 and "n=16/4" in l5 and set(STATE["auth"]) == {None}, f"T5 text mode end to end, no .env = no auth header (rc {rc}): {l5[:120]}")
    rc, out = probe("--out", os.path.join(td, "t6.json"), "--prompt-ids-file", pf, "--gen", "16", "--positions", "4",
                    "--deadline", "0")
    ck(rc == 0 and line(out).endswith("PARTIAL"), f"T6 --deadline 0 -> partial summary (rc {rc})")
    STATE["shift"] = 0.3
    rc, out = probe("--out", os.path.join(td, "t7.json"), "--prompt-ids-file", pf, "--gen", "32", "--positions", "8")
    l7 = line(out) if rc == 0 else ""
    ck(rc == 0 and num(l7, "raw conc dlp") > 0.1 and num(l7, "conc dlpc") < 1e-9 and num(l7, "solo dlpc") < 1e-9
       and num(l7, "conc kl_mean") < 1e-9 if rc == 0 else False,
       f"T7 constant per-position offset: raw dlp > 0.1, dlpc = KL = 0 (rc {rc}): {l7[:200]}")
    STATE["shift"] = 0.0
    # ---- trace mode (the default, r16z7 review): all positions vs one prompt_logprobs prefill, paired solo per prompt
    STATE["corrupt"] = 0.0
    rc, out = probe("--mode", "trace", "--out", os.path.join(td, "t9.json"), "--prompt-ids-file", pf, "--gen", "40",
                    "--env", envf)
    l9 = line(out) if rc == 0 else ""
    ck(rc == 0 and l9.startswith("CONCQ mode=trace") and num(l9, "conc kl") < 1e-9 and num(l9, "solo kl") < 1e-9
       and "streams=10/5" in l9 and "positions=%d" % (15 * 40) in l9 and "top1 conc=100.0% solo=100.0%" in l9,
       f"T9 trace, exact server: KL 0 at all 600 positions (5 solo + 10 conc streams x 40; alignment of prompt_logprobs) (rc {rc}): {l9[:170]}")
    STATE["corrupt"] = 0.6
    rc, out = probe("--mode", "trace", "--out", os.path.join(td, "t10.json"), "--prompt-ids-file", pf, "--gen", "40")
    l10 = line(out) if rc == 0 else ""
    ck(rc == 0 and num(l10, "conc kl") > 0.05 and num(l10, "solo kl") < 1e-9,
       f"T10 trace, corrupting concurrent decodes: conc KL >> solo KL = 0 (rc {rc}): {l10[:150]}")
    STATE["corrupt"] = 0.0
    rc, out = probe("--mode", "trace", "--out", os.path.join(td, "t11.json"), "--ctx", "400,500", "--streams", "2",
                    "--gen", "24", "--env", os.path.join(td, "missing-env"))
    l11 = line(out) if rc == 0 else ""
    ck(rc == 0 and "streams=4/2" in l11 and "positions=%d" % (6 * 24) in l11, f"T11 trace text mode end to end (rc {rc}): {l11[:150]}")
    rc, out = probe("--compare", os.path.join(td, "t10.json"), os.path.join(td, "t9.json"))
    ck(rc == 0 and "CONCQ-CMP t10.json -> t9.json: conc kl" in out, "T12 --compare on trace files: CONCQ-CMP conc kl")
    rc, out = probe("--mode", "trace", "--out", os.path.join(td, "t13.json"), "--ctx", "8800", "--gen", "300")
    ck(rc == 2 and "ctx + gen > 9000" in out, f"T13 trace refuses ctx + gen > 9000 without --force (rc {rc})")
    rc, out = probe("--out", os.path.join(td, "t14.json"), "--ctx", "400", "--streams", "1", "--gen", "16", "--deadline", "0")
    ck(rc == 0 and line(out).endswith("PARTIAL"), f"T14 trace (default mode) --deadline 0 -> partial (rc {rc})")
    # ---- --ab: OFF arms with corrupting concurrent decodes vs ON arms without, both over a decode-noise floor
    STATE["noise"] = 0.15
    fl = {}
    for nm, cor in (("off1", 0.5), ("off2", 0.5), ("on1", 0.0), ("on2", 0.0), ("on3", 0.0)):
        STATE["corrupt"] = cor
        rc, out = probe("--mode", "trace", "--out", os.path.join(td, nm + ".json"), "--prompt-ids-file", pf, "--gen", "40")
        fl[nm] = os.path.join(td, nm + ".json")
    STATE["corrupt"] = 0.0; STATE["noise"] = 0.0
    rc, out = probe("--ab", fl["off1"], fl["off2"], "--on", fl["on1"], fl["on2"])
    ck(rc == 0 and "verdict=BETTER" in out, f"T15 --ab corrupting OFF vs clean ON -> BETTER: {out.strip()[-160:]}")
    rc, out = probe("--ab", fl["on1"], fl["on2"], "--on", fl["off1"], fl["off2"])
    ck(rc == 0 and "verdict=WORSE" in out, f"T16 --ab reversed -> WORSE: {out.strip()[-160:]}")
    rc, out = probe("--ab", fl["on1"], fl["on2"], "--on", fl["on3"], fl["on2"])
    ck(rc == 0 and "verdict=NO-DIFF" in out, f"T17 --ab clean vs clean -> NO-DIFF: {out.strip()[-160:]}")
    rc, out = probe("--ab", os.path.join(td, "t2.json"), "--on", fl["on1"])
    ck(rc != 0 and "needs trace-mode files" in out, f"T18 --ab refuses a sampled-mode file (rc {rc})")
    srv.shutdown()
    print("test_conc_probe_mock: " + ("ALL OK" if fails == 0 else f"{fails} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
