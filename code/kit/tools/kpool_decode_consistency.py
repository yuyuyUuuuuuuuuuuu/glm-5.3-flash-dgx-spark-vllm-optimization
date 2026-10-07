#!/usr/bin/env python3
"""Decode-vs-prefill consistency probe for the kpool indexer tail ring (docs/KPOOL_RING.md). Operator-run on nodeA,
once before and once after GLM53_KPOOL_RING=1, with no other traffic.

The bug corrupts pooled indexer keys that SPEC DECODE builds (the model's own output); prefill builds every pool
correctly. So for each of 3 fixed real-text prompts (~6k tokens, same text source as quality_long.py):
  A  decode:  temperature 0, max_tokens --gen (default 1536), ignore_eos, logprobs 5, token ids returned.
              The generated text's pools are built by the decode kernel (verify steps with rejections).
  B  prefill: prompt = A's prompt ids + A's generated ids, max_tokens 1, prompt_logprobs 5, its own cache_salt
              (no prefix-cache reuse of A's blocks). Every pool is built by the prefill writer.
  Per generated position j: KL(A_j || B_j) over the top-5 union (missing-mass floor, as quality_long.py) and top-1
  agreement, reported per position bucket. Decode and prefill differ by numerics anyway (other GEMM shapes), so the
  number to compare is BEFORE vs AFTER on the same texts: with the ring the decode-built pools equal the prefill-built
  ones, so the divergence should fall toward that numerics floor and stop growing with j.

  run:      kpool_decode_consistency.py <out.json> [--gen N] [--force]
  compare:  kpool_decode_consistency.py --compare before.json after.json
Refuses to run while the server reports running or waiting requests (--force overrides). Reads VLLM_API_KEY from the
launcher .env (KPOOL_PROBE_ENV overrides the path; the key is never printed). KPOOL_PROBE_URL overrides the server.
"""
import glob
import json
import math
import os
import random
import sys
import time
import urllib.request

BASE = os.environ.get("KPOOL_PROBE_URL", "http://127.0.0.1:8888")
ENV = os.environ.get("KPOOL_PROBE_ENV", os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env"))
MODEL = "GLM-5.3-Flash-EXL3"
BUCKETS = (0, 128, 512, 1024, 1536, 4096)


def key():
    for line in open(ENV):
        if line.startswith("VLLM_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"no VLLM_API_KEY in {ENV}")


def texts():
    files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
    t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
    out = []
    for i in range(3):
        s = random.Random("kpool-ring-%d" % i).randrange(0, len(t) - 30000)
        out.append(t[s:s + 24000])
    return out


def http(path, body=None, k=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if k:
        headers["Authorization"] = "Bearer " + k
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if body is not None else raw.decode()


def busy():
    text = http("/metrics")
    tot = {"running": 0.0, "waiting": 0.0}
    for line in text.splitlines():
        for name in tot:
            if line.startswith(f"vllm:num_requests_{name}"):
                tot[name] += float(line.rsplit(" ", 1)[1])
    return tot


def ids_of(top):
    """{'token_id:123': lp} (return_tokens_as_token_ids) or {'123': {'logprob': lp}} (prompt_logprobs) -> {123: lp}."""
    out = {}
    for t, v in (top or {}).items():
        tid = int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else int(t)
        out[tid] = v["logprob"] if isinstance(v, dict) else float(v)
    return out


def kl_top(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}
    lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values()))
    zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)


def run(path, gen, force):
    state = busy()
    if (state["running"] or state["waiting"]) and not force:
        raise SystemExit(f"server busy (running={state['running']:.0f} waiting={state['waiting']:.0f}); "
                         "retry when idle or pass --force")
    k = key()
    salt = "kpool-ring-probe-%d" % time.time_ns()
    res = []
    for i, tx in enumerate(texts()):
        a = http("/v1/completions", {
            "model": MODEL, "prompt": tx, "max_tokens": gen, "temperature": 0, "ignore_eos": True, "logprobs": 5,
            "return_tokens_as_token_ids": True, "return_token_ids": True, "cache_salt": salt + "-a%d" % i}, k)
        ca = a["choices"][0]
        p_ids, g_ids = ca["prompt_token_ids"], ca["token_ids"]
        tops = ca["logprobs"]["top_logprobs"]
        if len(tops) != len(g_ids):
            raise SystemExit(f"text {i}: {len(tops)} top_logprobs for {len(g_ids)} generated tokens")
        b = http("/v1/completions", {
            "model": MODEL, "prompt": p_ids + g_ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 5,
            "cache_salt": salt + "-b%d" % i}, k)
        pl = b["choices"][0]["prompt_logprobs"]
        if len(pl) != len(p_ids) + len(g_ids):
            raise SystemExit(f"text {i}: {len(pl)} prompt_logprobs for {len(p_ids) + len(g_ids)} tokens")
        pre = [ids_of(pl[len(p_ids) + j]) for j in range(len(g_ids))]
        dec = [ids_of(t) for t in tops]
        res.append({"prompt_len": len(p_ids), "gen": g_ids, "decode": dec, "prefill": pre})
        print("text %d: prompt %d tokens, generated %d" % (i, len(p_ids), len(g_ids)), flush=True)
    json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "gen": gen, "texts": res}, open(path, "w"))
    summary(path)


def stats(path):
    d = json.load(open(path))
    per = {}
    for t in d["texts"]:
        for j, (pa, pb) in enumerate(zip(t["decode"], t["prefill"])):
            if not pa or not pb:
                continue
            pa = {int(x): v for x, v in pa.items()}
            pb = {int(x): v for x, v in pb.items()}
            bi = max(i for i, lo in enumerate(BUCKETS) if j >= lo)
            s = per.setdefault(bi, [0.0, 0, 0])
            s[0] += kl_top(pa, pb)
            s[1] += max(pa, key=pa.get) == max(pb, key=pb.get)
            s[2] += 1
    return d.get("when", "?"), per


def summary(path):
    when, per = stats(path)
    print(f"{path} ({when}): decode vs prefill over generated positions")
    tk = tn = ta = 0
    for bi in sorted(per):
        s = per[bi]
        hi = BUCKETS[bi + 1] if bi + 1 < len(BUCKETS) else "end"
        print("  gen pos [%s, %s): n=%d  KL mean %.5f  top-1 agree %.2f%%" % (BUCKETS[bi], hi, s[2], s[0] / s[2],
                                                                               100.0 * s[1] / s[2]))
        tk += s[0]; ta += s[1]; tn += s[2]
    print("  all: n=%d  KL mean %.5f  top-1 agree %.2f%%" % (tn, tk / tn, 100.0 * ta / tn))
    return tk / tn


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--compare"] and len(argv) == 3:
        before, after = summary(argv[1]), summary(argv[2])
        print("KL mean before %.5f -> after %.5f (%+.1f%%)" % (before, after, 100.0 * (after - before) / before))
    elif argv and not argv[0].startswith("--"):
        gen = int(argv[argv.index("--gen") + 1]) if "--gen" in argv else 1536
        run(argv[0], gen, "--force" in argv)
    else:
        raise SystemExit(__doc__)
