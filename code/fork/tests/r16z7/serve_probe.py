#!/usr/bin/env python3
"""r16z7: validate tools/prodcheck/conc_quality_probe.py on the nodeC mini rig through a REAL OpenAI server.

Runs INSIDE the kit-composed container of tests/r16z7/run_kit.sh (HANDOFF_DRIVER=/w/tests/r16z7/serve_probe.py): the
overlay chain has already composed site-packages exactly like a production rank (the kit's bundle + production's env +
this run's overrides, e.g. GLM53_KPOOL_TAIL_POSITIONS=2). This driver
  1. builds token-id prompts (the run_engine.py corpus + GLM-OCR tokenizer, --ctx / --solo-ctx tokens),
  2. starts `vllm.entrypoints.openai.api_server` on 127.0.0.1 with the mini model and the harness's engine settings
     (TP=1, workers in their own process like production's mp executor, FULL_AND_PIECEWISE CUDA graphs with the
     production capture sizes, fp8 KV, prefix caching, DFlash2 drafter with vLLM's synthetic acceptance so verify steps
     accept several drafts on the mini model, --skip-tokenizer-init: token-id prompts, token-id logprobs),
  3. runs the probe UNCHANGED against it (--prompt-ids-file, --no-auth),
  4. stops the server it started (its own process group only) and exits with the probe's rc.
Usage (from run_kit.sh): -- [--ctx 20000,24000,28000,32000] [--solo-ctx 26000] [--gen 192] [--positions 16]
"""
import argparse
import glob
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request

MODELS = (os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"))


def corpus_ids():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{MODELS}/GLM-OCR/tokenizer.json")
    files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
    t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
    return [int(x) for x in tok.encode(t).ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default=f"{MODELS}/GLM-5.3-Flash-handoff-mini2g")
    ap.add_argument("--ctx", default="20000,24000,28000,32000")
    ap.add_argument("--solo-ctx", type=int, default=26000)
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--kv-bytes", type=int, default=6 << 30)
    ap.add_argument("--gpu-util", type=float, default=0.3)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--boot-timeout", type=int, default=1500)
    ap.add_argument("--deadline", type=float, default=900.0)
    ap.add_argument("--mode", default="sampled", choices=("sampled", "trace"),
                    help="probe mode; trace: only the --ctx prompts (each is paired with its own solo decode)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_", "TF_EXL3", "VLLM_PREFIX"))}
    print(f"serve_probe {a.label}: flags {json.dumps(flags)}", flush=True)
    ids = corpus_ids()
    lens = [int(x) for x in a.ctx.split(",")] + ([a.solo_ctx] if a.mode == "sampled" else [])
    prompts = []
    for i, n in enumerate(lens):
        s = random.Random("concq-rig-%d-%d" % (i, n)).randrange(0, len(ids) - n - 1)
        prompts.append(ids[s:s + n])
    pf = os.path.join(a.out, "prompts.json")
    json.dump(prompts, open(pf, "w"))
    k_set = sorted(int(x) for x in os.environ.get("GLM53_ADAPTIVE_K_SET", "4,5,7").split(",") if x.strip())
    sizes = {1, 2, 4, 8, 16, 24, 32}
    sizes.update(n * (k + 1) for n in range(1, 5) for k in k_set + [7])
    syn = os.environ.get("HANDOFF_SPEC_SYNTHETIC", "0.9,0.85,0.8,0.7,0.6,0.5,0.4")
    spec = {"method": "dflash", "model": a.model + "-dflash2", "num_speculative_tokens": 7, "kv_cache_dtype": "auto",
            "draft_sample_method": "probabilistic", "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [float(x) for x in syn.split(",")]}
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model, "--served-model-name", "mini",
           "--skip-tokenizer-init", "--tensor-parallel-size", "1", "--distributed-executor-backend", "mp",
           "--dtype", "bfloat16", "--max-model-len", str(a.max_model_len), "--max-num-seqs", "4",
           "--max-num-batched-tokens", "16384", "--enable-prefix-caching", "--kv-cache-dtype", "fp8",
           "--kv-cache-memory-bytes", str(a.kv_bytes), "--gpu-memory-utilization", str(a.gpu_util),
           "--speculative-config", json.dumps(spec), "--seed", "0", "--max-logprobs", "20",
           "--compilation-config", json.dumps({"cudagraph_capture_sizes": sorted(sizes)}),
           "--host", "127.0.0.1", "--port", str(a.port)]
    log = open(os.path.join(a.out, "server.log"), "w")
    print("serve_probe: starting", " ".join(cmd[:4]), "...", flush=True)
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd="/tmp")
    rc = 1
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                print(f"serve_probe: server exited rc {srv.returncode} during boot (server.log tail):", flush=True)
                os.system(f"tail -n 30 {os.path.join(a.out, 'server.log')}")
                return 5
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > a.boot_timeout:
                print("serve_probe: server not healthy in time", flush=True)
                return 6
            time.sleep(5)
        print(f"serve_probe: server healthy after {time.time() - t0:.0f}s", flush=True)
        p = subprocess.run([sys.executable, "/w/tools/prodcheck/conc_quality_probe.py", "--url", f"http://127.0.0.1:{a.port}",
                            "--no-auth", "--model", "mini", "--prompt-ids-file", pf, "--gen", str(a.gen), "--mode", a.mode,
                            "--positions", str(a.positions), "--deadline", str(a.deadline),
                            "--out", os.path.join(a.out, "concq.json")], capture_output=True, text=True)
        sys.stdout.write(p.stdout); sys.stdout.write(p.stderr[-3000:])
        rc = p.returncode
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try:
                srv.wait(90)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
                srv.wait(30)
        log.close()
        for line in open(os.path.join(a.out, "server.log"), errors="replace"):
            if "glm53-kpool-tail" in line or "Traceback" in line or "Error" in line[:200]:
                print("server.log:", line.rstrip()[:240])
    return rc


if __name__ == "__main__":
    sys.exit(main())
