"""[dec-hostloop] plugin wiring in fresh interpreters (production image):
  P1 env unset  -> integrate.plugin_register() leaves GPUModelRunner / the sm90 builder untouched (production path);
  P2 GLM53_DEC_HOSTLOOP=1 -> patched when a worker imports vllm.v1.worker.gpu.model_runner (import hook), and no
     CUDA context is created by the plugin or the hook (the API server / engine core load the plugin too);
  P3 GLM53_DEC_HOSTLOOP_METER=8 alone -> meter hooks only, fast path off;
  P4 a fingerprint mismatch -> WARNING, fast path NOT installed;
  P5 GLM53_DEC_HOSTLOOP_WAKE=12,0 alone -> prepare_inputs hook only (wake starts at the first real decode step, not
     at import: no spinner, no CUDA context), fast path off; P6 an unusable WAKE value -> WARNING, nothing patched.
Run: tests/hostloop_gpu.sh python3 tests/test_hostloop_plugin.py
"""
import os
import subprocess
import sys

CHILD = r'''
import os, sys, logging
sys.path.insert(0, "/w")
logging.basicConfig(level=logging.INFO)
import torch
import integrate
integrate.plugin_register()
ctx_plugin = torch.cuda.is_initialized()
mode = os.environ.get("CHILD_MODE", "")
if mode == "badfp":
    import glm53_hostloop as H
    H.VERIFIED_FINGERPRINTS["GPUModelRunner.prepare_inputs"] = frozenset({"0000000000000000"})
import vllm.v1.worker.gpu.model_runner as MR
import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as SM90
R = MR.GPUModelRunner
print("RESULT", dict(
    cuda_after_plugin=ctx_plugin,
    cuda_after_import=torch.cuda.is_initialized(),
    prep=getattr(R.prepare_inputs, "_glm53_hostloop", False),
    post=getattr(R.postprocess_sampled, "_glm53_hostloop", False),
    sample=getattr(R.sample_tokens, "_glm53_hostloop", False),
    klh=getattr(SM90.FlashInferMLASparseSM90Builder._kv_lens_host, "_glm53_hostloop", False),
    coll=getattr(__import__("vllm.distributed.parallel_state", fromlist=["x"]).GroupCoordinator._all_reduce_out_place,
                 "_glm53_hostloop", False),
    fwdhook=getattr(__import__("vllm.v1.worker.gpu.async_utils", fromlist=["x"]).StepTimingCollector.forward_start,
                    "_glm53_hostloop", False),
    fast=__import__("glm53_hostloop").ST.enabled,
    wake=__import__("glm53_hostloop")._WAKE.raw,
    wake_started=__import__("glm53_hostloop")._WAKE.started))
'''


def run(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_DEC_HOSTLOOP")}
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
    if p.returncode or not line:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"child failed rc={p.returncode}")
    res = eval(line[-1][len("RESULT "):])
    warn = [x for x in (p.stdout + p.stderr).splitlines() if "glm53-hostloop" in x]
    return res, warn


fails = []
r, w = run({})
print("P1 unset:", r)
base_cuda = r["cuda_after_import"]          # what importing vLLM's model runner does on its own
if any(r[k] for k in ("prep", "post", "sample", "klh", "fast", "coll", "fwdhook")) or r["cuda_after_plugin"]:
    fails.append("P1")
r, w = run({"GLM53_DEC_HOSTLOOP": "1"})
print("P2 on:", r, w[-1:] if w else "")
if (not (r["prep"] and r["post"] and r["klh"] and r["fast"]) or r["sample"] or r["coll"] or r["cuda_after_plugin"]
        or r["cuda_after_import"] != base_cuda):
    fails.append("P2")
r, w = run({"GLM53_DEC_HOSTLOOP_METER": "8"})
print("P3 meter only:", r)
if not (r["prep"] and r["sample"] and r["coll"] and r["fwdhook"]) or r["post"] or r["klh"] or r["fast"]:
    fails.append("P3")
r, w = run({"GLM53_DEC_HOSTLOOP": "1", "CHILD_MODE": "badfp"})
print("P4 bad fingerprint:", r, [x[-160:] for x in w if "NOT installed" in x])
if r["prep"] or r["post"] or r["klh"] or r["fast"] or not any("NOT installed" in x for x in w):
    fails.append("P4")
r, w = run({"GLM53_DEC_HOSTLOOP_WAKE": "12,0"})
print("P5 wake only:", r)
if (not r["prep"] or r["post"] or r["klh"] or r["fast"] or r["sample"] or r["coll"] or r["wake"] != "12,0"
        or r["wake_started"] or r["cuda_after_plugin"]):
    fails.append("P5")
r, w = run({"GLM53_DEC_HOSTLOOP_WAKE": "x,y"})
print("P6 bad wake value:", r, [x[-120:] for x in w if "unusable" in x])
if r["prep"] or r["wake"] or not any("unusable" in x for x in w):
    fails.append("P6")
print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
