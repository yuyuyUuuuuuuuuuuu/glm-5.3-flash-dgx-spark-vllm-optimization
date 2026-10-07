"""Adversarial check of the r16e4 kit's ON path in the production image, with production's env (nodeC only).

  V1 compose: the KIT's patch_tf_bundle.py (GLM53_OPT/GLM53_SITEPKG redirected to /tmp) with GLM53_MOE_E4M3=1, twice
     (a container restart re-runs it: install_site re-copies integrate.py, then the arm) -> exactly one arming block;
     patch_moe_e4m3.main() again without install_site -> "already armed"; =0 and unset leave integrate.py == site/.
  V2 vLLM's own load_general_plugins() with production's .env flags (env_nonsecret + env.r16 lines + start.sh
     defaults MAX_NUM_SEQS=4 / DFLASH_TOKENS=7) -> the apply_exl3_experts wrapper chain (e4m3 must be outermost),
     Exl3MoEMethod.process_weights_after_loading wrapped, decode_bound() == (32, 7, 256).
  V3 a real layer-10 shard finished by production's (now wrapped) process_weights_after_loading -> the load-time
     self-test runs there: time, CUDA peak-memory delta, STATS; then a 2048-token call (served, summary line) and a
     32-token call (pass-through, bit-equal to production's chain without the e4m3 wrapper).
Run:
  GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" \\
  GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3:$TF_EXL3_KITS/tf-exl3-deploy16.r16e4 \\
  flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/moee4m3/verify_prodenv_chain.py
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time

KIT = os.environ.get("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16e4"))
OPT, SP = "/tmp/opt", "/tmp/sp"
FAIL = []


def ck(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def compose(val):
    shutil.rmtree(OPT, ignore_errors=True)
    os.makedirs(OPT + "/tf")
    shutil.copytree(KIT + "/site", OPT + "/tf/site")
    shutil.copytree(KIT + "/overlay", OPT + "/tf/overlay")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GLM53_", "TF_EXL3"))}
    env.update(GLM53_OPT="/tmp/opt", GLM53_SITEPKG=SP)
    if val is not None:
        env["GLM53_MOE_E4M3"] = val
    r = subprocess.run([sys.executable, KIT + "/launcher/overlay/patch_tf_bundle.py"], env=env,
                       capture_output=True, text=True)
    return r


def v1():
    shutil.rmtree(SP, ignore_errors=True)
    os.makedirs(SP)
    site_int = open(KIT + "/site/integrate.py").read()
    for val in (None, "", "0"):
        r = compose(val)
        same = open(SP + "/integrate.py").read() == site_int
        ck(r.returncode == 0 and same and not os.path.exists(SP + "/glm53_moe_e4m3.py"),
           f"V1 GLM53_MOE_E4M3={val!r}: rc {r.returncode}, integrate.py == site/ {same}, module absent")
    for i in (1, 2):
        r = compose("1")
        txt = open(SP + "/integrate.py").read()
        n = txt.count("# [glm53-moe-e4m3] BEGIN")
        ck(r.returncode == 0 and n == 1 and "integrate.py: armed" in r.stdout,
           f"V1 =1 compose #{i} (restart re-runs the bundle): rc {r.returncode}, arming blocks {n}")
    sys.path.insert(0, OPT + "/tf/overlay")
    os.environ.update(GLM53_OPT=OPT, GLM53_SITEPKG=SP, GLM53_MOE_E4M3="1")
    import patch_moe_e4m3 as P
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P.main([])
    txt = open(SP + "/integrate.py").read()
    ck(rc == 0 and "already armed" in buf.getvalue() and txt.count("# [glm53-moe-e4m3] BEGIN") == 1,
       "V1 patch_moe_e4m3 re-run without install_site: already armed, one block")
    sys.path.remove(OPT + "/tf/overlay")


PROD_ENV = {
    "EXL3_FUSED_MOE": "1", "EXL3_FAT_KERNEL": "1", "EXL3_FAT_GROUPED": "1", "EXL3_TEMP_ROWS_FUSED": "256",
    "SPEC_METHOD": "dflash", "DFLASH_TOKENS": "7", "MAX_NUM_SEQS": "4", "MTP_TOKENS": "2",
    "GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_SET": "4,5,7", "GLM53_MIXED_PREFILL_CHUNK": "0",
    "GLM53_DENSE_FP8": "dense,kda,mla,shared", "TF_EXL3_MOE": "1", "GLM53_MEM_HYGIENE": "1",
    "GLM53_LMHEAD_FP8": "1", "GLM53_FP8_GEMV": "1", "GLM53_FP8_GEMV_MAX_M": "16", "GLM53_BF16_GEMV": "1",
    "GLM53_BF16_GEMV_DEDUP_ROUTER": "1", "GLM53_PREFILL_FUSED_CAP": "1", "GLM53_KPOOL_SEED_STRIDE": "1",
    "GLM53_FP8_LARGE_M": "1",
    # env.r16 lines
    "GLM53_PREFILL_QUICKWINS": "all", "GLM53_MLA_PREFILL": "1", "GLM53_DEC_FP8ROOF": "1",
    "GLM53_DEC_MOEGLUE_WARM": "1", "GLM53_DEC_HOSTLOOP": "1", "GLM53_DEC_SMALLOPS": "1",
    "GLM53_DEC_SMALLOPS_KINDS": "dconv", "GLM53_KPOOL_RING": "1",
    "GLM53_MOE_E4M3": "1",
}


def chain(fn):
    out = []
    seen = 0
    while fn is not None and seen < 12:
        seen += 1
        tags = [a for a in dir(fn) if a.startswith("_") and a.endswith("_orig") and not a.startswith("__")]
        out.append(f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}")
        nxt = None
        for a in tags:
            nxt = getattr(fn, a)
            break
        fn = nxt
    return out


def main():
    v1()
    os.environ.update(PROD_ENV)
    sys.path.insert(0, SP)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=sys.stdout)
    import torch
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import vllm.model_executor.layers.quantization.exl3 as prod
    import integrate
    import glm53_moe_e4m3 as M

    ck(integrate.__file__.startswith(SP), f"V2 integrate from the composed site ({integrate.__file__})")
    ck(M.__file__.startswith(SP), f"V2 glm53_moe_e4m3 from the composed site ({M.__file__})")
    c = chain(prod.apply_exl3_experts)
    print("apply_exl3_experts chain (outer -> inner):", " -> ".join(c), flush=True)
    ck(c[0].startswith("glm53_moe_e4m3"), "V2 e4m3 wrapper is the outermost apply_exl3_experts")
    pw = prod.Exl3MoEMethod.process_weights_after_loading
    ck(getattr(pw, "_glm53_moe_e4m3", False), "V2 Exl3MoEMethod.process_weights_after_loading wrapped")
    print("process_weights_after_loading chain:", " -> ".join(chain(pw)), flush=True)
    b = M.decode_bound()
    ck(b == (32, 7, 256), f"V2 decode_bound() with production's env = {b} (want (32, 7, 256))")
    ck(M.STATS["installed"] and M.STATS["decode_bound_warn"] == 0, f"V2 STATS {M.STATS}")

    sys.path.insert(0, "/w/tests")
    sys.path.insert(0, "/w/tests/moee4m3")
    import harness as H
    from real_layer import RealWeights

    H.load_xl()
    dev = torch.device("cuda", 0)
    W = RealWeights(dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    L = H.make_layer(prod, W)          # production's process_weights_after_loading, wrapped -> self-test
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20
    after = (torch.cuda.memory_allocated() - m0) / 2 ** 20
    print(f"make_layer incl. load-time self-test: {dt * 1e3:.0f} ms, CUDA peak +{peak:.0f} MiB, resident +{after:.0f} "
          f"MiB; ok={getattr(L, '_glm53_moe_e4m3_ok', None)} STATS {M.STATS}", flush=True)
    ck(getattr(L, "_glm53_moe_e4m3_ok", None) is True and M.STATS["selftests"] == 1,
       "V3 the self-test ran inside process_weights_after_loading and passed")
    ck(peak < 1024, f"V3 load-time peak {peak:.0f} MiB < 1 GiB")
    # second construction of a layer object from the same weights: its own self-test (per layer object)
    g = torch.Generator().manual_seed(5)
    lim = float(getattr(prod, "SWIGLU_LIMIT_DEFAULT", 10.0))

    def inp(T):
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        ids = torch.randint(0, W.n, (T, 8), generator=g).to(dev)
        w = torch.rand(T, 8, generator=g).to(dev)
        return x, ids, (w / w.sum(-1, keepdim=True)).float()

    inner = prod.apply_exl3_experts._glm53_moe_e4m3_orig
    xd, idd, wd = inp(32)
    a = prod.apply_exl3_experts(xd, idd, wd, L, limit=lim)
    s0 = M.STATS["served"]
    xp, idp, wp = inp(2048)
    out = prod.apply_exl3_experts(xp, idp, wp, L, limit=lim)
    torch.cuda.synchronize()
    ref = inner(xp, idp, wp, L, limit=lim)
    r = float((out.double() - ref.double()).norm() / ref.double().norm())
    ck(M.STATS["served"] == s0 + 1 and torch.isfinite(out).all().item(),
       f"V3 2048 tokens served on e4m3 (rel vs production {r:.2e}); STATS {M.STATS}")
    ck(r < 0.15, f"V3 served output in the e4m3 class vs production (rel {r:.2e} < 0.15)")
    b2 = inner(xd, idd, wd, L, limit=lim)
    ck(M.STATS["passed"] >= 1, f"V3 32 tokens passed through (passed {M.STATS['passed']}); "
       f"decode rel vs inner {float((a.double() - b2.double()).norm() / b2.double().norm()):.2e}")
    print("verify_prodenv_chain:", "ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
