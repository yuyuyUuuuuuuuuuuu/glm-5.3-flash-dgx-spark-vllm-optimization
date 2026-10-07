"""Review: GLM53_MLA_PREFILL through vLLM's REAL plugin loader (vllm.plugins.load_general_plugins -> entry point
tf_exl3_moe = integrate:plugin_register, as the bundle's dist-info registers it), in fresh interpreters that first
import what each production process has imported at that point. Checks: the backend class vLLM resolves through its
registry is the wrapped one, no CUDA context is created by plugin loading (API server / engine core must stay
context-free), and the other production hooks still install."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROD_ENV = {
    "TF_EXL3_MOE": "1", "GLM53_PREFILL_FUSED_CAP": "1", "GLM53_MEM_HYGIENE": "1", "GLM53_FP8_GEMV": "1",
    "GLM53_FP8_GEMV_MAX_M": "16", "GLM53_BF16_GEMV": "1", "GLM53_BF16_GEMV_DEDUP_ROUTER": "1", "TF_EXL3_JIT": "0",
    "GLM53_MLA_PREFILL": "1",
}
CHILD = r"""
import importlib, json, logging, os, sys
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((r.levelname, r.getMessage()[:240]))
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.INFO)
logging.getLogger("vllm").addHandler(H())
pre = os.environ.get("PRE_IMPORT", "")
for m in [x for x in pre.split(",") if x]:
    importlib.import_module(m)
import torch
from vllm.plugins import load_general_plugins
load_general_plugins()
ctx_after_plugins = torch.cuda.is_initialized()
from vllm.v1.attention.backends.registry import AttentionBackendEnum
cls = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90.get_class()
impl = cls.get_impl_cls()
import glm53_mla_prefill as M
print("RESULT " + json.dumps({"cuda_ctx_after_plugins": ctx_after_plugins,
      "registry_impl_wrapped": bool(getattr(impl.forward_mqa, "__glm53_mla_prefill__", False)),
      "installed": M.STATE.installed, "ext": M.STATE.ext is not None,
      "hooks": sorted({r[1].split(" ")[0] for r in recs if r[0] == "INFO" and "install" in r[1]}),
      "warnings": [r[1] for r in recs if r[0] in ("WARNING", "ERROR")]}))
"""


def main() -> int:
    d = Path(tempfile.mkdtemp())
    di = d / "tf_exl3_moe-0.1.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: tf_exl3_moe\nVersion: 0.1.0\n")
    (di / "entry_points.txt").write_text("[vllm.general_plugins]\ntf_exl3_moe = integrate:plugin_register\n")
    fails = []
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = dict(PROD_ENV)
    if mode == "nomla":
        base.pop("GLM53_MLA_PREFILL")
    elif mode == "mlaonly":
        base = {"GLM53_MLA_PREFILL": "1", "TF_EXL3_JIT": "0"}
    elif mode == "none":
        base = {"TF_EXL3_JIT": "0"}
    print(f"### mode {mode}: {sorted(base)}", flush=True)
    for label, pre in (("fresh (plugin loader first)", ""),
                       ("API server (arg_utils imported)", "vllm.engine.arg_utils"),
                       ("engine core imported", "vllm.v1.engine.core"),
                       ("GPU worker imported", "vllm.v1.worker.gpu_worker")):
        env = dict(os.environ)
        for k in PROD_ENV:
            env.pop(k, None)
        env.update(base)
        env["PRE_IMPORT"] = pre
        env["PYTHONPATH"] = f"{d}:/w:" + env.get("PYTHONPATH", "")
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=240,
                           cwd="/tmp")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"=== {label}: no RESULT rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
            fails.append(label)
            continue
        r = json.loads(line[0][7:])
        want = "GLM53_MLA_PREFILL" in base
        ok = (r["registry_impl_wrapped"] == want) and (r["installed"] == want) and not r["cuda_ctx_after_plugins"]
        print(f"=== {label}: {'PASS' if ok else 'FAIL'} {json.dumps(r)}", flush=True)
        if not ok:
            fails.append(label)
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
