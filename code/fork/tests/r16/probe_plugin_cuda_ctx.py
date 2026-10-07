"""Which flag makes plugin loading create a CUDA context? Fresh interpreter per flag set, real plugin loader.
Each R16 flag alone and each R15 flag alone (no other GLM53_* / TF_EXL3_* variable), then on top of production's R15 env."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
R15 = {"TF_EXL3_MOE": "1", "GLM53_PREFILL_FUSED_CAP": "1", "GLM53_MEM_HYGIENE": "1", "GLM53_FP8_GEMV": "1",
       "GLM53_FP8_GEMV_MAX_M": "16", "GLM53_BF16_GEMV": "1", "GLM53_BF16_GEMV_DEDUP_ROUTER": "1", "GLM53_FP8_LARGE_M": "1",
       "GLM53_LMHEAD_FP8": "1", "GLM53_TF_PROFILE": "/tmp/glm53-prof", "TF_EXL3_JIT": "0"}
R16 = {"GLM53_PREFILL_QUICKWINS": "all", "GLM53_MLA_PREFILL": "1", "GLM53_DEC_FP8ROOF": "1", "GLM53_DEC_MOEGLUE_WARM": "1",
       "GLM53_DEC_HOSTLOOP": "1", "GLM53_DEC_SMALLOPS": "1", "GLM53_DEC_SMALLOPS_KINDS": "dconv"}
CHILD = r"""
import importlib, os, torch
for m in [x for x in os.environ.get("PRE_IMPORT", "").split(",") if x]:
    importlib.import_module(m)
from vllm.plugins import load_general_plugins
load_general_plugins()
print("RESULT", torch.cuda.is_initialized())
"""
d = Path(tempfile.mkdtemp()); di = d / "tf_exl3_moe-0.1.0.dist-info"; di.mkdir()
(di / "METADATA").write_text("Metadata-Version: 2.1\nName: tf_exl3_moe\nVersion: 0.1.0\n")
(di / "entry_points.txt").write_text("[vllm.general_plugins]\ntf_exl3_moe = integrate:plugin_register\n")
ALONE = [(f"{k} alone", {k: v, **({"GLM53_DEC_SMALLOPS_KINDS": "dconv"} if k == "GLM53_DEC_SMALLOPS" else {}), "TF_EXL3_JIT": "0"})
         for k, v in R16.items() if k != "GLM53_DEC_SMALLOPS_KINDS"] + [("r16 alone", {**R16, "TF_EXL3_JIT": "0"})]
ALONE += [(f"{k} alone (R15)", {k: v, "TF_EXL3_JIT": "0"}) for k, v in R15.items() if k != "TF_EXL3_JIT"]
sets = [("none", {}), ("r15", R15)] + ALONE + [(f"r15+{k}", {**R15, k: v, **({"GLM53_DEC_SMALLOPS_KINDS": "dconv"} if k == "GLM53_DEC_SMALLOPS" else {})}) for k, v in R16.items() if k != "GLM53_DEC_SMALLOPS_KINDS"] + [("r15+r16", {**R15, **R16})]
for pre in ("", "vllm.v1.engine.core"):
    for name, envs in sets:
        env = {k: v for k, v in os.environ.items() if not (k.startswith("GLM53_") or k.startswith("TF_EXL3"))}
        env.update(envs); env["PRE_IMPORT"] = pre
        env["PYTHONPATH"] = f"{d}:/w:" + env.get("PYTHONPATH", "")
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300, cwd="/tmp")
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        print(f"pre={pre or '-':22s} {name:32s} cuda_ctx_after_plugins={line[0][7:] if line else 'rc=%d %s' % (p.returncode, p.stderr[-300:])}", flush=True)
