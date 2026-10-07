"""Review: GLM53_MLA_PREFILL through the real plugin entry point (integrate.plugin_register), next to the other
production hooks, in fresh interpreters. Phases: off / on (AOT) / on but the extension is unavailable (JIT off)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

PROD_ENV = {
    "TF_EXL3_MOE": "1", "GLM53_PREFILL_FUSED_CAP": "1", "GLM53_MEM_HYGIENE": "1", "GLM53_FP8_GEMV": "1",
    "GLM53_FP8_GEMV_MAX_M": "16", "GLM53_BF16_GEMV": "1", "GLM53_BF16_GEMV_DEDUP_ROUTER": "1", "TF_EXL3_JIT": "0",
}
CHILD = r"""
import json, logging, os, sys
sys.path.insert(0, "/w")
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((r.levelname, r.name, r.getMessage()[:300]))
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.INFO)
for n in ("vllm", "vllm.glm53_mla_prefill"):
    logging.getLogger(n).addHandler(H())
if os.environ.get("HIDE_EXT"):
    sys.modules["glm53_mla_prefill_ext"] = None
if os.environ.get("IMPORT_VLLM_FIRST"):
    import vllm  # noqa
import integrate
integrate.plugin_register()
import glm53_mla_prefill as M
T = M.TARGET_MODULE
mod = sys.modules.get(T)
wrapped = bool(mod is not None and getattr(mod.FlashInferMLASparseSM90Impl.forward_mqa, "__glm53_mla_prefill__", False))
print("RESULT " + json.dumps({"installed": M.STATE.installed, "target_imported": mod is not None, "wrapped": wrapped,
      "variant": M.STATE.variant, "min_tokens": M.STATE.min_tokens, "mixed": M.STATE.mixed,
      "ext_loaded": M.STATE.ext is not None,
      "warnings": [r for r in recs if r[0] in ("WARNING", "ERROR")],
      "info": [r[2] for r in recs if r[0] == "INFO" and ("install" in r[2] or "loaded" in r[2])]}))
"""


def run(label, extra, drop=()):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(PROD_ENV)
    env.update(extra)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"=== {label}: no RESULT (rc {p.returncode})\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        return None
    r = json.loads(line[0][7:])
    print(f"=== {label}: installed={r['installed']} target_imported={r['target_imported']} wrapped={r['wrapped']} "
          f"ext_loaded={r['ext_loaded']} variant={r['variant']} min={r['min_tokens']} mixed={r['mixed']}")
    for l in (p.stdout + p.stderr).splitlines():
        if l.startswith("RESULT "):
            continue
        if any(k in l for k in ("glm53", "tf_exl3", "fp8_gemv", "WARNING", "Error", "error")):
            print("   |", l[:260])
    return r


def main() -> int:
    fails = []
    r = run("off (GLM53_MLA_PREFILL unset, other production hooks on)", {}, drop=("GLM53_MLA_PREFILL",))
    if r is None or r["installed"] or r["wrapped"] or r["ext_loaded"]:
        fails.append("off")
    r = run("on, AOT, plugin before vllm import", {"GLM53_MLA_PREFILL": "1"})
    if r is None or not (r["installed"] and r["wrapped"] and r["ext_loaded"]):
        fails.append("on")
    r = run("on, AOT, vllm imported first", {"GLM53_MLA_PREFILL": "1", "IMPORT_VLLM_FIRST": "1"})
    if r is None or not (r["installed"] and r["wrapped"]):
        fails.append("on-vllm-first")
    r = run("on, extension missing + JIT off", {"GLM53_MLA_PREFILL": "1", "HIDE_EXT": "1"})
    if r is None or r["wrapped"] or r["installed"]:
        fails.append("missing-ext")
    r = run("on, bad values", {"GLM53_MLA_PREFILL": "1", "GLM53_MLA_PREFILL_VARIANT": "7",
                               "GLM53_MLA_PREFILL_MIN_TOKENS": "abc", "GLM53_MLA_PREFILL_MIXED": "off"})
    if r is None or not (r["variant"] == 4 and r["min_tokens"] == 256 and r["mixed"] is False):
        fails.append("bad-values")
    r = run("typo value", {"GLM53_MLA_PREFILL": "enable"})
    if r is None or r["wrapped"]:
        fails.append("typo")
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
