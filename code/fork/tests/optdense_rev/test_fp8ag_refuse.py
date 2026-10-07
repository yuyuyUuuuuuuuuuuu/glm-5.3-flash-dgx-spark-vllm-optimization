"""opt-dense-rev: a refused W8A8 install with GLM53_DENSE_W8A8_FP8AG=1 says so at ERROR (the paired-collective hang
warning) and reports fp8ag_refused; without FP8AG the refusal report is unchanged. No GPU work.
Run: tests/gpu_run.sh python3 tests/optdense_rev/test_fp8ag_refuse.py"""
import logging
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fp8_w8a8 as W  # noqa: E402

recs = []
h = logging.Handler()
h.emit = lambda r: recs.append(r)
logging.getLogger("vllm.tf_fp8_w8a8").addHandler(h)
fake = types.ModuleType("fakeprod")            # no Glm53DenseFp8Method -> refused before any CUDA work
os.environ["GLM53_DENSE_W8A8"] = "1"
os.environ["GLM53_DENSE_W8A8_FP8AG"] = "1"
r1 = W.install(fake)
e1 = [x for x in recs if x.levelno >= logging.ERROR]
os.environ.pop("GLM53_DENSE_W8A8_FP8AG")
recs.clear()
r2 = W.install(fake)
e2 = [x for x in recs if x.levelno >= logging.ERROR]
ok = (not r1["installed"] and r1.get("fp8ag_refused") and len(e1) == 1 and "HANGS" in e1[0].getMessage()
      and not r2["installed"] and "fp8ag_refused" not in r2 and not e2)
print("r1", r1, "\nr2", r2, "\nerror lines", [x.getMessage()[:120] for x in e1])
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
