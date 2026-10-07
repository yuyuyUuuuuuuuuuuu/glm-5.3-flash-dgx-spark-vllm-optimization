"""Counterfactual for the 2026-09-27 review: the new checks fail on the pre-review patches and pass on the current ones.

Host-side, CPU only (no GPU container): the pre-review patch scripts come from git commit a4d51fa, the stock target files
from the production image (docker run --rm --network none ... cat, read-only). Nothing outside a temp dir is written.
  B-aot-cache: the patched qwen3_dflash.py text must differ for GLM53_DRAFT_FP8 = off / 1 / fc / layers,fc
               (vLLM's compile caches are invalidated by traced source text, never by the variable).
  F3 / A-default-on: patch A must fail closed with GLM53_SPEC_RESAMPLE_INDEPENDENT unset (no silent default).
Exit non-zero unless the pre-review patches fail both checks and the current patches pass both.
"""
import contextlib
import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRE = "a4d51fa"
IMAGE = "ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor"
V = "/usr/local/lib/python3.12/dist-packages/vllm"
QD = "model_executor/models/qwen3_dflash.py"
RSU = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
tmp = Path(tempfile.mkdtemp(prefix="glm53_cf_"))
site = tmp / "site"
stock = {}
for rel in (QD, RSU):
    (site / rel).parent.mkdir(parents=True, exist_ok=True)
    stock[rel] = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", IMAGE, f"{V}/{rel}"],
                                check=True, capture_output=True).stdout
for name in ("patch_drafter_fp8.py", "patch_spec_resample_noise.py"):
    (tmp / f"pre_{name}").write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"{PRE}:overlay/{name}"],
                                                    check=True, capture_output=True).stdout)
os.environ["GLM53_SITE"] = str(site)


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"cf_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


res = {}
for label, b_path, a_path in (("pre-review " + PRE, tmp / "pre_patch_drafter_fp8.py", tmp / "pre_patch_spec_resample_noise.py"),
                              ("current", REPO / "overlay/patch_drafter_fp8.py", REPO / "overlay/patch_spec_resample_noise.py")):
    B = load("b" + label[:3], b_path)
    shas = {}
    for env in ("off", "1", "fc", "layers,fc"):
        (site / QD).write_bytes(stock[QD])
        os.environ["GLM53_DRAFT_FP8"] = env
        with contextlib.redirect_stdout(io.StringIO()):
            B.main(["x"])
        shas[env] = hashlib.sha256((site / QD).read_bytes()).hexdigest()[:12]
    A = load("a" + label[:3], a_path)
    (site / RSU).write_bytes(stock[RSU])
    os.environ.pop("GLM53_SPEC_RESAMPLE_INDEPENDENT", None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            A.main(["x"])
        a_out = "patched with the variable unset (silent default)"
        a_ok = False
    except SystemExit as e:
        a_out = f"fails closed: {e}"
        a_ok = True
    b_ok = len(set(shas.values())) == 4
    res[label] = (b_ok, a_ok)
    print(f"{label}:\n  B patched qwen3_dflash.py sha256 per GLM53_DRAFT_FP8 {shas} -> {len(set(shas.values()))} distinct"
          f"\n  A with GLM53_SPEC_RESAMPLE_INDEPENDENT unset: {a_out}")
shutil.rmtree(tmp, ignore_errors=True)
pre, cur = res["pre-review " + PRE], res["current"]
ok = pre == (False, False) and cur == (True, True)
print(f"pre-review patches fail both new checks: {pre == (False, False)}; current patches pass both: {cur == (True, True)}")
print("ALL PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
