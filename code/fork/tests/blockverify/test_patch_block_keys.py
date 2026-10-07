#!/usr/bin/env python3
"""Host-only checks of overlay/patch_spec_block_keys.py and its bundle wiring (text only; no GPU, no vLLM import).

Sources: the image's two files as extracted in $TF_EXL3_ASSETS/vllm-src/vllm (sha256 pinned below).
  1. GLM53_REJECTION_METHOD: unset / empty / bogus fail closed and write nothing; standard writes nothing
  2. block without the resample-noise patch fails closed (both files untouched)
  3. block after the resample-noise patch: both files patched, compile, second run is a no-op (idempotent)
  4. drifted anchor / partially patched file / marker outside the edits fail closed, before either file is written
  5. the patched text: BLOCK_KEYS is a constexpr (standard compiles the stock code), both launches pass
     BLOCK_KEYS=use_block_verification, the drafter's switch comes from rejection_sample_method, the resample-noise
     patch's four anchors stay contiguous (re-running it is a no-op) and its "not exact" warning is marked emitted
  6. patch_tf_bundle.py (the fork's and launcher/overlay's copy are identical) runs the block-keys patch after the
     resample-noise patch, only when GLM53_REJECTION_METHOD is non-empty, and stops the container start on failure
Usage: python3 tests/blockverify/test_patch_block_keys.py   (exit 1 on any failure)
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("VLLM_SRC", os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "vllm-src/vllm")))
RSU = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
SPEC = "v1/worker/gpu/spec_decode/dflash2/speculator.py"
SHA = {RSU: "659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a",
       SPEC: "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48"}
FAIL: list[str] = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


for rel, sha in SHA.items():
    check(hashlib.sha256((SRC / rel).read_bytes()).hexdigest() == sha, f"image source {rel} is the pinned file")

tmp = Path(tempfile.mkdtemp(prefix="bk_"))


def fresh_site(name):
    site = tmp / name
    for rel in (RSU, SPEC):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, site / rel)
    return site


def snapshot(site):
    return {rel: (site / rel).read_text() for rel in (RSU, SPEC)}


def run_patch(path, site, env):
    """Run a patch module's main() the way patch_tf_bundle does, with GLM53_SITE and env overrides."""
    old = dict(os.environ)
    try:
        os.environ["GLM53_SITE"] = str(site)
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mod = load(f"m{abs(hash((str(path), str(site), str(sorted(env.items())))))}", path)
        try:
            rc = mod.main(["x"])
            return rc, None
        except SystemExit as e:
            return None, str(e)
    finally:
        os.environ.clear()
        os.environ.update(old)


BK = REPO / "overlay/patch_spec_block_keys.py"
RN = REPO / "overlay/patch_spec_resample_noise.py"

# 1
site = fresh_site("s1")
before = snapshot(site)
for val in (None, "", "bogus", "BLOCK", " block2"):
    rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": val})
    check(rc is None and err and "must be 'standard' or 'block'" in err and snapshot(site) == before,
          f"GLM53_REJECTION_METHOD={val!r}: fails closed, nothing written")
rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": "standard"})
check(rc == 0 and snapshot(site) == before, "GLM53_REJECTION_METHOD=standard: exit 0, nothing written")
# 2
rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": "block"})
check(rc is None and "run patch_spec_resample_noise.py first" in (err or "") and snapshot(site) == before,
      "block without the resample-noise patch: fails closed, both files untouched")
# 3
rc, err = run_patch(RN, site, {"GLM53_SPEC_RESAMPLE_INDEPENDENT": "1"})
check(rc == 0, "resample-noise patch applied")
mid = snapshot(site)
rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": "block"})
after = snapshot(site)
check(rc == 0 and after[RSU] != mid[RSU] and after[SPEC] != mid[SPEC], "block: both files patched")
for rel in (RSU, SPEC):
    try:
        compile(after[rel], rel, "exec")
        ok = True
    except SyntaxError:
        ok = False
    check(ok, f"patched {rel} compiles")
rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": "block"})
check(rc == 0 and snapshot(site) == after, "second block run: no-op (idempotent)")
rc, err = run_patch(RN, site, {"GLM53_SPEC_RESAMPLE_INDEPENDENT": "1"})
check(rc == 0 and snapshot(site) == after, "resample-noise patch re-run on the block-keys file: 'already present', "
      f"nothing written (the overlay chain can be re-run) ({err})")
rc, err = run_patch(BK, site, {"GLM53_REJECTION_METHOD": "block"})
check(rc == 0 and snapshot(site) == after, "whole chain re-run (resample-noise, block-keys): no-op")
# 4
bk = load("bk_anchor", BK)
for label, rel, mutate in (
        ("drifted rejection-kernel anchor", RSU, lambda s: s.replace(bk.R3_OLD, bk.R3_OLD.replace("u = tl_rand32", "u  = tl_rand32"))),
        ("drifted walk anchor", SPEC, lambda s: s.replace(bk.S2_OLD, bk.S2_OLD.replace("- 1", "-1"))),
        ("partially patched verifier", RSU, lambda s: s.replace(bk.R6_OLD, bk.R6_NEW)),
        ("marker outside the edits", SPEC, lambda s: s + "\n# [glm53-block-keys] stray\n")):
    s4 = fresh_site("s4_" + label.replace(" ", "_"))
    run_patch(RN, s4, {"GLM53_SPEC_RESAMPLE_INDEPENDENT": "1"})
    (s4 / rel).write_text(mutate((s4 / rel).read_text()))
    b4 = snapshot(s4)
    rc, err = run_patch(BK, s4, {"GLM53_REJECTION_METHOD": "block"})
    check(rc is None and "preflight failed" in (err or "") and snapshot(s4) == b4, f"{label}: fails closed, neither file written ({err})")
# 5
r, sp = after[RSU], after[SPEC]
check(r.count("    BLOCK_KEYS: tl.constexpr = False,  # [glm53-block-keys]\n") == 1
      and r.count("    BLOCK_KEYS: tl.constexpr,  # [glm53-block-keys]") == 1,
      "verifier: _rejection_kernel takes BLOCK_KEYS (constexpr, default False), _resample_kernel takes it (constexpr, "
      "passed by the only launch)")
rn = load("rn_anchor", RN)
check(all(r.count(new_) == 1 for _l, _o, new_ in rn.EDITS), "verifier: all four patched anchors of the resample-noise "
      "patch are still contiguous")
check("\n_GLM53_BLOCK_WARNED = True\n" in r and r.index("_GLM53_BLOCK_WARNED = False") < r.index("_GLM53_BLOCK_WARNED = True"),
      "verifier: the resample patch's 'block is not exact' warning is marked as emitted (it no longer applies)")
check(r.count("BLOCK_KEYS=use_block_verification,  # [glm53-block-keys]") == 2, "verifier: both launches pass BLOCK_KEYS=use_block_verification")
check("pos = pos + (tl.cast(i, tl.int64) << 32)" in r and "(tl.cast(resample_idx, tl.int64) << 32)" in r,
      "verifier: the row index goes into the Philox counter's high word (u draw and resample/bonus key)")
check("    if next_local_pos == 0:\n        return\n" in r and "has_next_row = logit_idx + 1 < tl.num_programs(0)\n" in r
      and "num_logits,  # [glm53-block-keys]" not in r,
      "verifier: the residual-mass kernel (block only) returns for the last row of a request; its bound is the grid "
      "(tl.num_programs(0)), not an integer argument (Triton would recompile per batch-shape class)")
check("    if use_block_verification:  # [glm53-block-keys]\n        _glm53_log_block_keys()\n" in r,
      "verifier: the block-keys marker is logged once in block mode")
check("BLOCK_KEYS: tl.constexpr = False,  # [glm53-block-keys]" in sp and "BLOCK_KEYS=self._glm53_block_keys," in sp
      and 'getattr(self.speculative_config, "rejection_sample_method", "standard") == "block"' in sp,
      "drafter: walk kernel constexpr defaults to False; the switch is the speculative config's rejection_sample_method")
check("position = position + (tl.cast(step, tl.int64) << 32)" in sp, "drafter: draft index goes into the counter's high word")
# 6
tb = (REPO / "overlay/patch_tf_bundle.py").read_text()
check(tb == (REPO / "launcher/overlay/patch_tf_bundle.py").read_text(), "patch_tf_bundle.py: fork overlay == launcher/overlay copy")
i_rn, i_bk = tb.index('("patch_spec_resample_noise.py"'), tb.index('("patch_spec_block_keys.py", "GLM53_REJECTION_METHOD")')
check(i_rn < i_bk, "patch_tf_bundle.py runs the block-keys patch after the resample-noise patch")
# run the real bundle script on a scratch OPT / SITEPKG / GLM53_SITE
opt = tmp / "opt"
(opt / "tf/site").mkdir(parents=True)
(opt / "tf/overlay").mkdir(parents=True)
for f in ("patch_spec_resample_noise.py", "patch_spec_block_keys.py"):
    shutil.copy(REPO / "overlay" / f, opt / "tf/overlay" / f)
(opt / "tf/site/marker.txt").write_text("x")
sitepkg = tmp / "sitepkg"
sitepkg.mkdir()
for label, env, want_rc, want in (
        ("unset", {"GLM53_REJECTION_METHOD": None}, 0, "patch_spec_block_keys.py: GLM53_REJECTION_METHOD unset -> skipped"),
        ("standard", {"GLM53_REJECTION_METHOD": "standard"}, 0, "not needed, files untouched"),
        ("block", {"GLM53_REJECTION_METHOD": "block"}, 0, "block verification with row-keyed randomness"),
        ("bogus", {"GLM53_REJECTION_METHOD": "blok"}, 1, "must be 'standard' or 'block'")):
    s6 = fresh_site("s6_" + label)
    import subprocess
    e = dict(os.environ, GLM53_OPT=str(opt), GLM53_SITEPKG=str(sitepkg), GLM53_SITE=str(s6),
             GLM53_SPEC_RESAMPLE_INDEPENDENT="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ("GLM53_DRAFT_FP8", "GLM53_DRAFT_LMHEAD_FP8", "GLM53_KPOOL_SEED_STRIDE", "GLM53_KPOOL_RING", "GLM53_REJECTION_METHOD"):
        e.pop(k, None)
    for k, v in env.items():
        if v is not None:
            e[k] = v
    p = subprocess.run([sys.executable, str(REPO / "overlay/patch_tf_bundle.py")], env=e, capture_output=True, text=True)
    txt = p.stdout + p.stderr
    patched = "[glm53-block-keys]" in (s6 / SPEC).read_text()
    check((p.returncode == 0) == (want_rc == 0) and want in txt and patched == (label == "block"),
          f"patch_tf_bundle.py with GLM53_REJECTION_METHOD {label}: rc={p.returncode}, drafter patched={patched}")
shutil.rmtree(tmp, ignore_errors=True)
print("ALL PASSED" if not FAIL else f"FAILED: {len(FAIL)}")
sys.exit(1 if FAIL else 0)
