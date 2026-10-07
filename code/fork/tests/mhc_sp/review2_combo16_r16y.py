"""The combined kit's FOUR feature switches in every one of the 16 combinations, in one bundle pass, plus a second
pass (idempotence) - the production overlay chain including kdaqkv (GLM53_KDA_STRIDED_QKV=1, as production runs).

For each (GLM53_DENSE_W8A8, GLM53_KDA_FLASHKDA, GLM53_MHC_SP, GLM53_MOE_E4M3) in {0,1}^4: run each gated patch of
the bundle chain in the bundle's own order (patch_kda_strided_qkv.py =1 always, then patch_dense_w8a8.py,
patch_flashkda.py, patch_mhc_sp.py and patch_moe_e4m3.py last, per patch_tf_bundle.py's PATCHES tuple) with its
switch ("1" on, "" = the empty string the launcher forwards when unset) from the combined kit's overlay/ on a fresh
fake site-packages tree; require rc 0 everywhere, a SECOND pass rc 0 and byte-identical, and the tree deltas to
compose exactly:
  off-tree (0,0,0,0) == the kdaqkv-only tree (== the previous kit's composed tree; off_equals_prev proves it
             against r16n on the real fs),
  delta(W8A8=1) = {integrate.py (the armed block), fp8_w8a8.py, tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so},
  delta(FK=1)   = {NV/kda.py, quickwins, glm53_flashkda.py, _flashkda_fp32_C.abi3.so},
  delta(SP=1)   = {NV/model.py, quickwins, moeglue},
  delta(E4=1)   = {integrate.py, glm53_moe_e4m3.py, glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so},
  tree(1,1,1,1) - tree(0,0,0,0) = the union of the four sets (the patches interfere with nothing; the two
             integrate.py armings compose: w8a8 inserts after the fp8_roof step, e4m3 appends at the end).

r16y integration note: the w8a8 branch carried its fp8_w8a8 conditional import in the bundle site integrate.py
itself; the kit ships r16n's integrate.py byte for byte and patch_dense_w8a8.py arms it at compose time instead
(=0 disarms), so the off-tree here is byte-identical to the previous kit's and the armed block participates in the
delta exactly like e4m3's.

Host-only. Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16y IMG_VLLM=$TF_EXL3_ASSETS/img_vllm_glm5next_r16x \
  python3 tests/mhc_sp/review2_combo16_r16y.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16y"))
import review2_compose3_r16x as R  # build()/run()/snapshot()/PATCHES  (KIT env override must precede the import)

FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


EXPECTED = {
    "w8a8": {"integrate.py", "fp8_w8a8.py", "tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so"},
    "flashkda": {R.NV + "/kda.py", "glm53_prefill_quickwins.py",
                 "glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"},
    "mhcsp": {R.NV + "/model.py", "glm53_prefill_quickwins.py", "glm53_moeglue.py"},
    "e4m3": {"integrate.py", "glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"},
}
# the bundle's own PATCHES order (patch_tf_bundle.py): kdaqkv first, dense_w8a8, flashkda, ..., moe_e4m3 LAST
LABELS = ("w8a8", "flashkda", "mhcsp", "e4m3")
NAMES = {"w8a8": "GLM53_DENSE_W8A8", "flashkda": "GLM53_KDA_FLASHKDA", "mhcsp": "GLM53_MHC_SP",
         "e4m3": "GLM53_MOE_E4M3"}
NAMES.update({"kdaqkv": R.PATCHES["kdaqkv"][1]})
FILES = {"w8a8": "patch_dense_w8a8.py", "flashkda": "patch_flashkda.py", "mhcsp": "patch_mhc_sp.py",
         "e4m3": "patch_moe_e4m3.py"}
FILES.update({"kdaqkv": R.PATCHES["kdaqkv"][0]})


def run(sp, ov, name, val):
    env = dict(os.environ, GLM53_SITEPKG=str(sp), GLM53_TF_OVERLAY=str(ov), GLM53_OPT=str(ov.parent.parent),
               PYTHONDONTWRITEBYTECODE="1", **{NAMES[name]: val})
    for k in ("GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY", "GLM53_MOEGLUE_PY", "GLM53_KDA_PY",
              "GLM53_FLA_KDA_PY", "GLM53_FLA_FUSED_RECURRENT_PY"):
        env.pop(k, None)
    import subprocess
    p = subprocess.run([sys.executable, str(ov / FILES[name])], env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def combo_tree(on: tuple[int, int, int, int]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        sp, ov = R.build(Path(td))
        # the bundle skips a patch whose switch is empty/unset WITHOUT running the patch script
        # (patch_tf_bundle.run_patch), so combo_tree runs exactly what the bundle would run, in the bundle's order
        seq = [("kdaqkv", "1")] + [(n, "1") for i, n in enumerate(LABELS) if on[i]]
        for name, val in seq:
            rc, out = run(sp, ov, name, val)
            check(rc == 0, f"combo {on}: pass 1 {name}={val!r} rc={rc}")
            if rc:
                print("      " + "\n      ".join(out.splitlines()[-6:]))
        s1 = R.snapshot(sp)
        for name, val in seq:
            rc, out = run(sp, ov, name, val)
            check(rc == 0, f"combo {on}: pass 2 {name}={val!r} rc={rc}")
            if rc:
                print("      " + "\n      ".join(out.splitlines()[-6:]))
        s2 = R.snapshot(sp)
        diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
        check(not diff, f"combo {on}: second pass byte-identical (changed: {diff})")
        return s1


def main() -> int:
    trees = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    on = (a, b, c, d)
                    print(f"== combo GLM53_DENSE_W8A8={a} GLM53_KDA_FLASHKDA={b} GLM53_MHC_SP={c} GLM53_MOE_E4M3={d}")
                    trees[on] = combo_tree(on)
    base = trees[(0, 0, 0, 0)]

    def delta(a_, b_):
        return sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))

    for i, name in enumerate(LABELS):
        on = [0, 0, 0, 0]
        on[i] = 1
        d = delta(base, trees[tuple(on)])
        exp = sorted(EXPECTED[name])
        check(d == exp, f"delta({name}=1) == {exp} (got {d})")
    union = sorted(set().union(*[set(EXPECTED[n]) for n in LABELS]))
    d = delta(base, trees[(1, 1, 1, 1)])
    check(d == union, f"delta(all four on) == the union of the four deltas (got {d})")
    for on in ((1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1),
               (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)):
        exp = sorted(set().union(*[set(EXPECTED[n]) for n, b in zip(LABELS, on) if b]))
        check(delta(base, trees[on]) == exp, f"delta({on}) == {exp}")
    # the single-feature deltas must not intersect beyond the shared integrate.py (w8a8 vs e4m3)
    inter = EXPECTED["w8a8"] & EXPECTED["e4m3"]
    check(inter == {"integrate.py"}, f"w8a8/e4m3 integrate.py is the only shared file (got {inter})")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
