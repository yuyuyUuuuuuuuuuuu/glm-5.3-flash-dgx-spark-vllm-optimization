"""r16z3's moe3 KNOBS in every combination with their parent features (and each other), in one bundle pass, plus a
second pass (idempotence): the two layer lists change NO composed file (the dial lives in the installed
glm53_moe_e4m3.py; WHICH layers run e4m3 / the f16 down is the engine-boot row, boot_checks), so each knob-on tree
must be byte-identical to its parent feature's tree. GLM53_MOE_FUSED16 is a proper switch with its own payload
(patch_moe_fused16.py installs glm53_moe_fused16.py + the SHARED e4m3 extension into site-packages and arms
integrate.py - its block goes BEFORE glm53_moe_e4m3's, which stays last), so its delta is exactly that file set and
the arm order is asserted on the composed integrate.py.

  fused16 axis: off | 1
  moe axis:     off | 1 (down e4m3, the default) | 1+f16 | 1 + GLM53_MOE_E4M3_LAYERS=13-44
                | 1 + GLM53_MOE_E4M3_DOWN_LAYERS=3-12
  fkda axis:    off | 1 + GLM53_KDA_FLASHKDA_V=3   (control: the moe3 knobs must not disturb the fkda tree)

For every (fused16, moe, fkda) of the 2 x 5 x 2 = 20 combinations: run each gated patch of the bundle chain in the
bundle's own order (kdaqkv = 1 always, then patch_flashkda.py, patch_moe_fused16.py, patch_moe_e4m3.py last);
require rc 0 everywhere, a SECOND pass rc 0 and byte-identical, and:
  tree(knob-on) == tree(parent feature, knob unset) for the two layer lists,
  delta(fused16=1) == {integrate.py, glm53_moe_fused16.py, glm53_moe_e4m3_ext....so} (the extension is the e4m3
  payload, shared: the fused16 patch copies it into site-packages even with e4m3 itself off),
  with e4m3 also armed the composed integrate.py carries BOTH blocks with the fused16 one BEFORE glm53_moe_e4m3's,
  and the composed modules / extension are byte-identical to THIS worktree's overlay files (the built .so is the
  moe3 build: kernels/moe_e4m3.cu of this branch).

Host-only. Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16z3 PREV_KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16z2rev \\
  IMG_VLLM=$TF_EXL3_ASSETS/img_vllm_glm5next_r16x python3 tests/mhc_sp/review2_knobs_r16z3.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z3"))
os.environ.setdefault("PREV_KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2rev"))
import review2_compose3_r16x as R  # build()/snapshot()/PATCHES/NV  (env overrides must precede the import)

FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


E4SO = "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"
FILES = {"fused16": "patch_moe_fused16.py", "e4m3": "patch_moe_e4m3.py", "flashkda": "patch_flashkda.py"}
ENVNAMES = {"fused16": "GLM53_MOE_FUSED16", "e4m3": "GLM53_MOE_E4M3", "flashkda": "GLM53_KDA_FLASHKDA"}
EXPECTED = {
    # patch_moe_fused16.py copies the SHARED e4m3 extension even with e4m3 itself off (its FILES)
    "fused16": {"integrate.py", "glm53_moe_fused16.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"},
    "e4m3": {"integrate.py", "glm53_moe_e4m3.py", E4SO},
    "flashkda": {R.NV + "/kda.py", "glm53_prefill_quickwins.py",
                 "glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"},
}


def run(sp, ov, name, val, extra=None, out=None):
    if name == "kdaqkv":
        fname, envname = R.PATCHES["kdaqkv"][0], R.PATCHES["kdaqkv"][1]
    else:
        fname, envname = FILES[name], ENVNAMES[name]
    env = dict(os.environ, GLM53_SITEPKG=str(sp), GLM53_TF_OVERLAY=str(ov), GLM53_OPT=str(ov.parent.parent),
               PYTHONDONTWRITEBYTECODE="1", **{envname: val})
    for k, v in (extra or {}).items():
        env[k] = v
    for k in ("GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY", "GLM53_MOEGLUE_PY", "GLM53_KDA_PY",
              "GLM53_FLA_KDA_PY", "GLM53_FLA_FUSED_RECURRENT_PY"):
        env.pop(k, None)
    import subprocess
    p = subprocess.run([sys.executable, str(ov / fname)], env=env, capture_output=True, text=True)
    if out is not None:
        out.append(p.stdout + p.stderr)
    return p.returncode, (p.stdout + p.stderr).strip()


def knob_tree(fused16: str, moe: str, fkda: str) -> tuple[dict, list[str]]:
    """fused16: "off" | "1"; moe: "off" | "1" | "1f16" | "1L" | "1DL"; fkda: "off" | "3".
    Returns (snapshot, compose outputs)."""
    with tempfile.TemporaryDirectory() as td:
        sp_dir, ov = R.build(Path(td))
        seq = [("kdaqkv", "1")]
        extra_by_name: dict[str, dict] = {}
        outs: list[str] = []
        if fkda != "off":
            seq.append(("flashkda", "1"))
            extra_by_name["flashkda"] = {"GLM53_KDA_FLASHKDA_V": "3"}
        if fused16 != "off":
            seq.append(("fused16", "1"))
        if moe != "off":
            seq.append(("e4m3", "1"))
            ex = {}
            if moe == "1f16":
                ex["GLM53_MOE_E4M3_DOWN"] = "f16"
            if moe == "1L":
                ex["GLM53_MOE_E4M3_LAYERS"] = "13-44"
            if moe == "1DL":
                ex["GLM53_MOE_E4M3_DOWN_LAYERS"] = "3-12"
            extra_by_name["e4m3"] = ex
        for p1 in (True, False):
            for name, val in seq:
                rc, o = run(sp_dir, ov, name, val, extra_by_name.get(name), out=outs)
                check(rc == 0, f"knobs fused16={fused16} moe={moe} fkda={fkda}: pass {2 if p1 is False else 1} {name}={val!r} rc={rc}")
                if rc:
                    print("      " + "\n      ".join(o.splitlines()[-6:]))
            s = R.snapshot(sp_dir)
            if p1:
                s1 = s
            else:
                s2 = s
        diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
        check(not diff, f"knobs fused16={fused16} moe={moe} fkda={fkda}: second pass byte-identical (changed: {diff})")
        if fused16 == "1" and moe != "off":
            integ = s1.get("integrate.py")
            check(integ is not None, "integrate.py present")
        return s1, outs


def arm_order() -> None:
    """with FUSED16=1 and e4m3 =1 both armed: the composed integrate.py carries BOTH blocks, the fused16 one BEFORE
    glm53_moe_e4m3's (that one must stay last: its arm() requires it at the end), and a SECOND armed pass refuses a
    moved block."""
    with tempfile.TemporaryDirectory() as td:
        sp_dir, ov = R.build(Path(td))
        for name, val, extra in (("kdaqkv", "1", None), ("fused16", "1", None), ("e4m3", "1", None)):
            rc, o = run(sp_dir, ov, name, val, extra)
            check(rc == 0, f"arm order: {name}={val} rc={rc}")
            if rc:
                print("      " + "\n      ".join(o.splitlines()[-6:]))
        integ = (sp_dir / "integrate.py").read_text()
        i = integ.index("# [glm53-moe-fused16] BEGIN")
        e = integ.index("# [glm53-moe-e4m3] BEGIN")
        check(i < e, "integrate.py: the fused16 block sits BEFORE glm53_moe_e4m3's")
        check(integ.rstrip().endswith("# [glm53-moe-e4m3] END"), "integrate.py: glm53_moe_e4m3's block is last")
        rc, o = run(sp_dir, ov, "fused16", "1", out=None)
        rc2, o2 = run(sp_dir, ov, "e4m3", "1")
        check(rc == 0 and rc2 == 0, f"arm order: idempotent second pass (rc {rc}, {rc2})")
        check("already armed" in o, "the fused16 arm reports 'already armed' on the second pass")


def main() -> int:
    trees, logs = {}, {}
    for f in ("off", "1"):
        for e in ("off", "1", "1f16", "1L", "1DL"):
            for k in ("off", "3"):
                key = (f, e, k)
                print(f"== knobs GLM53_MOE_FUSED16={f} GLM53_MOE_E4M3@LAYERS={e} GLM53_KDA_FLASHKDA@V={k}")
                trees[key], logs[key] = knob_tree(*key)
    base = trees[("off", "off", "off")]

    def delta(a_, b_):
        return sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))

    # the single-feature deltas keep their exact file sets (the kit's payloads)
    for name, key, feat in (("fused16=1", ("1", "off", "off"), "fused16"),
                            ("moe=1", ("off", "1", "off"), "e4m3"),
                            ("fkda=3", ("off", "off", "3"), "flashkda")):
        d = delta(base, trees[key])
        check(d == sorted(EXPECTED[feat]), f"delta({name}) == {sorted(EXPECTED[feat])} (got {d})")
    # every knob-on tree == its parent feature's tree (the layer dials change no file)
    for knob, parent in ((("off", "1f16", "off"), ("off", "1", "off")), (("off", "1L", "off"), ("off", "1", "off")),
                         (("off", "1DL", "off"), ("off", "1", "off")), (("off", "1L", "3"), ("off", "1", "3")),
                         (("1", "1L", "off"), ("1", "1", "off")), (("1", "1DL", "off"), ("1", "1", "off"))):
        check(trees[knob] == trees[parent], f"tree(knobs {knob}) == tree({parent}) (the layer dial changes no file)")
    # the union composes exactly (both arms present)
    d = delta(base, trees[("1", "1", "off")])
    check(d == sorted(EXPECTED["fused16"] | EXPECTED["e4m3"]), f"delta(fused16+e4m3) == the union (got {d})")
    arm_order()
    # the composed modules are byte-identical to THIS worktree's overlay files (one source of truth)
    with tempfile.TemporaryDirectory() as td2:
        sp_dir, ov = R.build(Path(td2))
        run(sp_dir, ov, "kdaqkv", "1")
        run(sp_dir, ov, "fused16", "1")
        run(sp_dir, ov, "e4m3", "1")
        repo_ov = R.REPO / "overlay"
        for mod in ("glm53_moe_e4m3.py", "glm53_moe_fused16.py", E4SO):
            check((sp_dir / mod).read_bytes() == (repo_ov / mod).read_bytes(),
                  f"the composed {mod} is byte-identical to this worktree's overlay file")
    # the fkda3 control still selects the fkda3 build (visible in the compose log: the staging print)
    check(any("GLM53_KDA_FLASHKDA_V=3: the fkda3 build" in o for o in logs[("off", "off", "3")]),
          "the fkda3 control still selects the fkda3 build")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
