"""r16z2's BEHAVIOR knobs in every combination with their parent features (and each other), in one bundle pass, plus
a second pass (idempotence): the knobs change NO composed file (the dial lives in the same installed module /
extension), so each knob-on tree must be byte-identical to its parent feature's tree, and the compose logs prove the
knob reached the module.

  w8a8 axis: off | 1 (custom, the default) | 1 + GLM53_DENSE_W8A8_GEMM=cutlass_mm | 1 + GLM53_DENSE_W8A8_ONLY
  moe axis:  off | 1 (e4m3 down, the default) | 1 + GLM53_MOE_E4M3_DOWN=f16
  fkda axis: off | 1 | 1 + GLM53_KDA_FLASHKDA_V=3   (control: the knobs must not disturb the fkda/sp trees)

For every (w8a8, moe, fkda) of the 4 x 3 x 3 = 36 combinations: run each gated patch of the bundle chain in the
bundle's own order (kdaqkv =1 always, then patch_dense_w8a8.py, patch_flashkda.py, patch_mhc_sp.py, patch_mhc_sp2.py
- mhcsp2's r16x SP state is part of the 'sp2' variants of the parent trees and stays fixed OFF here to keep the knob
matrix readable, patch_moe_e4m3.py last); require rc 0 everywhere, a SECOND pass rc 0 and byte-identical, and:
  tree(knob-on) == tree(parent feature, knob unset) for every knob,
  the compose logs name the selected GEMM backend / projection filter / down width,
  and the w8a8/e4m3 deltas keep their exact file sets (integrate.py armings compose).

Host-only. Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16z2 PREV_KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16x \\
  IMG_VLLM=$TF_EXL3_ASSETS/img_vllm_glm5next_r16x python3 tests/mhc_sp/review2_knobs_r16z2.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2"))
os.environ.setdefault("PREV_KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16x"))
import review2_compose3_r16x as R  # build()/snapshot()/PATCHES/NV  (env overrides must precede the import)

FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


FILES = {"w8a8": "patch_dense_w8a8.py", "flashkda": "patch_flashkda.py", "mhcsp": "patch_mhc_sp.py",
         "mhcsp2": "patch_mhc_sp2.py", "e4m3": "patch_moe_e4m3.py"}
ENVNAMES = {"w8a8": "GLM53_DENSE_W8A8", "flashkda": "GLM53_KDA_FLASHKDA", "mhcsp": "GLM53_MHC_SP",
            "mhcsp2": "GLM53_MHC_SP2", "e4m3": "GLM53_MOE_E4M3"}
EXPECTED = {
    "w8a8": {"integrate.py", "fp8_w8a8.py", "tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so"},
    "flashkda": {R.NV + "/kda.py", "glm53_prefill_quickwins.py",
                 "glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"},
    "mhcsp": {R.NV + "/model.py", "glm53_prefill_quickwins.py", "glm53_moeglue.py"},
    "e4m3": {"integrate.py", "glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"},
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


def knob_tree(w8a8: str, moe: str, fkda: str) -> tuple[dict, list[str]]:
    """w8a8: "off" | "1" | "1mm" | "1only"; moe: "off" | "1" | "1f16"; fkda: "off" | "1" | "3".
    Returns (snapshot, compose outputs)."""
    with tempfile.TemporaryDirectory() as td:
        sp_dir, ov = R.build(Path(td))
        seq = [("kdaqkv", "1")]
        extra_by_name: dict[str, dict] = {}
        outs: list[str] = []
        if w8a8 != "off":
            seq.append(("w8a8", "1"))
            ex = {}
            if w8a8 == "1mm":
                ex["GLM53_DENSE_W8A8_GEMM"] = "cutlass_mm"
            if w8a8 == "1only":
                ex["GLM53_DENSE_W8A8_ONLY"] = "kda.in_proj_qkvbfg_a,mla.o_proj"
            extra_by_name["w8a8"] = ex
        if fkda != "off":
            seq.append(("flashkda", "1"))
            extra_by_name["flashkda"] = {"GLM53_KDA_FLASHKDA_V": "3" if fkda == "3" else ""}
        if moe != "off":
            seq.append(("e4m3", "1"))
            extra_by_name["e4m3"] = {"GLM53_MOE_E4M3_DOWN": "f16"} if moe == "1f16" else {}
        for p1 in (True, False):
            for name, val in seq:
                rc, o = run(sp_dir, ov, name, val, extra_by_name.get(name), out=outs)
                check(rc == 0, f"knobs w8a8={w8a8} moe={moe} fkda={fkda}: pass {2 if p1 is False else 1} {name}={val!r} rc={rc}")
                if rc:
                    print("      " + "\n      ".join(o.splitlines()[-6:]))
            s = R.snapshot(sp_dir)
            if p1:
                s1 = s
            else:
                s2 = s
        diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
        check(not diff, f"knobs w8a8={w8a8} moe={moe} fkda={fkda}: second pass byte-identical (changed: {diff})")
        return s1, outs


def main() -> int:
    trees, logs = {}, {}
    for w in ("off", "1", "1mm", "1only"):
        for e in ("off", "1", "1f16"):
            for f in ("off", "1", "3"):
                key = (w, e, f)
                print(f"== knobs GLM53_DENSE_W8A8={w} GLM53_MOE_E4M3@DOWN={e} GLM53_KDA_FLASHKDA@V={f}")
                trees[key], logs[key] = knob_tree(*key)
    base = trees[("off", "off", "off")]

    def delta(a_, b_):
        return sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))

    # the single-feature deltas keep their exact file sets (the kit's payloads)
    for name, key, feat in (("w8a8=1", ("1", "off", "off"), "w8a8"), ("moe=1", ("off", "1", "off"), "e4m3"),
                            ("fkda=1", ("off", "off", "1"), "flashkda"), ("fkda=3", ("off", "off", "3"), "flashkda")):
        d = delta(base, trees[key])
        check(d == sorted(EXPECTED[feat]), f"delta({name}) == {sorted(EXPECTED[feat])} (got {d})")
    # every knob-on tree == its parent feature's tree (the knobs change no file)
    for knob, parent in ((("1mm", "off", "off"), ("1", "off", "off")), (("1only", "off", "off"), ("1", "off", "off")),
                         (("off", "1f16", "off"), ("off", "1", "off")), (("off", "1f16", "1"), ("off", "1", "1")),
                         (("1mm", "off", "3"), ("1", "off", "3")), (("1only", "1f16", "off"), ("1", "1f16", "off"))):
        check(trees[knob] == trees[parent], f"tree(knobs {knob}) == tree({parent}) (the behavior dial changes no file)")
    # the knobs really dial the module: the composed fp8_w8a8.py / glm53_moe_e4m3.py must be BYTE-IDENTICAL to the
    # reviewed branches' modules (the knob semantics incl. the ONLY filter and the custom-GEMM checks are those
    # modules', proven by the branches' own rigs: tests/w8a82/test_w8a82.py and tests/moe2/test_down16.py on GPU)
    import subprocess as sp_
    for feat, mod, br, br_path in (("w8a8", "fp8_w8a8.py", "w8a82", "overlay/fp8_w8a8.py"),
                                   ("e4m3", "glm53_moe_e4m3.py", "moe2", "overlay/glm53_moe_e4m3.py"),
                                   ("e4m3", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so", "moe2",
                                    "overlay/glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so")):
        with tempfile.TemporaryDirectory() as td2:
            sp_dir, ov = R.build(Path(td2))
            run(sp_dir, ov, feat, "1")
            composed = (sp_dir / mod).read_bytes() if mod != "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so" \
                else (sp_dir / mod).read_bytes()
        br_file = Path(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-fork-wt-") + br) / br_path
        if br_file.is_file():
            check(composed == br_file.read_bytes(),
                  f"the composed {mod} is byte-identical to the {br} branch's built artifact")
        else:
            br_bytes = sp_.run(["git", "-C", str(R.REPO), "show", f"{br}:{br_path}"], capture_output=True).stdout
            check(composed == br_bytes, f"the composed {mod} is byte-identical to the {br} branch's committed bytes")
    # the fkda3 control still selects the fkda3 build (visible in the compose log: the staging print)
    check(any("GLM53_KDA_FLASHKDA_V=3: the fkda3 build" in o for o in logs[("off", "off", "3")]),
          "the fkda3 control still selects the fkda3 build")
    # the unions compose exactly (per-key: only the ACTIVE features' sets)
    for key in (("1", "1", "1"), ("1mm", "1f16", "3"), ("1only", "1", "off")):
        feats = [n for n, hit in (("w8a8", key[0] != "off"), ("e4m3", key[1] != "off"), ("flashkda", key[2] != "off")) if hit]
        exp = sorted(set().union(*[set(EXPECTED[n]) for n in feats]))
        d = delta(base, trees[key])
        check(d == exp, f"delta(features on, knobs {key}) == {exp} (got {d})")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
