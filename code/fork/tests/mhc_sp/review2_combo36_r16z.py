"""The combined kit's SIX switch axes in every one of the 48 combinations, in one bundle pass, plus a second pass
(idempotence) - the production overlay chain including kdaqkv (GLM53_KDA_STRIDED_QKV=1, as production runs).

r16z's axes beyond r16y's four: the FlashKDA BUILD GLM53_KDA_FLASHKDA_V (unset/1/2/3 -> off / the shipped fkda
build / fkda2 / fkda3; all three stage under the same site names glm53_flashkda.py + _flashkda_fp32_C.abi3.so, so
the delta FILE set is the same and only the bytes differ) and the pipelined SP GLM53_MHC_SP2 (needs GLM53_MHC_SP=1;
the delta file set is mhcsp's). For each (GLM53_DENSE_W8A8, GLM53_KDA_FLASHKDA@V, GLM53_MHC_SP[@SP2],
GLM53_MOE_E4M3): run each gated patch of the bundle chain in the bundle's own order (patch_kda_strided_qkv.py =1
always, then patch_dense_w8a8.py, patch_flashkda.py, patch_mhc_sp.py, patch_mhc_sp2.py and patch_moe_e4m3.py last,
per patch_tf_bundle.py's PATCHES tuple) with its switch ("1" on, "" = the empty string the launcher forwards when
unset; SP2 additionally with GLM53_MHC_SP=1 in its env, FK variants with GLM53_KDA_FLASHKDA_V); require rc 0
everywhere, a SECOND pass rc 0 and byte-identical, and the tree deltas to compose exactly:
  off-tree (0,off,off,0) == the kdaqkv-only tree (== the previous kit's composed tree; off_equals_prev proves it
             against the r16x kit on the real fs),
  delta(W8A8=1)       = {integrate.py (the armed block), fp8_w8a8.py, tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so},
  delta(FK@V)         = {NV/kda.py, quickwins, glm53_flashkda.py, _flashkda_fp32_C.abi3.so} (V=3's kda.py call text
                        differs: the direct-output out=, fp eb0e8dedaee6deb6 vs 9715f9b548cfa694 for V unset/2),
  delta(SP)           = {NV/model.py, quickwins, moeglue} (SP2's bytes differ from SP's),
  delta(E4=1)         = {integrate.py, glm53_moe_e4m3.py, glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so},
  tree(all-on) - tree(off) = the union (the patches interfere with nothing; the two integrate.py armings compose),
  and the same combo under a DIFFERENT FlashKDA build / SP level has a DIFFERENT tree (the switch really switches).

Host-only. Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16z PREV_KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16x \
  IMG_VLLM=$TF_EXL3_ASSETS/img_vllm_glm5next_r16x python3 tests/mhc_sp/review2_combo36_r16z.py
"""
from __future__ import annotations

import itertools
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z"))
os.environ.setdefault("PREV_KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16x"))
import review2_compose3_r16x as R  # build()/run()/snapshot()/PATCHES/NV  (env overrides must precede the import)

FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


# the bundle's own PATCHES order (patch_tf_bundle.py): kdaqkv first, dense_w8a8, flashkda, mhc_sp, mhc_sp2, moe_e4m3 LAST
FILES = {"w8a8": "patch_dense_w8a8.py", "flashkda": "patch_flashkda.py", "mhcsp": "patch_mhc_sp.py",
         "mhcsp2": "patch_mhc_sp2.py", "e4m3": "patch_moe_e4m3.py"}
ENVNAMES = {"w8a8": "GLM53_DENSE_W8A8", "flashkda": "GLM53_KDA_FLASHKDA", "mhcsp": "GLM53_MHC_SP",
            "mhcsp2": "GLM53_MHC_SP2", "e4m3": "GLM53_MOE_E4M3"}
EXPECTED = {
    "w8a8": {"integrate.py", "fp8_w8a8.py", "tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so"},
    "flashkda": {R.NV + "/kda.py", "glm53_prefill_quickwins.py",
                 "glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"},
    "mhcsp": {R.NV + "/model.py", "glm53_prefill_quickwins.py", "glm53_moeglue.py"},
    "mhcsp2": {R.NV + "/model.py", "glm53_prefill_quickwins.py", "glm53_moeglue.py"},
    "e4m3": {"integrate.py", "glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"},
}
LABELS = ("w8a8", "flashkda", "mhcsp", "mhcsp2", "e4m3")


def run(sp, ov, name, val, extra=None):
    if name == "kdaqkv":   # R.PATCHES: kdaqkv's overlay file + env name (no fingerprint table of its own)
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
    return p.returncode, (p.stdout + p.stderr).strip()


def combo_tree(w8a8: int, fkda: str, sp: str, e4: int) -> dict:
    """fkda: "off" | "fkda" | "fkda2" | "fkda3" (the GLM53_KDA_FLASHKDA_V value, "" = unset for the shipped build);
    sp: "off" | "sp" | "sp2"."""
    with tempfile.TemporaryDirectory() as td:
        sp_dir, ov = R.build(Path(td))
        fk_on = fkda != "off"
        vval = {"off": "", "fkda": "", "fkda2": "2", "fkda3": "3"}[fkda]
        seq = [("kdaqkv", "1")]
        if w8a8:
            seq.append(("w8a8", "1"))
        if fk_on:
            seq.append(("flashkda", "1"))
        if sp != "off":
            seq.append(("mhcsp", "1"))
        if sp == "sp2":
            seq.append(("mhcsp2", "1"))
        if e4:
            seq.append(("e4m3", "1"))
        fk_env = {"GLM53_KDA_FLASHKDA_V": vval} if fk_on else None
        sp2_env = {"GLM53_MHC_SP": "1"}
        for p1 in (True, False):
            for name, val in seq:
                extra = fk_env if name == "flashkda" else (sp2_env if name == "mhcsp2" else None)
                rc, out = run(sp_dir, ov, name, val, extra)
                check(rc == 0, f"combo w8a8={w8a8} fkda={fkda}{('@' + vval) if fk_on else ''} sp={sp} e4={e4}: "
                               f"pass {2 if p1 is False else 1} {name}={val!r} rc={rc}")
                if rc:
                    print("      " + "\n      ".join(out.splitlines()[-6:]))
            s = R.snapshot(sp_dir)
            if p1:
                s1 = s
            else:
                s2 = s
        diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
        check(not diff, f"combo w8a8={w8a8} fkda={fkda} sp={sp} e4={e4}: second pass byte-identical (changed: {diff})")
        return s1


def main() -> int:
    trees = {}
    for a in (0, 1):
        for b in ("off", "fkda", "fkda2", "fkda3"):
            for c in ("off", "sp", "sp2"):
                for d in (0, 1):
                    on = (a, b, c, d)
                    print(f"== combo GLM53_DENSE_W8A8={a} GLM53_KDA_FLASHKDA@V={b} GLM53_MHC_SP[@SP2]={c} GLM53_MOE_E4M3={d}")
                    trees[on] = combo_tree(*on)
    base = trees[(0, "off", "off", 0)]

    def delta(a_, b_):
        return sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k))

    singles = {("w8a8", (1, "off", "off", 0)), ("flashkda", (0, "fkda", "off", 0)), ("fkda2", (0, "fkda2", "off", 0)),
               ("fkda3", (0, "fkda3", "off", 0)), ("mhcsp", (0, "off", "sp", 0)), ("mhcsp2", (0, "off", "sp2", 0)),
               ("e4m3", (0, "off", "off", 1))}
    for name, on in singles:
        d = delta(base, trees[on])
        exp = sorted(EXPECTED["mhcsp" if name == "mhcsp2" else ("flashkda" if name.startswith("fkda") else name)])
        check(d == exp, f"delta({name}) == {exp} (got {d})")
    # the file sets are shared, the BYTES are not: fkda/fkda2 share the kda.py call text (fp 9715f9b5...), fkda3's
    # call text differs (fp eb0e8dedaee6deb6); the fkda2/fkda3 installed pairs differ from the shipped pair
    check(trees[(0, "fkda", "off", 0)] != trees[(0, "fkda2", "off", 0)], "the fkda2 build's tree != the shipped build's")
    check(trees[(0, "fkda", "off", 0)] != trees[(0, "fkda3", "off", 0)], "the fkda3 build's tree != the shipped build's")
    check(trees[(0, "fkda2", "off", 0)] != trees[(0, "fkda3", "off", 0)], "the fkda3 build's tree != the fkda2 build's")
    # the snapshot holds HASHES: prove the call text + the kda_conv fingerprints on fresh trees (the file contents)
    def fk_texts(vval):
        import subprocess as sp_
        with tempfile.TemporaryDirectory() as td2:
            sp_dir, ov = R.build(Path(td2))
            env = dict(os.environ, GLM53_SITEPKG=str(sp_dir), GLM53_TF_OVERLAY=str(ov),
                       GLM53_OPT=str(ov.parent.parent), PYTHONDONTWRITEBYTECODE="1",
                       GLM53_KDA_FLASHKDA="1", GLM53_KDA_FLASHKDA_V=vval)
            for k in ("GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY", "GLM53_MOEGLUE_PY", "GLM53_KDA_PY",
                      "GLM53_FLA_KDA_PY", "GLM53_FLA_FUSED_RECURRENT_PY"):
                env.pop(k, None)
            p_ = sp_.run([sys.executable, str(ov / "patch_flashkda.py")], env=env, capture_output=True, text=True)
            check(p_.returncode == 0, f"fk_texts(V={vval!r}): patch rc={p_.returncode}")
            return (sp_dir / R.NV.lstrip("/") / "kda.py").read_text(), \
                   (sp_dir / "glm53_prefill_quickwins.py").read_text()
    kda1_t, qw1_t = fk_texts("")
    kda2_t, qw2_t = fk_texts("2")
    kda3_t, qw3_t = fk_texts("3")
    check(kda1_t == kda2_t, "V=2 leaves the kda.py call text byte-identical to the shipped build's (the fp stays 9715f9b5...)")
    check(kda1_t != kda3_t and "out=(None if use_spec else core_attn_out[:, :num_actual_tokens])" in kda3_t
          and "out=(None if use_spec" not in kda1_t,
          "V=3's kda.py carries the direct-output out= call (and the shipped one does not)")
    check('"9715f9b548cfa694"' in qw1_t and '"9715f9b548cfa694"' in qw2_t and "eb0e8dedaee6deb6" not in qw1_t
          and '"eb0e8dedaee6deb6"' in qw3_t and "9715f9b548cfa694" not in qw3_t,
          "the kda_conv VERIFIED fp follows the call text (9715f9b5... for the shipped/fkda2 call, eb0e8dedaee6deb6 for fkda3)")
    check(trees[(0, "fkda", "off", 0)][R.NV + "/kda.py"] == trees[(0, "fkda2", "off", 0)][R.NV + "/kda.py"],
          "V=2's composed kda.py hash == the shipped build's")
    check(trees[(0, "fkda", "off", 0)][R.NV + "/kda.py"] != trees[(0, "fkda3", "off", 0)][R.NV + "/kda.py"],
          "V=3's composed kda.py hash != the shipped build's")
    check(trees[(0, "off", "sp", 0)] != trees[(0, "off", "sp2", 0)], "the SP2 tree != the r16x SP tree")
    # compositions: every pair/triple combination of axes on (fkda at each of its three builds) sums to the union
    union = sorted(set().union(*[set(EXPECTED[n]) for n in LABELS]))
    d = delta(base, trees[(1, "fkda3", "sp2", 1)])
    check(d == union, f"delta(all six on, fkda3 + SP2) == the union of the five deltas (got {d})")
    d = delta(base, trees[(1, "fkda", "sp", 1)])
    check(d == union, f"delta(all six on, shipped fkda + SP) == the union (got {d})")
    for on in ((1, "fkda2", "sp2", 0), (1, "fkda3", "sp", 0), (0, "fkda2", "sp", 1), (0, "fkda3", "sp2", 1),
               (1, "off", "sp2", 1), (0, "fkda2", "off", 1), (1, "fkda3", "off", 0)):
        exp = sorted(set().union(*[set(EXPECTED[n]) for n, hit in
                                   (("w8a8", on[0]), ("flashkda", on[1] != "off"),
                                    ("mhcsp", on[2] != "off"), ("mhcsp2", on[2] == "sp2"), ("e4m3", on[3])) if hit]))
        check(delta(base, trees[on]) == exp, f"delta({on}) == {exp}")
    # the two integrate.py armings compose (w8a8 vs e4m3) and the fingerprints land in the tables
    inter = EXPECTED["w8a8"] & EXPECTED["e4m3"]
    check(inter == {"integrate.py"}, f"w8a8/e4m3 integrate.py is the only shared file (got {inter})")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
