"""The combined kit's THREE feature switches in every one of the 8 combinations, in one bundle pass, plus a second
pass (idempotence) - the production overlay chain including kdaqkv (GLM53_KDA_STRIDED_QKV=1, as production runs).

For each (GLM53_KDA_FLASHKDA, GLM53_MHC_SP, GLM53_MOE_E4M3) in {0,1}^3: run patch_kda_strided_qkv.py (=1, always)
and each feature's patch with its switch ("1" on, "" = the empty string the launcher forwards when unset) from the
combined kit's overlay/ on a fresh fake site-packages tree; require rc 0 everywhere, a SECOND pass rc 0 and
byte-identical, and the tree deltas to compose exactly:
  off-tree (0,0,0) == the kdaqkv-only tree (== the previous kit's composed tree, off_equals_prev proves it against
             r16n on the real fs),
  delta(FK=1) = {NV/kda.py, quickwins, glm53_flashkda.py, _flashkda_fp32_C.abi3.so},
  delta(SP=1) = {NV/model.py, quickwins, moeglue},
  delta(E4=1) = {integrate.py},
  tree(1,1,1) - tree(0,0,0) = the union of the three sets (the patches interfere with nothing).

Host-only. Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16x IMG_VLLM=<extracted vllm dir> python3 tests/mhc_sp/review2_combo8_r16x.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review2_compose3_r16x as R  # build()/run()/snapshot()/PATCHES

FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


EXPECTED = {
    "flashkda": {R.NV + "/kda.py", "glm53_prefill_quickwins.py",
                 "glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"},
    "mhcsp": {R.NV + "/model.py", "glm53_prefill_quickwins.py", "glm53_moeglue.py"},
    "e4m3": {"integrate.py", "glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"},
}
LABELS = ("flashkda", "mhcsp", "e4m3")


def combo_tree(sp_off, on: tuple[int, int, int]) -> dict:
    with tempfile.TemporaryDirectory() as td:
        sp, ov = R.build(Path(td))
        # the bundle skips a patch whose switch is empty/unset WITHOUT running the patch script
        # (patch_tf_bundle.run_patch), so combo_tree runs exactly what the bundle would run
        seq = [("kdaqkv", "1")] + [(n, "1") for i, n in enumerate(LABELS) if on[i]]
        for name, val in seq:
            rc, out = R.run(sp, ov, name, val)
            check(rc == 0, f"combo {on}: pass 1 {name}={val!r} rc={rc}")
            if rc:
                print("      " + "\n      ".join(out.splitlines()[-4:]))
        s1 = R.snapshot(sp)
        for name, val in seq:
            rc, out = R.run(sp, ov, name, val)
            check(rc == 0, f"combo {on}: pass 2 {name}={val!r} rc={rc}")
        s2 = R.snapshot(sp)
        diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
        check(not diff, f"combo {on}: second pass byte-identical (changed: {diff})")
        return s1


def main() -> int:
    trees = {}
    for on in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)):
        print(f"== combo GLM53_KDA_FLASHKDA={on[0]} GLM53_MHC_SP={on[1]} GLM53_MOE_E4M3={on[2]}")
        trees[on] = combo_tree(None, on)
    base = trees[(0, 0, 0)]

    def delta(a, b):
        return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))

    for i, name in enumerate(LABELS):
        on = [0, 0, 0]
        on[i] = 1
        d = delta(base, trees[tuple(on)])
        exp = sorted(EXPECTED[name])
        check(d == exp, f"delta({name}=1) == {exp} (got {d})")
    union = sorted(set().union(*[set(EXPECTED[n]) for n in LABELS]))
    d = delta(base, trees[(1, 1, 1)])
    check(d == union, f"delta(all three on) == the union of the three deltas (got {d})")
    for on in ((1, 1, 0), (1, 0, 1), (0, 1, 1)):
        exp = sorted(set().union(*[set(EXPECTED[n]) for n, b in zip(LABELS, on) if b]))
        check(delta(base, trees[on]) == exp, f"delta({on}) == {exp}")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
