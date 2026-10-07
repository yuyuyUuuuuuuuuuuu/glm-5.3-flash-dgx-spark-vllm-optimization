"""Adversarial review 2 (mhcsp) against the COMBINED kit (r16x): does the ACTUAL production overlay chain compose?

Production runs GLM53_KDA_STRIDED_QKV=1 (r16k) on top of the r16n bundle, so the chain the combined kit runs at
container start is FOUR gated patches: patch_kda_strided_qkv.py (kdaqkv) first, then patch_flashkda.py,
patch_mhc_sp.py and patch_moe_e4m3.py (the bundle order; kdaqkv has no fingerprint table of its own, the other
three extend quickwins/moeglue). This script runs that chain in one pass for every order of the three feature
patches (kdaqkv always first, like production), then a SECOND pass on the same tree and checks it is a clean no-op
(byte-identical), and that the final trees of all orders are identical.

Host-only (no torch, no GPU, nothing in any container is written): a fake site-packages tree under a temp dir holds
the image's vllm/models/glm5next/nvidia/{model,kda}.py and vllm/third_party/flash_linear_attention/ops/
{fused_recurrent,kda}.py (copied out of the image beforehand: IMG_VLLM=<dir> holding models/glm5next/nvidia/ and
third_party/flash_linear_attention/ops/) and the kits' site/ glm53_prefill_quickwins.py, glm53_moeglue.py,
integrate.py; the patches run from the COMBINED kit's overlay/ dir (GLM53_SITEPKG / GLM53_TF_OVERLAY / GLM53_OPT
point at the temp tree).

Checks per order: every patch rc 0; second pass rc 0 and byte-identical; final trees of all orders byte-identical;
the fingerprints the runtime drift guards will compute are in their tables (quickwins mhc_aux/mhc_mean on the SP
forwards, kda_conv on the FlashKDA _forward, moeglue WARM on the SP layer forward plain + quickwins-transplanted);
the quickwins transplant anchors (MHC_AUX, MHC_FINAL, KDA_CONV) still anchor exactly once on the final sources.

Run: KIT=$TF_EXL3_KITS/tf-exl3-deploy16.r16x IMG_VLLM=<extracted vllm dir> python3 tests/mhc_sp/review2_compose3_r16x.py
"""
from __future__ import annotations

import hashlib
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = Path(os.environ.get("KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16x")))
PREV = Path(os.environ.get("PREV_KIT", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16n")))
IMG = Path(os.environ["IMG_VLLM"])
NV = "vllm/models/glm5next/nvidia"
FLA = "vllm/third_party/flash_linear_attention/ops"
FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def build(root: Path) -> tuple[Path, Path]:
    sp = root / "site"
    (sp / NV).mkdir(parents=True)
    (sp / FLA).mkdir(parents=True)
    for f in ("model.py", "kda.py"):
        shutil.copy(IMG / "models/glm5next/nvidia" / f, sp / NV / f)
    for f in ("fused_recurrent.py", "kda.py"):
        shutil.copy(IMG / "third_party/flash_linear_attention/ops" / f, sp / FLA / f)
    for f in ("glm53_prefill_quickwins.py", "glm53_moeglue.py", "integrate.py"):
        assert (KIT / "site" / f).read_bytes() == (PREV / "site" / f).read_bytes(), (f, "combined vs previous kit")
        shutil.copy(KIT / "site" / f, sp / f)
    ov = root / "opt" / "tf" / "overlay"
    ov.mkdir(parents=True)
    for f in (KIT / "overlay").iterdir():
        if f.is_file():
            shutil.copy(f, ov / f.name)
    return sp, ov


# kdaqkv first: production's bundle order (the r16k patch predates the feature patches; it has no fingerprint
# table, but its edits must not move any anchor the other three patches or the quickwins transplants use)
PATCHES = {"kdaqkv": ("patch_kda_strided_qkv.py", "GLM53_KDA_STRIDED_QKV", "1"),
           "mhcsp": ("patch_mhc_sp.py", "GLM53_MHC_SP", "1"),
           "flashkda": ("patch_flashkda.py", "GLM53_KDA_FLASHKDA", "1"),
           "e4m3": ("patch_moe_e4m3.py", "GLM53_MOE_E4M3", "1")}


def run(sp: Path, ov: Path, name: str, val: str | None = None) -> tuple[int, str]:
    f, env_name, val = PATCHES[name] if val is None else (PATCHES[name][0], PATCHES[name][1], val)
    env = dict(os.environ, GLM53_SITEPKG=str(sp), GLM53_TF_OVERLAY=str(ov), GLM53_OPT=str(ov.parent.parent),
               PYTHONDONTWRITEBYTECODE="1", **{env_name: val})
    for k in ("GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY", "GLM53_MOEGLUE_PY", "GLM53_KDA_PY",
              "GLM53_FLA_KDA_PY", "GLM53_FLA_FUSED_RECURRENT_PY"):
        env.pop(k, None)
    p = subprocess.run([sys.executable, str(ov / f)], env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def snapshot(sp: Path) -> dict:
    return {str(p.relative_to(sp)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sp.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def fp_func(src: str, cls: str, fn: str, decorators: bool) -> str:
    import ast
    import textwrap
    lines = src.splitlines(keepends=True)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == fn:
                    start = min([d.lineno for d in m.decorator_list] + [m.lineno]) if decorators else m.lineno
                    seg = "".join(lines[start - 1:m.end_lineno])
                    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(seg))).encode()).hexdigest()[:16]
    raise ValueError(f"{cls}.{fn}")


def table_has(src: str, key_re: str, fp: str) -> bool:
    m = [ln for ln in src.splitlines() if re.search(key_re, ln) and "frozenset({" in ln]
    return len(m) == 1 and f'"{fp}"' in m[0]


def verify(sp: Path, label: str) -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(KIT / "overlay"))
    import patch_mhc_sp as P
    model = (sp / NV / "model.py").read_text()
    kda = (sp / NV / "kda.py").read_text()
    qw = (sp / "glm53_prefill_quickwins.py").read_text()
    mg = (sp / "glm53_moeglue.py").read_text()
    fps = P.sp_fingerprints(model)
    check(table_has(qw, r'\("mhc_aux", "Glm5NextModel.forward"\)', fps["model_forward_sp"]),
          f"{label}: quickwins mhc_aux VERIFIED has the SP model forward {fps['model_forward_sp']}")
    check(table_has(qw, r'\("mhc_mean", "Glm5NextModel.forward"\)', fps["model_forward_sp"]),
          f"{label}: quickwins mhc_mean(model) VERIFIED has the SP model forward")
    check(table_has(qw, r'\("mhc_mean", "Glm5NextDecoderLayer.forward"\)', fps["layer_forward_sp"]),
          f"{label}: quickwins mhc_mean(layer) VERIFIED has the SP layer forward {fps['layer_forward_sp']}")
    check(table_has(mg, r'"Glm5NextDecoderLayer.forward":', fps["layer_forward_sp"])
          and table_has(mg, r'"Glm5NextDecoderLayer.forward":', fps["layer_forward_sp_qw"]),
          f"{label}: moeglue WARM_VERIFIED has the SP layer forward plain + quickwins ({fps['layer_forward_sp_qw']})")
    kfp = fp_func(kda, "Glm5NextLinearAttention", "_forward", True)
    check(table_has(qw, r'\("kda_conv", "Glm5NextLinearAttention._forward"\)', kfp),
          f"{label}: quickwins kda_conv VERIFIED has the FlashKDA _forward {kfp}")
    # the quickwins transplant anchors still anchor once on the final sources
    for name in ("MHC_AUX", "MHC_FINAL", "KDA_CONV"):
        i = qw.index(f"\n{name} = (")
        chunk = qw[i:i + 6000]
        olds = re.findall(r'"""(.*?)"""', chunk, re.S)
        tgt = kda if name == "KDA_CONV" else model
        check(len(olds) >= 2 and tgt.count(olds[0]) == 1, f"{label}: quickwins {name} anchor count on the final "
              f"source = {tgt.count(olds[0]) if olds else 'n/a'} (need 1)")
    for name, path in (("model.py", NV + "/model.py"), ("kda.py", NV + "/kda.py"),
                       ("ops/kda.py", FLA + "/kda.py"), ("ops/fused_recurrent.py", FLA + "/fused_recurrent.py")):
        compile((sp / path).read_text(), name, "exec")
    check("glm53_moe_e4m3" in (sp / "integrate.py").read_text(), f"{label}: integrate.py armed for e4m3")
    sys.path.pop(0)


def main() -> int:
    finals = {}
    orders = [("kdaqkv", *o) for o in itertools.permutations(("mhcsp", "flashkda", "e4m3"))]
    for order in orders:
        label = ">".join(order)
        print(f"== order {label}")
        with tempfile.TemporaryDirectory() as td:
            sp, ov = build(Path(td))
            for n in order:
                rc, out = run(sp, ov, n)
                check(rc == 0, f"{label}: pass 1 {n} rc={rc}")
                print("      " + "\n      ".join(out.splitlines()[-3:]))
            s1 = snapshot(sp)
            verify(sp, label + " pass1")
            for n in order:
                rc, out = run(sp, ov, n)
                check(rc == 0, f"{label}: pass 2 {n} rc={rc}")
                if rc:
                    print("      " + "\n      ".join(out.splitlines()[-4:]))
            s2 = snapshot(sp)
            diff = sorted(k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k))
            check(not diff, f"{label}: second pass byte-identical (changed: {diff})")
            finals[label] = s1
    vals = list(finals.values())
    diffs = sorted(k for v in vals[1:] for k in set(v) | set(vals[0]) if v.get(k) != vals[0].get(k))
    check(not diffs, f"final trees identical across the {len(orders)} orders (differ: {sorted(set(diffs))})")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
