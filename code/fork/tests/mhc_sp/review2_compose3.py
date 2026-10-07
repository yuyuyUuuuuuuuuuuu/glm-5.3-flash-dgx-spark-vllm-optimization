"""Adversarial review 2 (mhcsp): does patch_mhc_sp.py compose with r16n's patch_flashkda.py and r16e4's
patch_moe_e4m3.py in ONE pass (both orders) and in a SECOND pass on the same tree?

Host-only (no torch, no GPU, nothing in any container is written): a fake site-packages tree under a temp dir holds
the image's vllm/models/glm5next/nvidia/{model,kda}.py (copied out of the image beforehand: IMG_VLLM=<dir> holding
models/glm5next/nvidia/) and the kits' site/ glm53_prefill_quickwins.py, glm53_moeglue.py, integrate.py; the three
patches run from the three kits' overlay/ dirs (GLM53_SITEPKG / GLM53_TF_OVERLAY / GLM53_OPT point at the temp tree).

Checks per order: every patch rc 0; second pass rc 0 and byte-identical; final trees of both orders byte-identical;
the fingerprints the runtime drift guards will compute are in their tables (quickwins mhc_aux/mhc_mean on the SP
forwards, kda_conv on the FlashKDA _forward, moeglue WARM on the SP layer forward plain + quickwins-transplanted);
the quickwins transplant anchors (MHC_AUX, MHC_FINAL, KDA_CONV) still anchor exactly once on the final sources.

Run: IMG_VLLM=<extracted vllm dir> python3 tests/mhc_sp/review2_compose3.py
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KITS = {"msp": Path(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16msp")), "n": Path(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16n")),
        "e4": Path(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16e4"))}
IMG = Path(os.environ["IMG_VLLM"])
NV = "vllm/models/glm5next/nvidia"
FAIL: list[str] = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def build(root: Path) -> tuple[Path, Path]:
    sp = root / "site"
    (sp / NV).mkdir(parents=True)
    for f in ("model.py", "kda.py"):
        shutil.copy(IMG / "models/glm5next/nvidia" / f, sp / NV / f)
    for f in ("glm53_prefill_quickwins.py", "glm53_moeglue.py", "integrate.py"):
        for k in KITS.values():                                   # identical in the three kits (checked below)
            assert (k / "site" / f).read_bytes() == (KITS["msp"] / "site" / f).read_bytes(), (k, f)
        shutil.copy(KITS["msp"] / "site" / f, sp / f)
    ov = root / "opt" / "tf" / "overlay"
    ov.mkdir(parents=True)
    for k in KITS.values():
        for f in (k / "overlay").iterdir():
            if f.is_file() and not (ov / f.name).exists():
                shutil.copy(f, ov / f.name)
    return sp, ov


PATCHES = {"mhcsp": ("patch_mhc_sp.py", "GLM53_MHC_SP"), "flashkda": ("patch_flashkda.py", "GLM53_KDA_FLASHKDA"),
           "e4m3": ("patch_moe_e4m3.py", "GLM53_MOE_E4M3")}


def run(sp: Path, ov: Path, name: str) -> tuple[int, str]:
    f, env_name = PATCHES[name]
    env = dict(os.environ, GLM53_SITEPKG=str(sp), GLM53_TF_OVERLAY=str(ov), GLM53_OPT=str(ov.parent.parent),
               PYTHONDONTWRITEBYTECODE="1", **{env_name: "1"})
    for k in ("GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY", "GLM53_MOEGLUE_PY", "GLM53_KDA_PY"):
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
    sys.path.insert(0, str(KITS["msp"] / "overlay"))
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
    ns: dict = {}
    qw_mod_src = qw
    for name in ("MHC_AUX", "MHC_FINAL", "KDA_CONV"):
        i = qw_mod_src.index(f"\n{name} = (")
        chunk = qw_mod_src[i:i + 6000]
        olds = re.findall(r'"""(.*?)"""', chunk, re.S)
        tgt = kda if name == "KDA_CONV" else model
        check(len(olds) >= 2 and tgt.count(olds[0]) == 1, f"{label}: quickwins {name} anchor count on the final "
              f"source = {tgt.count(olds[0]) if olds else 'n/a'} (need 1)")
    for name in ("model.py", "kda.py"):
        compile((sp / NV / name).read_text(), name, "exec")
    check("glm53_moe_e4m3" in (sp / "integrate.py").read_text(), f"{label}: integrate.py armed for e4m3")
    sys.path.pop(0)
    del ns


def main() -> int:
    finals = {}
    for order in (("mhcsp", "flashkda", "e4m3"), ("flashkda", "e4m3", "mhcsp"), ("e4m3", "mhcsp", "flashkda")):
        label = ">".join(order)
        print(f"== order {label}")
        with tempfile.TemporaryDirectory() as td:
            sp, ov = build(Path(td))
            for n in order:
                rc, out = run(sp, ov, n)
                check(rc == 0, f"{label}: pass 1 {n} rc={rc}")
                print("      " + "\n      ".join(out.splitlines()[-4:]))
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
    check(not diffs, f"final trees identical across the 3 orders (differ: {sorted(set(diffs))})")
    print("ALL OK" if not FAIL else f"FAILURES: {len(FAIL)}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
