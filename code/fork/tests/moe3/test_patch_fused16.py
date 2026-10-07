"""overlay/patch_moe_fused16.py (the bundle patch of GLM53_MOE_FUSED16), CPU only, temp dirs (never site-packages):
  P1 unset / empty / 0 -> nothing installed, integrate.py untouched; invalid values -> SystemExit
  P2 1 -> glm53_moe_fused16.py + the shared extension copied byte for byte; integrate.py = original + the block only
  P3 second run -> "already armed", unchanged; drifted block -> SystemExit; missing source -> SystemExit
  P4 composed with patch_moe_e4m3.py in BOTH orders: e4m3's block stays last, both patches idempotent on a second
     pass, plugin_register runs the original steps, then glm53_moe_fused16, then glm53_moe_e4m3; exceptions contained
  P5 patch_tf_bundle.py (overlay == launcher/overlay) runs patch_moe_fused16.py right before patch_moe_e4m3.py
Run: python3 tests/moe3/test_patch_fused16.py   (host python, no torch)
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "overlay" / "patch_moe_fused16.py"
PATCH_E4 = REPO / "overlay" / "patch_moe_e4m3.py"
SO = "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"
n_ok = n = 0
failed = []


def ck(cond, msg):
    global n_ok, n
    n += 1
    if cond:
        n_ok += 1
        print(f"ok   {msg}")
    else:
        failed.append(msg)
        print(f"FAIL {msg}")


def run(patch, opt, site, env_name, val):
    env = dict(os.environ, GLM53_OPT=str(opt), GLM53_SITEPKG=str(site))
    for k in ("GLM53_MOE_E4M3", "GLM53_MOE_FUSED16"):
        env.pop(k, None)
    if val is not None:
        env[env_name] = val
    return subprocess.run([sys.executable, str(patch)], env=env, capture_output=True, text=True)


def fresh(td: Path):
    opt = td / "opt"
    ovl = opt / "tf" / "overlay"
    site = td / "site"
    ovl.mkdir(parents=True)
    site.mkdir()
    for f in ("glm53_moe_fused16.py", "glm53_moe_e4m3.py"):
        shutil.copy2(REPO / "overlay" / f, ovl / f)
    (ovl / SO).write_bytes(b"\x7fELF fake extension bytes")
    shutil.copy2(REPO / "integrate.py", site / "integrate.py")
    return opt, ovl, site


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def order(td, site, fail_fused=False):
    stubdir = td / "stubs"
    stubdir.mkdir(exist_ok=True)
    (stubdir / "glm53_moe_e4m3.py").write_text("import builtins\ndef plugin_install():\n    builtins.ORDER.append('e4m3')\n")
    (stubdir / "glm53_moe_fused16.py").write_text(
        "import builtins\ndef plugin_install():\n    builtins.ORDER.append('fused16')\n" +
        ("    raise RuntimeError('boom')\n" if fail_fused else ""))
    (stubdir / "glm53_smallops_install.py").write_text(
        "import builtins\ndef plugin_install():\n    builtins.ORDER.append('smallops')\n")
    code = ("import builtins, os, sys\nbuiltins.ORDER = []\n"
            f"sys.path[:0] = [{str(stubdir)!r}, {str(site)!r}]\n"
            "for k in list(os.environ):\n    k.startswith(('GLM53_', 'TF_EXL3')) and os.environ.pop(k)\n"
            "import integrate\nintegrate.plugin_register()\nprint('ORDER', builtins.ORDER)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("ORDER")]
    return r.returncode, (line[0] if line else r.stderr[-300:])


def main():
    orig = (REPO / "integrate.py").read_text()
    pm, pe = load(PATCH), load(PATCH_E4)
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        opt, ovl, site = fresh(td)
        for v in (None, "", "0"):
            r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", v)
            ck(r.returncode == 0 and "nothing installed" in r.stdout and not (site / "glm53_moe_fused16.py").exists()
               and (site / "integrate.py").read_text() == orig, f"P1 GLM53_MOE_FUSED16={v!r}: nothing installed")
        for v in ("2", "on", "true"):
            r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", v)
            ck(r.returncode != 0 and (site / "integrate.py").read_text() == orig, f"P1 invalid {v!r}: refused")
        r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", "1")
        ck(r.returncode == 0 and "integrate.py: armed" in r.stdout, "P2 =1 applied")
        for f in ("glm53_moe_fused16.py", SO):
            ck((site / f).read_bytes() == (ovl / f).read_bytes(), f"P2 {f} copied byte for byte")
        armed = (site / "integrate.py").read_text()
        ck(armed == orig + pm.BLOCK, "P2 integrate.py = original + the block, nothing else")
        r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", "1")
        ck(r.returncode == 0 and "already armed" in r.stdout and (site / "integrate.py").read_text() == armed,
           "P3 second run: already armed, unchanged")
        (site / "integrate.py").write_text(orig + pm.BLOCK.replace("        _register()\n", "        _register()  # x\n"))
        r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", "1")
        ck(r.returncode != 0 and "drift" in (r.stderr + r.stdout), "P3 drifted block: refused")
        (site / "integrate.py").write_text(armed)
        (ovl / SO).unlink()
        r = run(PATCH, opt, site, "GLM53_MOE_FUSED16", "1")
        ck(r.returncode != 0 and "missing" in (r.stderr + r.stdout), "P3 missing extension: refused")
        (ovl / SO).write_bytes(b"\x7fELF fake extension bytes")
        # P4: both orders with the e4m3 patch
        for first, second in (("fused16", "e4m3"), ("e4m3", "fused16")):
            (site / "integrate.py").write_text(orig)
            seq = {"fused16": (PATCH, "GLM53_MOE_FUSED16"), "e4m3": (PATCH_E4, "GLM53_MOE_E4M3")}
            rs = [run(seq[k][0], opt, site, seq[k][1], "1") for k in (first, second)]
            txt = (site / "integrate.py").read_text()
            ck(all(r.returncode == 0 for r in rs) and txt == orig + pm.BLOCK + pe.BLOCK,
               f"P4 {first} then {second}: integrate.py = original + fused16 block + e4m3 block (e4m3 last)")
            rs = [run(seq[k][0], opt, site, seq[k][1], "1") for k in (first, second)]
            ck(all(r.returncode == 0 and "already armed" in r.stdout for r in rs)
               and (site / "integrate.py").read_text() == txt, f"P4 {first} then {second}: second pass idempotent")
            rc, line = order(td, site)
            ck(rc == 0 and line == "ORDER ['smallops', 'fused16', 'e4m3']",
               f"P4 {first} then {second}: plugin order original -> fused16 -> e4m3 ({line})")
        rc, line = order(td, site, fail_fused=True)
        ck(rc == 0 and line == "ORDER ['smallops', 'fused16', 'e4m3']",
           f"P4 an exception in glm53_moe_fused16.plugin_install is contained, e4m3 still runs ({line})")
    a = (REPO / "overlay" / "patch_tf_bundle.py").read_text()
    b = (REPO / "launcher" / "overlay" / "patch_tf_bundle.py").read_text()
    ck(a == b, "P5 overlay/patch_tf_bundle.py == launcher/overlay/patch_tf_bundle.py")
    i, j = a.find('("patch_moe_fused16.py", "GLM53_MOE_FUSED16")'), a.find('("patch_moe_e4m3.py", "GLM53_MOE_E4M3")')
    ck(0 <= i < j and a.index('("patch_mhc_sp2.py"') < i, "P5 patch_moe_fused16.py runs right before e4m3's")
    print(f"checks: {n_ok}/{n} passed")
    print("RESULT:", "PASS" if not failed else "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
