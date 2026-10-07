"""overlay/patch_moe_e4m3.py (the bundle patch of GLM53_MOE_E4M3), CPU only, in temp dirs (never site-packages):
  P1 unset / empty / 0 -> nothing installed, integrate.py untouched; an invalid value -> SystemExit
  P2 1 -> glm53_moe_e4m3.py + the extension copied byte for byte, integrate.py = the original + the marker block only
  P3 second run -> "already armed", integrate.py unchanged; a drifted block -> SystemExit; missing source -> SystemExit
  P4 the armed integrate.plugin_register runs every original step first and glm53_moe_e4m3.plugin_install LAST
     (a stub module records the order), exceptions in it never escape
  P5 patch_tf_bundle.py (overlay/ == launcher/overlay/) registers ("patch_moe_e4m3.py", "GLM53_MOE_E4M3") last
Run: python3 tests/moee4m3/test_patch_moe_e4m3.py   (host python is enough: no torch needed)
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
PATCH = REPO / "overlay" / "patch_moe_e4m3.py"
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


def run_patch(opt: Path, site: Path, val):
    env = dict(os.environ, GLM53_OPT=str(opt), GLM53_SITEPKG=str(site))
    env.pop("GLM53_MOE_E4M3", None)
    if val is not None:
        env["GLM53_MOE_E4M3"] = val
    return subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)


def fresh(td: Path):
    opt = td / "opt"
    ovl = opt / "tf" / "overlay"
    site = td / "site"
    ovl.mkdir(parents=True)
    site.mkdir()
    shutil.copy2(REPO / "overlay" / "glm53_moe_e4m3.py", ovl / "glm53_moe_e4m3.py")
    (ovl / "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so").write_bytes(b"\x7fELF fake extension bytes")
    shutil.copy2(REPO / "integrate.py", site / "integrate.py")
    return opt, ovl, site


def main():
    orig = (REPO / "integrate.py").read_text()
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        opt, ovl, site = fresh(td)
        for v in (None, "", "0"):
            r = run_patch(opt, site, v)
            ck(r.returncode == 0 and "nothing installed" in r.stdout and not (site / "glm53_moe_e4m3.py").exists()
               and (site / "integrate.py").read_text() == orig, f"P1 GLM53_MOE_E4M3={v!r}: nothing installed")
        for v in ("2", "on", "true"):
            r = run_patch(opt, site, v)
            ck(r.returncode != 0 and (site / "integrate.py").read_text() == orig, f"P1 invalid {v!r}: refused")
        r = run_patch(opt, site, "1")
        ck(r.returncode == 0 and "integrate.py: armed" in r.stdout, f"P2 =1 applied ({r.stdout.strip()[-80:]})")
        for f in ("glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"):
            ck((site / f).read_bytes() == (ovl / f).read_bytes(), f"P2 {f} copied byte for byte")
        armed = (site / "integrate.py").read_text()
        spec = importlib.util.spec_from_file_location("pm", PATCH)
        pm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pm)
        ck(armed == orig + pm.BLOCK, "P2 integrate.py = original + the marker block, nothing else")
        r = run_patch(opt, site, "1")
        ck(r.returncode == 0 and "already armed" in r.stdout and (site / "integrate.py").read_text() == armed,
           "P3 second run: already armed, unchanged")
        (site / "integrate.py").write_text(orig + pm.BLOCK.replace("        _register()\n", "        _register()  # edited\n"))
        r = run_patch(opt, site, "1")
        ck(r.returncode != 0 and "drift" in (r.stderr + r.stdout), "P3 drifted block: refused")
        (site / "integrate.py").write_text(armed)
        (ovl / "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so").unlink()
        r = run_patch(opt, site, "1")
        ck(r.returncode != 0 and "missing" in (r.stderr + r.stdout), "P3 missing extension: refused")
        # P4: the armed plugin_register order, with stub feature modules recording calls
        stubdir = td / "stubs"
        stubdir.mkdir()
        (stubdir / "glm53_moe_e4m3.py").write_text(
            "import builtins\ndef plugin_install():\n    builtins.ORDER.append('e4m3')\n")
        (stubdir / "glm53_smallops_install.py").write_text(
            "import builtins\ndef plugin_install():\n    builtins.ORDER.append('smallops')\n")
        code = (
            "import builtins, os, sys\nbuiltins.ORDER = []\n"
            f"sys.path[:0] = [{str(stubdir)!r}, {str(site)!r}]\n"
            "for k in list(os.environ):\n    k.startswith(('GLM53_', 'TF_EXL3')) and os.environ.pop(k)\n"
            "import integrate\nintegrate.plugin_register()\n"
            "print('ORDER', builtins.ORDER, getattr(integrate.plugin_register, '_glm53_moe_e4m3_armed', False))\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("ORDER")]
        ck(r.returncode == 0 and line and line[0] == "ORDER ['smallops', 'e4m3'] True",
           f"P4 armed plugin_register: the last original step first, glm53_moe_e4m3.plugin_install last ({line})")
        (stubdir / "glm53_moe_e4m3.py").write_text("def plugin_install():\n    raise RuntimeError('boom')\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        ck(r.returncode == 0, "P4 an exception in plugin_install does not escape plugin_register")
    a = (REPO / "overlay" / "patch_tf_bundle.py").read_text()
    b = (REPO / "launcher" / "overlay" / "patch_tf_bundle.py").read_text()
    ck(a == b, "P5 overlay/patch_tf_bundle.py == launcher/overlay/patch_tf_bundle.py")
    ck('    ("patch_moe_e4m3.py", "GLM53_MOE_E4M3"),' in a and
       a.index('("patch_moe_e4m3.py"') > a.index('("patch_mhc_sp2.py"'),
       "P5 patch_tf_bundle.py runs patch_moe_e4m3.py, after every other patch")
    print(f"checks: {n_ok}/{n} passed")
    print("RESULT:", "PASS" if not failed else "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
