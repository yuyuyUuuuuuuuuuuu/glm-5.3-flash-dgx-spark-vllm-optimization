"""[glm53-moe-fused16] Install the P16 routed-MoE prefill (GLM53_MOE_FUSED16, docs/MOE3.md) into the serving container.

Run by overlay/patch_tf_bundle.py (after install_site, before patch_moe_e4m3.py) only when GLM53_MOE_FUSED16 is
non-empty:
  1    copy glm53_moe_fused16.py and glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so (the extension it shares with
       GLM53_MOE_E4M3) from the bundle's overlay dir (/opt/glm53/tf/overlay) into site-packages, and arm the feature:
       a marked block in site-packages integrate.py that wraps integrate.plugin_register so
       glm53_moe_fused16.plugin_install() runs after every original plugin step (prefill cap / the TF K2 hook have
       fingerprinted production's apply_exl3_grouped_fat by then; the wrapper exposes production's function as
       _tf_exl3_orig for later checks). The block goes at the end of integrate.py, or right BEFORE glm53_moe_e4m3's
       block when that one is already there (it must stay last: its arm() requires it at the end).
  0    prints and touches nothing (stock).
  else SystemExit.
Unset/empty: patch_tf_bundle.py does not run this file at all (nothing of the feature reaches site-packages).

Idempotent: integrate.py with the marker block (anywhere) -> "already armed" (the block is compared, a different one =
drift -> SystemExit); files are copied every run (same bytes). Fail closed: a missing source file, an integrate.py
without plugin_register, or a block drift raises SystemExit. Atomic replace; pyc cleared.
Env overrides for tests: GLM53_OPT (default /opt/glm53), GLM53_SITEPKG (default site-packages).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV = "GLM53_MOE_FUSED16"
TAG = "[glm53-moe-fused16]"
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf" / "overlay"
SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
FILES = ("glm53_moe_fused16.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so")
E4M3_MARK = "# [glm53-moe-e4m3] BEGIN"
MARK_BEGIN = "# [glm53-moe-fused16] BEGIN armed by overlay/patch_moe_fused16.py (GLM53_MOE_FUSED16=1); docs/MOE3.md"
MARK_END = "# [glm53-moe-fused16] END"
BLOCK = f"""

{MARK_BEGIN}
def _glm53_moe_fused16_after(_register):
    def plugin_register() -> None:
        _register()
        try:  # after the original steps: wraps production's apply_exl3_grouped_fat (prefill E3 tier only)
            import glm53_moe_fused16
            glm53_moe_fused16.plugin_install()
        except Exception as exc:  # noqa: BLE001
            _log.warning("glm53_moe_fused16 not loaded: %r", exc)
    plugin_register.__doc__ = _register.__doc__
    plugin_register._glm53_moe_fused16_armed = True
    return plugin_register


plugin_register = _glm53_moe_fused16_after(plugin_register)
{MARK_END}
"""


def arm(integrate_py: Path) -> str:
    src = integrate_py.read_text()
    block = BLOCK.strip("\n")
    if MARK_BEGIN in src:
        i = src.index(MARK_BEGIN)
        j = src.find(MARK_END, i)
        if j < 0 or src[i:j + len(MARK_END)] != block or src.count(MARK_BEGIN) != 1:
            raise SystemExit(f"{TAG} {integrate_py}: a different / duplicated arming block (drift); refusing")
        e = src.find(E4M3_MARK)
        if e >= 0 and e < i:
            raise SystemExit(f"{TAG} {integrate_py}: the block sits after glm53_moe_e4m3's (moved); refusing")
        return "already armed"
    if "\ndef plugin_register(" not in src or "_log = " not in src:
        raise SystemExit(f"{TAG} {integrate_py}: no plugin_register / _log (not the tf-exl3-fork integrate.py)")
    e = src.find("\n\n" + E4M3_MARK)
    new = src + BLOCK if e < 0 else src[:e] + BLOCK + src[e:]
    tmp = integrate_py.with_suffix(".py.glm53-moe-fused16.tmp")
    tmp.write_text(new)
    os.chmod(tmp, integrate_py.stat().st_mode & 0o777)
    os.replace(tmp, integrate_py)
    pyc = integrate_py.parent / "__pycache__"
    for p in pyc.glob("integrate.*.pyc") if pyc.is_dir() else ():
        p.unlink()
    return "armed"


def main(argv=None) -> int:
    val = os.environ.get(ENV, "").strip()
    if val in ("", "0"):
        print(f"{TAG} {ENV}={val or '(unset)'} -> nothing installed (stock prefill MoE kernels)")
        return 0
    if val != "1":
        raise SystemExit(f"{TAG} {ENV} must be empty, 0 or 1 (not printed further)")
    for f in FILES:
        if not (OPT / f).is_file() or (OPT / f).stat().st_size == 0:
            raise SystemExit(f"{TAG} {ENV}=1 but {OPT / f} is missing")
    integrate_py = SITEPKG / "integrate.py"
    if not integrate_py.is_file():
        raise SystemExit(f"{TAG} {integrate_py} missing (the bundle's site/ must be installed first)")
    for f in FILES:
        tmp = SITEPKG / (f + ".glm53-moe-fused16.tmp")
        shutil.copy2(OPT / f, tmp)
        os.replace(tmp, SITEPKG / f)
        print(f"{TAG} {f}: installed")
    print(f"{TAG} integrate.py: {arm(integrate_py)} (glm53_moe_fused16.plugin_install runs after the original "
          f"plugin steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
