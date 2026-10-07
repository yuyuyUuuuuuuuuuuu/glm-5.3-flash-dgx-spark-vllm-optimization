"""[glm53-moe-e4m3] Install the e4m3 routed-MoE prefill (GLM53_MOE_E4M3, docs/MOE_E4M3.md) into the serving container.

Run by overlay/patch_tf_bundle.py (after install_site) only when GLM53_MOE_E4M3 is non-empty:
  1    copy glm53_moe_e4m3.py and glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so from the bundle's overlay dir
       (/opt/glm53/tf/overlay) into site-packages, and arm the feature: append a marked block at the END of
       site-packages integrate.py that wraps integrate.plugin_register (the vllm.general_plugins entry point the
       bundle's dist-info registers) so glm53_moe_e4m3.plugin_install() runs AFTER every other plugin step - its
       wrapper of apply_exl3_experts is then the outermost one, and moeglue / the TF K2 hook (decode) still see
       production's apply_exl3_experts when they fingerprint it.
  0    prints and touches nothing (stock).
  else SystemExit (start.sh's validate_numeric_config refuses it earlier).
Unset/empty: patch_tf_bundle.py does not run this file at all, so nothing of the feature reaches site-packages and the
composed tree is the previous kit's byte for byte (the module and the extension ship in the bundle's overlay dir, NOT
in site/).

Idempotent: integrate.py with the marker block -> "already armed" (the block is compared, a different one = drift ->
SystemExit); files are copied every run (same bytes). Fail closed: a missing source file, an integrate.py without
plugin_register, or a block drift raises SystemExit, which stops the container start. Atomic replace; pyc cleared.
Env overrides for tests: GLM53_OPT (default /opt/glm53), GLM53_SITEPKG (default site-packages).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV = "GLM53_MOE_E4M3"
TAG = "[glm53-moe-e4m3]"
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf" / "overlay"
SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
FILES = ("glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so")
MARK_BEGIN = "# [glm53-moe-e4m3] BEGIN armed by overlay/patch_moe_e4m3.py (GLM53_MOE_E4M3=1); docs/MOE_E4M3.md"
MARK_END = "# [glm53-moe-e4m3] END"
BLOCK = f"""

{MARK_BEGIN}
def _glm53_moe_e4m3_after(_register):
    def plugin_register() -> None:
        _register()
        try:  # LAST: glm53_moe_e4m3's apply_exl3_experts wrapper is the outermost one (prefill calls only)
            import glm53_moe_e4m3
            glm53_moe_e4m3.plugin_install()
        except Exception as exc:  # noqa: BLE001
            _log.warning("glm53_moe_e4m3 not loaded: %r", exc)
    plugin_register.__doc__ = _register.__doc__
    plugin_register._glm53_moe_e4m3_armed = True
    return plugin_register


plugin_register = _glm53_moe_e4m3_after(plugin_register)
{MARK_END}
"""


def arm(integrate_py: Path) -> str:
    src = integrate_py.read_text()
    if MARK_BEGIN in src:
        i = src.index(MARK_BEGIN)
        j = src.find(MARK_END, i)
        if j < 0 or src[i:j + len(MARK_END)] != BLOCK.strip("\n") or not src.endswith(BLOCK):
            raise SystemExit(f"{TAG} {integrate_py}: a different / moved arming block (drift); refusing")
        return "already armed"
    if "\ndef plugin_register(" not in src or "_log = " not in src:
        raise SystemExit(f"{TAG} {integrate_py}: no plugin_register / _log (not the tf-exl3-fork integrate.py)")
    tmp = integrate_py.with_suffix(".py.glm53-moe-e4m3.tmp")
    tmp.write_text(src + BLOCK)
    os.chmod(tmp, integrate_py.stat().st_mode & 0o777)
    os.replace(tmp, integrate_py)
    pyc = integrate_py.parent / "__pycache__"
    for p in pyc.glob("integrate.*.pyc") if pyc.is_dir() else ():
        p.unlink()
    return "armed"


def main(argv=None) -> int:
    val = os.environ.get(ENV, "").strip()
    if val in ("", "0"):
        print(f"{TAG} {ENV}={val or '(unset)'} -> nothing installed (stock prefill MoE)")
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
        tmp = SITEPKG / (f + ".glm53-moe-e4m3.tmp")
        shutil.copy2(OPT / f, tmp)
        os.replace(tmp, SITEPKG / f)
        print(f"{TAG} {f}: installed")
    print(f"{TAG} integrate.py: {arm(integrate_py)} (glm53_moe_e4m3.plugin_install runs last in plugin_register)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
