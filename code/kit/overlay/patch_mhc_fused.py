"""[glm53-mhc-fused] Install the fused mHC post + prenorm-GEMM prefill (GLM53_MHC_FUSED, opt-kdamhc) into the container.

Run by overlay/patch_tf_bundle.py (after install_site, LAST in the chain) only when GLM53_MHC_FUSED is non-empty:
  1    copy glm53_mhc_fused.py and glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so (tools/mhcfused/build.py) from
       the bundle's overlay dir (/opt/glm53/tf/overlay) into site-packages and append a marked block at the END of
       site-packages integrate.py that wraps integrate.plugin_register so glm53_mhc_fused.plugin_install() runs after
       every other plugin step in every vLLM process. The module re-points MHCFusedPostPreOp.forward_cuda (eager
       prefill calls >= GLM53_MHC_FUSED_MIN_T rows; decode and CUDA-graph capture keep production's op) and, under
       GLM53_MHC_SP2, model.py's _sp2_post_pre_into on the op's first call. Its first eligible call self-checks on
       deterministic synthetic rows (residual_cur bit for bit against mhc_post_tilelang, logits vs the tf32 GEMM; the
       same verdict on every TP rank; else it uninstalls itself; never raises into the engine).
  0    prints and touches nothing (stock).
  else SystemExit.
Unset/empty: patch_tf_bundle.py does not run this file at all; the module and the extension ship in the bundle's
overlay dir, NOT in site/, so the composed tree is the previous kit's byte for byte.

No collective: each TP rank decides alone, both ranks must still carry the same value (start.sh forwards it to both
ranks; boot_checks should compare the '[glm53-mhc-fused]' lines of the two ranks).
Idempotent / fail closed like patch_moe_e4m3.py (whose block it follows: install_site re-copies integrate.py on every
bundle pass, so the blocks are always re-appended in the chain's order). Env overrides for tests: GLM53_OPT,
GLM53_SITEPKG.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENV = "GLM53_MHC_FUSED"
TAG = "[glm53-mhc-fused]"
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf" / "overlay"
SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
FILES = ("glm53_mhc_fused.py", "glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so")
MARK_BEGIN = "# [glm53-mhc-fused] BEGIN armed by overlay/patch_mhc_fused.py (GLM53_MHC_FUSED=1); glm53_mhc_fused.py"
MARK_END = "# [glm53-mhc-fused] END"
BLOCK = f"""

{MARK_BEGIN}
def _glm53_mhc_fused_after(_register):
    def plugin_register() -> None:
        _register()
        try:  # LAST: MHCFusedPostPreOp.forward_cuda re-pointed for eager prefill calls only
            import glm53_mhc_fused
            glm53_mhc_fused.plugin_install()
        except Exception as exc:  # noqa: BLE001
            _log.warning("glm53_mhc_fused not loaded: %r", exc)
    plugin_register.__doc__ = _register.__doc__
    plugin_register._glm53_mhc_fused_armed = True
    return plugin_register


plugin_register = _glm53_mhc_fused_after(plugin_register)
{MARK_END}
"""


def arm(integrate_py: Path) -> str:
    src = integrate_py.read_text()
    if MARK_BEGIN in src:
        i = src.index(MARK_BEGIN)
        j = src.find(MARK_END, i)
        if j < 0 or src[i:j + len(MARK_END)] != BLOCK.strip("\n") or not src.endswith(BLOCK) \
                or src.count(MARK_BEGIN) != 1:
            raise SystemExit(f"{TAG} {integrate_py}: a different / moved arming block (drift); refusing")
        return "already armed"
    if "\ndef plugin_register(" not in src or "_log = " not in src:
        raise SystemExit(f"{TAG} {integrate_py}: no plugin_register / _log (not the tf-exl3-fork integrate.py)")
    tmp = integrate_py.with_suffix(".py.glm53-mhc-fused.tmp")
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
        print(f"{TAG} {ENV}={val or '(unset)'} -> nothing installed (production's mHC op)")
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
        tmp = SITEPKG / (f + ".glm53-mhc-fused.tmp")
        shutil.copy2(OPT / f, tmp)
        os.replace(tmp, SITEPKG / f)
        print(f"{TAG} {f}: installed")
    print(f"{TAG} integrate.py: {arm(integrate_py)} (glm53_mhc_fused.plugin_install runs last in plugin_register)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
