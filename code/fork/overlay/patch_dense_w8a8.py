#!/usr/bin/env python3
"""[glm53-w8a8] Install the W8A8 prefill module (fp8_w8a8.py + tf_fp8_w8a8_ext) into site-packages, ONLY when
GLM53_DENSE_W8A8=1 (bundle overlay, run by patch_tf_bundle.py like every overlay patch; unset/empty never reaches
this file, 0 is handled here itself, start.sh validates empty/0/1 and refuses anything else).

What the feature is (docs/DENSE_W8A8.md): pf3000 plan step 3, the W8A8 (per-token fp8 activations x the stored
per-channel fp8 weights) cutlass path for the dense + shared-expert FP8 linears of the PREFILL path only - the
pf3000 kill test measured 1.14-2.41x per GEMM against production's current path (TileLang W8A16 large-M + Marlin),
net about +510..519 ms per 13,824-token chunk per rank, with the weights kept in the Marlin repack (the standard
fp8 operand cutlass needs is repacked per call by the byte-exact kernel, no resident copy: KV pool 2.00M ->
1.96M). Decode keeps Marlin: the module declines CUDA-graph capture and M < GLM53_DENSE_W8A8_MIN_M (512), and
everything it cannot serve falls back to production's own apply (the wrapper wraps fp8_gemv's, which wraps
production's).

Install states:
  1        -> site-packages/fp8_w8a8.py + tf_fp8_w8a8_ext*.so copied from /opt/glm53/tf/overlay (atomic, verified,
              idempotent, stale .pyc removed) and the site integrate.py ARMED (see the r16y note below). The module
              itself is still INERT until vLLM loads its plugins: the armed block in integrate.plugin_register
              imports it only when GLM53_DENSE_W8A8 is set (same env), so an install without the env changes nothing.
  0        -> stock: any previously installed copy is REMOVED and integrate.py DISARMED (an on -> off toggle returns
              to the pristine tree, byte for byte) and nothing is installed.
  other    -> SystemExit (the launcher validated already; defense in depth).
Both files fail closed: a missing source, a copy that does not verify byte for byte, or a site-packages file that
is neither the source's bytes nor an already-installed copy of them is a SystemExit (stops the container start),
like every bundle overlay. Markers (boot_checks B.15):
  installed: "[glm53-w8a8] site-packages/fp8_w8a8.py: installed" / "...tf_fp8_w8a8_ext...: installed"
             + "[glm53-w8a8] integrate.py: armed ..."
  0:         "[glm53-w8a8] GLM53_DENSE_W8A8=0: stock"
Env overrides for tests: GLM53_SITEPKG (site-packages), GLM53_OPT (the bundle root, default /opt/glm53).

r16y integration (the one edit to the reviewed w8a8 wiring): the w8a8 branch carried the conditional fp8_w8a8
import in the bundle's site integrate.py itself, which would have made the kit's site/ differ from the previous
kit's (r16n) with every switch unset. The arming therefore moved here, to compose time, exactly like
overlay/patch_moe_e4m3.py arms integrate.py for GLM53_MOE_E4M3 (that block appends at the END of integrate.py,
this one inserts after the fp8_roof step - the two never overlap, and the plugin order is the one the w8a8 branch
shipped: fp8_w8a8 installs AFTER fp8_gemv, so its wrapper is the outer one). The branch's repo-root integrate.py
keeps the same block for the repo-level rigs; the kit's site integrate.py is r16n's until =1 arms it.
Env overrides for tests: GLM53_SITEPKG (site-packages), GLM53_OPT (the bundle root, default /opt/glm53).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MARK = "[glm53-w8a8]"
MODULE = "fp8_w8a8.py"
EXT = "tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so"

# ---- the site integrate.py arming (compose-time, patch_moe_e4m3.py's kind of marked block; see the docstring)
ARM_MARK_BEGIN = "# [glm53-w8a8] BEGIN armed by overlay/patch_dense_w8a8.py (GLM53_DENSE_W8A8=1); docs/DENSE_W8A8.md"
ARM_MARK_END = "# [glm53-w8a8] END"
# The anchor is the END of integrate.plugin_register's fp8_roof step (the reviewed w8a8 branch inserted its
# conditional import at the same place). Exactly one occurrence in the bundle's integrate.py; arming inserts the
# marked try-block below verbatim, disarming removes it again byte for byte.
ARM_ANCHOR = ('    except Exception as exc:  # noqa: BLE001\n'
              '        _log.warning("tf_fp8_roof not installed: %r", exc)\n')
ARM_BLOCK = (
    '    ' + ARM_MARK_BEGIN + '\n'
    '    # The W8A8 wrapper must install AFTER fp8_gemv\'s (above) so it is the outer one and a served prefill\n'
    '    # call never reaches the large-M path; everything the module declines falls through to production.\n'
    '    try:\n'
    '        if os.environ.get("GLM53_DENSE_W8A8", "").strip().lower() in {"1", "on", "true", "yes"}:\n'
    '            import fp8_w8a8\n'
    '            fp8_w8a8.plugin_register()\n'
    '    except Exception as exc:  # noqa: BLE001\n'
    '        _log.warning("tf_fp8_w8a8 not installed: %r", exc)\n'
    '    ' + ARM_MARK_END + '\n'
)


def _integrate_texts(src: str) -> tuple[str, str]:
    # (armed, disarmed) forms of an integrate.py source; raises if the anchor is absent or the marks are torn.
    if ARM_MARK_BEGIN in src or ARM_MARK_END in src:
        if src.count(ARM_MARK_BEGIN) != 1 or src.count(ARM_MARK_END) != 1 \
                or src.index(ARM_MARK_END) < src.index(ARM_MARK_BEGIN) \
                or src.count(ARM_ANCHOR + ARM_BLOCK) != 1:
            raise SystemExit(f"{MARK} site integrate.py: a different / moved arming block (drift); refusing")
        armed = src
        disarmed = src.replace(ARM_ANCHOR + ARM_BLOCK, ARM_ANCHOR, 1)
    else:
        if src.count(ARM_ANCHOR) != 1:
            raise SystemExit(f"{MARK} site integrate.py: the fp8_roof anchor found {src.count(ARM_ANCHOR)}x "
                             "(not the tf-exl3-fork integrate.py this kit was built against); refusing")
        armed = src.replace(ARM_ANCHOR, ARM_ANCHOR + ARM_BLOCK, 1)
        disarmed = src
    compile(armed, "integrate.py", "exec")   # the armed file must parse: a broken insertion stops the container
    return armed, disarmed


def arm(sitepkg: Path, do_arm: bool) -> str:
    # arm (=1) or disarm (=0) site-packages integrate.py; the disarm restores the stock bytes exactly.
    integrate_py = sitepkg / "integrate.py"
    if not integrate_py.is_file():
        raise SystemExit(f"{MARK} {integrate_py} missing (the bundle's site/ must be installed first)")
    src = integrate_py.read_text()
    armed, disarmed = _integrate_texts(src)
    target = armed if do_arm else disarmed
    if target == src:
        return ("already armed" if do_arm else "already disarmed (stock)")
    tmp = integrate_py.with_suffix(".py.glm53-w8a8.tmp")
    tmp.write_text(target)
    os.chmod(tmp, integrate_py.stat().st_mode & 0o777)
    os.replace(tmp, integrate_py)
    pyc = integrate_py.parent / "__pycache__"
    if pyc.is_dir():
        for p in pyc.glob("integrate.*.pyc"):
            p.unlink()
    return ("armed (fp8_w8a8.plugin_register installs after fp8_gemv's in plugin_register)" if do_arm
            else "disarmed (stock bytes restored)")


def targets(sitepkg: Path) -> list[Path]:
    return [sitepkg / MODULE, sitepkg / EXT]


def install_one(src: Path, dst: Path) -> str:
    """Atomic copy of src -> dst with verification; returns 'installed' or 'already current'."""
    data = src.read_bytes()
    if dst.exists():
        cur = dst.read_bytes()
        if cur == data:
            return "already current"
        if not dst.is_file():
            raise SystemExit(f"{MARK} {dst} exists and is not a file")
        # a foreign file (not ours, not the source's bytes): refuse instead of overwriting
        raise SystemExit(f"{MARK} {dst} exists with other content ({len(cur)} bytes, expected {len(data)}); "
                         "refusing to overwrite")
    tmp = dst.with_name(dst.name + ".w8a8.tmp")
    tmp.write_bytes(data)
    if tmp.read_bytes() != data:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{MARK} {dst}: the staged copy does not verify")
    os.replace(tmp, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass
    return "installed"


def remove_one(dst: Path) -> str:
    if not dst.exists():
        return "absent"
    dst.unlink()
    pyc = dst.parent / "__pycache__"
    if dst.suffix == ".py":
        for p in pyc.glob(dst.stem + ".*.pyc"):
            p.unlink(missing_ok=True)
    return "removed"


def main(argv: list[str]) -> int:
    del argv
    val = os.environ.get("GLM53_DENSE_W8A8", "").strip()
    opt = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf"
    sitepkg = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
    if val == "0":
        states = [remove_one(d) for d in targets(sitepkg)]
        print(f"{MARK} GLM53_DENSE_W8A8=0: stock "
              f"({'; '.join(f'{d.name}: {s}' for d, s in zip(targets(sitepkg), states))}; integrate.py: {arm(sitepkg, False)})")
        return 0
    if val != "1":
        raise SystemExit(f"{MARK} GLM53_DENSE_W8A8 must be 0 or 1 (got {val!r})")
    for name in (MODULE, EXT):
        src = opt / "overlay" / name
        if not src.is_file():
            raise SystemExit(f"{MARK} missing {src}: the bundle does not carry the W8A8 module "
                             "(a start.sh and patch_tf_bundle.py from this kit are required)")
    for src, dst in zip((opt / "overlay" / MODULE, opt / "overlay" / EXT), targets(sitepkg)):
        state = install_one(src, dst)
        print(f"{MARK} site-packages/{dst.name}: {state} (GLM53_DENSE_W8A8=1; inert until vLLM loads its plugins "
              f"with the same env)")
    print(f"{MARK} integrate.py: {arm(sitepkg, True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
