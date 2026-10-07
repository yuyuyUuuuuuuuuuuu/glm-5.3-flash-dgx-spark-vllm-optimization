#!/usr/bin/env python3
"""[glm53-mla-exactlens] Install the exact-length FA2 sparse-MLA planner (glm53_mla_exactlens.py) into site-packages and
arm it in the site integrate.py, ONLY when GLM53_MLA_EXACT_LENS=1 (bundle overlay, run by patch_tf_bundle.py like
every overlay patch; unset/empty never reaches this file, 0 is handled here, the start.sh validates empty/0/1).

What it fixes (docs/MLA_EXACT_LENS.md): production's FLASHINFER_MLA_SPARSE_SM90 builder plans FA2 with
``index_topk + ctx % index_kpool`` keys for every row with ctx >= index_topk, while the kpool indexer selects
``index_topk - index_kpool + ctx % index_kpool`` (511 pools + the tail): every decode row attends 4 extra keys (slot-0
copies and the NEXT row's first selected keys; the last row of a step reads stale slot ids an earlier call left in the
process-wide kv_indices, which can belong to another request). The module replaces the builder's build() (fingerprint-
pinned) so plan() gets the exact counts; self-tested on the image's real ops at the first build.

Install states (the same contract as overlay/patch_dense_w8a8.py):
  1     -> site-packages/glm53_mla_exactlens.py copied from /opt/glm53/tf/overlay (atomic, verified, idempotent; a
           foreign file of that name is refused) and the site integrate.py ARMED: a marked try-block after the
           glm53_hostloop step of plugin_register imports it (the module itself is inert unless the same env is on).
  0     -> stock: an installed copy is removed and integrate.py disarmed (byte for byte the stock file again).
  other -> SystemExit (the launcher validated already; defense in depth).
Markers (boot_checks): "[glm53-mla-exactlens] site-packages/glm53_mla_exactlens.py: installed|already current" +
"[glm53-mla-exactlens] integrate.py: armed ..."; "[glm53-mla-exactlens] GLM53_MLA_EXACT_LENS=0: stock (...)".
Env overrides for tests: GLM53_SITEPKG (site-packages), GLM53_OPT (the bundle root, default /opt/glm53).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-mla-exactlens]"
ENV = "GLM53_MLA_EXACT_LENS"
MODULE = "glm53_mla_exactlens.py"

ARM_MARK_BEGIN = ("# [glm53-mla-exactlens] BEGIN armed by overlay/patch_mla_exactlens.py (GLM53_MLA_EXACT_LENS=1); "
                  "docs/MLA_EXACT_LENS.md")
ARM_MARK_END = "# [glm53-mla-exactlens] END"
# The END of integrate.plugin_register's glm53_hostloop step: exactly one occurrence in the bundle's integrate.py.
ARM_ANCHOR = ('    except Exception as exc:  # noqa: BLE001\n'
              '        _log.warning("glm53_hostloop not loaded: %r", exc)\n')
ARM_BLOCK = (
    '    ' + ARM_MARK_BEGIN + '\n'
    '    try:\n'
    '        import glm53_mla_exactlens\n'
    '        glm53_mla_exactlens.plugin_install()\n'
    '    except Exception as exc:  # noqa: BLE001\n'
    '        _log.warning("glm53_mla_exactlens not loaded: %r", exc)\n'
    '    ' + ARM_MARK_END + '\n'
)


def integrate_texts(src: str) -> tuple[str, str]:
    """(armed, disarmed) forms of an integrate.py source; SystemExit on a missing anchor or torn marks."""
    if ARM_MARK_BEGIN in src or ARM_MARK_END in src:
        if src.count(ARM_MARK_BEGIN) != 1 or src.count(ARM_MARK_END) != 1 \
                or src.index(ARM_MARK_END) < src.index(ARM_MARK_BEGIN) \
                or src.count(ARM_ANCHOR + ARM_BLOCK) != 1:
            raise SystemExit(f"{MARK} site integrate.py: a different / moved arming block (drift); refusing")
        armed = src
        disarmed = src.replace(ARM_ANCHOR + ARM_BLOCK, ARM_ANCHOR, 1)
    else:
        if src.count(ARM_ANCHOR) != 1:
            raise SystemExit(f"{MARK} site integrate.py: the glm53_hostloop anchor found {src.count(ARM_ANCHOR)}x "
                             "(not the tf-exl3-fork integrate.py this kit was built against); refusing")
        armed = src.replace(ARM_ANCHOR, ARM_ANCHOR + ARM_BLOCK, 1)
        disarmed = src
    compile(armed, "integrate.py", "exec")
    return armed, disarmed


def arm(sitepkg: Path, do_arm: bool) -> str:
    integrate_py = sitepkg / "integrate.py"
    if not integrate_py.is_file():
        raise SystemExit(f"{MARK} {integrate_py} missing (the bundle's site/ must be installed first)")
    src = integrate_py.read_text()
    armed, disarmed = integrate_texts(src)
    target = armed if do_arm else disarmed
    if target == src:
        return "already armed" if do_arm else "already disarmed (stock)"
    tmp = integrate_py.with_suffix(".py.glm53-mla-exactlens.tmp")
    tmp.write_text(target)
    os.chmod(tmp, integrate_py.stat().st_mode & 0o777)
    os.replace(tmp, integrate_py)
    pyc = integrate_py.parent / "__pycache__"
    if pyc.is_dir():
        for p in pyc.glob("integrate.*.pyc"):
            p.unlink()
    return ("armed (glm53_mla_exactlens.plugin_install after the glm53_hostloop step)" if do_arm
            else "disarmed (stock bytes restored)")


def install_one(src: Path, dst: Path) -> str:
    data = src.read_bytes()
    if dst.exists():
        if not dst.is_file():
            raise SystemExit(f"{MARK} {dst} exists and is not a file")
        cur = dst.read_bytes()
        if cur == data:
            return "already current"
        raise SystemExit(f"{MARK} {dst} exists with other content ({len(cur)} bytes, expected {len(data)}); refusing")
    tmp = dst.with_name(dst.name + ".exactlens.tmp")
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
    for p in (dst.parent / "__pycache__").glob(dst.stem + ".*.pyc"):
        p.unlink(missing_ok=True)
    return "removed"


def main(argv: list[str]) -> int:
    del argv
    val = os.environ.get(ENV, "").strip()
    opt = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf"
    sitepkg = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
    dst = sitepkg / MODULE
    if val == "0":
        state = remove_one(dst)
        print(f"{MARK} {ENV}=0: stock ({MODULE}: {state}; integrate.py: {arm(sitepkg, False)})")
        return 0
    if val != "1":
        raise SystemExit(f"{MARK} {ENV} must be 0 or 1 (got {val!r})")
    integrate_py = sitepkg / "integrate.py"
    if not integrate_py.is_file():
        raise SystemExit(f"{MARK} {integrate_py} missing (the bundle's site/ must be installed first)")
    integrate_texts(integrate_py.read_text())   # preflight: a drifted integrate.py refuses BEFORE anything is written
    src = opt / "overlay" / MODULE
    if not src.is_file():
        raise SystemExit(f"{MARK} missing {src}: the bundle does not carry the module (a start.sh and "
                         "patch_tf_bundle.py from this kit are required)")
    print(f"{MARK} site-packages/{MODULE}: {install_one(src, dst)} ({ENV}=1; inert until vLLM loads its plugins with "
          f"the same env)")
    print(f"{MARK} integrate.py: {arm(sitepkg, True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
