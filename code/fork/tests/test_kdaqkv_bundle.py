"""Host-only test of the GLM53_KDA_STRIDED_QKV wiring (no GPU, no torch).

Exercises the real dispatch path — overlay/patch_tf_bundle.py's PATCHES table running the shipped
overlay/patch_kda_strided_qkv.py — against a scratch site-packages seeded with the two target files
(the 487ecf187 text), and asserts:
  - unset/empty env  -> the bundle prints "unset -> skipped (stock)" and the files are BYTE-IDENTICAL
    (the default-OFF contract);
  - env = 1          -> both files become exactly the overlay's prepare() output (the text the GPU rig
    loads as `fla_patched`), a second run says "already present" and changes nothing;
  - a drifted file   -> SystemExit naming the preflight, nothing written (fail closed);
  - env = 0          -> stock, both files byte-identical (review addition);
  - env not in 0/1   -> the overlay refuses to install.

Run: python3 tests/test_kdaqkv_bundle.py   (any host; the target text comes from $GLM53_FLA_SRC_VLLM,
default $TF_EXL3_ASSETS/vllm-src — the image's own text is checked for the same anchors by
tests/test_kda_strided_qkv.py on the GPU).
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import tempfile
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GLM53_FLA_SRC_VLLM", os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "vllm-src"))) / "vllm"
OPS = SRC / "third_party/flash_linear_attention/ops"
TARGETS = ("fused_recurrent.py", "kda.py")
ENVVARS = ("GLM53_KDA_STRIDED_QKV", "GLM53_OPT", "GLM53_SITEPKG")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_bundle(env: str | None, opt_root: Path, sitepkg: Path, uniq: str) -> str:
    """Run overlay/patch_tf_bundle.py's main() once, with GLM53_KDA_STRIDED_QKV=<env> (None = unset)."""
    old = {k: os.environ.pop(k, None) for k in ENVVARS}
    os.environ["GLM53_OPT"] = str(opt_root / "glm53")
    os.environ["GLM53_SITEPKG"] = str(sitepkg)
    if env is not None:
        os.environ["GLM53_KDA_STRIDED_QKV"] = env
    try:
        out = StringIO()
        spec = importlib.util.spec_from_file_location(f"ptb_{uniq}", REPO / "overlay/patch_tf_bundle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with contextlib.redirect_stdout(out):
            rc = mod.main()
        return out.getvalue() + ("" if rc is None else f"(rc={rc})")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_scratch(root: Path) -> dict[str, str]:
    """A scratch OPT tree (empty site/ + the shipped overlay) and a site-packages with the pristine
    targets. Other PATCHES entries are skipped by their unset env vars; install_site() copies an
    empty site/ (0 entries)."""
    opt_root = root / "opt"
    (opt_root / "glm53" / "tf" / "site").mkdir(parents=True)
    (opt_root / "glm53" / "tf" / "overlay").mkdir(parents=True)
    shutil.copy2(REPO / "overlay/patch_kda_strided_qkv.py", opt_root / "glm53/tf/overlay")
    sitepkg = root / "dist-packages"
    opsdir = sitepkg / "vllm/third_party/flash_linear_attention/ops"
    opsdir.mkdir(parents=True)
    texts = {t: (OPS / t).read_text() for t in TARGETS}
    for t, body in texts.items():
        (opsdir / t).write_text(body)
    return texts


def main() -> None:
    checks = 0
    failed: list[str] = []

    def ok(cond: bool, msg: str) -> None:
        nonlocal checks
        checks += 1
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            failed.append(msg)

    pk = load("pk_overlay", REPO / "overlay/patch_kda_strided_qkv.py")
    expected = {
        t: pk.prepare(source=(OPS / t).read_text(), hunks=h)
        for t, h in ((TARGETS[0], pk.FREC_HUNKS), (TARGETS[1], pk.KDA_HUNKS))
    }

    root = Path(tempfile.mkdtemp(prefix="kdaqkv-bundle-"))
    texts = build_scratch(root)
    opt_root = root / "opt"
    sitepkg = root / "dist-packages"
    opsdir = sitepkg / "vllm/third_party/flash_linear_attention/ops"

    # --- OFF: unset -> skipped, byte-identical
    out = run_bundle(None, opt_root, sitepkg, "off")
    ok("patch_kda_strided_qkv.py: GLM53_KDA_STRIDED_QKV unset -> skipped (stock)" in out,
       f"OFF: bundle skipped the feature ({out.strip().splitlines()[-1][:100]})")
    for t in TARGETS:
        ok((opsdir / t).read_text() == texts[t], f"OFF: {t} byte-identical (default-OFF contract)")

    # --- ON: applies exactly the overlay's bytes, then idempotent
    out1 = run_bundle("1", opt_root, sitepkg, "on1")
    ok("patched" in out1, f"ON: bundle applied ({out1.strip().splitlines()[-1][:100]})")
    for t in TARGETS:
        ok((opsdir / t).read_text() == expected[t], f"ON: {t} == the overlay's prepare() output")
    out2 = run_bundle("1", opt_root, sitepkg, "on2")
    ok("already present" in out2, f"ON twice: idempotent ({out2.strip().splitlines()[-1][:100]})")
    for t in TARGETS:
        ok((opsdir / t).read_text() == expected[t], f"ON twice: {t} unchanged")

    # --- drift: fail closed, nothing written. [review] start from PRISTINE fused_recurrent.py so "nothing written"
    # is observable (the implementer's version drifted kda.py after fused_recurrent.py was already patched, where a
    # partial write could not be told apart from "already present")
    (opsdir / "fused_recurrent.py").write_text(texts["fused_recurrent.py"])
    drifted = texts["kda.py"].replace("q=q.contiguous(),", "q=q.contiguous(),  # drifted")
    (opsdir / "kda.py").write_text(drifted)
    try:
        run_bundle("1", opt_root, sitepkg, "drift")
        ok(False, "drift: the bundle aborted")
    except SystemExit as exc:
        ok("preflight failed" in str(exc), f"drift: fail closed ({str(exc).strip()[:90]})")
    except Exception as exc:  # noqa: BLE001
        ok(False, f"drift: unexpected {type(exc).__name__}: {exc}")
    ok((opsdir / "fused_recurrent.py").read_text() == texts["fused_recurrent.py"],
       "drift: fused_recurrent.py NOT written (both files preflighted before any write)")
    ok((opsdir / "kda.py").read_text() == drifted, "drift: kda.py left as found")

    # --- [review] explicit 0 (the r16k start.sh accepts empty/0/1): stock, both files byte-identical
    for t in TARGETS:
        (opsdir / t).write_text(texts[t])
    out0 = run_bundle("0", opt_root, sitepkg, "zero")
    ok("GLM53_KDA_STRIDED_QKV=0: stock, files untouched" in out0, f"value 0: stock ({out0.strip().splitlines()[-1][:100]})")
    for t in TARGETS:
        ok((opsdir / t).read_text() == texts[t], f"value 0: {t} byte-identical")

    # --- wrong value: refuses to install
    try:
        run_bundle("true", opt_root, sitepkg, "wrong")
        ok(False, "value 'true': refused")
    except SystemExit as exc:
        ok("must be exactly 1" in str(exc), f"value 'true': refused ({str(exc).strip()[:90]})")

    shutil.rmtree(root)
    print(f"checks: {checks - len(failed)}/{checks} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
