"""opt-kdamhc: overlay/patch_mhc_fused.py mechanics (CPU, no torch). Exit 1 on failure.

P0 unset / 0 -> nothing copied, integrate.py byte-identical
P1 =1 -> both files copied, integrate.py armed (block at the end), second run 'already armed' (byte-identical)
P2 a bundle pass with GLM53_MOE_E4M3=1 too: install_site re-copies integrate.py, moe arms, then mhc_fused arms: both
   blocks present once, mhc_fused's last; a second full pass reproduces the same bytes
P3 the armed plugin_register (exec'd on a stub integrate.py) calls the previous register THEN plugin_install;
   an exception inside plugin_install is logged, not raised
P4 fail closed: bad value, missing extension, drifted block, foreign integrate.py -> SystemExit
Run: python3 tests/opt_kdamhc/test_patch_mhc_fused.py <kit dir>   (kit dir has site/integrate.py)
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2"))
FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def load(name, path, env):
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    base = Path(tempfile.mkdtemp(prefix="pmf-", dir=os.environ.get("SCRATCH", None)))
    opt = base / "opt"
    ov = opt / "tf" / "overlay"
    sp = base / "site"
    ov.mkdir(parents=True)
    sp.mkdir()
    ext = "glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so"
    shutil.copy2(ROOT / "overlay" / "glm53_mhc_fused.py", ov)
    (ov / ext).write_bytes(b"\x7fELF-stub")
    for f in ("glm53_moe_e4m3.py", "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"):
        (ov / f).write_bytes(b"stub")
    pristine = (KIT / "site" / "integrate.py").read_text()

    def install_site():
        (sp / "integrate.py").write_text(pristine)

    env = {"GLM53_OPT": str(opt), "GLM53_SITEPKG": str(sp)}
    install_site()
    os.environ.pop("GLM53_MHC_FUSED", None)
    P = load("patch_mhc_fused", ROOT / "overlay" / "patch_mhc_fused.py", env)
    M = load("patch_moe_e4m3", ROOT / "overlay" / "patch_moe_e4m3.py", env)
    for v in ("", "0"):
        os.environ["GLM53_MHC_FUSED"] = v
        P.main([])
        check((sp / "integrate.py").read_text() == pristine and not (sp / ext).exists(), f"P0 value {v!r}: untouched")
    os.environ["GLM53_MHC_FUSED"] = "1"
    P.main([])
    s1 = (sp / "integrate.py").read_text()
    check((sp / ext).read_bytes() == b"\x7fELF-stub" and (sp / "glm53_mhc_fused.py").read_bytes() ==
          (ROOT / "overlay" / "glm53_mhc_fused.py").read_bytes(), "P1 both files copied")
    check(s1 == pristine + P.BLOCK, "P1 integrate.py = pristine + block")
    P.main([])
    check((sp / "integrate.py").read_text() == s1, "P1 second run: already armed, byte-identical")

    def bundle_pass():
        install_site()
        os.environ["GLM53_MOE_E4M3"] = "1"
        M.main([])
        P.main([])
        return (sp / "integrate.py").read_text()
    a = bundle_pass()
    b = bundle_pass()
    check(a == pristine + M.BLOCK + P.BLOCK, "P2 moe block then mhc_fused block, each once")
    check(a == b, "P2 a second full bundle pass reproduces the same bytes")

    # P3: semantics of the armed wrapper on a stub integrate.py
    calls = []
    stub = "import logging\n_log = logging.getLogger('stub')\n\ndef plugin_register():\n    CALLS.append('prev')\n"
    mod = type(sys)("glm53_mhc_fused")
    mod.plugin_install = lambda: calls.append("install")
    sys.modules["glm53_mhc_fused"] = mod
    ns = {"CALLS": calls}
    exec(compile(stub + P.BLOCK, "integrate_stub.py", "exec"), ns)
    ns["plugin_register"]()
    check(calls == ["prev", "install"] and ns["plugin_register"]._glm53_mhc_fused_armed, "P3 previous register, then "
          "plugin_install")

    def boom():
        raise RuntimeError("x")
    mod.plugin_install = boom
    try:
        ns["plugin_register"]()
        ok = True
    except Exception:  # noqa: BLE001
        ok = False
    check(ok, "P3 an exception in plugin_install is logged, not raised")

    # P4: fail closed
    os.environ["GLM53_MHC_FUSED"] = "yes"
    check(raises(lambda: P.main([])), "P4 bad value -> SystemExit")
    os.environ["GLM53_MHC_FUSED"] = "1"
    (ov / ext).rename(ov / (ext + ".x"))
    check(raises(lambda: P.main([])), "P4 missing extension -> SystemExit")
    (ov / (ext + ".x")).rename(ov / ext)
    install_site()
    (sp / "integrate.py").write_text(pristine + P.BLOCK.replace("runs", "ran").replace("LAST", "last"))
    check(raises(lambda: P.main([])), "P4 drifted block -> SystemExit")
    (sp / "integrate.py").write_text(pristine + P.BLOCK + "\n# trailing\n")
    check(raises(lambda: P.main([])), "P4 moved block (not at the end) -> SystemExit")
    (sp / "integrate.py").write_text("x = 1\n")
    check(raises(lambda: P.main([])), "P4 foreign integrate.py -> SystemExit")
    shutil.rmtree(base, ignore_errors=True)
    print("RESULT:", "ALL OK" if not FAIL else f"{len(FAIL)} FAIL", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
