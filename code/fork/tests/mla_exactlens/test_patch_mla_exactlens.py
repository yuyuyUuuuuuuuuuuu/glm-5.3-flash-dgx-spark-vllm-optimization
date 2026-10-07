#!/usr/bin/env python3
"""[glm53-mla-exactlens] host test of overlay/patch_mla_exactlens.py (no GPU, no torch).
P1 =1 on a stock site: module installed byte-identical to the bundle copy, integrate.py armed, armed file compiles and
   the block sits right after the glm53_hostloop step; second =1 run: 'already current' / 'already armed', no change
P2 =0 after =1: module removed, integrate.py byte-identical to the stock file; =0 again is a no-op
P3 refusals (SystemExit, nothing written): a value other than 0/1, a missing bundle module, a foreign file of the
   module's name, an integrate.py without the anchor, a torn arming block
P4 the bundle overlay copy overlay/glm53_mla_exactlens.py is byte-identical to the repo-root module
Run: python3 tests/mla_exactlens/test_patch_mla_exactlens.py"""
import importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAILS = []
def ck(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)

def load():
    spec = importlib.util.spec_from_file_location("pmx", REPO / "overlay/patch_mla_exactlens.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run(m, env, site, opt):
    old = dict(os.environ)
    os.environ.update({"GLM53_SITEPKG": str(site), "GLM53_OPT": str(opt)})
    if env is None:
        os.environ.pop("GLM53_MLA_EXACT_LENS", None)
    else:
        os.environ["GLM53_MLA_EXACT_LENS"] = env
    try:
        m.main([]); return "ok"
    except SystemExit as e:
        return f"exit:{e}"
    finally:
        os.environ.clear(); os.environ.update(old)

def snapshot(d):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}

def main():
    m = load()
    stock = subprocess.run(["git", "-C", str(REPO), "show", "r16z2rev:integrate.py"], check=True,
                           capture_output=True).stdout
    root = Path(tempfile.mkdtemp(prefix="pmx.", dir=os.environ.get("PMX_TMP")))
    try:
        site, opt = root / "site", root / "opt"
        (opt / "tf/overlay").mkdir(parents=True); site.mkdir()
        shutil.copy(REPO / "glm53_mla_exactlens.py", opt / "tf/overlay/glm53_mla_exactlens.py")
        (site / "integrate.py").write_bytes(stock)
        r = run(m, "1", site, opt)
        integ = (site / "integrate.py").read_text()
        ck(r == "ok" and (site / "glm53_mla_exactlens.py").read_bytes() == (REPO / "glm53_mla_exactlens.py").read_bytes(),
           "P1 =1 installs the module byte-identical")
        ck(m.ARM_ANCHOR + m.ARM_BLOCK in integ and integ.count(m.ARM_MARK_BEGIN) == 1, "P1 integrate.py armed after the hostloop step")
        compile(integ, "integrate.py", "exec")
        ck(True, "P1 armed integrate.py compiles")
        s1 = snapshot(site)
        r2 = run(m, "1", site, opt)
        ck(r2 == "ok" and snapshot(site) == s1, "P1 second =1 run changes nothing")
        r3 = run(m, "0", site, opt)
        ck(r3 == "ok" and not (site / "glm53_mla_exactlens.py").exists() and (site / "integrate.py").read_bytes() == stock,
           "P2 =0 removes the module and restores integrate.py byte for byte")
        s0 = snapshot(site)
        ck(run(m, "0", site, opt) == "ok" and snapshot(site) == s0, "P2 =0 again is a no-op")
        for val in ("2", "yes", "on"):
            ck(run(m, val, site, opt).startswith("exit:") and snapshot(site) == s0, f"P3 value {val!r} refused, nothing written")
        (opt / "tf/overlay/glm53_mla_exactlens.py").rename(opt / "tf/overlay/x.py")
        ck(run(m, "1", site, opt).startswith("exit:") and snapshot(site) == s0, "P3 missing bundle module refused")
        (opt / "tf/overlay/x.py").rename(opt / "tf/overlay/glm53_mla_exactlens.py")
        (site / "glm53_mla_exactlens.py").write_text("# foreign\n")
        ck(run(m, "1", site, opt).startswith("exit:") and (site / "glm53_mla_exactlens.py").read_text() == "# foreign\n",
           "P3 a foreign module file is refused (not overwritten)")
        (site / "glm53_mla_exactlens.py").unlink()
        (site / "integrate.py").write_text(stock.decode().replace("glm53_hostloop not loaded", "hostloop gone"))
        ck(run(m, "1", site, opt).startswith("exit:") and not (site / "glm53_mla_exactlens.py").exists(),
           "P3 integrate.py without the hostloop anchor refused before anything is installed")
        torn = stock.decode().replace(m.ARM_ANCHOR, m.ARM_ANCHOR + "    " + m.ARM_MARK_BEGIN + "\n", 1)
        (site / "integrate.py").write_text(torn)
        ck(run(m, "1", site, opt).startswith("exit:") and (site / "integrate.py").read_text() == torn
           and not (site / "glm53_mla_exactlens.py").exists(), "P3 a torn arming block refused, nothing written")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    ov = REPO / "overlay/glm53_mla_exactlens.py"
    ck(ov.is_file() and ov.read_bytes() == (REPO / "glm53_mla_exactlens.py").read_bytes(),
       "P4 overlay/glm53_mla_exactlens.py == repo-root glm53_mla_exactlens.py")
    print(f"RESULT: {'ALL OK' if not FAILS else str(len(FAILS)) + ' FAIL'}")
    return 1 if FAILS else 0

sys.exit(main())
