#!/usr/bin/env python3
"""FKDA2/FKDA3: the wrapper's extension pin (glm53_flashkda.EXT_SHA256), one subprocess per case (configure()
caches and the op library loads once per process). Production image, tests/fkda/gpu_run.sh with FKDA_USER=root
(installs the overlay the way patch_flashkda.py does, like tests/fkda/wrapper_unit.py).
  --version fkda2 (default) | fkda3 : which pinned build is installed (wu.install stages its overlay pair)
  pinned  : the overlay .so (installed) -> configures; boot line ends with the pinned build name (fkda2 or fkda3 precision build)
  other   : the OTHER build's EXTENSION first on sys.path (staged as _flashkda_fp32_C.abi3.so, like the fkda2 era's
            --other build dir: configure imports the extension lazily, so sys.path[0] decides which .so loads)
            -> RuntimeError at configure (the sha does not match this wrapper's pin)
  override: the same other extension with STATE['allow_any_ext'] -> configures, boot line says NOT the validated build
Exit 0 only if all three behave."""
import shutil
import subprocess
import sys
import tempfile

CASE = r'''
import sys, torch
sys.path.insert(0, "/w/tests/fkda")
import wrapper_unit as wu
from pathlib import Path
wu.install(Path("/usr/local/lib/python3.12/dist-packages"), "{ver}")
if "{other}":
    sys.path.insert(0, "{other}")
import glm53_flashkda as F
F.STATE["allow_any_ext"] = {allow}
from vllm.v1.worker.workspace import init_workspace_manager
init_workspace_manager(torch.device("cuda"))
try:
    F.configure(wu.FakeLayer(), wu.FakeCfg())
    print("CONFIGURED", flush=True)
except RuntimeError as e:
    print("REFUSED", e, flush=True)
'''


def run(other, allow, ver):
    r = subprocess.run([sys.executable, "-c", CASE.format(other=other, allow=allow, ver=ver)],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    print("\n".join(l for l in out.splitlines() if "glm53-kda-flashkda" in l or l.startswith(("CONFIGURED", "REFUSED"))))
    return out


def main():
    ver = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "fkda2"
    other_ver = "fkda3" if ver == "fkda2" else "fkda2"
    # the "other" build: the other build's EXTENSION staged under the site name (configure imports it lazily, so
    # sys.path[0] decides which .so loads; this wrapper's pin then refuses it)
    other = ""
    if "--other" in sys.argv:
        other = sys.argv[sys.argv.index("--other") + 1]
    else:
        other = tempfile.mkdtemp(prefix="fkda-other-")
        shutil.copy2("/w/overlay/" + wu_ext(other_ver), other + "/_flashkda_fp32_C.abi3.so")
    ok = True
    o = run("", False, ver)
    ok &= "CONFIGURED" in o and "precision build (" in o and "buffers RESERVED at boot" in o
    print("pinned:", "ok" if ok else "FAIL")
    o = run(other, False, ver)
    r2 = "REFUSED" in o and "precision build (" in o and "is not the" in o
    print("other build refused:", "ok" if r2 else "FAIL")
    o = run(other, True, ver)
    r3 = "CONFIGURED" in o and "NOT the validated build" in o
    print("override:", "ok" if r3 else "FAIL")
    ok = ok and r2 and r3
    print(f"WRAPPER PIN ({ver}):", "ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def wu_ext(ver):
    """the overlay extension file of a build (tests/fkda/wrapper_unit.VERSIONS)."""
    sys.path.insert(0, "/w/tests/fkda")
    import wrapper_unit
    return wrapper_unit.VERSIONS[ver][1]


if __name__ == "__main__":
    main()
