"""opt-moe2: run an existing test with the lean mainloop (GLM53_MOE_E4M3_MAINLOOP=1) forced on in-process: every
fused call that test makes (variants 0 / 16, + 2048 for TG) then runs variant + 8192. The flag is set right after the
test imports the extension helpers (harness.load_xl is the first thing every test calls), and re-set after any
install() / uninstall() the test does (both reset it from the environment / to off), so the whole test sees it.
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe2/run_with_ms.py tests/optmoe/test_tg.py
"""
from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import harness as H  # noqa: E402

_orig_load_xl = H.load_xl
_state = {"patched": False}


def _force(M):
    M.MS["on"] = True
    if _state["patched"]:
        return
    _state["patched"] = True
    oi, ou, orun = M.install, M.uninstall, M.run

    def install(*a, **k):
        r = oi(*a, **k)
        M.MS["on"] = True
        return r

    def uninstall(*a, **k):
        r = ou(*a, **k)
        M.MS["on"] = True
        return r

    def run(*a, **k):
        sched = k.get("sched")
        if sched is not None and "ms" not in sched:
            k["sched"] = dict(sched, ms=True)
        return orun(*a, **k)

    M.install, M.uninstall, M.run = install, uninstall, run
    print("run_with_ms: GLM53_MOE_E4M3_MAINLOOP forced on (fused variants + 8192)", flush=True)


def load_xl():
    xl = _orig_load_xl()
    try:
        import glm53_moe_e4m3 as M
    except ImportError:
        for p in (os.path.join(HERE, "..", "..", "overlay"),):
            sys.path.insert(0, p)
        import glm53_moe_e4m3 as M
    _force(M)
    return xl


H.load_xl = load_xl
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
