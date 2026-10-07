"""opt-moe-rev fixes on top of opt-moe (FOLD_SHARED observability + torch.compile guard), on the stand-in moe_runner
of tests/optmoe/test_fold.py with the real layer-10 experts:
  1. the first folded served call logs "glm53_moe_e4m3 fold: first served call folded" exactly once (boot-checkable:
     vLLM's profile run is a served call), and a served call that is NOT folded while FOLD is active logs the
     "was NOT folded" WARNING exactly once
  2. under torch.compile tracing (torch.compiler.is_compiling() True) the wrappers pass through: no arming, no fold,
     vLLM's add runs, and _unpack never drops the shared half even with a stale pending flag
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe_rev/test_rev.py"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402
from test_fold import STUB, inputs, rel  # noqa: E402

CHK = H.Checks()


class Rec(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, r):
        self.msgs.append((r.levelno, r.getMessage()))


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    rec = Rec()
    M._log.addHandler(rec)
    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    tmpd = tempfile.mkdtemp(dir="/w/tests/optmoe_rev")
    stub_py = os.path.join(tmpd, "stub_moe_runner.py")
    with open(stub_py, "w") as f:
        f.write(STUB)
    spec = importlib.util.spec_from_file_location("stub_moe_runner", stub_py)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    why = M._install_fold(R)
    os.unlink(stub_py)
    os.rmdir(tmpd)
    CHK(why == "ok", f"[stub] wrappers installed ({why})")
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": "bf16",
                                   "GLM53_MOE_E4M3_FOLD_SHARED": "1"}, load_selftest=False)
    CHK(rep["installed"] and M.FOLD["on"], "[hook] FOLD installed")
    T = 4289
    x, s, ids, w = inputs(T, "real", 41, dev)
    rf = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32"}).clone()

    def apply_fn(xx):
        return prod.apply_exl3_experts(xx, ids, w, L, limit=C.LIMIT)

    def shared_fn(xx):
        return s.clone()

    def count(sub, lvl=None):
        return sum(1 for lv, m in rec.msgs if sub in m and (lvl is None or lv == lvl))

    # 1. logging
    for _ in range(3):
        o = R.MoERunner(apply_fn, shared_fn).forward(x)
    CHK(count("fold: first served call folded", logging.INFO) == 1, "[log] first fold logged once (3 folded calls)")
    CHK(count("was NOT folded") == 0, "[log] no not-folded warning while every call folds")
    for _ in range(2):
        rr = R.MoERunner(apply_fn, shared_fn, scale=2.5)
        o = rr.forward(x)
    CHK(rr.adds == 1 and count("was NOT folded", logging.WARNING) == 1 and rel(o, s.double() + 2.5 * rf.double()) < 6e-3,
        "[log] ineligible runner while FOLD is on: not folded, vLLM add, WARNING once")
    # 2. compile guard
    real = torch.compiler.is_compiling
    f0 = M.STATS["folded"]
    try:
        torch.compiler.is_compiling = lambda: True
        rc = R.MoERunner(apply_fn, shared_fn)
        o = rc.forward(x)
        CHK(rc.adds == 1 and M.STATS["folded"] == f0 and rel(o, s.double() + rf.double()) < 6e-3,
            f"[compile] is_compiling: no fold, vLLM's add ({rel(o, s.double() + rf.double()):.2e})")
        M._FOLD_CTX["pending"] = True
        sh, fu = R._unpack((s, rf))
        CHK(sh is s, "[compile] is_compiling: _unpack keeps the shared half even with a stale pending flag")
        M._FOLD_CTX["pending"] = False
    finally:
        torch.compiler.is_compiling = real
    rn = R.MoERunner(apply_fn, shared_fn)
    o = rn.forward(x)
    CHK(rn.adds == 0 and M.STATS["folded"] == f0 + 1, "[compile] back to eager: folds again")
    M.uninstall(prod)
    CHK(not M.FOLD["wrapped"], "[stub] uninstall restores")
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
