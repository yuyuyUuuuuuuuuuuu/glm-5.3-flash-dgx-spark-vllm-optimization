"""opt-moe2-rev: the load-time self-test with GLM53_MOE_E4M3_MAINLOOP=1 (+ ACC=bf16, TG default on).
  1. which fused variants the self-test actually launches (ext.fused wrapped): must be the served + 8192 ones
  2. FB_EXPECT=fallback (with OPTMOE_EXT = a build with -DME_AT_SM_EXTRA=8192, i.e. a MAINLOOP launch that asks for
     more dynamic shared memory than a GB10 block gets): the self-test must RAISE, the layer must fall back to
     production's path, and the production path (apply_exl3_experts on a prefill-sized call, then a plain torch op)
     must keep working - no stale CUDA error surfacing in an unrelated kernel launch check
Run: GPU_RUN_ENV="OPTMOE_EXT=/w/...so;FB_EXPECT=fallback" GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 \
     tests/gpu_run.sh python3 tests/optmoe2rev/fallback_ms.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    expect = os.environ.get("FB_EXPECT", "serve")
    ext = M._ext()
    seen = []
    orig = ext.fused

    def fused(*a):
        seen.append(int(a[-1]))
        return orig(*a)
    ext.fused = fused
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_MAINLOOP": "1", "GLM53_MOE_E4M3_ACC": "bf16"})
    CHK(rep["installed"] and rep.get("load_hook") and rep.get("mainloop") == 1, f"installed with MAINLOOP ({rep})")
    s0 = dict(M.STATS)
    L = make_real_layer(prod, dev)
    ok = getattr(L, "_glm53_moe_e4m3_ok", None)
    print(f"  self-test: layer ok={ok}, fused variants launched {seen}, selftest_raised "
          f"{M.STATS['selftest_raised'] - s0['selftest_raised']}, failed {M.STATS['selftest_failed'] - s0['selftest_failed']}",
          flush=True)
    CHK(len(seen) >= 1 and all(v >= 8192 for v in seen), f"self-test launched only MAINLOOP variants ({seen})")
    T = 1024
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", T, 5, dev)
    w = C.weights_for(T, 5, dev).float()
    err = None
    try:
        s1 = dict(M.STATS)
        y = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
        z = float((torch.ones(4, device=dev) * 2).sum().item())
        torch.cuda.synchronize()
        served = M.STATS["served"] - s1["served"]
        print(f"  after the self-test: prefill call ok (served by e4m3: {served}), |y| {float(y.float().abs().mean()):.4e}, "
              f"torch op {z}", flush=True)
    except Exception as exc:  # noqa: BLE001
        err = exc
        print(f"  after the self-test: the next call RAISED {exc!r}", flush=True)
    if expect == "fallback":
        CHK(ok is False and M.STATS["selftest_raised"] > s0["selftest_raised"], "self-test raised -> layer falls back")
        CHK(err is None, f"production fallback path keeps working after the refused launch ({err!r})")
    else:
        CHK(ok is True, "self-test passed, layer served")
        CHK(err is None, f"served prefill call works ({err!r})")
    M.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
