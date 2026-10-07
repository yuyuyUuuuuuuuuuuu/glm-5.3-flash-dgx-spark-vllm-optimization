"""GLM53_MOE_E4M3 wiring (glm53_moe_e4m3.py + integrate.plugin_register) on nodeC, real layer-10 experts.

  W1 OFF == previous: with the knob unset / "" / 0 / off, integrate.plugin_register() leaves production's
     apply_exl3_experts the very same function object (nothing wrapped: the byte-identical guarantee), and the
     outputs of a decode-sized and a prefill call equal the pre-registration ones within production's own
     run-to-run spread (its decode and E3 kernels use fp32 atomics; measured and printed).
  W2 ON: plugin_register() installs the wrapper last (outermost); decode-sized calls pass through bit-identically;
     a CUDA graph captured at T=64 through the hooked function replays equal to eager production; prefill calls are
     served (self-test once per layer); the served output is the e4m3 class vs production (rel ~6.5e-2) and equals
     glm53_moe_e4m3.run().
  W3 production stack: TF fork (integrate.install, K2) + prefill cap 1 + GLM53_MOE_E4M3: decode unchanged vs the
     stack without it, prefill served.
Run (both production module versions):
  GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moee4m3/test_wiring.py
  GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" \\
  GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moee4m3/test_wiring.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def inputs(T, seed, dev, kind="real"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    return x, C.routing(kind, T, seed, dev), C.weights_for(T, seed, dev)


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    xl = H.load_xl()
    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    import integrate
    import glm53_moe_e4m3 as M

    for k in ("GLM53_MOE_E4M3", "TF_EXL3_MOE", "GLM53_PREFILL_FUSED_CAP", "GLM53_DEC_MOEGLUE"):
        os.environ.pop(k, None)
    base = prod.apply_exl3_experts
    xd, idd, wd = inputs(64, 1, dev)
    xp, idp, wp = inputs(2048, 2, dev)
    ref_d = base(xd, idd, wd, L, limit=C.LIMIT)
    ref_p = base(xp, idp, wp, L, limit=C.LIMIT)
    noise = max(rel(base(xp, idp, wp, L, limit=C.LIMIT).float(), ref_p.float()) for _ in range(3))
    tol_p = max(4 * noise, 1e-5)          # the e4m3 route differs from production by ~6e-2
    # production's decode kernel itself is not bitwise reproducible (fp32 atomics): its own run-to-run spread
    noise_d = max(rel(base(xd, idd, wd, L, limit=C.LIMIT).float(), ref_d.float()) for _ in range(4))
    tol_d = max(4 * noise_d, 1e-4)        # the routes differ by ~6e-2 (e4m3 class); fp32-atomic spreads are 1e-8..1e-5
    print(f"  production's own run-to-run spread: decode T=64 {noise_d:.1e}, prefill T=2048 {noise:.1e}", flush=True)
    # ---- W1
    for v in (None, "", "0"):
        if v is None:
            os.environ.pop("GLM53_MOE_E4M3", None)
        else:
            os.environ["GLM53_MOE_E4M3"] = v
        integrate.plugin_register()
        CHK(prod.apply_exl3_experts is base, f"W1 knob {v!r}: apply_exl3_experts is production's object")
        CHK(not M.STATS["installed"], f"W1 knob {v!r}: module reports not installed")
        d = prod.apply_exl3_experts(xd, idd, wd, L, limit=C.LIMIT)
        p = prod.apply_exl3_experts(xp, idp, wp, L, limit=C.LIMIT)
        rd = rel(d.float(), ref_d.float())
        CHK(rd <= tol_d, f"W1 knob {v!r}: decode-sized output == previous ({rd:.1e} <= production's spread bound {tol_d:.1e})")
        rp = rel(p.float(), ref_p.float())
        CHK(rp <= tol_p, f"W1 knob {v!r}: prefill output == previous ({rp:.1e} <= {tol_p:.1e}; E3 spread {noise:.1e})")
    print(f"  W1 OFF == previous: function identity + outputs (prefill diff within E3's own noise {noise:.1e})", flush=True)
    # ---- W2: the kit's ON state = site-packages integrate.py armed by overlay/patch_moe_e4m3.py (here: a temp copy)
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("pm", os.path.join(H.REPO, "overlay", "patch_moe_e4m3.py"))
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    tdir = tempfile.mkdtemp(prefix="moee4m3-armed-")
    armed_py = os.path.join(tdir, "integrate.py")
    with open(os.path.join(H.REPO, "integrate.py")) as f_in, open(armed_py, "w") as f_out:
        f_out.write(f_in.read())
    from pathlib import Path
    CHK(pm.arm(Path(armed_py)) == "armed", "W2 patch_moe_e4m3.arm on a copy of integrate.py")
    spec = importlib.util.spec_from_file_location("integrate_armed", armed_py)
    integ_armed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integ_armed)
    os.environ["GLM53_MOE_E4M3"] = "1"
    integ_armed.plugin_register()
    hooked = prod.apply_exl3_experts
    CHK(getattr(hooked, "_glm53_moe_e4m3", False) and hooked._glm53_moe_e4m3_orig is base,
        "W2 knob 1: wrapper installed directly over production's apply (outermost)")
    integ_armed.plugin_register()     # a second registration must not stack a second wrapper
    CHK(prod.apply_exl3_experts is hooked, "W2 second plugin_register: no double wrap")
    s0 = dict(M.STATS)
    d = prod.apply_exl3_experts(xd, idd, wd, L, limit=C.LIMIT)
    CHK(rel(d.float(), ref_d.float()) <= tol_d and M.STATS["served"] == s0["served"] and M.STATS["passed"] == s0["passed"] + 1,
        "W2 decode-sized call: passed through to production (output within production's own spread)")
    # graph capture of a decode-sized call through the hook
    static_x, static_i, static_w = xd.clone(), idd.clone(), wd.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            prod.apply_exl3_experts(static_x, static_i, static_w, L, limit=C.LIMIT)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = prod.apply_exl3_experts(static_x, static_i, static_w, L, limit=C.LIMIT)
    x2, i2, w2 = inputs(64, 7, dev)
    static_x.copy_(x2); static_i.copy_(i2); static_w.copy_(w2)
    graph.replay()
    torch.cuda.synchronize()
    eager = base(x2, i2, w2, L, limit=C.LIMIT)
    rg = rel(gout.float(), eager.float())
    CHK(rg < 1e-6, f"W2 CUDA graph (T=64) through the hook: replay == eager production ({rg:.1e})")
    # prefill served
    L.__dict__.pop("_glm53_moe_e4m3_ok", None)
    s0 = dict(M.STATS)
    p = prod.apply_exl3_experts(xp, idp, wp, L, limit=C.LIMIT)
    CHK(M.STATS["served"] == s0["served"] + 1 and M.STATS["selftests"] == s0["selftests"] + 1,
        "W2 prefill call served after one self-test")
    rr = rel(p.float(), M.run(prod, xp, idp, wp, L, C.LIMIT).to(xp.dtype).float())
    CHK(rr < 1e-5, f"W2 served == run() ({rr:.1e})")
    rcls = rel(p.float(), ref_p.float())
    CHK(0.02 < rcls < 0.12, f"W2 served vs production: the e4m3 class difference ({rcls:.4f})")
    s0 = dict(M.STATS)
    prod.apply_exl3_experts(xp, idp, wp, L, limit=C.LIMIT)
    CHK(M.STATS["selftests"] == s0["selftests"], "W2 self-test once per layer")
    print(f"  W2 ON: decode pass-through, graph replay {rg:.1e}, prefill served (vs production {rcls:.4f})",
          flush=True)
    M.uninstall(prod)
    CHK(prod.apply_exl3_experts is base, "W2 uninstall")
    os.environ.pop("GLM53_MOE_E4M3", None)
    # ---- W4: load-time self-test (process_weights_after_loading hook), the summary line, the decode bound
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1"})
    CHK(rep["installed"] and rep.get("load_hook"), f"W4 installed with the load-time hook ({rep})")
    s0 = dict(M.STATS)
    L4 = make_real_layer(prod, dev)
    CHK(getattr(L4, "_glm53_moe_e4m3_ok", None) is True and M.STATS["layers"] == s0["layers"] + 1 and
        M.STATS["selftests"] == s0["selftests"] + 1, "W4 the layer was self-tested during process_weights_after_loading")
    M._SUMMARY["logged"] = None
    line = M.log_summary(256)
    CHK("layers served" in line and f"{M.STATS['layers_ok']}/{M.STATS['layers']} layers served" in line,
        f"W4 summary: {line}")
    s0 = dict(M.STATS)
    prod.apply_exl3_experts(xp, idp, wp, L4, limit=C.LIMIT)
    CHK(M.STATS["selftests"] == s0["selftests"] and M.STATS["served"] == s0["served"] + 1,
        "W4 first prefill call served without a second self-test")
    b, k, cap = M.decode_bound({"MAX_NUM_SEQS": "4", "SPEC_METHOD": "dflash", "DFLASH_TOKENS": "7",
                                "EXL3_TEMP_ROWS_FUSED": "256"})
    CHK((b, k, cap) == (32, 7, 256), f"W4 production decode bound 4 x 8 = 32 <= 256 ({b}, {k}, {cap})")
    b, k, cap = M.decode_bound({"MAX_NUM_SEQS": "64", "SPEC_METHOD": "dflash", "DFLASH_TOKENS": "7",
                                "EXL3_TEMP_ROWS_FUSED": "256"})
    w0 = M.STATS["decode_bound_warn"]
    M.uninstall(prod)
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "MAX_NUM_SEQS": "64", "SPEC_METHOD": "dflash",
                                   "DFLASH_TOKENS": "7", "EXL3_TEMP_ROWS_FUSED": "256"})
    CHK(b == 512 and M.STATS["decode_bound_warn"] == w0 + 1, "W4 MAX_NUM_SEQS 64 x 8 = 512 > 256: DECODE BOUND warning")
    M.uninstall(prod)
    pw = prod.Exl3MoEMethod.process_weights_after_loading
    CHK(not getattr(pw, "_glm53_moe_e4m3", False), "W4 uninstall restores process_weights_after_loading")
    for v in ("on", "true", "2"):
        rep = M.install(prod, environ={"GLM53_MOE_E4M3": v})
        CHK(not rep["installed"] and prod.apply_exl3_experts is base, f"W4 value {v!r} refused (only 1 enables)")
    # ---- W3 production stack
    import glm53_prefill_cap as PC

    tf = H.load_tf()
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    CHK(rep["installed"], f"W3 TF installed ({rep.get('reason')})")
    L2 = make_real_layer(prod, dev)            # built after TF so its build hook registers the layer
    assert PC.install(prodmod=prod, n=1)["installed"]
    stack = prod.apply_exl3_experts
    d0 = stack(xd, idd, wd, L2, limit=C.LIMIT)
    p0 = stack(xp, idp, wp, L2, limit=C.LIMIT)
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1"})
    CHK(rep["installed"], "W3 GLM53_MOE_E4M3 over the TF + prefill-cap stack")
    s0 = dict(M.STATS)
    d1 = prod.apply_exl3_experts(xd, idd, wd, L2, limit=C.LIMIT)
    CHK(rel(d1.float(), d0.float()) <= tol_d and M.STATS["served"] == s0["served"] and M.STATS["passed"] == s0["passed"] + 1,
        "W3 decode-sized call: passed through to the stack")
    p1 = prod.apply_exl3_experts(xp, idp, wp, L2, limit=C.LIMIT)
    CHK(M.STATS["served"] == s0["served"] + 1, "W3 prefill served")
    r3 = rel(p1.float(), p0.float())
    CHK(0.02 < r3 < 0.12, f"W3 prefill vs the stack: e4m3 class ({r3:.4f})")
    print(f"  W3 stack (TF K2 + prefill cap 1 + e4m3): decode passed through, prefill served (vs stack {r3:.4f})", flush=True)
    M.uninstall(prod)
    PC.uninstall(prod)
    _ = tf
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
