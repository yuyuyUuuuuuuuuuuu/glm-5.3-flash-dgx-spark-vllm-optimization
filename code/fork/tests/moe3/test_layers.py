"""GLM53_MOE_E4M3_LAYERS (layer-selective e4m3) composed with GLM53_MOE_FUSED16, real layer-10 experts.
  L1 parsing: lists / ranges, invalid values refuse install, unset = every layer (the previous behaviour)
  L2 a selected layer (layer_name ...layers.10...) is served by e4m3 (== run()); an unselected one passes to
     production's apply (with FUSED16 installed: the P16 grouped path) and gets no self-test; summary line counts
  L3 a layer without layer_name is not selected when a list is set; unset list: served as before
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe3/test_layers.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_moe_fused16 as F
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    # L1
    CHK(M.layers_mode({}) is None and M.layers_mode({"GLM53_MOE_E4M3_LAYERS": ""}) is None, "unset / empty = all")
    CHK(M.layers_mode({"GLM53_MOE_E4M3_LAYERS": "3-5, 10,44"}) == frozenset({3, 4, 5, 10, 44}), "list + ranges")
    for bad in ("x", "5-3", ",", "-1"):
        r = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_LAYERS": bad}, load_selftest=False)
        CHK(not r["installed"], f"invalid {bad!r} refused")
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    base = prod.apply_exl3_experts
    assert F.install(prod, environ={"GLM53_MOE_FUSED16": "1"}, load_selftest=False)["installed"]
    T = 13824
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", T, 5, dev)
    w = C.weights_for(T, 5, dev).float()
    y_f16 = base(x, ids, w, L, limit=C.LIMIT)                       # FUSED16 (production arithmetic)
    y_e4 = M.run(prod, x, ids, w, L, C.LIMIT).to(torch.bfloat16)    # the e4m3 arithmetic

    def close(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())
    for sel, name, want_e4 in (("10", "model.layers.10.mlp.experts", True), ("3-9,11-44", "model.layers.10.mlp.experts", False),
                               ("10", None, False), ("", "model.layers.10.mlp.experts", True)):
        M.uninstall(prod)
        for attr in ("_glm53_moe_e4m3_ok",):
            if hasattr(L, attr):
                delattr(L, attr)
        if name is None:
            if hasattr(L, "layer_name"):
                delattr(L, "layer_name")
        else:
            L.layer_name = name
        for k in list(M.STATS):
            if isinstance(M.STATS[k], int):
                M.STATS[k] = 0
        M._SUMMARY["logged"] = None
        env = {"GLM53_MOE_E4M3": "1"}
        if sel:
            env["GLM53_MOE_E4M3_LAYERS"] = sel
        r = M.install(prod, environ=env, load_selftest=False)
        CHK(r["installed"], f"install {sel!r}")
        s0 = F.STATS["served"]
        y = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
        d_e4, d_f16 = close(y, y_e4), close(y, y_f16)
        line = M.log_summary(256)
        print(f"L2 LAYERS={sel!r} layer_name={name}: rel to e4m3 {d_e4:.2e}, to FUSED16 {d_f16:.2e}; e4m3 served "
              f"{M.STATS['served']}, unselected {M.STATS['layers_unselected']}, P16 served {F.STATS['served'] - s0}; "
              f"summary: {line}", flush=True)
        if want_e4:
            CHK(d_e4 < 1e-2 and M.STATS["served"] == 1 and F.STATS["served"] == s0, f"{sel!r}: e4m3 must serve")
        else:
            CHK(d_f16 < 1e-4 and M.STATS["served"] == 0 and M.STATS["layers_unselected"] == 1
                and F.STATS["served"] == s0 + 1 and "unselected" in line, f"{sel!r}: production (P16) path expected")
        CHK(("GLM53_MOE_E4M3_LAYERS" in line) == bool(sel), "summary suffix only with a list")
    M.uninstall(prod)
    F.uninstall(prod)
    PC.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
