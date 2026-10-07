"""Adversarial review checks for GLM53_MOE_E4M3 (review of commits 34af5ce..f36b090), nodeC, real layer-10 experts.

  R1 persistent kernel with any grid (1, 2, 7, default) and lag (1, 12, 10000): same output (atomics order only)
  R2 the fused kernel while another stream keeps the SMs busy (not every CTA co-resident at launch): completes, same
  R3 back-to-back calls with different segment counts (flag / ticket reset between calls): reproducible
  R4 MNBT 16384 x top-8 (several requests' tokens in one batch): fused == sequential kernels; finite
  R5 duplicated experts inside one token's top-8, vs the spec in torch
  R6 expert_map with every expert non-local: zero output, no hang
  R7 small activations (a16 near the fp16 subnormal range) vs the spec
  R8 NaN / inf in one token: does it stay in that token, and does it propagate (vs production)?
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moee4m3/review_adv.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
CHK = H.Checks()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def spec_ref(M, L, x, ids, w):
    T = x.shape[0]
    ref = torch.zeros(T, 4096, dtype=torch.float32, device=x.device)
    xh = x.half()
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= len(L._exl3_inners):
            continue
        tok, kk = (ids == e).nonzero(as_tuple=True)
        d = M.spec_ffn(xh.index_select(0, tok), L._exl3_inners[e], C.LIMIT)
        ref.index_add_(0, tok, d * w[tok, kk].unsqueeze(-1).float())
    return ref


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    n_exp = len(L._exl3_inners)

    def inputs(T, kind="real", seed=None):
        seed = T if seed is None else seed
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        ids = C.routing(kind, T, seed, dev)
        w = C.weights_for(T, seed, dev).float()
        return x, ids, w

    # R1 grid / lag sweep
    x, ids, w = inputs(1024)
    base = M.run(prod, x, ids, w, L, C.LIMIT).clone()
    for sd in ({"grid": 1}, {"grid": 2}, {"grid": 7}, {"lag": 1}, {"lag": 10000}, {"grid": 3, "lag": 1}):
        t0 = time.time()
        o = M.run(prod, x, ids, w, L, C.LIMIT, sched=sd)
        torch.cuda.synchronize()
        r = rel(o, base)
        print(f"  R1 sched {sd}: rel-L2 vs default {r:.2e} ({time.time() - t0:.2f}s)", flush=True)
        CHK(r < 1e-5, f"R1 {sd} {r}")

    # R2 SMs busy on another stream while the persistent kernel launches
    side = torch.cuda.Stream(device=dev)
    a = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        for _ in range(40):
            a = a @ a
            a = a / a.norm()
    o = M.run(prod, x, ids, w, L, C.LIMIT)
    torch.cuda.synchronize()
    r = rel(o, base)
    print(f"  R2 concurrent side-stream GEMMs: rel-L2 {r:.2e}", flush=True)
    CHK(r < 1e-5, f"R2 {r}")
    hp = torch.cuda.Stream(device=dev, priority=-5)
    with torch.cuda.stream(hp):
        for _ in range(40):
            a = a @ a
            a = a / a.norm()
    o = M.run(prod, x, ids, w, L, C.LIMIT)
    torch.cuda.synchronize()
    r = rel(o, base)
    print(f"  R2 concurrent high-priority side-stream GEMMs: rel-L2 {r:.2e}", flush=True)
    CHK(r < 1e-5, f"R2hp {r}")

    # R3 alternating shapes (ticket / gu_cnt / ready reset)
    xb, idsb, wb = inputs(13824)
    big = M.run(prod, xb, idsb, wb, L, C.LIMIT).clone()
    xs, idss, ws = inputs(300, "collapsed", 77)
    small = M.run(prod, xs, idss, ws, L, C.LIMIT).clone()
    worst = 0.0
    for i in range(6):
        ob = M.run(prod, xb, idsb, wb, L, C.LIMIT)
        os_ = M.run(prod, xs, idss, ws, L, C.LIMIT)
        worst = max(worst, rel(ob, big), rel(os_, small))
    torch.cuda.synchronize()
    print(f"  R3 6x alternating T=13824 / T=300: worst rel-L2 vs first {worst:.2e}", flush=True)
    CHK(worst < 1e-5, f"R3 {worst}")
    del xb, idsb, wb, big, ob

    # R4 MNBT x top-8
    x4, ids4, w4 = inputs(16384)
    of = M.run(prod, x4, ids4, w4, L, C.LIMIT).clone()
    osq = M.run(prod, x4, ids4, w4, L, C.LIMIT, sched={"mode": "streams", "nchunks": 1})
    torch.cuda.synchronize()
    r = rel(of, osq)
    print(f"  R4 T=16384 fused vs sequential kernels: rel-L2 {r:.2e}, finite {bool(torch.isfinite(of).all())}", flush=True)
    CHK(r < 1e-5 and bool(torch.isfinite(of).all()), f"R4 {r}")
    del x4, ids4, w4, of, osq
    torch.cuda.empty_cache()

    # R5 duplicated experts in a token's top-8
    T = 300
    x5, ids5, w5 = inputs(T, "real", 5)
    ids5 = ids5.clone()
    ids5[:, 1] = ids5[:, 0]                  # expert k0 twice
    ids5[::3, 2:5] = ids5[::3, 0:1]          # every 3rd token: the same expert 5 times
    o5 = M.run(prod, x5, ids5, w5, L, C.LIMIT)
    r = rel(o5, spec_ref(M, L, x5, ids5, w5))
    print(f"  R5 duplicated experts vs spec: rel-L2 {r:.2e}", flush=True)
    CHK(r < 5e-3, f"R5 {r}")

    # R6 every expert non-local
    emap = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    o6 = M.run(prod, x5, ids5, w5, L, C.LIMIT, expert_map=emap)
    torch.cuda.synchronize()
    print(f"  R6 all non-local: max |out| {float(o6.abs().max()):.3e}", flush=True)
    CHK(float(o6.abs().max()) == 0.0, "R6 nonzero")

    # R7 tiny activations
    for sc7 in (1e-2, 1e-3):
        x7 = (x5.float() * sc7).to(torch.bfloat16)
        o7 = M.run(prod, x7, ids5, w5, L, C.LIMIT)
        ref7 = spec_ref(M, L, x7, ids5, w5)
        r = rel(o7, ref7)
        print(f"  R7 activations x{sc7:g} vs spec: rel-L2 {r:.2e}, |ref| {float(ref7.norm()):.3e}, finite "
              f"{bool(torch.isfinite(o7).all())}", flush=True)
        CHK(r < 5e-3 and float(ref7.norm()) > 0, f"R7 {sc7} {r}")

    # R8 NaN / inf in one token
    x8, ids8, w8 = inputs(512, "real", 8)
    clean = M.run(prod, x8, ids8, w8, L, C.LIMIT).clone()
    emapp = prod.pin_exl3_expert_map(L, dev)
    for name, val in (("nan", float("nan")), ("inf(bf16 1e5 -> fp16)", 1e5)):
        xx = x8.clone()
        xx[7, 100] = val
        oe = M.run(prod, xx, ids8, w8, L, C.LIMIT)
        op = prod.apply_exl3_fused_moe(xx, ids8, w8, L, L._exl3_inners, emapp, C.LIMIT)
        others = torch.ones(512, dtype=torch.bool, device=dev)
        others[7] = False
        r_oth = rel(oe[others], clean[others])
        print(f"  R8 {name} in token 7: e4m3 row7 finite={bool(torch.isfinite(oe[7]).all())} "
              f"nan={int(torch.isnan(oe[7]).sum())} max|row7|={float(oe[7].nan_to_num(0).abs().max()):.3e}; "
              f"production row7 finite={bool(torch.isfinite(op[7]).all())} nan={int(torch.isnan(op[7]).sum())}; "
              f"other tokens rel vs clean {r_oth:.2e}", flush=True)
        CHK(r_oth < 1e-5, f"R8 {name} leaks into other tokens")
    CHK.summary()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
