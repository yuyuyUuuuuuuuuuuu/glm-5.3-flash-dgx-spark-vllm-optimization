"""opt-moe GLM53_MOE_E4M3_ACC=bf16 (the bf16 accumulator): real layer-10 experts (TP=2 rank-0 shard), production shapes.

  1. numerics at production shapes (T 13,824 / 13,856 / 4,289 / 2,048 / 300, real + collapsed routing, x8 activations):
     the bf16-accumulator output (variant 0 = 8 values per atomic op, variant 1 = 4 values per op, variant 16 = the
     f16 down + bf16 accumulator) vs the fp32-accumulator result of the same variant (fp32): rel-L2 < 6e-3 and max
     error < 3e-2 of the output's max; and vs the fp32 spec in fp64: the bf16 accumulator adds < 1 % of the e4m3
     path's own distance to production (variance ratio), i.e. it is not a quality change of the e4m3 class
  2. the output really is the zeroed accumulator: a NaN-filled bf16 out is fully overwritten by gather2 (finite)
  3. expert_map with half of the experts non-local: rows whose every expert is non-local are exactly 0; T=1, T=17;
     one expert holding every row; an all-zero x row gives an exactly-zero output row
  4. the hook: GLM53_MOE_E4M3_ACC=bf16 installs, the self-test runs both accumulators (acc_bf16_rel_l2 reported and
     < ACC_SELFTEST_TOL), the served call returns bf16 == run(acc bf16); unset / "" / "f32" = the fp32 accumulator
     (the served output == run() fp32 cast, i.e. unchanged); invalid values refused; uninstall resets
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/test_acc.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
CHK = H.Checks()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def inputs(T, kind, seed, dev, xscale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, 4096, generator=g) * xscale).to(torch.bfloat16).to(dev)
    ids = C.routing(kind, T, seed, dev)
    w = C.weights_for(T, seed, dev).float()
    return x, ids, w


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    # 1. numerics
    for T, kind, seed, xs in ((13824, "real", 13, 1.0), (13856, "collapsed", 14, 1.0), (4289, "real", 15, 1.0),
                              (2048, "collapsed", 16, 1.0), (300, "real", 17, 1.0), (1024, "real", 18, 8.0)):
        x, ids, w = inputs(T, kind, seed, dev, xs)
        tag = f"T={T} {kind} x{xs:g}"
        f0 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "f32"}).clone()
        f16 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "f32", "variant": 16}).clone()
        b8 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "bf16", "variant": 0}).clone()
        b4 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "bf16", "variant": 1}).clone()
        b16 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "bf16", "variant": 16}).clone()
        po = prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, C.LIMIT).float()
        cast = rel(f0.to(torch.bfloat16), f0)
        line = [f"fp32 cast {cast:.2e}"]
        for name, o, ref in (("b8", b8, f0), ("b4", b4, f0), ("b16", b16, f16)):
            CHK(o.dtype == torch.bfloat16 and bool(torch.isfinite(o).all()), f"[{tag}] {name}: bf16, finite")
            r = rel(o, ref)
            mx = float((o.double() - ref.double()).abs().max() / ref.double().abs().max())
            line.append(f"{name} vs fp32 acc {r:.2e} (max {mx:.1e})")
            CHK(r < 6e-3, f"[{tag}] {name} vs the fp32 accumulator {r:.2e} < 6e-3")
            CHK(mx < 3e-2, f"[{tag}] {name} max error {mx:.1e} < 3e-2 of max")
        # quality class: distance to production (E3, fp16 operands) barely moves
        pf, pb = rel(f0.to(torch.bfloat16), po), rel(b8, po)
        pf16, pb16 = rel(f16.to(torch.bfloat16), po), rel(b16, po)
        line.append(f"vs production: f32 {pf:.4f} bf16 {pb:.4f} (var x{(pb / pf) ** 2:.4f}); d16 f32 {pf16:.4f} "
                    f"bf16 {pb16:.4f} (var x{(pb16 / pf16) ** 2:.4f})")
        CHK((pb / pf) ** 2 < 1.01, f"[{tag}] bf16 acc adds < 1 % to the e4m3 distance-to-production variance")
        CHK((pb16 / pf16) ** 2 < 1.015, f"[{tag}] d16: bf16 acc adds < 1.5 % to the distance-to-production variance")
        print(f"  [{tag}] " + " | ".join(line), flush=True)
        del x, ids, w, f0, f16, b8, b4, b16, po
        torch.cuda.empty_cache()
    # 2. the accumulator is zeroed by gather2 (a NaN-filled output is fully overwritten)
    ext = M._ext()
    x, ids, w = inputs(4289, "real", 21, dev)
    t = M.plan(prod, ids.to(torch.long), w, len(L._exl3_inners), emap)
    a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
    out = torch.full((4289, 4096), float("nan"), dtype=torch.bfloat16, device=dev)
    P = L._exl3_ptrs
    ext.gather2(x, t["local"], t["pos"], P["gate_suh"], a8, asc, out, t["topk"], len(L._exl3_inners))
    sync = torch.zeros(1 + 2 * int(t["seg_expert"].numel()), dtype=torch.int32, device=dev)
    ext.fused(a8, asc, a8d, dsc, a16, out, P["gate_trellis"], P["up_trellis"], P["gate_svh"], P["up_svh"],
              P["down_trellis"], P["down_suh"], P["down_svh"], t["row_token"], t["row_weight"], t["seg_expert"],
              t["seg_row0"], t["seg_rows"], t["num_segs"], sync, float(C.LIMIT), 12, 0, 0)
    ref = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "f32"})
    CHK(bool(torch.isfinite(out).all()) and rel(out, ref) < 6e-3,
        f"[zeroing] NaN-prefilled bf16 out fully overwritten, == fp32 acc ({rel(out, ref):.2e})")
    # 3. edge cases
    n_exp = len(L._exl3_inners)
    half_map = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    half_map[: n_exp // 2] = torch.arange(n_exp // 2, dtype=torch.long, device=dev)
    x, ids, w = inputs(2048, "real", 22, dev)
    ids[:5] = torch.arange(n_exp // 2, n_exp // 2 + 8, device=dev)        # rows 0-4: every expert non-local
    bo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=half_map, sched={"acc": "bf16"})
    fo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=half_map, sched={"acc": "f32"})
    CHK(bool((bo[:5] == 0).all()) and rel(bo, fo) < 6e-3,
        f"[edge expert_map half non-local] all-non-local rows exactly 0, rest == fp32 acc ({rel(bo, fo):.2e})")
    g = torch.Generator().manual_seed(5)
    for tag, T, ids_fn in (("T=1", 1, lambda T: C.routing("real", T, 3, dev)),
                           ("T=17", 17, lambda T: C.routing("real", T, 3, dev)),
                           ("one expert holds all", 512, lambda T: torch.cat(
                               [torch.full((T, 1), 7, device=dev), C.routing("real", T, 4, dev)[:, 1:]], 1))):
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        if T > 1:
            x[0] = 0
        ids = ids_fn(T)
        w = C.weights_for(T, 9, dev).float()
        bo = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "bf16"})
        fo = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "f32"})
        zero_ok = T == 1 or bool((bo[0] == 0).all())
        CHK(rel(bo, fo) < 6e-3 and bool(torch.isfinite(bo).all()) and zero_ok,
            f"[edge {tag}] == fp32 acc ({rel(bo, fo):.2e}), finite, zero row exact ({zero_ok})")
    # 4. hook
    for bad in ("fp16", "1", "BF16x", "f64"):
        rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": bad})
        CHK(not rep["installed"] and not M.ACC["bf16"], f"[hook] GLM53_MOE_E4M3_ACC={bad!r} refused")
    base = prod.apply_exl3_experts
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": "bf16"}, load_selftest=False)
    CHK(rep["installed"] and rep.get("acc_bf16") is True and M.ACC["bf16"], f"[hook] bf16 installed: {rep}")
    for a in ("_glm53_moe_e4m3_ok",):
        if hasattr(L, a):
            delattr(L, a)
    st = M.selftest(prod, L, C.LIMIT)
    print(f"  [hook] self-test with the bf16 accumulator: {st}", flush=True)
    CHK(st["ok"] and "acc_bf16_rel_l2" in st and st["acc_bf16_rel_l2"] < M.ACC_SELFTEST_TOL and st["rel_l2"] < 3e-3,
        f"[hook] self-test runs both accumulators and passes ({st})")
    for d16 in (False, True):
        M.DOWN16["on"] = d16
        try:
            st = M.selftest(prod, L, C.LIMIT)
        finally:
            M.DOWN16["on"] = False
        CHK(st["ok"], f"[hook] self-test (DOWN16={d16}) passes with the bf16 accumulator: {st}")
    x, ids, w = inputs(13824, "real", 23, dev)
    hooked = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
    direct = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "bf16"})
    f32 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"acc": "f32"})
    CHK(hooked.dtype == torch.bfloat16 and rel(hooked, direct) < 2e-3 and rel(hooked, f32) < 6e-3,
        f"[hook] served call is the bf16 accumulator (vs run(bf16) {rel(hooked, direct):.2e}, vs fp32 "
        f"{rel(hooked, f32):.2e})")
    xh = x.half()
    hh = prod.apply_exl3_experts(xh, ids, w, L, limit=C.LIMIT)
    CHK(hh.dtype == torch.float16 and rel(hh, f32) < 2e-3,
        f"[hook] fp16 input keeps the fp32 accumulator ({rel(hh, f32):.2e})")
    M.uninstall(prod)
    CHK(prod.apply_exl3_experts is base and not M.ACC["bf16"], "[hook] uninstall restores the function and the mode")
    for val in (None, "", "f32", "F32"):
        env = {"GLM53_MOE_E4M3": "1"} if val is None else {"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": val}
        rep = M.install(prod, environ=env, load_selftest=False)
        ok = rep["installed"] and rep.get("acc_bf16") is False and not M.ACC["bf16"]
        o = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
        r = M.run(prod, x, ids, w, L, C.LIMIT).to(torch.bfloat16)
        diff = float((o.float() != r.float()).float().mean())
        CHK(ok and diff < 1e-3, f"[hook] GLM53_MOE_E4M3_ACC={val!r} -> fp32 accumulator (served == run().to(bf16) "
            f"except {diff:.1e} of elements: the fp32 atomics order)")
        M.uninstall(prod)
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "0", "GLM53_MOE_E4M3_ACC": "bf16"}, load_selftest=False)
    CHK(not rep["installed"] and prod.apply_exl3_experts is base and not M.ACC["bf16"],
        "[hook] GLM53_MOE_E4M3=0 with ACC=bf16: nothing installed")
    PC.uninstall(prod)
    CHK.summary()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
