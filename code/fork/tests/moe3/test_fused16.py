"""GLM53_MOE_FUSED16 tests (real layer-10 experts, TP=2 rank-0 shard; production module of the image, prefill cap 1 as
deployed). Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe3/test_fused16.py
  E  exp_prod == (float) exp((double) a) for every float in [-87, 88]
  B  every schedule, through production's own grouped arguments: h13 / h2 BIT-IDENTICAL to production's, out within
     production's own fp32-atomics spread (x100, floor 2e-7); NaN / inf / huge activations; expert_map with non-local
     experts; top-k 4; mixed prefill+decode sizes
  H  the hook: production's apply_exl3_experts with the wrapper == without it (bf16 output: differing elements are
     the atomics-order class only), served / passed counts by size, capture pass-through, self-test at load,
     knob values, fingerprint refusal, uninstall, no double wrap
"""
from __future__ import annotations

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ndiff16(u, v):
    n = 0
    for i in range(0, u.shape[0], 8192):
        n += int((u[i:i + 8192].view(torch.int16) != v[i:i + 8192].view(torch.int16)).sum())
    return n


def rel(u, v):
    num = den = 0.0
    for i in range(0, u.shape[0], 2048):
        a, b = u[i:i + 2048].double(), v[i:i + 2048].double()
        m = torch.isfinite(b)
        num += float((a[m] - b[m]).pow(2).sum())
        den += float(b[m].pow(2).sum())
    return (num / max(den, 1e-300)) ** 0.5


def test_exp(F):
    E = F._ext()
    cnt = torch.zeros(2, dtype=torch.int64, device="cuda")
    for b0, b1 in ((0, bits(88.0) + 1), (0x80000000, bits(-87.0) + 1)):
        for s in range(b0, b1, 1 << 26):
            E.exp_check(s, min(s + (1 << 26), b1), cnt)
    torch.cuda.synchronize()
    bad, fb = cnt.tolist()
    print(f"E exp_prod vs (float) exp((double) a), every float in [-87, 88]: mismatches {bad}, fallbacks {fb}", flush=True)
    CHK(bad == 0, f"exp_prod mismatches {bad}")


def grouped_args(prod, L, x, ids, w):
    a = {}
    orig = prod.apply_exl3_grouped_fat

    def capture(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit):
        a.update(xh=xh.clone(), counts=counts.clone(), token_sorted=token_sorted.clone(),
                 weight_sorted=weight_sorted.clone(), cap=cap, limit=limit, out_in=out.clone())
        return orig(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit)
    prod.apply_exl3_grouped_fat = capture
    try:
        prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
    finally:
        prod.apply_exl3_grouped_fat = orig
    return a


def check_schedules(prod, F, L, tag, x, ids, w, modes=("p16b", "sep"), extra=None):
    a = grouped_args(prod, L, x, ids, w)
    if not a:
        print(f"B {tag}: no grouped call", flush=True)
        return
    orig = prod.apply_exl3_grouped_fat
    dev = x.device
    rows_cap = a["token_sorted"].numel()
    sc = prod._grouped_scratch(dev, rows_cap, 4096, 1024)
    ref = []
    for _ in range(2):
        o = a["out_in"].clone()
        orig(a["xh"], o, a["counts"], a["token_sorted"], a["weight_sorted"], L, a["cap"], a["limit"])
        ref.append(o)
    nf = int(a["counts"][a["counts"] > a["cap"]].sum())
    h13p, h2p = sc["h13"][:nf].clone(), sc["h2"][:nf].clone()
    spread = rel(ref[0], ref[1])
    tol = max(100 * spread, 2e-7)
    for mode in modes:
        for sched in ([{"mode": mode}] + ([{"mode": mode, "nchunks": 1}, {"mode": mode, "nchunks": 4}]
                                          if mode == "sep" else [])):
            sc["h13"][:nf].zero_()
            sc["h2"][:nf].zero_()
            keep = {}
            o = a["out_in"].clone()
            F.run(prod, a["xh"], o, a["counts"], a["token_sorted"], a["weight_sorted"], L, a["cap"], a["limit"],
                  keep=keep, sched=sched)
            n13, n2 = ndiff16(sc["h13"][:nf], h13p), ndiff16(sc["h2"][:nf], h2p)
            d = rel(o, ref[0])
            fin = bool((torch.isfinite(o) == torch.isfinite(ref[0])).all())
            print(f"B {tag} {sched}: fat rows {nf}: h13 diff {n13}, h2 diff {n2}; out rel {d:.2e} (prod spread "
                  f"{spread:.2e}); finite pattern equal {fin}", flush=True)
            CHK(n13 == 0 and n2 == 0, f"{tag} {sched}: h13/h2 not bit-identical ({n13}, {n2})")
            CHK(d <= tol, f"{tag} {sched}: out rel {d} > {tol}")
            CHK(fin, f"{tag} {sched}: finite pattern differs")
            del o, keep
    del a, ref, h13p, h2p
    torch.cuda.empty_cache()


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_fused16 as F
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    test_exp(F)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]

    def inputs(T, kind, seed, xs=1.0, topk=None):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(T, 4096, generator=g) * xs).to(torch.bfloat16).to(dev)
        ids = C.routing(kind, T, seed, dev)
        w = C.weights_for(T, seed, dev).float()
        if topk:
            ids, w = ids[:, :topk].contiguous(), w[:, :topk].contiguous()
            w = w / w.sum(-1, keepdim=True)
        return x, ids, w

    # ---- B: schedules vs production's kernels
    for T, kind in ((13824, "real"), (13856, "real"), (13824, "collapsed"), (8192, "real"), (4289, "real"),
                    (4289, "collapsed"), (2048, "real"), (300, "real")):
        x, ids, w = inputs(T, kind, 100 + T)
        check_schedules(prod, F, L, f"T={T} {kind}", x, ids, w)
    x, ids, w = inputs(3000, "real", 7, xs=40.0)                   # large activations: fp16 overflow -> inf in h13
    x[5, :64] = float("inf")
    x[9, 100:200] = float("nan")
    check_schedules(prod, F, L, "T=3000 x40 + inf/nan rows", x, ids, w)
    x, ids, w = inputs(2500, "real", 8, topk=4)
    check_schedules(prod, F, L, "T=2500 top-4", x, ids, w)
    # expert_map: half of the global experts are non-local (TP-style local map)
    n_exp = len(L._exl3_inners)
    em = torch.full((2 * n_exp,), -1, dtype=torch.int32, device=dev)
    em[torch.arange(0, 2 * n_exp, 2, device=dev)] = torch.arange(n_exp, dtype=torch.int32, device=dev)
    saved_map = L.expert_map
    L.expert_map = em
    try:
        g = torch.Generator().manual_seed(9)
        T = 4000
        ids = torch.stack([torch.randperm(2 * n_exp, generator=g)[:8] for _ in range(T)]).to(dev)
        w = torch.rand(T, 8, generator=g).to(dev)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        if hasattr(prod, "_EXL3_EXPERT_MAP_CACHE"):
            prod._EXL3_EXPERT_MAP_CACHE.clear()
        check_schedules(prod, F, L, "T=4000 expert_map half non-local", x, ids, w)
    finally:
        L.expert_map = saved_map
        for attr in ("_exl3_expert_map_dev", "_exl3_pinned_expert_map"):
            if hasattr(L, attr):
                delattr(L, attr)

    # ---- H: the hook
    base = prod.apply_exl3_experts
    orig_gf = prod.apply_exl3_grouped_fat
    for v in ("", "0"):
        r = F.install(prod, environ={"GLM53_MOE_FUSED16": v})
        CHK(not r["installed"] and prod.apply_exl3_grouped_fat is orig_gf, f"value {v!r} must install nothing")
    for v in ("2", "on", "true"):
        r = F.install(prod, environ={"GLM53_MOE_FUSED16": v})
        CHK(not r["installed"] and prod.apply_exl3_grouped_fat is orig_gf, f"value {v!r} must be refused")
    # fingerprint refusal
    real_tables = prod.build_grouped_fat_tables

    def build_grouped_fat_tables(*a, **k):        # a different function body
        return real_tables(*a, **k)
    prod.build_grouped_fat_tables = build_grouped_fat_tables
    r = F.install(prod, environ={"GLM53_MOE_FUSED16": "1"})
    CHK(not r["installed"] and "fingerprint" in r["reason"], f"fingerprint drift must refuse: {r}")
    prod.build_grouped_fat_tables = real_tables
    # install, self-test at load through process_weights_after_loading
    r = F.install(prod, environ={"GLM53_MOE_FUSED16": "1"})
    CHK(r["installed"] and r.get("load_hook"), f"install: {r}")
    r2 = F.install(prod, environ={"GLM53_MOE_FUSED16": "1"})
    CHK(r2["reason"] == "already installed" and prod.apply_exl3_grouped_fat._glm53_moe_fused16_orig is orig_gf,
        "no double wrap")
    L2 = make_real_layer(prod, dev)                    # production's process_weights_after_loading -> self-test
    CHK(getattr(L2, "_glm53_moe_fused16_ok", None) is True, "load-time self-test passed on the second layer")
    CHK(F.STATS["layers_ok"] >= 1 and F.STATS["selftest_failed"] == 0, f"stats {F.STATS}")
    del L2
    torch.cuda.empty_cache()
    for T, kind, want in ((13824, "real", "p16b"), (8192, "real", "sep"), (4289, "collapsed", "sep"),
                          (4095, "real", None), (300, "real", None)):
        x, ids, w = inputs(T, kind, 500 + T)
        F.uninstall(prod)
        y0 = base(x, ids, w, L, limit=C.LIMIT)
        y0b = base(x, ids, w, L, limit=C.LIMIT)
        F.install(prod, environ={"GLM53_MOE_FUSED16": "1"})
        s0 = dict(F.STATS)
        y1 = base(x, ids, w, L, limit=C.LIMIT)
        nd = int((y1.view(torch.int16) != y0.view(torch.int16)).sum())
        nd_aa = int((y0b.view(torch.int16) != y0.view(torch.int16)).sum())
        served = F.STATS["served_" + want] - s0["served_" + want] if want else 0
        passed = F.STATS["passed"] - s0["passed"]
        print(f"H T={T} {kind}: bf16 output elements differing from unhooked production {nd} of {y0.numel()} "
              f"(production vs itself: {nd_aa}); rel {rel(y1.float(), y0.float()):.2e}; served {want} {served}, "
              f"passed {passed}", flush=True)
        CHK(y1.dtype == y0.dtype and y1.shape == y0.shape, "dtype/shape")
        # fp32 sum-order differences can only flip a bf16 rounding: every differing element must be ONE bf16 ulp
        # away (same sign), and rare
        a32, b32 = y1.float(), y0.float()
        ulp = torch.maximum(a32.abs(), b32.abs()) * 2.0 ** -7                # >= one bf16 ulp of the larger value
        rms = b32.pow(2).mean(dim=1, keepdim=True).sqrt()
        far = int(((a32 - b32).abs() > ulp + 1e-5 * rms).sum())
        mx = float(((a32 - b32).abs() / rms.clamp_min(1e-30)).max())
        print(f"H T={T}: elements beyond one bf16 ulp + 1e-5 x row rms: {far}; max |diff| / row rms {mx:.2e}", flush=True)
        del a32, b32, ulp, rms
        CHK(far == 0 and nd <= 1e-3 * y0.numel(), f"T={T}: hooked output differs beyond the atomics class")
        if want:
            CHK(served == 1 and passed == 0, f"T={T}: expected one {want} call")
        else:
            CHK(passed == 1 and nd == 0 or (passed == 1 and nd <= nd_aa * 10 + 10), f"T={T}: pass-through expected")
    # capture: a decode-sized call never reaches the grouped tier; a capture of a prefill-sized grouped call passes
    x, ids, w = inputs(4096, "real", 77)
    s0 = dict(F.STATS)
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        gr = torch.cuda.CUDAGraph()
        out = torch.zeros(4096, 4096, dtype=torch.float32, device=dev)
        xh = torch.zeros(4096, 4096, dtype=torch.float16, device=dev)
        counts = torch.zeros(len(L._exl3_inners), dtype=torch.long, device=dev)
        try:
            with torch.cuda.graph(gr, stream=st):
                F._serve(prod, lambda *aa: None, xh, out, counts, torch.zeros(128, dtype=torch.long, device=dev),
                         torch.zeros(128, dtype=torch.float16, device=dev), L, 1, C.LIMIT)
        except Exception as exc:  # noqa: BLE001
            print(f"H capture probe raised {exc!r}", flush=True)
    CHK(F.STATS["passed"] - s0["passed"] == 1 and F.STATS["served"] == s0["served"], "capture must pass through")
    F.uninstall(prod)
    CHK(prod.apply_exl3_grouped_fat is orig_gf, "uninstall restores production's function")
    cls = prod.Exl3MoEMethod
    CHK(not getattr(cls.process_weights_after_loading, "_glm53_moe_fused16", False), "uninstall restores pwal")
    PC.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
