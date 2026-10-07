"""Paired A/B of two TF kernel variants in one process (docs/OPTIMIZATION.md, acceptance protocol).

Variants are selected with the extension's test hook ext.set_variant(id) (0 = shipped, 1 = the master kernels,
2.. experiments); a captured CUDA graph keeps the variant it was captured with. For every T and routing kind:
12 production-argument sets (3 layers x 4 routings, n = 288, K = 4096, N = 1024, topk = 8, cold weights), one
CUDA graph of the 12 bare TF calls per variant, >= 9 alternating rounds; the statistic is the median over rounds of
the per-round ratio B / A (pairing cancels nodeC drift), with the per-variant median us and spread.

Correctness of what is compared, per T and kind:
  - bitwise: single-contribution routing (topk = 1, so no atomic-order noise) through the production apply's own
    argument capture: out of variant B == out of variant A (torch.equal), eager and graph replay;
  - E.1: the timed 12-set graphs are replayed once more and each output compared with production exl3_moe.
Optional per-stage timing (--stages): graph replay of grouped gate/up and grouped down alone per variant.

--mode apply: A / B are the K2 switch (0 = TF_EXL3_APPLY off: production's routing prelude + the TF exl3_moe path;
1 = on: the hooked apply serves the call from the router ids); the graphs hold 12 production apply calls
(apply_exl3_fused_moe, returning its fp32 out) and the E.1 read-back compares each with production's apply + exl3_moe.
--mode full: A / B are "<variant>:<K2 switch>", e.g. `1:0 0:1` = the master TF (master kernels, production's
prelude + the exl3_moe path) vs the shipped TF (variant 0, K2 on), paired in one process on production apply graphs.

Every configuration line (per-variant median us, GB/s, spreads, the bitwise / E.1 verdicts) is printed; keep the
whole output as the log of a run (docs/logs/), not only the summary.

Usage: tests/gpu_run.sh python3 -u tests/bench_variant_ab.py A B [--t 1,2,...] [--kinds rand,corr40] [--rounds 9]
       [--stages] [--repeat N] [--mode kernel|apply|full]
"""
from __future__ import annotations

import argparse

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LAYERS, ROUTES = 3, 4
LIMIT = 10.0
T_ALL = (1, 2, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128)
PER_EXPERT = 3 * (K // 16) * (N // 16) * 128        # bytes of gate + up + down trellis of one expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--t", default=",".join(map(str, T_ALL)))
    ap.add_argument("--kinds", default="rand,corr40")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--mode", default="kernel", choices=("kernel", "apply", "full"))
    opt = ap.parse_args()
    Ts = [int(v) for v in opt.t.split(",")]
    kinds = opt.kinds.split(",")

    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    ext = tf.load_ext()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    orig, disp = rep["orig"], xl.exl3_moe
    tf.CFG.strict = True
    layers = []
    for i in range(LAYERS):
        W = H.Weights(NEXP, K, N, dev, seed=500 + i)
        layers.append((W, H.make_layer(prod, W)))
    def spec(sv):                                   # full: (variant, K2 switch); kernel: variant; apply: K2 switch
        if opt.mode == "full":
            v_, a_ = sv.split(":")
            return int(v_), int(a_)
        return int(sv)

    VA, VB = spec(opt.a), spec(opt.b)
    print(f"mode {opt.mode}: A = {VA}, B = {VB} (kernel variants: {ext.num_variants()}); rounds {opt.rounds}; "
          f"repeat {opt.repeat}; T {Ts}; kinds {kinds}", flush=True)

    def graph_of(fns):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for f in fns:
                f()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for f in fns:
                f()
        return gr

    def timed(gr, calls, reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / (reps * calls)

    def paired(ga, gb, calls):
        ga.replay(); gb.replay()
        torch.cuda.synchronize()
        t1 = timed(ga, calls, 1)
        reps = max(2, int(20000 / (t1 * calls)))
        ta, tb = [], []
        for r in range(opt.rounds):
            order = ((ga, ta), (gb, tb)) if r % 2 == 0 else ((gb, tb), (ga, ta))
            for gr, acc in order:
                acc.append(timed(gr, calls, reps))
        med = lambda v: sorted(v)[len(v) // 2]
        ratio = med([b / a for a, b in zip(ta, tb)])
        return med(ta), med(tb), ratio, (max(ta) - min(ta)) / med(ta), (max(tb) - min(tb)) / med(tb)

    def tf_call(args):
        p = tf._plan(args, require_ok=False)
        assert p is not None
        tf._launch_kernels(p, args)

    def with_variant(v, fn):
        if opt.mode in ("apply", "full"):
            var, apl = (int(ext.variant()), v) if opt.mode == "apply" else v
            old, old_v = tf.CFG.apply, int(ext.variant())
            tf.CFG.apply = bool(apl)
            ext.set_variant(var)
            try:
                return fn()
            finally:
                tf.CFG.apply = old
                ext.set_variant(old_v)
        old = ext.variant()
        ext.set_variant(v)
        try:
            return fn()
        finally:
            ext.set_variant(old)

    orig_apply = getattr(prod.apply_exl3_fused_moe, "_tf_exl3_orig", prod.apply_exl3_fused_moe)

    results = []
    for rep_i in range(opt.repeat):
        for kind in kinds:
            for T in Ts:
                g = torch.Generator().manual_seed(9000 + T + (0 if kind == "rand" else 31))
                sets, distinct, empty, raw = [], 0.0, 0.0, []
                for li, (W, layer) in enumerate(layers):
                    for _ in range(ROUTES):
                        x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                        ids = H.routing_ids(kind, T, NEXP, TOPK, g, dev)
                        w = H.random_weights(T, TOPK, g, dev)
                        args = H.capture_args(prod, xl, x, ids, w, layer, LIMIT)
                        args = H.with_out(args, torch.zeros(T, K, dtype=torch.float32, device=dev))
                        sets.append(args)
                        raw.append((x, ids, w, layer))
                        distinct += int(torch.unique(ids).numel())
                        P = T * TOPK
                        cnt = torch.bincount(ids.flatten(), minlength=NEXP)
                        empty += tf.s_cap(P, NEXP) - int(((cnt + 15) // 16).sum())
                distinct /= len(sets)
                empty /= len(sets)
                # bitwise: single-contribution routing, B vs A (eager and graph)
                x1 = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                ids1 = torch.stack([torch.randperm(NEXP, generator=g)[:1] for _ in range(T)]).to(dev)
                w1 = H.random_weights(T, 1, g, dev)
                a1 = H.capture_args(prod, xl, x1, ids1, w1, layers[0][1], LIMIT)
                outs = []
                for v in (VA, VB):
                    if opt.mode != "kernel":
                        L0 = layers[0][1]
                        o = with_variant(v, lambda: prod.apply_exl3_fused_moe(x1, ids1, w1, L0, L0._exl3_inners, None,
                                                                             LIMIT))
                        box = []
                        gr1 = with_variant(v, lambda: graph_of([lambda: box.append(prod.apply_exl3_fused_moe(
                            x1, ids1, w1, L0, L0._exl3_inners, None, LIMIT))]))
                        gr1.replay()
                        torch.cuda.synchronize()
                        outs.append((o, box[-1].clone()))
                        del gr1
                        continue
                    o = torch.zeros(T, K, dtype=torch.float32, device=dev)
                    with_variant(v, lambda: tf_call(H.with_out(a1, o)))
                    og = torch.zeros(T, K, dtype=torch.float32, device=dev)
                    gr1 = with_variant(v, lambda: graph_of([lambda: tf_call(H.with_out(a1, og))]))
                    og.zero_()
                    gr1.replay()
                    torch.cuda.synchronize()
                    outs.append((o, og.clone()))
                    del gr1
                same_tab = (opt.mode == "apply" or (opt.mode == "kernel" and ext.variant_sk_table(VA) ==
                            ext.variant_sk_table(VB)) or (opt.mode == "full" and ext.variant_sk_table(VA[0]) ==
                                                          ext.variant_sk_table(VB[0])))
                if same_tab:                   # same operations in the same order: bitwise
                    beq = torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1]) and torch.equal(
                        outs[0][0], outs[0][1])
                else:                          # another split-count table: another fp32 summation tree -> E.1
                    beq = tf.passes_e1(tf.compare(outs[1][0], outs[0][0])) and torch.equal(outs[0][0], outs[0][1]) \
                        and torch.equal(outs[1][0], outs[1][1])
                ck(beq, f"{kind} T={T}: single-contribution out of variant {VB} vs variant {VA} "
                        f"({'bitwise' if same_tab else 'E.1'})")
                gouts = {}
                if opt.mode != "kernel":
                    def mk_apply(tag):
                        gouts[tag] = []
                        return [lambda r=r: gouts[tag].append(prod.apply_exl3_fused_moe(
                            r[0], r[1], r[2], r[3], r[3]._exl3_inners, None, LIMIT)) for r in raw]
                    ga = with_variant(VA, lambda: graph_of(mk_apply("a")))
                    gb = with_variant(VB, lambda: graph_of(mk_apply("b")))
                else:
                    fns = [lambda a=a: tf_call(a) for a in sets]
                    ga = with_variant(VA, lambda: graph_of(fns))
                    gb = with_variant(VB, lambda: graph_of(fns))
                ta, tb, ratio, sa, sb = paired(ga, gb, len(sets))
                # E.1 read-back of the timed graphs vs production
                worst = 0.0
                for tag, gr in (("a", ga), ("b", gb)):
                    for a in sets:
                        a[1].zero_()
                    gr.replay()
                    torch.cuda.synchronize()
                    got = [a[1] for a in sets] if opt.mode == "kernel" else gouts[tag][-len(sets):]
                    for a, y in zip(sets, got):
                        o_x = torch.zeros(T, K, dtype=torch.float32, device=dev)
                        orig(*H.with_out(a, o_x))
                        torch.cuda.synchronize()
                        m = tf.compare(y, o_x)
                        worst = max(worst, m["rel_l2"])
                        ck(tf.passes_e1(m), f"{kind} T={T}: graph replay vs orig {m}")
                del ga, gb
                gbs = lambda us: distinct * PER_EXPERT / (us * 1e-6) / 1e9
                r = dict(kind=kind, T=T, distinct=distinct, empty=empty, ta=ta, tb=tb, ratio=ratio, sa=sa, sb=sb,
                         worst=worst, beq=beq)
                line = (f"{kind:6s} T={T:3d} distinct {distinct:6.1f} empty-seg {empty:6.1f} | A {ta:8.1f} us "
                        f"({gbs(ta):5.1f} GB/s, {gbs(ta) / 2.5:4.1f}%) B {tb:8.1f} us ({gbs(tb):5.1f} GB/s, "
                        f"{gbs(tb) / 2.5:4.1f}%) | B/A {ratio:.4f} | spread A {sa * 100:4.1f}% B {sb * 100:4.1f}% | "
                        f"{'bitwise' if same_tab else 'E.1(split table differs)'} {beq} | E.1 worst {worst:.2e}")
                if opt.stages:
                    plans = [(a, tf._plan(a, require_ok=False)) for a in sets]
                    sc = plans[0][1].scratch
                    tabs = []
                    for a, p in plans:
                        S = tf.s_cap(p.P, p.n)
                        t = {"pe": torch.empty(p.P, dtype=torch.int32, device=dev),
                             "se": torch.empty(S, dtype=torch.int32, device=dev),
                             "s0": torch.empty(S, dtype=torch.int32, device=dev),
                             "sr": torch.empty(S, dtype=torch.int32, device=dev),
                             "ns": torch.empty(1, dtype=torch.int32, device=dev), "S": S}
                        ext.route_prep(a[2], p.P, p.R, t["pe"], t["se"], t["s0"], t["sr"], t["ns"])
                        tabs.append(t)
                    dsk = lambda P: int(ext.split_counts(P, K, N)[1])      # evaluated at capture (per variant)
                    gsk = lambda P: int(ext.split_counts(P, K, N)[0])
                    st = {}
                    for kind_s in ("grouped_gu", "grouped_down"):
                        fs = []
                        for (a, p), t in zip(plans, tabs):
                            if kind_s == "grouped_gu":
                                fs.append(lambda a=a, p=p, t=t: ext.grouped(sc.xg, sc.xu, a[13], a[16], t["se"], t["s0"],
                                          t["sr"], t["ns"], sc.z, 2, K, N, p.P, gsk(p.P), tf.GATEUP_CFG[0],
                                          tf.GATEUP_CFG[1], t["S"]))
                            else:
                                fs.append(lambda a=a, p=p, t=t: ext.grouped(sc.xd, sc.xd, a[19], a[19], t["se"], t["s0"],
                                          t["sr"], t["ns"], sc.z, 1, N, K, p.P, dsk(p.P), tf.DOWN_CFG[0],
                                          tf.DOWN_CFG[1], t["S"]))
                        g1 = with_variant(VA, lambda: graph_of(fs))
                        g2 = with_variant(VB, lambda: graph_of(fs))
                        st[kind_s] = paired(g1, g2, len(fs))
                        del g1, g2
                    mb = {"grouped_gu": 2 / 3, "grouped_down": 1 / 3}
                    line += " | stages " + " ".join(
                        f"{k} A {v[0]:.1f} B {v[1]:.1f} us ({gbs(v[1]) * mb[k]:5.1f} GB/s) B/A {v[2]:.4f}"
                        for k, v in st.items())
                    r["stages"] = st
                print(line, flush=True)
                results.append(r)

    print()
    print(f"summary B/A ({VB} / {VA}, mode {opt.mode}), median of per-round ratios; < 1 = B faster")
    for kind in kinds:
        rs = [r for r in results if r["kind"] == kind]
        print(f"  {kind:6s} " + "  ".join(f"T{r['T']}:{r['ratio']:.3f}" for r in rs))
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
