"""E.5: decode A/B timing, TF path vs production exllamav3_ext.exl3_moe (docs/DESIGN.md §E.5).

n = 288, K = 4096, N = 1024, topk = 8, 3 distinct layers (5.2 GiB of trellis) x 4 routing sets cycled, so every
call reads cold weights (T = 1: 12 x 48 MiB distinct working set >> L2). Routings:
  rand    independent random top-8 per row, T in {1, 2, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128}
          (128 = R: the largest single-launch decode call, P = P_cap = 1024);
  corr40  DFlash-like correlated routing (harness.correlated_ids: sequences of k+1 in {5, 6, 8} rows, each drawing its
          rows' experts from a 40-expert pool of its own), T in {5, 8, 12, 16, 24, 32, 48, 64} (the production range).
Per configuration: distinct experts per call, empty segment blocks (S_cap - nseg), and every GB/s figure is the
distinct experts' trellis bytes over the time, also as % of the measured 250 GB/s read ceiling (tests/bw_ceiling.py).
Measured, 20 warmups + 200 calls each, CUDA events:
  bare   : the exl3_moe call alone on pre-built production args (XL: orig(*args); TF: dispatcher(*args))
  apply  : production apply_exl3_fused_moe incl. routing (exl3_moe = orig vs = dispatcher)
  each eagerly and under CUDA-graph replay (graph of 12 calls over 3 layers sharing the TF scratch, replayed;
  per-call time = total / calls)
TF per-stage breakdown (graph replay of each stage alone), touched-expert GB/s, and the T window where the
graph-replay production-apply speedup is >= 1.05 (-> TF_EXL3_TOKENS). Timings are reported, not asserted.
Correctness of what is timed, per T (all 12 sets):
  - bare eager: dispatcher (TF) vs orig, E.1 tolerances, TF path taken;
  - bare graph: after timing, all outputs zeroed and the 12-call TF graph replayed once; each of the 12
    outputs vs eager orig, E.1 tolerances (the multi-layer, shared-scratch replay that production runs);
  - apply eager: production apply_exl3_fused_moe (fp32 out) with the dispatcher vs with orig, E.1 tolerances;
  - apply graph: after timing, the 12-call TF apply graph replayed once; each output vs the eager orig apply,
    E.1 tolerances.
"""
from __future__ import annotations

import os
import time

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LAYERS, ROUTES = 3, 4
WARM, ITERS, ROUNDS = 20, 216, 9
CEIL = 250.0                     # GB/s, tests/bw_ceiling.py (streaming 250.1, scattered 2 MiB chunks 247-251)
T_RAND = (1, 2, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128)
T_CORR = (5, 8, 12, 16, 24, 32, 48, 64)
LIMIT = 10.0


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    orig, disp = rep["orig"], xl.exl3_moe
    tf.CFG.strict = True
    # production baseline = the exllamav3_ext.exl3_moe this process has: stock, or the native thin-decode kernel
    # (a rebuilt exllamav3_ext bound over the image's + GLM53_EXL3_MOE_FAST=1, as the launcher does). With
    # TF_EXL3_BENCH_SHARED_SUH=1 gate and up share their input rotation, so the launcher overlay aliases the up_suh
    # pointer table (FAST=1) and the thin kernel takes its shared-input variant (skips one Hadamard).
    shared_suh = os.environ.get("TF_EXL3_BENCH_SHARED_SUH", "0") == "1"
    print(f"baseline (XL): {rep.get('orig_kernel')}; K2 apply path {'on' if rep.get('apply_hook') else 'off'}; "
          f"shared gate/up suh {shared_suh}", flush=True)
    # bench-only: time another grouped-GEMV variant (docs/OPTIMIZATION.md; 1 = the master kernel)
    variant = int(os.environ.get("TF_EXL3_BENCH_VARIANT", "0"))
    tf.load_ext().set_variant(variant)
    print(f"TF grouped-GEMV variant {variant} (0 = shipped)", flush=True)
    layers = []
    for i in range(LAYERS):
        W = H.Weights(NEXP, K, N, dev, seed=500 + i, shared_suh=shared_suh)
        layers.append((W, H.make_layer(prod, W)))
        if i == 0:
            L0 = layers[0][1]
            print(f"layer: _exl3_shared_w13_suh {getattr(L0, '_exl3_shared_w13_suh', None)}, up_suh table aliased "
                  f"to gate_suh {L0._exl3_ptrs['up_suh'] is L0._exl3_ptrs['gate_suh']}", flush=True)
        ck(tf.REG.get((0, layers[-1][1]._exl3_ptrs["gate_trellis"].data_ptr())) is not None, f"layer {i} not registered")
    trellis_gib = sum(W.nbytes() for W, _ in layers) / H.GiB
    per_expert = 3 * (K // 16) * (N // 16) * 128
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}: {props.multi_processor_count} SMs; exl3_moe concurrency "
          f"{int(xl.exl3_moe_max_concurrency(0))}; {LAYERS} layers x {NEXP} experts = {trellis_gib:.2f} GiB of weights; "
          f"per expert {per_expert / 2**20:.1f} MiB; TF extension {tf.EXT_SOURCE}")

    def timed(fn, calls_per_fn):
        for _ in range(max(1, WARM // calls_per_fn)):
            fn()
        torch.cuda.synchronize()
        reps = max(1, ITERS // calls_per_fn)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / (reps * calls_per_fn)

    def ab(fn_x, fn_t, calls_per_fn):
        """XL and TF alternated in ROUNDS rounds (other GPU clients share nodeC; the order flips every round);
        median per-call us of each and spreads. The per-round speedups XL/TF are kept in ratio_log."""
        for _ in range(max(1, WARM // calls_per_fn)):
            fn_x()
            fn_t()
        torch.cuda.synchronize()
        reps = max(1, ITERS // ROUNDS // calls_per_fn)
        tx, tt = [], []
        for r_ in range(ROUNDS):
            for fn, acc in (((fn_x, tx), (fn_t, tt)) if r_ % 2 == 0 else ((fn_t, tt), (fn_x, tx))):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(reps):
                    fn()
                e.record()
                torch.cuda.synchronize()
                acc.append(s.elapsed_time(e) * 1000.0 / (reps * calls_per_fn))
        med = lambda v: sorted(v)[len(v) // 2]
        ratio_log.append(med([a / b for a, b in zip(tx, tt)]))
        return med(tx), med(tt), (max(tx) - min(tx)) / med(tx), (max(tt) - min(tt)) / med(tt)

    ratio_log = []

    graph_outs = {}

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
            outs = [f() for f in fns]
        graph_outs[id(gr)] = outs                               # graph-owned outputs (apply returns its result)
        return gr

    rows = []
    for kind, T in [("rand", t_) for t_ in T_RAND] + [("corr40", t_) for t_ in T_CORR]:
        g = torch.Generator().manual_seed(9000 + T + (0 if kind == "rand" else 31))
        sets = []                                              # (layer, x_bf16, ids, w, args)
        distinct = empty = 0.0
        for li, (W, layer) in enumerate(layers):
            for _ in range(ROUTES):
                x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                ids = H.routing_ids(kind, T, NEXP, TOPK, g, dev)
                w = H.random_weights(T, TOPK, g, dev)
                args = H.capture_args(prod, xl, x, ids, w, layer, LIMIT)
                args = H.with_out(args, torch.zeros(T, K, dtype=torch.float32, device=dev))
                sets.append((layer, x, ids, w, args))
                distinct += int(torch.unique(ids).numel())
                cnt = torch.bincount(ids.flatten(), minlength=NEXP)
                empty += tf.s_cap(T * TOPK, NEXP) - int(((cnt + 15) // 16).sum())
        distinct /= len(sets)
        empty /= len(sets)
        # correctness of what is being timed: bare eager (all 12 sets)
        ref_bare = []
        worst_c = {"bare_eager": 0.0, "bare_graph": 0.0, "apply_eager": 0.0, "apply_graph": 0.0}
        for layer, x, ids, w, args in sets:
            o_x = torch.zeros(T, K, dtype=torch.float32, device=dev)
            o_t = torch.zeros_like(o_x)
            orig(*H.with_out(args, o_x))
            c0 = tf.COUNTERS["tf_calls"]
            disp(*H.with_out(args, o_t))
            torch.cuda.synchronize()
            m = tf.compare(o_t, o_x)
            worst_c["bare_eager"] = max(worst_c["bare_eager"], m["rel_l2"])
            ck(tf.COUNTERS["tf_calls"] == c0 + 1, f"{kind} T={T}: TF path not taken")
            ck(tf.passes_e1(m), f"{kind} T={T}: TF result out of tolerance {m}")
            ref_bare.append(o_x)

        bare_x = [lambda a=s[4]: orig(*a) for s in sets]
        bare_t = [lambda a=s[4]: disp(*a) for s in sets]

        def apply_fns(use_tf):
            fs = []
            for layer, x, ids, w, _ in sets:
                def f(layer=layer, x=x, ids=ids, w=w, use_tf=use_tf):
                    xl.exl3_moe = disp if use_tf else orig
                    try:
                        return prod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, None, LIMIT)
                    finally:
                        xl.exl3_moe = disp
                fs.append(f)
            return fs

        app_x, app_t = apply_fns(False), apply_fns(True)
        # apply eager: production apply_exl3_fused_moe (returns its fp32 out) with the dispatcher vs with orig
        ref_apply = []
        for fx, ft in zip(app_x, app_t):
            yx, yt = fx(), ft()
            torch.cuda.synchronize()
            m = tf.compare(yt, yx)
            worst_c["apply_eager"] = max(worst_c["apply_eager"], m["rel_l2"])
            ck(yt.dtype == torch.float32 and tf.passes_e1(m), f"{kind} T={T}: eager apply TF vs orig {m}")
            ref_apply.append(yx.clone())

        def seq(fs):
            def run():
                for f in fs:
                    f()
            return run

        r = {"T": T, "kind": kind, "distinct": distinct, "empty": empty, "spread": 0.0}
        ratio_log.clear()
        r["bare_eager_xl"], r["bare_eager_tf"], s1, s2 = ab(seq(bare_x), seq(bare_t), len(sets))
        r["spread"] = max(r["spread"], s1, s2)
        r["apply_eager_xl"], r["apply_eager_tf"], s1, s2 = ab(seq(app_x), seq(app_t), len(sets))
        r["spread"] = max(r["spread"], s1, s2)
        grs = {}
        for name, fs in (("bare_graph_xl", bare_x), ("bare_graph_tf", bare_t), ("apply_graph_xl", app_x),
                         ("apply_graph_tf", app_t)):
            c0 = tf.COUNTERS["tf_calls"]
            grs[name] = graph_of(fs)
            took_tf = tf.COUNTERS["tf_calls"] > c0
            ck(took_tf == name.endswith("_tf"), f"{name}: TF path taken={took_tf}")
        for k in ("bare_graph", "apply_graph"):
            r[k + "_xl"], r[k + "_tf"], s1, s2 = ab(grs[k + "_xl"].replay, grs[k + "_tf"].replay, len(sets))
            r["spread"] = max(r["spread"], s1, s2)
        # the timed TF graphs, replayed once more and read back (12 calls, 3 layers, one shared scratch)
        for s_ in sets:
            s_[4][1].zero_()                                    # bare calls accumulate into their args' outputs
        grs["bare_graph_tf"].replay()
        torch.cuda.synchronize()
        for s_, o_x in zip(sets, ref_bare):
            m = tf.compare(s_[4][1], o_x)
            worst_c["bare_graph"] = max(worst_c["bare_graph"], m["rel_l2"])
            ck(tf.passes_e1(m), f"{kind} T={T}: bare TF graph replay out of tolerance {m}")
        grs["apply_graph_tf"].replay()
        torch.cuda.synchronize()
        for y, y_ref in zip(graph_outs[id(grs["apply_graph_tf"])], ref_apply):
            m = tf.compare(y, y_ref)
            worst_c["apply_graph"] = max(worst_c["apply_graph"], m["rel_l2"])
            ck(tf.passes_e1(m), f"{kind} T={T}: apply TF graph replay vs eager orig apply {m}")
        r["check"] = worst_c
        for gr_ in grs.values():
            graph_outs.pop(id(gr_), None)
        del grs

        # TF per-stage breakdown: graph replay of each stage alone over the 12 sets; every set has its own
        # route tables (its own experts), so the grouped stages read cold weights as in the full call
        ext = tf.load_ext()
        stages = {}
        plans = [(s_[4], tf.plan(s_[4])) for s_ in sets]
        for a, p in plans:
            tf._launch_kernels(p, a)                          # valid activations in the shared scratch
        sc = plans[0][1].scratch
        tabs = []
        for a, p in plans:
            S = tf.s_cap(p.P, p.n)
            t = {"pe": torch.empty(p.P, dtype=torch.int32, device=dev),
                 "se": torch.empty(S, dtype=torch.int32, device=dev), "s0": torch.empty(S, dtype=torch.int32, device=dev),
                 "sr": torch.empty(S, dtype=torch.int32, device=dev), "ns": torch.empty(1, dtype=torch.int32, device=dev),
                 "S": S}
            ext.route_prep(a[2], p.P, p.R, t["pe"], t["se"], t["s0"], t["sr"], t["ns"])
            tabs.append(t)

        def dsk(P):                                           # down K splits of the shipped variant
            return int(ext.split_counts(P, K, N)[1])

        def gsk(P):                                           # gate/up K splits of the shipped variant
            return int(ext.split_counts(P, K, N)[0])

        def st(stage):
            fs = []
            for (a, p), t in zip(plans, tabs):
                if stage == "route_prep":
                    fs.append(lambda a=a, p=p, t=t: ext.route_prep(a[2], p.P, p.R, t["pe"], t["se"], t["s0"], t["sr"],
                                                                   t["ns"]))
                elif stage == "rot_in":
                    fs.append(lambda a=a, t=t: ext.rot_in(a[0], a[3], t["pe"], a[14], a[17], sc.xg, sc.xu))
                elif stage == "grouped_gu":
                    fs.append(lambda a=a, p=p, t=t: ext.grouped(sc.xg, sc.xu, a[13], a[16], t["se"], t["s0"], t["sr"],
                                                                t["ns"], sc.z, 2, K, N, p.P, gsk(p.P), *tf.GATEUP_CFG[:2], t["S"]))
                elif stage == "gateup_epi":
                    fs.append(lambda a=a, p=p, t=t: ext.gateup_epilogue(sc.z, t["pe"], a[15], a[18], a[20], sc.xd,
                                                                        p.P, N, gsk(p.P), LIMIT))
                elif stage == "grouped_down":
                    fs.append(lambda a=a, p=p, t=t: ext.grouped(sc.xd, sc.xd, a[19], a[19], t["se"], t["s0"], t["sr"],
                                                                t["ns"], sc.z, 1, N, K, p.P, dsk(p.P), *tf.DOWN_CFG[:2], t["S"]))
                elif stage == "down_epi":
                    fs.append(lambda a=a, p=p, t=t: ext.down_epilogue(sc.z, t["pe"], a[3], a[4], a[21], a[1], dsk(p.P)))
            return fs

        for stage in ("route_prep", "rot_in", "grouped_gu", "gateup_epi", "grouped_down", "down_epi"):
            gr = graph_of(st(stage))
            stages[stage] = timed(gr.replay, len(sets))
            del gr
        r["stages"] = stages
        r["ratios"] = list(ratio_log)          # per-round-median speedups: bare eager, apply eager, bare graph, apply graph
        # plan() + dispatcher host overhead (Python), pre-built args
        a = sets[0][4]
        t0 = time.perf_counter()
        for _ in range(2000):
            tf.plan(a)
        r["plan_us"] = (time.perf_counter() - t0) / 2000 * 1e6
        rows.append(r)
        gbps = distinct * per_expert / (r["bare_graph_tf"] * 1e-6) / 1e9
        gbps_x = distinct * per_expert / (r["bare_graph_xl"] * 1e-6) / 1e9
        stage_bytes = {"grouped_gu": 2 / 3, "grouped_down": 1 / 3}
        # the progress / correctness lines are labelled with the routing kind of this configuration (a loop
        # variable of the same name once overwrote it with the last stage name)
        ck(kind == r["kind"] and kind in ("rand", "corr40"), f"routing-kind label {kind!r} != {r['kind']!r}")
        print(f"{kind:6s} T={T:3d} distinct {distinct:5.1f} empty-seg {empty:5.1f}: bare graph XL {r['bare_graph_xl']:7.1f} "
              f"TF {r['bare_graph_tf']:7.1f} us ({gbps_x:5.1f} / {gbps:5.1f} GB/s = {gbps / CEIL * 100:4.1f}% of {CEIL:.0f}) "
              f"round spread {r['spread'] * 100:.0f}% | stages " +
              " ".join(f"{k} {v:.1f}" + (f" ({distinct * per_expert * stage_bytes[k] / (v * 1e-6) / 1e9:5.1f} GB/s "
                                          f"{distinct * per_expert * stage_bytes[k] / (v * 1e-6) / 1e9 / CEIL * 100:4.1f}%)"
                                          if k in stage_bytes else "")
                       for k, v in stages.items()) + f" | plan() {r['plan_us']:.1f} us", flush=True)
        print(f"{kind:6s} T={T:3d} correctness vs orig (worst rel_l2 over 12 sets, tol 2e-3): bare eager {worst_c['bare_eager']:.2e}, "
              f"bare graph replay {worst_c['bare_graph']:.2e}, apply eager {worst_c['apply_eager']:.2e}, "
              f"apply graph replay {worst_c['apply_graph']:.2e}", flush=True)

    print()
    print(f"E.5 decode A/B (median us per call over {ROUNDS} alternating rounds; speedup = XL / TF; n=288 K=4096 N=1024"
          f" topk=8, cold weights; GB/s = distinct experts' trellis bytes / bare-graph TF time, % of the {CEIL:.0f} GB/s"
          f" ceiling; floor = those bytes at {CEIL:.0f} GB/s)")
    hdr = (f"{'kind':6s} {'T':>3} {'dist':>5} {'empty':>5} | {'bare eager XL':>13} {'TF':>7} {'x':>5} | {'bare graph XL':>13} "
           f"{'TF':>7} {'x':>5} | {'apply eager XL':>14} {'TF':>7} {'x':>5} | {'apply graph XL':>14} {'TF':>7} {'x':>5} | "
           f"{'TF GB/s':>7} {'%ceil':>5} {'floor':>7} {'gu GB/s':>7} {'%':>4} {'dn GB/s':>7} {'%':>4}")
    print(hdr)
    window = []
    for r in rows:
        sp = {k: r[f"{k}_xl"] / r[f"{k}_tf"] for k in ("bare_eager", "bare_graph", "apply_eager", "apply_graph")}
        byt = r["distinct"] * per_expert
        gbps = byt / (r["bare_graph_tf"] * 1e-6) / 1e9
        gu = byt * 2 / 3 / (r["stages"]["grouped_gu"] * 1e-6) / 1e9
        dn = byt / 3 / (r["stages"]["grouped_down"] * 1e-6) / 1e9
        print(f"{r['kind']:6s} {r['T']:>3} {r['distinct']:>5.1f} {r['empty']:>5.1f} | {r['bare_eager_xl']:>13.1f} "
              f"{r['bare_eager_tf']:>7.1f} {sp['bare_eager']:>5.2f} | "
              f"{r['bare_graph_xl']:>13.1f} {r['bare_graph_tf']:>7.1f} {sp['bare_graph']:>5.2f} | "
              f"{r['apply_eager_xl']:>14.1f} {r['apply_eager_tf']:>7.1f} {sp['apply_eager']:>5.2f} | "
              f"{r['apply_graph_xl']:>14.1f} {r['apply_graph_tf']:>7.1f} {sp['apply_graph']:>5.2f} | {gbps:>7.1f} "
              f"{gbps / CEIL * 100:>5.1f} {byt / (CEIL * 1e9) * 1e6:>7.1f} {gu:>7.1f} {gu / CEIL * 100:>4.0f} "
              f"{dn:>7.1f} {dn / CEIL * 100:>4.0f}")
        if r["kind"] == "rand" and sp["apply_graph"] >= 1.05:
            window.append(r["T"])
    print("per-stage graph replay (us) and GB/s (% of the ceiling) of the TF call:")
    for r in rows:
        byt = r["distinct"] * per_expert
        st = r["stages"]
        print(f"  {r['kind']:6s} T={r['T']:3d} total {r['bare_graph_tf']:8.1f} | " + " ".join(
            f"{k} {v:.1f}" for k, v in st.items()) + f" | sum {sum(st.values()):.1f} | gu "
            f"{byt * 2 / 3 / (st['grouped_gu'] * 1e-6) / 1e9:5.1f} GB/s ({byt * 2 / 3 / (st['grouped_gu'] * 1e-6) / 1e9 / CEIL * 100:4.1f}%) "
            f"down {byt / 3 / (st['grouped_down'] * 1e-6) / 1e9:5.1f} GB/s ({byt / 3 / (st['grouped_down'] * 1e-6) / 1e9 / CEIL * 100:4.1f}%)"
            f" | per-round-median speedups (bare eager, apply eager, bare graph, apply graph) "
            + " ".join(f"{v:.3f}" for v in r["ratios"]))
    rows_r = [r for r in rows if r["kind"] == "rand"]
    print(f"T (random routing) with graph-replay production-apply speedup >= 1.05: {window}")
    if window:
        print(f"suggested TF_EXL3_TOKENS={min(window)}:{max(window)}"
              + ("" if window == [r['T'] for r in rows_r if min(window) <= r['T'] <= max(window)] else " (window not contiguous)"))
    else:
        print("no T qualifies: TF must stay disabled (correct-but-slower is not deployed)")
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
