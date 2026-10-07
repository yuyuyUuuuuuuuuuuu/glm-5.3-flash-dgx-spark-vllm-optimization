"""Speed of GLM53_MOE_E4M3 vs production's prefill routed MoE on nodeC, real layer-10 experts (TP=2 rank 0 shard).

Production = apply_exl3_fused_moe as deployed for prefill (prefill cap GLM53_PREFILL_FUSED_CAP=1 installed: experts with
1 row on the thin kernel, the rest on E3 = tables + fm_gather + fm_gateup + fm_down), env of tests/prefill_cap_common.
E4M3 = glm53_moe_e4m3.run (routing tables + gather + gate/up + actq + down). Both produce the fp32 [T, 4096] sum that
apply_exl3_experts then casts to x.dtype (not timed in either).

(1) per call, alternating order, BENCH_ROUNDS rounds x 2 calls, median; (2) per phase with CUDA events (median of
BENCH_REPS) for both paths. T in BENCH_T (default 13824, 4289, 1791), routing BENCH_KINDS (real, zipf, collapsed;
tests/prefill_cap_common.routing, calibrated to production's per-layer kernel times).
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moee4m3/bench_moe_e4m3.py
"""
from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def ev(fn, reps):
    for _ in range(2):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def phases(prod, M, L, x, ids, w, reps):
    E = M._ext()
    ext3 = prod.load_fat_moe_ext()
    p = L._exl3_ptrs
    dev = x.device
    T = x.shape[0]
    xh = x.half()
    n_exp = len(L._exl3_inners)
    # E4M3 phases
    t = M.plan(prod, ids, w, n_exp, None)
    a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
    out = torch.zeros(T, 4096, dtype=torch.float32, device=dev)
    f_plan = lambda: M.plan(prod, ids, w, n_exp, None)  # noqa: E731
    f_g = lambda: E.gather2(xh, t["local"], t["pos"], p["gate_suh"], a8, asc, out, t["topk"], n_exp)  # noqa: E731
    f_gu = lambda: E.gateup(a8, asc, p["gate_trellis"], p["up_trellis"], p["gate_svh"], p["up_svh"], a16,  # noqa: E731
                            t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], C.LIMIT, 0, 0, 1, 0)
    f_aq = lambda: E.actq(a16, t["row_expert"], p["down_suh"], a8d, dsc, t["seg_row0"], t["seg_rows"], t["num_segs"], 0, 1)  # noqa: E731
    f_dn = lambda: E.down(a8d, dsc, p["down_trellis"], p["down_svh"], out, t["row_token"], t["row_weight"],  # noqa: E731
                          t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], 0, 0, 1, 0)
    me = [ev(f, reps) for f in (f_plan, f_g, f_gu, f_aq, f_dn)]
    # E3 phases (production's own functions, cap 1 as the prefill cap runs it)
    local = prod.map_topk_to_local(ids, n_exp, None)
    topk = ids.shape[1]
    flat_token = torch.arange(T, device=dev, dtype=torch.long).repeat_interleave(topk)
    flat_weight = w.reshape(-1).to(torch.float16)
    order = local.argsort()
    ts, ws = flat_token[order], flat_weight[order]
    cnt = torch.zeros(n_exp + 1, dtype=torch.long, device=dev)
    cnt.scatter_add_(0, local.long(), torch.ones_like(local, dtype=torch.long))
    counts = cnt[:n_exp]
    rows_cap = ts.numel()
    sc = prod._grouped_scratch(dev, rows_cap, 4096, 1024)
    h13, h2 = sc["h13"][:rows_cap], sc["h2"][:rows_cap]
    tile = int(ext3.exl3_fat_moe_tile_rows_gateup())
    t3 = prod.build_grouped_fat_tables(counts, 1, ts, ws, rows_cap, tile)
    f_tab = lambda: prod.build_grouped_fat_tables(counts, 1, ts, ws, rows_cap, tile)  # noqa: E731
    f_g3 = lambda: ext3.exl3_fat_moe_gather(xh, t3["row_token"], t3["row_expert"], p["gate_suh"], h13, t3["num_rows"])  # noqa: E731
    f_gu3 = lambda: ext3.exl3_fat_moe_gateup(h13, p["gate_trellis"], p["up_trellis"], p["gate_svh"], p["up_svh"],  # noqa: E731
                                             p["down_suh"], h2, t3["seg_expert"], t3["seg_row0"], t3["seg_rows"],
                                             t3["num_segs"], C.LIMIT)
    f_d3 = lambda: ext3.exl3_fat_moe_down(h2, p["down_trellis"], p["down_svh"], out, t3["row_token"], t3["row_weight"],  # noqa: E731
                                          t3["seg_expert"], t3["seg_row0"], t3["seg_rows"], t3["num_segs"])
    e3 = [ev(f, reps) for f in (f_tab, f_g3, f_gu3, f_d3)]
    return me, e3


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,4289,1791").split(",")]
    kinds = os.environ.get("BENCH_KINDS", "real,zipf,collapsed").split(",")
    reps = int(os.environ.get("BENCH_REPS", "5"))
    rounds = int(os.environ.get("BENCH_ROUNDS", "5"))
    do_phases = os.environ.get("BENCH_PHASES", "1") == "1"
    summary = []
    summary_apply = []
    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            seed = T + (0 if kind == "real" else 50000 if kind == "collapsed" else 90000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()
            hist = C.histogram(ids)
            emap = prod.pin_exl3_expert_map(L, dev)
            f_prod = lambda: prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, C.LIMIT)  # noqa: E731
            f_me = lambda: M.run(prod, x, ids, w, L, C.LIMIT)  # noqa: E731
            times = {"prod": [], "e4m3": []}
            for rnd in range(rounds):
                order = ("prod", "e4m3") if rnd % 2 == 0 else ("e4m3", "prod")
                for name in order:
                    f = f_prod if name == "prod" else f_me
                    f()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    torch.cuda.synchronize()
                    s.record()
                    for _ in range(2):
                        f()
                    e.record()
                    torch.cuda.synchronize()
                    times[name].append(s.elapsed_time(e) / 2)
            mp, mm = statistics.median(times["prod"]), statistics.median(times["e4m3"])
            sp = lambda v: (max(v) - min(v)) / statistics.median(v) * 100  # noqa: E731
            c = hist["counts"]
            rows = int(c.sum())
            flop = 2.0 * rows * (4096 * 2048 + 1024 * 4096)
            for sc_s in [v for v in os.environ.get("BENCH_SCHED", "").split(",") if v]:
                if sc_s.startswith("f"):     # f<lag>[v<variant>]: fused kernel with that lag (debug variant)
                    lg, _, vv = sc_s[1:].partition("v")
                    sd = {"mode": "fused", "lag": int(lg), "variant": int(vv or 0)}
                else:
                    nc, gg, dg = (int(u) for u in sc_s.split(":"))
                    sd = {"mode": "streams", "nchunks": nc, "gu_grid": gg, "dn_grid": dg}
                f_s = lambda: M.run(prod, x, ids, w, L, C.LIMIT, sched=sd)  # noqa: E731
                print(f"  [sched {kind} T={T}] {sd}: {ev(f_s, reps):.2f} ms", flush=True)
            print(f"[call {kind} T={T}] {C.fmt_hist(hist)} | production {mp:.2f} ms (spread {sp(times['prod']):.1f}%) | "
                  f"e4m3 {mm:.2f} ms (spread {sp(times['e4m3']):.1f}%, {flop / mm / 1e9:.1f} TFLOPS) | delta "
                  f"{mm - mp:+.2f} ms (x{mp / mm:.3f})", flush=True)
            summary.append((kind, T, mp, mm))
            if os.environ.get("BENCH_APPLY", "1") == "1":
                # through production's apply_exl3_experts (incl. its x.dtype cast): as deployed (prefill cap 1) vs the
                # same with GLM53_MOE_E4M3 installed on top
                base = prod.apply_exl3_experts
                f_a0 = lambda: base(x, ids, w, L, limit=C.LIMIT)  # noqa: E731
                rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1"})
                assert rep["installed"], rep
                hooked = prod.apply_exl3_experts
                f_a1 = lambda: hooked(x, ids, w, L, limit=C.LIMIT)  # noqa: E731
                f_a1()                                       # self-test (once per layer) outside the timing
                assert getattr(L, "_glm53_moe_e4m3_ok", False), "self-test failed"
                ta = {"prod": [], "e4m3": []}
                for rnd in range(rounds):
                    for name in (("prod", "e4m3") if rnd % 2 == 0 else ("e4m3", "prod")):
                        ta[name].append(ev(f_a0 if name == "prod" else f_a1, 2))
                M.uninstall(prod)
                a0, a1 = statistics.median(ta["prod"]), statistics.median(ta["e4m3"])
                print(f"  [apply {kind} T={T}] apply_exl3_experts: production (prefill cap 1) {a0:.2f} ms | with "
                      f"GLM53_MOE_E4M3=1 {a1:.2f} ms | delta {a1 - a0:+.2f} ms (x{a0 / a1:.3f})", flush=True)
                summary_apply.append((kind, T, a0, a1))
            if do_phases:
                me, e3 = phases(prod, M, L, x, ids, w, reps)
                gu_flop, dn_flop = 2.0 * rows * 4096 * 2048, 2.0 * rows * 1024 * 4096
                print(f"  [phases {kind} T={T}] e4m3: plan {me[0]:.2f} gather {me[1]:.2f} gate/up {me[2]:.2f} "
                      f"({gu_flop / me[2] / 1e9:.0f} TFLOPS) actq {me[3]:.2f} down {me[4]:.2f} ({dn_flop / me[4] / 1e9:.0f} "
                      f"TFLOPS) = {sum(me):.2f} | E3 (cap 1): tables {e3[0]:.2f} gather {e3[1]:.2f} gate/up {e3[2]:.2f} "
                      f"down {e3[3]:.2f} = {sum(e3):.2f} ms", flush=True)
            del x
            torch.cuda.empty_cache()
    PC.uninstall(prod)
    print("SUMMARY kind T prod_ms e4m3_ms delta_ms speedup   (apply_exl3_fused_moe vs glm53_moe_e4m3.run)")
    for kind, T, a, b in summary:
        print(f"SUMMARY {kind} {T} {a:.3f} {b:.3f} {b - a:+.3f} {a / b:.3f}")
    print("SUMMARY_APPLY kind T prod_ms e4m3_ms delta_ms speedup   (apply_exl3_experts, unhooked vs hooked)")
    for kind, T, a, b in summary_apply:
        print(f"SUMMARY_APPLY {kind} {T} {a:.3f} {b:.3f} {b - a:+.3f} {a / b:.3f}")
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
