"""K2 (docs/OPTIMIZATION.md): production's decode apply_exl3_fused_moe served from the router ids.

With integrate.install(), production's apply_exl3_fused_moe is replaced by a hook that, while exl3_moe is the TF
dispatcher, runs tf.apply_fused (route_ids -> rot_in(bf16 x) -> grouped -> epilogues) instead of production's
routing prelude + exl3_moe, and delegates everything else to production's own apply.

  (a) route_ids == production's prelude + route_prep: nseg / segment tables / pair_expert identical; inside every
      expert group the (token, fp16 weight) multiset identical to production's token_sorted / weight_sorted
      (production's argsort order inside a group is unspecified); fp32 / bf16 / fp16 weights; n = 288 and n = 32
      through an EP map; sentinel -1, ids >= n, duplicate ids in a row, a hot expert (count = T, = R at T = 128), an expert
      over the cap R (count 2T > R), all-sentinel rows, an empty expert map
  (b) rot_in with the bf16 hidden state == rot_in on production's x2d.contiguous().half(): bit-identical
  (c) E.1: hooked apply vs production apply with the original exl3_moe, T = 1..16, 24, 32, 64, 128, incl. the
      routing edge cases of (a) and fp32 / bf16 router weights
  (d) bitwise: hooked apply (K2) == the TF exl3_moe path (TF_EXL3_APPLY off) on one-route-per-token routing
  (e) side effects of production's decode branch (_exl3_last_fat_fallback / _reason) are set
  (f) delegation: T = R + 1, non-int64 ids, weights shaped differently, fp32 x, a non-contiguous ids view,
      an expert map on the CPU (production must raise its own RuntimeError), TF_EXL3_TOKENS outside, TF disabled,
      exl3_moe not the dispatcher -> production's apply runs (same result as the unhooked apply)
  (g) E.4: the hooked apply captured in a CUDA graph (K2 chosen at capture), 100 replays with fresh routing vs the
      eager original apply; no host sync (sync debug "error" + positive control); allocations per call: 1 (out,
      as production allocates it) vs production's own count
  (h) uninstall restores exl3_moe, the build hook and apply_exl3_fused_moe; a later third-party patch of the
      apply is kept
"""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    ext = tf.load_ext()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep.get("apply_hook") is True, f"apply hook not installed: {rep}")
    orig, disp = rep["orig"], xl.exl3_moe
    orig_apply = prod.apply_exl3_fused_moe._tf_exl3_orig
    tf.CFG.strict = True
    W = H.Weights(NEXP, K, N, dev, seed=41)
    layer = H.make_layer(prod, W)
    ck(tf.REG.get((0, layer._exl3_ptrs["gate_trellis"].data_ptr())) is not None, "layer not registered")
    k2st = tf.REG[(0, layer._exl3_ptrs["gate_trellis"].data_ptr())].selftest.get("apply", {})
    print(f"K2 load-time self-test: {k2st}")
    ck(k2st.get("ok") is True and k2st.get("cases", 0) >= 2, f"K2 self-test {k2st}")
    W32 = H.Weights(32, K, N, dev, seed=42)
    layer32 = H.make_layer(prod, W32)
    R = int(layer._exl3_fused_temps[0].shape[1])
    g = torch.Generator().manual_seed(2026)

    def ref_apply(x, ids, w, L, emap):
        """production's own apply with the original exl3_moe"""
        xl.exl3_moe = orig
        try:
            return orig_apply(x, ids, w, L, L._exl3_inners, emap, LIMIT)
        finally:
            xl.exl3_moe = disp

    def hooked(x, ids, w, L, emap):
        return prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, LIMIT)

    def ids_case(case, T, n, gen):
        if case == "random":
            return H.random_ids(T, n, TOPK, gen, dev)
        ids = H.random_ids(T, n, TOPK, gen, "cpu")
        if case == "sentinel":                       # -1 and >= n ids
            ids[0, 0] = -1
            ids[T - 1, TOPK - 1] = n + 5
        elif case == "duplicates":                   # the same expert twice in a row
            ids[:, 1] = ids[:, 0]
        elif case == "hot":                          # one expert in every row (count = T); T = R -> count = R
            ids[:, 0] = 7
        elif case == "all_sentinel_row":
            ids[T // 2, :] = -1
        elif case == "overcap":                      # expert 7 twice in every row: count 2T > R for T > R / 2
            ids[:, 0] = 7
            ids[:, 1] = 7
        elif case == "corr":
            return H.correlated_ids(T, n, TOPK, 40, gen, dev)
        return ids.to(dev)

    # ---- (a) route_ids vs production prelude + route_prep -------------------------------------------------------
    i32 = dict(dtype=torch.int32, device=dev)
    mism, cases = 0, 0
    for it in range(400):
        n = (NEXP, 32)[it % 2]
        use_map = n == 32 and it % 4 == 1
        T = int(torch.randint(1, 129, (1,), generator=g))
        case = ("random", "sentinel", "duplicates", "hot", "all_sentinel_row", "overcap")[it % 6]
        wdt = (torch.float32, torch.bfloat16, torch.float16)[it % 3]
        emap = None
        if use_map:
            emap = torch.full((64,), -1, dtype=torch.long)
            emap[torch.randperm(64, generator=g)[:32]] = torch.arange(32)
            if it % 40 == 1:
                emap = torch.empty(0, dtype=torch.long)       # empty map: every id non-local
            emap = emap.to(dev)
        ids = ids_case(case, T, 64 if use_map else n, g)
        w = (torch.rand(T, TOPK, generator=g) * 2.5).to(wdt).to(dev)
        P = T * TOPK
        # production: map_topk_to_local + argsort + gathers + scatter_add (apply_exl3_fused_moe :1616-1631)
        local = prod.map_topk_to_local(ids, n, emap)
        order = local.argsort()
        ts_p = torch.arange(T, device=dev, dtype=torch.long).repeat_interleave(TOPK)[order]
        ws_p = w.reshape(-1).to(torch.float16)[order]
        ec = torch.zeros(n + 1, dtype=torch.long, device=dev)
        ec.scatter_add_(0, local.long(), torch.ones(local.shape, dtype=torch.long, device=dev))
        S = tf.s_cap(P, n)
        pe_p, se_p, s0_p, sr_p, ns_p = (torch.full((P,), 7, **i32), torch.full((S,), -9, **i32),
                                        torch.full((S,), -9, **i32), torch.full((S,), -9, **i32), torch.full((1,), -9, **i32))
        ext.route_prep(ec, P, R, pe_p, se_p, s0_p, sr_p, ns_p)
        pe_k, se_k, s0_k, sr_k, ns_k = (torch.full((P,), 7, **i32), torch.full((S,), -9, **i32),
                                        torch.full((S,), -9, **i32), torch.full((S,), -9, **i32), torch.full((1,), -9, **i32))
        ts_k = torch.full((P,), -9, dtype=torch.long, device=dev)
        ws_k = torch.full((P,), float("nan"), dtype=torch.float16, device=dev)
        ext.route_ids(ids, w, emap, n, R, pe_k, se_k, s0_k, sr_k, ns_k, ts_k, ws_k)
        torch.cuda.synchronize()
        nk = int(ns_k)
        same = (nk == int(ns_p) and torch.equal(pe_k, pe_p) and torch.equal(se_k[:nk], se_p[:nk])
                and torch.equal(s0_k[:nk], s0_p[:nk]) and torch.equal(sr_k[:nk], sr_p[:nk]))
        # (token, weight) multiset per group: groups are the runs of the sorted local id
        loc_sorted = local[order]
        key_p = loc_sorted * (1 << 40) + ts_p * (1 << 16) + ws_p.view(torch.int16).long().bitwise_and(0xFFFF)
        key_k = loc_sorted * (1 << 40) + ts_k * (1 << 16) + ws_k.view(torch.int16).long().bitwise_and(0xFFFF)
        same = same and torch.equal(torch.sort(key_p).values, torch.sort(key_k).values)
        cases += 1
        if not same:
            mism += 1
            if mism <= 3:
                print(f"  route_ids mismatch it={it} n={n} map={use_map} T={T} case={case} w={wdt}: nseg {nk}/{int(ns_p)}")
    print(f"(a) route_ids vs production prelude + route_prep: {cases - mism}/{cases} identical (n in {{288, 32+EP map}},"
          f" T 1..128, fp32/bf16/fp16 weights, sentinel/duplicates/hot/all-sentinel rows/empty map)")
    ck(mism == 0, f"(a) route_ids mismatches: {mism}")

    # ---- (b) rot_in bf16 x == rot_in on x.half() ------------------------------------------------------------------
    B, P = 9, 40
    xb = (torch.randn(B, K, generator=g) * 4).to(torch.bfloat16).to(dev)
    tsb = torch.randint(0, B, (P,), generator=g).to(dev)
    peb = torch.randint(0, NEXP, (P,), generator=g).int().to(dev)
    ptr = layer._exl3_ptrs
    outs = []
    for xin in (xb, xb.half()):
        xg = torch.full((P, K), float("nan"), dtype=torch.float16, device=dev)
        xu = torch.full_like(xg, float("nan"))
        ext.rot_in(xin, tsb, peb, ptr["gate_suh"], ptr["up_suh"], xg, xu)
        outs.append((xg, xu))
    torch.cuda.synchronize()
    okb = torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    print(f"(b) rot_in(bf16 x) bit-identical to rot_in(x.half()): {okb}")
    ck(okb, "(b) bf16 rot_in differs")

    # ---- (c) E.1 hooked apply vs production apply (orig exl3_moe); (e) side effects ---------------------------
    worst, n_c, fails = 0.0, 0, 0
    for T in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 24, 32, 64, 128):
        for case in ("random", "sentinel", "duplicates", "hot", "all_sentinel_row", "overcap", "corr"):
            if T < 2 and case in ("all_sentinel_row",):
                continue
            for L, emap, nn in ((layer, None, NEXP), (layer32, "map", 64)):
                if L is layer32 and case not in ("random", "sentinel"):
                    continue
                em = None
                if emap == "map":
                    em = torch.full((64,), -1, dtype=torch.long)
                    em[torch.randperm(64, generator=g)[:32]] = torch.arange(32)
                    em = em.to(dev)
                    L.expert_map = em
                x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                ids = ids_case(case, T, nn, g)
                w = H.random_weights(T, TOPK, g, dev)
                if T % 2:
                    w = w.to(torch.bfloat16)
                L._exl3_last_fat_fallback = "junk"
                L._exl3_last_fat_reason = "junk"
                c0 = tf.COUNTERS["tf_apply_calls"]
                y = hooked(x, ids, w, L, em)
                took = tf.COUNTERS["tf_apply_calls"] == c0 + 1
                side = (L._exl3_last_fat_fallback, L._exl3_last_fat_reason) == ("none", "no_fat_experts")
                y_ref = ref_apply(x, ids, w, L, em)
                torch.cuda.synchronize()
                m = tf.compare(y, y_ref)
                n_c += 1
                worst = max(worst, m["rel_l2"])
                ok = took and side and y.dtype == torch.float32 and y.shape == (T, K) and tf.passes_e1(m)
                if not ok:
                    fails += 1
                    print(f"  (c) T={T} case={case} n={nn}: took K2={took} side effects={side} {m}")
                L.expert_map = None
    print(f"(c) hooked apply (K2) vs production apply + exl3_moe: {n_c - fails}/{n_c} within E.1 (K2 taken, side "
          f"effects set), worst rel_l2 {worst:.2e}")
    ck(fails == 0, f"(c) {fails} K2 cases failed")

    # ---- (d) bitwise vs the TF exl3_moe path on one route per token ------------------------------------------
    bad = 0
    for T in (1, 5, 8, 16, 64, 128):
        x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
        ids = torch.stack([torch.randperm(NEXP, generator=g)[:1] for _ in range(T)]).to(dev)
        if T > 1:
            ids[1, 0] = -1
        w = H.random_weights(T, 1, g, dev)
        y_k2 = hooked(x, ids, w, layer, None)
        tf.CFG.apply = False
        c0 = tf.COUNTERS["tf_apply_calls"]
        y_tf = hooked(x, ids, w, layer, None)                     # K2 off: production apply -> TF dispatcher
        ck(tf.COUNTERS["tf_apply_calls"] == c0, "(d) K2 ran with TF_EXL3_APPLY off")
        tf.CFG.apply = True
        torch.cuda.synchronize()
        if not torch.equal(y_k2, y_tf):
            bad += 1
            print(f"  (d) T={T}: max |diff| {float((y_k2 - y_tf).abs().max()):.3e}")
    print(f"(d) K2 bit-identical to the TF exl3_moe path on one-route-per-token routing: {6 - bad}/6")
    ck(bad == 0, "(d) K2 not bit-identical to the exl3_moe path")

    # ---- (f) delegation ---------------------------------------------------------------------------------------
    def delegated(x, ids, w, L=layer, emap=None, expect_raise=None):
        c0, d0 = tf.COUNTERS["tf_apply_calls"], tf.COUNTERS["apply_delegated"]
        err_h = err_r = None
        try:
            y = hooked(x, ids, w, L, emap)
        except Exception as e:  # noqa: BLE001
            err_h, y = e, None
        try:
            yr = ref_apply(x, ids, w, L, emap)
        except Exception as e:  # noqa: BLE001
            err_r, yr = e, None
        torch.cuda.synchronize()
        dg = tf.COUNTERS["tf_apply_calls"] == c0 and tf.COUNTERS["apply_delegated"] == d0 + 1
        if expect_raise:
            return dg and type(err_h) is type(err_r) and err_h is not None and str(err_h) == str(err_r)
        if err_h is not None or err_r is not None:
            return False
        return dg and tf.passes_e1(tf.compare(y, yr))

    T = 8
    x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
    ids = H.random_ids(T, NEXP, TOPK, g, dev)
    w = H.random_weights(T, TOPK, g, dev)
    res = {}
    xR = torch.randn(R + 1, K, generator=g).to(torch.bfloat16).to(dev)
    res["T=R+1"] = delegated(xR, H.random_ids(R + 1, NEXP, TOPK, g, dev), H.random_weights(R + 1, TOPK, g, dev))
    res["int32 ids"] = delegated(x, ids.int(), w)
    res["weights [T*topk]"] = delegated(x, ids, w.reshape(-1))
    res["fp32 x"] = delegated(x.float(), ids, w)
    idsT = torch.stack([ids, ids], dim=2)[:, :, 0]                  # a non-contiguous view with the same values
    res["non-contiguous ids"] = (not idsT.is_contiguous()) and delegated(x, idsT, w)
    emc = torch.arange(NEXP, dtype=torch.long)                     # CPU expert map: production raises
    res["cpu expert map (production raises)"] = delegated(x, ids, w, emap=emc, expect_raise=True)
    tf.CFG.tokens_lo, tf.CFG.tokens_hi = 16, 1 << 30
    res["TF_EXL3_TOKENS=16: excludes T=8"] = delegated(x, ids, w)
    tf.CFG.tokens_lo, tf.CFG.tokens_hi = 1, 1 << 30
    tf.set_enabled(False)
    res["TF disabled"] = delegated(x, ids, w)
    tf.set_enabled(True)
    xl.exl3_moe = orig                                              # exl3_moe is not the dispatcher
    c0 = tf.COUNTERS["tf_apply_calls"]
    y1 = prod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, None, LIMIT)
    y2 = orig_apply(x, ids, w, layer, layer._exl3_inners, None, LIMIT)
    torch.cuda.synchronize()
    xl.exl3_moe = disp
    res["exl3_moe not the dispatcher"] = tf.COUNTERS["tf_apply_calls"] == c0 and float((y1 - y2).norm()) <= 1e-6 * float(
        y2.norm())
    print(f"(f) delegation to production's apply: {res}")
    ck(all(res.values()), f"(f) delegation {res}")

    # ---- (g) CUDA graph, host sync, allocations --------------------------------------------------------------
    Ts = (1, 5, 8, 64)
    static = {T: (torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev), H.random_ids(T, NEXP, TOPK, g, dev),
                  H.random_weights(T, TOPK, g, dev)) for T in Ts}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for T in Ts:
            hooked(*static[T], layer, None)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graphs, gouts, chosen = {}, {}, True
    for T in Ts:
        c0 = tf.COUNTERS["tf_apply_calls"]
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gouts[T] = hooked(*static[T], layer, None)
        chosen &= tf.COUNTERS["tf_apply_calls"] == c0 + 1
        graphs[T] = gr
    ck(chosen, "(g) K2 not chosen at capture")
    fails_g, worst_g = 0, 0.0
    for i in range(100):
        T = Ts[i % len(Ts)]
        kind = "rand" if i % 2 == 0 else "corr40"
        new = (torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev), H.routing_ids(kind, T, NEXP, TOPK, g, dev),
               H.random_weights(T, TOPK, g, dev))
        for dst, src in zip(static[T], new):
            dst.copy_(src)
        graphs[T].replay()
        y = gouts[T].clone()
        y_ref = ref_apply(*new, layer, None)
        torch.cuda.synchronize()
        m = tf.compare(y, y_ref)
        worst_g = max(worst_g, m["rel_l2"])
        if not tf.passes_e1(m):
            fails_g += 1
    print(f"(g) K2 captured at T={Ts} (chosen at capture: {chosen}); 100 interleaved replays with fresh routing: "
          f"{100 - fails_g}/100 within E.1 vs the eager original apply, worst rel_l2 {worst_g:.2e}")
    ck(fails_g == 0, f"(g) {fails_g} replays out of tolerance")
    x, ids, w = static[8]
    hooked(x, ids, w, layer, None)
    torch.cuda.synchronize()
    c0 = tf.COUNTERS["tf_apply_calls"]
    torch.cuda.set_sync_debug_mode("error")
    err = None
    try:
        hooked(x, ids, w, layer, None)
    except RuntimeError as e:
        err = e
    finally:
        torch.cuda.set_sync_debug_mode(0)
    control = False
    torch.cuda.set_sync_debug_mode("error")
    try:
        _ = x[0, 0].item()
    except RuntimeError:
        control = True
    finally:
        torch.cuda.set_sync_debug_mode(0)
    ck(err is None and tf.COUNTERS["tf_apply_calls"] == c0 + 1 and control, f"(g) host sync in the K2 path: {err}")
    torch.cuda.synchronize()
    s0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(100):
        hooked(x, ids, w, layer, None)
    torch.cuda.synchronize()
    s1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(100):
        ref_apply(x, ids, w, layer, None)
    torch.cuda.synchronize()
    s2 = torch.cuda.memory_stats()["allocation.all.allocated"]
    print(f"(g) no host sync in K2 (sync debug 'error': raised={err is not None}, positive control raised={control}); "
          f"allocations per call: K2 {(s1 - s0) / 100:.1f} (out), production apply {(s2 - s1) / 100:.1f}")
    ck(s1 - s0 == 100, f"(g) K2 allocates {(s1 - s0) / 100} blocks per call (expected exactly 1: out)")
    del graphs, gouts

    # ---- TF_EXL3_APPLY: the K2 switch (read at install) ------------------------------------------------------
    c_off, c_on, c_bad = (tf.configure({"TF_EXL3_APPLY": "0"}), tf.configure({}), tf.configure({"TF_EXL3_APPLY": "off"}))
    env_ok = c_off.apply is False and c_on.apply is True and c_bad.apply is False
    tf.configure()
    rep_off = None
    integrate.uninstall(prodmod=prod, ext=xl)
    import os
    os.environ["TF_EXL3_APPLY"] = "0"
    try:
        rep_off = integrate.install(prodmod=prod, ext=xl, force=True)
        hooked_off = getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False)
    finally:
        os.environ.pop("TF_EXL3_APPLY", None)
        integrate.uninstall(prodmod=prod, ext=xl)
        tf.configure()
    integrate.install(prodmod=prod, ext=xl, force=True)
    hooked_on = getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False)
    print(f"TF_EXL3_APPLY: parse 0/unset/off -> {c_off.apply}/{c_on.apply}/{c_bad.apply}; install with TF_EXL3_APPLY=0: "
          f"dispatcher installed {bool(rep_off and rep_off['installed'])}, apply hooked {hooked_off}; default: apply hooked "
          f"{hooked_on}")
    ck(env_ok and rep_off["installed"] and not hooked_off and hooked_on, "TF_EXL3_APPLY switch")
    tf.CFG.strict = True

    # ---- (h) uninstall -------------------------------------------------------------------------------------------
    rep_u = integrate.uninstall(prodmod=prod, ext=xl)
    ok_h = (rep_u["restored_apply"] and rep_u["restored_exl3_moe"] and rep_u["restored_build"]
            and prod.apply_exl3_fused_moe is orig_apply and xl.exl3_moe is orig)
    integrate.install(prodmod=prod, ext=xl, force=True)
    third = lambda *a: orig_apply(*a)                              # a later third-party patch of the apply
    prod.apply_exl3_fused_moe = third
    rep_u2 = integrate.uninstall(prodmod=prod, ext=xl)
    ok_h = ok_h and prod.apply_exl3_fused_moe is third and not rep_u2["restored_apply"]
    prod.apply_exl3_fused_moe = orig_apply
    print(f"(h) uninstall restores apply/exl3_moe/build hook: {rep_u}; a later third-party apply patch kept: "
          f"{not rep_u2['restored_apply']}")
    ck(ok_h, "(h) uninstall")
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
