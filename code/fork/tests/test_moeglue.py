"""moeglue (GLM53_DEC_MOEGLUE, glm53_moeglue.py, docs/DEC_MOEGLUE.md): production's decode apply_exl3_experts in
five launches (glue_prep | grouped g/u | gateup_epilogue | grouped d | glue_finish).

  (a) glue_prep == route_ids + rot_in: identical nseg / segment tables; every (token, expert) pair's xg / xu row and
      fp16 weight bit-identical (rows matched through pair_expert / token_sorted); inside each expert group the pairs
      in pair-index order (stable); inv[j] = the pair's row; ids int32 and int64; n = 288 and n = 32 + EP map;
      sentinel -1 / ids >= n / duplicates / hot expert / all-sentinel row / over-cap expert / empty map
  (b) bitwise proof against K2 on production-shaped weights: for every T in 1..128 and routing case, the glue output
      == bf16( sum over slots k = 0..topk-1, in order, in fp32, of K2's output with only slot k routed ) -- K2 with
      one valid route per token writes exactly that pair's fp32 contribution (0 + v), the split-count tables are the
      same (same P), so this is exactly "same per-pair values, summed in slot order"; ids int32 and int64, fp32 / bf16
      router weights, EP map (n = 32). Routings with an expert over the cap R (its pairs are dropped, as production
      drops them; the per-slot calls would not drop them) are checked in (c) only.
  (c) vs production's own apply_exl3_experts (K2, fp32 atomics, .to(bf16)): rel_l2 and max |diff| in bf16 ulps,
      and vs fp64 of K2's per-slot contributions: glue and K2 errors; glue deterministic (two runs bitwise equal)
  (d) the hooked apply_exl3_experts captured in a CUDA graph over 2 layers, 100 replays with fresh routing (ids int32)
      == eager glue bit for bit; no host sync in the eager call (sync debug mode "error")
  (e) delegation to production's apply (exact result of the unhooked function): fused=False, EXL3_FUSED_MOE=0,
      fp32 x, int16 ids, T > R, a layer without a moeglue verdict, TF K2 disabled; install refused on a fingerprint
      mismatch / TF not installed; uninstall restores the function
"""
from __future__ import annotations

import os

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
    import glm53_moeglue as MG
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep0 = MG.install(prodmod=prod, force=True)
    ck(not rep0["installed"] and "dispatcher" in (rep0["reason"] or ""), f"install before TF must refuse: {rep0}")
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep.get("apply_hook") is True, f"K2 apply hook not installed: {rep}")
    rep = MG.install(prodmod=prod, force=True)
    print(f"moeglue install: {rep}")
    ck(rep["installed"], f"moeglue not installed: {rep}")
    orig_experts = prod.apply_exl3_experts._glm53_moeglue_orig
    tf.CFG.strict = True
    MG.CFG.strict = True
    W = H.Weights(NEXP, K, N, dev, seed=41)
    layer = H.make_layer(prod, W)
    W32 = H.Weights(32, K, N, dev, seed=42)
    layer32 = H.make_layer(prod, W32)
    for L, nm in ((layer, "n=288"), (layer32, "n=32")):
        v = MG.VERDICT.get((0, L._exl3_ptrs["gate_trellis"].data_ptr()))
        print(f"load-time moeglue self-test {nm}: {v}")
        ck(v is not None and v[0], f"moeglue self-test {nm}: {v}")
    R = int(layer._exl3_fused_temps[0].shape[1])
    g = torch.Generator().manual_seed(20260928)

    def ids_case(case, T, n, gen):
        if case == "corr":
            return H.correlated_ids(T, n, TOPK, 40, gen, dev)
        ids = H.random_ids(T, n, TOPK, gen, "cpu")
        if case == "sentinel":
            ids[0, 0] = -1
            ids[T - 1, TOPK - 1] = n + 5
        elif case == "duplicates":
            ids[:, 1] = ids[:, 0]
        elif case == "hot":
            ids[:, 0] = 7
        elif case == "all_sentinel_row":
            ids[T // 2, :] = -1
        elif case == "overcap":
            ids[:, 0] = 7
            ids[:, 1] = 7
        return ids.to(dev)

    def ep_map(gen):
        em = torch.full((64,), -1, dtype=torch.long)
        em[torch.randperm(64, generator=gen)[:32]] = torch.arange(32)
        return em.to(dev)

    # ---- (a) glue_prep vs route_ids + rot_in ---------------------------------------------------------------------
    i32 = dict(dtype=torch.int32, device=dev)
    mism, cases = 0, 0
    for it in range(240):
        n = (NEXP, 32)[it % 2]
        L = layer if n == NEXP else layer32
        use_map = n == 32 and it % 4 == 1
        T = int(torch.randint(1, 129, (1,), generator=g))
        case = ("random", "sentinel", "duplicates", "hot", "all_sentinel_row", "overcap", "corr")[it % 7]
        if case == "all_sentinel_row" and T < 2:
            T = 2
        wdt = (torch.float32, torch.bfloat16, torch.float16)[it % 3]
        idt = (torch.int32, torch.int64)[(it // 2) % 2]
        emap = None
        if use_map:
            emap = ep_map(g) if it % 40 != 1 else torch.empty(0, dtype=torch.long, device=dev)
        ids = ids_case(case, T, 64 if use_map else n, g)
        w = (torch.rand(T, TOPK, generator=g) * 2.5).to(wdt).to(dev)
        x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
        P = T * TOPK
        S = tf.s_cap(P, n)
        ptr = L._exl3_ptrs
        tabs = []
        for _ in range(2):
            tabs.append(dict(pe=torch.full((P,), 7, **i32), se=torch.full((S,), -9, **i32),
                             s0=torch.full((S,), -9, **i32), sr=torch.full((S,), -9, **i32),
                             ns=torch.full((1,), -9, **i32), ts=torch.full((P,), -9, dtype=torch.long, device=dev),
                             ws=torch.full((P,), float("nan"), dtype=torch.float16, device=dev),
                             xg=torch.full((P, K), float("nan"), dtype=torch.float16, device=dev),
                             xu=torch.full((P, K), float("nan"), dtype=torch.float16, device=dev),
                             inv=torch.full((P,), -9, **i32)))
        k2, gl = tabs
        ext.route_ids(ids.long(), w, emap, n, R, k2["pe"], k2["se"], k2["s0"], k2["sr"], k2["ns"], k2["ts"], k2["ws"])
        ext.rot_in(x, k2["ts"], k2["pe"], ptr["gate_suh"], ptr["up_suh"], k2["xg"], k2["xu"])
        ext.glue_prep(x, ids.to(idt).contiguous(), w, emap, n, R, ptr["gate_suh"], ptr["up_suh"], gl["xg"], gl["xu"],
                      gl["pe"], gl["se"], gl["s0"], gl["sr"], gl["ns"], gl["ts"], gl["ws"], gl["inv"],
                      ptr["gate_svh"], ptr["up_svh"], ptr["down_suh"], ptr["down_svh"], N, bool(it % 2))
        torch.cuda.synchronize()
        nk = int(gl["ns"])
        same = (nk == int(k2["ns"]) and torch.equal(gl["se"][:nk], k2["se"][:nk])
                and torch.equal(gl["s0"][:nk], k2["s0"][:nk]) and torch.equal(gl["sr"][:nk], k2["sr"][:nk]))
        # the row of every pair: glue puts pair j at inv[j]; K2 puts it somewhere inside its expert group
        local = prod.map_topk_to_local(ids.long(), n, emap).reshape(-1)
        inv = gl["inv"].long()
        same = same and bool((inv >= 0).all()) and torch.equal(torch.sort(inv).values, torch.arange(P, device=dev))
        tok = torch.arange(T, device=dev).repeat_interleave(TOPK)
        same = same and torch.equal(gl["ts"][inv], tok)
        same = same and torch.equal(gl["ws"][inv], w.reshape(-1).to(torch.float16))
        same = same and torch.equal(gl["pe"].sort().values, k2["pe"].sort().values)
        # stable: sorted rows are (local id, pair index) ascending
        key = local * P + torch.arange(P, device=dev)
        same = same and torch.equal(torch.argsort(key), torch.argsort(inv))
        # xg / xu rows of computed pairs: match (token, expert, occurrence) between the two layouts
        pe_g = gl["pe"][inv]
        live = (pe_g >= 0).nonzero().flatten()
        if live.numel():
            rows_g = inv[live]
            # K2 row of the same pair: the K2 rows of expert e hold the tokens of that group in some order; pair j
            # (token t) of expert e with occurrence o (duplicates: the o-th pair of t in this group)
            occ = torch.zeros(P, dtype=torch.long, device=dev)
            k2_rows = torch.full((P,), -1, dtype=torch.long, device=dev)
            k2_map = {}
            pe_k, ts_k = k2["pe"].tolist(), k2["ts"].tolist()
            for r_, (e_, t_) in enumerate(zip(pe_k, ts_k)):
                k2_map.setdefault((e_, t_), []).append(r_)
            pos = []
            for j in live.tolist():
                key_ = (int(pe_g[j]), int(tok[j]))
                lst = k2_map.get(key_)
                pos.append(lst.pop(0) if lst else -1)
            k2r = torch.tensor(pos, device=dev)
            same = same and bool((k2r >= 0).all())
            if same:
                same = torch.equal(gl["xg"][rows_g], k2["xg"][k2r]) and torch.equal(gl["xu"][rows_g], k2["xu"][k2r])
        cases += 1
        if not same:
            mism += 1
            if mism <= 3:
                print(f"  glue_prep mismatch it={it} n={n} map={use_map} T={T} case={case} ids={idt} w={wdt}")
    print(f"(a) glue_prep vs route_ids + rot_in: {cases - mism}/{cases} identical (segment tables, rows of every pair "
          f"bitwise, stable order, inv, weights; n 288 / 32+EP map, T 1..128, int32/int64 ids, 3 weight dtypes)")
    ck(mism == 0, f"(a) glue_prep mismatches: {mism}")

    # ---- (b)/(c) bitwise vs K2 slot decomposition; vs production's apply -----------------------------------------
    def glue_call(x, ids, w, L):
        c0 = MG.COUNTERS.get("served_eager", 0) + MG.COUNTERS.get("served_captured", 0)
        y = prod.apply_exl3_experts(x, ids, w, L, limit=LIMIT)
        served = MG.COUNTERS.get("served_eager", 0) + MG.COUNTERS.get("served_captured", 0) == c0 + 1
        return y, served

    def k2_slots(x, ids, w, L, emap):
        """fp32 per-slot contributions from K2 (one valid route per token per call, same P)"""
        outs = []
        for k in range(TOPK):
            idk = torch.full_like(ids, -1)
            idk[:, k] = ids[:, k]
            o = tf.apply_fused(x, idk.long().contiguous(), w, L, L._exl3_inners, emap, LIMIT)
            assert o is not None
            outs.append(o)
        return outs

    nb, fails, worst_rel, worst_ulp, n_c, n_bit_cases = 0, 0, 0.0, 0.0, 0, 0
    err_glue, err_k2, err_glue_at = 0.0, 0.0, None
    for T in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 64, 128):
        for case in ("random", "sentinel", "duplicates", "hot", "all_sentinel_row", "overcap", "corr"):
            if T < 2 and case == "all_sentinel_row":
                continue
            for L, nn in ((layer, NEXP), (layer32, 64)):
                if L is layer32 and case not in ("random", "sentinel"):
                    continue
                emap = ep_map(g) if L is layer32 else None
                L.expert_map = emap
                x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                ids = ids_case(case, T, nn, g)
                w = H.random_weights(T, TOPK, g, dev)
                if T % 2:
                    w = w.to(torch.bfloat16)
                idt = (torch.int32, torch.int64)[T % 2]
                ids_in = ids.to(idt).contiguous()
                y, served = glue_call(x, ids_in, w, L)
                y2, _ = glue_call(x, ids_in, w, L)
                MG.STATE.enabled = False
                y_prod = prod.apply_exl3_experts(x, ids_in, w, L, limit=LIMIT)      # production (K2 + atomics)
                MG.STATE.enabled = True
                ok = served and y.dtype == torch.bfloat16 and y.shape == (T, K) and torch.equal(y, y2)
                loc = prod.map_topk_to_local(ids.long(), L._exl3_ptrs["gate_trellis"].numel(), emap).reshape(-1)
                over = bool((torch.bincount(loc)[:-1] > R).any()) if loc.numel() else False
                if not over:     # eligibility (count <= R) differs per slot when an expert is over the cap
                    parts = k2_slots(x, ids, w, L, emap)
                    acc = torch.zeros_like(parts[0])
                    for p_ in parts:
                        acc = acc + p_
                    ref = acc.to(torch.bfloat16)
                    bit = torch.equal(y, ref)
                    nb += bit
                    n_bit_cases += 1
                    ok = ok and bit
                    if not bit:
                        dd = (y.float() - ref.float()).abs()
                        print(f"  (b) not bitwise: T={T} case={case} n={nn}: {int((dd > 0).sum())} values differ, "
                              f"max {float(dd.max()):.3e}")
                    # fp64 truth of the same per-pair contributions
                    tru = sum(p_.double() for p_ in parts)
                    den = max(tru.norm().item(), 1e-30)
                    eg = (y.double() - tru).norm().item() / den
                    ek = (y_prod.double() - tru).norm().item() / den
                    if eg > err_glue:
                        err_glue, err_glue_at = eg, (T, case, nn)
                    err_k2 = max(err_k2, ek)
                d = (y.float() - y_prod.float())
                rel = d.norm().item() / max(y_prod.float().norm().item(), 1e-30)
                # |diff| relative to the largest |value| of its row (bf16 ulp at the row's scale = 2^-8 of it)
                rowmax = y_prod.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
                ulp = float((d.abs() / rowmax).max()) * 256.0
                worst_rel, worst_ulp = max(worst_rel, rel), max(worst_ulp, ulp)
                ok = ok and rel <= 1e-4 and ulp <= 2.0
                n_c += 1
                if not ok:
                    fails += 1
                    print(f"  (b/c) T={T} case={case} n={nn} ids={idt}: served={served} rel={rel:.2e} ulp={ulp}")
                L.expert_map = None
    torch.cuda.synchronize()
    print(f"(b) glue == bf16(slot-ordered fp32 sum of K2's per-pair contributions): {nb}/{n_bit_cases} bitwise "
          f"(every routing without an expert over the cap R); (c) vs production's apply (K2, atomics): {n_c - fails}/{n_c} ok, worst "
          f"rel_l2 {worst_rel:.2e}, worst |diff| {worst_ulp:.2f} bf16 ulp at the row's max; vs the fp64 sum of the "
          f"per-pair values: glue {err_glue:.2e} (at {err_glue_at}), production {err_k2:.2e}; glue deterministic")
    ck(fails == 0, f"(b/c) failures: {fails}")

    # ---- (d) CUDA graph -----------------------------------------------------------------------------------------
    T = 5
    xs = [torch.zeros(T, K, dtype=torch.bfloat16, device=dev) for _ in range(2)]
    idss = [torch.zeros(T, TOPK, dtype=torch.int32, device=dev) for _ in range(2)]
    wss = [torch.zeros(T, TOPK, dtype=torch.float32, device=dev) for _ in range(2)]
    Ls = [layer, layer]
    W2 = H.Weights(NEXP, K, N, dev, seed=43)
    Ls[1] = H.make_layer(prod, W2)

    def fill():
        for i in range(2):
            xs[i].copy_(torch.randn(T, K, generator=g).to(torch.bfloat16))
            idss[i].copy_(H.correlated_ids(T, NEXP, TOPK, 40, g, "cpu").to(torch.int32))
            wss[i].copy_(H.random_weights(T, TOPK, g, "cpu"))

    fill()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            prod.apply_exl3_experts(xs[i], idss[i], wss[i], Ls[i], limit=LIMIT)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    c0 = MG.COUNTERS.get("served_captured", 0)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        gouts = [prod.apply_exl3_experts(xs[i], idss[i], wss[i], Ls[i], limit=LIMIT) for i in range(2)]
    ck(MG.COUNTERS.get("served_captured", 0) == c0 + 2, "moeglue not taken during capture")
    bad = 0
    for rep_ in range(100):
        fill()
        gr.replay()
        eager = [prod.apply_exl3_experts(xs[i], idss[i], wss[i], Ls[i], limit=LIMIT) for i in range(2)]
        torch.cuda.synchronize()
        bad += sum(not torch.equal(a, b) for a, b in zip(gouts, eager))
    print(f"(d) CUDA graph, 2 layers x 100 replays with fresh routing: {200 - bad}/200 bit-identical to eager glue")
    ck(bad == 0, "(d) graph replay differs")
    torch.cuda.set_sync_debug_mode("error")
    try:
        prod.apply_exl3_experts(xs[0], idss[0], wss[0], Ls[0], limit=LIMIT)
        nosync = True
    except RuntimeError as exc:
        nosync = False
        print(f"  sync: {exc}")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print(f"(d) no host synchronisation in the eager glue call: {nosync}")
    ck(nosync, "(d) glue synchronizes")

    # ---- (e) delegation ---------------------------------------------------------------------------------------
    def same_as_prod(label, x, ids, w, L, **kw):
        c0 = sum(v for k_, v in MG.COUNTERS.items() if k_.startswith("served"))
        try:
            y = prod.apply_exl3_experts(x, ids, w, L, limit=LIMIT, **kw)
        except Exception as exc:  # noqa: BLE001
            y = exc
        served = sum(v for k_, v in MG.COUNTERS.items() if k_.startswith("served")) != c0
        MG.STATE.enabled = False
        try:
            y0 = prod.apply_exl3_experts(x, ids, w, L, limit=LIMIT, **kw)
        except Exception as exc:  # noqa: BLE001
            y0 = exc
        MG.STATE.enabled = True
        if isinstance(y, Exception) or isinstance(y0, Exception):
            eq = type(y) is type(y0)
        else:
            eq = y.dtype == y0.dtype and y.shape == y0.shape and \
                 (tf.compare(y.float(), y0.float())["rel_l2"] <= 1e-3)
        print(f"(e) {label}: delegated {not served}, same as production {eq}")
        ck(not served and eq, f"(e) {label}")

    T = 4
    x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
    ids = H.random_ids(T, NEXP, TOPK, g, dev).to(torch.int32)
    w = H.random_weights(T, TOPK, g, dev)
    MG.CFG.strict = False
    tf.CFG.strict = False
    same_as_prod("fp32 x", x.float(), ids, w, layer)
    same_as_prod("int16 ids", x, ids.to(torch.int16), w, layer)
    same_as_prod("T > R", torch.randn(R + 1, K, generator=g).to(torch.bfloat16).to(dev),
                 H.random_ids(R + 1, NEXP, TOPK, g, dev).to(torch.int32), H.random_weights(R + 1, TOPK, g, dev), layer)
    os.environ["EXL3_FUSED_MOE"] = "0"
    try:
        c0 = sum(v for k_, v in MG.COUNTERS.items() if k_.startswith("served"))
        r = MG.try_glue(x, ids, w, layer, LIMIT, None)
        ck(r is None and sum(v for k_, v in MG.COUNTERS.items() if k_.startswith("served")) == c0,
           "(e) EXL3_FUSED_MOE=0 must delegate")
        print("(e) EXL3_FUSED_MOE=0: delegated True")
    finally:
        os.environ.pop("EXL3_FUSED_MOE", None)
    ck(MG.try_glue(x, ids, w, layer, LIMIT, False) is None, "(e) fused=False must delegate")
    print("(e) fused=False: delegated True")
    key = (0, layer._exl3_ptrs["gate_trellis"].data_ptr())
    saved = MG.VERDICT.pop(key)
    same_as_prod("layer without a moeglue verdict", x, ids, w, layer)
    MG.VERDICT[key] = (False, "test")
    same_as_prod("layer whose self-test failed", x, ids, w, layer)
    MG.VERDICT[key] = saved
    tf.STATE.apply_enabled = False
    same_as_prod("TF K2 disabled", x, ids, w, layer)
    tf.STATE.apply_enabled = True
    # install refusals / uninstall
    MG.uninstall(prod)
    ck(prod.apply_exl3_experts is orig_experts, "uninstall must restore apply_exl3_experts")
    saved_v = dict(MG.VERIFIED)
    MG.VERIFIED["apply_exl3_experts"] = frozenset({"0000000000000000"})
    r = MG.install(prodmod=prod, force=True)
    MG.VERIFIED.update(saved_v)
    print(f"(e) install with an unverified apply_exl3_experts: {r['installed']} ({r['reason']})")
    ck(not r["installed"] and prod.apply_exl3_experts is orig_experts, "(e) fingerprint mismatch must refuse")
    r = MG.install(prodmod=prod, force=None)
    ck(not r["installed"] or MG.env_enabled(), "(e) env off must not install")
    print(f"(e) env {MG.ENV} unset: installed {r['installed']} ({r['reason']})")
    H.report_peak(8.0)
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
