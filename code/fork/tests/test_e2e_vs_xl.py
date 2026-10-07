"""E.1: TF path vs production exllamav3_ext.exl3_moe on identical, production-built arguments
(docs/DESIGN.md §E.1).

Flow per case: integrate.install() (dispatcher + build hook) -> layers built by production's own
process_weights_after_loading (hooked build_exl3_fused_state -> tf.preflight incl. self-test) -> production
apply_exl3_fused_moe with a recorder in place of exl3_moe yields the exact positional args -> orig(*args)
(concurrency C = max, as production) vs tf.launch (STRICT) on fresh zero outputs.
Tolerances (E.1): global rel_l2 <= 2e-3, per-row rel_l2 <= 1e-2 over rows with ||ref_row|| > 1e-3*max,
non-finite positions identical, rows that are ~0 in the reference stay ~0. Noise floor XL(C=max) vs
XL(C=1) <= 1e-6. Mutation check: zeroing one pair's weight must move that row by > 0.1.
Delegation: plan() is None and the dispatcher's result equals orig within 1e-6 (C=1, deterministic).
Cases: T in {1, 2, 4, 8, 16, 64} x n in {32, 288} and T in {65, 96, 127, 128} (n = 288; 128 also n = 32, skewed with
count = R and with sentinels): every B <= R that plan() serves, up to P = P_cap = 1024 (asserted to have run).
"""
from __future__ import annotations

import torch

import harness as H

K, N, TOPK = 4096, 1024, 8


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep["installed"], f"install failed: {rep}")
    orig = rep["orig"]
    disp = xl.exl3_moe
    ck(getattr(disp, "_tf_exl3_dispatch", False), "dispatcher not installed")
    ck(prod._exl3_moe_accepts_num_active(disp) == prod._exl3_moe_accepts_num_active(orig),
       "num_active detection differs between dispatcher and orig")
    print(f"install: {rep['reason']}, num_active detection (orig/dispatcher): "
          f"{prod._exl3_moe_accepts_num_active(orig)}/{prod._exl3_moe_accepts_num_active(disp)}")
    tf.CFG.strict = True

    layers, weights = {}, {}
    for n in (32, 288):
        W = H.Weights(n, K, N, dev, seed=100 + n)
        L_ = H.make_layer(prod, W)                          # production path; build hook -> preflight
        layers[n], weights[n] = L_, W
        info = tf.REG.get((0, L_._exl3_ptrs["gate_trellis"].data_ptr()))
        ck(info is not None and info.ok, f"n={n}: layer not registered by preflight")
        print(f"layer n={n}: registered={info is not None}, self-test {info.selftest if info else None}")
    # calibrate svh_d so std(out) ~ 1 (in place: pointers unchanged)
    g = torch.Generator().manual_seed(1)
    for n, L_ in layers.items():
        x = torch.randn(16, K, generator=g).half().to(dev)
        ids = H.random_ids(16, n, TOPK, g, dev)
        w = H.random_weights(16, TOPK, g, dev)
        a = H.capture_args(prod, xl, x, ids, w, L_, 10.0)
        o = torch.zeros(16, K, dtype=torch.float32, device=dev)
        orig(*H.with_out(a, o))
        s = float(o.std())
        weights[n].rescale_svh_d(1.0 / s)
        o.zero_()
        orig(*H.with_out(a, o))
        print(f"calibrated n={n}: std(out) {s:.3g} -> {float(o.std()):.3f}")

    worst = {"rel_l2": 0.0, "row_rel_max": 0.0, "noise": 0.0, "P_max": 0, "B_max": 0}
    ncases = [0]

    def run_case(name, n, x2d, ids, w, L, expert_map=None, tables=None, expect_zero_rows=(), expect_all_zero=False,
                 layer=None):
        layer = layers[n] if layer is None else layer
        args = H.capture_args(prod, xl, x2d, ids, w, layer, L, expert_map=expert_map)
        if tables is not None:
            args = args[:13] + tuple(tables) + args[22:]
        B = x2d.shape[0]
        out_xl = torch.zeros(B, K, dtype=torch.float32, device=dev)
        out_x1 = torch.zeros_like(out_xl)
        out_tf = torch.zeros_like(out_xl)
        orig(*H.with_out(args, out_xl))
        orig(*H.with_temps(H.with_out(args, out_x1), H.c1_temps(layer)))
        p = tf.plan(H.with_out(args, out_tf))
        if not ck(p is not None, f"{name}: plan() returned None"):
            return None
        tf.launch(p, H.with_out(args, out_tf), None)
        torch.cuda.synchronize()
        m = tf.compare(out_tf, out_xl)
        noise = tf.compare(out_xl, out_x1)["rel_l2"]
        ok = tf.passes_e1(m)
        for r in expect_zero_rows:
            ok &= bool((out_tf[r] == 0).all()) and bool((out_xl[r] == 0).all())
        if expect_all_zero:
            ok &= bool((out_tf == 0).all()) and bool((out_xl == 0).all())
        ncases[0] += 1
        worst["P_max"] = max(worst["P_max"], args[3].numel())
        worst["B_max"] = max(worst["B_max"], B)
        worst["rel_l2"] = max(worst["rel_l2"], m["rel_l2"] if m["ref_norm"] > 0 else 0.0)
        worst["row_rel_max"] = max(worst["row_rel_max"], m["row_rel_max"])
        worst["noise"] = max(worst["noise"], noise)
        print(f"  {name:44s} L={L:4.1f} P={args[3].numel():4d} nseg={int(p.scratch.nseg):3d} rel_l2 {m['rel_l2']:.2e} "
              f"row_max {m['row_rel_max']:.2e} finite_eq {m['finite_equal']} noise(XL C=max vs C=1) {noise:.1e} "
              f"{'ok' if ok else 'FAIL'}")
        ck(ok, f"{name} L={L}: {m}")
        ck(noise <= 1e-6, f"{name} L={L}: XL noise floor {noise:.2e} > 1e-6")
        return args, out_xl, out_tf

    Ls = (10.0, 1.0, 0.0)
    print("-- random distinct experts")
    for n in (32, 288):
        for T in (1, 2, 4, 8, 16, 64):
            gg = torch.Generator().manual_seed(1000 * n + T)
            x = torch.randn(T, K, generator=gg).half().to(dev)
            ids = H.random_ids(T, n, TOPK, gg, dev)
            w = H.random_weights(T, TOPK, gg, dev)
            for L in Ls:
                run_case(f"random n={n} T={T}", n, x, ids, w, L)

    print("-- T = 65..128: B up to R, P up to P_cap = R * topk = 1024 (full scratch, S_cap up to 352)")
    for n in (288, 32):
        for T in ((65, 96, 127, 128) if n == 288 else (128,)):
            gg = torch.Generator().manual_seed(2000 * n + T)
            x = torch.randn(T, K, generator=gg).half().to(dev)
            ids = H.random_ids(T, n, TOPK, gg, dev)
            w = H.random_weights(T, TOPK, gg, dev)
            for L in Ls:
                run_case(f"random n={n} T={T}", n, x, ids, w, L)
    n, T = 288, 128
    gg = torch.Generator().manual_seed(128)
    x = torch.randn(T, K, generator=gg).half().to(dev)
    ids = torch.stack([torch.cat([torch.tensor([0]), 1 + torch.randperm(n - 1, generator=gg)[:TOPK - 1]])
                       for _ in range(T)]).to(dev)
    w = H.random_weights(T, TOPK, gg, dev)
    for L in Ls:
        run_case("T=128 skewed: count(e0) = R = 128 (8 segs)", n, x, ids, w, L)
    ids = H.random_ids(T, n, TOPK, gg, dev)
    ids[::5, 0] = -1
    ids[3::7, 5] = n + 1
    for L in Ls:
        run_case("T=128 sentinel ids -1 and >= n", n, x, ids, w, L)
    info288 = tf.REG[(0, layers[288]._exl3_ptrs["gate_trellis"].data_ptr())]
    print(f"   largest TF case: B = {worst['B_max']}, P = {worst['P_max']} (layer P_cap {info288.P_cap})")
    ck(worst["P_max"] == info288.P_cap and worst["B_max"] == int(layers[288]._exl3_fused_temps[0].shape[1]),
       "no TF case at P = P_cap / B = R")

    print("-- N = 768 (intermediate % 256 != 0: exl3_moe's TILESIZE_N=128 instance; E.0 second dims)")
    W768 = H.Weights(32, K, 768, dev, seed=768)
    l768 = H.make_layer(prod, W768)
    ck((0, l768._exl3_ptrs["gate_trellis"].data_ptr()) in tf.REG, "N=768 layer not registered")
    gg = torch.Generator().manual_seed(768)
    a = H.capture_args(prod, xl, torch.randn(16, K, generator=gg).half().to(dev), H.random_ids(16, 32, TOPK, gg, dev),
                       H.random_weights(16, TOPK, gg, dev), l768, 10.0)
    o = torch.zeros(16, K, dtype=torch.float32, device=dev)
    orig(*H.with_out(a, o))
    W768.rescale_svh_d(1.0 / float(o.std()))
    for T in (1, 8, 64):
        x = torch.randn(T, K, generator=gg).half().to(dev)
        ids = H.random_ids(T, 32, TOPK, gg, dev)
        w = H.random_weights(T, TOPK, gg, dev)
        for L in Ls:
            run_case(f"random N=768 n=32 T={T}", 32, x, ids, w, L, layer=l768)

    print("-- skewed")
    n = 288
    gg = torch.Generator().manual_seed(7)
    T = 64
    x = torch.randn(T, K, generator=gg).half().to(dev)
    ids = torch.stack([torch.cat([torch.tensor([0]), 1 + torch.randperm(n - 1, generator=gg)[:TOPK - 1]])
                       for _ in range(T)]).to(dev)
    w = H.random_weights(T, TOPK, gg, dev)
    for L in Ls:
        run_case("skewed: every token has expert 0 (4 segs)", n, x, ids, w, L)
    same = torch.randperm(n, generator=gg)[:TOPK]
    ids = same.repeat(T, 1).to(dev)
    for L in Ls:
        run_case("skewed: all tokens pick the same 8", n, x, ids, w, L)

    print("-- cap edges (duplicate ids per token, synthetic)")
    for n in (32, 288):
        R = int(layers[n]._exl3_fused_temps[0].shape[1])
        T = R // TOPK                                        # 16 tokens x 8 = 128 = R pairs, all expert 5
        x = torch.randn(T, K, generator=gg).half().to(dev)
        ids = torch.full((T, TOPK), 5, dtype=torch.long, device=dev)
        w = H.random_weights(T, TOPK, gg, dev)
        for L in Ls:
            run_case(f"cap: count(e5) = R = {R}, n={n}", n, x, ids, w, L)
        T = R // TOPK + 1                                    # 17 tokens, 136 pairs, 129 of them expert 5
        x = torch.randn(T, K, generator=gg).half().to(dev)
        ids = torch.full((T, TOPK), 5, dtype=torch.long)
        others = [e for e in range(n) if e != 5]
        ids[T - 1, 1:] = torch.tensor(others)[torch.randperm(len(others), generator=gg)[:TOPK - 1]]
        ids = ids.to(dev)
        w = H.random_weights(T, TOPK, gg, dev)
        for L in Ls:
            run_case(f"cap: count(e5) = R+1 = {R + 1} (skipped), n={n}", n, x, ids, w, L)

    print("-- sentinel / non-local")
    n = 288
    T = 8
    x = torch.randn(T, K, generator=gg).half().to(dev)
    ids = H.random_ids(T, n, TOPK, gg, dev)
    ids[0, 0] = -1
    ids[1, 3] = n + 5
    ids[2, 7] = -1
    ids[3, 1] = 10 ** 6
    w = H.random_weights(T, TOPK, gg, dev)
    for L in Ls:
        run_case("sentinel: ids -1 and >= n (no expert_map)", n, x, ids, w, L)
    emap = torch.where(torch.arange(2 * n) >= n, torch.arange(2 * n) - n, torch.full((2 * n,), -1)).to(dev)
    ids_g = torch.stack([torch.randperm(2 * n, generator=gg)[:TOPK] for _ in range(T)]).to(dev)
    for L in Ls:
        run_case("EP expert_map: half the experts non-local", n, x, ids_g, w, L, expert_map=emap)
    T = 4
    x4 = torch.randn(T, K, generator=gg).half().to(dev)
    ids = H.random_ids(T, n, TOPK, gg, dev)
    ids[2, :] = -1
    w4 = H.random_weights(T, TOPK, gg, dev)
    for L in Ls:
        run_case("one token with all ids sentinel (row 2 = 0)", n, x4, ids, w4, L, expect_zero_rows=(2,))
    ids = torch.full((T, TOPK), -1, dtype=torch.long, device=dev)
    for L in Ls:
        run_case("all pairs sentinel (nseg = 0, out = 0)", n, x4, ids, w4, L, expect_all_zero=True)

    print("-- unaligned counts 15, 16, 17, 33")
    T = 40
    x = torch.randn(T, K, generator=gg).half().to(dev)
    rows = []
    for t in range(T):
        have = [e for e, c in ((1, 15), (2, 16), (3, 17), (4, 33)) if t < c]
        rest = [int(e) for e in (5 + torch.randperm(n - 5, generator=gg))[: TOPK - len(have)]]
        rows.append(have + rest)
    ids = torch.tensor(rows, dtype=torch.long, device=dev)
    w = H.random_weights(T, TOPK, gg, dev)
    for L in Ls:
        res = run_case("unaligned counts", n, x, ids, w, L)
    args = res[0]
    ec = args[2]
    print(f"   counts of experts 1..4: {ec[1:5].tolist()}")
    ck(ec[1:5].tolist() == [15, 16, 17, 33], "unaligned counts not as constructed")

    print("-- separate per-expert tensors (pointer-table generality)")
    n = 32
    S = H.SeparateExperts(n, K, N, dev, seed=0, like=weights[n])
    ok = tf.register(S.ptrs, layers[n]._exl3_fused_temps, K, N, S.regions(), orig=orig, name="separate")
    ck(ok, "separate-tensor tables not registered")
    tabs = [S.ptrs[k] for k in ("gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
                                "down_trellis", "down_suh", "down_svh")]
    for T in (1, 8, 64):
        gg2 = torch.Generator().manual_seed(77 + T)
        x = torch.randn(T, K, generator=gg2).half().to(dev)
        ids = H.random_ids(T, n, TOPK, gg2, dev)
        w = H.random_weights(T, TOPK, gg2, dev)
        for L in (10.0, 1.0):
            r_sep = run_case(f"separate tensors n={n} T={T}", n, x, ids, w, L, tables=tabs)
            r_stk = run_case(f"stacked (same values) n={n} T={T}", n, x, ids, w, L)
            rel = float((r_sep[2] - r_stk[2]).norm() / r_stk[2].norm())
            ck(rel <= 1e-6, f"separate vs stacked TF differ {rel:.2e}")

    print("-- mutation check (tolerance has teeth)")
    n, T = 288, 8
    gg = torch.Generator().manual_seed(4242)
    x = torch.randn(T, K, generator=gg).half().to(dev)
    ids = H.random_ids(T, n, TOPK, gg, dev)
    w = H.random_weights(T, TOPK, gg, dev)
    args, out_xl, out_tf = run_case("mutation base", n, x, ids, w, 10.0)
    ts, ws = args[3], args[4]
    cand = (ts == 0).nonzero().flatten()
    j = int(cand[torch.argmax(ws[cand].float())])
    ws_mut = ws.clone()
    ws_mut[j] = 0
    out_mut = torch.zeros_like(out_tf)
    a_mut = H.with_out(args, out_mut)
    a_mut = a_mut[:4] + (ws_mut,) + a_mut[5:]
    tf.launch(tf.plan(a_mut), a_mut, None)
    torch.cuda.synchronize()
    row_rel = float((out_mut[0] - out_xl[0]).norm() / out_xl[0].norm())
    other = float((out_mut[1:] - out_xl[1:]).norm() / out_xl[1:].norm())
    print(f"  zeroed pair j={j} (token 0, w={float(ws[j]):.3f}): row-0 rel_l2 {row_rel:.3f} (> 0.1 required), "
          f"other rows rel {other:.2e}")
    ck(row_rel > 0.1, f"mutation not detected: {row_rel}")
    ck(other <= 2e-3, "mutation leaked into other rows")

    print("-- A10 write-before-read: scratch poisoned with NaN before each call")
    sc = tf._SCRATCH[(0, K, N)]
    for T, sentinel in ((8, False), (64, True), (1, False)):
        gg = torch.Generator().manual_seed(1010 + T)
        x = torch.randn(T, K, generator=gg).half().to(dev)
        ids = H.random_ids(T, 288, TOPK, gg, dev)
        if sentinel:
            ids[::3, 0] = -1
        w = H.random_weights(T, TOPK, gg, dev)
        for t_ in (sc.xg, sc.xu, sc.xd, sc.z):
            t_.fill_(float("nan"))
        sc.pair_expert.fill_(7)                      # stale "valid" rows must be overwritten by route_prep
        sc.seg_expert.fill_(123)
        r_ = run_case(f"A10 poisoned scratch T={T} sentinel={sentinel}", 288, x, ids, w, 10.0)
        ck(r_ is not None and bool(torch.isfinite(r_[2]).all()), f"A10: non-finite output T={T}")

    print(f"-- E.1 summary over {ncases[0]} TF cases: worst global rel_l2 {worst['rel_l2']:.2e} (tol 2e-3), "
          f"worst row rel {worst['row_rel_max']:.2e} (tol 1e-2), worst XL noise floor {worst['noise']:.1e} (tol 1e-6)")

    # ---------------- delegation cases ---------------------------------------------------------------------
    print("-- delegation (plan() is None; dispatcher result == orig, C=1)")
    n, T = 288, 8
    layer = layers[n]
    gg = torch.Generator().manual_seed(99)
    x = torch.randn(T, K, generator=gg).half().to(dev)
    ids = H.random_ids(T, n, TOPK, gg, dev)
    w = H.random_weights(T, TOPK, gg, dev)
    base = H.with_temps(H.capture_args(prod, xl, x, ids, w, layer, 10.0), H.c1_temps(layer))

    def delegate(name, args, compare=True):
        o1 = torch.zeros(args[0].shape[0], args[0].shape[1], dtype=torch.float32, device=dev)
        o2 = torch.zeros_like(o1)
        p = tf.plan(H.with_out(args, o1))
        d0 = tf.COUNTERS["delegated"]
        c0 = tf.COUNTERS["tf_calls"]
        disp(*H.with_out(args, o1))
        orig(*H.with_out(args, o2))
        torch.cuda.synchronize()
        delegated = tf.COUNTERS["delegated"] == d0 + 1 and tf.COUNTERS["tf_calls"] == c0
        if compare:
            fin = torch.isfinite(o2)
            same = torch.equal(torch.isfinite(o1), fin)
            den = float(o2[fin].norm()) or 1.0
            rel = float((o1[fin] - o2[fin]).norm()) / den
        else:
            same, rel = True, 0.0
        print(f"  {name:48s} plan None: {p is None}, delegated: {delegated}, rel vs orig {rel:.1e}")
        ck(p is None and delegated and same and rel <= 1e-6, f"delegation {name}")

    Tb = int(layer._exl3_fused_temps[0].shape[1]) + 1
    xb = torch.randn(Tb, K, generator=gg).half().to(dev)
    idsb = H.random_ids(Tb, n, TOPK, gg, dev)
    wb = H.random_weights(Tb, TOPK, gg, dev)
    delegate("B = R + 1 (prefill-sized)", H.with_temps(H.capture_args(prod, xl, xb, idsb, wb, layer, 10.0),
                                                      H.c1_temps(layer)))
    delegate("K_gate = 3", base[:10] + (3,) + base[11:])
    delegate("act_function = 1 (GELU)", base[:9] + (1,) + base[10:])
    delegate("weight_sorted bf16", base[:4] + (base[4].to(torch.bfloat16),) + base[5:])
    delegate("unregistered pointer table (clone)", base[:13] + (base[13].clone(),) + base[14:])
    delegate("act_limit = inf", base[:28] + (float("inf"),))
    for (Kx, Nx) in ((3968, 1024), (4096, 1088)):
        Wx = H.Weights(8, Kx, Nx, dev, seed=5)
        Lx = H.make_layer(prod, Wx)
        ck(tf.REG.get((0, Lx._exl3_ptrs["gate_trellis"].data_ptr())) is None,
           f"ineligible shape K={Kx} N={Nx} was registered")
        xx = torch.randn(4, Kx, generator=gg).half().to(dev)
        ax = H.with_temps(H.capture_args(prod, xl, xx, H.random_ids(4, 8, 4, gg, dev),
                                         H.random_weights(4, 4, gg, dev), Lx, 10.0), H.c1_temps(Lx))
        # exl3_moe itself only supports N % 128 == 0 (its tiles cover 1024 of 1088 columns and the down GEMM reads
        # uninitialized temp columns: two identical orig calls differ by ~1e-2), so N=1088 is checked for
        # plan()/delegation only; K=3968 (hidden % 256 != 0, supported by exl3_moe) carries the numeric check.
        delegate(f"ineligible shape K={Kx} N={Nx} (pre-flight rejected)", ax, compare=(Nx % 128 == 0))
        del Wx, Lx

    print("-- 30-argument call (num_active) through the dispatcher")
    seen = []

    def fake_orig(*a):
        seen.append(len(a))
        return orig(*a[:29])

    fake_orig.__doc__ = orig.__doc__
    d30 = integrate._make_dispatcher(tf, fake_orig)
    a30 = base[:9] + (1,) + base[10:] + (-1,)              # delegated (GELU) with num_active
    d30(*H.with_out(a30, torch.zeros(T, K, dtype=torch.float32, device=dev)))
    ck(seen == [30], f"orig did not receive exactly 30 args: {seen}")
    o29 = torch.zeros(T, K, dtype=torch.float32, device=dev)
    o30 = torch.zeros_like(o29)
    d30(*H.with_out(base, o29))
    d30(*H.with_out(base + (-1,), o30))
    torch.cuda.synchronize()
    rel = float((o29 - o30).norm() / o29.norm())
    print(f"  delegated 30-arg call: orig received {seen[0]} args; TF path 29 vs 30 args rel {rel:.1e}; "
          f"orig calls {len(seen)}")
    ck(seen == [30] and rel <= 1e-6, "30-arg TF path")

    def fake_na(*a):
        pass

    fake_na.__doc__ = "exl3_moe(arg0, ..., num_active: int)"
    d_na = integrate._make_dispatcher(tf, fake_na)
    ck(prod._exl3_moe_accepts_num_active(d_na) is True and prod._exl3_moe_accepts_num_active(fake_na) is True,
       "doc-based num_active detection not preserved")
    print(f"  detector on a num_active-accepting orig: orig {prod._exl3_moe_accepts_num_active(fake_na)}, "
          f"dispatcher {prod._exl3_moe_accepts_num_active(d_na)}")

    integrate.uninstall(prodmod=prod, ext=xl)
    ck(xl.exl3_moe is orig and not getattr(prod.build_exl3_fused_state, "_tf_exl3_hook", False),
       "uninstall did not restore")
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
