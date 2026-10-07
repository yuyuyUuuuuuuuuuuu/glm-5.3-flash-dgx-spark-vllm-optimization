"""Stage unit tests (docs/DESIGN.md §E.2 E-U1..E-U4) + down_epilogue exactness + JIT == AOT build.

E-U1 rot_in                vs exllamav3_ext.had_r_128(x_row, out, suh_e, None, 1.0): torch.equal (bit-identical)
E-U2 grouped (ptr tables)  interleaved w13 / w2, e in {0, n-1}, gate/up/down vs float64 X16 @ unpack(trellis_e):
                           rel <= 1e-5
E-U2c grouped variants     every kernel variant's Z == the master kernel's, gate/up SK 2 / 4 / 8, down SK 1 / 2, at
                           S_cap 5 and at S_cap 65543 > 65535 (the ORD 0 fallback of the ORD 1 / 2 grids)
E-U7 whole call            moe_forward at S_cap > 65535 (P = 62000, small dims) runs (the ORD 0 fallback) and equals
                           the same tokens served by two ORD 1 calls, bit for bit
E-U3 route_prep            vs the §B.4 torch oracle: exact equality, 1000 random routings, n in {288, 1100}
E-U4 gateup_epilogue       vs a numpy emulation of exl3_moe's fp16 step 3: <= 1 fp16 ulp on <= 1% of elements, at
                           SK 2, 4 and 8 (every gate/up split count production uses)
E-U5 down_epilogue         one pair per token (no atomic races) vs numpy emulation: bit-identical
tile cfg / scratch         tf_exl3_moe's tile constants == the compiled extension's; the persistent Z holds
                           ext.z_need(P) for every P <= P_cap (the pre-launch check's own bound)
build                      the active extension must be the in-place AOT build (never a silent JIT fallback); the JIT
                           build of the same sources gives moe_forward results equal up to fp32 atomic order
"""
from __future__ import annotations

import numpy as np
import torch

import harness as H

K, N, NEXP = 4096, 1024, 288


def fwht_np(v: np.ndarray) -> np.ndarray:
    """float32 Walsh-Hadamard over the last axis (128), strides 1,2,...,64 — the kernel's butterfly order."""
    v = v.astype(np.float32).copy()
    s = 1
    while s < 128:
        w = v.reshape(v.shape[:-1] + (128 // (2 * s), 2, s))
        a = w[..., 0, :].copy()
        b = w[..., 1, :].copy()
        w[..., 0, :] = a + b
        w[..., 1, :] = a - b
        v = w.reshape(v.shape)
        s *= 2
    return v


R_SCALE = np.float32(0.088388347648)


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16)


def oracle_route(c: torch.Tensor, P: int, R: int, S_cap: int):
    """§B.4 torch-op oracle."""
    n = c.numel() - 1
    cnt = c[:n]
    end = torch.cumsum(cnt, 0)
    start = end - cnt
    elig = (cnt > 0) & (cnt <= R) & (end <= P) & (start >= 0)
    nsg = torch.where(elig, (cnt + 15) // 16, torch.zeros_like(cnt))
    send = torch.cumsum(nsg, 0)
    soff = send - nsg
    nseg = int(send[-1])
    j = torch.arange(P, device=c.device)
    eo = torch.searchsorted(end, j, right=True)
    ok = eo < n
    ec = eo.clamp(max=n - 1)
    pair_expert = torch.where(ok & elig[ec], eo, torch.full_like(eo, -1)).int()
    s = torch.arange(S_cap, device=c.device)
    se = torch.searchsorted(send, s, right=True)
    sv = s < nseg
    sc = se.clamp(max=n - 1)
    q = s - soff[sc]
    seg_expert = torch.where(sv, se, torch.full_like(se, -1))
    seg_row0 = start[sc] + 16 * q
    seg_rows = torch.minimum(torch.full_like(q, 16), cnt[sc] - 16 * q)
    return pair_expert, min(nseg, S_cap), seg_expert.int(), seg_row0.int(), seg_rows.int()


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    ext = tf.load_ext()
    ck = H.Checks()
    dev = torch.device("cuda", 0)
    W = H.Weights(NEXP, K, N, dev, seed=11)
    layer = H.make_layer(prod, W)
    ptr = layer._exl3_ptrs
    g = torch.Generator().manual_seed(5)

    # ---------------- E-U1 rot_in bit-identical to had_r_128 ------------------------------------------------
    B, P = 6, 40
    x = torch.randn(B, K, generator=g).half().to(dev) * 3
    ts = torch.randint(0, B, (P,), generator=g).to(dev)
    pe = torch.randint(0, NEXP, (P,), generator=g).int().to(dev)
    pe[3] = -1
    xg = torch.full((P, K), float("nan"), dtype=torch.float16, device=dev)
    xu = torch.full_like(xg, float("nan"))
    ext.rot_in(x, ts, pe, ptr["gate_suh"], ptr["up_suh"], xg, xu)
    torch.cuda.synchronize()
    ok_all, nd = True, 0
    for j in range(P):
        e = int(pe[j])
        if e < 0:
            ok_all &= bool(torch.isnan(xg[j]).all())      # skipped rows untouched
            continue
        t = int(ts[j])
        for mat, buf in ((0, xg), (1, xu)):
            ref = torch.empty((1, K), dtype=torch.float16, device=dev)
            xl.had_r_128(x[t:t + 1].contiguous(), ref, W.w13_suh[e, mat].contiguous(), None, 1.0)
            eq = torch.equal(buf[j:j + 1], ref)
            ok_all &= eq
            nd += int((buf[j:j + 1] != ref).sum())
    print(f"E-U1 rot_in vs had_r_128: {P} pairs x 2 mats, bit-identical={ok_all}, differing values={nd}")
    ck(ok_all, "E-U1 rot_in not bit-identical to had_r_128")

    # rot_in with a strided (row-padded) x view
    xpad = torch.randn(B, K + 64, generator=g).half().to(dev)
    xv = xpad[:, :K]
    ext.rot_in(xv, ts, pe, ptr["gate_suh"], ptr["up_suh"], xg, xu)
    xg2 = torch.empty_like(xg)
    xu2 = torch.empty_like(xu)
    ext.rot_in(xv.contiguous(), ts, pe, ptr["gate_suh"], ptr["up_suh"], xg2, xu2)
    torch.cuda.synchronize()
    valid = pe >= 0
    ok = torch.equal(xg[valid], xg2[valid]) and torch.equal(xu[valid], xu2[valid])
    print(f"E-U1b rot_in strided x (row stride {xv.stride(0)}) == contiguous: {ok}")
    ck(ok, "E-U1b strided x differs")

    # ---------------- E-U2 grouped via pointer tables vs float64 -------------------------------------------
    import exl3_format_ref as R

    def seg_tables(rows_per_expert):   # [(e, row0, rows)] -> device tables, padded by 2 unused entries
        pad = [(-1, 0, 0)] * 2
        se = torch.tensor([r[0] for r in rows_per_expert + pad], dtype=torch.int32, device=dev)
        s0 = torch.tensor([r[1] for r in rows_per_expert + pad], dtype=torch.int32, device=dev)
        sr = torch.tensor([r[2] for r in rows_per_expert + pad], dtype=torch.int32, device=dev)
        ns = torch.tensor([len(rows_per_expert)], dtype=torch.int32, device=dev)
        return se, s0, sr, ns

    Pg = 32
    segs = [(0, 0, 16), (0, 16, 3), (NEXP - 1, 19, 13)]   # expert 0 split in 2 segments; last expert
    se, s0, sr, ns = seg_tables(segs)
    X0 = (torch.randn(Pg, K, generator=g) * 2).half().to(dev)
    X1 = (torch.randn(Pg, K, generator=g) * 2).half().to(dev)
    Z = torch.full((2 * 4 * Pg * N,), float("nan"), dtype=torch.float32, device=dev)
    Z8 = torch.full_like(Z, float("nan"))
    ext.grouped(X0, X1, ptr["gate_trellis"], ptr["up_trellis"], se, s0, sr, ns, Z, 2, K, N, Pg, 4, 4, 4, len(segs) + 2)
    ext.grouped(X0, X1, ptr["gate_trellis"], ptr["up_trellis"], se, s0, sr, ns, Z8, 2, K, N, Pg, 4, 8, 4, len(segs) + 2)
    torch.cuda.synchronize()
    rows = torch.zeros(Pg, dtype=torch.bool)
    for (e, r0, rn) in segs:
        rows[r0:r0 + rn] = True
    same_nt = torch.equal(Z.view(2, 4, Pg, N)[:, :, rows.to(dev)], Z8.view(2, 4, Pg, N)[:, :, rows.to(dev)])
    print(f"E-U2 grouped gate/up nt=4 (production config) bit-identical to TensorFold's nt=8: {same_nt}")
    ck(same_nt, "E-U2 nt=4 vs nt=8 differ")
    Zs = Z.view(2, 4, Pg, N).double().sum(1).cpu()
    worst = 0.0
    for (e, r0, rn) in segs:
        for mat, X in ((0, X0), (1, X1)):
            wq = R.unpack(W.w13_trellis[e, mat].cpu(), 4).double()
            ref = X[r0:r0 + rn].double().cpu() @ wq
            rel = float((Zs[mat, r0:r0 + rn] - ref).norm() / ref.norm())
            worst = max(worst, rel)
    print(f"E-U2 grouped gate/up (ptr tables, interleaved w13, e in {{0,{NEXP - 1}}}): worst rel {worst:.2e} (tol 1e-5)")
    ck(worst <= 1e-5, f"E-U2 gate/up rel {worst:.2e} > 1e-5")

    Xd = (torch.randn(Pg, N, generator=g) * 2).half().to(dev)
    Zd = torch.full((Pg * K,), float("nan"), dtype=torch.float32, device=dev)
    ext.grouped(Xd, Xd, ptr["down_trellis"], ptr["down_trellis"], se, s0, sr, ns, Zd, 1, N, K, Pg, 1, 4, 4, len(segs) + 2)
    torch.cuda.synchronize()
    Zd = Zd.view(Pg, K).double().cpu()
    worst = 0.0
    for (e, r0, rn) in segs:
        wq = R.unpack(W.w2_trellis[e].cpu(), 4).double()
        ref = Xd[r0:r0 + rn].double().cpu() @ wq
        worst = max(worst, float((Zd[r0:r0 + rn] - ref).norm() / ref.norm()))
    print(f"E-U2 grouped down (ptr tables, w2 stacked): worst rel {worst:.2e} (tol 1e-5)")
    ck(worst <= 1e-5, f"E-U2 down rel {worst:.2e} > 1e-5")

    # ---------------- E-U2c every grouped-GEMV kernel variant writes bit-identical Z (docs/OPTIMIZATION.md) ------
    # gate/up SK = 2, 4, 8 (production's split counts: 8 at P <= 32, 4 up to P = 512, 2 above, table 6) and down
    # SK = 1 / 2 on the segments above (e in {0, n-1}, a 2-segment expert, partial rows); variant 1 = the master
    # kernel; every other variant (incl. the shipped 0) must equal it bit for bit on every row the segments own, and
    # leave the other rows untouched. Each launch is repeated with S_cap = 65543 > 65535 (seg tables padded; nseg
    # unchanged), where ORD 1 / 2 cannot put S_cap in gridDim.y / z and must fall back to ORD 0: same Z again.
    nv = int(ext.num_variants())
    v_saved = int(ext.variant())
    ok_var, bad = True, []
    S_big = 65543
    pad_big = lambda t: torch.cat([t, torch.full((S_big - t.numel(),), -1, dtype=t.dtype, device=dev)])
    se_b, s0_b, sr_b = pad_big(se), pad_big(s0), pad_big(sr)
    for kind, mats, KK, NN, A0, A1, t0, t1, SKs in (("gate/up", 2, K, N, X0, X1, "gate_trellis", "up_trellis", (2, 4, 8)),
                                                   ("down", 1, N, K, Xd, Xd, "down_trellis", "down_trellis", (1, 2))):
        for SKv in SKs:
            outs = {}
            for v in range(nv):
                ext.set_variant(v)
                for big in (False, True):
                    Zv = torch.full((mats * SKv * Pg * NN,), float("nan"), dtype=torch.float32, device=dev)
                    if big:
                        try:
                            ext.grouped(A0, A1, ptr[t0], ptr[t1], se_b, s0_b, sr_b, ns, Zv, mats, KK, NN, Pg, SKv, 4,
                                        4, S_big)
                        except RuntimeError as e:          # a variant whose fallback has no compiled instance
                            bad.append((kind, SKv, v, "S_cap 65543 raised: " + str(e).splitlines()[0][:80]))
                    else:
                        ext.grouped(A0, A1, ptr[t0], ptr[t1], se, s0, sr, ns, Zv, mats, KK, NN, Pg, SKv, 4, 4,
                                    len(segs) + 2)
                    outs[(v, big)] = Zv.view(mats * SKv, Pg, NN)
            torch.cuda.synchronize()
            ref = outs[(1, False)]
            for (v, big), o_ in outs.items():
                same = torch.equal(o_[:, rows.to(dev)], ref[:, rows.to(dev)]) and bool(
                    torch.isnan(o_[:, ~rows.to(dev)]).all())
                if not same:
                    ok_var = False
                    bad.append((kind, SKv, v, "S_cap 65543" if big else "S_cap 5"))
    ext.set_variant(v_saved)
    bad_v = sorted({b_[2] for b_ in bad})
    print(f"E-U2c grouped variants 0..{nv - 1} (gate/up SK 2, 4, 8; down SK 1, 2; S_cap 5 and 65543 > 65535) "
          f"bit-identical to the master kernel: {ok_var}" + (f"; failing variants {bad_v}, {len(bad)} cases, first "
                                                              f"{bad[:3]}" if bad else ""))
    ck(ok_var, f"E-U2c grouped variants {bad_v} differ from the master kernel ({len(bad)} cases, first {bad[:2]})")
    ck(int(ext.variant()) == 0, "E-U2c: shipped variant not restored")

    # ---------------- E-U7 whole call with S_cap > 65535 (the grouped grid's ORD 0 fallback) --------------------
    # moe_forward accepts P up to 2^20, so S_cap = min(P, n) + ceil(P / 16) can exceed 65535, the gridDim.y limit of
    # the shipped ORD 1 grid; the launch must then run ORD 0 (the check that it can runs before route_prep / rot_in
    # are enqueued). Small dims (K = 256, N = 128), n = 62000 experts whose pointer tables cycle over 4 real experts,
    # one route per token (token t -> expert t): P = 62000 -> S_cap = 65875. The whole call must equal, bit for bit,
    # the same tokens served by two calls of 31000 pairs (S_cap 32938: the ORD 1 grid), both at the same split
    # counts (P > 512). No atomic-order noise: each output row has one contribution.
    Ks, Ns, nsm, nbig = 256, 128, 4, 62000
    Ws = H.Weights(nsm, Ks, Ns, dev, seed=13)
    cyc = torch.arange(nbig) % nsm

    def ptab(t):                                        # t[e] for e < nsm -> int64 pointer table over nbig experts
        base = torch.tensor([int(t[e].data_ptr()) for e in range(nsm)], dtype=torch.int64)
        return base[cyc].to(dev)

    tabs7 = [ptab(Ws.w13_trellis[:, 0]), ptab(Ws.w13_suh[:, 0]), ptab(Ws.w13_svh[:, 0]),
             ptab(Ws.w13_trellis[:, 1]), ptab(Ws.w13_suh[:, 1]), ptab(Ws.w13_svh[:, 1]),
             ptab(Ws.w2_trellis), ptab(Ws.w2_suh), ptab(Ws.w2_svh)]
    x7 = torch.randn(nbig, Ks, generator=g).half().to(dev)
    w7 = (torch.rand(nbig, generator=g) * 0.5 + 0.25).half().to(dev)

    def call7(ta, tb):                                  # tokens ta..tb-1, token t -> expert t
        Pc = tb - ta
        cnt = torch.zeros(nbig + 1, dtype=torch.int64, device=dev)
        cnt[ta:tb] = 1
        S7 = int(ext.s_cap(Pc, nbig))
        out = torch.zeros(Pc, Ks, dtype=torch.float32, device=dev)
        nanh = lambda n_: torch.full((n_,), float("nan"), dtype=torch.float16, device=dev)
        i32 = lambda n_: torch.full((n_,), -9, dtype=torch.int32, device=dev)
        Z7 = torch.full((int(ext.z_need(Pc, Ks, Ns)),), float("nan"), dtype=torch.float32, device=dev)
        err = None
        try:
            ext.moe_forward(x7[ta:tb], out, cnt, torch.arange(Pc, dtype=torch.int64, device=dev), w7[ta:tb], *tabs7,
                            nanh(Pc * Ks), nanh(Pc * Ks), nanh(Pc * Ns), Z7, i32(Pc), i32(S7), i32(S7), i32(S7),
                            i32(1), 128, Ns, 10.0)
            torch.cuda.synchronize()
        except RuntimeError as e:
            err = str(e).splitlines()[0]
        return out, S7, err

    ext.set_variant(0)
    whole = call7(0, nbig)
    halves = [call7(0, nbig // 2), call7(nbig // 2, nbig)]
    sk_same = tuple(ext.split_counts(nbig, Ks, Ns)) == tuple(ext.split_counts(nbig // 2, Ks, Ns))
    ok7 = whole[2] is None and all(h[2] is None for h in halves)
    same7 = ok7 and torch.equal(whole[0], torch.cat([halves[0][0], halves[1][0]]))
    fin7 = ok7 and bool(torch.isfinite(whole[0]).all()) and float(whole[0].abs().max()) > 0
    print(f"E-U7 whole call, shipped variant, P = {nbig}: S_cap {whole[1]} (> 65535: ORD 0 fallback) ran: "
          f"{whole[2] is None} {whole[2] or ''}; == two calls at S_cap {halves[0][1]} (ORD 1) bit for bit: {same7}; "
          f"finite and non-zero: {fin7}; same split counts {tuple(ext.split_counts(nbig, Ks, Ns))}: {sk_same}")
    ck(whole[1] > 65535 and halves[0][1] <= 65535 and sk_same, "E-U7 setup does not straddle the S_cap limit")
    ck(ok7, f"E-U7 moe_forward raised: {whole[2] or [h[2] for h in halves]}")
    ck(same7 and fin7, "E-U7 whole call at S_cap > 65535 differs from the ORD 1 calls")
    del whole, halves

    # ---------------- E-U3 route_prep vs oracle ------------------------------------------------------------
    gr = torch.Generator().manual_seed(3)
    for n in (288, 1100):
        mism = 0
        cases = 0
        for it in range(1000):
            kind = it % 5
            Rcap = [1, 16, 128, 128, 3][kind]
            P = int(torch.randint(1, 1025, (1,), generator=gr))
            if kind == 4:
                c = torch.zeros(n + 1, dtype=torch.int64)
                c[n] = P                                                       # all sentinel
            else:
                ids = torch.randint(0, n + 1, (P,), generator=gr)             # n = sentinel bucket
                if kind == 2:                                                  # skew: a hot expert over cap
                    hot = int(torch.randint(0, n, (1,), generator=gr))
                    ids[: P // 2] = hot
                if kind == 3:                                                  # inconsistent: sum(c) > P (REQ-S)
                    ids = torch.cat([ids, torch.randint(0, n, (7,), generator=gr)])
                c = torch.bincount(ids, minlength=n + 1).to(torch.int64)
            c = c.to(dev)
            S_cap = tf.s_cap(P, n)
            pe_k = torch.full((P,), 7, dtype=torch.int32, device=dev)
            se_k = torch.full((S_cap,), -9, dtype=torch.int32, device=dev)
            s0_k = torch.full_like(se_k, -9)
            sr_k = torch.full_like(se_k, -9)
            ns_k = torch.full((1,), -9, dtype=torch.int32, device=dev)
            ext.route_prep(c, P, Rcap, pe_k, se_k, s0_k, sr_k, ns_k)
            pe_o, ns_o, se_o, s0_o, sr_o = oracle_route(c, P, Rcap, S_cap)
            nk = int(ns_k)
            same = (nk == ns_o and torch.equal(pe_k, pe_o) and torch.equal(se_k[:nk], se_o[:nk])
                    and torch.equal(s0_k[:nk], s0_o[:nk]) and torch.equal(sr_k[:nk], sr_o[:nk]))
            cases += 1
            if not same:
                mism += 1
                if mism <= 3:
                    print(f"  mismatch n={n} it={it} kind={kind} P={P} R={Rcap} nseg k/o {nk}/{ns_o}")
        print(f"E-U3 route_prep vs oracle n={n}: {cases - mism}/{cases} identical")
        ck(mism == 0, f"E-U3 route_prep mismatches n={n}: {mism}")

    # ---------------- E-U4 gateup_epilogue vs numpy fp16 emulation -----------------------------------------
    Pe = 64
    pe = torch.randint(0, NEXP, (Pe,), generator=g).int().to(dev)
    # Z so that g0/u0 ~ N(0, s^2) with s such that g,u after svh have std ~5 (clamp exercised at L=10)
    svh_scale = float(W.w13_svh.float().abs().mean())
    zs = 5.0 / svh_scale / 2.0
    # SK: production's gate/up splits (table 6): 8 at P <= 32, 4 up to P = 512, 2 above
    for L, SKq in ((10.0, 4), (1.0, 4), (0.0, 4), (10.0, 8), (0.0, 8), (10.0, 2), (0.0, 2)):
        Zq = (torch.randn(2, SKq, Pe, N, generator=g) * zs).float().to(dev)
        xd = torch.empty((Pe, N), dtype=torch.float16, device=dev)
        ext.gateup_epilogue(Zq.view(-1), pe, ptr["gate_svh"], ptr["up_svh"], ptr["down_suh"], xd, Pe, N, SKq, L)
        torch.cuda.synchronize()
        z = Zq.cpu().numpy()
        sg = z[0, 0]
        su = z[1, 0]
        for s in range(1, SKq):
            sg = (sg + z[0, s]).astype(np.float32)
            su = (su + z[1, s]).astype(np.float32)
        e_idx = pe.cpu().numpy()
        svg = W.w13_svh[:, 0].cpu().numpy()[e_idx].astype(np.float64)
        svu = W.w13_svh[:, 1].cpu().numpy()[e_idx].astype(np.float64)
        sud = W.w2_suh.cpu().numpy()[e_idx].astype(np.float64)
        g0 = f16(sg).astype(np.float32).reshape(Pe, N // 128, 128)
        u0 = f16(su).astype(np.float32).reshape(Pe, N // 128, 128)
        gh = f16((fwht_np(g0) * R_SCALE).astype(np.float32)).reshape(Pe, N).astype(np.float64)
        uh = f16((fwht_np(u0) * R_SCALE).astype(np.float32)).reshape(Pe, N).astype(np.float64)
        gh = f16(gh * svg).astype(np.float64)
        uh = f16(uh * svu).astype(np.float64)
        ex = f16(np.exp(-gh)).astype(np.float64)
        sm = f16(1.0 + ex).astype(np.float64)
        rc = f16(1.0 / sm).astype(np.float64)
        gs = f16(gh * rc).astype(np.float64)
        if L != 0.0:
            Lh = float(f16(L))
            uh = np.minimum(np.maximum(uh, -Lh), Lh)
            gs = np.minimum(gs, Lh)
        a = f16(gs * uh).astype(np.float64)
        a = f16(a * sud).astype(np.float32).reshape(Pe, N // 128, 128)
        ref = f16((fwht_np(a) * R_SCALE).astype(np.float32)).reshape(Pe, N)
        got = xd.cpu().numpy()
        gi = got.view(np.int16).astype(np.int32)
        ri = ref.view(np.int16).astype(np.int32)
        # ulp distance on the fp16 total order
        def ordered(i):
            return np.where(i < 0, -32768 - i, i)
        ulp = np.abs(ordered(gi) - ordered(ri))
        frac = float((ulp > 0).mean())
        print(f"E-U4 gateup_epilogue L={L} SK={SKq}: max ulp {int(ulp.max())}, differing {frac * 100:.3f}% "
              f"(|g|>L fraction {(np.abs(gh) > (L or 1e9)).mean() * 100:.1f}%)")
        ck(int(ulp.max()) <= 1 and frac <= 0.01, f"E-U4 L={L} SK={SKq}: max ulp {int(ulp.max())}, frac {frac:.4f}")

    # ---------------- E-U5 down_epilogue exactness (one pair per token) -------------------------------------
    Bd = 24
    tsd = torch.randperm(Bd, generator=g).to(dev)                   # a permutation: one pair per token
    ped = torch.randint(0, NEXP, (Bd,), generator=g).int().to(dev)
    ped[5] = -1
    wsd = (torch.rand(Bd, generator=g) * 2.5).half().to(dev)
    Zd = (torch.randn(Bd * K, generator=g) * 30).float().to(dev)
    out = torch.zeros(Bd, K, dtype=torch.float32, device=dev)
    ext.down_epilogue(Zd, ped, tsd, wsd, ptr["down_svh"], out, 1)
    torch.cuda.synchronize()
    zz = f16(Zd.cpu().numpy()).astype(np.float32).reshape(Bd, K // 128, 128)
    hz = fwht_np(zz).reshape(Bd, K)
    rs = (np.float32(0.088388347648) * wsd.cpu().numpy().astype(np.float32)).astype(np.float32)
    svd = W.w2_svh.cpu().numpy()[ped.cpu().numpy().clip(0)].astype(np.float32)
    val = ((hz * rs[:, None]).astype(np.float32) * svd).astype(np.float32)
    ref = np.zeros((Bd, K), dtype=np.float32)
    tsn = tsd.cpu().numpy()
    for j in range(Bd):
        if int(ped[j]) >= 0:
            ref[tsn[j]] = val[j]
    same = np.array_equal(out.cpu().numpy(), ref)
    print(f"E-U5 down_epilogue (single pair per token) bit-identical to emulation: {same}; "
          f"row of skipped pair stays 0: {bool((out[tsd[5]] == 0).all())}")
    ck(same and bool((out[tsd[5]] == 0).all()), "E-U5 down_epilogue mismatch")

    Zd2 = (torch.randn(2 * Bd * K, generator=g) * 30).float().to(dev)
    out2 = torch.zeros(Bd, K, dtype=torch.float32, device=dev)
    ext.down_epilogue(Zd2, ped, tsd, wsd, ptr["down_svh"], out2, 2)
    torch.cuda.synchronize()
    z2 = Zd2.cpu().numpy().reshape(2, Bd, K)
    zz2 = f16((z2[0] + z2[1]).astype(np.float32)).astype(np.float32).reshape(Bd, K // 128, 128)
    val2 = ((fwht_np(zz2).reshape(Bd, K) * rs[:, None]).astype(np.float32) * svd).astype(np.float32)
    ref2 = np.zeros((Bd, K), dtype=np.float32)
    for j in range(Bd):
        if int(ped[j]) >= 0:
            ref2[tsn[j]] = val2[j]
    same2 = np.array_equal(out2.cpu().numpy(), ref2)
    print(f"E-U5b down_epilogue with 2 K splits (T=1 production config) bit-identical to emulation: {same2}")
    ck(same2, "E-U5b down_epilogue SK=2 mismatch")

    # ---------------- tile configuration / scratch sizing come from the extension (TA-3) ------------------
    cfg_ok = (tf.GATEUP_CFG == tuple(ext.GATEUP_CFG) and tf.DOWN_CFG == tuple(ext.DOWN_CFG)
              and tf.DOWN_SMALL == tuple(ext.DOWN_SMALL))
    print(f"tile cfg: python {tf.GATEUP_CFG} {tf.DOWN_CFG} {tf.DOWN_SMALL} == extension "
          f"{tuple(ext.GATEUP_CFG)} {tuple(ext.DOWN_CFG)} {tuple(ext.DOWN_SMALL)}: {cfg_ok}")
    ck(cfg_ok, "python tile constants differ from the compiled extension's")
    info = tf.REG.get((0, ptr["gate_trellis"].data_ptr()))
    if info is None:   # make_layer ran without the build hook in this test: register for the scratch check
        ck(tf.register(ptr, layer._exl3_fused_temps, K, N, tf._regions_of_layer(layer), selftest=False),
           "register for the scratch check")
        info = tf.REG[(0, ptr["gate_trellis"].data_ptr())]
    sc = tf._SCRATCH[info.scratch_key]
    need = [int(ext.z_need(P_, K, N)) for P_ in range(1, info.P_cap + 1)]
    ok_all_p = max(need) <= sc.z.numel() and int(ext.z_need_max(info.P_cap, K, N)) == sc.z.numel()
    print(f"scratch Z {sc.z.numel()} floats >= z_need(P) for every P in 1..P_cap={info.P_cap} (max {max(need)}, "
          f"argmax P={need.index(max(need)) + 1}): {ok_all_p}")
    ck(ok_all_p, "scratch Z smaller than the extension's per-call requirement for some P <= P_cap")

    # ---------------- build: JIT == AOT ------------------------------------------------------------------
    import importlib

    ck(bool(tf.EXT_SOURCE) and tf.EXT_SOURCE.startswith("aot:"), f"AOT==JIT: active extension is {tf.EXT_SOURCE}, not AOT")
    aot = ext
    jit = importlib.import_module("tf_exl3_moe").load_ext("jit")
    tf.use_ext(aot)                                                  # keep AOT as the active one
    T = 8
    gx = torch.Generator().manual_seed(9)
    x2d = torch.randn(T, K, generator=gx).half().to(dev)
    ids = H.random_ids(T, NEXP, 8, gx, dev)
    w = H.random_weights(T, 8, gx, dev)
    args = H.capture_args(prod, xl, x2d, ids, w, layer, 10.0)
    tf.register(layer._exl3_ptrs, layer._exl3_fused_temps, K, N, tf._regions_of_layer(layer), selftest=False)
    p = tf._plan(args, require_ok=False)
    outs = []
    for m in (aot, jit):
        o = torch.zeros(T, K, dtype=torch.float32, device=dev)
        a = H.with_out(args, o)
        tf.use_ext(m)
        tf._launch_kernels(p, a)
        torch.cuda.synchronize()
        outs.append(o)
    tf.use_ext(aot)
    # atomics make the fp32 sum order vary; compare with a tight tolerance instead of bit equality
    rel = float((outs[0] - outs[1]).norm() / outs[0].norm())
    print(f"build: AOT ({aot.__file__}) vs JIT ({jit.__file__}) moe_forward rel {rel:.2e}; active again: {tf.EXT_SOURCE}")
    ck(rel <= 1e-6, f"AOT vs JIT differ: {rel:.2e}")
    ck(tf.EXT_SOURCE.startswith("aot:"), "AOT not restored as the active extension")

    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
