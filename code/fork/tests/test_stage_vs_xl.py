"""E-U6: TF stages vs the production exllamav3_ext.exl3_moe BINARY on the binary's own intermediates.

Why: E.1's global tolerance (2e-3) cannot tell PAR numerics from NATIVE ones (NATIVE is ~7e-4 from XL, well
inside it), and E-U4/E-U5 compare against numpy emulations written from the XL *sources* (G4: the binary is not
proven identical to them). This test compares each TF stage against what the XL binary itself computed.

How: exl3_moe with concurrency-1 temps runs every expert in one kernel group and leaves the LAST processed
expert's intermediates in the temps (docs/ref/xl_exl3_moe_kernel.cuh:146-219):
    temp_state_u[0, :c]        = xu  (input Hadamard of x * suh_u, fp16)
    temp_intermediate_u[0, :c] = u0  (up GEMM output, fp16)
    temp_intermediate_g[0, :c] = xd  (gate/up epilogue output = down GEMM input, fp16; written over g0)
    temp_state_g[0, :c]        = d0  (down GEMM output, fp16; written over xg)
One call routes T tokens to a single expert e* (every other route is a sentinel), so rows 0..T-1 of the temps are
pairs j = 0..T-1 of token_sorted, and every output row has exactly one contribution (no atomic-order noise).
The up pointer tables are pointed at the GATE trellis/suh (svh stay distinct): XL's g0 is then the same
deterministic computation as its u0, so the surviving u0 is also XL's g0.

  (a) rot_in                        TF xg, xu            == XL xu                  (bit-identical)
  (b) gateup_epilogue (SK 2, 4, 8)  TF xd from Z = XL u0 == XL xd                  (bit-identical, L in {10, 1, 0})
  (c) down_epilogue (SK 1 and 2)    TF out from Z = XL d0 == XL out                (bit-identical, L in {10, 1, 0})
  (d) the same (b)/(c) with the NATIVE build (TF_PARITY=0) must NOT pass (b)/(c): the test discriminates.
  (e) GEMM: TF up GEMM (fp32 split sums, rounded to fp16) vs XL u0 — the only remaining TF-vs-XL difference
      source (XL rounds a cross-CTA split-K partial to fp16); reported, bounded by the E.1 tolerance.
Z for (b)/(c) is XL's fp16 value plus a sub-half-ulp perturbation (v * 2^-12): PAR rounds the split sums to fp16
(A8-e / A5) and must land exactly on XL's value; a build without that rounding does not.
"""
from __future__ import annotations

import os

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 32, 8
E_STAR, T = 5, 24                     # 24 rows: XL's m loop runs a full 16-row tile and a partial 8-row tile
Ls = (10.0, 1.0, 0.0)


def ulp_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in fp16 ulps (ordered bit patterns; +0 == -0)."""
    def ordered(t):
        bits = t.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return (ordered(a) - ordered(b)).abs()


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    ext = tf.load_ext()
    ck = H.Checks()
    dev = torch.device("cuda", 0)
    orig = H.prod_orig(xl)
    W = H.Weights(NEXP, K, N, dev, seed=606)
    layer = H.make_layer(prod, W)
    keys = ("gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
            "down_trellis", "down_suh", "down_svh")
    tabs = [layer._exl3_ptrs[k] for k in keys]
    tabs[3], tabs[4] = tabs[0], tabs[1]                   # up := gate (trellis, suh); svh stay distinct
    SK_GU = tf.GATEUP_CFG[2]
    g = torch.Generator().manual_seed(66)
    x = torch.randn(T, K, generator=g).half().to(dev)
    ids = torch.full((T, TOPK), -1, dtype=torch.long)
    ids[:, 0] = E_STAR
    ids = ids.to(dev)
    w = H.random_weights(T, TOPK, g, dev)
    ec, ts, ws = tf.production_routing(ids, w, NEXP)
    P = ts.numel()
    R = int(layer._exl3_fused_temps[0].shape[1])
    ck(int(ec[E_STAR]) == T and int(ec[:NEXP].sum()) == T, "routing: exactly T pairs on e*")

    # TF route tables for this call
    S = int(ext.s_cap(P, NEXP))
    i32 = dict(dtype=torch.int32, device=dev)
    pe, se, s0, sr, ns = (torch.empty(P, **i32), torch.empty(S, **i32), torch.empty(S, **i32), torch.empty(S, **i32),
                          torch.empty(1, **i32))
    ext.route_prep(ec, P, R, pe, se, s0, sr, ns)
    torch.cuda.synchronize()
    ck(bool((pe[:T] == E_STAR).all()) and bool((pe[T:] == -1).all()), "route_prep: rows 0..T-1 are e*, rest -1")

    def xl_run(L):
        temps = tuple(torch.full((1,) + tuple(t.shape[1:]), float("nan"), dtype=t.dtype, device=dev)
                      for t in layer._exl3_fused_temps)
        out = torch.zeros(T, K, dtype=torch.float32, device=dev)
        args = (x, out, ec, ts, ws, *temps, 0, 4, 4, 4, *tabs, True, False, True, False, True, False, float(L))
        orig(*args)
        torch.cuda.synchronize()
        return {"xu": temps[1][0, :T].clone(), "u0": temps[3][0, :T].clone(), "xd": temps[2][0, :T].clone(),
                "d0": temps[0][0, :T].clone(), "out": out}

    ref = {L: xl_run(L) for L in Ls}
    again = xl_run(10.0)
    det = all(torch.equal(again[k], ref[10.0][k]) for k in ("xu", "u0", "xd", "d0", "out"))
    print(f"XL (C=1) deterministic across two identical calls (temps and out): {det}")
    ck(det, "XL concurrency-1 run not deterministic: the stage comparison premise fails")
    fin = all(bool(torch.isfinite(ref[L][k]).all()) for L in Ls for k in ("xu", "u0", "xd", "d0"))
    ck(fin, "XL temps rows 0..T-1 not all written (NaN left)")

    # (a) rot_in
    xg = torch.full((P, K), float("nan"), dtype=torch.float16, device=dev)
    xu = torch.full_like(xg, float("nan"))
    ext.rot_in(x, ts, pe, tabs[1], tabs[4], xg, xu)
    torch.cuda.synchronize()
    a_ok = torch.equal(xu[:T], ref[10.0]["xu"]) and torch.equal(xg[:T], ref[10.0]["xu"])
    print(f"(a) rot_in vs XL xu (in-kernel, {T} rows x {K}): bit-identical {a_ok}")
    ck(a_ok, "(a) rot_in differs from XL's own input Hadamard")

    # (e) GEMM-level difference (the source of the E.1 noise)
    nt, wp = tf.GATEUP_CFG[0], tf.GATEUP_CFG[1]
    Z = torch.full((2 * SK_GU * P * N,), float("nan"), dtype=torch.float32, device=dev)
    ext.grouped(xg, xu, tabs[0], tabs[3], se, s0, sr, ns, Z, 2, K, N, P, SK_GU, nt, wp, S)
    torch.cuda.synchronize()
    Zv = Z.view(2, SK_GU, P, N)
    acc = [Zv[m_, 0, :T].clone() for m_ in (0, 1)]
    for s in range(1, SK_GU):                            # the kernel's fixed fp32 order ((z0 + z1) + z2) + z3
        acc = [acc[m_] + Zv[m_, s, :T] for m_ in (0, 1)]
    g0_tf, u0_tf = acc[0].half(), acc[1].half()
    u0_xl = ref[10.0]["u0"]
    ud = ulp_dist(u0_tf, u0_xl)
    rel_u0 = float((u0_tf.float() - u0_xl.float()).norm() / u0_xl.float().norm())
    frac_u0 = float((ud > 0).float().mean())
    print(f"(e) up GEMM TF (fp32 split sums -> fp16) vs XL u0: {frac_u0 * 100:.2f}% of {u0_xl.numel()} values differ, "
          f"max {int(ud.max())} ulp, >1 ulp {float((ud > 1).float().mean()) * 100:.3f}%, rel_l2 {rel_u0:.2e}")
    ck(torch.equal(g0_tf, u0_tf), "(e) TF gate and up GEMM differ on identical inputs")
    ck(rel_u0 <= tf.E1_GLOBAL_TOL, f"(e) GEMM-level TF vs XL rel_l2 {rel_u0:.2e} > E.1 tolerance")

    def epilogues(m, label):
        """(b) and (c) with extension m; returns {case: (identical, frac differing, max ulp / max rel)}."""
        res = {}
        for L in Ls:
            r = ref[L]
            zin = r["u0"].float()
            zin = zin + zin * 2.0 ** -12                   # < 1/2 ulp: PAR's fp16 rounding must remove it
            for skg in (2, SK_GU, 8):                      # production's gate/up splits (table 6): 8 at P <= 32, 2 at P > 512
                Zb = torch.zeros(2, skg, P, N, dtype=torch.float32, device=dev)
                Zb[0, 0, :T] = zin
                Zb[1, 0, :T] = zin
                xd = torch.full((P, N), float("nan"), dtype=torch.float16, device=dev)
                m.gateup_epilogue(Zb.view(-1), pe, tabs[2], tabs[5], tabs[7], xd, P, N, skg, L)
                torch.cuda.synchronize()
                d = ulp_dist(xd[:T], r["xd"])
                res[f"xd L={L:g} SK={skg}"] = (torch.equal(xd[:T], r["xd"]), float((d > 0).float().mean()),
                                              int(d.max()))
            for sk in (1, 2):
                zd = r["d0"].float()
                zd = zd + zd * 2.0 ** -12
                Zd = torch.zeros(sk, P, K, dtype=torch.float32, device=dev)
                Zd[0, :T] = zd
                out = torch.zeros(T, K, dtype=torch.float32, device=dev)
                m.down_epilogue(Zd.view(-1), pe, ts, ws, tabs[8], out, sk)
                torch.cuda.synchronize()
                diff = (out != r["out"])
                rel = float((out - r["out"]).norm() / r["out"].norm())
                res[f"out L={L:g} SK={sk}"] = (torch.equal(out, r["out"]), float(diff.float().mean()), rel)
        for k, v in res.items():
            unit = "ulp" if k.startswith("xd") else "rel"
            print(f"  {label:6s} {k:16s} identical to XL {str(v[0]):5s}  differing {v[1] * 100:6.2f}%  max "
                  f"{v[2]:.3g} {unit}")
        return res

    print("(b)/(c) PAR (the shipped AOT build) vs XL on XL's own intermediates:")
    par = epilogues(ext, "PAR")
    for k, v in par.items():
        ck(v[0], f"PAR {k}: not bit-identical to XL ({v[1] * 100:.2f}% differ, max {v[2]})")

    # (d) NATIVE control: the same comparison must reject it
    os.environ["TF_PARITY"] = "0"
    try:
        nat = tf.load_ext("jit")
    finally:
        os.environ.pop("TF_PARITY", None)
        tf.use_ext(ext)
    ck(nat.parity() == 0, "NATIVE build reports parity != 0")
    print(f"(d) NATIVE ({nat.__file__}) on the same inputs:")
    natr = epilogues(nat, "NATIVE")
    rejected = all(not v[0] for v in natr.values())
    print(f"(d) every NATIVE case rejected by the bit-identity criterion: {rejected}")
    ck(rejected, "the stage comparison does not discriminate PAR from NATIVE")
    ck(tf.EXT_SOURCE.startswith("aot:"), "AOT not restored as the active extension")
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
