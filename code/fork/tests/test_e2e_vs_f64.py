"""E.2: TF and production exl3_moe vs the float64 reference of kernels/exl3_format_ref.py (docs/DESIGN.md §E.2).

Reference per routed pair (t, e, w16): g = x16[t] @ W_gate_e, u = x16[t] @ W_up_e (float64, the EXL3 layer
definition y = ((((x * suh) @ H) @ W_q) @ H) * svh), a = min(silu(g), L) * clip(u, -L, L) if L != 0 else
silu(g) * u (exl3_moe's post-SiLU clamp), d = a @ W_down_e, out[t] += float(w16) * d.
Criteria: e_tf <= 1.25 * e_xl + 2.5e-4 (TF may not be less accurate than production) and e_tf <= 5e-3.
Sizes: n = 16 experts, T in {1, 4, 16}, L in {10, 1, 0} (CPU float64 cost).
"""
from __future__ import annotations

import os

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 16, 8


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import exl3_format_ref as R
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    orig = rep["orig"]
    tf.CFG.strict = True
    W = H.Weights(NEXP, K, N, dev, seed=2024)
    layer = H.make_layer(prod, W)
    ck(tf.REG.get((0, layer._exl3_ptrs["gate_trellis"].data_ptr())) is not None, "layer not registered")
    g = torch.Generator().manual_seed(8)
    x = torch.randn(16, K, generator=g).half().to(dev)
    a = H.capture_args(prod, xl, x, H.random_ids(16, NEXP, TOPK, g, dev), H.random_weights(16, TOPK, g, dev), layer, 10.0)
    o = torch.zeros(16, K, dtype=torch.float32, device=dev)
    orig(*H.with_out(a, o))
    W.rescale_svh_d(1.0 / float(o.std()))

    # float64 per-expert weights, CPU
    print("unpacking 16 x 3 trellises (float64 reference) ...", flush=True)
    Wq = {}
    for e in range(NEXP):
        Wq[e] = (R.unpack(W.w13_trellis[e, 0].cpu(), 4).double(), R.unpack(W.w13_trellis[e, 1].cpu(), 4).double(),
                 R.unpack(W.w2_trellis[e].cpu(), 4).double())
    sc = {e: tuple(t.double().cpu() for t in (W.w13_suh[e, 0], W.w13_svh[e, 0], W.w13_suh[e, 1], W.w13_svh[e, 1],
                                              W.w2_suh[e], W.w2_svh[e])) for e in range(NEXP)}

    def lin(xv, wq, suh, svh):
        return R.rotate(R.rotate(xv * suh, -1) @ wq, -1) * svh

    def reference(x16, ids, w, L):
        xs = x16.double().cpu()
        w16 = w.to(torch.float16).double().cpu()
        ids = ids.cpu()
        out = torch.zeros(xs.shape[0], K, dtype=torch.float64)
        for t in range(xs.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0 or e >= NEXP:
                    continue
                wg, wu, wd = Wq[e]
                suh_g, svh_g, suh_u, svh_u, suh_d, svh_d = sc[e]
                gg = lin(xs[t:t + 1], wg, suh_g, svh_g)
                uu = lin(xs[t:t + 1], wu, suh_u, svh_u)
                s = gg * torch.sigmoid(gg)
                if L != 0.0:
                    s = torch.clamp(s, max=L)
                    uu = torch.clamp(uu, min=-L, max=L)
                d = lin(s * uu, wd, suh_d, svh_d)
                out[t] += w16[t, k] * d[0]
        return out

    worst = 0.0
    cases = []
    for T in (1, 4, 16):
        gT = torch.Generator().manual_seed(300 + T)
        x16 = torch.randn(T, K, generator=gT).half().to(dev)
        ids = H.random_ids(T, NEXP, TOPK, gT, dev)
        if T >= 4:
            ids[1, 2] = -1                                   # a sentinel route too
        w = H.random_weights(T, TOPK, gT, dev)
        for L in (10.0, 1.0, 0.0):
            args = H.capture_args(prod, xl, x16, ids, w, layer, L)
            out_xl = torch.zeros(T, K, dtype=torch.float32, device=dev)
            out_tf = torch.zeros_like(out_xl)
            orig(*H.with_out(args, out_xl))
            p = tf.plan(H.with_out(args, out_tf))
            ck(p is not None, "plan None")
            tf.launch(p, H.with_out(args, out_tf), None)
            torch.cuda.synchronize()
            ref = reference(x16, ids, w, L)
            rn = float(ref.norm())
            e_tf = float((out_tf.double().cpu() - ref).norm()) / rn
            e_xl = float((out_xl.double().cpu() - ref).norm()) / rn
            ok = e_tf <= 1.25 * e_xl + 2.5e-4 and e_tf <= 5e-3
            worst = max(worst, e_tf)
            cases.append((T, L, args, ref, e_tf))
            print(f"  T={T:2d} L={L:4.1f}: e_tf {e_tf:.3e}  e_xl {e_xl:.3e}  e_tf/e_xl {e_tf / e_xl:.3f}  "
                  f"(need e_tf <= {1.25 * e_xl + 2.5e-4:.3e} and <= 5e-3) {'ok' if ok else 'FAIL'}", flush=True)
            ck(ok, f"E.2 T={T} L={L}: e_tf {e_tf:.3e} e_xl {e_xl:.3e}")
    print(f"E.2 worst e_tf {worst:.3e}")

    # NATIVE build (TF_PARITY=0: fp32 epilogues, no fp16 roundings) must also be within the E.2 bound
    par_ext = tf._EXT
    os.environ["TF_PARITY"] = "0"
    try:
        nat = tf.load_ext("jit")
    finally:
        os.environ.pop("TF_PARITY", None)
    ck(nat.parity() == 0, "NATIVE build reports parity != 0")
    tf.use_ext(nat)
    worst_nat = 0.0
    for T, L, args, ref, e_tf in cases:
        out_n = torch.zeros(args[0].shape[0], K, dtype=torch.float32, device=dev)
        p = tf.plan(H.with_out(args, out_n))
        tf.launch(p, H.with_out(args, out_n), None)
        torch.cuda.synchronize()
        e_n = float((out_n.double().cpu() - ref).norm()) / float(ref.norm())
        worst_nat = max(worst_nat, e_n)
        print(f"  NATIVE T={T:2d} L={L:4.1f}: e_native {e_n:.3e} (PAR e_tf {e_tf:.3e})")
        ck(e_n <= 5e-3, f"NATIVE T={T} L={L}: {e_n:.3e}")
    tf.use_ext(par_ext)
    ck(tf.EXT_SOURCE.startswith("aot:"), "AOT not restored as the active extension")
    print(f"E.2 NATIVE ({nat.__file__}) worst e_native {worst_nat:.3e}")
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
