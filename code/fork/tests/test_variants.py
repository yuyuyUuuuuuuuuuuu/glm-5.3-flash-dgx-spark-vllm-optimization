"""Grouped-GEMV kernel variants in the full pipeline (docs/OPTIMIZATION.md): bit-identical outputs.

Variant ids of the extension (0 = shipped, 1 = the master kernel, 2.. experiments) that share a split-count table
perform the same operations in the same order, so on routing with one route per token (no atomic-order noise in
down_epilogue) the production apply's output must be bitwise equal among them. For T in {1, 8, 64, 128}: the hooked
production apply (K2 path) captured once per variant into a CUDA graph, then 1000 replays in total with fresh
x / ids / weights copied into the static inputs; after each replay every variant's output must equal (torch.equal)
the first variant of its split-count table, and every table's output must pass E.1 against variant 1's.

Above P = 512 (reached only with several routes per token, where the atomics make the apply output order-dependent)
the shipped split-count table 6 runs 2 gate/up splits, and the K3b epilogues drop 2*SK*4 Z lines per block. That
branch is checked on the bare forward path (moe_forward) with one route per token and B = P tokens (P = 513, 520,
640, 768, 1024): every scratch buffer is poisoned with NaN and flushed out of L2 before each call, so a line dropped
while still live is re-read from DRAM as NaN (the gate/up partials, 16 KiB per pair, are mostly L2-resident when the
epilogue runs, so a wrong discard drops a dirty line); the shipped variant 0 must equal variant 12 (table 6 without
the discards) bit for bit, twice; variants 15 (table 1: gate/up 4 splits) and 1 (master) must agree within E.1.
"""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP = 4096, 1024, 288
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
    integrate.install(prodmod=prod, ext=xl, force=True)
    tf.CFG.strict = True
    layer = H.make_layer(prod, H.Weights(NEXP, K, N, dev, seed=91))
    nv = int(ext.num_variants())
    g = torch.Generator().manual_seed(17)

    def fresh(T):
        ids = torch.stack([torch.randperm(NEXP, generator=g)[:1] for _ in range(T)])
        if T > 1:
            ids[int(torch.randint(0, T, (1,), generator=g))] = -1          # a sentinel route
        return (torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev), ids.to(dev), H.random_weights(T, 1, g, dev))

    Ts = (1, 8, 64, 128)
    static = {T: fresh(T) for T in Ts}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for T in Ts:
            prod.apply_exl3_fused_moe(*static[T], layer, layer._exl3_inners, None, LIMIT)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graphs, outs = {}, {}
    for T in Ts:
        for v in range(nv):
            ext.set_variant(v)
            gr = torch.cuda.CUDAGraph()
            c0 = tf.COUNTERS["tf_apply_calls"]
            with torch.cuda.graph(gr):
                outs[(T, v)] = prod.apply_exl3_fused_moe(*static[T], layer, layer._exl3_inners, None, LIMIT)
            ck(tf.COUNTERS["tf_apply_calls"] == c0 + 1, f"T={T} variant {v}: K2 not chosen at capture")
            graphs[(T, v)] = gr
    ext.set_variant(0)
    tab = {v: int(ext.variant_sk_table(v)) for v in range(nv)}
    first = {}
    for v in [1] + [u for u in range(nv) if u != 1]:
        first.setdefault(tab[v], v)
    bad, bad_e1, worst = {}, {}, 0.0
    reps = 0
    for i in range(1000 // len(Ts)):
        for T in Ts:
            for dst, src in zip(static[T], fresh(T)):
                dst.copy_(src)
            ref = {}
            for v in [1] + [u for u in range(nv) if u != 1]:
                graphs[(T, v)].replay()
                y = outs[(T, v)]
                if first[tab[v]] == v:
                    ref[tab[v]] = y.clone()
                    if v != 1:
                        m = tf.compare(y, ref[tab[1]])
                        worst = max(worst, m["rel_l2"])
                        if not tf.passes_e1(m):
                            bad_e1[(T, v)] = bad_e1.get((T, v), 0) + 1
                elif not torch.equal(y, ref[tab[v]]):
                    bad[(T, v)] = bad.get((T, v), 0) + 1
            reps += 1
    torch.cuda.synchronize()
    print(f"{reps} fresh-routing replays (T in {Ts}, one route per token), variants 0..{nv - 1} (split-count tables "
          f"{tab}): bit-identical within each table in every replay: {not bad} {bad if bad else ''}; tables vs "
          f"variant 1 within E.1: {not bad_e1} (worst rel_l2 {worst:.2e})")
    ck(not bad and not bad_e1 and reps == 1000, f"variant outputs differ: {bad} {bad_e1}")
    ck(int(ext.variant()) == 0, "shipped variant not restored")

    # ---- forward path above P = 512: table 6's gate/up SK 2 and its K3b discards, bitwise (docstring) ----
    tabs = [layer._exl3_ptrs[k] for k in tf._PTR_KEYS]
    R = int(layer._exl3_fused_temps[0].shape[1])
    flush = torch.empty((96 << 20) // 4, dtype=torch.float32, device=dev)     # 96 MiB > 4x the 24 MiB L2

    def fwd(v, x, cnt, ts, ws, P):
        ext.set_variant(v)
        out = torch.zeros(P, K, dtype=torch.float32, device=dev)
        S = int(ext.s_cap(P, NEXP))
        xg = torch.full((P * K,), float("nan"), dtype=torch.float16, device=dev)
        xu = torch.full_like(xg, float("nan"))
        xd = torch.full((P * N,), float("nan"), dtype=torch.float16, device=dev)
        Z = torch.full((int(ext.z_need(P, K, N)),), float("nan"), dtype=torch.float32, device=dev)
        i32 = lambda n_, v_: torch.full((n_,), v_, dtype=torch.int32, device=dev)
        flush.fill_(float(v))                          # the NaN lines go to DRAM: a stale read returns NaN
        ext.moe_forward(x, out, cnt, ts, ws, *tabs, xg, xu, xd, Z, i32(P, 7), i32(S, 123), i32(S, 0), i32(S, 0),
                        i32(1, 0), R, N, LIMIT)
        torch.cuda.synchronize()
        return out

    gp = torch.Generator().manual_seed(29)
    fwd_ok, fwd_worst, sks = True, 0.0, {}
    for P in (513, 520, 640, 768, 1024):
        ext.set_variant(0)
        sks[P] = tuple(int(v) for v in ext.split_counts(P, K, N))
        e = torch.randint(0, NEXP + 1, (P,), generator=gp)                    # NEXP = a sentinel route
        order = torch.argsort(e, stable=True)
        x = torch.randn(P, K, generator=gp).half().to(dev)
        ws = (torch.rand(P, generator=gp) * 0.5 + 0.25)[order].half().to(dev)
        cnt = torch.bincount(e, minlength=NEXP + 1).to(torch.int64).to(dev)
        ts = order.to(dev)
        o = {v: fwd(v, x, cnt, ts, ws, P) for v in (12, 0, 15, 1)}
        o0b = fwd(0, x, cnt, ts, ws, P)
        same = torch.equal(o[0], o[12]) and torch.equal(o[0], o0b) and bool(torch.isfinite(o[0]).all())
        e1 = {v: tf.compare(o[0], o[v]) for v in (15, 1)}
        fwd_worst = max([fwd_worst] + [m["rel_l2"] for m in e1.values()])
        ok = same and all(tf.passes_e1(m) for m in e1.values()) and sks[P] == (2, 1)
        print(f"forward P={P} (split counts {sks[P]}): variant 0 == variant 12 (no K3b) and == a repeat, finite: "
              f"{same}; vs table 1 (variant 15) rel_l2 {e1[15]['rel_l2']:.2e}, vs master (variant 1) "
              f"{e1[1]['rel_l2']:.2e} (E.1: {all(tf.passes_e1(m) for m in e1.values())})")
        fwd_ok &= ok
    ext.set_variant(0)
    ck(fwd_ok, f"forward path above P = 512 (gate/up SK 2 + K3b) not bitwise / E.1 (split counts {sks})")
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
