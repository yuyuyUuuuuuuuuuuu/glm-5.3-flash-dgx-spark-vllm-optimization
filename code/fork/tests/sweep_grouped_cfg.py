"""A13 exploration: grouped-GEMV launch configurations (n tiles a block, warps a block, K splits) for the gate/up
and down matrices at decode sizes, graph replay over 12 cold routing sets (3 layers x 4 routes), n=288,
K=4096, N=1024. Each config is checked for correctness (split-summed Z vs the default config, rel <= 1e-5)
before timing. Informational; exits non-zero only on a correctness failure."""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LAYERS, ROUTES = 3, 4


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    xl = H.load_xl()
    tf = H.load_tf()
    ext = tf.load_ext()
    ck = H.Checks()
    dev = torch.device("cuda", 0)
    layers = [H.make_layer(prod, H.Weights(NEXP, K, N, dev, seed=700 + i)) for i in range(LAYERS)]
    P_cap = 512
    xg = torch.randn(P_cap, K, device=dev).half()
    xu = torch.randn(P_cap, K, device=dev).half()
    xd = torch.randn(P_cap, N, device=dev).half()
    Z = torch.empty(16 * 2 * P_cap * N, device=dev)

    def graph_of(fns):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in fns:
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for f in fns:
                f()
        return gr

    def timed(gr, calls, reps=10):
        gr.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / (reps * calls)

    import os

    gu_cfgs = [(8, 4, 4), (4, 4, 4), (4, 8, 4), (4, 4, 8), (8, 4, 8), (2, 4, 8)]
    dn_cfgs = [(8, 4, 1), (4, 4, 1), (4, 8, 1), (4, 4, 2), (8, 4, 2), (2, 4, 2)]
    # optional narrowing (docs/OPTIMIZATION.md side sweeps): SWEEP_T="1,2", SWEEP_GU="4,4,4;4,4,8", SWEEP_DN=...
    parse = lambda v: [tuple(int(x) for x in c.split(",")) for c in v.split(";")]
    if os.environ.get("SWEEP_GU"):
        gu_cfgs = parse(os.environ["SWEEP_GU"])
    if os.environ.get("SWEEP_DN"):
        dn_cfgs = parse(os.environ["SWEEP_DN"])
    ROUNDS = int(os.environ.get("SWEEP_ROUNDS", "5"))
    Ts = tuple(int(v) for v in os.environ.get("SWEEP_T", "1,2,4,8,16,32,64").split(","))
    for T in Ts:
        g = torch.Generator().manual_seed(T)
        sets = []
        for L in layers:
            for _ in range(ROUTES):
                ids = H.random_ids(T, NEXP, TOPK, g, dev)
                ec, ts, ws = tf.production_routing(ids, torch.ones(T, TOPK, device=dev), NEXP)
                P = ts.numel()
                S = tf.s_cap(P, NEXP)
                t = {k: torch.empty(n_, dtype=torch.int32, device=dev) for k, n_ in
                     (("pe", P), ("se", S), ("s0", S), ("sr", S), ("ns", 1))}
                ext.route_prep(ec, P, 128, t["pe"], t["se"], t["s0"], t["sr"], t["ns"])
                sets.append((L._exl3_ptrs, P, S, t))
        mib = sum(int(torch.unique(st[3]["se"][: int(st[3]["ns"])]).numel()) for st in sets) / len(sets) * 2 * 2
        for kind, cfgs, mats, KK, NN, X0, X1, tk0, tk1, mb in (
                ("gate/up", gu_cfgs, 2, K, N, xg, xu, "gate_trellis", "up_trellis", mib),
                ("down", dn_cfgs, 1, N, K, xd, xd, "down_trellis", "down_trellis", mib / 2)):
            graphs, ref = {}, None
            for (nt, w, sk) in cfgs:
                if KK % (16 * sk * w) or NN % (16 * nt):
                    continue
                fns = [lambda p=p, P=P, S=S, t=t, nt=nt, w=w, sk=sk: ext.grouped(X0, X1, p[tk0], p[tk1], t["se"], t["s0"],
                       t["sr"], t["ns"], Z, mats, KK, NN, P, sk, nt, w, S) for (p, P, S, t) in sets]
                fns[-1]()
                torch.cuda.synchronize()
                P = sets[-1][1]
                zs = Z[: mats * sk * P * NN].view(mats, sk, P, NN).sum(1)
                if ref is None:
                    ref = zs.clone()
                rel = float((zs - ref).norm() / ref.norm())
                ck(rel <= 1e-5, f"{kind} cfg {(nt, w, sk)} T={T}: rel {rel:.2e}")
                graphs[(nt, w, sk)] = (graph_of(fns), len(fns))
            times = {c: [] for c in graphs}
            for _ in range(ROUNDS):
                for c, (gr, n_) in graphs.items():
                    times[c].append(timed(gr, n_, reps=4))
            res = []
            for c, v in times.items():
                us = sorted(v)[ROUNDS // 2]
                res.append(f"{c[0]},{c[1]},{c[2]}:{us:7.1f}us/{mb * 2**20 / (us * 1e-6) / 1e9:5.1f}")
            print(f"T={T:2d} {kind:7s} ({mb:6.1f} MiB/call, median of {ROUNDS} rounds, us / GB/s) " + " ".join(res), flush=True)
            del graphs
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
