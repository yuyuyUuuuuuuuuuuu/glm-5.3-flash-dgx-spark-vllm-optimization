"""P2 (docs/OPTIMIZATION.md): what the fp32 split-K partials Z and the fp16 intermediates cost inside the pipeline.

Per configuration: 12 production-argument sets (3 layers x 4 routings, n = 288, K = 4096, N = 1024, topk = 8, cold
weights), every set with its own route tables, all sharing the TF scratch (as production does). CUDA graphs of the
12 sets running only the listed stages: [gu], [gu, gue], [gue], [down], [down, de], [de], [rot_in], [rot_in, gu];
alternating rounds (order flipped every round), median per-call us.
  gue in the pipeline   = t[gu, gue] - t[gu];  excess = that - t[gue] (stage-alone gue re-reads a Z hot in L2)
  de  in the pipeline   = t[down, de] - t[down];  excess = that - t[de]
  rot_in -> gu          = t[rot_in, gu] - t[gu] - t[rot_in]
Go for the split-K fixup fusion track (F1/F2/F3/C) if the gue + de excess is >= 1% of the call at T = 32 or 64.
C2 sweep: [gu(SK), gue(SK)] for SK in {4, 2, 1} at T >= 16 (xd vs SK = 4, rel <= 1e-5).
"""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LAYERS, ROUTES, ROUNDS = 3, 4, 9
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
    layers = [H.make_layer(prod, H.Weights(NEXP, K, N, dev, seed=500 + i)) for i in range(LAYERS)]

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
            for f in fns:
                f()
        return gr

    def timed(gr, calls, reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / (reps * calls)

    med = lambda v: sorted(v)[len(v) // 2]
    PER_EXPERT = 3 * (K // 16) * (N // 16) * 128
    for kind, T in (("rand", 8), ("rand", 16), ("rand", 32), ("rand", 64), ("rand", 128), ("corr40", 16),
                    ("corr40", 32), ("corr40", 64)):
        g = torch.Generator().manual_seed(9000 + T + (0 if kind == "rand" else 31))
        sets, distinct = [], 0.0
        for layer in layers:
            for _ in range(ROUTES):
                x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
                ids = H.routing_ids(kind, T, NEXP, TOPK, g, dev)
                w = H.random_weights(T, TOPK, g, dev)
                a = H.capture_args(prod, xl, x, ids, w, layer, LIMIT)
                sets.append(H.with_out(a, torch.zeros(T, K, dtype=torch.float32, device=dev)))
                distinct += int(torch.unique(ids).numel())
        distinct /= len(sets)
        plans = [(a, tf._plan(a, require_ok=False)) for a in sets]
        for a, p in plans:
            tf._launch_kernels(p, a)                          # valid activations in the shared scratch
        sc = plans[0][1].scratch
        tabs = []
        for a, p in plans:
            S = tf.s_cap(p.P, p.n)
            t = {"pe": torch.empty(p.P, dtype=torch.int32, device=dev), "se": torch.empty(S, dtype=torch.int32, device=dev),
                 "s0": torch.empty(S, dtype=torch.int32, device=dev), "sr": torch.empty(S, dtype=torch.int32, device=dev),
                 "ns": torch.empty(1, dtype=torch.int32, device=dev), "S": S}
            ext.route_prep(a[2], p.P, p.R, t["pe"], t["se"], t["s0"], t["sr"], t["ns"])
            tabs.append(t)
        dsk = lambda P: int(ext.split_counts(P, K, N)[1])
        GU_NT, GU_W, _ = tf.GATEUP_CFG
        GU_SK = tf.GATEUP_CFG[2]                             # C2 sweep baseline (explicit SK below)

        def stage(name, a, p, t, sk=GU_SK):
            if name == "rot":
                return lambda: ext.rot_in(a[0], a[3], t["pe"], a[14], a[17], sc.xg, sc.xu)
            if name == "gu":
                return lambda: ext.grouped(sc.xg, sc.xu, a[13], a[16], t["se"], t["s0"], t["sr"], t["ns"], sc.z, 2, K, N,
                                           p.P, sk, GU_NT, GU_W, t["S"])
            if name == "gue":
                return lambda: ext.gateup_epilogue(sc.z, t["pe"], a[15], a[18], a[20], sc.xd, p.P, N, sk, LIMIT)
            if name == "down":
                return lambda: ext.grouped(sc.xd, sc.xd, a[19], a[19], t["se"], t["s0"], t["sr"], t["ns"], sc.z, 1, N, K,
                                           p.P, dsk(p.P), tf.DOWN_CFG[0], tf.DOWN_CFG[1], t["S"])
            if name == "de":
                return lambda: ext.down_epilogue(sc.z, t["pe"], a[3], a[4], a[21], a[1], dsk(p.P))
            raise KeyError(name)

        combos = {"gu": ("gu",), "gu+gue": ("gu", "gue"), "gue": ("gue",), "down": ("down",),
                  "down+de": ("down", "de"), "de": ("de",), "rot": ("rot",), "rot+gu": ("rot", "gu"),
                  "full": ("rot", "gu", "gue", "down", "de")}
        graphs = {}
        for cname, st in combos.items():
            fns = []
            for (a, p), t in zip(plans, tabs):
                for s_ in st:
                    fns.append(stage(s_, a, p, t))
            graphs[cname] = graph_of(fns)
        for gr in graphs.values():
            gr.replay()
        torch.cuda.synchronize()
        t1 = timed(graphs["full"], len(sets), 1)
        reps = max(2, int(20000 / (t1 * len(sets))))
        times = {c: [] for c in graphs}
        names = list(graphs)
        for r in range(ROUNDS):
            for c in (names if r % 2 == 0 else list(reversed(names))):
                times[c].append(timed(graphs[c], len(sets), reps))
        m = {c: med(v) for c, v in times.items()}
        pd = lambda a, b: med([x - y for x, y in zip(times[a], times[b])])
        gue_in = pd("gu+gue", "gu")
        de_in = pd("down+de", "down")
        rot_gu = med([x - y - z for x, y, z in zip(times["rot+gu"], times["gu"], times["rot"])])
        ex = (gue_in - m["gue"]) + (de_in - m["de"])
        print(f"{kind:6s} T={T:3d} distinct {distinct:5.1f} | full {m['full']:8.1f} us | gu {m['gu']:.1f} gu+gue "
              f"{m['gu+gue']:.1f} gue-alone {m['gue']:.1f} -> gue in pipeline {gue_in:.1f} (excess {gue_in - m['gue']:+.1f})"
              f" | down {m['down']:.1f} down+de {m['down+de']:.1f} de-alone {m['de']:.1f} -> de in pipeline {de_in:.1f} "
              f"(excess {de_in - m['de']:+.1f}) | rot {m['rot']:.1f} rot->gu excess {rot_gu:+.1f} | gue+de excess "
              f"{ex:+.1f} us = {ex / m['full'] * 100:+.2f}% of the call | spread full "
              f"{(max(times['full']) - min(times['full'])) / m['full'] * 100:.1f}%", flush=True)
        del graphs
        # C2: gate/up with SK in {4, 2, 1} (+ its epilogue), correctness of xd vs SK = 4
        if T >= 16:
            res = {}
            xd_ref = None
            gr_sk = {}
            for sk in (4, 2, 1):
                a, p = plans[-1]
                t = tabs[-1]
                stage("gu", a, p, t, sk)()
                stage("gue", a, p, t, sk)()
                torch.cuda.synchronize()
                xd = sc.xd[:p.P].clone()
                if xd_ref is None:
                    xd_ref = xd
                valid = t["pe"] >= 0
                rel = float((xd[valid].float() - xd_ref[valid].float()).norm() / xd_ref[valid].float().norm())
                ck(rel <= 1e-3, f"C2 SK={sk}: xd rel {rel:.2e} vs SK=4")
                fns = []
                for (a, p), t in zip(plans, tabs):
                    fns += [stage("gu", a, p, t, sk), stage("gue", a, p, t, sk)]
                gr_sk[sk] = graph_of(fns)
                res[sk] = rel
            tt = {sk: [] for sk in gr_sk}
            for r in range(ROUNDS):
                for sk in ((4, 2, 1) if r % 2 == 0 else (1, 2, 4)):
                    tt[sk].append(timed(gr_sk[sk], len(sets), reps))
            print(f"       C2 [gu(SK), gue(SK)]: " + " ".join(
                f"SK={sk} {med(tt[sk]):.1f} us (ratio vs SK=4 {med([x / y for x, y in zip(tt[sk], tt[4])]):.4f}, xd rel "
                f"{res[sk]:.1e})" for sk in (4, 2, 1)), flush=True)
            del gr_sk
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
