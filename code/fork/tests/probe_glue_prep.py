"""Probe: the decode MoE prelude, K2 (ids .to(long) + zeros(out) + route_ids + rot_in) vs glue_prep, alone and with
the shared expert's FP8 gate/up GEMV running concurrently on a second stream (as in production), 42 layers with
their own suh tables (cold), CUDA graphs, interleaved rounds. Prints per-layer us (graph time / 42)."""
from __future__ import annotations

import argparse

import torch

import harness as H
import moeglue_rig as MR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="1,5,8,16")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    dev = torch.device("cuda", 0)
    integrate.install(prodmod=prod, ext=xl, force=True)
    E = tf.load_ext()
    layers = MR.make_layers(prod, dev, 3, 42)
    rig = MR.Rig(prod, layers, dev)
    n = MR.NEXP
    for T in [int(t) for t in a.T.split(",")]:
        kind = "corr40" if T >= 5 else "rand"
        sets = MR.make_sets(kind, T, len(layers), 1, dev, seed=77 + T)
        P = T * MR.TOPK
        S = P + (P + 15) // 16
        i32 = lambda m: torch.zeros(m, dtype=torch.int32, device=dev)
        pe, se, sr0, srn, ns, inv = i32(P), i32(S), i32(S), i32(S), i32(1), i32(P)
        ts = torch.zeros(P, dtype=torch.int64, device=dev)
        ws = torch.zeros(P, dtype=torch.float16, device=dev)
        xg = torch.zeros(P, MR.K, dtype=torch.float16, device=dev)
        xu = torch.zeros_like(xg)
        R = int(layers[0]._exl3_fused_temps[0].shape[1])

        def k2pre(li, x, ids, w):
            L = layers[li]
            p = L._exl3_ptrs
            ids64 = ids.to(torch.long)
            torch.zeros(T, MR.K, dtype=torch.float32, device=dev)
            E.route_ids(ids64, w, None, n, R, pe, se, sr0, srn, ns, ts, ws)
            E.rot_in(x, ts, pe, p["gate_suh"], p["up_suh"], xg, xu)

        def glue(pf):
            def f(li, x, ids, w):
                p = layers[li]._exl3_ptrs
                E.glue_prep(x, ids, w, None, n, R, p["gate_suh"], p["up_suh"], xg, xu, pe, se, sr0, srn, ns, ts, ws,
                            inv, p["gate_svh"], p["up_svh"], p["down_suh"], p["down_svh"], MR.N, pf)
            return f

        variants = {"k2pre": k2pre, "glue": glue(False), "glue_pf": glue(True)}
        graphs = {}
        for name, fn in variants.items():
            for conc in (False, True):
                def call(s, fn=fn, conc=conc):
                    li, x, ids, w = s
                    main = torch.cuda.current_stream()
                    if conc:
                        rig.aux.wait_stream(main)
                        with torch.cuda.stream(rig.aux):
                            rig.fp8(x, rig.sh_gu[li], MR.SH_GU)
                    fn(li, x, ids, w)
                    if conc:
                        main.wait_stream(rig.aux)
                fns = [lambda s=s, call=call: call(s) for s in sets]
                graphs[f"{name}{'+fp8' if conc else ''}"], _ = MR.capture(fns, dev)
        res = MR.ab_rounds(graphs, len(sets), a.rounds, a.reps)
        base = res["k2pre"]
        print(f"{kind} T={T}: " + " | ".join(
            f"{k} {MR.med(v):.1f} us ({100 * MR.spread(v):.0f}%)" for k, v in res.items()), flush=True)
        del graphs
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
