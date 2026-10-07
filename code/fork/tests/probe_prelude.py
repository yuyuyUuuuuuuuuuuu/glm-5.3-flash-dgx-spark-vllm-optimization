"""P3 (docs/OPTIMIZATION.md): what production's decode apply costs around the exl3_moe call.

Production apply_exl3_fused_moe (the real module) is run with exllamav3_ext.exl3_moe replaced by a no-op that keeps
the original __doc__ (so the num_active detector answers the same), i.e. exactly the prelude: map_topk_to_local,
arange/repeat_interleave, weights -> fp16, argsort, the two gathers, expert_count zeros + scatter_add,
zeros(out), x2d.contiguous().half(). 12 calls (one layer, 12 routings) are captured into one CUDA graph and
replayed; per-call time = replay / 12, median of alternating rounds with an empty-graph control. The same routings
are then run through the full production apply with the TF dispatcher (graph) and the bare TF call (graph) for
the difference "apply - bare", which should match the prelude.
Go for K2 (apply-level fused routing) if the prelude costs >= 10 us per call.
"""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
CALLS, ROUNDS = 12, 9
LIMIT = 10.0


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    orig, disp = rep["orig"], xl.exl3_moe
    W = H.Weights(NEXP, K, N, dev, seed=77)
    layer = H.make_layer(prod, W)

    def noop(*args):
        return None

    noop.__doc__ = orig.__doc__

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

    def timed(gr, reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            gr.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / (reps * CALLS)

    med = lambda v: sorted(v)[len(v) // 2]
    for kind, T in (("rand", 1), ("corr40", 5), ("corr40", 8), ("corr40", 16), ("corr40", 32), ("rand", 8),
                    ("rand", 64)):
        g = torch.Generator().manual_seed(123 + T)
        calls = []
        for _ in range(CALLS):
            x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
            ids = H.routing_ids(kind, T, NEXP, TOPK, g, dev)
            w = H.random_weights(T, TOPK, g, dev)
            calls.append((x, ids, w))

        def apply_with(fn):
            def mk(x, ids, w):
                def f():
                    xl.exl3_moe = fn
                    try:
                        return prod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, None, LIMIT)
                    finally:
                        xl.exl3_moe = disp
                return f
            return [mk(*c) for c in calls]

        g_pre = graph_of(apply_with(noop))                    # prelude only
        g_app = graph_of(apply_with(disp))                    # full production apply, TF path
        args = [H.capture_args(prod, xl, x, ids, w, layer, LIMIT) for (x, ids, w) in calls]
        args = [H.with_out(a, torch.zeros(T, K, dtype=torch.float32, device=dev)) for a in args]
        g_bare = graph_of([lambda a=a: disp(*a) for a in args])
        g_nil = graph_of([lambda: None])                      # empty-graph control (replay launch overhead)
        for gr in (g_pre, g_app, g_bare, g_nil):
            gr.replay()
        torch.cuda.synchronize()
        reps = 20
        t = {"pre": [], "app": [], "bare": [], "nil": []}
        for r in range(ROUNDS):
            order = list(t.keys()) if r % 2 == 0 else list(reversed(t.keys()))
            for k in order:
                gr = {"pre": g_pre, "app": g_app, "bare": g_bare, "nil": g_nil}[k]
                t[k].append(timed(gr, reps))
        m = {k: med(v) for k, v in t.items()}
        diff = med([a - b for a, b in zip(t["app"], t["bare"])])
        print(f"{kind:6s} T={T:3d}: prelude-only graph {m['pre']:6.1f} us/call | full apply (TF) {m['app']:7.1f} | "
              f"bare TF {m['bare']:7.1f} | apply - bare {diff:6.1f} (paired median) | empty graph {m['nil'] * CALLS:5.1f} us/replay"
              f" | spread prelude {(max(t['pre']) - min(t['pre'])) / m['pre'] * 100:.1f}% | "
              f"prelude share of apply {m['pre'] / m['app'] * 100:.1f}%", flush=True)
        ck(m["pre"] > 0, "prelude timing")
        del g_pre, g_app, g_bare, g_nil
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
