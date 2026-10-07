"""opt-moe2-rev: independent same-process A/B of GLM53_MOE_E4M3_MAINLOOP (fused kernel only, real layer-10 experts,
real routing, TG on): shipped (2048) vs MAINLOOP (10240), bf16 and fp32 accumulators, plus an A/A control arm
(2048 timed as a separate arm) to size the noise of a paired delta on a shared GPU.
  warm: rounds of BENCH_N back-to-back calls per arm, arm order alternating (A B A' / A' B A), per-round paired deltas
  cold: a 256 MiB write before every single timed call (L2 24 MiB flushed; weights and activations re-read from DRAM),
        pairs alternate the order
Reports the median paired delta, its min..max over rounds, and how many rounds had B < A.
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe2rev/bench_ab_ms.py
Env: BENCH_T (13824,4289), BENCH_SEEDS (2 routings), BENCH_ROUNDS (15), BENCH_N (5), BENCH_COLD (30 pairs)
"""
from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def ev_time(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def summ(name, a, b):
    d = [y - x for x, y in zip(a, b)]
    md = statistics.median(d)
    q = sorted(d)
    lo, hi = q[len(q) // 4], q[(3 * len(q)) // 4]
    neg = sum(1 for v in d if v < 0)
    return (f"{name}: A {statistics.median(a):.2f} B {statistics.median(b):.2f} | paired delta median {md:+.3f} "
            f"(IQR {lo:+.3f}..{hi:+.3f}, min {q[0]:+.3f}, max {q[-1]:+.3f}), B<A in {neg}/{len(d)}")


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    ext = M._ext()
    P = L._exl3_ptrs
    n_exp = len(L._exl3_inners)
    emap = prod.pin_exl3_expert_map(L, dev)
    rounds = int(os.environ.get("BENCH_ROUNDS", "15"))
    n = int(os.environ.get("BENCH_N", "5"))
    ncold = int(os.environ.get("BENCH_COLD", "30"))
    flush = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
    for T in [int(v) for v in os.environ.get("BENCH_T", "13824,4289").split(",")]:
        for seed in [int(v) for v in os.environ.get("BENCH_SEEDS", "101,202").split(",")]:
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            ids = C.routing("real", T, seed, dev)
            w = C.weights_for(T, seed, dev).float()
            t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
            a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
            ext.gather_tok(x, P["gate_suh"], a8, asc, None)
            outs = {"f32": torch.zeros(T, 4096, dtype=torch.float32, device=dev),
                    "bf16": torch.zeros(T, 4096, dtype=torch.bfloat16, device=dev)}
            sync = torch.zeros(1 + 2 * int(t["seg_expert"].numel()), dtype=torch.int32, device=dev)

            def fused(acc, var):
                sync.zero_()
                ext.fused(a8, asc, a8d, dsc, a16, outs[acc], P["gate_trellis"], P["up_trellis"], P["gate_svh"],
                          P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"], t["row_token"],
                          t["row_weight"], t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], sync,
                          float(C.LIMIT), 12, 0, var)
            nsegs = int(t["num_segs"].item())
            for acc in ("bf16", "f32"):
                A = lambda: fused(acc, 2048)        # noqa: E731
                B = lambda: fused(acc, 10240)       # noqa: E731
                for f in (A, B, A, B):
                    f()
                ta, tb, ta2 = [], [], []
                for r in range(rounds):
                    if r % 2 == 0:
                        ta.append(ev_time(A, n)); tb.append(ev_time(B, n)); ta2.append(ev_time(A, n))
                    else:
                        ta2.append(ev_time(A, n)); tb.append(ev_time(B, n)); ta.append(ev_time(A, n))
                print(f"[T={T} seed {seed} {nsegs} segs {acc} warm x{rounds}x{n}] "
                      + summ("ship->MAINLOOP", ta, tb) + " || " + summ("A/A control", ta, ta2), flush=True)
                ca, cb, ca2 = [], [], []
                for r in range(ncold):
                    order = (("a", A), ("b", B), ("a2", A)) if r % 2 == 0 else (("a2", A), ("b", B), ("a", A))
                    for k, f in order:
                        flush.zero_()
                        v = ev_time(f, 1)
                        {"a": ca, "b": cb, "a2": ca2}[k].append(v)
                print(f"[T={T} seed {seed} {acc} cold x{ncold}] " + summ("ship->MAINLOOP", ca, cb) + " || "
                      + summ("A/A control", ca, ca2), flush=True)
            del x, outs
            torch.cuda.empty_cache()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
