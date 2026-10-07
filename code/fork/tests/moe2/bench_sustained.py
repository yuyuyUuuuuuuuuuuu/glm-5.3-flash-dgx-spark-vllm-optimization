"""Adversarial speed check: does the e4m3 path keep its x1.5 under SUSTAINED load (power / clock throttling of the
mma-dense kernel), not only in the short alternating bench?  Real layer-10 experts (TP=2 rank 0 shard), T=13824 real
routing. For each path: BURST back-to-back calls (default 160, ~6-10 s, i.e. longer than one 13,824-token chunk's
whole MoE time), per-call CUDA-event times; first-10 vs last-40 medians. Then the same with the paths interleaved
call by call (the bench's own pattern) for comparison.
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe2/bench_sustained.py
"""
from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def burst(fn, n):
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    torch.cuda.synchronize()
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in evs]


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    T = int(os.environ.get("BENCH_T", "13824"))
    n = int(os.environ.get("BURST", "160"))
    g = torch.Generator().manual_seed(T)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", T, T, dev)
    w = C.weights_for(T, T, dev).float()
    emap = prod.pin_exl3_expert_map(L, dev)
    f_prod = lambda: prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, C.LIMIT)  # noqa: E731
    f_me = lambda: M.run(prod, x, ids, w, L, C.LIMIT)  # noqa: E731
    for f in (f_prod, f_me):
        f()
    res = {}
    for name, f in (("prod", f_prod), ("e4m3", f_me), ("prod2", f_prod), ("e4m3_2", f_me)):
        t = burst(f, n)
        res[name] = t
        print(f"[burst {name}] {n} calls {sum(t) / 1000:.2f} s: first10 med {statistics.median(t[:10]):.2f} ms, "
              f"last40 med {statistics.median(t[-40:]):.2f} ms, max {max(t):.2f} ms, p10 {sorted(t)[n // 10]:.2f} ms",
              flush=True)
    inter = {"prod": [], "e4m3": []}
    for i in range(40):
        for name, f in ((("prod", f_prod), ("e4m3", f_me)) if i % 2 == 0 else (("e4m3", f_me), ("prod", f_prod))):
            inter[name] += burst(f, 1)
    for name in inter:
        print(f"[interleaved {name}] med {statistics.median(inter[name]):.2f} ms", flush=True)
    sp = lambda a, b: statistics.median(res[a][-40:]) / statistics.median(res[b][-40:])  # noqa: E731
    print(f"SUSTAINED speedup (last40 medians): {sp('prod', 'e4m3'):.3f} / {sp('prod2', 'e4m3_2'):.3f}; "
          f"interleaved {statistics.median(inter['prod']) / statistics.median(inter['e4m3']):.3f}", flush=True)
    PC.uninstall(prod)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
