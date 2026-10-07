"""opt-moe: the bf16 accumulator (sched acc=bf16) vs the shipped fp32 accumulator + .to(bf16), real layer-10 experts
(TP=2 rank-0 shard), production's prefill cap 1 installed. Per call (apply-level work of the e4m3 path: plan + gather2
+ fused + the cast), interleaved rounds, median. Error: every arm's bf16 output vs the fp32-accumulator result (fp32,
before its cast); the fp32 arm's own cast error and run-to-run (atomics order) are the floor to compare with.
Env: BENCH_T (13824,4289), BENCH_KINDS (real,collapsed), BENCH_ROUNDS (9), BENCH_N (3).
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/bench_acc.py"""
from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def sample(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def rel(a, b):
    a = a.double(); b = b.double()
    return float((a - b).norm() / b.norm())


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    print("smem/SM", torch.cuda.get_device_properties(dev).shared_memory_per_multiprocessor if hasattr(
        torch.cuda.get_device_properties(dev), "shared_memory_per_multiprocessor") else "?",
        "SMs", torch.cuda.get_device_properties(dev).multi_processor_count, flush=True)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,4289").split(",")]
    kinds = os.environ.get("BENCH_KINDS", "real,collapsed").split(",")
    rounds = int(os.environ.get("BENCH_ROUNDS", "9"))
    n = int(os.environ.get("BENCH_N", "3"))
    arms = {"f32": {}, "b16v8": {"acc": "bf16", "variant": 0}, "b16v4": {"acc": "bf16", "variant": 1}}
    if os.environ.get("BENCH_D16", "0") == "1":
        arms.update({"f32_d16": {"variant": 16}, "b16_d16": {"acc": "bf16", "variant": 16}})
    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            seed = T + (0 if kind == "real" else 50000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()

            def call(sch):
                o = M.run(prod, x, ids, w, L, C.LIMIT, sched=sch or None)
                return o.to(torch.bfloat16)
            # numerics
            ref = M.run(prod, x, ids, w, L, C.LIMIT).clone()           # fp32 accumulator, fp32
            ref2 = M.run(prod, x, ids, w, L, C.LIMIT).clone()
            ref16 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": 16}).clone() if "f32_d16" in arms else None
            line = [f"f32 run-to-run {rel(ref2, ref):.2e}", f"f32 cast {rel(ref.to(torch.bfloat16), ref):.2e}"]
            for k, sch in arms.items():
                if k == "f32":
                    continue
                o = call(sch)
                r = ref16 if "d16" in k else ref
                assert o.dtype == torch.bfloat16 and torch.isfinite(o).all()
                d = (o.double() - r.double()).abs()
                line.append(f"{k} {rel(o, r):.2e} (max {float(d.max() / r.abs().max()):.1e})")
            print(f"[{kind} T={T}] ERR " + " | ".join(line), flush=True)
            fns = {k: (lambda sch=sch: call(sch)) for k, sch in arms.items()}
            for f in fns.values():
                f()
            t = {k: [] for k in fns}
            names = list(fns)
            for r in range(rounds):
                for k in (names if r % 2 == 0 else names[::-1]):
                    t[k].append(sample(fns[k], n))
            med = {k: statistics.median(v) for k, v in t.items()}
            print(f"[{kind} T={T}] TIME " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms  (b16v8-f32 "
                  f"{med['b16v8'] - med['f32']:+.2f})", flush=True)
            del x, ref, ref2
            torch.cuda.empty_cache()
    PC.uninstall(prod)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
