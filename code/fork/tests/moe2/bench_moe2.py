"""moe2 before/after, per apply call, real layer-10 experts (TP=2 rank 0 shard), interleaved rounds (median):
  prod      production's apply_exl3_experts (prefill cap 1 installed, as deployed), unhooked
  e4m3_old  the e4m3 path as in combo: x.half() copy, gather from fp16, fused variant 0, .to(bf16)
  e4m3      moe2: gather reads bf16 directly (bit-identical a8, test_down16), fused variant 0, .to(bf16)
  e4m3_d16  moe2 + GLM53_MOE_E4M3_DOWN=f16 (fused variant 16), .to(bf16)
  hook / hook_d16  the same through the installed wrapper (production's apply_exl3_experts, hooked)
Env: BENCH_T (default 13824,4289,1791,512,300), BENCH_KINDS (real,collapsed), BENCH_ROUNDS (9), BENCH_N (3 calls/sample)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe2/bench_moe2.py
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


def sample(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,4289,1791,512,300").split(",")]
    kinds = os.environ.get("BENCH_KINDS", "real,collapsed").split(",")
    rounds = int(os.environ.get("BENCH_ROUNDS", "9"))
    n = int(os.environ.get("BENCH_N", "3"))
    base = prod.apply_exl3_experts
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1"}, load_selftest=False)
    assert rep["installed"]
    hooked = prod.apply_exl3_experts
    rows = []
    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            seed = T + (0 if kind == "real" else 50000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()
            fns = {
                "prod": lambda: base(x, ids, w, L, limit=C.LIMIT),
                "e4m3_old": lambda: M.run(prod, x.half(), ids, w, L, C.LIMIT).to(torch.bfloat16),
                "e4m3": lambda: M.run(prod, x, ids, w, L, C.LIMIT).to(torch.bfloat16),
                "e4m3_d16": lambda: M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": 16}).to(torch.bfloat16),
                "hook": lambda: hooked(x, ids, w, L, limit=C.LIMIT),
            }

            def hook16():
                M.DOWN16["on"] = True
                try:
                    return hooked(x, ids, w, L, limit=C.LIMIT)
                finally:
                    M.DOWN16["on"] = False
            fns["hook_d16"] = hook16
            for f in fns.values():
                f()
            t = {k: [] for k in fns}
            names = list(fns)
            for r in range(rounds):
                order = names if r % 2 == 0 else names[::-1]
                for k in order:
                    t[k].append(sample(fns[k], n))
            med = {k: statistics.median(v) for k, v in t.items()}
            rows.append((kind, T, med))
            print(f"[{kind} T={T}] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms", flush=True)
            del x
            torch.cuda.empty_cache()
    M.uninstall(prod)
    PC.uninstall(prod)
    print("TABLE kind T prod e4m3_old e4m3 e4m3_d16 hook hook_d16 | old->new  d16-new  speedup(prod/new, prod/d16)")
    for kind, T, m in rows:
        print(f"TABLE {kind} {T} {m['prod']:.2f} {m['e4m3_old']:.2f} {m['e4m3']:.2f} {m['e4m3_d16']:.2f} {m['hook']:.2f} "
              f"{m['hook_d16']:.2f} | {m['e4m3'] - m['e4m3_old']:+.2f} {m['e4m3_d16'] - m['e4m3']:+.2f} | "
              f"x{m['prod'] / m['e4m3']:.3f} x{m['prod'] / m['e4m3_d16']:.3f}")
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
