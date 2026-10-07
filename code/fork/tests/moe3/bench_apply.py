"""moe3: per apply_exl3_experts call (what the MoE layer calls; incl. routing, thin kernel, zero-fill, casts), real
layer-10 experts (TP=2 rank-0 shard), production module with prefill cap 1 installed (as deployed), interleaved rounds:
  prod       production, unhooked
  fused16    GLM53_MOE_FUSED16=1 (production arithmetic, P16 schedules)
  e4m3       GLM53_MOE_E4M3=1 (the e4m3 path; outermost wrapper, serves the whole call)
  e4m3_d16   GLM53_MOE_E4M3=1 + GLM53_MOE_E4M3_DOWN=f16
Env (GPU_RUN_ENV): BENCH_T (13824,13856,4289,1791), BENCH_KINDS (real,collapsed), BENCH_ROUNDS (9), BENCH_N (3)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe3/bench_apply.py
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
    import glm53_moe_fused16 as F
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    base = prod.apply_exl3_experts
    assert F.install(prod, environ={"GLM53_MOE_FUSED16": "1"}, load_selftest=False)["installed"]
    fused_gf = prod.apply_exl3_grouped_fat
    orig_gf = fused_gf._glm53_moe_fused16_orig
    assert M.install(prod, environ={"GLM53_MOE_E4M3": "1"}, load_selftest=False)["installed"]
    hooked = prod.apply_exl3_experts
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,13856,4289,1791").split(",")]
    kinds = os.environ.get("BENCH_KINDS", "real,collapsed").split(",")
    rounds = int(os.environ.get("BENCH_ROUNDS", "9"))
    n = int(os.environ.get("BENCH_N", "3"))
    rows = []

    def with_gf(gf, fn):
        prod.apply_exl3_grouped_fat = gf
        try:
            return fn()
        finally:
            prod.apply_exl3_grouped_fat = fused_gf

    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            seed = T + (0 if kind == "real" else 50000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()

            def e4(d16):
                M.DOWN16["on"] = d16
                try:
                    return hooked(x, ids, w, L, limit=C.LIMIT)
                finally:
                    M.DOWN16["on"] = False
            fns = {
                "prod": lambda: with_gf(orig_gf, lambda: base(x, ids, w, L, limit=C.LIMIT)),
                "fused16": lambda: base(x, ids, w, L, limit=C.LIMIT),
                "e4m3": lambda: e4(False),
                "e4m3_d16": lambda: e4(True),
            }
            for f in fns.values():
                f()
            t = {k: [] for k in fns}
            names = list(fns)
            for r in range(rounds):
                for k in (names if r % 2 == 0 else names[::-1]):
                    t[k].append(sample(fns[k], n))
            med = {k: statistics.median(v) for k, v in t.items()}
            rows.append((kind, T, med))
            print(f"[{kind} T={T}] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms", flush=True)
            del x
            torch.cuda.empty_cache()
    M.uninstall(prod)
    F.uninstall(prod)
    PC.uninstall(prod)
    print("TABLE kind T prod fused16 e4m3 e4m3_d16 | fused16-prod e4m3-fused16 e4m3_d16-fused16")
    for kind, T, m in rows:
        print(f"TABLE {kind} {T} {m['prod']:.2f} {m['fused16']:.2f} {m['e4m3']:.2f} {m['e4m3_d16']:.2f} | "
              f"{m['fused16'] - m['prod']:+.2f} {m['e4m3'] - m['fused16']:+.2f} {m['e4m3_d16'] - m['fused16']:+.2f}")
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
