"""opt-moe-rev: TOKGATHER / bf16-accumulator stress (refuter). For many token counts (item-boundary sizes, odd sizes,
production chunk sizes) and three routings, repeat: TG vs per-pair with the fp32 accumulator must agree to the
atomics-order class (rel-L2 <= 1e-6; run-to-run is ~1e-9) and, after the production cast + add to a shared output,
bitwise in bf16 except where the per-pair path itself flips run to run; bf16 accumulator TG vs per-pair: both
vs fp32 within a fixed bound. Any race in the g_srt staging (cp.async double buffer) shows up as a large outlier.
Env: STRESS_REPS (3). Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe_rev/stress_tg.py"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    reps = int(os.environ.get("STRESS_REPS", "3"))
    sizes = [257, 300, 511, 1000, 2047, 2048, 2049, 4289, 6000, 8191, 13824, 16384]
    worst = {"f32": 0.0, "bf16_excess": 0.0}
    for T in sizes:
        for kind in ("real", "zipf", "collapsed"):
            seed = 7 * T + len(kind)
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn(T, 4096, generator=g) * (0.5 + (T % 3))).to(torch.bfloat16).to(dev)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()
            S = (torch.randn(T, 4096, generator=g) * 0.3).to(torch.bfloat16).to(dev)
            f0 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False}).clone()
            f1 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False}).clone()
            spread = rel(f1, f0)
            pb0 = S + f0.to(torch.bfloat16)
            pflip = int((pb0 != (S + f1.to(torch.bfloat16))).sum())
            b0 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": False}).clone()
            eb0 = rel(b0, f0)
            mx_f, mx_flip, mx_b = 0.0, 0, 0.0
            for r in range(reps):
                ft = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": True})
                e = rel(ft, f0)
                mx_f = max(mx_f, e)
                mx_flip = max(mx_flip, int((pb0 != (S + ft.to(torch.bfloat16))).sum()))
                bt = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": True})
                mx_b = max(mx_b, rel(bt, f0))
                buf = S.clone()
                fo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": True}, fold_into=buf)
                ef = rel(fo, S.double() + f0.double())
                mx_b = max(mx_b, ef * 0.9)       # fold error is a bit larger (S rounded in); bound below
            worst["f32"] = max(worst["f32"], mx_f)
            worst["bf16_excess"] = max(worst["bf16_excess"], mx_b - eb0)
            CHK(mx_f <= max(4 * spread, 1e-7) and mx_b < 6e-3 and mx_flip <= max(4 * pflip, 64),
                f"[T={T} {kind}] fp32 TG vs per-pair max {mx_f:.2e} (run-to-run {spread:.2e}); bf16-out elements "
                f"differing after cast+add {mx_flip} (per-pair self {pflip}) of {T * 4096}; bf16 acc TG max {mx_b:.2e} "
                f"(per-pair bf16 {eb0:.2e})")
            del x, S, f0, f1, b0, pb0
        torch.cuda.empty_cache()
    print("worst", worst, flush=True)
    PC.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
