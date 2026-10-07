"""opt-moe2: tests/optmoe_rev/bench_e2e.py with the GLM53_MOE_E4M3_MAINLOOP=1 arms (lean mainloop, "_ms").
opt-moe-rev: end-to-end A/B of the routed-MoE call AS vLLM COMPOSES IT (refuter's bench).
Every arm = fresh shared-expert output S (S_buf.copy_(S), same cost in every arm) + the routed call + whatever
combines them: production (fp32 acc, per-pair gather, .to(bf16), S + routed) vs ACC=bf16 / +TOKGATHER / +FOLD.
Real layer-10 experts (TP=2 rank-0 shard), real-ish routing, prefill cap 1 installed. Two L2 conditions:
  warm = calls back to back; cold = a 256 MiB write between calls (outside the timed region).
Each call timed with its own CUDA events; interleaved arms; median over BENCH_N samples.
Env: BENCH_T (13824,4289), BENCH_N (15), BENCH_KIND (real).
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe2/bench_e2e_ms.py"""
from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe_rev"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


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
    print("tok_gather_ok", M.tok_gather_ok(L), flush=True)
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,4289").split(",")]
    N = int(os.environ.get("BENCH_N", "15"))
    kind = os.environ.get("BENCH_KIND", "real")
    flush = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    for T in Ts:
        g = torch.Generator().manual_seed(1000 + T)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        S = (torch.randn(T, 4096, generator=g) * 0.3).to(torch.bfloat16).to(dev)
        ids = C.routing(kind, T, 1000 + T, dev)
        w = C.weights_for(T, 1000 + T, dev).float()
        Sb = torch.empty_like(S)

        def mk(acc, tg, fold, ms=False):
            sch = {"acc": acc, "tg": tg, "ms": ms}

            def f():
                Sb.copy_(S)
                if fold:
                    o = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched=sch, fold_into=Sb)
                    assert o is Sb
                    return o
                o = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched=sch).to(torch.bfloat16)
                return Sb + o
            return f
        arms = {"prod": mk("f32", False, False), "f32_tg": mk("f32", True, False),
                "f32_tg_ms": mk("f32", True, False, True), "acc_tg_fold": mk("bf16", True, True),
                "acc_tg_fold_ms": mk("bf16", True, True, True)}
        copy_only = lambda: Sb.copy_(S)  # noqa: E731
        # numerics: every arm vs the production arm, and production's own run-to-run
        p0 = arms["prod"]().clone()
        p1 = arms["prod"]().clone()
        line = [f"prod run-to-run {rel(p1, p0):.2e}"]
        for k, f in arms.items():
            if k == "prod":
                continue
            o = f().clone()
            assert torch.isfinite(o).all()
            line.append(f"{k} {rel(o, p0):.2e}")
        print(f"[T={T} {kind}] ERR vs prod: " + " | ".join(line), flush=True)
        for cond in ("warm", "cold"):
            t = {k: [] for k in list(arms) + ["copy"]}
            fns = dict(arms, copy=copy_only)
            for f in fns.values():
                f()
            names = list(fns)
            for r in range(N):
                for k in (names if r % 2 == 0 else names[::-1]):
                    if cond == "cold":
                        flush.fill_(r & 0xFF)
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    fns[k]()
                    e.record()
                    e.synchronize()
                    t[k].append(s.elapsed_time(e))
            med = {k: statistics.median(v) for k, v in t.items()}
            base = med["prod"]
            print(f"[T={T} {kind} {cond}] " + " | ".join(
                f"{k} {v:.2f}" + ("" if k in ("prod", "copy") else f" ({v - base:+.2f})") for k, v in med.items())
                + " ms (each arm includes the S copy)", flush=True)
        del x, S, Sb
        torch.cuda.empty_cache()
    PC.uninstall(prod)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
