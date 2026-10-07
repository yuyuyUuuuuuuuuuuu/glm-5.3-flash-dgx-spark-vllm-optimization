"""opt-moe: where the e4m3 routed-MoE call spends its time (bf16 accumulator and fp32), real layer-10 experts,
T=13,824 / 4,289 real routing: plan() (routing tables, torch + seg_tables), gather2 (incl. zeroing), fused, the whole
run(); CUDA events around each part, BENCH_N calls per sample, median of BENCH_ROUNDS.
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/bench_parts.py"""
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


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    pr = torch.cuda.get_device_properties(dev)
    print(f"device L2 {getattr(pr, 'L2_cache_size', -1) / 2**20:.1f} MiB, SMs {pr.multi_processor_count}", flush=True)
    L = make_real_layer(prod, dev)
    ext = M._ext()
    P = L._exl3_ptrs
    n_exp = len(L._exl3_inners)
    emap = prod.pin_exl3_expert_map(L, dev)
    rounds = int(os.environ.get("BENCH_ROUNDS", "7"))
    n = int(os.environ.get("BENCH_N", "5"))
    variant = int(os.environ.get("BENCH_VARIANT", "0"))
    for T in [int(v) for v in os.environ.get("BENCH_T", "13824,4289").split(",")]:
        g = torch.Generator().manual_seed(T)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        ids = C.routing("real", T, T, dev)
        w = C.weights_for(T, T, dev).float()
        t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
        a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
        outs = {"f32": torch.empty(T, 4096, dtype=torch.float32, device=dev),
                "bf16": torch.empty(T, 4096, dtype=torch.bfloat16, device=dev)}
        need = 1 + 2 * int(t["seg_expert"].numel())
        sync = torch.zeros(need, dtype=torch.int32, device=dev)

        def gather(o):
            ext.gather2(x, t["local"], t["pos"], P["gate_suh"], a8, asc, o, t["topk"], n_exp)

        def fused(o, var=variant, lag=12):
            sync.zero_()
            ext.fused(a8, asc, a8d, dsc, a16, o, P["gate_trellis"], P["up_trellis"], P["gate_svh"], P["up_svh"],
                      P["down_trellis"], P["down_suh"], P["down_svh"], t["row_token"], t["row_weight"],
                      t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], sync, float(C.LIMIT), lag, 0, var)
        fns = {
            "plan": lambda: M.plan(prod, ids.to(torch.long), w, n_exp, emap),
            "gather_f32": lambda: gather(outs["f32"]),
            "gather_bf16": lambda: gather(outs["bf16"]),
            "gather_nozero": lambda: gather(None),
            "fused_f32": lambda: fused(outs["f32"]),
            "fused_bf16": lambda: fused(outs["bf16"]),
            "cast": lambda: outs["f32"].to(torch.bfloat16),
            "run_f32": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32"}).to(torch.bfloat16),
            "run_bf16": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"}),
        }
        for v in [int(u) for u in os.environ.get("BENCH_FV", "").split(",") if u]:
            fns[f"fused_bf16_v{v}"] = (lambda v=v: fused(outs["bf16"], v))
            fns[f"run_bf16_v{v}"] = (lambda v=v: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap,
                                                        sched={"acc": "bf16", "variant": v}))
        if os.environ.get("BENCH_TG"):
            # TG variants read token rows: fill a8[:T] with gather_tok first (the per-pair variants then time on
            # whatever is in the buffer; timing only)
            ext.gather_tok(x, P["gate_suh"], a8, asc, None)
            fns["gather_tok"] = lambda: ext.gather_tok(x, P["gate_suh"], a8, asc, outs["bf16"])
            fns["fused_bf16_tg"] = lambda: fused(outs["bf16"], variant + 2048)
            fns["fused_f32_tg"] = lambda: fused(outs["f32"], variant + 2048)
            if os.environ.get("BENCH_TGPROBE"):
                fns["fused_bf16_tgprobe"] = lambda: fused(outs["bf16"], 4096)
                fns["fused_bf16_tgstage"] = lambda: fused(outs["bf16"], 4097)
        if os.environ.get("BENCH_E0"):
            # probe: every segment uses expert 0's weights (weights L2-resident; garbage result, timing only)
            se0 = torch.zeros_like(t["seg_expert"])

            def fused_e0(o, var=variant):
                sync.zero_()
                ext.fused(a8, asc, a8d, dsc, a16, o, P["gate_trellis"], P["up_trellis"], P["gate_svh"], P["up_svh"],
                          P["down_trellis"], P["down_suh"], P["down_svh"], t["row_token"], t["row_weight"],
                          se0, t["seg_row0"], t["seg_rows"], t["num_segs"], sync, float(C.LIMIT), 12, 0, var)
            fns["fused_bf16_e0"] = lambda: fused_e0(outs["bf16"])
        for lg in [int(u) for u in os.environ.get("BENCH_LAG", "").split(",") if u]:
            fns[f"fused_bf16_lag{lg}"] = (lambda lg=lg: fused(outs["bf16"], variant, lg))
        for v in [int(u) for u in os.environ.get("BENCH_FV32", "").split(",") if u]:
            fns[f"fused_f32_v{v}"] = (lambda v=v: fused(outs["f32"], v))
        for f in fns.values():
            f()
        tt = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                tt[k].append(sample(fns[k], n))
        med = {k: statistics.median(v) for k, v in tt.items()}
        nsegs = int(t["num_segs"].item())
        print(f"[T={T} real, {nsegs} segments] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms",
              flush=True)
        del x, outs
        torch.cuda.empty_cache()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
