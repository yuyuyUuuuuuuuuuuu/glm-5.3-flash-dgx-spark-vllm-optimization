"""W8A8 dense GEMMs at production prefill shapes, through the real hooked apply (docs/DENSE_W8A8.md).

For every production dense FP8 shape (per rank, TP=2, GLM53_DENSE_FP8=dense,kda,mla,shared; real weights from the
partial checkpoint where it has them):
  - production's CURRENT path through the hooked Glm53DenseFp8Method.apply (GLM53_FP8_LARGE_M=1, as in production:
    the TileLang large-M path for the big shapes, Marlin for the rest);
  - the W8A8 path through the SAME hooked apply with GLM53_DENSE_W8A8=1 installed ON TOP of fp8_gemv (per-call
    Marlin->fp8 repack + per-token quant + cutlass pieces), per M = 13824 and at the tail-chunk size M = 1791;
  - the repack kernel and the per-token quant alone (the W8A8 overhead over the GEMMs);
  - a rel_l2 of W8A8 vs production's output and vs an fp32 reference (the kill-test class 2.4e-2..2.7e-2);
  - an M sweep on two shapes (threshold choice);
then per 13,824-token chunk per rank totals, the memory cost (persistent repack scratch + transient peak of a full
chunk of W8A8 calls, against the 3.50 GiB/rank resident copy the kill test measured), and the M-sweep table.

Run: GPU_RUN_ENV="TF_EXL3_JIT=1" GPU_RUN_RO="$TF_EXL3_MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw-partial" \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/bench_fp8_w8a8.py
"""
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pf3000"))
from harness import run_main, gpu_guard, load_prod  # noqa: E402
import torch  # noqa: E402
from bench_test_c import SHAPES, load_real  # noqa: E402

dev = "cuda"
M = 13824
TAIL = 1791
g = torch.Generator(device=dev).manual_seed(0)


def timed(fn, reps=3, rounds=5, order=("a", "b")):
    """Median of `rounds` interleaved rounds of `reps` calls (the bench_test_c methodology)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = {a: [] for a in order}
    for r in range(rounds):
        for a in order[r % len(order):] + order[:r % len(order)]:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(reps):
                fn()
            e1.record()
            torch.cuda.synchronize()
            ts[a].append(e0.elapsed_time(e1) / reps)
    return statistics.median(ts[order[0]])


def main():
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M",
              "GLM53_DEC_FP8ROOF", "GLM53_DENSE_W8A8"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"          # production's value (env_nonsecret.txt:52)
    os.environ["GLM53_DENSE_FP8"] = "dense,kda,mla,shared"   # production's value (env_nonsecret.txt:51)
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    cls = prod.Glm53DenseFp8Method
    rep_f = F.install(prod)
    assert rep_f["installed"] and F.STATE.large, rep_f
    prod_apply = cls.apply                          # fp8_gemv's wrapper = production's current path
    os.environ["GLM53_DENSE_W8A8"] = "1"
    rep_w = W.install(prod)
    assert rep_w.get("installed") and W.STATE.enabled, rep_w
    assert cls.apply._tf_w8a8_orig is prod_apply, "w8a8 must wrap fp8_gemv's apply"
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}, {p.name} cc {p.major}.{p.minor}; fp8_gemv {rep_f['reason']}, "
          f"w8a8 {rep_w['reason']} (min M {W.CFG.min_m}, piece {W.CFG.piece_bytes >> 20} MiB); "
          f"median of 5 interleaved rounds", flush=True)

    tot = {"prod": 0.0, "w8a8": 0.0, "repack": 0.0, "quant": 0.0, "w8a8_tail": 0.0, "prod_tail": 0.0}
    resident = 0
    E = W.ext()
    F_L = F.ext_large()
    import inspect
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters

    for name, (n, k, grp, pre, calls, src) in SHAPES.items():
        w = load_real(src) if src else (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        m.process_weights_after_loading(lay)
        del w
        npad = lay.weight.shape[1] // 4
        x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        xt = (torch.randn(TAIL, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        resident += calls * n * k

        def prod_cur():
            return prod_apply(m, lay, x)

        def w8a8_cur():
            return cls.apply(m, lay, x)

        def prod_tail():
            return prod_apply(m, lay, xt)

        def w8a8_tail():
            return cls.apply(m, lay, xt)

        def repack_only():
            W.repack(W._scratch(n, k, dev), lay.weight, n, k)

        def quant_only():
            W.quant_per_token(x[:2048])

        ms_prod = timed(prod_cur)
        ms_w8a8 = timed(w8a8_cur)
        ms_prod_t = timed(prod_tail)
        ms_w8a8_t = timed(w8a8_tail)
        ms_repack = timed(repack_only)
        ms_quant = timed(quant_only)
        # numerics vs production and vs an fp32 reference (Gaussian activations)
        yp = prod_cur().float()
        yw = w8a8_cur().float()
        rl_prod = ((yw - yp).norm() / yp.norm()).item()
        del yp, yw
        xs = (torch.randn(2048, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        W.repack(W._scratch(n, k, dev), lay.weight, n, k)     # the scratch must hold THIS layer's fp8 for the ref
        yref = torch.mm(xs.float(), (W._scratch(n, k, dev).float() *
                                     W.large_alpha(lay.weight_scale, n)[:n][:, None].float()).t())
        yw2 = cls.apply(m, lay, xs).float()
        rl_ref = ((yw2 - yref).norm() / yref.norm()).item()
        del yref, yw2, xs
        fl = 2.0 * M * n * k
        note = "" if grp != "draft" else "  [drafter group: NOT served by W8A8, same path in both arms]"
        tot["prod"] += calls * ms_prod
        tot["w8a8"] += calls * ms_w8a8
        tot["repack"] += calls * ms_repack
        tot["quant"] += calls * (ms_quant * (-(-M // 2048)))
        tot["w8a8_tail"] += calls * ms_w8a8_t
        tot["prod_tail"] += calls * ms_prod_t
        print(f"{name:15s} [{n}x{k}] Npad={npad} calls={calls:2d} real={'Y' if src else 'N'}: "
              f"production {ms_prod:7.3f} ms | W8A8 {ms_w8a8:7.3f} ms ({ms_repack:5.2f} repack + "
              f"{ms_quant * (-(-M // 2048)):5.2f} quant/GEMMs) | speedup {ms_prod / ms_w8a8:4.2f}x | "
              f"tail M={TAIL}: {ms_prod_t:6.3f} -> {ms_w8a8_t:6.3f} ({ms_prod_t / ms_w8a8_t:4.2f}x) | "
              f"rel_l2 W8A8-vs-prod {rl_prod:.2e}, vs fp32 ref {rl_ref:.2e}{note}", flush=True)
        del lay, m, x, xt
        torch.cuda.empty_cache()

    scratch = sum(b.numel() * b.element_size() for b in W.STATE.scratch.values())
    alpha = sum(t.numel() * t.element_size() for t in W.ALPHA.values())
    print("\n==== per 13,824-token chunk per rank (calls per chunk from DEC_FP8ROOF/FP8_LARGE_M) ====")
    print(f"production current (TileLang large-M + Marlin): {tot['prod']:8.1f} ms/chunk")
    print(f"W8A8 (repack + per-token quant + cutlass):      {tot['w8a8']:8.1f} ms/chunk "
          f"-> {tot['prod'] - tot['w8a8']:+.1f} ms")
    print(f"  of which repack kernel {tot['repack']:.1f} ms, per-token quant {tot['quant']:.1f} ms")
    print(f"tail chunk M={TAIL}: production {tot['prod_tail']:.1f} ms -> W8A8 {tot['w8a8_tail']:.1f} ms "
          f"({tot['prod_tail'] - tot['w8a8_tail']:+.1f} ms)")
    print(f"memory per rank: persistent repack scratch {scratch / 2**20:.1f} MiB + cached scales {alpha / 2**20:.1f} "
          f"MiB (transient per call <= the scratch + one piece); the resident standard-layout copy the kill test "
          f"rejected is {resident / 2**30:.2f} GiB/rank")
    # ---- memory of one full chunk through the W8A8 path
    print("\n==== peak CUDA allocation of a full W8A8 chunk (M=13824, every shape once per call count) ====")
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for name, (n, k, grp, pre, calls, src) in SHAPES.items():
        w = load_real(src) if src else (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        m.process_weights_after_loading(lay)
        del w
        x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        for _ in range(calls):
            cls.apply(m, lay, x)
        del lay, m, x
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    above = (peak - base) / 2 ** 30
    print(f"peak above the pre-chunk allocation: {above:.3f} GiB (weights excluded: each layer's Marlin payload was "
          f"already resident); persistent scratch {scratch / 2**20:.1f} MiB of that")
    print(f"KV pool equivalent at 8,039 B/token/rank: {scratch / 8039:.0f} tokens "
          f"(2.00M -> {2_003_436 - scratch / 8039:.0f})")
    print(f"\ncounters: {W.summary()}")

    # ---- M sweep (threshold choice): in_proj (big, pieces) and shared.gate_up (small, single call)
    print("\n==== M sweep (ms, W8A8 vs production current) ====")
    for name in ("kda.in_proj", "shared.gate_up"):
        n, k, grp, pre, calls, src = SHAPES[name]
        w = load_real(src) if src else (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        m.process_weights_after_loading(lay)
        del w
        row = f"   {name:15s} [{n}x{k}]:"
        for mm in (256, 512, 1024, 1791, 4608, 13824):
            x = (torch.randn(mm, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)

            def pc():
                return prod_apply(m, lay, x)

            def wc():
                return cls.apply(m, lay, x)

            a = timed(pc)
            b = timed(wc)
            row += f" M{mm}:{a:.2f}/{b:.2f}({a / b:.2f}x)"
        print(row, flush=True)
        del lay, m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_main(main)
