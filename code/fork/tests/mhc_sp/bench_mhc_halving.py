"""MHC-SP kill-tests (a) and (b) on nodeC, single GPU (docs/MHC_SP.md sections 2-3).

(a) halving: the production mHC prefill op chain (mhc_fused_post_pre_tilelang = post + tf32 prenorm +
    pre_big_fuse_with_norm, then the aux/final qw_post_mean) timed at T = 13,824 / 6,912 / 4,289 inside CUDA
    graphs, alternating rounds like tests/bench_smallops.py. Production per chunk = 90 fused calls + 6
    post_means (ANATOMY.md 2.3: 881.6 ms at 13,824). Kill if 90*fused(6912)+6*mean(6912) > 441 ms
    (the gross <= 0.44 s/chunk gate) or if the saving is < 0.25 s.
(b) shard numerics: the same production ops run on the two rank halves of a 13,824-token step vs the full-T run,
    same rows compared bitwise; also at 1,536 tokens where compute_num_split picks split_k > 1 (fp32-order
    differences expected there, not at 13,824/6,912 where split_k == 1 on both sides).

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc_sp/bench_mhc_halving.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROUNDS = int(os.environ.get("BENCH_ROUNDS", "11"))
REPS = int(os.environ.get("BENCH_REPS", "5"))
DEV = torch.device("cuda", 0)
N, H = 4, 4096
H3 = 2 * N + N * N          # 24
HK = N * H                  # 16384


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def timed(graphs: dict, calls: int):
    names = list(graphs)
    for n in names:
        for _ in range(3):
            graphs[n].replay()
    torch.cuda.synchronize()
    res = {n: [] for n in names}
    for r in range(ROUNDS):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(REPS):
                graphs[n].replay()
            e.record()
            torch.cuda.synchronize()
            res[n].append(s.elapsed_time(e) * 1000.0 / (REPS * calls))
    out = {}
    for n, v in res.items():
        v = sorted(v)
        med = v[len(v) // 2]
        out[n] = (med, (v[-1] - v[0]) / med)
    return out


def make(T: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    fn = (torch.randn(H3, HK, generator=g, device=DEV) * 0.02).float()
    scale = torch.tensor([1.1, 0.9, 1.05], device=DEV)
    base = (torch.randn(H3, generator=g, device=DEV) * 0.1)
    norm = (torch.rand(H, generator=g, device=DEV) + 0.5).bfloat16()
    res = torch.randn(T, N, H, generator=g, device=DEV).bfloat16()
    x = (torch.randn(T, H, generator=g, device=DEV) * 0.5).bfloat16()
    post = torch.sigmoid(torch.randn(T, N, 1, generator=g, device=DEV)) * 2
    comb = torch.rand(T, N, N, generator=g, device=DEV)
    comb = comb / comb.sum(-1, keepdim=True)
    return dict(fn=fn, scale=scale, base=base, norm=norm, res=res, x=x, post=post, comb=comb)


def fused_call(w, t):
    """One production mhc_fused_post_pre_tilelang call (model.py hc_fused_post_pre's exact arguments)."""
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    return mhc_fused_post_pre_tilelang(
        t["x"], t["res"], t["post"], t["comb"], w["fn"], w["scale"], w["base"],
        1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, w["norm"], 1e-5,
    )


_QW = None


def _qw_post_mean():
    global _QW
    if _QW is None:
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p = os.path.join(root, "glm53_prefill_quickwins.py")
        spec = importlib.util.spec_from_file_location("glm53_prefill_quickwins", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _QW = m
    return _QW.qw_post_mean


def post_mean_call(w, t):
    r = _qw_post_mean()(t["x"], t["res"], t["post"], t["comb"], False)
    assert r is not None, "qw_post_mean refused the layout"
    return r[1]


def main():
    torch.manual_seed(0)
    print(f"gpu {torch.cuda.get_device_name(0)} sms {torch.cuda.get_device_properties(0).multi_processor_count}")
    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
    for T in (13824, 6912, 4289, 1536):
        print(f"compute_num_split(T={T}) = {compute_num_split(64, HK, (T + 63) // 64)}")

    # ---- (b) shard numerics (before the graphs: fresh allocations, full-T vs halves) -------------------------
    print("\n== (b) mHC ops on the two rank halves vs the full-T run (same rows)")
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang, mhc_pre_tilelang
    for T in (13824, 6912, 1536):
        w = make(T)
        r_full = mhc_fused_post_pre_tilelang(
            w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"],
            1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, w["norm"], 1e-5)
        outs = []
        for lo in (0, T // 2):
            t = {k: (v[lo:lo + T // 2] if k in ("res", "x", "post", "comb") else v) for k, v in w.items()}
            outs.append(mhc_fused_post_pre_tilelang(
                t["x"], t["res"], t["post"], t["comb"], w["fn"], w["scale"], w["base"],
                1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, w["norm"], 1e-5))
        names = ("residual_cur", "post_mix", "comb_mix", "layer_input")
        for i, nm in enumerate(names):
            eq = all(torch.equal(r_full[i][lo:lo + T // 2], o[i]) for lo, o in ((0, outs[0]), (T // 2, outs[1])))
            worst = max((r_full[i][lo:lo + T // 2].float() - o[i].float()).abs().max().item()
                        for lo, o in ((0, outs[0]), (T // 2, outs[1])))
            print(f"   T={T:6d} fused {nm:<12s} bitwise={eq} maxabs={worst:.3e}")
        # layer-0 path: hc_pre (with hc_expand'd residual)
        res4 = w["res"]
        p0 = mhc_pre_tilelang(res4, w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0, 20,
                              norm_weight=w["norm"], norm_eps=1e-5)
        hs = []
        for lo in (0, T // 2):
            hs.append(mhc_pre_tilelang(res4[lo:lo + T // 2], w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0,
                                       20, norm_weight=w["norm"], norm_eps=1e-5))
        for i, nm in enumerate(("post_mix", "comb_mix", "layer_input")):
            eq = all(torch.equal(p0[i][lo:lo + T // 2], h[i]) for lo, h in ((0, hs[0]), (T // 2, hs[1])))
            worst = max((p0[i][lo:lo + T // 2].float() - h[i].float()).abs().max().item()
                        for lo, h in ((0, hs[0]), (T // 2, hs[1])))
            print(f"   T={T:6d} hc_pre {nm:<12s} bitwise={eq} maxabs={worst:.3e}")
        del w, r_full, outs, p0, hs
        torch.cuda.empty_cache()

    # ---- (a) halving (eager, production's prefill runs eager; alternating order each round) -------------------
    print("\n== (a) per-call time of the production mHC chain (eager, median of ROUNDS rounds)")
    variants = []
    for T in (13824, 6912, 4289):
        w = make(T)
        t2 = {k: (v[: T // 2] if k in ("res", "x", "post", "comb") else v) for k, v in w.items()}
        variants.append((f"T={T}", w, w))
        variants.append((f"T={T // 2} (SP shard)", w, t2))

    def one(label, w, t):
        if label.endswith("fused"):
            fused_call(w, t)
        else:
            post_mean_call(w, t)

    for _, w, t in variants:                     # warmup/JIT
        for _ in range(2):
            fused_call(w, t)
            post_mean_call(w, t)
    torch.cuda.synchronize()
    res = {}
    for r in range(ROUNDS):
        order = [(f"{n} fused", w, t) for n, w, t in variants]
        order += [(f"{n} mean", w, t) for n, w, t in variants]
        if r % 2:
            order.reverse()
        for n, w, t in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(REPS):
                one(n, w, t)
            e.record()
            torch.cuda.synchronize()
            res.setdefault(n, []).append(s.elapsed_time(e) * 1000.0 / REPS)

    def med(n):
        v = sorted(res[n])
        return v[len(v) // 2], (v[-1] - v[0]) / v[len(v) // 2]

    for T, lbl_full, lbl_half in ((13824, "T=13824", "T=6912 (SP shard)"), (4289, "T=4289", "T=2144 (SP shard)")):
        f, fs = med(f"{lbl_full} fused")
        m, ms = med(f"{lbl_full} mean")
        fh, _ = med(f"{lbl_half} fused")
        chunk, chunk_sp = 90 * f + 6 * m, 90 * fh + 6 * m
        print(f"   T={T:6d}: fused {f * 1000:7.0f} us (spread {fs * 100:4.1f}%), post_mean {m * 1000:6.0f} us "
              f"-> per-13,824-chunk mHC {chunk / 1000:7.1f} ms")
        print(f"          shard T={T // 2}: fused {fh * 1000:7.0f} us -> SP {chunk_sp / 1000:7.1f} ms "
              f"(ratio {chunk_sp / chunk:.3f}); saving {(chunk - chunk_sp) / 1000:.1f} ms/chunk")


if __name__ == "__main__":
    main()
