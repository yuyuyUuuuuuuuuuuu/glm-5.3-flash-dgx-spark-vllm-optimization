"""opt-kdamhc-rev: adversarial re-check of the fused mHC post+prenorm kernel through the AOT .so (the kit's binary, not
the JIT build the original test used). Exit 1 on failure.

A1  ragged / odd / SP-shard sizes: residual_cur BITWISE mhc_post_tilelang on ALL rows (not the 1,024-row engine check)
A2  heavy-tailed residual (outlier channels x300, a few rows x1e3) and comb with exact zeros: still bitwise
A3  determinism: two launches on the same inputs give identical residual_cur / logits / sqrsum (TP ranks in non-SP
    steps compute the same rows redundantly and must agree bit for bit)
A4  logits and sqrsum accuracy vs an fp64 reference, fused fp32 vs production tf32 (claim: fp32 is closer)
A5  interleaved timing of the whole op (production vs hooked), 5 rounds x 11 reps, medians per round
Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc_rev/test_mhc_fused_adv.py
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "overlay"))      # glm53_mhc_fused.py + the AOT glm53_mhc_fused_ext .so
sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
os.environ["GLM53_MHC_FUSED_MIN_T"] = "17"
import torch  # noqa: E402
from bench_mhc_halving import make  # noqa: E402

FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    import glm53_mhc_fused as F
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    import glm53_mhc_fused_ext
    print("ext:", glm53_mhc_fused_ext.__file__, flush=True)
    F.install()
    with set_current_vllm_config(VllmConfig()):
        op = MHCFusedPostPreOp()
    prod = F.STATE["orig"]
    H = 4096

    def args(w):
        return (w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0, 20,
                1, 1, w["norm"], 1e-5)

    def kernel_only(w, M):
        rc = torch.full_like(w["res"], float("nan"))
        g = torch.full((1, M, 24), float("nan"), device="cuda")
        s = torch.full((1, M), float("nan"), device="cuda")
        F.post_gemm(w["x"], w["res"], w["post"].view(M, 4), w["comb"], w["fn"], rc, g, s)
        return rc, g, s

    for M, heavy in ((1024, False), (1025, False), (1537, False), (2145, False), (4289, False), (6913, False),
                     (6912, True), (13824, True), (18432, False)):
        w = make(M, seed=11 + M)
        if heavy:
            r = w["res"].float()
            r[:, :, 17] *= 300.0
            r[:, 2, 4000] *= 300.0
            r[::997] *= 1e3
            w["res"] = r.bfloat16()
            c = w["comb"].clone()
            c[::5, 1, :] = 0.0
            w["comb"] = c
        rc, g, s = kernel_only(w, M)
        ref = torch.empty_like(w["res"])
        mhc_post_tilelang(w["comb"].reshape(M, 4, 4), w["res"], w["post"].reshape(M, 4), w["x"].reshape(M, H), ref,
                          4, H)
        torch.cuda.synchronize()
        tag = f"M={M}{' heavy' if heavy else ''}"
        check(torch.equal(rc, ref), f"A1/A2 {tag}: residual_cur bitwise mhc_post_tilelang on all {M} rows")
        rc2, g2, s2 = kernel_only(w, M)
        check(torch.equal(rc, rc2) and torch.equal(g, g2) and torch.equal(s, s2), f"A3 {tag}: deterministic")
        # A4 accuracy vs fp64 of the bf16 residual_cur (what production's GEMM consumes) and of the fp32 new residual
        n_splits = 1
        g_t = torch.empty(n_splits, M, 24, device="cuda")
        s_t = torch.empty(n_splits, M, device="cuda")
        tf32_hc_prenorm_gemm(ref.view(M, 4 * H), w["fn"], g_t, s_t, n_splits)
        fn64 = w["fn"].double()
        g64 = ref.view(M, 4 * H).double() @ fn64.t()
        s64 = (ref.view(M, 4 * H).double() ** 2).sum(-1)

        def rel(a, b):
            return ((a.double() - b).norm() / b.norm()).item()
        print(f"   {tag}: logits rel vs fp64(bf16 residual): fused {rel(g[0], g64):.2e}  prod tf32 {rel(g_t[0], g64):.2e};"
              f" sqrsum rel: fused {rel(s[0], s64):.2e} prod {rel(s_t[0], s64):.2e}", flush=True)
        check(bool(torch.isfinite(g).all()) and bool(torch.isfinite(s).all()), f"A4 {tag}: finite logits/sqrsum")
        del rc, rc2, g, g2, s, s2, ref, g64
        torch.cuda.empty_cache()
    # A5 interleaved timing at the production shapes
    for M in (6912, 13824, 4289, 2145):
        w = make(M, seed=5)
        a = args(w)
        for _ in range(3):
            prod(op, *a); op.forward_cuda(*a)
        torch.cuda.synchronize()
        rp, rf = [], []
        for rnd in range(5):
            for which, lst in ((0, rp), (1, rf)) if rnd % 2 == 0 else ((1, rf), (0, rp)):
                ts = []
                for _ in range(11):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    (prod(op, *a) if which == 0 else op.forward_cuda(*a))
                    e1.record(); torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1))
                lst.append(statistics.median(ts))
        print(f"   A5 M={M}: production {statistics.median(rp):.3f} ms (rounds {['%.3f' % x for x in rp]}), "
              f"fused {statistics.median(rf):.3f} ms (rounds {['%.3f' % x for x in rf]}), "
              f"delta {statistics.median(rf) - statistics.median(rp):+.3f} ms/call", flush=True)
    print(f"served calls {F.STATE['calls']}, production calls {F.STATE['prod']}, installed {F.STATE['installed']}")
    print(f"RESULT: {'ALL OK' if not FAIL else f'{len(FAIL)} FAILED'}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
