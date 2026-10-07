"""opt-kdamhc-rev: what the fused kernel's logits are closer to. References in fp64:
  R_bf16 = residual_cur as stored (bf16, the operand production's prefill GEMM and the next layer consume)
  R_fp32 = the unrounded new residual (post*x + sum_k comb*res, the decode branch's operand)
for production tf32, fused ROUND_A=0 and fused ROUND_A=1; plus whole-op timing of ROUND_A=0 vs 1 at 6,912 rows.
Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc_rev/probe_round_a.py"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "overlay"))
sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
import torch  # noqa: E402
from bench_mhc_halving import make  # noqa: E402


def main():
    import glm53_mhc_fused_ext as E
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    H = 4096
    for M in (6912, 2145):
        w = make(M, seed=7)
        res, x, post, comb, fn = w["res"], w["x"], w["post"].reshape(M, 4).contiguous(), w["comb"].contiguous(), w["fn"]
        rb = torch.empty_like(res)
        mhc_post_tilelang(comb.view(M, 4, 4), res, post, x.view(M, H), rb, 4, H)
        r32 = (post.double()[:, :, None] * x.double()[:, None, :]
               + torch.einsum("tkj,tkh->tjh", comb.double(), res.double()))           # unrounded new residual
        g_bf = rb.view(M, 4 * H).double() @ fn.double().t()
        g_32 = r32.view(M, 4 * H) @ fn.double().t()
        gt = torch.empty(1, M, 24, device="cuda"); st = torch.empty(1, M, device="cuda")
        tf32_hc_prenorm_gemm(rb.view(M, 4 * H), fn, gt, st, 1)
        outs = {"prod tf32": gt[0]}
        for ra in (0, 1):
            rc = torch.empty_like(res); g = torch.empty(1, M, 24, device="cuda"); s = torch.empty(1, M, device="cuda")
            E.post_prenorm(comb.view(M, 4, 4), res, post, x.view(M, H), fn, rc, g, s, 9, ra)
            assert torch.equal(rc, rb)
            outs[f"fused ROUND_A={ra}"] = g[0]

        def rel(a, b):
            return ((a.double() - b).norm() / b.norm()).item()
        print(f"M={M}: |R_bf16 - R_fp32| logits rel {rel(g_bf, g_32):.2e}")
        for k, v in outs.items():
            print(f"   {k:18s} vs fp64(R_bf16) {rel(v, g_bf):.2e}   vs fp64(R_fp32) {rel(v, g_32):.2e}", flush=True)
        # timing ROUND_A 0 vs 1 (kernel only)
        rc = torch.empty_like(res); g = torch.empty(1, M, 24, device="cuda"); s = torch.empty(1, M, device="cuda")
        t = {0: [], 1: []}
        for rnd in range(6):
            for ra in ((0, 1) if rnd % 2 == 0 else (1, 0)):
                for _ in range(2):
                    E.post_prenorm(comb.view(M, 4, 4), res, post, x.view(M, H), fn, rc, g, s, 9, ra)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(10):
                    E.post_prenorm(comb.view(M, 4, 4), res, post, x.view(M, H), fn, rc, g, s, 9, ra)
                b.record(); torch.cuda.synchronize()
                t[ra].append(a.elapsed_time(b) / 10)
        print(f"   kernel time ROUND_A=0 {statistics.median(t[0]):.3f} ms, ROUND_A=1 {statistics.median(t[1]):.3f} ms",
              flush=True)


if __name__ == "__main__":
    main()
