"""opt-kdamhc: the fused mHC post + prenorm GEMM CUDA kernel (kernels/mhc_post_prenorm.cu) vs production.

Per M (6,912 = the SP shard of a 13,824 chunk, 3,456 = an SP2 sub-chunk, 13,824 = TP, 2,232 = the 32k tail's SP
shard, 37 = a ragged small size):
  - residual_cur BITWISE == mhc_post_tilelang's;
  - the 24 dot products / sqrsum vs an fp64 reference of the decode-branch arithmetic (fp32 new_r, fp32 fn), next to
    production's prefill branch (deep_gemm tf32 on bf16 residual_cur) and decode branch (mhc_fused_tilelang);
  - pre_big_fuse on each GEMM output: post/comb/layer_input distance from the decode branch's (dvp-relevant) and
    from the prefill branch's;
  - time: post + tf32 GEMM (production) vs the fused kernel per cfg.
Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/bench_post_prenorm.py
"""
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
from bench_mhc_halving import make  # noqa: E402

H, NS, H3 = 4096, 4, 24
FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def tmed(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def ext():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    return load(name="opt_mhc_post_prenorm", sources=[os.path.join(ROOT, "kernels", "mhc_post_prenorm.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def main():
    E = ext()
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_tilelang, mhc_post_tilelang, mhc_pre_big_fuse_with_norm_tilelang)
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    for M in [int(v) for v in os.environ.get("MS", "6912,3456,13824,2232,37").split(",")]:
        w = make(M, seed=3)
        x, res, post, comb, fn = w["x"], w["res"], w["post"].view(M, NS).contiguous(), w["comb"].contiguous(), w["fn"]
        dev = x.device
        rc_p = torch.empty_like(res)
        g_p, s_p = torch.empty(1, M, H3, device=dev), torch.empty(1, M, device=dev)
        post_k = lambda: mhc_post_tilelang(comb, res, post, x, rc_p, NS, H)
        gemm_k = lambda: tf32_hc_prenorm_gemm(rc_p.view(M, NS * H), fn, g_p, s_p, 1)
        post_k(); gemm_k()
        rc_d = torch.empty_like(res)
        g_d, s_d = torch.empty(1, M, H3, device=dev), torch.empty(1, M, device=dev)
        mhc_fused_tilelang(comb, res, post, x, fn.view(H3, NS, H), g_d, s_d, rc_d, NS, H, H3, tile_n=12, n_splits=1)
        # fp64 reference of the decode arithmetic (fp32 new_r)
        nr = post.view(M, NS, 1).double() * x.view(M, 1, H).double() + torch.einsum(
            "tkj,tkh->tjh", comb.double(), res.double())
        ref = nr.reshape(M, NS * H) @ fn.double().t()
        ref_sq = nr.reshape(M, -1).pow(2).sum(-1)
        torch.cuda.synchronize()
        print(f"M={M}: prefill branch vs fp64 ref: gemm rel {rel(g_p[0], ref):.2e}, sqrsum rel {rel(s_p[0], ref_sq):.2e}; "
              f"decode branch: gemm rel {rel(g_d[0], ref):.2e}, sqrsum rel {rel(s_d[0], ref_sq):.2e}", flush=True)

        def pre(g, s, rc):
            o = (torch.empty(M, NS, device=dev), torch.empty(M, NS * NS, device=dev),
                 torch.empty(M, H, dtype=torch.bfloat16, device=dev))
            mhc_pre_big_fuse_with_norm_tilelang(g, s, w["scale"], w["base"], rc, o[0], o[1], o[2], w["norm"], H,
                                                1e-5, 1e-6, 1e-6, 2.0, 20, 1e-5, 1, NS)
            return o
        o_p, o_d = pre(g_p, s_p, rc_p), pre(g_d, s_d, rc_d)
        t_prod = tmed(post_k) + tmed(gemm_k)
        print(f"   production post+GEMM {t_prod:.3f} ms; prefill vs decode branch: layer_input rel "
              f"{rel(o_p[2], o_d[2]):.2e} ({100 * (o_p[2] != o_d[2]).float().mean().item():.1f}% elems differ), "
              f"post max {((o_p[0] - o_d[0]).abs().max().item()):.1e}", flush=True)
        for ra in (0, 1):
            for cfg in (0, 1, 2, 3, 4, 5, 6):
                rc_f = torch.empty_like(res)
                g_f, s_f = torch.empty(1, M, H3, device=dev), torch.empty(1, M, device=dev)
                fk = lambda: E.post_prenorm(comb, res, post, x, fn, rc_f, g_f, s_f, cfg, ra)
                fk(); torch.cuda.synchronize()
                check(torch.equal(rc_f, rc_p), f"M={M} round_a={ra} cfg={cfg}: residual_cur bitwise == mhc_post_tilelang")
                if cfg == 0:
                    o_f = pre(g_f, s_f, rc_f)
                    print(f"   fused round_a={ra}: gemm rel vs fp64 {rel(g_f[0], ref):.2e} (vs prefill {rel(g_f[0], g_p[0]):.2e},"
                          f" vs decode {rel(g_f[0], g_d[0]):.2e}); sqrsum rel {rel(s_f[0], ref_sq):.2e}; layer_input vs decode "
                          f"{rel(o_f[2], o_d[2]):.2e} ({100 * (o_f[2] != o_d[2]).float().mean().item():.1f}% differ), vs prefill "
                          f"{rel(o_f[2], o_p[2]):.2e} ({100 * (o_f[2] != o_p[2]).float().mean().item():.1f}% differ)", flush=True)
                tf = tmed(fk)
                print(f"   fused round_a={ra} cfg={cfg}: {tf:.3f} ms (saves {t_prod - tf:+.3f} ms/call)", flush=True)
        del w
    print(f"RESULT: {'ALL OK' if not FAIL else f'{len(FAIL)} FAILED'}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
