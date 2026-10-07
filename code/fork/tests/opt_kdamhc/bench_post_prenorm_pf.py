"""opt-kdamhc: register-prefetch variants (cfg 7..10) of the fused mHC post + prenorm kernel vs their base cfgs.

Per M: every output (residual_cur, the 24 dot products, sqrsum) of cfg 7/8/9/10 must be BITWISE the base cfg's
(1/0/3/6: the prefetch only moves loads), residual_cur bitwise mhc_post_tilelang's; timing interleaved, median of 25,
plus production's post + tf32 GEMM for reference. Exit 1 on failure.
Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/bench_post_prenorm_pf.py
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
PAIRS = ((1, 7), (0, 8), (3, 9), (6, 10))
FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def main():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    E = load(name="opt_mhc_post_prenorm_pf", sources=[os.path.join(ROOT, "kernels", "mhc_post_prenorm.cu")],
             extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    for M in [int(v) for v in os.environ.get("MS", "6912,3456,13824,2232,2145,37").split(",")]:
        w = make(M, seed=5)
        x, res, post, comb, fn = w["x"], w["res"], w["post"].view(M, NS).contiguous(), w["comb"].contiguous(), w["fn"]
        dev = x.device
        rc_p = torch.empty_like(res)
        g_p, s_p = torch.empty(1, M, H3, device=dev), torch.empty(1, M, device=dev)
        post_k = lambda: mhc_post_tilelang(comb, res, post, x, rc_p, NS, H)
        gemm_k = lambda: tf32_hc_prenorm_gemm(rc_p.view(M, NS * H), fn, g_p, s_p, 1)
        post_k(); gemm_k()
        outs = {}
        for ra in (0, 1):
            for cfg in sorted({c for p in PAIRS for c in p}):
                rc_f = torch.empty_like(res)
                g_f, s_f = torch.empty(1, M, H3, device=dev), torch.empty(1, M, device=dev)
                E.post_prenorm(comb, res, post, x, fn, rc_f, g_f, s_f, cfg, ra)
                outs[(ra, cfg)] = (rc_f, g_f, s_f)
            torch.cuda.synchronize()
            for b, p in PAIRS:
                rb, gb, sb = outs[(ra, b)]
                rp, gp, sp = outs[(ra, p)]
                check(torch.equal(rp, rc_p) and torch.equal(rp, rb) and torch.equal(gp, gb) and torch.equal(sp, sb),
                      f"M={M} round_a={ra} cfg {p} == cfg {b} bitwise (residual_cur also == mhc_post_tilelang)")
        # interleaved timing, round_a 0 (the default)
        fns = {"prod": lambda: (post_k(), gemm_k())}
        for cfg in sorted({c for p in PAIRS for c in p}):
            rc_f, g_f, s_f = outs[(0, cfg)]
            fns[cfg] = (lambda c=cfg, r=rc_f, g=g_f, s=s_f: E.post_prenorm(comb, res, post, x, fn, r, g, s, c, 0))
        ts = {k: [] for k in fns}
        for _ in range(3):
            for f in fns.values():
                f()
        torch.cuda.synchronize()
        for _ in range(25):
            for k, f in fns.items():
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); f(); b.record(); torch.cuda.synchronize()
                ts[k].append(a.elapsed_time(b))
        med = {k: statistics.median(v) for k, v in ts.items()}
        gbs = lambda ms: (M * (2 * NS * H * 2 + H * 2)) / (ms * 1e-3) / 1e9
        line = "  ".join(f"cfg{k} {med[k]:.3f}" for k in sorted(k for k in med if k != "prod"))
        print(f"M={M}: production post+GEMM {med['prod']:.3f} ms | {line} ms", flush=True)
        for b, p in PAIRS:
            print(f"   cfg {b} -> {p}: {med[b]:.3f} -> {med[p]:.3f} ms ({med[p] - med[b]:+.3f}; {gbs(med[b]):.0f} -> "
                  f"{gbs(med[p]):.0f} GB/s of residual+x read + residual_cur write)", flush=True)
    print("RESULT:", "ALL OK" if not FAIL else f"{len(FAIL)} FAIL", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
