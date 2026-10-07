"""Correctness of kernels/gemv_bf16.cu (docs/BF16_GEMV.md): every M 1..64 on the production shapes, every plan
knob (S, wn, kch), all out modes, strided x, determinism, ticket counters left at zero, CUDA-graph replay.

Error bound: |y - ref64| <= tol * sum_k |x_k w_k| (the fp32-accumulation bound, ref64 = float64 GEMM), and the
same metric for cuBLAS (F.linear) is printed next to it. bf16 outputs are compared after rounding ref64 to bf16
(at most 1 bf16 ulp apart)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from harness import Checks, run_main  # noqa: E402
import glm53_bf16_gemv as G  # noqa: E402

SHAPES = [  # name, N, K
    ("router", 288, 4096), ("idx_wk_wp", 160, 4096), ("idx_kpool_gate", 128, 4096), ("idx_wq_b", 4096, 1536),
    ("draft_conv", 1024, 4096), ("draft_ctx_kv", 5120, 4096), ("draft_fc", 4096, 20480),
]


def rel_err(y: torch.Tensor, ref: torch.Tensor, mag: torch.Tensor) -> float:
    return float(((y.double() - ref).abs() / mag.clamp_min(1e-30)).max())


def main() -> None:
    ck = Checks()
    ext = G.load_ext()
    print("ext", G.EXT_SOURCE, "version", ext.version())
    dev = torch.device("cuda")
    torch.manual_seed(0)
    for name, N, K in SHAPES:
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ctx = G.GemmCtx(N, K, dev)
        w64 = w.double()
        ms = range(1, 65) if N * K <= 4096 * 4096 else (1, 5, 8, 9, 16, 17, 33, 40, 64)
        worst, worst_cub = 0.0, 0.0
        for M in ms:
            xb = torch.randn(M, K + 64, device=dev).to(torch.bfloat16)
            x = xb[:, 32:32 + K]   # strided rows, 16-byte aligned (32 * 2 B = 64 B offset)
            ref = x.double() @ w64.t()
            mag = x.double().abs() @ w64.abs().t()
            y1 = G.gemm(x, w, ctx, out_mode=1)
            e = rel_err(y1, ref, mag)
            worst = max(worst, e)
            worst_cub = max(worst_cub, rel_err(torch.nn.functional.linear(x, w).float(), ref, mag))
            ck(e < 2e-6, f"{name} M={M} fp32 out rel err {e:.3g}")
            y0 = G.gemm(x, w, ctx, out_mode=0)
            d0 = (y0.float() - ref.to(torch.bfloat16).float()).abs()
            ulp = ref.abs().to(torch.bfloat16).float() * 2 ** -7 + (mag * 4e-6).float()
            ck(bool((d0 <= ulp).all()), f"{name} M={M} bf16 out more than 1 ulp from ref")
            y2 = G.gemm(x, w, ctx, out_mode=2)
            ck(torch.equal(y2, y1.to(torch.bfloat16).float()), f"{name} M={M} out_mode 2 != bf16(fp32 out)")
            ck(torch.equal(y0.float(), y2), f"{name} M={M} out_mode 0 != out_mode 2")
            ck(int(ctx.counters.abs().sum()) == 0, f"{name} M={M} counters not reset")
        print(f"{name:15s} N={N:5d} K={K:5d}: max rel err (vs sum|xw|) new {worst:.3g}  cuBLAS {worst_cub:.3g}")

    # every plan knob on two shapes, a few M: all must give the same fp32 result up to summation order,
    # and each plan must be bitwise deterministic run to run
    for (name, N, K) in (("router", 288, 4096), ("idx_wq_b", 4096, 1536)):
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ctx = G.GemmCtx(N, K, dev)
        ctx.ws = torch.zeros(64 * 64 * N, device=dev)
        ctx.counters = torch.zeros(N // 8, dtype=torch.int32, device=dev)
        n_plans = 0
        for M in (1, 7, 8, 12, 16, 31, 48, 64):
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            ref = x.double() @ w.double().t()
            mag = x.double().abs() @ w.double().abs().t()
            for S, wn, kch in itertools.product((1, 2, 3, 4, 8, 16), (1, 2, 4, 8), (128, 256, 512)):
                if not G._valid(N, K, S, wn, kch, M):
                    continue
                y = G.gemm(x, w, ctx, out_mode=1, plan=(S, wn, kch))
                y_again = G.gemm(x, w, ctx, out_mode=1, plan=(S, wn, kch))
                e = rel_err(y, ref, mag)
                ck(e < 2e-6, f"{name} M={M} plan {(S, wn, kch)} rel err {e:.3g}")
                ck(torch.equal(y, y_again), f"{name} M={M} plan {(S, wn, kch)} not deterministic")
                ck(int(ctx.counters.abs().sum()) == 0, f"{name} M={M} plan {(S, wn, kch)} counters not reset")
                n_plans += 1
        print(f"{name}: {n_plans} (M, plan) combinations OK")

    # fp32 head gate vs production's torch.mm(x.float(), W.float().t())
    N, K = 32, 4096
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    wt32 = w.t().contiguous().float()
    ctx = G.GemmCtx(N, K, dev, f32=True)
    worst, worst_cub = 0.0, 0.0
    for M in range(1, 65):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        mag = x.double().abs() @ w.double().abs().t()
        y = G.gemm_f32(x, w, ctx)
        y_again = G.gemm_f32(x, w, ctx)
        e = rel_err(y, ref, mag)
        worst = max(worst, e)
        worst_cub = max(worst_cub, rel_err(torch.mm(x.float(), wt32), ref, mag))
        ck(e < 1e-6, f"head gate M={M} rel err {e:.3g}")
        ck(torch.equal(y, y_again), f"head gate M={M} not deterministic")
        ck(int(ctx.counters.abs().sum()) == 0, f"head gate M={M} counters not reset")
    print(f"head gate fp32 N=32 K=4096: max rel err new {worst:.3g}  cuBLAS fp32 {worst_cub:.3g}")

    # informational: bf16 outputs vs the exactly rounded value (round_bf16(float64 GEMM)), cuBLAS vs the kernel
    for name, N, K in SHAPES[:5]:
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ctx = G.GemmCtx(N, K, dev)
        for M in (8, 32):
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            exact = (x.double() @ w.double().t()).to(torch.bfloat16)
            cub = torch.nn.functional.linear(x, w)
            new = G.gemm(x, w, ctx, out_mode=0)

            def ulps(a):   # |a - exact| in units of the bf16 ulp at |exact| (outputs with |exact| > 1e-3)
                e = exact.float()
                keep = e.abs() > 1e-3
                ulp = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(1e-30))) - 7)
                return ((a.float() - e).abs() / ulp)[keep].round()
            print(f"  {name:15s} M={M:2d}: outputs != round_bf16(exact): cuBLAS {float((cub != exact).float().mean()):6.2%}"
                  f" (max {int(ulps(cub).max())} ulp), kernel {float((new != exact).float().mean()):6.2%}"
                  f" (max {int(ulps(new).max())} ulp)")

    # CUDA graph capture + replay with new inputs (x copied into the captured buffer)
    N, K = 288, 4096
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ctx = G.GemmCtx(N, K, dev)
    for M in (5, 8, 40, 64):
        xs = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            G.gemm(xs, w, ctx, out_mode=2)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg = G.gemm(xs, w, ctx, out_mode=2)
        for _ in range(3):
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            xs.copy_(x)
            gr.replay()
            torch.cuda.synchronize()
            ck(torch.equal(yg, G.gemm(x, w, ctx, out_mode=2)), f"graph replay M={M} != eager")
        ck(int(ctx.counters.abs().sum()) == 0, f"graph M={M} counters not reset")
    print("graph replay OK")
    ck.summary()


if __name__ == "__main__":
    run_main(main)
