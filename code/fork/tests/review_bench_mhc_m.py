"""Reviewer: as-wired mHC A/B for every M 1..16 (same method as bench_smallops_ab.py, mhc only).
REVIEW_SERVE_M_MAX=16 reproduces the pre-review wiring (override served M <= 16).
Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh bash -c 'BENCH_ROUNDS=15 python3 tests/review_bench_mhc_m.py'"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_smallops import ab, graph_of  # noqa: E402


def main():
    dev = torch.device("cuda", 0)
    os.environ["GLM53_DEC_SMALLOPS"] = "1"
    import glm53_smallops_install as I
    I.plugin_install()
    from vllm.model_executor.kernels.mhc import tilelang as W
    prod_mhc = W.mhc_fused_post_pre_tilelang
    L = 89
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for i in range(L):
        lay = torch.nn.Module()
        lay.hc_ffn_fn = torch.nn.Parameter((torch.randn(24, 16384, device=dev) * 0.02).bfloat16().float(),
                                           requires_grad=False)
        lay.hc_ffn_scale = torch.nn.Parameter(torch.rand(3, device=dev), requires_grad=False)
        lay.hc_ffn_base = torch.nn.Parameter(torch.randn(24, device=dev) * 0.1, requires_grad=False)
        model.layers.append(lay)
    model.layers.insert(0, torch.nn.Module())
    I.post_load(model)
    if os.environ.get("REVIEW_SERVE_M_MAX"):        # measure the kernel beyond the shipped range (pre-review: 16)
        I.MHC_SERVE_M_MAX = int(os.environ["REVIEW_SERVE_M_MAX"])
    print(f"served M range 1..{I.MHC_SERVE_M_MAX}")
    norm = (torch.rand(4096, device=dev) + 0.5).bfloat16()
    lays = [model.layers[i + 1] for i in range(L)]
    Ms = [int(m) for m in os.environ.get("BENCH_MS", ",".join(map(str, range(1, 17)))).split(",")]
    print("M  prod_us  so_us  delta_us(+ = faster)  spreadA spreadB")
    for M in Ms:
        x = (torch.randn(M, 4096, device=dev) * 0.5).bfloat16()
        res = torch.randn(M, 4, 4096, device=dev).bfloat16()
        post = torch.sigmoid(torch.randn(M, 4, 1, device=dev)) * 2
        comb = torch.rand(M, 4, 4, device=dev)
        comb = comb / comb.sum(-1, keepdim=True)

        def a():
            for lay in lays:
                prod_mhc(x, res, post, comb, lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base, 1e-6, 1e-6, 1e-6, 2.0,
                         20, 1, 1, norm, 1e-5)

        def b():
            for lay in lays:
                torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, lay.hc_ffn_fn, lay.hc_ffn_scale,
                                                           lay.hc_ffn_base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
        r = ab({"A": graph_of(a), "B": graph_of(b)}, L)
        print(f"{M:2d} {r['A'][0]:8.2f} {r['B'][0]:6.2f} {r['A'][0] - r['B'][0]:+8.2f}   {r['A'][1]*100:5.1f}% {r['B'][1]*100:5.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
