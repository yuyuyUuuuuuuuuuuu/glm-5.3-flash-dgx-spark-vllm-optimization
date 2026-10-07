"""Reviewer: mHC A/B in a decode-like context. Per layer: an L2-thrashing read kernel (stands in for the FP8 GEMV /
MoE weight streams between two mHC ops), then the mHC op whose residual/post/comb inputs are the PREVIOUS op's
outputs (chained, as in the model). A = production function, B = torch.ops.vllm op with the smallops override.
Reports per-call delta = (A - B) / 89. Filler is identical in A and B."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_smallops import ab, graph_of  # noqa: E402


def main():
    dev = torch.device("cuda", 0)
    p = torch.cuda.get_device_properties(dev)
    print(f"L2 {p.L2_cache_size / 2**20:.1f} MiB, SMs {p.multi_processor_count}")
    os.environ["GLM53_DEC_SMALLOPS"] = "1"
    import glm53_smallops_install as I
    import glm53_smallops as SO
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
    lays = [model.layers[i + 1] for i in range(L)]
    norm = (torch.rand(4096, device=dev) + 0.5).bfloat16()
    fill_mb = int(os.environ.get("FILL_MB", "32"))
    filler = torch.empty(fill_mb * 2**20 // 16, 4, dtype=torch.int32, device=dev)
    filler.random_()
    sink = torch.zeros(1, dtype=torch.int32, device=dev)
    ext = SO.load_ext()

    kind = os.environ.get("FILLER", "sum")
    fbuf = filler.view(torch.float32)

    def fill():
        if kind == "evict_last":
            ext.l2_prefetch(filler, sink, 48)      # streaming read with an L2 evict_last policy
        elif kind == "sum":
            torch.sum(fbuf)                        # streaming read, default cache policy (reduce kernel)
    for M in [int(m) for m in os.environ.get("BENCH_MS", "5,6,8").split(",")]:
        x = (torch.randn(M, 4096, device=dev) * 0.5).bfloat16()
        res0 = torch.randn(M, 4, 4096, device=dev).bfloat16()
        post0 = torch.sigmoid(torch.randn(M, 4, 1, device=dev)) * 2
        comb0 = torch.rand(M, 4, 4, device=dev)
        comb0 = comb0 / comb0.sum(-1, keepdim=True)

        def chain(op):
            def run():
                res, post, comb = res0, post0, comb0
                for lay in lays:
                    fill()
                    res, post, comb, _ = op(x, res, post, comb, lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base,
                                            1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
                return res
            return run

        def fonly():
            for _ in lays:
                fill()
        r = ab({"A": graph_of(chain(prod_mhc)), "B": graph_of(chain(torch.ops.vllm.mhc_fused_post_pre_tilelang)),
                "F": graph_of(fonly)}, L)
        print(f"M={M:2d} filler {kind} {fill_mb} MiB: A-F {r['A'][0] - r['F'][0]:6.2f} us  B-F {r['B'][0] - r['F'][0]:6.2f} us  "
              f"delta A-B {r['A'][0] - r['B'][0]:+6.2f} us/call  (A {r['A'][0]:.2f} sp {r['A'][1]*100:.1f}%, "
              f"B {r['B'][0]:.2f} sp {r['B'][1]*100:.1f}%, F {r['F'][0]:.2f})", flush=True)
        # correctness of the chained result, graph vs eager production
        a_out = chain(prod_mhc)()
        b_out = chain(torch.ops.vllm.mhc_fused_post_pre_tilelang)()
        print(f"   chained 89-layer residual bitwise: {torch.equal(a_out.view(torch.int16), b_out.view(torch.int16))}")
    print(I.COUNTERS)


if __name__ == "__main__":
    main()
