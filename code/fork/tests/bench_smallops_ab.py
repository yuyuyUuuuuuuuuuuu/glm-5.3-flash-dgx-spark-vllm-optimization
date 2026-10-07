"""GLM53_DEC_SMALLOPS paired A/B of the ops as wired (docs/DEC_SMALLOPS.md), CUDA graphs, interleaved rounds:
  mhc    one decode step's 89 mhc_fused_post_pre calls on 89 distinct registered fp32 weights (cold from DRAM):
         A = production's function (mhc_fused_tilelang + mhc_pre_big_fuse_with_norm), B = torch.ops.vllm op with the
         GLM53_DEC_SMALLOPS override (smallops kernel on the bf16 copy + production's pre kernel)
  dconv  one drafter step's 20 grouped convs (5 layers x 2 convs x prepare/finish, T = 8 x reqs):
         A = production _grouped_conv, B = the installed wrapper (custom op glm53_so::dconv)
Run under the lock: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/bench_smallops_ab.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_smallops import ab, graph_of, report  # noqa: E402


def main():
    dev = torch.device("cuda", 0)
    os.environ["GLM53_DEC_SMALLOPS"] = "1"
    import glm53_smallops_install as I
    I.plugin_install()
    from vllm.model_executor.kernels.mhc import tilelang as W
    from vllm.model_executor.models import qwen3_dflash2 as D
    prod_mhc = W.mhc_fused_post_pre_tilelang
    prod_gc = D._grouped_conv
    # ---- mhc: 89 layers with registered weights
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
    model.layers.insert(0, torch.nn.Module())          # "layers.0" without hc params
    out = I.post_load(model)
    print(f"mhc: {out['mhc']['weights']} weights registered, {out['mhc']['bytes'] / 2**20:.2f} MiB", flush=True)
    norm = (torch.rand(4096, device=dev) + 0.5).bfloat16()
    for M in [int(m) for m in os.environ.get("BENCH_MS", "5,6,8,16").split(",")]:
        x = (torch.randn(M, 4096, device=dev) * 0.5).bfloat16()
        res = torch.randn(M, 4, 4096, device=dev).bfloat16()
        post = torch.sigmoid(torch.randn(M, 4, 1, device=dev)) * 2
        comb = torch.rand(M, 4, 4, device=dev)
        comb = comb / comb.sum(-1, keepdim=True)
        lays = [model.layers[i + 1] for i in range(L)]

        def a():
            for lay in lays:
                prod_mhc(x, res, post, comb, lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base, 1e-6, 1e-6, 1e-6, 2.0,
                         20, 1, 1, norm, 1e-5)

        def b():
            for lay in lays:
                torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, lay.hc_ffn_fn, lay.hc_ffn_scale,
                                                           lay.hc_ffn_base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
        s0 = I.COUNTERS["mhc_served"]
        r = ab({"production mhc op (fused + pre)": graph_of(a), "GLM53_DEC_SMALLOPS mhc op": graph_of(b)}, L)
        want = 3 * L if M <= I.MHC_SERVE_M_MAX else 0          # M > MHC_SERVE_M_MAX stays on production
        assert I.COUNTERS["mhc_served"] - s0 == want, "override served an unexpected number of calls"
        report(f"mHC fused post+pre op, M={M}, {L} distinct weights per graph (one decode step)", r,
               "production mhc op (fused + pre)")
    # ---- dconv
    H, gs, taps, G = 4096, 16, 2, 256
    LD = 20
    from safetensors import safe_open
    draft = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
    bks = []
    if os.path.isfile(os.path.join(draft, "model.safetensors")):
        with safe_open(os.path.join(draft, "model.safetensors"), framework="pt", device="cpu") as f:
            bks = [f.get_tensor(k).to(dev) for k in sorted(f.keys()) if k.endswith("base_kernel")]
    if not bks:
        bks = [(torch.randn(2, taps, H, device=dev) * 0.3).bfloat16() for _ in range(10)]
    Stand = type("DFlashGroupedConv", (torch.nn.Module,), {"__module__": D.__name__})
    m = Stand()
    m.base_kernel = torch.nn.Parameter(bks[0].clone(), requires_grad=False)
    m.block_size, m.taps, m.group_size, m.num_groups = 8, taps, gs, G
    holder = torch.nn.Module()
    holder.conv = m
    I.post_load(holder)
    wrap = D._grouped_conv
    assert getattr(wrap, "_glm53_so_hook", False)
    for T in (8, 16, 32):
        xs = [torch.randn(T, H, device=dev).bfloat16() for _ in range(LD)]
        cs = [(torch.randn(T, 2, taps, G, device=dev) * 0.3).bfloat16() for _ in range(LD)]
        bs = [bks[i % len(bks)][i % 2].contiguous() for i in range(LD)]

        def pa():
            for i in range(LD):
                prod_gc(xs[i], cs[i][:, i % 2], bs[i], 8, G, gs, taps)

        def pb():
            for i in range(LD):
                wrap(xs[i], cs[i][:, i % 2], bs[i], 8, G, gs, taps)
        r = ab({"production _grouped_conv": graph_of(pa), "GLM53_DEC_SMALLOPS dconv": graph_of(pb)}, LD)
        report(f"DFlash2 grouped conv as wired, T={T}, {LD} calls per graph (one drafter step)", r,
               "production _grouped_conv")
    print(f"counters {I.COUNTERS}")


if __name__ == "__main__":
    main()
