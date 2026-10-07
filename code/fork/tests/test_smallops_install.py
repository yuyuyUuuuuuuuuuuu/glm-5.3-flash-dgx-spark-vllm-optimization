"""GLM53_DEC_SMALLOPS installer end to end (docs/DEC_SMALLOPS.md), on nodeC in the production image:

  - off: plugin_install() with the env unset touches nothing (no op, no loader hook).
  - dconv: post_load on a stand-in drafter conv module (the real class's module and name, real DFlash2 base kernels)
    replaces qwen3_dflash2._grouped_conv; DFlashGroupedConv._convolve then returns bitwise production's result
    (eager, inside a CUDA graph, and through torch.compile(fullgraph=True)); fallback inputs (T < taps, fp32) go to
    production's function; uninstall restores the original.
  - mhc: post_load on stand-in decoder layers holding the real hc_{attn,ffn}_fn / scale / base of checkpoint layers
    (fp32 parameters, like vLLM's) registers bf16 copies and overrides vllm::mhc_fused_post_pre_tilelang; the op called
    as production calls it (torch.ops.vllm...) returns bitwise production's four outputs for M = 1..32 (<= 8 served,
    > 8 production: MHC_SERVE_M_MAX), eager and in a CUDA graph and via torch.compile(fullgraph=True); an unregistered fn, a weight
    modified after the copy (_version) and norm_weight=None go to production; uninstall restores production's kernel.
Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4:$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c \
     TF_EXL3_JIT=1 tests/gpu_run.sh python3 tests/test_smallops_install.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_smallops_kernels import DRAFT, ckpt_tensors  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def beq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def graph_run(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return out


def test_off():
    import glm53_smallops_install as I
    os.environ.pop(I.ENV, None)
    I.plugin_install()
    check(not I._STATE["ops"] and not I._STATE["loader"], "env unset: nothing registered, loader not hooked")


def test_dconv(dev):
    import glm53_smallops_install as I
    from vllm.model_executor.models import qwen3_dflash2 as D
    orig = D._grouped_conv
    from safetensors import safe_open
    with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
        bk = f.get_tensor("layers.0.attention_conv.base_kernel").to(dev)
    taps, H = bk.shape[1], bk.shape[2]
    Stand = type("DFlashGroupedConv", (torch.nn.Module,), {"__module__": D.__name__})
    m = Stand()
    m.base_kernel = torch.nn.Parameter(bk.clone(), requires_grad=False)
    m.block_size, m.taps, m.group_size = 8, taps, 16
    m.num_groups = H // 16
    model = torch.nn.Module()
    model.conv = m
    out = I.post_load(model)
    check(out["dconv"] == 1 and getattr(D._grouped_conv, "_glm53_so_hook", False),
          f"dconv wired on a drafter conv module ({out})")
    gen = torch.Generator(device=dev).manual_seed(5)
    n = 0
    ok = True
    for T in (8, 16, 24, 32, 5, 3):
        x = torch.randn(T, H, generator=gen, device=dev).bfloat16()
        coeff = (torch.randn(T, 2, taps, H // 16, generator=gen, device=dev) * 0.5).bfloat16()
        for side in (0, 1):
            ref = orig(x, coeff[:, side], m.base_kernel[side], 8, H // 16, 16, taps)
            got = D.DFlashGroupedConv._convolve(m, x, coeff[:, side], side)
            gg = graph_run(lambda: D.DFlashGroupedConv._convolve(m, x, coeff[:, side], side))
            ok &= beq(ref, got) and beq(ref, gg)
            n += 2
    check(ok, f"DFlashGroupedConv._convolve bitwise == production (eager + CUDA graph), {n} calls; "
              f"counters {I.COUNTERS}")
    served = I.COUNTERS["dconv_served"]
    x1 = torch.randn(1, H, device=dev).bfloat16()
    c1 = torch.randn(1, 2, taps, H // 16, device=dev).bfloat16()
    try:
        ref = orig(x1, c1[:, 0], m.base_kernel[0], 8, H // 16, 16, taps)
    except Exception as e:  # noqa: BLE001
        ref = e
    try:
        got = D._grouped_conv(x1, c1[:, 0], m.base_kernel[0], 8, H // 16, 16, taps)
    except Exception as e:  # noqa: BLE001
        got = e
    same = (isinstance(ref, Exception) and isinstance(got, Exception) and type(ref) is type(got)) or \
        (isinstance(ref, torch.Tensor) and isinstance(got, torch.Tensor) and beq(ref, got))
    check(same and I.COUNTERS["dconv_served"] == served,
          f"T=1 < taps goes to production (same result/exception: {type(ref).__name__})")
    xf = torch.randn(8, H, device=dev)
    cf = torch.randn(8, taps, H // 16, device=dev)
    check(torch.equal(D._grouped_conv(xf, cf, m.base_kernel[0].float(), 8, H // 16, 16, taps),
                      orig(xf, cf, m.base_kernel[0].float(), 8, H // 16, 16, taps)),
          "fp32 input goes to production")

    x = torch.randn(8, H, device=dev).bfloat16()
    c = (torch.randn(8, 2, taps, H // 16, device=dev) * 0.5).bfloat16()

    def f(x, c):
        return D._grouped_conv(x, c[:, 1], m.base_kernel[1], 8, H // 16, 16, taps) * 1.0

    try:
        cf_ = torch.compile(f, fullgraph=True, dynamic=False)
        got = cf_(x, c)
        ref = orig(x, c[:, 1], m.base_kernel[1], 8, H // 16, 16, taps) * 1.0
        check(beq(ref, got), "torch.compile(fullgraph=True) through the dconv wrapper: bitwise == production")
    except Exception as e:  # noqa: BLE001
        check(False, f"torch.compile(fullgraph=True) through the dconv wrapper: {type(e).__name__}: {e}"[:400])
    I.uninstall()
    check(D._grouped_conv is orig, "uninstall restores qwen3_dflash2._grouped_conv")


def test_mhc(dev):
    import glm53_smallops_install as I
    from vllm.model_executor.kernels.mhc import tilelang as W
    prod = W.mhc_fused_post_pre_tilelang
    names = [f"model.language_model.layers.{i}.hc_{s}_{t}" for i in (0, 1, 10) for s in ("attn", "ffn")
             for t in ("fn", "scale", "base")]
    real = ckpt_tensors(names)
    print(f"mhc: {len(real)} real tensors")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for i in (0, 1, 10):
        lay = torch.nn.Module()
        for s in ("attn", "ffn"):
            for t in ("fn", "scale", "base"):
                k = f"model.language_model.layers.{i}.hc_{s}_{t}"
                if k in real:
                    setattr(lay, f"hc_{s}_{t}", torch.nn.Parameter(real[k].float().to(dev), requires_grad=False))
        model.layers.append(lay)
    # layer indices in the module tree are 0, 1, 2 (checkpoint 0, 1, 10): "layers.0" = checkpoint layer 0
    before = torch.cuda.memory_allocated(dev)
    out = I.post_load(model)
    mib = (torch.cuda.memory_allocated(dev) - before) / 2**20
    nw = out["mhc"]["weights"] if out["mhc"] else 0
    check(nw == 5 and I._STATE["mhc_override"], f"mhc: {nw} weights registered (layer 0 attn skipped), op overridden, "
                                               f"+{mib:.2f} MiB ({out['rejected']})")
    gen = torch.Generator(device=dev).manual_seed(9)
    norm = (torch.rand(4096, generator=gen, device=dev) + 0.5).bfloat16()
    lay = model.layers[1]
    fn, sc, bs = lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base
    ok, bad = True, []
    served0 = I.COUNTERS["mhc_served"]
    for M in list(range(1, 17)) + [20, 24, 32]:
        x = (torch.randn(M, 4096, generator=gen, device=dev) * 0.5).bfloat16()
        res = (torch.randn(M, 4, 4096, generator=gen, device=dev)).bfloat16()
        post = torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2.0
        comb = torch.rand(M, 4, 4, generator=gen, device=dev)
        comb = comb / comb.sum(-1, keepdim=True)
        args = (x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
        ref = prod(*args)
        got = torch.ops.vllm.mhc_fused_post_pre_tilelang(*args)
        gg = graph_run(lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(*args))
        for a, b, c in zip(ref, got, gg):
            if not (beq(a, b) and beq(a, c)):
                ok = False
                bad.append(M)
                break
    served = I.COUNTERS["mhc_served"] - served0
    nmax = I.MHC_SERVE_M_MAX
    check(ok, f"torch.ops.vllm.mhc_fused_post_pre_tilelang bitwise == production, M 1..16 + 20/24/32, eager + CUDA "
              f"graph (bad M {bad}); served {served} calls (M <= {nmax} only), counters {I.COUNTERS}")
    check(served == nmax * 3, f"served exactly the M <= {nmax} calls (eager, graph warmup, capture): {served} == "
                              f"{nmax * 3}")

    M = 5
    x = (torch.randn(M, 4096, generator=gen, device=dev) * 0.5).bfloat16()
    res = (torch.randn(M, 4, 4096, generator=gen, device=dev)).bfloat16()
    post = torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2.0
    comb = torch.rand(M, 4, 4, generator=gen, device=dev)
    comb = comb / comb.sum(-1, keepdim=True)

    def f(x, res, post, comb):
        r, p, c, li = torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6,
                                                                  2.0, 20, 1, 1, norm, 1e-5)
        return r, p, c, li * 1.0

    try:
        got = torch.compile(f, fullgraph=True, dynamic=False)(x, res, post, comb)
        ref = prod(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
        check(all(beq(a, b) for a, b in zip(ref, got)), "torch.compile(fullgraph=True) calling the op: bitwise")
    except Exception as e:  # noqa: BLE001
        check(False, f"torch.compile(fullgraph=True) calling the op: {type(e).__name__}: {e}"[:400])

    s0 = dict(I.COUNTERS)
    other = fn.clone()
    torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, other, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1,
                                               norm, 1e-5)
    torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1,
                                               None, 1e-5)
    check(I.COUNTERS["mhc_served"] == s0["mhc_served"] and I.COUNTERS["mhc_prod"] == s0["mhc_prod"] + 2,
          "unregistered fn and norm_weight=None -> production")
    with torch.no_grad():
        fn.mul_(1.0)                              # bumps _version (in-place write)
    r2 = torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1,
                                                    norm, 1e-5)
    rp = prod(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
    check(I.COUNTERS["mhc_served"] == s0["mhc_served"] and all(beq(a, b) for a, b in zip(r2, rp)),
          "weight modified in place after the copy -> production")
    I.uninstall()
    s1 = dict(I.COUNTERS)
    torch.ops.vllm.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1,
                                               norm, 1e-5)
    check(I.COUNTERS == s1, "uninstall: the op runs production's kernel again (override gone)")


def main():
    dev = torch.device("cuda", 0)
    test_off()
    os.environ["GLM53_DEC_SMALLOPS"] = "1"
    import glm53_smallops_install as I
    I.plugin_install()
    check(I._STATE["ops"] and I._STATE["loader"], "env on: op registered, loader hooked")
    test_dconv(dev)
    test_mhc(dev)
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
