"""Reviewer's adversarial checks for GLM53_DEC_SMALLOPS (independent of the implementer's tests).

mhc (installed override, called as the model calls it: torch.ops.vllm.mhc_fused_post_pre_tilelang) vs production's
Python function (_ORIG), all four outputs bitwise:
  a) every real hc_*_fn available in the local checkpoint, chained like the model (outputs of layer i feed layer i+1),
     M = 1..16, three input scales (1x, 300x large residual, 1e-3x small)
  b) CUDA graph captured once, replayed on 6 fresh data sets copied into the static inputs
  c) layout edge cases: outer shape (2, 3); 2-D post; fp32 norm_weight; non-contiguous x; misaligned residual;
     M = 0; fn view with other strides
  d) fp64 reference for the kernel part (split-summed partials and residual_cur) at M = 5: rel_l2 / max-abs of
     production and of the override (identical when bitwise)
dconv (installed wrapper through the module global, like DFlashGroupedConv._convolve):
  e) real DFlash2 base kernels + real kernel_projection coefficients as production lays them out; T 8..40 incl. T not
     a multiple of the block; strided x rows; CUDA graph replay on fresh data; x with inf/nan (NaN positions equal)
memory: reserved/allocated delta of the mhc bf16 copies under the allocator config in PYTORCH_CUDA_ALLOC_CONF.
"""
import glob
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []
CKPT = os.environ.get("GLM53_CKPT", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-Uncensored-NVFP4"))
DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def beq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {torch.bfloat16: torch.int16, torch.float32: torch.int32}.get(a.dtype)
    return torch.equal(a.view(iv), b.view(iv)) if iv else torch.equal(a, b)


def all_beq(r1, r2):
    return all(beq(a, b) for a, b in zip(r1, r2))


def ckpt_hc():
    out = {}
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        hit = [k for k in hdr if ".hc_" in k and "layers." in k]
        if hit:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for k in hit:
                    out[k] = sf.get_tensor(k)
    return out


def mhc_tests(dev, I, prod):
    real = ckpt_hc()
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in real})
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for li in range(max(layers) + 1):
        lay = torch.nn.Module()
        for s in ("attn", "ffn"):
            for t in ("fn", "scale", "base"):
                k = f"model.language_model.layers.{li}.hc_{s}_{t}"
                if k in real:
                    setattr(lay, f"hc_{s}_{t}", torch.nn.Parameter(real[k].float().to(dev), requires_grad=False))
        model.layers.append(lay)
    torch.cuda.synchronize()
    a0, r0 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
    out = I.post_load(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a1, r1 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
    nw = out["mhc"]["weights"]
    print(f"mhc: {nw} weights registered from layers {layers}; allocated +{(a1 - a0) / 2**20:.2f} MiB, reserved "
          f"+{(r1 - r0) / 2**20:.2f} MiB (alloc conf {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}); "
          f"per weight {(a1 - a0) / max(nw, 1) / 2**20:.3f} MiB")
    chain = []
    for li, lay in enumerate(model.layers):
        for s in ("attn", "ffn"):
            if hasattr(lay, f"hc_{s}_fn") and not (li == 0 and s == "attn"):
                chain.append((f"{li}.{s}", getattr(lay, f"hc_{s}_fn"), getattr(lay, f"hc_{s}_scale"),
                              getattr(lay, f"hc_{s}_base")))
    gen = torch.Generator(device=dev).manual_seed(123)
    norm = (torch.rand(4096, generator=gen, device=dev) * 0.8 + 0.6).bfloat16()
    op = torch.ops.vllm.mhc_fused_post_pre_tilelang
    # a) chained real weights
    bad = []
    n = 0
    s_before = I.COUNTERS["mhc_served"]
    for scale_r, scale_x in ((1.0, 1.0), (300.0, 20.0), (1e-3, 1e-3)):
        for M in range(1, 17):
            x0 = (torch.randn(M, 4096, generator=gen, device=dev) * scale_x).bfloat16()
            res = (torch.randn(M, 1, 4096, generator=gen, device=dev) * scale_r).expand(M, 4, 4096).contiguous()
            res = (res + torch.randn(M, 4, 4096, generator=gen, device=dev) * 0.05 * scale_r).bfloat16()
            post = torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2.0
            comb = torch.rand(M, 4, 4, generator=gen, device=dev)
            comb = comb / comb.sum(-1, keepdim=True)
            rp_, pp_, cp_ = res, post, comb
            for name, fn, sc, bs in chain:
                args = (x0, rp_, pp_, cp_, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
                ref = I._ORIG["mhc"](*args)
                got = op(*args)
                n += 1
                if not all_beq(ref, got):
                    bad.append((scale_r, M, name))
                    break
                rp_, pp_, cp_, li = ref
                x0 = (li.float() * scale_x).bfloat16()          # next layer's input derived from this one
    served = I.COUNTERS["mhc_served"] - s_before
    check(not bad, f"a) chained real weights ({len(chain)} ops/chain), M 1..16, 3 scales: {n} calls bitwise, "
                   f"{served} served; bad {bad[:5]}")
    # b) graph replay on fresh data
    name, fn, sc, bs = chain[len(chain) // 2]
    ok = True
    for M in (1, 5, 6, 7, 8, 9, 12, 15, 16):
        sx = torch.zeros(M, 4096, dtype=torch.bfloat16, device=dev)
        sr = torch.zeros(M, 4, 4096, dtype=torch.bfloat16, device=dev)
        sp = torch.zeros(M, 4, 1, device=dev)
        scm = torch.zeros(M, 4, 4, device=dev)

        def f():
            return op(sx, sr, sp, scm, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gout = f()
        for trial in range(6):
            sx.copy_((torch.randn(M, 4096, generator=gen, device=dev) * (trial + 1)).bfloat16())
            sr.copy_((torch.randn(M, 4, 4096, generator=gen, device=dev) * 3 ** trial).bfloat16())
            sp.copy_(torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2)
            c = torch.rand(M, 4, 4, generator=gen, device=dev)
            scm.copy_(c / c.sum(-1, keepdim=True))
            g.replay()
            ref = I._ORIG["mhc"](sx, sr, sp, scm, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5)
            torch.cuda.synchronize()
            if not all_beq(ref, gout):
                ok = False
                print(f"   graph replay mismatch M={M} trial={trial}")
    check(ok, "b) CUDA graph captured once, replayed on 6 fresh data sets per M (1,5..9,12,15,16): bitwise")

    # c) layout edge cases (each: override result == production result, or both raise the same exception type)
    def same(call_a, call_b):
        try:
            ra = call_a()
        except Exception as e:  # noqa: BLE001
            ra = e
        try:
            rb = call_b()
        except Exception as e:  # noqa: BLE001
            rb = e
        if isinstance(ra, Exception) or isinstance(rb, Exception):
            return type(ra) is type(rb), f"{type(ra).__name__}/{type(rb).__name__}"
        return all_beq(ra, rb), "tensors"
    M = 6
    x = (torch.randn(M, 4096, generator=gen, device=dev)).bfloat16()
    res = (torch.randn(M, 4, 4096, generator=gen, device=dev) * 2).bfloat16()
    post = torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2
    comb = torch.rand(M, 4, 4, generator=gen, device=dev)
    comb = comb / comb.sum(-1, keepdim=True)
    cases = {}
    base_args = [x, res, post, comb, fn, sc, bs, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, norm, 1e-5]

    def mk(**kw):
        a = list(base_args)
        for i, v in kw.items():
            a[int(i[1:])] = v
        return a
    cases["outer (2,3)"] = mk(a0=x.view(2, 3, 4096), a1=res.view(2, 3, 4, 4096), a2=post.view(2, 3, 4, 1),
                              a3=comb.view(2, 3, 4, 4))
    cases["2-D post"] = mk(a2=post.view(M, 4))
    cases["fp32 norm_weight"] = mk(a14=norm.float())
    big = torch.randn(M, 2, 4096, generator=gen, device=dev).bfloat16()
    cases["non-contig x"] = mk(a0=big[:, 0, :])
    flat = torch.zeros(M * 4 * 4096 + 8, dtype=torch.bfloat16, device=dev)
    flat[1:1 + M * 4 * 4096].copy_(res.flatten())
    cases["misaligned residual"] = mk(a1=flat[1:1 + M * 4 * 4096].view(M, 4, 4096))
    cases["M=0"] = mk(a0=x[:0], a1=res[:0], a2=post[:0], a3=comb[:0])
    cases["fn transposed-view"] = mk(a4=fn.t().contiguous().t())
    for cname, a in cases.items():
        s0 = dict(I.COUNTERS)
        ok, how = same(lambda: I._ORIG["mhc"](*a), lambda: op(*a))
        sv = I.COUNTERS["mhc_served"] - s0["mhc_served"]
        check(ok, f"c) {cname}: override == production ({how}), served {sv}")
    # d) fp64 reference of the kernel part at M = 5
    import glm53_smallops as SO
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_tilelang
    M = 5
    x = (torch.randn(M, 4096, generator=gen, device=dev)).bfloat16()
    res = (torch.randn(M, 4, 4096, generator=gen, device=dev) * 4).bfloat16()
    post = torch.sigmoid(torch.randn(M, 4, 1, generator=gen, device=dev)) * 2
    comb = torch.rand(M, 4, 4, generator=gen, device=dev)
    comb = comb / comb.sum(-1, keepdim=True)
    yp_p = torch.empty(8, M, 24, device=dev)
    rp_p = torch.empty(8, M, device=dev)
    ro_p = torch.empty_like(res)
    mhc_fused_tilelang(comb, res, post.view(M, 4), x, fn.view(24, 4, 4096), yp_p, rp_p, ro_p, 4, 4096, 24,
                       tile_n=2, n_splits=8)
    wb = I._MHC_W[fn.data_ptr()][2]
    yp_s, rp_s, ro_s = SO.mhc_fused(comb.view(M, 16), post.view(M, 4), res, x, wb, 8)
    nr = post.double().view(M, 4, 1) * x.double().view(M, 1, 4096) + torch.einsum("mkj,mkh->mjh", comb.double(),
                                                                                  res.double())
    y64 = torch.einsum("njh,mjh->mn", fn.double().view(24, 4, 4096), nr)
    r64 = (nr * nr).sum((1, 2))

    def err(a, ref):
        d = a.double() - ref
        return float(d.norm() / ref.norm()), float(d.abs().max())
    e_yp, e_ys = err(yp_p.sum(0), y64), err(yp_s.sum(0), y64)
    e_rp, e_rs = err(rp_p.sum(0), r64), err(rp_s.sum(0), r64)
    e_op, e_os = err(ro_p, nr), err(ro_s, nr)
    print(f"d) fp64 ref M=5: partial-sum yp rel_l2/maxabs prod {e_yp[0]:.3e}/{e_yp[1]:.3e} so {e_ys[0]:.3e}/{e_ys[1]:.3e}; "
          f"sqrsum prod {e_rp[0]:.3e} so {e_rs[0]:.3e}; residual_cur(bf16) prod {e_op[0]:.3e}/{e_op[1]:.3e} "
          f"so {e_os[0]:.3e}/{e_os[1]:.3e}")
    check(beq(yp_p, yp_s) and beq(rp_p, rp_s) and beq(ro_p, ro_s), "d) kernel part bitwise at M=5 (fp64 errors equal)")


def dconv_tests(dev, I):
    from safetensors import safe_open
    from vllm.model_executor.models import qwen3_dflash2 as D
    with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
        bks = {k: f.get_tensor(k).to(dev) for k in f.keys() if k.endswith("base_kernel")}
        prj = {k: f.get_tensor(k).to(dev) for k in f.keys() if k.endswith("kernel_projection.weight")}
    cfg = json.load(open(os.path.join(DRAFT, "config.json")))
    dc = cfg.get("dflash_config", {})
    taps, gs = int(dc.get("conv_kernel_size", 2)), int(dc.get("conv_group_size", 16))
    H = cfg["hidden_size"]
    G = H // gs
    print(f"dconv: H {H} taps {taps} gs {gs}, {len(bks)} base kernels")
    Stand = type("DFlashGroupedConv", (torch.nn.Module,), {"__module__": D.__name__})
    mods = torch.nn.Module()
    for i, (k, v) in enumerate(sorted(bks.items())):
        m = Stand()
        m.base_kernel = torch.nn.Parameter(v.clone(), requires_grad=False)
        m.block_size, m.taps, m.group_size, m.num_groups = 8, taps, gs, G
        m.kernel_projection_w = prj[k[: -len("base_kernel")] + "kernel_projection.weight"]
        mods.add_module(f"c{i}", m)
    orig = D._grouped_conv
    out = I.post_load(mods)
    check(getattr(D._grouped_conv, "_glm53_so_hook", False), f"e) dconv wired ({out['dconv']} modules)")
    gen = torch.Generator(device=dev).manual_seed(7)
    ok, n, nan_ok = True, 0, True
    for m in mods.children():
        for T in (8, 13, 16, 21, 24, 32, 40):
            hs_big = (torch.randn(T, H + 64, generator=gen, device=dev) * 1.5).bfloat16()
            for hs in (hs_big[:, :H].contiguous(), hs_big[:, :H], hs_big[:, 64:]):
                coefficients = torch.nn.functional.linear(hs, m.kernel_projection_w).reshape(T, 2, taps, G)
                for side in (0, 1):
                    ref = orig(hs, coefficients[:, side], m.base_kernel[side], 8, G, gs, taps)
                    got = D.DFlashGroupedConv._convolve(m, hs, coefficients[:, side], side)
                    n += 1
                    ok &= beq(ref, got)
        hs = (torch.randn(16, H, generator=gen, device=dev)).bfloat16()
        hs[3, 5] = float("inf")
        hs[8, 7] = float("nan")
        hs[7, 9] = float("-inf")                 # t=7 is the previous block's last row for t=8 (pos 0, masked)
        coefficients = torch.nn.functional.linear(torch.nan_to_num(hs), m.kernel_projection_w).reshape(16, 2, taps, G)
        ref = orig(hs, coefficients[:, 0], m.base_kernel[0], 8, G, gs, taps)
        got = D.DFlashGroupedConv._convolve(m, hs, coefficients[:, 0], 0)
        nan_ok &= torch.equal(torch.isnan(ref), torch.isnan(got)) and torch.equal(
            torch.nan_to_num(ref.float(), nan=0.0), torch.nan_to_num(got.float(), nan=0.0))
    check(ok, f"e) dconv via DFlashGroupedConv._convolve bitwise, real base + real projection layout, strided rows, "
              f"T not multiple of block: {n} calls")
    check(nan_ok, "e) dconv with inf/nan in x: same NaN positions, other values bitwise")
    m = next(iter(mods.children()))
    T = 8
    sx = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    sco = torch.zeros(T, 2 * taps * G, dtype=torch.bfloat16, device=dev)

    def f():
        return D.DFlashGroupedConv._convolve(m, sx, sco.view(T, 2, taps, G)[:, 1], 1)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gout = f()
    ok = True
    for trial in range(6):
        sx.copy_((torch.randn(T, H, generator=gen, device=dev) * (trial + 1)).bfloat16())
        sco.copy_(torch.nn.functional.linear(sx, m.kernel_projection_w))
        g.replay()
        ref = orig(sx, sco.view(T, 2, taps, G)[:, 1], m.base_kernel[1], 8, G, gs, taps)
        torch.cuda.synchronize()
        ok &= beq(ref, gout)
    check(ok, "e) dconv CUDA graph captured once, replayed on 6 fresh data sets: bitwise")


def main():
    dev = torch.device("cuda", 0)
    os.environ["GLM53_DEC_SMALLOPS"] = "1"
    import glm53_smallops_install as I
    I.plugin_install()
    from vllm.model_executor.kernels.mhc import tilelang as W
    prod = W.mhc_fused_post_pre_tilelang
    dconv_tests(dev, I)
    mhc_tests(dev, I, prod)
    print(f"counters {I.COUNTERS}")
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
