"""GLM53_DEC_SMALLOPS kernels vs the production ops they replace: bitwise equality (docs/DEC_SMALLOPS.md).

  dconv      vs vllm.model_executor.models.qwen3_dflash2._grouped_conv (the production DFlash2 file; identical to the
             launcher overlay's qwen3_dflash2.py, md5 196c5504...), T = 1..64, block_size 8 (and 5, 6, 7), taps 2
             (the checkpoint) and 1, 3; real DFlash2 base kernels + kernel_projection when the drafter is mounted;
             special values (+-0, subnormal-range, large) in the activations.
  mhc_fused  vs vllm.model_executor.kernels.mhc.tilelang_kernels.mhc_fused_tilelang called exactly as production's
             mhc_fused_post_pre_tilelang does (small-FMA path: tile_n 2 / splits 8 for M < 8, tile_n 3 / splits 4
             for 8 <= M <= 16), all three outputs (yp partials, rp partials, residual_cur); weights: real hc_attn_fn /
             hc_ffn_fn of several layers (GLM53_CKPT, bf16 in the checkpoint, fp32 parameter in vLLM) through both
             the fp32 and the bf16-copy kernel, plus random fp32 weights that are NOT bf16-representable (fp32 kernel).
Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4:$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c \
     TF_EXL3_JIT=1 tests/gpu_run.sh python3 tests/test_smallops_kernels.py
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glm53_smallops as SO  # noqa: E402

CKPT = os.environ.get("GLM53_CKPT", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-Uncensored-NVFP4"))
DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def ckpt_tensors(names):
    """name -> tensor from the local checkpoint shards (header scan, no full load)."""
    out = {}
    files = sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors")))
    if not files:
        return out
    from safetensors import safe_open
    want = set(names)
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        hit = [k for k in hdr if k in want]
        if not hit:
            continue
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in hit:
                out[k] = sf.get_tensor(k)
        if len(out) == len(want):
            break
    return out


# ---------------------------------------------------------------------------------------------------------------
def test_dconv(dev):
    from vllm.model_executor.models import qwen3_dflash2 as D
    prod = D._grouped_conv
    H, gs = 4096, 16
    G = H // gs
    gen = torch.Generator(device=dev).manual_seed(1)
    real = None
    if os.path.isfile(os.path.join(DRAFT, "model.safetensors")):
        from safetensors import safe_open
        with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if "conv" in k]
            real = {k: f.get_tensor(k) for k in keys}
        print(f"dconv: real drafter conv tensors: {len(real)} ({sorted(real)[:4]} ...)")
    n_cases = 0
    for taps in (2, 1, 3):
        bases = []
        if taps == 2 and real:
            for k, v in sorted(real.items()):
                if k.endswith("base_kernel"):
                    bases.append((k, v.to(dev)))
        if not bases:
            bases.append(("random", (torch.randn(2, taps, H, generator=gen, device=dev) * 0.3).bfloat16()))
        projs = {k[: -len("kernel_projection.weight")]: v.to(dev) for k, v in (real or {}).items()
                 if k.endswith("kernel_projection.weight")} if taps == 2 else {}
        for bname, base_all in bases[:6]:
            proj = projs.get(bname[: -len("base_kernel")]) if bname != "random" else None
            for T in (1, 2, 3, 5, 8, 13, 16, 24, 32, 40, 64):
                if T < taps:   # production's _grouped_conv itself fails there (F.pad of an empty slice); T = 8 * reqs
                    continue
                for bs in ((8, 5, 6, 7) if T in (16, 40) else (8,)):
                    x = (torch.randn(T, H, generator=gen, device=dev) * 2.0).bfloat16()
                    if T >= 3:   # special values
                        x[0, :16] = 0.0
                        x[1, :16] = -0.0
                        x[2, :8] = torch.tensor([1e-30, -1e-30, 3e38, -3e38, 1e-40, 65504., 1.0, -1.0])
                    if proj is not None:
                        coeff = torch.nn.functional.linear(x, proj).reshape(T, 2, taps, G)
                    else:
                        coeff = (torch.randn(T, 2, taps, G, generator=gen, device=dev) * 0.5).bfloat16()
                    for side in (0, 1):
                        ref = prod(x, coeff[:, side], base_all[side], bs, G, gs, taps)
                        got = SO.dconv(x, coeff[:, side], base_all[side].contiguous(), gs, bs)
                        n_cases += 1
                        if not bitwise_equal(ref, got):
                            d = (ref.float() - got.float()).abs()
                            neq = (ref.view(torch.int16) != got.view(torch.int16))
                            check(False, f"dconv taps={taps} base={bname} T={T} bs={bs} side={side}: "
                                         f"{int(neq.sum())} differing, max |d| {float(d.nan_to_num().max()):.3g}")
                            return
    check(True, f"dconv bitwise == production _grouped_conv on {n_cases} cases (taps 1/2/3, T 1..64, block 5..8, "
                f"real drafter base kernels{' + real kernel_projection' if real else ''})")


# ---------------------------------------------------------------------------------------------------------------
def test_mhc_fused(dev):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_tilelang
    hc, H = 4, 4096
    n3 = hc * (2 + hc)
    names = [f"model.language_model.layers.{i}.hc_{s}_fn" for i in (0, 1, 3, 10, 22, 44) for s in ("attn", "ffn")]
    real = ckpt_tensors(names)
    print(f"mhc: {len(real)} real hc_*_fn tensors from {CKPT}")
    weights = [(k.split("layers.")[1], v.float().to(dev)) for k, v in sorted(real.items())]
    gen = torch.Generator(device=dev).manual_seed(2)
    rnd = torch.randn(n3, hc * H, generator=gen, device=dev) * 0.02    # fp32, not bf16-representable
    weights.append(("random-fp32", rnd))
    for name, w in weights:
        exact_bf16 = bool(torch.equal(w, w.bfloat16().float()))
        if name != "random-fp32":
            check(exact_bf16, f"mhc weight {name}: every fp32 value is a bf16 value (bf16 copy exact)")
    total = 0
    for name, w in weights:
        variants = [("fp32", w)]
        if torch.equal(w, w.bfloat16().float()):
            variants.append(("bf16", w.bfloat16().contiguous()))
        for M in range(1, 17):
            tile_n = 2 if M < 8 else 3
            S = 8 if M < 8 else 4
            for trial in range(3):
                x = (torch.randn(M, H, generator=gen, device=dev) * 0.5).bfloat16()
                res = (torch.randn(M, hc, H, generator=gen, device=dev) * 1.5).bfloat16()
                post = torch.sigmoid(torch.randn(M, hc, generator=gen, device=dev)) * 2.0
                comb = torch.rand(M, hc, hc, generator=gen, device=dev)
                comb = comb / comb.sum(-1, keepdim=True)
                if trial == 2:
                    res[0, 0, :32] = 0.0
                    x[0, :32] = -0.0
                yp_ref = torch.empty(S, M, n3, device=dev)
                rp_ref = torch.empty(S, M, device=dev)
                ro_ref = torch.empty_like(res)
                mhc_fused_tilelang(comb, res, post, x, w.view(n3, hc, H), yp_ref, rp_ref, ro_ref, hc, H, n3,
                                   tile_n=tile_n, n_splits=S)
                for vname, wv in variants:
                    yp, rp, ro = SO.mhc_fused(comb, post, res, x, wv, S)
                    total += 1
                    ok = bitwise_equal(yp, yp_ref) and bitwise_equal(rp, rp_ref) and bitwise_equal(ro, ro_ref)
                    if not ok:
                        dy = (yp - yp_ref).abs().max().item()
                        dr = (rp - rp_ref).abs().max().item()
                        do = (ro.float() - ro_ref.float()).abs().max().item()
                        nd = int((yp.view(torch.int32) != yp_ref.view(torch.int32)).sum())
                        check(False, f"mhc_fused {name} {vname} M={M} trial={trial}: yp max|d| {dy:.3g} ({nd} of "
                                     f"{yp.numel()} differ) rp {dr:.3g} residual {do:.3g}")
                        return
    torch.cuda.synchronize()
    check(True, f"mhc_fused bitwise == production mhc_fused_tilelang (yp, rp, residual_cur) on {total} calls: "
                f"M 1..16, {len(weights)} weights (real hc_*_fn fp32 and bf16 copy, random fp32)")


def main():
    dev = torch.device("cuda", 0)
    SO.load_ext()
    print(f"ext {SO.EXT_SOURCE}")
    test_dconv(dev)
    test_mhc_fused(dev)
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
