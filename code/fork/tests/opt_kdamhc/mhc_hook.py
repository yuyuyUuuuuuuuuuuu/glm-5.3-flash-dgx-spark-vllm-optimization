"""opt-kdamhc (test rig): route the PREFILL branch of MHCFusedPostPreOp through kernels/mhc_post_prenorm.cu.

install(mode): mode "fused0" = decode-consistent dot products (fp32 new residual x fp32 fn, the arithmetic class of the
decode branch mhc_fused_tilelang), "fused1" = the bf16-rounded residual_cur x fp32 fn (the prefill branch's operand,
without the tf32 truncation). Token counts <= 16 (decode) and CUDA-graph captures keep production's op. The
pre_big_fuse kernel is production's, with n_splits = 1. Must run before the model is constructed (CustomOp binds
forward_cuda at __init__)."""
from __future__ import annotations

import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = {"ext": None, "calls": 0, "mode": None, "cfg": int(os.environ.get("OPT_MHC_CFG", "1"))}


def ext():
    if STATE["ext"] is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        STATE["ext"] = load(name="opt_mhc_post_prenorm", sources=[os.path.join(ROOT, "kernels", "mhc_post_prenorm.cu")],
                            extra_cuda_cflags=["-O3"], verbose=False)
    return STATE["ext"]


def fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, round_a: int):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_tilelang, mhc_pre_big_fuse_with_norm_tilelang)
    hc, H = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    res = residual.reshape(-1, hc, H).contiguous()
    T = res.shape[0]
    dev = res.device
    rc = torch.empty_like(res)
    g = torch.empty(1, T, 2 * hc + hc * hc, dtype=torch.float32, device=dev)
    s = torch.empty(1, T, dtype=torch.float32, device=dev)
    ext().post_prenorm(comb_res_mix.reshape(T, hc, hc).contiguous(), res, post_layer_mix.reshape(T, hc).contiguous(),
                       x.reshape(T, H).contiguous(), fn.contiguous(), rc, g, s, STATE["cfg"], round_a)
    pm = torch.empty(T, hc, dtype=torch.float32, device=dev)
    cm = torch.empty(T, hc * hc, dtype=torch.float32, device=dev)
    li = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(g, s, hc_scale, hc_base, rc, pm, cm, li, H, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                  hc_post_mult_value, sinkhorn_repeat, 1, hc)
    else:
        nw = norm_weight.to(torch.bfloat16).contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(g, s, hc_scale, hc_base, rc, pm, cm, li, nw, H, rms_eps, hc_pre_eps,
                                            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps, 1, hc)
    STATE["calls"] += 1
    return rc.view(*outer, hc, H), pm.view(*outer, hc, 1), cm.view(*outer, hc, hc), li.view(*outer, H)


def install(mode: str) -> None:
    if mode not in ("fused0", "fused1"):
        return
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    orig = MHCFusedPostPreOp.forward_cuda
    ra = 1 if mode == "fused1" else 0
    ext()
    STATE["mode"] = mode

    def forward_cuda(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                     hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                     norm_eps=0.0):
        T = residual.numel() // (residual.shape[-1] * residual.shape[-2])
        if T <= 16 or torch.cuda.is_current_stream_capturing() or residual.shape[-1] % 64:
            return orig(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps)
        return fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, ra)

    MHCFusedPostPreOp.forward_cuda = forward_cuda
