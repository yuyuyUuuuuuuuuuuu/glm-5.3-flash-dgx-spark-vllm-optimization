"""opt-kdamhc: fused mHC post + prenorm GEMM on the PREFILL branch of MHCFusedPostPreOp (GLM53_MHC_FUSED; prototype).

Production's prefill branch of vllm's mhc_fused_post_pre_tilelang (num_tokens > 16) runs mhc_post_tilelang (writes the
new residual, [T, 4, 4096] bf16) and then deep_gemm's tf32_hc_prenorm_gemm, which reads that residual straight back to
form the 24 mHC mixing logits + the row square sums; pre_big_fuse then turns them into post/comb/layer_input.
kernels/mhc_post_prenorm.cu computes the logits while the new residual is still on chip:
  * residual_cur: BITWISE mhc_post_tilelang's (same fp32 FMA order, bf16 RNE; tests/opt_kdamhc/post_arith_probe.py);
  * the logits: ROUND_A=0 (default, "decode-consistent") = fp32 new residual x fp32 fn with fp32 accumulation, the
    arithmetic of the DECODE branch (mhc_fused_tilelang). Production's prefill logits are tf32 (rel 1.7e-3 vs fp64; the
    decode branch 1.6e-7), so today a token's mHC mixing differs between its prefill and its decode in ~23 % of the
    layer_input elements; with ROUND_A=0 they agree in 100.0 % (bench_post_prenorm.py). GLM53_MHC_FUSED_ROUND_A=1 uses
    the bf16-rounded residual instead (the prefill operand without the tf32 truncation of fn).
  * pre_big_fuse(_with_norm) is production's kernel with n_splits = 1.
Speed (nodeC, production kernels vs this, cfg 9 = BM16/HT64 + register prefetch, bench_post_prenorm_pf.py):
post+GEMM 3.51 -> 2.61 ms at the SP shard M=6,912 (-0.90 ms/call, ~ -81 ms per 13,824-token chunk at 90 calls),
6.84 -> 5.22 ms at M=13,824 (TP), 1.74 -> 1.31 at 3,456, 1.10 -> 0.82 at 2,145 (the 32k tail's odd-T SP2 shard).

Served: eager prefill calls with num_tokens > 16, hc_mult 4, hidden % 64 == 0, fn [24, 4*hidden] fp32 contiguous;
decode (<= 16 tokens) and CUDA-graph capture keep production's op. First eligible call per process: a self-check on deterministic synthetic rows
(fixed seed, the layer's fn; opt-kdamhc-rev) compares residual_cur with mhc_post_tilelang bit for bit and the logits
with production's tf32 result (rel <= 1e-2); a failure uninstalls (logged).
Under GLM53_MHC_SP2 the pipelined helper _sp2_post_pre_into (model.py) mirrors the production op directly; install()
also re-points that helper when the model module carries it.

Kernel: the AOT extension glm53_mhc_fused_ext (kernels/mhc_post_prenorm.cu, built into overlay/ by
tools/mhcfused/build.py in the production image, -O3 WITHOUT --use_fast_math: TileLang does not flush denormals
either); GLM53_MHC_FUSED_JIT=1 (tests only) JIT-builds it.

Kit wiring: overlay/patch_mhc_fused.py (GLM53_MHC_FUSED=1) copies this module + the extension into site-packages and
arms integrate.plugin_register, which calls plugin_install() in every vLLM process (never raises). TP: the module has
no collective; each rank decides its self-check alone on identical synthetic inputs (deterministic: same .so, same
GPU type, replicated fn), so both ranks reach the same verdict. Under GLM53_MHC_SP2 the model
module is imported after the plugins load, so the SP2 helper is re-pointed lazily on the first call of the op (the
engine's warm-up/capture calls it before any real prefill).
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger("vllm.glm53_mhc_fused")
ENV = "GLM53_MHC_FUSED"
STATE = {"ext": None, "installed": False, "orig": None, "checked": False, "calls": 0, "prod": 0,
         "round_a": int(os.environ.get("GLM53_MHC_FUSED_ROUND_A", "0") or 0),
         # kernel tile config (kernels/mhc_post_prenorm.cu): 9 = BM16/HT64 with the next chunk prefetched into
         # registers (bench_post_prenorm_pf.py: 2.61 ms at 6,912 rows vs 3.02 for the first cut's cfg 1, 3.51 production)
         "cfg": int(os.environ.get("GLM53_MHC_FUSED_CFG", "9") or 9), "sp2": False, "sp2_done": False,
         # below ~1k rows production's op is as fast or faster (37 rows: 0.069 vs 0.219 ms; 2,232: 1.62 vs 1.49 ms)
         "min_t": int(os.environ.get("GLM53_MHC_FUSED_MIN_T", "1024") or 1024)}


def plugin_install() -> None:
    """Called from the armed integrate.plugin_register (every vLLM process); inert unless GLM53_MHC_FUSED=1."""
    try:
        if not env_enabled():
            return
        install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_mhc_fused NOT installed (production's mHC op unchanged): %r", exc)


def env_enabled(environ=None) -> bool:
    e = os.environ if environ is None else environ
    return (e.get(ENV, "") or "").strip().lower() in ("1", "on", "true", "yes")


def ext():
    if STATE["ext"] is None:
        try:
            import glm53_mhc_fused_ext as m  # AOT (kit)
        except ImportError:
            if os.environ.get("GLM53_MHC_FUSED_JIT", "") != "1":
                raise
            from torch.utils.cpp_extension import load
            here = os.path.dirname(os.path.abspath(__file__))
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
            m = load(name="glm53_mhc_fused_ext_jit", sources=[os.path.join(here, "kernels", "mhc_post_prenorm.cu")],
                     extra_cuda_cflags=["-O3"], verbose=False)
        STATE["ext"] = m
    return STATE["ext"]


def post_gemm(x, residual, post, comb, fn, residual_cur, gemm_out, sqrsum):
    """residual_cur [T,4,H] (written), gemm_out [1,T,24] fp32, sqrsum [1,T] fp32 from the inputs of mhc_post."""
    T = residual.shape[0]
    ext().post_prenorm(comb.reshape(T, 4, 4), residual, post.reshape(T, 4), x.reshape(T, -1), fn,
                       residual_cur, gemm_out, sqrsum, STATE["cfg"], STATE["round_a"])


def _eligible(x, residual, post, comb, fn) -> bool:
    import torch
    if residual.dim() < 2 or residual.shape[-2] != 4 or residual.shape[-1] % 64:
        return False
    H = residual.shape[-1]
    T = residual.numel() // (4 * H)
    if T <= 16 or T < STATE["min_t"] or torch.cuda.is_current_stream_capturing():
        return False
    return (residual.dtype == torch.bfloat16 and x.dtype == torch.bfloat16 and fn.dtype == torch.float32
            and post.dtype == torch.float32 and comb.dtype == torch.float32 and tuple(fn.shape) == (24, 4 * H)
            and residual.is_contiguous() and x.is_contiguous() and fn.is_contiguous() and post.is_contiguous()
            and comb.is_contiguous() and residual.is_cuda)


CHECK_ROWS = 1024


def _first_check(H, fn, dev):
    """The one-time self-check, on DETERMINISTIC synthetic inputs (opt-kdamhc-rev): CHECK_ROWS rows from a fixed-seed
    device generator, with this layer's real fn. residual_cur must be bitwise mhc_post_tilelang's and the logits within
    1e-2 of production's tf32 GEMM. The verdict depends only on the kernel binary, the GPU and fn - not on the
    activations of whichever call came first - so every TP rank reaches the same verdict (fn is replicated). The
    first cut checked the first rows of the first finite call: under GLM53_MHC_SP each rank checked a different shard,
    and a one-rank uninstall would have left the ranks computing replicated rows (non-SP steps) with different
    arithmetic. It also no longer waits for a finite call (the engine's dummy profile run)."""
    import torch
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang
    T = CHECK_ROWS
    g0 = torch.Generator(device=dev).manual_seed(0x6D6863)
    residual = torch.randn(T, 4, H, generator=g0, device=dev).bfloat16()
    x = (torch.randn(T, H, generator=g0, device=dev) * 0.5).bfloat16()
    post = (torch.sigmoid(torch.randn(T, 4, generator=g0, device=dev)) * 2).contiguous()
    comb = torch.rand(T, 4, 4, generator=g0, device=dev)
    comb = (comb / comb.sum(-1, keepdim=True)).contiguous()
    rc = torch.empty_like(residual)
    g = torch.empty(1, T, 24, dtype=torch.float32, device=dev)
    s = torch.empty(1, T, dtype=torch.float32, device=dev)
    post_gemm(x, residual, post, comb, fn, rc, g, s)
    ref = torch.empty_like(residual)
    mhc_post_tilelang(comb, residual, post, x, ref, 4, H)
    g_ref = torch.empty_like(g)
    s_ref = torch.empty(1, T, dtype=torch.float32, device=dev)
    try:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
        tf32_hc_prenorm_gemm(ref.view(T, 4 * H), fn, g_ref, s_ref, 1)
    except Exception:  # noqa: BLE001 - no deep_gemm: fp32 reference
        g_ref[0] = ref.view(T, 4 * H).float() @ fn.t()
    rel = ((g.double() - g_ref.double()).norm() / g_ref.double().norm().clamp_min(1e-30)).item()
    same = torch.equal(rc, ref)
    ok = same and bool(torch.isfinite(g).all()) and rel <= 1e-2
    _log.info("glm53_mhc_fused: self-check (%d synthetic rows, fixed seed, this layer's fn): residual_cur %s "
              "mhc_post_tilelang, logits rel %.2e vs production's tf32 GEMM -> %s", T, "==" if same else "!=", rel,
              "serving" if ok else "UNINSTALLED (production's op serves)")
    return ok


def fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps, out=None):
    """The prefill branch of mhc_fused_post_pre_tilelang with the post + GEMM fused. out = optional preallocated
    (residual_cur, post_mix_cur [T,4], comb_mix_cur [T,16], layer_input_cur [T,H]) row slices (the SP2 helper)."""
    import torch
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_tilelang, mhc_pre_big_fuse_with_norm_tilelang)
    hc, H = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    res = residual.reshape(-1, hc, H)
    T = res.shape[0]
    dev = res.device
    if out is None:
        rc = torch.empty_like(res)
        pm = torch.empty(T, hc, dtype=torch.float32, device=dev)
        cm = torch.empty(T, hc * hc, dtype=torch.float32, device=dev)
        li = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    else:
        rc, pm, cm, li = out
    g = torch.empty(1, T, 2 * hc + hc * hc, dtype=torch.float32, device=dev)
    s = torch.empty(1, T, dtype=torch.float32, device=dev)
    if not STATE["checked"]:
        STATE["checked"] = True
        if not _first_check(H, fn, dev):
            uninstall()
            return None
    post_gemm(x, res, post_layer_mix, comb_res_mix, fn, rc, g, s)
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(g, s, hc_scale, hc_base, rc, pm, cm, li, H, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                  hc_post_mult_value, sinkhorn_repeat, 1, hc)
    else:
        nw = norm_weight if (norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()) else \
            norm_weight.to(torch.bfloat16).contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(g, s, hc_scale, hc_base, rc, pm, cm, li, nw, H, rms_eps, hc_pre_eps,
                                            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps, 1, hc)
    STATE["calls"] += 1
    return (rc.view(*outer, hc, H), pm.view(*outer, hc, 1), cm.view(*outer, hc, hc), li.view(*outer, H))


def install() -> bool:
    if STATE["installed"]:
        return True
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    ext()
    orig = MHCFusedPostPreOp.forward_cuda

    def forward_cuda(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                     hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                     norm_eps=0.0):
        if not STATE["sp2_done"] and STATE["installed"]:
            _install_sp2()             # lazy: the model module is imported after the plugins load
        if STATE["installed"] and _eligible(x, residual, post_layer_mix, comb_res_mix, fn):
            r = fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                               hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_weight,
                               norm_eps)
            if r is not None:
                return r
        STATE["prod"] += 1
        return orig(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps)

    forward_cuda._glm53_mhc_fused = True
    STATE["orig"] = orig
    MHCFusedPostPreOp.forward_cuda = forward_cuda
    STATE["installed"] = True
    _install_sp2()
    _log.info("glm53_mhc_fused: installed (prefill mHC post + prenorm GEMM fused, logits %s; decode/capture keep "
              "production's op%s)", "fp32 decode-consistent" if STATE["round_a"] == 0 else "bf16-residual x fp32",
              "; SP2 helper re-pointed" if STATE["sp2"] else "")
    return True


def _install_sp2() -> None:
    """GLM53_MHC_SP2: model.py's _sp2_post_pre_into mirrors the production op; route its post + GEMM here too."""
    import sys
    mod = sys.modules.get("vllm.models.glm5next.nvidia.model")
    if mod is None:
        return                         # not imported yet: retried on the op's next call
    STATE["sp2_done"] = True           # the model module is final now: decide once
    if not hasattr(mod, "_sp2_post_pre_into") or getattr(mod._sp2_post_pre_into, "_glm53_mhc_fused", False):
        return
    orig = mod._sp2_post_pre_into

    def _sp2_post_pre_into(x, residual, post, comb, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                           hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
                           residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur):
        if STATE["installed"] and _eligible(x, residual, post, comb, fn) and residual_cur.is_contiguous():
            r = fused_post_pre(x, residual, post, comb, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                               hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
                               out=(residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur))
            if r is not None:
                return
        orig(x, residual, post, comb, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
             hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
             residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)

    _sp2_post_pre_into._glm53_mhc_fused = True
    mod._sp2_post_pre_into = _sp2_post_pre_into
    STATE["sp2"] = True
    _log.info("glm53_mhc_fused: GLM53_MHC_SP2 helper _sp2_post_pre_into re-pointed (prefill shards fused too)")


def uninstall() -> None:
    if not STATE["installed"]:
        return
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    if STATE["orig"] is not None:
        MHCFusedPostPreOp.forward_cuda = STATE["orig"]
    STATE["installed"] = False
