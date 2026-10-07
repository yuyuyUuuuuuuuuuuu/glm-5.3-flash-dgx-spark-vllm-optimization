"""GLM53_DEC_SMALLOPS: serve two small decode ops with bit-identical faster kernels (docs/DEC_SMALLOPS.md).

Enable: GLM53_DEC_SMALLOPS in {1, on, true, yes} (read once, when the vLLM plugin loads; unset = nothing happens: no op is
registered, no module or op is touched). Revert: unset it and restart vLLM.
Sub-switch (only read when enabled):
  GLM53_DEC_SMALLOPS_KINDS   comma list, default all: dconv, mhc

  dconv  DFlash2 drafter grouped dynamic conv (qwen3_dflash2._grouped_conv; 4 calls per drafter layer = 20 per step):
         production's 10 PyTorch eager kernels -> 1 kernel (glm53_smallops.dconv), bitwise the same bf16 result.
         Wiring: the module global _grouped_conv of the drafter's module is replaced by a wrapper that calls the custom
         op glm53_so::dconv (opaque to torch.compile, fake impl); the op falls back to the original function for any
         input the kernel does not take (dtype, layout, alignment, T < taps). Installed only when the original function
         is the fingerprinted version the kernel was verified against, and after a self-test on the real base kernels.
  mhc    mHC fused post-mapping + pre-norm GEMM partials (vllm mhc_fused_tilelang, decode M <= 8, 89 calls per step):
         glm53_smallops.mhc_fused reads a bf16 copy of each layer's hc_{attn,ffn}_fn (GLM's values are bf16 values
         stored as fp32: the copy is exact, checked per weight) once per CTA for all tokens; same FMA order, same
         reductions -> bitwise production's partials and residual. The following mhc_pre_big_fuse_with_norm kernel is
         production's own. Wiring: the CUDA kernel of the custom op vllm::mhc_fused_post_pre_tilelang is overridden
         (torch.library, graphs traced by torch.compile are unchanged); the override serves a call only when fn is a
         registered weight (same tensor, unmodified since the copy: _version), M <= 8, norm_weight given, the expected
         shapes/dtypes/layout; everything else (prefill, layer 0's standalone pre, unknown weights) runs production's
         function with exactly the arguments received. Memory: 0.75 MiB per weight (89 weights = 66.75 MiB per rank).

Self-tests (post-load, on the real weights, both ranks the same since weights and inputs are identical): a kind whose
self-test fails is not installed and production's path stays. Decisions never depend on the rank; outputs are bitwise
production's either way, so collectives can never diverge.
torch.compile cache: vllm_config.additional_config["glm53_dec_smallops"] gets a tag (version, kinds), so no graph traced
in one mode is reused in the other (the dconv wrapper is traced when the drafter is compiled).
"""
import logging
import os
import sys
import threading

import torch

_log = logging.getLogger("vllm.glm53_dec_smallops")
_TRUE = frozenset({"1", "on", "true", "yes"})
ENV = "GLM53_DEC_SMALLOPS"
ENV_KINDS = "GLM53_DEC_SMALLOPS_KINDS"
ALL_KINDS = ("dconv", "mhc")
VERSION = 1

# Production functions the wiring re-implements or replaces (integrate.source_fingerprint = sha256 of ast.dump).
# Measured in the image (vllm 0.1.dev20051+g487ecf187) and the launcher overlay qwen3_dflash2.py (md5 196c5504...).
FP_GROUPED_CONV = frozenset({"238266efc04dc1cd"})          # qwen3_dflash2._grouped_conv
FP_MHC_FUSED_POST_PRE = frozenset({"4924eeb1bfbe2394"})    # kernels/mhc/tilelang.mhc_fused_post_pre_tilelang

MHC_HC, MHC_HID, MHC_N3 = 4, 4096, 24
# Largest M the mHC override serves. The kernel is bitwise for M <= 16, but it is only faster for M <= 8 (review
# 2026-09-28, nodeC paired A/B of the whole op, 89 cold weights, us/call production - override: M 1..8 +3.35 +2.47
# +2.27 +2.04 +2.53 +2.36 +2.91 +0.79; M 9..14 -1.44 -0.74 -0.36 -0.04 -0.68 -0.28 (slower); M 15/16 +0.71/+0.67;
# decode-like context with an L2-thrashing read between ops: M=10 -0.50). Production M 10/12 (2 requests) would
# lose, so M > 8 stays on production's kernel.
MHC_SERVE_M_MAX = 8

_LOCK = threading.Lock()
_STATE: dict = {"ops": False, "loader": False, "tag": None, "kinds": (), "mhc_override": False, "dconv_mods": set(),
                "logged": set()}
_ORIG: dict = {}
_LIBS: list = []                  # torch.library.Library objects: must stay alive while their registrations are used
_MHC_W: dict = {}                 # fn.data_ptr() -> (fn, fn._version, bf16 copy, name)
COUNTERS = {"mhc_served": 0, "mhc_prod": 0, "dconv_served": 0, "dconv_prod": 0}


def env_enabled(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


def env_kinds(environ=None) -> tuple[str, ...]:
    env = os.environ if environ is None else environ
    v = env.get(ENV_KINDS)
    if v is None or not v.strip():
        return ALL_KINDS
    ks = tuple(k.strip() for k in v.split(",") if k.strip())
    bad = [k for k in ks if k not in ALL_KINDS]
    if bad:
        raise ValueError(f"{ENV_KINDS}: unknown kind(s) {bad}; known {ALL_KINDS}")
    return ks


def _once(key, level, msg, *args) -> None:
    if key in _STATE["logged"]:
        return
    _STATE["logged"].add(key)
    _log.log(level, msg, *args)


def _fingerprint(fn):
    import integrate
    return integrate.source_fingerprint(getattr(fn, "_glm53_so_orig", fn))


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


# ---------------------------------------------------------------------------------------------------------------
# dconv: custom op + module-global wrapper

def _dconv_takes(x, delta, base, gs: int, block_size: int) -> bool:
    if not (x.is_cuda and x.dtype == torch.bfloat16 and delta.dtype == torch.bfloat16 and base.dtype == torch.bfloat16):
        return False
    if x.dim() != 2 or delta.dim() != 3 or base.dim() != 2:
        return False
    T, H = x.shape
    taps = base.shape[0]
    if not (1 <= taps <= 4) or gs < 1 or block_size < 1 or H % 8 or H % gs:
        return False
    if T < max(1, taps):             # production's F.pad path differs there: keep its exact behaviour
        return False
    if tuple(delta.shape) != (T, taps, H // gs) or delta.stride(2) != 1 or base.shape[1] != H:
        return False
    if x.stride(1) != 1 or x.stride(0) % 8 or not base.is_contiguous():
        return False
    return _aligned(x) and _aligned(base) and delta.device == x.device and base.device == x.device


def _dconv_op_impl(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, gs: int, block_size: int) -> torch.Tensor:
    if _dconv_takes(x, delta, base, gs, block_size):
        import glm53_smallops as SO
        COUNTERS["dconv_served"] += 1
        return SO.dconv(x, delta, base, gs, block_size)
    COUNTERS["dconv_prod"] += 1
    _once(("dconv_prod", tuple(x.shape), str(x.dtype)), logging.INFO,
          "glm53_dec_smallops: dconv input %s %s -> production _grouped_conv", tuple(x.shape), x.dtype)
    taps = base.shape[0]
    return _ORIG["grouped_conv"](x, delta, base, block_size, x.shape[1] // gs, gs, taps)


def _dconv_op_fake(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, gs: int, block_size: int) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)


def _grouped_conv_so(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    """Drop-in for qwen3_dflash2._grouped_conv (same signature, bitwise the same result)."""
    if (hidden_states.dim() == 2 and base.dim() == 2 and base.shape[0] == taps
            and hidden_states.shape[-1] == num_groups * group_size):
        return torch.ops.glm53_so.dconv(hidden_states, delta, base, int(group_size), int(block_size))
    return _ORIG["grouped_conv"](hidden_states, delta, base, block_size, num_groups, group_size, taps)


_grouped_conv_so._glm53_so_hook = True


def register_ops() -> None:
    if _STATE["ops"]:
        return
    from torch.library import Library

    from vllm.utils.torch_utils import direct_register_custom_op
    lib = Library("glm53_so", "FRAGMENT")
    _LIBS.append(lib)
    direct_register_custom_op(op_name="dconv", op_func=_dconv_op_impl, mutates_args=[], fake_impl=_dconv_op_fake,
                              target_lib=lib)
    _STATE["ops"] = True


def _selftest_dconv(conv_mod, orig) -> tuple[bool, str]:
    """Kernel vs production on this module's real base kernels (both sides), T = 8..32 x block sizes seen in decode."""
    import glm53_smallops as SO
    base_all = conv_mod.base_kernel.detach()
    taps, H = base_all.shape[1], base_all.shape[2]
    gs, G, bs = conv_mod.group_size, conv_mod.num_groups, conv_mod.block_size
    gen = torch.Generator(device=base_all.device).manual_seed(11)
    for T in sorted({bs, 2 * bs, 3 * bs, 4 * bs, 8, 16, 24, 32}):
        if T < taps:
            continue
        x = (torch.randn(T, H, generator=gen, device=base_all.device) * 2.0).to(torch.bfloat16)
        coeff = (torch.randn(T, 2, taps, G, generator=gen, device=base_all.device) * 0.5).to(torch.bfloat16)
        for side in (0, 1):
            base = base_all[side]
            ref = orig(x, coeff[:, side], base, bs, G, gs, taps)
            if not _dconv_takes(x, coeff[:, side], base, gs, bs):
                return False, f"kernel does not take the production layout (T={T})"
            got = SO.dconv(x, coeff[:, side], base, gs, bs)
            if not torch.equal(ref.view(torch.int16), got.view(torch.int16)):
                n = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
                return False, f"T={T} side={side}: {n} of {ref.numel()} values differ"
    torch.cuda.synchronize(base_all.device)
    return True, ""


def _wire_dconv(conv_mod, qual: str) -> tuple[bool, str]:
    modname = type(conv_mod).__module__
    mod = sys.modules.get(modname)
    if mod is None:
        return False, f"module {modname} not loaded"
    cur = getattr(mod, "_grouped_conv", None)
    if getattr(cur, "_glm53_so_hook", False):
        return True, "already wired"
    if not callable(cur):
        return False, f"{modname}._grouped_conv missing"
    fp = _fingerprint(cur)
    if fp not in FP_GROUPED_CONV:
        return False, f"{modname}._grouped_conv fingerprint {fp} not in {sorted(FP_GROUPED_CONV)}"
    ok, why = _selftest_dconv(conv_mod, cur)
    if not ok:
        return False, "self-test: " + why
    _ORIG["grouped_conv"] = cur
    mod._grouped_conv = _grouped_conv_so
    _STATE["dconv_mods"].add(modname)
    return True, "wired"


# ---------------------------------------------------------------------------------------------------------------
# mhc: bf16 weight copies + CUDA-kernel override of vllm::mhc_fused_post_pre_tilelang

def _mhc_impl(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
              hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6):
    orig = _ORIG["mhc"]
    args = (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps)
    e = _MHC_W.get(fn.data_ptr()) if norm_weight is not None else None
    if e is None or fn.dtype != e[0].dtype or fn.shape != e[0].shape or fn.stride() != e[0].stride():
        COUNTERS["mhc_prod"] += 1
        return orig(*args)
    if fn._version != e[1]:
        _once(("mhc_stale", e[3]), logging.WARNING, "glm53_dec_smallops: %s was modified after its bf16 copy was made; "
              "production's mHC kernel serves it from now on", e[3])
        COUNTERS["mhc_prod"] += 1
        return orig(*args)
    served = _mhc_small(e[2], *args)
    if served is None:
        COUNTERS["mhc_prod"] += 1
        return orig(*args)
    COUNTERS["mhc_served"] += 1
    return served


def _mhc_small(wb, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
               hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps):
    """Production's small-M branch of mhc_fused_post_pre_tilelang (norm_weight given) with the smallops kernel in
    place of mhc_fused_tilelang; None = not taken (the caller runs production's function)."""
    if residual.dim() < 2 or residual.shape[-2] != MHC_HC or residual.shape[-1] != MHC_HID:
        return None
    if (residual.dtype != torch.bfloat16 or x.dtype != torch.bfloat16 or post_layer_mix.dtype != torch.float32
            or comb_res_mix.dtype != torch.float32 or hc_scale.dtype != torch.float32
            or hc_base.dtype != torch.float32 or fn.dtype != torch.float32):
        return None
    outer_shape = residual.shape[:-2]
    M = residual.numel() // (MHC_HC * MHC_HID)
    if not (1 <= M <= MHC_SERVE_M_MAX):
        return None
    if tuple(x.shape) != (*outer_shape, MHC_HID) or tuple(comb_res_mix.shape) != (*outer_shape, MHC_HC, MHC_HC):
        return None
    if tuple(post_layer_mix.shape) not in ((*outer_shape, MHC_HC, 1), (*outer_shape, MHC_HC)):
        return None
    if tuple(fn.shape) != (MHC_N3, MHC_HC * MHC_HID) or tuple(hc_scale.shape) != (3,) or tuple(hc_base.shape) != (MHC_N3,):
        return None
    if tuple(norm_weight.shape) != (MHC_HID,):
        return None
    if not (x.is_contiguous() and residual.is_contiguous() and post_layer_mix.is_contiguous()
            and comb_res_mix.is_contiguous() and _aligned(x) and _aligned(residual)):
        return None
    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()
    import glm53_smallops as SO
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_pre_big_fuse_with_norm_tilelang
    dev = residual.device
    residual_flat = residual.view(M, MHC_HC, MHC_HID)
    S = SO.mhc_splits(M)                               # production: 8 if M < 8 (hidden <= 4096) else 4
    gemm_out_mul = torch.empty(S, M, MHC_N3, dtype=torch.float32, device=dev)
    gemm_out_sqrsum = torch.empty(S, M, dtype=torch.float32, device=dev)
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(M, MHC_HC, dtype=torch.float32, device=dev)
    comb_mix_cur = torch.empty(M, MHC_HC * MHC_HC, dtype=torch.float32, device=dev)
    layer_input_cur = torch.empty(M, MHC_HID, dtype=torch.bfloat16, device=dev)
    SO.mhc_fused(comb_res_mix.view(M, MHC_HC * MHC_HC), post_layer_mix.view(M, MHC_HC), residual_flat,
                 x.view(M, MHC_HID), wb, S, gemm_out_mul, gemm_out_sqrsum, residual_cur)
    mhc_pre_big_fuse_with_norm_tilelang(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual_cur, post_mix_cur,
                                        comb_mix_cur, layer_input_cur, norm_weight, MHC_HID, rms_eps, hc_pre_eps,
                                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, norm_eps, S, MHC_HC)
    return (residual_cur.view(*outer_shape, MHC_HC, MHC_HID), post_mix_cur.view(*outer_shape, MHC_HC, 1),
            comb_mix_cur.view(*outer_shape, MHC_HC, MHC_HC), layer_input_cur.view(*outer_shape, MHC_HID))


def _selftest_mhc(entries) -> tuple[bool, str]:
    """Override path vs production's function, all four outputs bitwise, M = 1..MHC_SERVE_M_MAX, on up to 3 real
    weights."""
    orig = _ORIG["mhc"]
    gen = None
    for name, fn, wb, scale, base in entries[:3]:
        dev = fn.device
        if gen is None:
            gen = torch.Generator(device=dev).manual_seed(12)
        nw = (torch.rand(MHC_HID, generator=gen, device=dev) + 0.5).to(torch.bfloat16)
        for M in range(1, MHC_SERVE_M_MAX + 1):
            x = (torch.randn(M, MHC_HID, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            res = (torch.randn(M, MHC_HC, MHC_HID, generator=gen, device=dev) * 1.5).to(torch.bfloat16)
            post = torch.sigmoid(torch.randn(M, MHC_HC, 1, generator=gen, device=dev)) * 2.0
            comb = torch.rand(M, MHC_HC, MHC_HC, generator=gen, device=dev)
            comb = comb / comb.sum(-1, keepdim=True)
            args = (x, res, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, nw, 1e-5)
            ref = orig(*args)
            got = _mhc_small(wb, *args)
            if got is None:
                return False, f"{name} M={M}: kernel does not take the production layout"
            for i, (a, b) in enumerate(zip(ref, got)):
                ia = a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.int32)
                ib = b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.int32)
                if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(ia, ib):
                    return False, f"{name} M={M}: output {i} differs"
    torch.cuda.synchronize()
    return True, ""


def _wire_mhc(model) -> dict:
    out = {"weights": 0, "bytes": 0, "rejected": []}
    from vllm.model_executor.kernels.mhc import tilelang as W
    fp = _fingerprint(W.mhc_fused_post_pre_tilelang)
    if fp not in FP_MHC_FUSED_POST_PRE:
        out["rejected"].append(f"mhc_fused_post_pre_tilelang fingerprint {fp} not in {sorted(FP_MHC_FUSED_POST_PRE)}")
        return out
    entries = []
    for qual, m in model.named_modules():
        for side in ("attn", "ffn"):
            fn = getattr(m, f"hc_{side}_fn", None)
            if not isinstance(fn, torch.Tensor) or fn.dtype != torch.float32 or not fn.is_cuda:
                continue
            name = f"{qual}.hc_{side}_fn"
            if qual.endswith("layers.0") and side == "attn":
                continue                      # layer 0's attn fn only feeds the standalone hc_pre (not this op)
            if tuple(fn.shape) != (MHC_N3, MHC_HC * MHC_HID) or not fn.is_contiguous():
                out["rejected"].append(f"{name}: shape {tuple(fn.shape)}")
                continue
            with torch.no_grad():
                wb = fn.detach().to(torch.bfloat16).contiguous()
                if not torch.equal(wb.float(), fn.detach()):
                    out["rejected"].append(f"{name}: not exactly bf16-representable")
                    del wb
                    continue
            entries.append((name, fn, wb, getattr(m, f"hc_{side}_scale"), getattr(m, f"hc_{side}_base")))
    if not entries:
        return out
    _ORIG.setdefault("mhc", W.mhc_fused_post_pre_tilelang)
    ok, why = _selftest_mhc(entries)
    if not ok:
        out["rejected"].append("self-test: " + why)
        return out
    with _LOCK:
        for name, fn, wb, _, _ in entries:
            _MHC_W[fn.data_ptr()] = (fn, fn._version, wb, name)
            out["weights"] += 1
            out["bytes"] += wb.numel() * wb.element_size()
        if not _STATE["mhc_override"]:
            from torch.library import Library
            lib = Library("vllm", "IMPL")
            lib.impl("mhc_fused_post_pre_tilelang", _mhc_impl, "CUDA")
            _LIBS.append(lib)
            _STATE["mhc_override"] = True
    return out


# ---------------------------------------------------------------------------------------------------------------

def _tag_compile_cache(kinds) -> None:
    tag = f"v{VERSION}:{','.join(kinds)}"
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        ac = cfg.additional_config
        if not isinstance(ac, dict):
            raise TypeError(f"additional_config is {type(ac).__name__}, not a dict")
        if ac.get("glm53_dec_smallops") != tag:
            ac["glm53_dec_smallops"] = tag
        _STATE["tag"] = tag
    except Exception as exc:  # noqa: BLE001
        _STATE["tag"] = None
        _log.warning("glm53_dec_smallops: could not tag the compile cache (%r)", exc)


def post_load(model, environ=None) -> dict:
    kinds = env_kinds(environ)
    if _STATE["tag"] is None:
        _tag_compile_cache(kinds)
    out = {"dconv": 0, "mhc": None, "rejected": []}
    if "dconv" in kinds:
        for qual, m in model.named_modules():
            if type(m).__name__ == "DFlashGroupedConv" and hasattr(m, "base_kernel"):
                ok, why = _wire_dconv(m, qual)
                if ok:
                    out["dconv"] += 1
                else:
                    out["rejected"].append(f"{qual}: {why}")
                    break                          # one module-global: the first verdict holds for all
    if "mhc" in kinds and any(isinstance(getattr(m, "hc_ffn_fn", None), torch.Tensor) for m in model.modules()):
        out["mhc"] = _wire_mhc(model)
        out["rejected"] += out["mhc"]["rejected"]
    served = []
    if out["dconv"]:
        served.append(f"dconv ({out['dconv']} conv modules, {sorted(_STATE['dconv_mods'])})")
    if out["mhc"] and out["mhc"]["weights"]:
        served.append(f"mhc ({out['mhc']['weights']} weights, bf16 copies {out['mhc']['bytes'] / 2**20:.2f} MiB)")
    if served or out["rejected"]:
        _log.info("glm53_dec_smallops on %s: %s; %d rejected%s; compile-cache tag %s", type(model).__name__,
                  "; ".join(served) or "nothing", len(out["rejected"]),
                  (": " + " | ".join(out["rejected"][:6])) if out["rejected"] else "", _STATE["tag"])
    for r in out["rejected"]:
        if "self-test" in r or "fingerprint" in r:
            _log.warning("glm53_dec_smallops: NOT serving: %s (production's op stays)", r)
    return out


def _hook_loader() -> bool:
    if _STATE["loader"]:
        return True
    import vllm.model_executor.model_loader.base_loader as BL
    orig = BL.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        try:
            post_load(model)
        except Exception as exc:  # noqa: BLE001 - never break model loading; the model keeps production's path
            _log.warning("glm53_dec_smallops: post-load wiring failed (%r); anything wired before the failure keeps "
                         "its per-call fallback to production", exc)

    process_weights_after_loading._glm53_so_orig = orig
    BL.process_weights_after_loading = process_weights_after_loading
    _STATE["loader"] = True
    return True


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_DEC_SMALLOPS is on."""
    on = env_enabled()
    _log.info("glm53_dec_smallops plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              "installing" if on else "off, production ops unchanged")
    if not on:
        return
    try:
        kinds = env_kinds()
        _STATE["kinds"] = kinds
        import glm53_smallops as SO
        SO.load_ext()
        register_ops()
        _hook_loader()
        _log.info("glm53_dec_smallops: ext %s, kinds %s; wired after weight loading", SO.EXT_SOURCE, ",".join(kinds))
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_dec_smallops not installed (production ops unchanged): %r", exc)


def uninstall() -> None:
    """Tests: restore the module globals, drop the op override (Library destruction unregisters it)."""
    for modname in list(_STATE["dconv_mods"]):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(getattr(mod, "_grouped_conv", None), "_glm53_so_hook", False):
            mod._grouped_conv = _ORIG["grouped_conv"]
    _STATE["dconv_mods"].clear()
    _MHC_W.clear()
    keep = [lib for lib in _LIBS if getattr(lib, "ns", None) == "glm53_so"]
    for lib in _LIBS:
        if lib not in keep:
            lib._destroy()
    _LIBS[:] = keep
    _STATE["mhc_override"] = False


def summary() -> dict:
    return {"counters": dict(COUNTERS), "mhc_weights": len(_MHC_W), "dconv_modules": sorted(_STATE["dconv_mods"]),
            "tag": _STATE["tag"]}
