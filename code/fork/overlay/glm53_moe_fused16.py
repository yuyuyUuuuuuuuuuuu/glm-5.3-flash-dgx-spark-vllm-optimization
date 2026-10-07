"""glm53_moe_fused16 - production's routed-MoE PREFILL arithmetic, faster (GLM53_MOE_FUSED16, docs/MOE3.md).

What it replaces: production's E3 grouped fat tier, ``apply_exl3_grouped_fat`` (exllamav3 ``fm_gather_kernel`` +
``fm_gateup_kernel`` + ``fm_down_kernel``), called by name from ``apply_exl3_fused_moe`` for every prefill MoE call.
Everything else of production's apply (routing, the thin kernel for experts at or below the cap, the zero-fill, casts,
tables, scratch buffers) is unchanged. What runs instead (kernels/moe_e4m3.cu, the glm53_moe_e4m3_ext .so):
  - the SAME per-element arithmetic: exllamav3's three kernels ported statement for statement (fp16 rows x fp16 trellis
    decode on mma.sync m16n8k16, fp32 accumulate, k in the same order, the same epilogues incl. the fp16-rounded
    suh products and the fp16 route weight) - except the SiLU's ``(float) exp((double) g)``, which is replaced by
    ``exp_prod``: the same function bit for bit (exhaustively checked over every float in [-87, 88]; ambiguous
    roundings and the range edges take the double statement), mostly in fp32 - the FP64 exp costs ~3.6 ms of the
    ~31 ms gate/up kernel per 13,824-token layer call on GB10;
  - a better schedule: tokens >= 11264 (production's 13,824-token chunks): ``p16b``, ONE persistent launch (2 CTAs x
    8 warps per SM, exllamav3's CTA shape) that interleaves gather, gate/up and down items with per-segment dataflow
    flags, so the DRAM-bound gather / fp32 scatter overlap the compute-bound gate/up; 2048 <= tokens < 11264:
    ``sep``, the token-ordered gather + the ported gate/up + down, with the down of segment-chunk c on a side stream
    concurrently with the gate/up of chunk c+1; below 2048 tokens: production's kernels (no measured gain).
Numerics: h13 and h2 BIT-IDENTICAL to production's; ``out`` = production's up to the order of the fp32 atomic adds
(production's own run-to-run nondeterminism; rel-L2 ~7e-8 vs production's own 4e-9..8e-9 run to run).

Knob: GLM53_MOE_FUSED16 unset / "" / "0" = install nothing (production byte-identical); "1" = on; anything else =
WARNING, nothing installed. Installs only if production's apply_exl3_grouped_fat / build_grouped_fat_tables /
_grouped_scratch carry the verified source fingerprints. A layer is served when it has production's pointer tables,
K4, hidden 4096, local intermediate 1024, a shared gate/up input rotation, and passed its one-time self-test at model
load (Exl3MoEMethod.process_weights_after_loading wrapped): a synthetic 2304-token grouped call on the layer's real
experts through production's own kernels and through both schedules - served only if h2 is bit-identical and out
agrees within 1e-6 rel-L2. Otherwise (and during CUDA-graph capture) the wrapped production function runs.
"""
from __future__ import annotations

import logging
import os

import torch

_log = logging.getLogger("vllm.glm53_moe_fused16")

ENV = "GLM53_MOE_FUSED16"
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
HIDDEN, INTER = 4096, 1024
TILE = 64                      # production's fat-row tile (exllamav3_fat_moe_tile_rows_*)
# Schedules (tests/moe3/bench_p16.py, real layer-10 experts, ms per grouped call vs production's):
#   p16b (one persistent launch, 2 CTAs/SM)      13824: 54.2 vs 61.9   12288: 49.2 vs 54.2   10240: 42.5 vs 45.3
#   sep  (gather16 + pgu + pdn, nchunks streams)  8192: 35.2 vs 37.4    6144: 27.6 vs 28.9    4289: 20.5 vs 21.0
#   below sep_min_t (4096): no gain measured (2048: 13.02 vs 13.26, 1791: 12.39 vs 12.25) -> production's kernels;
#   sep uses 2 stream chunks from 6144 tokens (8192: 35.40 vs 35.68), 1 below (4289: 20.64 vs 20.85)
SCHED = {"mode": "auto", "p16b_min_t": 11264, "sep_min_t": 4096, "nchunks": 2, "nchunks_min_t": 6144, "lg": 8, "ld": 24, "grid": 0,
         "variant": 0}
SELFTEST_TOKENS = 2304
SELFTEST_TOL = 1e-6            # out rel-L2 vs production's (fp32 atomic order only: measured 3e-9 .. 7e-8)
# production functions whose contract this relies on (sha256 of ast.dump, first 16 hex = glm53_prefill_cap's)
VERIFIED_FINGERPRINTS = {
    "apply_exl3_grouped_fat": frozenset({"b84e24e452c4a9eb"}),
    "build_grouped_fat_tables": frozenset({"5ebe863a15223fc0"}),
    "_grouped_scratch": frozenset({"a9c935fdbbc67d11"}),
}
STATS = {"installed": False, "calls": 0, "served": 0, "served_p16b": 0, "served_sep": 0, "passed": 0, "layers": 0,
         "layers_ok": 0, "layers_fallback": 0, "selftest_failed": 0, "selftest_raised": 0}
_FALLBACK_WHY: dict = {}
_SUMMARY = {"logged": None}
_SYNC: dict = {}
_SIDE: dict = {}
_LOGGED: set = set()
_EXT = None
EXT_SYMBOLS = ("p16b", "pgu", "pdn", "gather16", "exp_check")


def _ext():
    global _EXT
    if _EXT is None:
        import glm53_moe_e4m3_ext as e

        _EXT = e
    return _EXT


def enabled(environ=None) -> bool:
    """unset / empty / 0 = off, 1 = on, anything else = ValueError (install() then refuses with a WARNING)."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV) or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV} must be empty, 0 or 1 (got {raw!r})")


def layer_reason(layer) -> str | None:
    if not getattr(layer, "_exl3_ptrs", None) or not getattr(layer, "_exl3_inners", None):
        return "no pointer tables / inners"
    if int(getattr(layer, "_exl3_bits", 0)) != 4:
        return f"bits {getattr(layer, '_exl3_bits', None)} != 4"
    if int(getattr(layer, "_exl3_hidden_size", 0)) != HIDDEN or int(getattr(layer, "_exl3_intermediate_local", 0)) != INTER:
        return "shape (hidden, intermediate_local) != (4096, 1024)"
    if not bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        return "gate/up input rotations differ"
    if len(layer._exl3_inners) > 1024:
        return "more than 1024 local experts"
    return None


def source_fingerprint(fn) -> str | None:
    import ast
    import hashlib
    import inspect
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _unwrap(fn):
    seen = 0
    while getattr(fn, "_tf_exl3_orig", None) is not None and seen < 8:
        fn = fn._tf_exl3_orig
        seen += 1
    return fn


def compatibility(prodmod) -> tuple[bool, str]:
    bad = []
    for name, ok in VERIFIED_FINGERPRINTS.items():
        fn = getattr(prodmod, name, None)
        if not callable(fn):
            bad.append(f"{name} missing")
            continue
        fp = source_fingerprint(_unwrap(fn))
        if fp not in ok:
            bad.append(f"{name} fingerprint {fp} not in {sorted(ok)}")
    if not callable(getattr(prodmod, "map_topk_to_local", None)):
        bad.append("map_topk_to_local missing")
    return (not bad), "; ".join(bad)


def mode_for(tokens: int, sched: dict | None = None) -> str | None:
    sc = dict(SCHED)
    if sched:
        sc.update(sched)
    if sc["mode"] != "auto":
        return sc["mode"]
    if tokens >= int(sc["p16b_min_t"]):
        return "p16b"
    if tokens >= int(sc["sep_min_t"]):
        return "sep"
    return None


def run(prodmod, xh, out, counts, token_sorted, weight_sorted, layer, cap, limit, keep=None, sched=None):
    """Same contract as production's apply_exl3_grouped_fat: adds every fat expert's (count > cap) weighted FFN output
    into out (fp32 [T, 4096], zeroed and holding the thin kernel's experts) in place. Same tables (production's
    build_grouped_fat_tables, tile 64) and scratch (production's _grouped_scratch h13 / h2) as production."""
    ext = _ext()
    sc_ = dict(SCHED)
    if sched:
        sc_.update(sched)
    mode = sc_["mode"]
    if mode == "auto":
        mode = mode_for(int(xh.shape[0])) or "sep"
    if mode not in ("p16b", "sep"):
        raise ValueError(f"glm53_moe_fused16: unknown schedule {mode!r}")
    ptrs = layer._exl3_ptrs
    dev = ptrs["gate_trellis"].device
    rows_cap = int(token_sorted.numel())
    scratch = prodmod._grouped_scratch(dev, rows_cap, HIDDEN, INTER)
    h13 = scratch["h13"][:rows_cap]
    h2 = scratch["h2"][:rows_cap]
    t = prodmod.build_grouped_fat_tables(counts, cap, token_sorted.contiguous(), weight_sorted.contiguous(), rows_cap,
                                         TILE)
    rw = t["row_weight"].to(torch.float32)            # production's fp16 route weights, exactly
    variant = int(sc_["variant"])
    if mode == "sep" or variant & 1024:
        # rows visited token by token: x read once from DRAM (production's fm_gather re-reads it per routed expert)
        perm = torch.argsort(t["row_token"]).to(torch.int32)
        ext.gather16(xh, perm, t["row_token"], t["row_expert"], ptrs["gate_suh"], h13, t["num_rows"])
    if mode == "p16b":
        need = 1 + 3 * int(t["seg_expert"].numel())
        sync = _SYNC.get(dev)
        if sync is None or sync.numel() < need:
            sync = _SYNC[dev] = torch.zeros(max(need, 8192), dtype=torch.int32, device=dev)
        sync[:need].zero_()
        ext.p16b(xh, h13, h2, out, ptrs["gate_trellis"], ptrs["up_trellis"], ptrs["gate_svh"], ptrs["up_svh"],
                 ptrs["gate_suh"], ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"], t["row_token"],
                 t["row_expert"], rw, t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], sync,
                 float(limit), int(sc_["lg"]), int(sc_["ld"]), int(sc_["grid"]), variant)
    else:
        nc = int(sc_["nchunks"]) if int(xh.shape[0]) >= int(sc_.get("nchunks_min_t", 0)) else 1

        def gu(c):
            ext.pgu(h13, ptrs["gate_trellis"], ptrs["up_trellis"], ptrs["gate_svh"], ptrs["up_svh"], ptrs["down_suh"],
                    h2, t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], float(limit), 0, c, nc)

        def dn(c):
            ext.pdn(h2, ptrs["down_trellis"], ptrs["down_svh"], out, t["row_token"], rw, t["seg_expert"],
                    t["seg_row0"], t["seg_rows"], t["num_segs"], 0, c, nc)
        if nc <= 1:
            gu(0)
            dn(0)
        else:
            # gate/up of chunk c+1 (main stream) runs concurrently with the down of chunk c (side stream): the down's
            # DRAM-bound fp32 scatter overlaps the compute-bound gate/up (2 CTAs/SM each: they share the SMs)
            main = torch.cuda.current_stream(dev)
            side = _SIDE.get(dev)
            if side is None:
                side = _SIDE[dev] = torch.cuda.Stream(device=dev)
            side.wait_stream(main)
            evs = []
            for c in range(nc):
                gu(c)
                ev = torch.cuda.Event()
                ev.record(main)
                evs.append(ev)
            with torch.cuda.stream(side):
                for c in range(nc):
                    side.wait_event(evs[c])
                    dn(c)
            main.wait_stream(side)
    if keep is not None:
        keep.update(t)
        keep.update(h13=h13, h2=h2)


# ---------------------------------------------------------------------------------------------------------
# the production hook (apply_exl3_grouped_fat) and its per-layer gate

def _prod_args(prodmod, layer, x, ids, w, cap):
    """Production's own preparation of the grouped call (apply_exl3_fused_moe, overlay :1469-1482)."""
    n_exp = len(layer._exl3_inners)
    tokens = int(x.shape[0])
    expert_map = prodmod.pin_exl3_expert_map(layer, x.device) if hasattr(prodmod, "pin_exl3_expert_map") else None
    local = prodmod.map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = w.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, local.long(), torch.ones(local.shape, dtype=torch.long, device=x.device))
    return x.contiguous().half(), counts[:n_exp], flat_token[order], flat_weight[order]


def selftest(prodmod, orig, layer, limit: float, tokens: int = SELFTEST_TOKENS) -> dict:
    """A synthetic grouped call on the layer's real experts (every local expert routed, cap 1) through production's
    own grouped kernels (orig) and through each P16 schedule: h2 must be bit-identical, out within SELFTEST_TOL."""
    dev = layer._exl3_ptrs["gate_trellis"].device
    n_exp = len(layer._exl3_inners)
    g = torch.Generator(device="cpu").manual_seed(4321)
    topk = min(8, n_exp)
    ids = torch.argsort(torch.rand(tokens, n_exp, generator=g), dim=1)[:, :topk].to(dev)
    w = torch.rand(tokens, topk, generator=g).to(dev)
    w = w / w.sum(-1, keepdim=True)
    x = (torch.randn(tokens, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    xh, counts, ts, ws = _prod_args(prodmod, layer, x, ids, w, 1)
    rows_cap = int(ts.numel())
    out_ref = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=dev)
    orig(xh, out_ref, counts, ts, ws, layer, 1, limit)
    scratch = prodmod._grouped_scratch(dev, rows_cap, HIDDEN, INTER)
    nfat = int(counts[counts > 1].sum())
    h2_ref = scratch["h2"][:nfat].clone()
    res = {"ok": True}
    for mode in ("p16b", "sep"):
        o = torch.zeros_like(out_ref)
        run(prodmod, xh, o, counts, ts, ws, layer, 1, limit, sched={"mode": mode})
        same = bool(torch.equal(scratch["h2"][:nfat], h2_ref))
        err = float((o.double() - out_ref.double()).norm() / out_ref.double().norm().clamp_min(1e-300))
        res[mode] = {"h2_bitwise": same, "out_rel": err}
        res["ok"] = res["ok"] and same and err <= SELFTEST_TOL and bool(torch.isfinite(o).all())
    return res


def check_layer(prodmod, orig, layer, limit: float) -> bool:
    ok = getattr(layer, "_glm53_moe_fused16_ok", None)
    if ok is not None:
        return ok
    STATS["layers"] += 1
    why = layer_reason(layer)
    if why is not None:
        ok = False
        _log.warning("glm53_moe_fused16: layer not served (%s); production kernels kept", why)
    else:
        try:
            r = selftest(prodmod, orig, layer, float(limit))
            ok = r["ok"]
            if ok and "selftest" not in _LOGGED:
                _LOGGED.add("selftest")
                _log.info("glm53_moe_fused16 layer self-test passed (h2 bit-identical to production's; out rel-L2 "
                          "p16b %.1e, sep %.1e)", r["p16b"]["out_rel"], r["sep"]["out_rel"])
            if not ok:
                why = "self-test FAILED"
                STATS["selftest_failed"] += 1
                _log.warning("glm53_moe_fused16 layer self-test FAILED (%s): production kernels kept", r)
        except Exception as exc:  # noqa: BLE001
            ok = False
            why = "self-test raised"
            STATS["selftest_raised"] += 1
            _log.warning("glm53_moe_fused16 layer self-test raised %r: production kernels kept", exc)
    if ok:
        STATS["layers_ok"] += 1
    else:
        STATS["layers_fallback"] += 1
        _FALLBACK_WHY[why] = _FALLBACK_WHY.get(why, 0) + 1
    layer._glm53_moe_fused16_ok = ok
    return ok


def log_summary() -> str:
    why = ", ".join(f"{k} x{v}" for k, v in sorted(_FALLBACK_WHY.items())) or "none"
    line = (f"glm53_moe_fused16 summary: {STATS['layers_ok']}/{STATS['layers']} layers served, "
            f"{STATS['layers_fallback']} fell back ({why}); self-test FAILED {STATS['selftest_failed']}, raised "
            f"{STATS['selftest_raised']}")
    key = (STATS["layers"], STATS["layers_ok"])
    if _SUMMARY["logged"] != key:
        _SUMMARY["logged"] = key
        (_log.warning if STATS["layers_fallback"] else _log.info)("%s", line)
    return line


def _serve(prodmod, orig, xh, out, counts, token_sorted, weight_sorted, layer, cap, limit):
    STATS["calls"] += 1
    if _SUMMARY["logged"] is None or _SUMMARY["logged"][0] != STATS["layers"]:
        log_summary()
    mode = mode_for(int(xh.shape[0]))
    if (mode is None or torch.cuda.is_current_stream_capturing() or int(xh.shape[-1]) != HIDDEN
            or xh.dtype != torch.float16 or out.dtype != torch.float32
            or not check_layer(prodmod, orig, layer, limit)):
        STATS["passed"] += 1
        return orig(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit)
    run(prodmod, xh, out, counts, token_sorted, weight_sorted, layer, cap, limit, sched={"mode": mode})
    STATS["served"] += 1
    STATS["served_" + mode] += 1
    if "active" not in _LOGGED:
        _LOGGED.add("active")
        _log.info("glm53_moe_fused16 active: first grouped prefill call with %d tokens ran on the %s schedule "
                  "(production arithmetic)", int(xh.shape[0]), mode)
    return None


def _hook_load(prodmod, orig_ref) -> bool:
    cls = getattr(prodmod, "Exl3MoEMethod", None)
    fn = getattr(cls, "process_weights_after_loading", None) if cls is not None else None
    if fn is None:
        return False
    if getattr(fn, "_glm53_moe_fused16", False):
        return True
    limit = float(getattr(prodmod, "SWIGLU_LIMIT_DEFAULT", 10.0))

    def process_weights_after_loading(self, layer, *a, **k):
        r = fn(self, layer, *a, **k)
        if getattr(layer, "_exl3_inners", None):
            check_layer(prodmod, orig_ref[0], layer, limit)
        return r

    process_weights_after_loading.__doc__ = fn.__doc__
    process_weights_after_loading._glm53_moe_fused16 = True
    process_weights_after_loading._glm53_moe_fused16_orig = fn
    cls.process_weights_after_loading = process_weights_after_loading
    return True


def install(prodmod=None, environ=None, load_selftest: bool = True) -> dict:
    """Wrap prodmod.apply_exl3_grouped_fat. Installs nothing when the knob is off."""
    try:
        on = enabled(environ)
    except ValueError as exc:
        _log.warning("glm53_moe_fused16 not installed: %s (production kernels unchanged)", exc)
        return {"installed": False, "reason": str(exc)}
    if not on:
        return {"installed": False, "reason": "off"}
    if prodmod is None:
        import importlib

        prodmod = importlib.import_module(PROD_MODULE)
    orig = getattr(prodmod, "apply_exl3_grouped_fat", None)
    if orig is None:
        _log.warning("glm53_moe_fused16 not installed: production module has no apply_exl3_grouped_fat")
        return {"installed": False, "reason": "no apply_exl3_grouped_fat"}
    if getattr(orig, "_glm53_moe_fused16", False):
        return {"installed": True, "reason": "already installed"}
    ok, why = compatibility(prodmod)
    if not ok:
        _log.warning("glm53_moe_fused16 not installed: %s (production kernels unchanged)", why)
        return {"installed": False, "reason": why}
    try:
        e = _ext()
        missing = [n for n in EXT_SYMBOLS if not hasattr(e, n)]
        if missing:
            raise RuntimeError(f"extension lacks {missing}")
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moe_fused16 not installed: extension glm53_moe_e4m3_ext unavailable (%r)", exc)
        return {"installed": False, "reason": f"extension: {exc!r}"}

    def apply_exl3_grouped_fat(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit):
        return _serve(prodmod, orig, xh, out, counts, token_sorted, weight_sorted, layer, cap, limit)

    apply_exl3_grouped_fat.__doc__ = getattr(orig, "__doc__", None)
    apply_exl3_grouped_fat._glm53_moe_fused16 = True
    apply_exl3_grouped_fat._glm53_moe_fused16_orig = orig
    apply_exl3_grouped_fat._tf_exl3_orig = orig          # fingerprint checks of other modules see production's
    prodmod.apply_exl3_grouped_fat = apply_exl3_grouped_fat
    load_hook = _hook_load(prodmod, [orig]) if load_selftest else False
    STATS["installed"] = True
    _log.info("glm53_moe_fused16 installed: GLM53_MOE_FUSED16=1, E3 grouped prefill (tokens >= %d) on production's "
              "arithmetic in the P16 schedules (h2 bit-identical, out = production's up to fp32 atomic order); "
              "per-layer self-test %s (pid %d)", SCHED["sep_min_t"],
              "at load time" if load_hook else "at the first eligible call", os.getpid())
    return {"installed": True, "reason": "ok", "load_hook": load_hook}


def uninstall(prodmod=None) -> None:
    import importlib

    prodmod = prodmod or importlib.import_module(PROD_MODULE)
    fn = getattr(prodmod, "apply_exl3_grouped_fat", None)
    while fn is not None and getattr(fn, "_glm53_moe_fused16", False):
        prodmod.apply_exl3_grouped_fat = fn._glm53_moe_fused16_orig
        fn = prodmod.apply_exl3_grouped_fat
    cls = getattr(prodmod, "Exl3MoEMethod", None)
    pw = getattr(cls, "process_weights_after_loading", None) if cls is not None else None
    while pw is not None and getattr(pw, "_glm53_moe_fused16", False):
        cls.process_weights_after_loading = pw._glm53_moe_fused16_orig
        pw = cls.process_weights_after_loading
    STATS["installed"] = False


def plugin_install() -> None:
    """Called by integrate.plugin_register (armed by overlay/patch_moe_fused16.py). Never raises into vLLM."""
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moe_fused16 not installed (production kernels unchanged): %r", exc)
