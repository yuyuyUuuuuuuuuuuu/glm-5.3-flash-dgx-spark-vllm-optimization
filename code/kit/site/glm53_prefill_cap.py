"""GLM53_PREFILL_FUSED_CAP: a smaller thin/fat split for production's E3 prefill branch (docs/PREFILL_CAP.md).

Production's apply_exl3_fused_moe (docs/ref/prod_live/overlay_exl3.py:1444) serves a MoE layer call with
tokens > cap (cap = layer._exl3_fused_temps[0].shape[1] = EXL3_TEMP_ROWS_FUSED, 256 in production) on the E3 branch
when EXL3_FAT_GROUPED=1, EXL3_MOE_ROW_TILE is off and the layer's tier resolved to "grouped": one fused exl3_moe launch
("thin": it skips every expert whose row count exceeds the temps' dim 1) plus apply_exl3_grouped_fat for the experts
with count > cap. The thin kernel re-decodes the whole expert for every 16-row block and runs only
exl3_moe_max_concurrency() experts at a time (6 on GB10), so experts with 17..256 rows are far cheaper on the grouped
kernels (64-row tiles, every fat expert of the layer in one launch).

With GLM53_PREFILL_FUSED_CAP=n (1 <= n < cap) a wrapper around apply_exl3_fused_moe runs exactly those E3 calls with
layer._exl3_fused_temps swapped for a set of temps with n rows (same dtypes, same concurrency; allocated once per
(device, hidden, intermediate, concurrency, n) and shared by all layers, like production's own), restored in
`finally`. Production's own code then sends experts with <= n rows to the thin kernel and experts with > n rows to
E3 (its cap is read from the temps). Nothing else changes:
  - decode / any call with tokens <= cap (every CUDA-graph capture size) is passed through untouched, so the fused
    single-launch decode path and the TF fork's K2 / exl3_moe paths see exactly the temps they see without this;
  - a call that would not take the E3 branch (tier not "grouped", EXL3_FAT_GROUPED=0, EXL3_MOE_ROW_TILE=1) is passed
    through (the other fat tiers would get slower with a small cap);
  - a call made while a CUDA graph is being captured is passed through (the temps allocation must not happen there);
  - E3's scratch is sized from token_sorted.numel() = tokens * top-k (independent of the cap), and its segment
    table bound (rows_cap / tile + n_experts) holds for any cap, so no capacity changes.
Numerics: experts with n < rows <= cap move from the thin kernel (fp16 gate/up temps and fp16 activation
min(silu(g), L)) to the E3 kernels (fp32 activation silu(min(g, L))): the class production already uses for experts with
> cap rows; fp32 output accumulation order changes. Measured closer to the float64 reference than the thin kernel.

Enable: GLM53_PREFILL_FUSED_CAP=<n> (read once, when the vLLM plugin loads). Unset, empty, "0" or "off": nothing is
installed. An invalid value or a production module whose E3 code is not a fingerprinted version: WARNING, nothing is
installed. Revert: unset and restart vLLM.
Composes with the TF fork's K2 apply hook in either order (both keep the next function in `_tf_exl3_orig`; the K2
fingerprint check follows it). Production calls apply_exl3_fused_moe by module-global name from apply_exl3_experts.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import logging
import os
import textwrap

import torch

_log = logging.getLogger("vllm.glm53_prefill_cap")
ENV = "GLM53_PREFILL_FUSED_CAP"
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
_OFF = frozenset({"", "0", "off", "false", "no"})

# Production functions whose behaviour this relies on (sha256 of ast.dump of the source, first 16 hex; the same
# fingerprint as integrate.source_fingerprint). Identical in the image's quantization/exl3.py and the launcher
# overlay df864b5 (tests/test_prefill_cap.py prints and checks them against both).
#   apply_exl3_fused_moe: the E3 branch condition and cap = int(temps[0].shape[1]) read per call
#   apply_exl3_grouped_fat / build_grouped_fat_tables: `cap` is used only as the "count > cap" fat threshold
#   _grouped_scratch: capacity from rows = token_sorted.numel(), independent of cap
VERIFIED_FINGERPRINTS = {
    "apply_exl3_fused_moe": frozenset({"fa19d59307dd88bf"}),
    "apply_exl3_grouped_fat": frozenset({"b84e24e452c4a9eb"}),
    "build_grouped_fat_tables": frozenset({"5ebe863a15223fc0"}),
    "_grouped_scratch": frozenset({"a9c935fdbbc67d11"}),
}
REQUIRED = ("apply_exl3_fused_moe", "grouped_fat_enabled", "fused_moe_row_tile_enabled")

STATS = {"swapped": 0, "passthrough_tokens": 0, "passthrough_tier": 0, "passthrough_capture": 0,
         "passthrough_cap": 0, "temps_allocs": 0, "temps_bytes": 0}
_TEMPS: dict[tuple, tuple] = {}
_LOGGED: set = set()
_STATE = {"n": 0}


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def parse_env(environ=None) -> int:
    """0 = off. Raises ValueError for a value that is neither off nor an integer >= 1."""
    raw = (os.environ if environ is None else environ).get(ENV)
    if raw is None or raw.strip().lower() in _OFF:
        return 0
    try:
        n = int(raw.strip())
    except ValueError:
        raise ValueError(f"{ENV}={raw!r} is not an integer") from None
    if n < 1:
        raise ValueError(f"{ENV}={raw!r} must be >= 1 (or unset / 0 to disable)")
    return n


def source_fingerprint(fn) -> str | None:
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
    bad = [f"{n} missing" for n in REQUIRED if not callable(getattr(prodmod, n, None))]
    for name, ok in VERIFIED_FINGERPRINTS.items():
        fn = getattr(prodmod, name, None)
        if not callable(fn):
            bad.append(f"{name} missing")
            continue
        fp = source_fingerprint(_unwrap(fn))
        if fp not in ok:
            bad.append(f"{name} fingerprint {fp} not in {sorted(ok)}")
    return (not bad), "; ".join(bad)


def small_temps(temps, n: int) -> tuple:
    """Production's temps layout (concurrency, rows, hidden) x 2 + (concurrency, rows, intermediate) x 2, fp16, with
    rows = n. One set per (device, hidden, intermediate, concurrency, n), shared by every layer (as production's)."""
    t0, t2 = temps[0], temps[2]
    conc, _, hidden = (int(v) for v in t0.shape)
    inter = int(t2.shape[2])
    key = (str(t0.device), hidden, inter, conc, n, t0.dtype, t2.dtype)
    s = _TEMPS.get(key)
    if s is None:
        s = (torch.empty((conc, n, hidden), dtype=t0.dtype, device=t0.device),
             torch.empty((conc, n, hidden), dtype=temps[1].dtype, device=t0.device),
             torch.empty((conc, n, inter), dtype=t2.dtype, device=t0.device),
             torch.empty((conc, n, inter), dtype=temps[3].dtype, device=t0.device))
        _TEMPS[key] = s
        STATS["temps_allocs"] += 1
        STATS["temps_bytes"] = sum(t.numel() * t.element_size() for v in _TEMPS.values() for t in v)
    return s


def set_cap(n: int) -> None:
    """Tests: change n of an installed wrapper (0 = pass everything through)."""
    _STATE["n"] = int(n)


def _make_hook(prodmod, orig):
    grouped_on = prodmod.grouped_fat_enabled
    row_tiles = prodmod.fused_moe_row_tile_enabled
    capturing = torch.cuda.is_current_stream_capturing
    state = _STATE

    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit):
        n = state["n"]
        temps = getattr(layer, "_exl3_fused_temps", None)
        try:
            tokens = int(x2d.shape[0])
            cap = int(temps[0].shape[1])
        except Exception:  # noqa: BLE001 - production raises its own error for a layer without temps
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        if n <= 0 or tokens <= cap:                        # decode and every capture size: untouched
            STATS["passthrough_tokens"] += 1
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        if n >= cap:
            STATS["passthrough_cap"] += 1
            _log_once("cap", logging.WARNING, "glm53 prefill fused cap: %s=%d >= the layer's fused cap %d; not applied",
                      ENV, n, cap)
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        if (getattr(layer, "_exl3_fat_effective_tier", None) != "grouped" or not grouped_on() or row_tiles()):
            STATS["passthrough_tier"] += 1
            _log_once("tier", logging.WARNING, "glm53 prefill fused cap: a prefill call on a layer that does not take "
                      "the E3 grouped branch (tier %r, EXL3_FAT_GROUPED %s, EXL3_MOE_ROW_TILE %s); passed through",
                      getattr(layer, "_exl3_fat_effective_tier", None), grouped_on(), row_tiles())
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        if capturing():
            STATS["passthrough_capture"] += 1
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        try:
            small = small_temps(temps, n)
        except Exception as exc:  # noqa: BLE001 - review: never fail a prefill over the 120 KiB helper allocation
            STATS["passthrough_error"] = STATS.get("passthrough_error", 0) + 1
            _log_once("small_temps", logging.WARNING, "glm53 prefill fused cap: small temps unavailable (%r); passed through", exc)
            return orig(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_fused_temps = small
        try:
            out = orig(x2d, ids, weights, layer, inners, expert_map, limit)
        finally:
            layer._exl3_fused_temps = temps
        STATS["swapped"] += 1
        if STATS["swapped"] == 1:
            _log.info("glm53 prefill fused cap active: first prefill MoE call with %d tokens ran with thin cap %d "
                      "(layer cap %d; E3 serves experts with > %d rows); small temps %d B", tokens, n, cap, n,
                      STATS["temps_bytes"])
        return out

    apply_exl3_fused_moe.__doc__ = getattr(orig, "__doc__", None)
    apply_exl3_fused_moe._glm53_prefill_cap_hook = True
    apply_exl3_fused_moe._tf_exl3_orig = orig
    return apply_exl3_fused_moe


def install(prodmod=None, *, n: int | None = None) -> dict:
    """Wrap prodmod.apply_exl3_fused_moe. n None = from the env (0/unset -> nothing is installed). Idempotent."""
    report = {"installed": False, "reason": None, "n": 0}
    try:
        cap_n = parse_env() if n is None else int(n)
    except ValueError as exc:
        report["reason"] = str(exc)
        _log.warning("glm53 prefill fused cap NOT installed: %s", exc)
        return report
    if cap_n <= 0:
        report["reason"] = f"{ENV} not set"
        return report
    if prodmod is None:
        prodmod = importlib.import_module(PROD_MODULE)
    cur = getattr(prodmod, "apply_exl3_fused_moe", None)
    if getattr(cur, "_glm53_prefill_cap_hook", False):
        set_cap(cap_n)
        report.update(installed=True, reason="already installed", n=cap_n)
        return report
    ok, why = compatibility(prodmod)
    if not ok:
        report["reason"] = f"production module not verified: {why}"
        _log.warning("glm53 prefill fused cap NOT installed (%s=%d): %s; production's E3 branch unchanged",
                     ENV, cap_n, why)
        return report
    set_cap(cap_n)
    prodmod.apply_exl3_fused_moe = _make_hook(prodmod, cur)
    report.update(installed=True, reason="ok", n=cap_n)
    _log.info("glm53 prefill fused cap installed: %s=%d (MoE calls with tokens > the fused cap on E3-grouped layers: "
              "thin kernel for experts <= %d rows, E3 for the rest; decode unchanged)", ENV, cap_n, cap_n)
    return report


def uninstall(prodmod=None) -> dict:
    if prodmod is None:
        prodmod = importlib.import_module(PROD_MODULE)
    cur = getattr(prodmod, "apply_exl3_fused_moe", None)
    if getattr(cur, "_glm53_prefill_cap_hook", False):
        prodmod.apply_exl3_fused_moe = cur._tf_exl3_orig
        set_cap(0)
        return {"restored": True}
    set_cap(0)
    return {"restored": False}


def plugin_install() -> None:
    """Called from integrate.plugin_register (every vLLM process); never raises."""
    try:
        n = parse_env()
    except ValueError as exc:
        _log.warning("glm53 prefill fused cap NOT installed: %s", exc)
        return
    if n <= 0:
        return
    try:
        install(n=n)
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53 prefill fused cap install failed (production unchanged): %r", exc)
