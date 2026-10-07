# VENDORED COPY (reference only, never installed): branch moefq working tree overlay/glm53_moe_e4m3_emu.py
# (HEAD 86b69f2 + its uncommitted per-token-row fix), sha256 18990ef25b781456..., copied 2026-10-01 by branch moee4m3 as the
# numerics SPEC the e4m3 kernel (kernels/moe_e4m3.cu) is checked against. Do not edit; re-copy on a spec change.
"""glm53_moe_e4m3_emu — SPEED-IGNORING quality emulation of an e4m3 routed-MoE prefill (GLM53_MOE_E4M3_EMU).

What it is for
--------------
Before building an e4m3 routed-MoE kernel (weeks of work), measure what such a kernel would COST in output
quality. This module does NOT implement that kernel: it replaces the routed-expert arithmetic of the prefill
path with the arithmetic an e4m3 kernel would perform - weights decoded to fp16 in the rotated domain and
rounded to e4m3, activations quantized to e4m3 with a per-token scale, fp32 accumulate - with plain torch ops,
any number of launches and as much host sync as it likes (it is expected to be several times slower than the
production prefill MoE). Decode is untouched: the gate is exactly production's own prefill branch (tokens >
the fused temp rows), so CUDA-graph capture and every decode step keep production's bytes.

The numerics are the offline analysis of branch moee4m3 (tests/moee4m3/real_expert_err.py, the 'e8'/per-token
column of docs/logs/pf3000/moee4m3_real_expert_err.log: ~3.8% gate / ~6.5% FFN-output rel error per expert vs
the production path), moved from numpy on CPU shards into the serving process on the live weights:

  W_q    = decode(trellis)                                     kernels/exl3_format_ref.py semantics: the mcg
                                                                codebook fp16 values in tile order, [K, N]
  W_e4m3 = W_q -> float8_e4m3fn                                no extra scale: the codebook values are O(1)
                                                                (max |v| 3.949), so the e4m3 rounding IS the
                                                                kernel's weight error (rel_rms 2.66%)
  x_rot  = H(x * suh)                                          routed domain, fp32 Hadamard
  x_e4m3 = (x_rot / (amax/448)) -> e4m3 (saturating) * scale   per-token scale, at every GEMM input
  y      = x_e4m3 @ W_e4m3.T                                   torch fp32 GEMM = fp32 accumulate
  out    = H(y) * svh

per expert, three times (gate / up / down), with production's own operand widths where an e4m3 kernel would
see them: x2d.cast(half) and act.cast(half) - the shipped apply_exl3_python_loop / exl3_moe contract, cast
BEFORE suh, like the fused kernel. gate/up outputs stay fp32 (the accumulator), the SwiGLU clamp is
production's own statement (silu(gate.clamp(max=limit)) * up.clamp(-limit, limit), MOE_ACT_SILU = 0), the
contribution is weighted by the router's own topk weight and accumulated in fp32, and the result is returned
as x.dtype: the same output contract as apply_exl3_experts.

Where it hooks
--------------
overlay/patch_moe_e4m3_emu.py (run by patch_tf_bundle.py when GLM53_MOE_E4M3_EMU is non-empty) copies this
module into site-packages and appends an arming block at the END of site-packages
``vllm/model_executor/layers/quantization/exl3.py``: on module import it calls install(), which wraps the
production module-global ``apply_exl3_experts`` (the shipped routed-expert apply Exl3MoEMethod.apply calls by
module-global name). Appending at EOF keeps every function's text - and with it the ast fingerprints moeglue
and the K2 apply hook pin (their VERIFIED dicts) - byte identical, and works for the image's exl3.py AND the
launcher overlay's. The wrapper is armed BEFORE moeglue's plugin_install wraps the same attribute, so moeglue
stays the outermost wrapper and hands every call it declines (every prefill call) to this wrapper; a call the
gate declines goes straight to the stock function. Unset/0 knob -> the patch installs NOTHING - not even a
wrapper, the composed tree stays the previous kit's byte for byte.

  GLM53_MOE_E4M3_EMU            ""/0 (default: not even a wrapper - stock) | 1
  GLM53_MOE_E4M3_EMU_MODE       empty/e4m3 (the emulation) | prod (same plumbing, production's arithmetic:
                                the harness control the handoff OFF==prod run checks)
  GLM53_MOE_E4M3_EMU_CACHE_MB   decoded-e4m3-weight cache budget in MiB (default 2048, per device, LRU,
                                counts the fp8 W_e4m3 bytes only; 0 = decode every call)
  GLM53_MOE_E4M3_EMU_CHUNK      tokens per GEMM chunk (default 2048: bounds the fp32 temporaries to a few
                                hundred MiB per expert at production shapes)

Memory: the largest transients are one chunk's fp32 activations and one expert's fp32 weight matrix - a few
GB even at absurd sizes, never near nodeC's 40 GiB per-process cap; no dense [tokens, tokens] or
[experts, experts] tensor is ever built. State: nothing is registered with vLLM, no CUDA graph is touched
(prefill only), no allocator behavior changes.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

_log = logging.getLogger("vllm.glm53_moe_e4m3_emu")

ENV = "GLM53_MOE_E4M3_EMU"
ENV_MODE = "GLM53_MOE_E4M3_EMU_MODE"
ENV_CACHE = "GLM53_MOE_E4M3_EMU_CACHE_MB"
ENV_CHUNK = "GLM53_MOE_E4M3_EMU_CHUNK"
_BITS = 4
_E4M3_MAX = 448.0                                   # float8_e4m3fn's largest finite value
_HAD = 128                                          # kernels/exl3_format_ref.HAD
TEMP_ROWS_FALLBACK = 128                            # == EXL3_TEMP_ROWS_FUSED's default (prod_exl3_reference)

PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"

STATS = {
    "installed": False,
    "calls": 0,            # apply_exl3_experts calls the wrapper saw (any gate outcome)
    "emulated": 0,         # calls actually emulated (prefill)
    "bypassed": 0,         # calls handed to the original (tokens <= cap: decode / capture)
    "experts": 0,          # (layer, expert) FFNs computed
    "chunks": 0,           # GEMM chunks executed
    "pairs": 0,            # routed (token, expert) pairs emulated
    "decode_ms": 0.0,      # host time in the weight decode + e4m3 rounding (first calls mostly)
    "wall_ms": 0.0,        # host time inside the emulated calls
    "cached_bytes": 0,     # current fp8 weight-cache footprint
    "evictions": 0,
}
_LOGGED: set = set()


# ---------------------------------------------------------------------------------------------------------
# configuration (parsed once per process, at install; never per call)

class _Cfg:
    def __init__(self) -> None:
        self.enabled = False
        self.mode = "e4m3"          # 'prod' = the same plumbing, production's arithmetic (tests only)
        self.cache_bytes = 2048 * (1 << 20)
        self.chunk = 2048


CFG = _Cfg()
_TRUE = frozenset({"1", "on", "true", "yes"})


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    v = int(raw.strip())
    if not lo <= v <= hi:
        raise ValueError(f"{name}={raw!r} outside [{lo}, {hi}]")
    return v


def _configure() -> None:
    """Parse the knobs; a malformed value raises and install() refuses (production runs unchanged)."""
    raw = (os.environ.get(ENV) or "").strip().lower()
    if raw in ("", "0", "off", "no", "false"):
        CFG.enabled = False
        return
    if raw not in _TRUE:
        raise ValueError(f"{ENV} must be empty, 0 or 1 (got {raw!r})")
    CFG.enabled = True
    mode = (os.environ.get(ENV_MODE) or "").strip().lower()
    if mode and mode not in ("e4m3", "prod"):
        raise ValueError(f"{ENV_MODE} must be empty, e4m3 or prod (got {mode!r})")
    CFG.mode = mode or "e4m3"
    CFG.cache_bytes = _env_int(ENV_CACHE, 2048, 0, 1 << 20) * (1 << 20)
    CFG.chunk = _env_int(ENV_CHUNK, 2048, 8, 1 << 22)


# ---------------------------------------------------------------------------------------------------------
# the exl3 format: the mcg codebook, the tile order, the trellis decode (kernels/exl3_format_ref.py)

_MCG, _MASK, _FLIP = 0xCBAC1FED, 0x8FFF8FFF, 0x3B603B60
_CODEBOOK: dict[torch.device, torch.Tensor] = {}
_HADM: dict[torch.device, torch.Tensor] = {}
_P_OF: dict[torch.device, torch.Tensor] = {}


def _codebook(device: torch.device) -> torch.Tensor:
    """The fp16 value of every 16-bit state, [65536] (one fp16 addition, rounded to nearest even)."""
    cb = _CODEBOOK.get(device)
    if cb is not None:
        return cb
    import numpy as np

    s = np.arange(65536, dtype=np.uint64)
    x = (s * _MCG) & 0xFFFFFFFF
    x = (x & _MASK) ^ _FLIP
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16)
    hi = (x >> 16).astype(np.uint16).view(np.float16)
    vals = (lo.astype(np.float64) + hi.astype(np.float64)).astype(np.float16)    # exact sum, 1 fp16 rounding
    cb = torch.from_numpy(vals).to(device)
    _CODEBOOK[device] = cb
    return cb


def _hadamard(device: torch.device) -> torch.Tensor:
    """The 128x128 Sylvester Hadamard matrix / sqrt(128), fp32 (Sylvester recursion)."""
    h = _HADM.get(device)
    if h is not None:
        return h
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < _HAD:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = (h / (_HAD ** 0.5)).contiguous()
    _HADM[device] = h
    return h


def _p_of(device: torch.device) -> torch.Tensor:
    """Inverse of the tile order: p_of[r * 16 + c] = the stream index p whose value lands at (r, c)."""
    po = _P_OF.get(device)
    if po is not None:
        return po
    import numpy as np

    p = np.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    inv = np.zeros(256, dtype=np.int64)
    inv[rows * 16 + cols] = p
    po = torch.from_numpy(inv).to(device)
    _P_OF[device] = po
    return po


def decode_wq(trellis: torch.Tensor) -> torch.Tensor:
    """W_q [K, N] fp16, the weight in the rotated domain (the tiles' fp16 codebook values, in place)."""
    bits = _BITS
    t = trellis
    if t.dtype != torch.int16 or t.dim() != 3 or t.shape[-1] != 16 * bits:
        raise ValueError(f"trellis must be int16 [K/16, N/16, {16 * bits}], got {t.dtype} {tuple(t.shape)}")
    kt, nt, nw = int(t.shape[0]), int(t.shape[1]), 8 * bits
    u = t.to(torch.int64) & 0xFFFF                                   # the int16 words, unsigned
    words = u[..., 0::2] | (u[..., 1::2] << 16)                      # little-endian pairs: uint32 values
    p = torch.arange(256, device=t.device, dtype=torch.int64)
    first = p * bits + bits - 16 + 256 * bits                        # the state's first bit (non-negative)
    last = first + 16                                                # one past its last bit
    i0 = (first // 32) % nw
    i1 = ((last - 1) // 32) % nw
    shift = ((last - 1) // 32 + 1) * 32 - last
    st = (((words[..., i0] << 32) | words[..., i1]) >> shift) & 0xFFFF     # [kt, nt, 256] states
    vals = _codebook(t.device).index_select(0, st.reshape(-1)).reshape(kt, nt, 256)
    vals = vals.index_select(-1, _p_of(t.device))                    # [kt, nt, r * 16 + c]
    return vals.view(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16).contiguous()


# ---------------------------------------------------------------------------------------------------------
# the decoded-weight cache (per device, LRU, byte-budgeted: one entry = an expert's three fp8 W_e4m3)

class _WeightCache:
    """LRU keyed by the trellis storage pointer (stable for the life of the model)."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.entries: OrderedDict[int, tuple[dict, int]] = OrderedDict()

    def get(self, packs: dict) -> dict:
        key = int(packs["gate"].trellis.untyped_storage().data_ptr())
        hit = self.entries.get(key)
        if hit is not None:
            self.entries.move_to_end(key)
            return hit[0]
        import time

        t0 = time.perf_counter()
        w = {proj: (decode_wq(inner.trellis).to(torch.float8_e4m3fn), inner.suh, inner.svh)
             for proj, inner in packs.items()}
        nbytes = sum(wq.numel() for wq, _s, _v in w.values())
        STATS["decode_ms"] += (time.perf_counter() - t0) * 1000.0
        if self.budget:                                # 0 = decode every call
            while self.entries and STATS["cached_bytes"] + nbytes > self.budget:
                _k, (_v, n) = self.entries.popitem(last=False)
                STATS["cached_bytes"] -= n
                STATS["evictions"] += 1
            self.entries[key] = (w, nbytes)
            STATS["cached_bytes"] += nbytes
        return w


_CACHE: dict[torch.device, _WeightCache] = {}


def _cache(device: torch.device) -> _WeightCache:
    c = _CACHE.get(device)
    if c is None:
        c = _CACHE[device] = _WeightCache(CFG.cache_bytes)
    return c


# ---------------------------------------------------------------------------------------------------------
# the arithmetic

def _rotate(x: torch.Tensor) -> torch.Tensor:
    """H / sqrt(128) on every block of 128 along the last dim (fp32)."""
    h = _hadamard(x.device)
    shape = x.shape
    return (x.reshape(-1, shape[-1] // _HAD, _HAD) @ h.t()).reshape(shape)


def _quant_tok(x: torch.Tensor) -> torch.Tensor:
    """Per-token (last dim) e4m3 with a per-token scale: amax/448, saturating cast, rescale."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    sc = amax / _E4M3_MAX
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    return (x / sc).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32) * sc


def _proj(x_f16: torch.Tensor, w: tuple[torch.Tensor, torch.Tensor, torch.Tensor], chunk: int,
          mode: str) -> torch.Tensor:
    """One EXL3 projection in the routed domain: x fp16 [t, K] -> [t, N] fp32.

    w = (W_e4m3 fp8 [K, N], suh fp16 [K], svh fp16 [N]). mode 'e4m3' quantizes the activations per token and
    multiplies by the rounded weight; mode 'prod' is production's own arithmetic (fp16 operand, fp32
    accumulate, no rounding) and serves only the tests that pin the emulation against the offline analysis.
    """
    wq, suh, svh = w
    n = int(x_f16.shape[0])
    wq32 = wq.to(torch.float32) if mode == "e4m3" else None
    if wq32 is None:
        wq32 = wq.to(torch.float16).to(torch.float32)      # prod mode: the fp16 weight, no rounding
    out = torch.empty((n, wq32.shape[1]), dtype=torch.float32, device=x_f16.device)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        # suh applied in fp32: the kernel promotes the fp16 operands to float for the multiply (an fp16
        # product would overflow on the down projection's large FFN intermediates)
        xs = x_f16[r0:r1].float() * suh.float()
        # W_q is [K, N] in the rotated domain: y[t, n] = sum_k x_rot[t, k] * W_q[k, n] (exl3_format_ref)
        y = torch.mm(_quant_tok(_rotate(xs)) if mode == "e4m3" else _rotate(xs), wq32)
        out[r0:r1] = _rotate(y) * svh.float()
        STATS["chunks"] += 1
    return out


def _ffn(x_f16: torch.Tensor, packs: dict, limit: float, mode: str, chunk: int) -> torch.Tensor:
    """One expert's SwiGLU FFN in the emulated arithmetic: [t, H] fp16 -> [t, H] fp32."""
    g = _proj(x_f16, packs["gate"], chunk, mode)
    u = _proj(x_f16, packs["up"], chunk, mode)
    act = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    return _proj(act.contiguous().half(), packs["down"], chunk, mode)


# ---------------------------------------------------------------------------------------------------------
# the production-module hook


def _cap(layer) -> int:
    """Production's own prefill threshold: the fused kernel serves tokens <= the temp rows."""
    temps = getattr(layer, "_exl3_fused_temps", None)
    try:
        return int(temps[0].shape[1]) if temps is not None else TEMP_ROWS_FALLBACK
    except (TypeError, IndexError, ValueError):
        return TEMP_ROWS_FALLBACK


def _emu_apply(prodmod, orig, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, layer,
               *, limit: float, fused=None) -> torch.Tensor:
    import time

    STATS["calls"] += 1
    tokens = int(x.shape[-2])
    cap = _cap(layer)
    if tokens <= cap or torch.cuda.is_current_stream_capturing():    # decode / capture: production's bytes
        STATS["bypassed"] += 1
        return orig(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
    t0 = time.perf_counter()
    STATS["emulated"] += 1
    topk = int(topk_ids.reshape(tokens, -1).shape[-1])
    STATS["pairs"] += tokens * topk
    if "started" not in _LOGGED:
        _LOGGED.add("started")
        _log.warning("GLM53_MOE_E4M3_EMU=1: the prefill routed MoE runs the e4m3 emulation (weights -> e4m3, "
                     "activations -> per-token e4m3, fp32 accumulate) from this call on; decode is production's "
                     "own. This is a QUALITY probe: its speed is NOT the speed of an e4m3 kernel.")
    dev = x.device
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("glm53_moe_e4m3_emu: the layer has no _exl3_inners (EXL3 experts not built)")
    x2d = x.reshape(tokens, int(x.shape[-1])).contiguous()
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = prodmod.pin_exl3_expert_map(layer, x2d.device)
    out = torch.zeros(tokens, int(x2d.shape[-1]), dtype=torch.float32, device=dev)
    flat = ids.reshape(-1).cpu().tolist()                # host loop over the active experts: not the point
    cache = _cache(dev)
    chunk = CFG.chunk
    for e_raw in dict.fromkeys(int(v) for v in flat):    # unique, first-seen order
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            if expert_map.numel() <= e:
                continue
            mapped = int(expert_map[e].item())
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        mask = ids == e_raw
        if not bool(mask.any()):
            continue
        pairs = mask.reshape(-1).nonzero(as_tuple=True)[0]     # flat (token * topk + k) indices
        rows = torch.div(pairs, topk, rounding_mode="floor")   # the token every routed pair belongs to
        tok = torch.unique(rows)                               # each active token's FFN computed ONCE
        d = _ffn(x2d.index_select(0, tok).contiguous().half(), cache.get(inners[e]), float(limit), CFG.mode, chunk)
        STATS["experts"] += 1
        pos = torch.searchsorted(tok, rows)                    # pair -> its token's position in `tok`
        scale = weights.reshape(-1, 1).index_select(0, pairs).to(torch.float32)
        out.index_add_(0, rows, d.index_select(0, pos) * scale)   # per-pair weighted sum over token rows
        del d
    STATS["wall_ms"] += (time.perf_counter() - t0) * 1000.0
    return out.to(dtype=x.dtype)


def install(prodmod=None) -> bool:
    """Wrap prodmod.apply_exl3_experts (outermost). False = not installed (knob off, module missing, drift)."""
    try:
        _configure()
    except ValueError as exc:
        _log.warning("glm53_moe_e4m3_emu not installed: %r (production kernels unchanged)", exc)
        return False
    if not CFG.enabled:
        return False
    if prodmod is None:
        import importlib

        prodmod = importlib.import_module(PROD_MODULE)
    orig = getattr(prodmod, "apply_exl3_experts", None)
    if orig is None or getattr(orig, "_glm53_moe_e4m3_emu", False):
        _log.warning("glm53_moe_e4m3_emu not installed: %s", "no apply_exl3_experts" if orig is None
                     else "already installed")
        return False

    def apply_exl3_experts(x, topk_ids, topk_weights, layer, *, limit=10.0, fused=None):
        return _emu_apply(prodmod, orig, x, topk_ids, topk_weights, layer, limit=float(limit), fused=fused)

    apply_exl3_experts.__doc__ = getattr(orig, "__doc__", None)
    apply_exl3_experts._glm53_moe_e4m3_emu = True
    apply_exl3_experts._glm53_moe_e4m3_emu_orig = orig
    prodmod.apply_exl3_experts = apply_exl3_experts
    STATS["installed"] = True
    _log.info("glm53_moe_e4m3_emu installed on %s.apply_exl3_experts: prefill routed MoE runs the e4m3 "
              "emulation while GLM53_MOE_E4M3_EMU=1; cache %d MiB, chunk %d tokens", PROD_MODULE,
              CFG.cache_bytes >> 20, CFG.chunk)
    return True


def uninstall(prodmod=None) -> None:
    """Restore the original (tests)."""
    import importlib

    prodmod = prodmod or importlib.import_module(PROD_MODULE)
    fn = getattr(prodmod, "apply_exl3_experts", None)
    if fn is not None and getattr(fn, "_glm53_moe_e4m3_emu", False):
        prodmod.apply_exl3_experts = fn._glm53_moe_e4m3_emu_orig
        STATS["installed"] = False


def plugin_install() -> None:
    """Called by integrate.plugin_install() after every other feature: the wrapper must be the outermost."""
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moe_e4m3_emu not installed (production kernels unchanged): %r", exc)
