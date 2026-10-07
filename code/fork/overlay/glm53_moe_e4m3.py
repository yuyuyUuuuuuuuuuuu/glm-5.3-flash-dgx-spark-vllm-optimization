"""glm53_moe_e4m3 — the routed-MoE PREFILL on e4m3 tensor cores (GLM53_MOE_E4M3, docs/MOE_E4M3.md).

Replaces the arithmetic of production's prefill routed-expert path (E3 grouped fat kernels + the thin kernel for
experts at or below the fused cap) with kernels/moe_e4m3.cu (extension glm53_moe_e4m3_ext; module + extension ship in
the bundle's overlay dir, installed into site-packages only by overlay/patch_moe_e4m3.py when GLM53_MOE_E4M3=1): the EXL3 trellis is
decoded in registers, rounded to e4m3 (no scale) and multiplied on mma.sync m16n8k32 e4m3 with per-row e4m3
activations and fp32 accumulation. The numerics are the e4m3 quality emulation's (branch moefq,
glm53_moe_e4m3_emu = GLM53_MOE_E4M3_EMU): the kernel is the fast implementation of that spec, so the emulation's KL
measurement is the quality cost of turning this on.

Hook: wraps the production module-global ``apply_exl3_experts`` (what Exl3MoEMethod.apply calls, the same hook point
as the emulation). A call is served when ALL hold, else it goes to the wrapped function unchanged:
  - tokens > the layer's fused temp rows (production's own prefill branch; decode and every CUDA-graph capture size
    are at or below it) and no stream capture is in progress;
  - ``fused`` is not False, the layer has production's pointer tables (``_exl3_ptrs``), K4 trellis, hidden 4096,
    local intermediate 1024, gate/up sharing one input rotation (``_exl3_shared_w13_suh``: one gathered row serves
    gate and up) and passed its one-time self-test against the spec (below).
Unset / empty / 0: install() does nothing at all (not even a wrapper): production byte-identical; any other value
than 1: WARNING, nothing installed. NOTE: "tokens > the fused cap" is the gate, not "is a prefill": with
GLM53_MIXED_PREFILL_CHUNK=0 the decode/verify tokens scheduled into the same step as a prefill chunk go through this
path too (the KL measurement covers exactly that), and a pure decode/verify step goes through it if MAX_NUM_SEQS x
(K+1) exceeds the cap (checked and WARNED at install and in the summary; 4 x 8 = 32 <= 256 in production).

Self-test (once per layer, at MODEL LOAD: Exl3MoEMethod.process_weights_after_loading is wrapped, so it runs before
the profile run and before the engine is ready; lazily at the first eligible call if that method is missing): a
192-token synthetic call routed to 8 of the layer's real experts through this path and through the spec's torch
arithmetic; the layer is served only if the outputs agree to rel-L2 <= 5e-3 (measured ~1e-3). A Python-level failure
or exception keeps the layer on production's path for the life of the process (WARNING, counted); a device-side
fault (illegal address, the fused kernel's watchdog __trap) is a sticky CUDA error that kills the process - at load
time, i.e. a failed boot, not a silent fallback. Summary line at the first apply call: "glm53_moe_e4m3 summary:
N/M layers served, F fell back (...)".

Memory: none of its own at steady state; the per-call buffers are carved from production's grouped-fat scratch
(``_grouped_scratch``: h13 fp16 [rows, 4096] holds the e4m3 gate/up rows + the e4m3 down rows + both scale
vectors, h2 fp16 [rows, 1024] holds the fp16 SwiGLU output), which production allocates for the same prefill
calls anyway. Per call: routing tables O(tokens x top-k) and the fp32 output [tokens, 4096] (production allocates
the same output).
"""
from __future__ import annotations

import logging
import os

import torch

_log = logging.getLogger("vllm.glm53_moe_e4m3")

ENV = "GLM53_MOE_E4M3"
# GLM53_MOE_E4M3_DOWN: unset / "" / "e4m3" = the down projection on e4m3 too (the original spec); "f16" = gate/up on
# e4m3, the down projection on production's operand widths (fp16 rotated input x fp16 trellis decode, fp32 accumulate,
# mma m16n8k16): removes 2 of the 4 e4m3 roundings (docs/MOE2.md: ~35-40 % of the added noise variance) for ~+1.5 ms
# per 13,824-token layer call. Only read when GLM53_MOE_E4M3=1; any other value: install() refuses (WARNING).
ENV_DOWN = "GLM53_MOE_E4M3_DOWN"
DOWN16 = {"on": False, "layers": None}
# GLM53_MOE_E4M3_LAYERS (branch moe3, docs/MOE3.md): unset / "" = every MoE layer (the original behaviour, unchanged);
# else a comma list of model layer indices and ranges ("3-13,20,40-44"): only those layers' prefill calls take the e4m3
# path, every other layer stays on production's path (with GLM53_MOE_FUSED16=1: production's arithmetic in the faster
# P16 schedule). The index is read from the RoutedExperts layer_name ("...layers.<i>.mlp.experts"); a layer without
# one is not selected. Only read when GLM53_MOE_E4M3=1; an unparsable value: install() refuses (WARNING).
ENV_LAYERS = "GLM53_MOE_E4M3_LAYERS"
LAYERS = {"sel": None}
# GLM53_MOE_E4M3_DOWN_LAYERS (branch moe3): unset / "" = GLM53_MOE_E4M3_DOWN decides for every layer (unchanged);
# else a layer list (same syntax as GLM53_MOE_E4M3_LAYERS): exactly those layers run the down projection on fp16
# operands (fused variant 16, the f16-down spec in their self-test), every other served layer the e4m3 down - the
# ~+4 ms/layer quality fix only where the layers are sensitive. Only read when GLM53_MOE_E4M3=1; unparsable: refused.
ENV_DOWN_LAYERS = "GLM53_MOE_E4M3_DOWN_LAYERS"
# GLM53_MOE_E4M3_ACC (branch opt-moe, docs/OPT_MOE.md): unset / "" / "f32" = the fp32 accumulator + production's
# .to(bf16) (unchanged); "bf16" = the down epilogue rounds each weighted contribution to bf16 and adds it with the L2's
# bf16 atomics (red.add.noftz.v4.bf16x2) straight into the bf16 output that apply returns (gather2 zeroes it): half the
# scatter bytes, half the zeroing, no cast pass (-7.0 ms per 13,824-token layer call, -2.8 at 4,289). Numerics: a
# bf16 rounding per contribution and per partial sum (<= 8 terms): 3.9e-3 rel-L2 vs the fp32 accumulator (the cast
# alone 1.7e-3), against the e4m3 path's 6.5e-2 vs production (0.3 % of its added variance). Only read when
# GLM53_MOE_E4M3=1; any other value: install() refuses (WARNING).
ENV_ACC = "GLM53_MOE_E4M3_ACC"
ACC = {"bf16": False}
# GLM53_MOE_E4M3_FOLD_SHARED (branch opt-moe, docs/OPT_MOE.md): unset / "" / "0" = unchanged; "1" (requires
# GLM53_MOE_E4M3_ACC=bf16) = a served call accumulates the routed contributions straight into the shared experts'
# bf16 output (vLLM's MoERunner runs the shared experts BEFORE the routed ones for prefill-sized batches,
# SharedExpertsOrder.NO_OVERLAP) instead of a zeroed buffer, and vLLM's own "shared_output + fused_output" add is
# skipped for that call: no zeroing pass in gather2, no add pass (~2 ms per 13,824-token layer call). Installed as two
# runtime wrappers on vllm's moe_runner (MoERunner.forward arms the fold only when the runner's result is exactly
# shared + routed: routed_scaling_factor 1.0, no routed output transform, fused output not pre-reduced, not
# sequence-parallel, no zero-expert router; moe_runner._unpack drops the shared half of the folded call's result).
# Numerics: the shared output becomes the first term of the bf16 accumulation (vs one bf16 add at the end). Only read
# when GLM53_MOE_E4M3=1; any other value, or "1" without ACC=bf16: install() refuses (WARNING).
# GLM53_MOE_E4M3_TOKGATHER (branch opt-moe): unset / "" / "1" = on, "0" = off. On: when every expert of a layer has the
# same w13 suh (checked once per layer, cached on the layer; true for GLM-5.3-Flash's EXL3 checkpoint), the gathered
# e4m3 gate/up input of a (token, expert) pair does not depend on the expert, so gather_tok writes ONE row per token
# (T rows instead of T x topk: 57 MB instead of 453 MB at 13,824 tokens) and the fused kernel's gate/up jobs read
# their A rows through row_token (fused variants + 2048). Same bytes, same arithmetic as the per-pair rows (the
# output is in the fp32 / bf16 atomics-order class of the per-pair path). Only read when GLM53_MOE_E4M3=1.
ENV_TG = "GLM53_MOE_E4M3_TOKGATHER"
TG = {"on": True}
ENV_FOLD = "GLM53_MOE_E4M3_FOLD_SHARED"
FOLD = {"on": False, "wrapped": False}
# GLM53_MOE_E4M3_MAINLOOP (branch opt-moe2, docs/OPT_MOE2.md): unset / "" / "0" = the shipped fused kernel
# (unchanged); "1" = fused variants + 8192: the same mainloop with fewer instructions per stage - the trellis decode
# with the mask / XOR applied after the half-word permutation as one LOP3 each and byte-aligned fields as one PRMT
# (the same integer function), and the A copy addresses of a job computed once into a per-thread shared-memory table
# instead of per stage (-2.0 ms per 13,824-token layer call, -0.5 at 4,289; the mainloop is issue-bound). The same
# bytes, fragments and mma order: every intermediate (a16, a8d, dsc) is bit-identical and the output is in the
# atomics-order class of the shipped kernel. 16 KB more dynamic shared memory (100,880 B per CTA incl. static with TG).
# Only read when GLM53_MOE_E4M3=1; any other value: install() refuses (WARNING).
ENV_MS = "GLM53_MOE_E4M3_MAINLOOP"
MS = {"on": False}
_FOLD_CTX = {"se": None, "pending": False}
ACC_SELFTEST_TOL = 8e-3     # bf16 accumulator vs the fp32 accumulator on the self-test call (measured 3.9e-3)
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
HIDDEN, INTER = 4096, 1024
SELFTEST_TOL = 5e-3
TEMP_ROWS_FALLBACK = 128
# Schedule: the segments are split into NCHUNKS chunks; gate/up of chunk c+1 (main stream, GU_GRID CTAs) runs
# concurrently with actq + down of chunk c (side stream, DN_GRID CTAs), so the down's DRAM-bound fp32 scatter-add
# overlaps the compute-bound gate/up. The first gate/up and the last down use every SM. NCHUNKS 1 = sequential.
SCHED = {"mode": "fused", "lag": 12, "nchunks": 1, "gu_grid": 0, "dn_grid": 0}
_SIDE: dict = {}
_SYNC: dict = {}

STATS = {"installed": False, "calls": 0, "served": 0, "passed": 0, "selftests": 0, "selftest_failed": 0,
         "selftest_raised": 0, "tokens": 0, "layers": 0, "layers_ok": 0, "layers_fallback": 0,
         "decode_bound_warn": 0, "layers_unselected": 0, "folded": 0, "tg_layers": 0}
_FALLBACK_WHY: dict = {}
_SUMMARY = {"logged": None}
_LOGGED: set = set()
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        import glm53_moe_e4m3_ext as e

        _EXT = e
    return _EXT


def down_mode(environ=None) -> bool:
    """True for GLM53_MOE_E4M3_DOWN=f16; False for unset / "" / "e4m3"; anything else = ValueError."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_DOWN) or "").strip().lower()
    if raw in ("", "e4m3"):
        return False
    if raw == "f16":
        return True
    raise ValueError(f"{ENV_DOWN} must be empty, e4m3 or f16 (got {raw!r})")


def acc_mode(environ=None) -> bool:
    """True for GLM53_MOE_E4M3_ACC=bf16; False for unset / "" / "f32"; anything else = ValueError."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_ACC) or "").strip().lower()
    if raw in ("", "f32"):
        return False
    if raw == "bf16":
        return True
    raise ValueError(f"{ENV_ACC} must be empty, f32 or bf16 (got {raw!r})")


def tg_mode(environ=None) -> bool:
    """True for GLM53_MOE_E4M3_TOKGATHER unset / "" / "1"; False for "0"; anything else = ValueError."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_TG) or "").strip()
    if raw in ("", "1"):
        return True
    if raw == "0":
        return False
    raise ValueError(f"{ENV_TG} must be empty, 0 or 1 (got {raw!r})")


def tok_gather_ok(layer) -> bool:
    """Every local expert's gate suh equals expert 0's and gate suh == up suh (cached on the layer; one host sync)."""
    ok = getattr(layer, "_glm53_tok_gather", None)
    if ok is None:
        try:
            s = layer.w13_suh
            ok = bool(getattr(layer, "_exl3_shared_w13_suh", False)) and s.dim() == 3 and int(s.shape[1]) == 2 and \
                bool(torch.equal(s[:, 0], s[0:1, 0].expand_as(s[:, 0])))
        except Exception:  # noqa: BLE001
            ok = False
        layer._glm53_tok_gather = ok
        if ok:
            STATS["tg_layers"] += 1
    return ok


def ms_mode(environ=None) -> bool:
    """True for GLM53_MOE_E4M3_MAINLOOP=1; False for unset / "" / "0"; anything else = ValueError."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_MS) or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV_MS} must be empty, 0 or 1 (got {raw!r})")


def fold_mode(environ=None) -> bool:
    """True for GLM53_MOE_E4M3_FOLD_SHARED=1; False for unset / "" / "0"; anything else = ValueError."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_FOLD) or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV_FOLD} must be empty, 0 or 1 (got {raw!r})")


def fold_buffer(se, tokens: int, x: torch.Tensor):
    """The shared experts' pending output if the routed sum may be accumulated into it, else None."""
    if se is None:
        return None
    try:
        buf = se._output[se._output_idx]
    except Exception:  # noqa: BLE001
        return None
    if (not isinstance(buf, torch.Tensor) or buf.dtype != torch.bfloat16 or x.dtype != torch.bfloat16
            or buf.dim() != 2 or tuple(buf.shape) != (tokens, HIDDEN) or not buf.is_contiguous()
            or buf.device != x.device or buf.data_ptr() == x.data_ptr()):
        return None
    return buf


def _fold_allowed(runner) -> bool:
    """The runner's result is exactly shared_output + fused_output (then the optional all-reduce)."""
    try:
        if getattr(runner, "_shared_experts", None) is None:
            return False
        if float(getattr(runner, "routed_scaling_factor", 1.0)) != 1.0:
            return False
        if getattr(runner, "routed_output_transform", None) is not None:
            return False
        if bool(runner._fused_output_is_reduced):
            return False
        if bool(getattr(runner.moe_config, "is_sequence_parallel", False)):
            return False
        if type(getattr(runner, "router", None)).__name__ == "ZeroExpertRouter":
            return False
        if bool(getattr(runner._shared_experts, "enable_dbo", False)):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


# the lines of MoERunner.forward the fold relies on (vLLM drift = no fold): the tuple from the custom op is unpacked by
# the module-level _unpack, and the result is exactly shared_output + fused_output when shared_output is not None
FOLD_FINGERPRINT = ("shared_output, fused_output = _unpack(result)", "if shared_output is not None:",
                    "result = shared_output + fused_output", "self._shared_experts.output if self._shared_experts")


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _install_fold(R=None) -> str:
    """Wrap vllm's MoERunner.forward and moe_runner._unpack (idempotent). Returns "ok" or the reason it did not.
    R: the moe_runner module (tests pass a stand-in)."""
    if FOLD["wrapped"]:
        return "ok"
    if R is None:
        try:
            import vllm.model_executor.layers.fused_moe.runner.moe_runner as R
        except Exception as exc:  # noqa: BLE001
            return f"moe_runner import failed: {exc!r}"
    cls = getattr(R, "MoERunner", None)
    fwd = getattr(cls, "forward", None) if cls is not None else None
    unpack = getattr(R, "_unpack", None)
    if fwd is None or unpack is None or not hasattr(cls, "_apply_quant_method"):
        return "moe_runner has no MoERunner.forward / _unpack / _apply_quant_method"
    try:
        import inspect

        src = inspect.getsource(fwd) + inspect.getsource(cls._apply_quant_method)
    except Exception as exc:  # noqa: BLE001
        return f"MoERunner source unavailable ({exc!r})"
    missing = [f for f in FOLD_FINGERPRINT if f not in src]
    if missing:
        return f"MoERunner.forward / _apply_quant_method changed (missing {missing})"

    def forward(self, *a, **k):
        # opt-moe-rev: under torch.compile tracing (vLLM CompilationMode != NONE) the Python state below would be
        # evaluated at trace time while the custom op runs at replay: never arm there (the traced graph = vLLM's own,
        # _serve sees se None and does not fold). Production runs CompilationMode.NONE, where this is always False.
        if _is_compiling():
            return fwd(self, *a, **k)
        prev = (_FOLD_CTX["se"], _FOLD_CTX["pending"])
        _FOLD_CTX["se"] = self._shared_experts if (FOLD["on"] and _fold_allowed(self)) else None
        _FOLD_CTX["pending"] = False
        try:
            return fwd(self, *a, **k)
        finally:
            _FOLD_CTX["se"], _FOLD_CTX["pending"] = prev

    def _unpack(result):
        if not _is_compiling() and _FOLD_CTX["pending"] and isinstance(result, tuple) and len(result) == 2:
            _FOLD_CTX["pending"] = False
            return (None, result[1])     # result[1] already holds shared + routed
        return unpack(result)

    forward.__doc__ = fwd.__doc__
    forward._glm53_moe_e4m3 = True
    forward._glm53_moe_e4m3_orig = fwd
    _unpack._glm53_moe_e4m3 = True
    _unpack._glm53_moe_e4m3_orig = unpack
    cls.forward = forward
    R._unpack = _unpack
    FOLD["wrapped"] = True
    FOLD["module"] = R
    return "ok"


def _uninstall_fold() -> None:
    if not FOLD["wrapped"]:
        return
    R = FOLD.pop("module")

    fn = R.MoERunner.forward
    while getattr(fn, "_glm53_moe_e4m3", False):
        R.MoERunner.forward = fn = fn._glm53_moe_e4m3_orig
    fn = R._unpack
    while getattr(fn, "_glm53_moe_e4m3", False):
        R._unpack = fn = fn._glm53_moe_e4m3_orig
    FOLD["wrapped"] = False


def layers_mode(environ=None):
    """None for unset / "" (every layer); else the frozenset of selected layer indices. ValueError if unparsable."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_LAYERS) or "").strip()
    if raw == "":
        return None
    sel = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo < 0 or hi < lo or hi > 4096:
                raise ValueError(f"{ENV_LAYERS}: bad range {part!r}")
            sel.update(range(lo, hi + 1))
        else:
            v = int(part)
            if v < 0 or v > 4096:
                raise ValueError(f"{ENV_LAYERS}: bad layer {part!r}")
            sel.add(v)
    if not sel:
        raise ValueError(f"{ENV_LAYERS}={raw!r} selects no layer")
    return frozenset(sel)


def down_layers_mode(environ=None):
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_DOWN_LAYERS) or "").strip()
    if raw == "":
        return None
    return layers_mode({ENV_LAYERS: raw})


def down16_for(layer) -> bool:
    sel = DOWN16.get("layers")
    if sel is None:
        return bool(DOWN16["on"])
    return layer_index(layer) in sel


def layer_index(layer):
    import re

    m = re.search(r"layers\.(\d+)\.", str(getattr(layer, "layer_name", "") or ""))
    return int(m.group(1)) if m else None


def layer_selected(layer) -> bool:
    sel = LAYERS["sel"]
    return sel is None or layer_index(layer) in sel


def enabled(environ=None) -> bool:
    """unset / empty / 0 = off, 1 = on, anything else = ValueError (install() then refuses with a WARNING) - exactly
    what the kit's start.sh accepts (_glm53_validate_bool_flag) and docs/MOE_E4M3.md states."""
    env = os.environ if environ is None else environ
    raw = (env.get(ENV) or "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV} must be empty, 0 or 1 (got {raw!r})")


def _cap(layer) -> int:
    temps = getattr(layer, "_exl3_fused_temps", None)
    try:
        return int(temps[0].shape[1]) if temps is not None else TEMP_ROWS_FALLBACK
    except (TypeError, IndexError, ValueError):
        return TEMP_ROWS_FALLBACK


def layer_reason(layer) -> str | None:
    """None if the layer's shapes/format are served, else why not (static per layer)."""
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


# ---------------------------------------------------------------------------------------------------------
# the computation

def _buffers(prodmod, device, P: int):
    """(a8 u8 [P,4096], a8d u8 [P,1024], asc f32 [P], dsc f32 [P], a16 f16 [P,1024]) views into production's
    grouped scratch (or a private cache when the module has none)."""
    sc = None
    gs = getattr(prodmod, "_grouped_scratch", None) if prodmod is not None else None
    if gs is not None:
        sc = gs(device, P, HIDDEN, INTER)
    else:
        key = ("own", str(device))
        sc = _OWN.get(key)
        if sc is None or int(sc["h13"].shape[0]) < max(P, 256):
            n = max(P, 256)
            sc = _OWN[key] = {"h13": torch.empty((n, HIDDEN), dtype=torch.float16, device=device),
                              "h2": torch.empty((n, INTER), dtype=torch.float16, device=device)}
    h13, h2 = sc["h13"], sc["h2"]
    assert int(h13.shape[0]) >= P and int(h2.shape[0]) >= P and h13.is_contiguous() and h2.is_contiguous()
    flat = h13.view(-1).view(torch.uint8)                      # rows x 8192 bytes
    o = 0
    a8 = flat[o:o + P * HIDDEN].view(P, HIDDEN); o += P * HIDDEN
    a8d = flat[o:o + P * INTER].view(P, INTER); o += P * INTER
    asc = flat[o:o + 4 * P].view(torch.float32); o += 4 * P
    dsc = flat[o:o + 4 * P].view(torch.float32); o += 4 * P
    assert o <= flat.numel()
    return a8, a8d, asc, dsc, h2[:P]


_OWN: dict = {}


def plan(prodmod, ids: torch.Tensor, weights: torch.Tensor, n_exp: int, expert_map):
    """Routing tables on the device, no host sync. Rows = routed (token, k) pairs sorted by local expert (stable),
    pairs of non-local / invalid experts last (never computed)."""
    T, topk = int(ids.shape[0]), int(ids.shape[1])
    P = T * topk
    dev = ids.device
    if prodmod is not None and hasattr(prodmod, "map_topk_to_local"):
        local = prodmod.map_topk_to_local(ids, n_exp, expert_map)
    else:
        flat = ids.reshape(-1)
        local = torch.where((flat < 0) | (flat >= n_exp), torch.full_like(flat, n_exp), flat)
    local = local.reshape(-1).to(torch.long)
    order = torch.argsort(local, stable=True)
    counts = torch.zeros(n_exp + 1, dtype=torch.long, device=dev)
    counts.scatter_add_(0, local, torch.ones_like(local))
    pos = torch.empty(P, dtype=torch.int32, device=dev)
    pos[order] = torch.arange(P, dtype=torch.int32, device=dev)
    t = {
        "T": T, "topk": topk, "P": P, "n_exp": n_exp,
        "local": local.to(torch.int32),
        "pos": pos,
        "row_token": torch.div(order, topk, rounding_mode="floor"),
        "row_weight": weights.reshape(-1).to(torch.float32).index_select(0, order),
        "row_expert": local.index_select(0, order).to(torch.int32),
    }
    se, sr0, sr, ns, nr = _ext().seg_tables(counts[:n_exp].contiguous(), int(_ext().tile_rows()), P)
    t.update(seg_expert=se, seg_row0=sr0, seg_rows=sr, num_segs=ns, num_rows=nr)
    return t


def t_topk_ok(ids: torch.Tensor) -> bool:
    return int(ids.shape[-1]) in (2, 4, 8)


def run(prodmod, x2d: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, layer, limit: float,
        expert_map=None, out: torch.Tensor | None = None, keep: dict | None = None, sched: dict | None = None,
        fold_into: torch.Tensor | None = None) -> torch.Tensor:
    """The routed experts of one layer for every token of x2d: fp32 [T, 4096] (sum over the routed experts of
    weight x expert FFN). keep: if a dict, the intermediate buffers are stored in it (tests). fold_into: a bf16
    [T, 4096] tensor (the shared experts' output) the routed sum is ADDED into when the bf16 accumulator applies to
    this call; the caller checks `result is fold_into` (otherwise it was ignored and the result is the plain sum)."""
    ext = _ext()
    ptrs = layer._exl3_ptrs
    n_exp = len(layer._exl3_inners)
    T = int(x2d.shape[0])
    dev = x2d.device
    xc = x2d.contiguous()
    # gather2 reads bf16 directly (rounded to fp16 in the kernel = x.half(), bit for bit): no extra pass over x
    xh = xc if (xc.dtype == torch.bfloat16 and t_topk_ok(ids.reshape(T, -1))) else xc.half()
    t = plan(prodmod, ids.reshape(T, -1).to(torch.long), weights.reshape(T, -1), n_exp, expert_map)
    a8, a8d, asc, dsc, a16 = _buffers(prodmod, dev, t["P"])
    sc_ = dict(SCHED)
    if sched:
        sc_.update(sched)
    # opt-moe: acc "bf16" = the bf16 accumulator (fused schedule only): the down epilogue adds bf16-rounded
    # contributions in bf16 (red.add.noftz.bf16x2) into a bf16 [T, 4096] output, zeroed by gather2
    acc = sc_.get("acc", "bf16" if ACC["bf16"] else "f32")
    acc_bf16 = (acc == "bf16" and sc_.get("mode") == "fused" and t["topk"] in (2, 4, 8)
                and x2d.dtype == torch.bfloat16)        # a bf16 result only where apply returns bf16 anyway
    if (fold_into is not None and out is None and acc_bf16 and fold_into.dtype == torch.bfloat16
            and tuple(fold_into.shape) == (T, HIDDEN) and fold_into.is_contiguous()):
        out = fold_into                  # accumulate onto the shared output: gather2 must not zero it (out given)
    variant = int(sc_.get("variant", 16 if down16_for(layer) else 0))
    tg = (bool(sc_.get("tg", TG["on"])) and sc_.get("mode") == "fused" and t["topk"] in (2, 4, 8)
          and variant in (0, 16) and tok_gather_ok(layer))
    # opt-moe2: the lean mainloop (+ 8192), any shipped fused variant (0 / 16, then + 2048 for TG), same arithmetic
    if bool(sc_.get("ms", MS["on"])) and sc_.get("mode") == "fused" and variant in (0, 16):
        variant += 8192
    if tg:
        # one gathered row per token (a8[t], asc[t]); the fused gate/up jobs index it through row_token
        if out is None:
            out = torch.empty(T, HIDDEN, dtype=torch.bfloat16 if acc_bf16 else torch.float32, device=dev)
            ext.gather_tok(xh, ptrs["gate_suh"], a8, asc, out)
        else:
            ext.gather_tok(xh, ptrs["gate_suh"], a8, asc, None)
        variant += 2048
    elif t["topk"] in (2, 4, 8):
        # single-pass gather; it also zeroes the fp32 accumulator (one fewer launch / memset)
        if out is None:
            out = torch.empty(T, HIDDEN, dtype=torch.bfloat16 if acc_bf16 else torch.float32, device=dev)
            ext.gather2(xh, t["local"], t["pos"], ptrs["gate_suh"], a8, asc, out, t["topk"], n_exp)
        else:
            ext.gather2(xh, t["local"], t["pos"], ptrs["gate_suh"], a8, asc, None, t["topk"], n_exp)
    else:
        if out is None:
            out = torch.zeros(T, HIDDEN, dtype=torch.float32, device=dev)
        ext.gather(xh, t["local"], t["pos"], ptrs["gate_suh"], a8, asc, t["topk"], n_exp, 0)
    if sc_.get("mode") == "fused":
        need = 1 + 2 * int(t["seg_expert"].numel())
        sync = _SYNC.get(dev)
        if sync is None or sync.numel() < need:
            sync = _SYNC[dev] = torch.zeros(max(need, 4096), dtype=torch.int32, device=dev)
        sync[:need].zero_()
        ext.fused(a8, asc, a8d, dsc, a16, out, ptrs["gate_trellis"], ptrs["up_trellis"], ptrs["gate_svh"],
                  ptrs["up_svh"], ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"], t["row_token"],
                  t["row_weight"], t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], sync, float(limit),
                  int(sc_.get("lag", 12)), int(sc_.get("grid", 0)), variant)
        if keep is not None:
            keep.update(t)
            keep.update(a8=a8, a8d=a8d, asc=asc, dsc=dsc, a16=a16, tg=tg)
            if tg:
                # the per-pair view of what the kernel read: row r of the gate/up input = a8[row_token[r]]
                rtk = t["row_token"][: t["P"]]
                keep.update(a8=a8.index_select(0, rtk), asc=asc.index_select(0, rtk))
        return out
    if DOWN16["on"]:
        raise RuntimeError("GLM53_MOE_E4M3_DOWN=f16 exists only in the fused schedule")
    nc = int(sc_["nchunks"])
    gg = int(sc_["gu_grid"])
    dg = int(sc_["dn_grid"])

    def gu(c, grid):
        ext.gateup(a8, asc, ptrs["gate_trellis"], ptrs["up_trellis"], ptrs["gate_svh"], ptrs["up_svh"], a16,
                   t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], float(limit), 0, c, nc, grid)

    def dn(c, grid):
        ext.actq(a16, t["row_expert"], ptrs["down_suh"], a8d, dsc, t["seg_row0"], t["seg_rows"], t["num_segs"], c, nc)
        ext.down(a8d, dsc, ptrs["down_trellis"], ptrs["down_svh"], out, t["row_token"], t["row_weight"],
                 t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], 0, c, nc, grid)

    if nc <= 1:
        gu(0, 0)
        dn(0, 0)
    else:
        main = torch.cuda.current_stream(dev)
        side = _SIDE.get(dev)
        if side is None:
            side = _SIDE[dev] = torch.cuda.Stream(device=dev)
        evs = []
        for c in range(nc):
            gu(c, 0 if c == 0 else gg)
            ev = torch.cuda.Event()
            ev.record(main)
            evs.append(ev)
        with torch.cuda.stream(side):
            for c in range(nc):
                side.wait_event(evs[c])
                dn(c, 0 if c == nc - 1 else dg)
        main.wait_stream(side)
    if keep is not None:
        keep.update(t)
        keep.update(a8=a8, a8d=a8d, asc=asc, dsc=dsc, a16=a16)
    return out


# ---------------------------------------------------------------------------------------------------------
# the spec in torch (for the self-test; same arithmetic as glm53_moe_e4m3_emu, kernels/exl3_format_ref.py format)

_MCG, _MASK, _FLIP = 0xCBAC1FED, 0x8FFF8FFF, 0x3B603B60
_SPEC_CACHE: dict = {}


def _spec_tables(device):
    tb = _SPEC_CACHE.get(device)
    if tb is not None:
        return tb
    s = torch.arange(65536, dtype=torch.int64)
    x = (s * _MCG) & 0xFFFFFFFF
    x = (x & _MASK) ^ _FLIP
    lo = (x & 0xFFFF).to(torch.int16).view(torch.float16).double()
    hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).double()
    cb = (lo + hi).to(torch.float16).to(device)
    p = torch.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    inv = torch.zeros(256, dtype=torch.long)
    inv[rows * 16 + cols] = p
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    tb = (cb, inv.to(device), (h / 128 ** 0.5).to(device))
    _SPEC_CACHE[device] = tb
    return tb


def spec_decode(trellis: torch.Tensor) -> torch.Tensor:
    """W_q [K, N] fp16 from int16 [K/16, N/16, 64] (exl3_format_ref.unpack)."""
    cb, inv, _ = _spec_tables(trellis.device)
    kt, nt = int(trellis.shape[0]), int(trellis.shape[1])
    u = trellis.to(torch.int64) & 0xFFFF
    w = u[..., 0::2] | (u[..., 1::2] << 16)
    p = torch.arange(256, device=trellis.device, dtype=torch.int64)
    first = p * 4 + 4 - 16 + 256 * 4
    last = first + 16
    i0, i1 = (first // 32) % 32, ((last - 1) // 32) % 32
    sh = ((last - 1) // 32 + 1) * 32 - last
    st = (((w[..., i0] << 32) | w[..., i1]) >> sh) & 0xFFFF
    vals = cb.index_select(0, st.reshape(-1)).reshape(kt, nt, 256).index_select(-1, inv)
    return vals.view(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16).contiguous()


def _spec_rot(x):
    h = _spec_tables(x.device)[2]
    s = x.shape
    return (x.reshape(-1, s[-1] // 128, 128) @ h.t()).reshape(s)


def _spec_q(x):
    sc = x.abs().amax(dim=-1, keepdim=True) / 448.0
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    return (x / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * sc


def spec_proj(x16, trellis, suh, svh):
    w = spec_decode(trellis).to(torch.float8_e4m3fn).to(torch.float32)
    y = _spec_q(_spec_rot(x16.float() * suh.float())) @ w
    return _spec_rot(y) * svh.float()


def spec_proj16(x16, trellis, suh, svh):
    """Production's operand widths: h = fp16(H(float(x16) * suh) * r), y = h @ W_q (fp16 decode), fp32 accumulate."""
    w = spec_decode(trellis).to(torch.float32)
    y = _spec_rot(x16.float() * suh.float()).half().float() @ w
    return _spec_rot(y) * svh.float()


def spec_ffn(x16, inner, limit, down16=None):
    if down16 is None:
        down16 = DOWN16["on"]
    g = spec_proj(x16, inner["gate"].trellis, inner["gate"].suh, inner["gate"].svh)
    u = spec_proj(x16, inner["up"].trellis, inner["up"].suh, inner["up"].svh)
    a = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    proj = spec_proj16 if down16 else spec_proj
    return proj(a.half(), inner["down"].trellis, inner["down"].suh, inner["down"].svh)


def selftest(prodmod, layer, limit: float, tokens: int = 192, experts=None) -> dict:
    """Synthetic call on the layer's real experts through run() and through the spec (fp32 matmul precision)."""
    dev = layer._exl3_ptrs["gate_trellis"].device
    n_exp = len(layer._exl3_inners)
    g = torch.Generator(device="cpu").manual_seed(1234)
    if experts is None:
        experts = sorted(set([0, n_exp - 1] + torch.randperm(n_exp, generator=g)[:6].tolist()))[:8]
    E = torch.tensor(experts, dtype=torch.long)
    topk = min(8, len(experts))
    ids = torch.stack([E[torch.randperm(len(experts), generator=g)[:topk]] for _ in range(tokens)]).to(dev)
    w = torch.rand(tokens, topk, generator=g).to(dev)
    w = w / w.sum(-1, keepdim=True)
    x = (torch.randn(tokens, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    out = run(prodmod, x.reshape(tokens, HIDDEN), ids, w, layer, limit, sched={"acc": "f32"})
    outb = run(prodmod, x.reshape(tokens, HIDDEN), ids, w, layer, limit) if ACC["bf16"] else None
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=dev)
        xh = x.half()
        for e in experts:
            m = ids == e
            tok, kk = m.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            d = spec_ffn(xh.index_select(0, tok), layer._exl3_inners[e], limit, down16=down16_for(layer))
            ref.index_add_(0, tok, d * w[tok, kk].unsqueeze(-1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    err = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
    mx = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
    r = {"rel_l2": err, "max_rel": mx, "ok": bool(err <= SELFTEST_TOL and torch.isfinite(out).all())}
    if outb is not None:
        # GLM53_MOE_E4M3_ACC=bf16: the served (bf16-accumulator) output against the fp32-accumulator output just checked
        eb = float((outb.double() - out.double()).norm() / out.double().norm().clamp_min(1e-30))
        r["acc_bf16_rel_l2"] = eb
        r["ok"] = bool(r["ok"] and outb.dtype == torch.bfloat16 and eb <= ACC_SELFTEST_TOL and torch.isfinite(outb).all())
    return r


# ---------------------------------------------------------------------------------------------------------
# the production hook

def _moeglue_join() -> None:
    """This wrapper is outside moeglue's (installed after it) and does not call it for the calls it serves; moeglue's
    warm (GLM53_DEC_MOEGLUE_WARM) forks only for decode-sized calls, but if one were pending, join it here exactly as
    moeglue's own wrapper would at the end of apply_exl3_experts."""
    import sys

    mg = sys.modules.get("glm53_moeglue")
    warm = getattr(mg, "WARM", None) if mg is not None else None
    if warm is not None and any(getattr(warm, "pending", {}).values()):
        mg.warm_join()


def check_layer(prodmod, layer, limit: float) -> bool:
    """The per-layer gate, once: shape / format checks, then the self-test against the spec. Sets and returns
    layer._glm53_moe_e4m3_ok; counts the outcome (STATS, the summary line). A Python-level failure keeps the layer on
    production's path; a device-side fault (illegal address, the fused kernel's watchdog trap) is a STICKY CUDA error
    that no fallback can catch - the process (both ranks, at TP=2) dies, which is why this runs at load time."""
    ok = getattr(layer, "_glm53_moe_e4m3_ok", None)
    if ok is not None:
        return ok
    if not layer_selected(layer):
        STATS["layers_unselected"] += 1
        layer._glm53_moe_e4m3_ok = False           # GLM53_MOE_E4M3_LAYERS: production's path, no self-test
        return False
    STATS["layers"] += 1
    why = layer_reason(layer)
    if why is not None:
        ok = False
        _log.warning("glm53_moe_e4m3: layer not served (%s); production path kept", why)
    else:
        try:
            STATS["selftests"] += 1
            r = selftest(prodmod, layer, float(limit))
            ok = r["ok"]
            if ok and "selftest" not in _LOGGED:
                _LOGGED.add("selftest")
                _log.info("glm53_moe_e4m3 layer self-test passed (rel-L2 %.2e vs the spec, max %.2e%s)",
                          r["rel_l2"], r["max_rel"],
                          f"; bf16 accumulator {r['acc_bf16_rel_l2']:.2e} vs fp32" if "acc_bf16_rel_l2" in r else "")
            if not ok:
                why = "self-test FAILED"
                STATS["selftest_failed"] += 1
                _log.warning("glm53_moe_e4m3 layer self-test FAILED (%s): production path kept", r)
        except Exception as exc:  # noqa: BLE001
            ok = False
            why = "self-test raised"
            STATS["selftest_raised"] += 1
            _log.warning("glm53_moe_e4m3 layer self-test raised %r: production path kept", exc)
    if ok:
        STATS["layers_ok"] += 1
    else:
        STATS["layers_fallback"] += 1
        _FALLBACK_WHY[why] = _FALLBACK_WHY.get(why, 0) + 1
    layer._glm53_moe_e4m3_ok = ok
    return ok


def decode_bound(environ=None, cap: int | None = None) -> tuple[int, int, int]:
    """(MAX_NUM_SEQS x (K+1), K, cap) from the serving env (start.sh forwards MAX_NUM_SEQS, SPEC_METHOD,
    DFLASH_TOKENS / MTP_TOKENS, EXL3_TEMP_ROWS_FUSED to both containers): the largest pure decode/verify step."""
    env = os.environ if environ is None else environ

    def num(name, default):
        try:
            return int(str(env.get(name, "") or default).split()[0])
        except ValueError:
            return default
    seqs = num("MAX_NUM_SEQS", 4)
    meth = (env.get("SPEC_METHOD") or "").strip().lower()
    k = num("DFLASH_TOKENS", 7) if meth == "dflash" else num("MTP_TOKENS", 0) if meth == "mtp" else 0
    if cap is None:
        cap = num("EXL3_TEMP_ROWS_FUSED", TEMP_ROWS_FALLBACK)
    return seqs * (k + 1), k, cap


def log_summary(cap: int | None = None) -> str:
    """The per-process summary (boot_checks greps it): layers served / fell back, and the decode-bound check."""
    why = ", ".join(f"{k} x{v}" for k, v in sorted(_FALLBACK_WHY.items())) or "none"
    bound, k, cap = decode_bound(cap=cap)
    line = (f"glm53_moe_e4m3 summary: {STATS['layers_ok']}/{STATS['layers']} layers served, "
            f"{STATS['layers_fallback']} fell back ({why}); self-test FAILED {STATS['selftest_failed']}, raised "
            f"{STATS['selftest_raised']}; decode bound MAX_NUM_SEQS x (K+1) = {bound} (K={k}) vs fused cap {cap}")
    if LAYERS["sel"] is not None:
        line += (f"; {ENV_LAYERS}: {len(LAYERS['sel'])} indices selected, {STATS['layers_unselected']} layers "
                 f"unselected (production's path)")
    if ACC["bf16"]:
        line += f"; {ENV_ACC}=bf16 (bf16 accumulator)"
    if FOLD["on"]:
        line += f"; {ENV_FOLD}=1 (routed sum accumulated into the shared experts' output)"
    line += (f"; token gather {STATS['tg_layers']}/{STATS['layers_ok']} served layers" if TG["on"]
             else f"; {ENV_TG}=0 (per-pair gather)")
    if MS["on"]:
        line += f"; {ENV_MS}=1 (lean mainloop, fused variants + 8192)"
    key = (STATS["layers"], STATS["layers_ok"], cap, STATS["layers_unselected"])
    if _SUMMARY["logged"] != key:
        _SUMMARY["logged"] = key
        (_log.warning if STATS["layers_fallback"] else _log.info)("%s", line)
        if bound > cap:
            STATS["decode_bound_warn"] += 1
            _log.warning("glm53_moe_e4m3 DECODE BOUND EXCEEDED: MAX_NUM_SEQS x (K+1) = %d > fused cap %d - pure "
                         "decode/verify steps can exceed the cap and would run on the e4m3 path (raise "
                         "EXL3_TEMP_ROWS_FUSED or lower MAX_NUM_SEQS / K)", bound, cap)
    return line


def _serve(prodmod, orig, x, topk_ids, topk_weights, layer, limit, fused):
    STATS["calls"] += 1
    tokens = int(x.shape[-2])
    cap = _cap(layer)
    if _SUMMARY["logged"] is None or _SUMMARY["logged"][0] != STATS["layers"]:
        log_summary(cap)                       # first call (vLLM's profile run): the load-time self-tests are done
    if (fused is False or tokens <= cap or torch.cuda.is_current_stream_capturing()
            or int(x.shape[-1]) != HIDDEN):
        STATS["passed"] += 1
        return orig(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
    if not check_layer(prodmod, layer, limit):  # normally decided at load time (_hook_load); lazily otherwise
        STATS["passed"] += 1
        return orig(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
    x2d = x.reshape(tokens, HIDDEN)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = prodmod.pin_exl3_expert_map(layer, x2d.device)
    fb = fold_buffer(_FOLD_CTX["se"], tokens, x2d) if (FOLD["on"] and not _FOLD_CTX["pending"]) else None
    out = run(prodmod, x2d, ids, weights, layer, float(limit), expert_map=expert_map, fold_into=fb)
    if fb is not None and out is fb:
        _FOLD_CTX["pending"] = True      # moe_runner._unpack drops the shared half: out = shared + routed
        STATS["folded"] += 1
        if "fold" not in _LOGGED:
            _LOGGED.add("fold")
            _log.info("glm53_moe_e4m3 fold: first served call folded (%d tokens): routed sum accumulated into the "
                      "shared experts' output, vLLM's add skipped", tokens)
    elif FOLD["on"] and "nofold" not in _LOGGED:
        # opt-moe-rev: a served call that did not fold although GLM53_MOE_E4M3_FOLD_SHARED=1 is active (the speedup
        # is silently absent otherwise: the counters are in-process only). Correct either way (vLLM adds as before).
        _LOGGED.add("nofold")
        why = ("no foldable shared-expert output (runner not eligible, aux-stream order, or shape/dtype)" if fb is None
               else "the bf16 accumulator did not apply to this call")
        _log.warning("glm53_moe_e4m3 fold: a served call (%d tokens) was NOT folded: %s; vLLM adds as before",
                     tokens, why)
    _moeglue_join()
    STATS["served"] += 1
    STATS["tokens"] += tokens
    if "active" not in _LOGGED:
        _LOGGED.add("active")
        _log.info("glm53_moe_e4m3 active: first prefill routed-MoE call with %d tokens ran on the e4m3 kernels",
                  tokens)
    layer._exl3_last_apply = "e4m3"
    return out.to(dtype=x.dtype)


def _hook_load(prodmod) -> bool:
    """Wrap Exl3MoEMethod.process_weights_after_loading: each EXL3 MoE layer is checked (shapes + self-test) right
    after production built its pointer tables, i.e. during model load - before vLLM's profile run and long before the
    engine reports ready, whatever the warmup does. False if the class / method is missing (then lazily at the first
    eligible call: vLLM's profile run executes every MoE layer with max_num_batched_tokens tokens)."""
    cls = getattr(prodmod, "Exl3MoEMethod", None)
    fn = getattr(cls, "process_weights_after_loading", None) if cls is not None else None
    if fn is None:
        return False
    if getattr(fn, "_glm53_moe_e4m3", False):
        return True
    limit = float(getattr(prodmod, "SWIGLU_LIMIT_DEFAULT", 10.0))

    def process_weights_after_loading(self, layer, *a, **k):
        r = fn(self, layer, *a, **k)
        if getattr(layer, "_exl3_inners", None):
            check_layer(prodmod, layer, limit)
        return r

    process_weights_after_loading.__doc__ = fn.__doc__
    process_weights_after_loading._glm53_moe_e4m3 = True
    process_weights_after_loading._glm53_moe_e4m3_orig = fn
    cls.process_weights_after_loading = process_weights_after_loading
    return True


def install(prodmod=None, environ=None, load_selftest: bool = True) -> dict:
    """Wrap prodmod.apply_exl3_experts. Returns a report; installs nothing when the knob is off."""
    try:
        on = enabled(environ)
        d16 = down_mode(environ) if on else False
        sel = layers_mode(environ) if on else None
        dsel = down_layers_mode(environ) if on else None
        accb = acc_mode(environ) if on else False
        tgm = tg_mode(environ) if on else True
        fold = fold_mode(environ) if on else False
        msm = ms_mode(environ) if on else False
        if fold and not accb:
            raise ValueError(f"{ENV_FOLD}=1 requires {ENV_ACC}=bf16")
    except ValueError as exc:
        _log.warning("glm53_moe_e4m3 not installed: %s (production kernels unchanged)", exc)
        return {"installed": False, "reason": str(exc)}
    if not on:
        return {"installed": False, "reason": "off"}
    if prodmod is None:
        import importlib

        prodmod = importlib.import_module(PROD_MODULE)
    orig = getattr(prodmod, "apply_exl3_experts", None)
    if orig is None:
        _log.warning("glm53_moe_e4m3 not installed: production module has no apply_exl3_experts")
        return {"installed": False, "reason": "no apply_exl3_experts"}
    if getattr(orig, "_glm53_moe_e4m3", False):
        return {"installed": True, "reason": "already installed"}
    try:
        e = _ext()
        assert int(e.version()) == 1
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moe_e4m3 not installed: extension glm53_moe_e4m3_ext unavailable (%r)", exc)
        return {"installed": False, "reason": f"extension: {exc!r}"}
    for need in ("pin_exl3_expert_map", "map_topk_to_local"):
        if not hasattr(prodmod, need):
            _log.warning("glm53_moe_e4m3 not installed: production module has no %s", need)
            return {"installed": False, "reason": f"no {need}"}

    DOWN16["on"] = d16
    DOWN16["layers"] = dsel
    LAYERS["sel"] = sel
    ACC["bf16"] = accb
    TG["on"] = tgm
    MS["on"] = msm
    fold_why = _install_fold() if fold else "off"
    FOLD["on"] = fold and fold_why == "ok"
    if fold and not FOLD["on"]:
        _log.warning("glm53_moe_e4m3: %s=1 NOT active (%s); the routed output is added by vLLM as before",
                     ENV_FOLD, fold_why)

    def apply_exl3_experts(x, topk_ids, topk_weights, layer, *, limit=10.0, fused=None):
        return _serve(prodmod, orig, x, topk_ids, topk_weights, layer, limit, fused)

    apply_exl3_experts.__doc__ = getattr(orig, "__doc__", None)
    apply_exl3_experts._glm53_moe_e4m3 = True
    apply_exl3_experts._glm53_moe_e4m3_orig = orig
    prodmod.apply_exl3_experts = apply_exl3_experts
    load_hook = _hook_load(prodmod) if load_selftest else False
    STATS["installed"] = True
    bound, k, cap = decode_bound(environ)
    when = "per-layer self-test at load time" if load_hook else "per-layer self-test at the first eligible call"
    _log.info("glm53_moe_e4m3 installed: GLM53_MOE_E4M3=1, prefill routed MoE (tokens > fused cap) on the e4m3 "
              "kernels (down projection: %s%s); %s (pid %d). NOTE: every apply call with more tokens than the fused cap is served - with "
              "GLM53_MIXED_PREFILL_CHUNK=0 that includes decode/verify tokens batched into a prefill step",
              "fp16 (GLM53_MOE_E4M3_DOWN=f16)" if d16 else "e4m3",
              ("; accumulator: bf16 (GLM53_MOE_E4M3_ACC=bf16)" if accb else "")
              + ("; routed sum folded into the shared experts' output (GLM53_MOE_E4M3_FOLD_SHARED=1)"
                 if FOLD["on"] else "")
              + ("; mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)" if msm else ""), when, os.getpid())
    if bound > cap:
        STATS["decode_bound_warn"] += 1
        _log.warning("glm53_moe_e4m3 DECODE BOUND EXCEEDED: MAX_NUM_SEQS x (K+1) = %d > EXL3_TEMP_ROWS_FUSED %d - "
                     "pure decode/verify steps can exceed the fused cap and would run on the e4m3 path", bound, cap)
    if sel is not None:
        _log.info("glm53_moe_e4m3: %s selects %d layer indices (%s); every other MoE layer stays on production's path",
                  ENV_LAYERS, len(sel), ",".join(str(v) for v in sorted(sel)))
    return {"installed": True, "reason": "ok", "load_hook": load_hook, "down16": d16,
            "layers": None if sel is None else sorted(sel), "down_layers": None if dsel is None else sorted(dsel),
            "acc_bf16": accb, "fold_shared": FOLD["on"], "tok_gather": tgm, "mainloop": int(msm)}


def uninstall(prodmod=None) -> None:
    import importlib

    prodmod = prodmod or importlib.import_module(PROD_MODULE)
    fn = getattr(prodmod, "apply_exl3_experts", None)
    while fn is not None and getattr(fn, "_glm53_moe_e4m3", False):
        prodmod.apply_exl3_experts = fn._glm53_moe_e4m3_orig
        fn = prodmod.apply_exl3_experts
    cls = getattr(prodmod, "Exl3MoEMethod", None)
    pw = getattr(cls, "process_weights_after_loading", None) if cls is not None else None
    while pw is not None and getattr(pw, "_glm53_moe_e4m3", False):
        cls.process_weights_after_loading = pw._glm53_moe_e4m3_orig
        pw = cls.process_weights_after_loading
    STATS["installed"] = False
    DOWN16["on"] = False
    DOWN16["layers"] = None
    LAYERS["sel"] = None
    ACC["bf16"] = False
    TG["on"] = True
    MS["on"] = False
    FOLD["on"] = False
    _uninstall_fold()


def plugin_install() -> None:
    """Called LAST by integrate.plugin_register (armed there by overlay/patch_moe_e4m3.py): the wrapper is the outermost
    one. Never raises into vLLM's plugin loader."""
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moe_e4m3 not installed (production kernels unchanged): %r", exc)
