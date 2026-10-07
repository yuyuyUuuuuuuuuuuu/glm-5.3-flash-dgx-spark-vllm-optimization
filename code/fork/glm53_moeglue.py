"""moeglue — production's decode MoE apply in five launches instead of eleven (docs/DEC_MOEGLUE.md).

Enable: ``GLM53_DEC_MOEGLUE`` in {1, on, true, yes} (read once, at plugin load; default off = production unchanged).
Requires TF_EXL3_MOE=1 with its K2 apply path (the layers' TF registration, scratch and self-tests are reused).
Optional ``GLM53_DEC_MOEGLUE_PREFETCH`` (default 1): glue_prep also prefetches each active expert's epilogue vectors
(svh gate/up, suh down, svh down; 14 KiB per expert) into L2.

What production runs per decode MoE layer on the main stream after the router (launcher overlay exl3.py
Exl3MoEMethod.apply -> apply_exl3_experts -> apply_exl3_fused_moe, K2-hooked):
    topk_ids.to(torch.long) copy | torch.zeros(out) | route_ids | rot_in | grouped g/u | gateup_epilogue | grouped d |
    down_epilogue (fp32 atomics) | out.to(x.dtype)
With moeglue (apply_exl3_experts wrapped; the int32 router ids go straight in):
    glue_prep | grouped g/u | gateup_epilogue | grouped d | glue_finish
glue_prep = map_topk_to_local + a stable sort position per pair + the segment table + rot_in (+ L2 prefetch);
glue_finish = down_epilogue per pair + the per-token sum in slot order + the cast to x.dtype. Everything up to each
pair's fp32 contribution is bit-identical to K2; the per-token sum of the topk contributions is taken in slot order
(deterministic) instead of K2's atomics in arrival order (not deterministic run to run). One valid route per token ->
bit-identical output (the per-layer self-test checks it on the layer's real weights).

Served: exactly the calls production would send to its fused apply (fused None + EXL3_FUSED_MOE on, or fused True)
while TF's dispatcher and K2 apply hook are installed and K2 would serve the call (tf_exl3_moe.plan_apply's rules,
ids int32 or int64), x bf16 or fp16, on a layer whose moeglue self-test passed. Anything else -> production's own
apply_exl3_experts with exactly the arguments received (which then runs K2 / the dispatcher as before).
The decision depends only on shapes, dtypes and load-time state (identical on both TP ranks: the self-test runs on each
rank's own shard; a failing layer falls back on that rank only, which changes no collective).
Revert: unset GLM53_DEC_MOEGLUE on both ranks and restart (captured graphs keep what they captured).

warm (``GLM53_DEC_MOEGLUE_WARM`` in {1, on, true, yes}; independent of GLM53_DEC_MOEGLUE, default off): right after the
o_proj GEMV of every decoder layer whose MLP is a MoE, a side stream reads that layer's next small weights into the L2
(GLM53_DEC_MOEGLUE_WARM_SET letters, in this order: f = hc_ffn_fn (the fused mHC post/pre of the MoE sublayer), r =
router gate, g = shared expert gate_up, d = shared expert down; default "frgd", at most GLM53_DEC_MOEGLUE_WARM_MIB MiB,
default 16) while the o_proj all-reduce and the mHC kernels (which read almost nothing from DRAM) run on the main
stream. The fork is inside the o_proj's quant-method apply (a custom op under torch.compile, between the GEMV and the
all-reduce), the join is at the end of that layer's apply_exl3_experts (same CUDA-graph segment: no attention op in
between). Nothing is written; every output is bit-identical. Only calls with <= GLM53_DEC_MOEGLUE_WARM_MAX_M tokens
(default 64) fork. GLM53_DEC_MOEGLUE_WARM_BLOCKS (default 16) sets the warm kernel's grid (x 256 threads, 8 loads in
flight each): few blocks leave the SMs to the mHC kernels.
Breakable CUDA graphs (vLLM VLLM_USE_BREAKABLE_CUDAGRAPH=1, auto-enabled for Glm5Next*: production's PIECEWISE graphs
are chains of separate CUDA graphs cut at every @eager_break_during_capture op): a fork must rejoin before its graph
segment ends. The o_proj -> all-reduce -> mHC -> router -> MoE window has no eager break (tests/r16/
test_r16_breakable.py), so the warm forks and joins in one segment as designed; as a guard, BreakableCUDAGraphCapture.
add_eager / _end_segment are wrapped so that a warm still pending when a segment ends is joined into it first (at a
break: WARNING once, and the warm no longer forks inside breakable segments; at the end of a capture, normal or raising:
silent), i.e. a capture can neither fail on an unjoined warm nor leave a stale fork behind.
"""
from __future__ import annotations

import logging
import os
import types

import torch

ENV = "GLM53_DEC_MOEGLUE"
ENV_PREFETCH = "GLM53_DEC_MOEGLUE_PREFETCH"
ENV_WARM = "GLM53_DEC_MOEGLUE_WARM"
ENV_WARM_SET = "GLM53_DEC_MOEGLUE_WARM_SET"
ENV_WARM_MIB = "GLM53_DEC_MOEGLUE_WARM_MIB"
ENV_WARM_BLOCKS = "GLM53_DEC_MOEGLUE_WARM_BLOCKS"
ENV_WARM_MAX_M = "GLM53_DEC_MOEGLUE_WARM_MAX_M"
WARM_VERSION = 1
WARM_MAX_REGIONS = 6                  # kernels/exl3.cu WARM_MAX_REGIONS
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
_TRUE = frozenset({"1", "on", "true", "yes"})
_log = logging.getLogger("vllm.glm53_moeglue")

# The production functions this wrapper re-implements the decode branch of (ast fingerprints, integrate.
# source_fingerprint): apply_exl3_experts (x2d / ids / weights reshapes, pin_exl3_expert_map, use_fused rule, the
# fused call, _exl3_last_apply, out.to(x.dtype)), pin_exl3_expert_map (called as is) and Exl3MoEMethod.apply (calls
# apply_exl3_experts by module-global name with limit=float(swiglu_limit or default)). Verified: the image's
# quantization/exl3.py and the launcher overlay exl3.py df864b5 (both identical in these three functions).
VERIFIED = {
    "apply_exl3_experts": frozenset({"9cfe8d06cda3bb15"}),
    "pin_exl3_expert_map": frozenset({"a451bf90a4558e30"}),
    "Exl3MoEMethod.apply": frozenset({"02bec38995a4dcc2"}),
}


# warm: the vLLM functions whose order the fork / join relies on (o_proj GEMV -> all-reduce -> hc_fused_post_pre(
# hc_ffn_fn) -> mlp(gate -> experts) inside one decoder layer, no attention op between the o_proj and the MoE apply).
# Verified on the image's vLLM (production's patches do not touch them).
WARM_VERIFIED = {
    "RowParallelLinear.forward": frozenset({"0d071001b54bbb4b"}),
    # f543fb24079672d5 = 9c0fe21938cdc177 as recompiled by GLM53_PREFILL_QUICKWINS item mhc_mean (glm53_prefill_quickwins
    # MHC_FINAL: only the last layer's final hc_post + hc_contract, after the MLP, becomes _glm53_qw_final_post, which
    # runs production's statements below MIN_T tokens). The o_proj -> all-reduce -> hc_fused_post_pre(hc_ffn_fn) -> mlp
    # order the fork / join relies on is unchanged (deploy-r16: without it, quickwins=all disarmed warm with a WARNING).
    "Glm5NextDecoderLayer.forward": frozenset({"9c0fe21938cdc177", "f543fb24079672d5"}),
    "Glm5NextMoE.forward": frozenset({"5492b934207c3148"}),
}


class _Cfg:
    def __init__(self) -> None:
        self.prefetch = True
        self.strict = False           # tests: raise instead of falling back
        self.warm_set = "frgd"
        self.warm_mib = 16.0
        self.warm_blocks = 16
        self.warm_unroll = 8
        self.warm_max_m = 64


class _State:
    def __init__(self) -> None:
        self.enabled = False
        self.reason = "not installed"
        self.prodmod = None
        self.ext = None               # exllamav3_ext module (production's)
        self.tfext = None             # tf_exl3_moe_ext (moe_forward_glue, l2_warm)
        self.hooked = False           # apply_exl3_experts wrapped (glue and / or the warm join)
        self.warm = False
        self.warm_reason = "not installed"
        self.loader_hooked = False
        self.tag = None


CFG = _Cfg()
STATE = _State()
VERDICT: dict[tuple, tuple[bool, str]] = {}      # (device, gate_trellis table ptr) -> (passed, detail)
COUNTERS: dict[str, int] = {}
_LOGGED: set = set()


def _count(k: str) -> None:
    COUNTERS[k] = COUNTERS.get(k, 0) + 1


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def env_enabled(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


def env_prefetch(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV_PREFETCH)
    return v is None or not v.strip() or v.strip().lower() in _TRUE


def env_warm(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV_WARM)
    return v is not None and v.strip().lower() in _TRUE


def warm_cfg_from_env(environ=None) -> str | None:
    """Parse the warm knobs into CFG; returns an error string for an invalid value (install refuses then)."""
    env = os.environ if environ is None else environ
    st = env.get(ENV_WARM_SET, "").strip().lower() or "frgd"
    if not st or any(c not in "frgd" for c in st) or len(set(st)) != len(st):
        return f"{ENV_WARM_SET}={st!r}: letters from f, r, g, d, each once"
    try:
        mib = float(env.get(ENV_WARM_MIB, "").strip() or 16)
        blocks = int(env.get(ENV_WARM_BLOCKS, "").strip() or 16)
        max_m = int(env.get(ENV_WARM_MAX_M, "").strip() or 64)
    except ValueError as exc:
        return f"warm knob not a number: {exc}"
    if not 0.25 <= mib <= 20 or not 1 <= blocks <= 1024 or not 1 <= max_m <= 512:
        return f"warm knobs out of range: MIB {mib} (0.25..20), BLOCKS {blocks} (1..1024), MAX_M {max_m} (1..512)"
    CFG.warm_set, CFG.warm_mib, CFG.warm_blocks, CFG.warm_max_m = st, mib, blocks, max_m
    return None


def fingerprints(prodmod) -> dict:
    import integrate

    out = {}
    for name in VERIFIED:
        obj = prodmod
        for part in name.split("."):
            obj = getattr(obj, part, None)
        fn = getattr(obj, "_glm53_moeglue_orig", obj)
        out[name] = integrate.source_fingerprint(fn) if callable(fn) else None
    return out


# ---------------------------------------------------------------------------------------------------------------
# per-layer self-test (tf_exl3_moe.POST_REGISTER_HOOKS: runs at load, eagerly, on the layer's real tables)

def _fake_layer(tables, temps):
    import tf_exl3_moe as tf

    return types.SimpleNamespace(_exl3_ptrs=dict(zip(tf._PTR_KEYS, tables)), _exl3_fused_temps=temps, _exl3_k=4)


def selftest_layer(info, tables, temps, limit) -> tuple[bool, str]:
    """moeglue vs K2 (itself checked against production's exl3_moe just before, in tf_exl3_moe._register) on this
    layer's tables: bf16 x, fp32 router weights, B in {1, min(8, R)}:
      (a) one route per token, ids int32 and int64: bit-identical bf16 output (== K2's out.to(bf16));
      (b) topk routes per token incl. one sentinel id: rel_l2 vs K2's out.to(bf16) <= 1e-3 (only the fp32 summation
          order of a token's contributions differs), and bit-identical between two runs (deterministic)."""
    import tf_exl3_moe as tf

    dev = torch.device("cuda", info.device)
    g = torch.Generator(device="cpu").manual_seed(20260928)
    n, R = info.n, int(temps[0].shape[1])
    layer = _fake_layer(tables, temps)
    inners = [None] * n
    topk, Bs = tf.selftest_shapes(info, R)
    worst, cases = 0.0, 0
    for B in Bs:
        for k in (1, topk):
            if B * k > min(info.P_cap, tf._ROUTE_IDS_MAX_PAIRS):
                continue
            x = torch.randn(B, info.K, generator=g).to(torch.bfloat16).to(dev)
            ids = torch.stack([torch.randperm(n, generator=g)[:k] for _ in range(B)]).to(dev)
            if B > 1:
                ids[B - 1, k - 1] = -1
            w = (torch.softmax(torch.randn(B, k, generator=g), -1) * 2.5).to(dev)
            ref = tf.apply_fused(x, ids, w, layer, inners, None, float(limit))
            if ref is None:
                return False, f"K2 did not serve the self-test call B={B} topk={k}"
            ref = ref.to(torch.bfloat16)
            for idt in (torch.int32, torch.int64):
                o1 = tf.apply_glue(x, ids.to(idt).contiguous(), w, layer, inners, None, float(limit), CFG.prefetch)
                o2 = tf.apply_glue(x, ids.to(idt).contiguous(), w, layer, inners, None, float(limit), CFG.prefetch)
                if o1 is None or o2 is None:
                    return False, f"moeglue did not serve the self-test call B={B} topk={k} ids {idt}"
                torch.cuda.synchronize(dev)
                cases += 1
                if not torch.equal(o1, o2):
                    return False, f"not deterministic B={B} topk={k} ids {idt}"
                if k == 1:
                    if not torch.equal(o1, ref):
                        return False, f"one route per token not bit-identical to K2 (B={B}, ids {idt})"
                else:
                    d = (o1.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
                    fin = bool(torch.equal(torch.isfinite(o1), torch.isfinite(ref)))
                    worst = max(worst, d)
                    if not fin or not d <= 1e-3:
                        return False, f"topk {k} B={B} ids {idt}: rel_l2 {d:.2e} vs K2, finite pattern equal {fin}"
    return True, f"{cases} cases, one route per token bitwise, topk routes rel_l2 max {worst:.1e} vs K2"


def _post_register(info, tables, temps, limit) -> None:
    key = (info.device, tables[0].data_ptr())
    try:
        ok, detail = selftest_layer(info, tables, temps, limit)
    except Exception as exc:  # noqa: BLE001 - a self-test that cannot run is a failed self-test
        if CFG.strict:
            raise
        ok, detail = False, f"self-test raised {type(exc).__name__}: {str(exc)[:200]}"
    VERDICT[key] = (ok, detail)
    _count("selftest_passed" if ok else "selftest_failed")
    if ok:
        _log_once("st_ok", logging.INFO, "glm53_moeglue: layer self-test passed (%s); further layers are counted",
                  detail)
    else:
        _log.warning("glm53_moeglue: layer self-test FAILED, this layer stays on the K2 path: %s", detail)


# ---------------------------------------------------------------------------------------------------------------
# the serving path

def try_glue(x, topk_ids, topk_weights, layer, limit, fused):
    """The routed output production's apply_exl3_experts would return, or None (then production's own runs)."""
    import tf_exl3_moe as tf

    prod = STATE.prodmod
    use_fused = prod.fused_moe_enabled() if fused is None else bool(fused)
    if not use_fused:
        _count("deleg_not_fused")
        return None
    inners = getattr(layer, "_exl3_inners", None)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    if not inners or not ptrs:
        _count("deleg_no_state")
        return None
    ext = STATE.ext
    if not getattr(getattr(ext, "exl3_moe", None), "_tf_exl3_dispatch", False) or \
            not getattr(getattr(prod, "apply_exl3_fused_moe", None), "_tf_exl3_apply_hook", False):
        _count("deleg_tf_not_installed")
        return None
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dim() < 2:
        _count("deleg_x")
        return None
    gt = ptrs.get("gate_trellis")
    v = VERDICT.get((x.get_device(), gt.data_ptr() if isinstance(gt, torch.Tensor) else None))
    if v is None or not v[0]:
        _count("deleg_untested" if v is None else "deleg_selftest_failed")
        if v is None:
            _log_once("untested", logging.WARNING, "glm53_moeglue: a layer without a moeglue self-test was called "
                      "(registered before install?): production's apply serves it (counted, not repeated)")
        return None
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = prod.pin_exl3_expert_map(layer, x2d.device)
    why: list = []
    out = tf.apply_glue(x2d, ids, weights, layer, inners, expert_map, limit, CFG.prefetch, why)
    if out is None:
        reason = why[0] if why else "unknown"
        cat = reason.split(":", 1)[0]
        _count("deleg_" + cat)
        if cat in ("contract", "tf_error", "unregistered"):
            _log_once("deleg:" + reason[:80], logging.WARNING, "glm53_moeglue: a call (B=%s) was handed to "
                      "production's apply: %s (counted, not repeated)", tokens, reason)
        return None
    layer._exl3_last_apply = "fused"
    _count("served_captured" if torch.cuda.is_current_stream_capturing() else "served_eager")
    return out


def _make_hook(orig):
    default_limit = orig.__kwdefaults__.get("limit") if getattr(orig, "__kwdefaults__", None) else 10.0

    def apply_exl3_experts(x, topk_ids, topk_weights, layer, *, limit=default_limit, fused=None):
        try:
            if STATE.enabled:
                try:
                    out = try_glue(x, topk_ids, topk_weights, layer, float(limit), fused)
                except Exception as exc:
                    if CFG.strict or not isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError, KeyError,
                                                          AttributeError)) or "CUDA" in str(exc) or \
                            "capture" in str(exc):
                        raise
                    _count("deleg_error")
                    _log_once("err:" + repr(exc)[:80], logging.WARNING, "glm53_moeglue: %r; production's apply "
                              "serves this call (counted, not repeated)", exc)
                    out = None
                if out is not None:
                    return out
            return orig(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
        finally:
            if WARM.pending:                           # warm: join the side stream forked at this layer's o_proj
                warm_join()

    apply_exl3_experts.__doc__ = orig.__doc__
    apply_exl3_experts._glm53_moeglue_orig = orig
    return apply_exl3_experts


# ---------------------------------------------------------------------------------------------------------------
# warm (GLM53_DEC_MOEGLUE_WARM): L2 warm-up of the MoE sublayer's small weights during the o_proj all-reduce + mHC

class _WarmState:
    def __init__(self) -> None:
        self.handles: list = []       # handle -> _WarmHandle
        self.side: dict = {}          # device index -> torch.cuda.Stream
        self.sink: dict = {}          # device index -> int32[4] (never written in practice)
        self.pending: dict = {}       # device index -> "cap" / "eager" (capture state of the fork) while not joined
        self.bk_skip = False          # a warm was joined at a breakable-CUDA-graph break: no fork inside segments
        self.bk_guard = None          # "hooked" | "absent" (no breakable capture in this vLLM) | refusal reason


class _WarmHandle:
    __slots__ = ("qm", "layer", "base", "regions", "nbytes", "n", "label")

    def __init__(self, qm, layer, base, regions, n, label):
        self.qm, self.layer, self.base, self.regions, self.n, self.label = qm, layer, base, regions, n, label
        self.nbytes = sum(t.numel() * t.element_size() for t in regions)


WARM = _WarmState()
_WRAPPED_QM: dict = {}


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _stale(p, cap: bool) -> bool:
    """A pending fork made in the other capture state (an eager fork seen during a capture, or the reverse) cannot be
    joined: waiting on an event recorded outside the capture invalidates it (cudaErrorStreamCaptureInvalidated)."""
    return (p == "cap") != cap


def warm_join() -> None:
    """Main stream waits for every forked warm (end of the MoE apply). Cheap: the warm finished long before.
    A fork made in the other capture state is dropped instead (it only read weights; nothing depends on it)."""
    cap = None
    for d, p in list(WARM.pending.items()):
        if p:
            cap = _capturing() if cap is None else cap
            if _stale(p, cap):
                _count("warm_stale_dropped")
                _log_once("warm_stale", logging.WARNING, "glm53_moeglue warm: a fork made %s was still pending %s; "
                          "dropped without a join (counted, not repeated)", "eagerly" if p != "cap" else
                          "under capture", "during a capture" if cap else "eagerly")
            else:
                torch.cuda.current_stream(d).wait_stream(WARM.side[d])
                _count("warm_joined")
            WARM.pending[d] = False


def _warm_fork(h: _WarmHandle, x: torch.Tensor) -> None:
    """After the o_proj GEMV was enqueued: side stream waits for it, then reads h.regions into the L2."""
    if not STATE.warm:
        return
    if WARM.bk_skip and _bk_active():
        _count("warm_bk_skipped")
        return
    k = x.shape[-1] if x.dim() else 1
    M = x.numel() // max(1, k)
    if M == 0 or M > CFG.warm_max_m:
        _count("warm_skip_m")
        return
    d = x.device.index
    side = WARM.side.get(d)
    if side is None:
        _count("warm_no_stream")
        return
    cur = torch.cuda.current_stream(x.device)
    cap = _capturing()
    p = WARM.pending.get(d)
    if p:                                              # never in the model's order: join the stale fork first
        if _stale(p, cap):                             # e.g. an eager fork whose MoE never ran, then a capture
            _count("warm_stale_dropped")
            _log_once("warm_stale", logging.WARNING, "glm53_moeglue warm: a fork made %s was still pending %s; "
                      "dropped without a join (counted, not repeated)", "eagerly" if p != "cap" else
                      "under capture", "during a capture" if cap else "eagerly")
        else:
            cur.wait_stream(side)
            _count("warm_unjoined")
            _log_once("warm_unjoined", logging.WARNING, "glm53_moeglue warm: an o_proj forked again before its MoE "
                      "joined the previous fork (joined now; counted, not repeated)")
        WARM.pending[d] = False
    side.wait_stream(cur)
    WARM.pending[d] = "cap" if cap else "eager"        # truthy while not joined; the capture state it was made in
    try:
        with torch.cuda.stream(side):
            STATE.tfext.l2_warm(h.regions, CFG.warm_blocks, CFG.warm_unroll, WARM.sink[d])
    except Exception as exc:  # noqa: BLE001 - join at once: nothing may stay forked
        cur.wait_stream(side)
        WARM.pending[d] = False
        if CFG.strict or "CUDA" in str(exc):
            raise
        _count("warm_error")
        _log_once("warm_err:" + repr(exc)[:60], logging.WARNING, "glm53_moeglue warm: %r (warm skipped for this "
                  "call; counted, not repeated)", exc)
        return
    _count("warm_captured" if cap else "warm_eager")


def _linear_warm(x: torch.Tensor, handle: int) -> torch.Tensor:
    h = WARM.handles[handle]
    y = h.base.apply(h.qm, h.layer, x, None)          # production's (possibly fp8_gemv-hooked) apply, unchanged
    _warm_fork(h, x)
    return y


@torch.library.custom_op("glm53_moeglue::linear_warm", mutates_args=())
def _linear_warm_op(x: torch.Tensor, weight: torch.Tensor, handle: int, size_n: int) -> torch.Tensor:
    return _linear_warm(x, handle)


@_linear_warm_op.register_fake
def _(x, weight, handle, size_n):
    return x.new_empty(tuple(x.shape[:-1]) + (size_n,))


def _wrap_o_proj(layer, handle: int, n: int) -> bool:
    qm = getattr(layer, "quant_method", None)
    if qm is None or not callable(getattr(type(qm), "apply", None)):
        return False
    base = getattr(type(qm), "_glm53_warm_base", type(qm))
    cls = _WRAPPED_QM.get(base)
    if cls is None:
        def apply(self, layer, x, bias=None):
            h = getattr(layer, "_glm53_warm_handle", None)
            if h is None or bias is not None or not STATE.warm:
                return base.apply(self, layer, x, bias)
            if torch.compiler.is_compiling():         # opaque to dynamo; the fork happens at run / capture time
                return torch.ops.glm53_moeglue.linear_warm(x, layer.weight, h, layer._glm53_warm_n)
            return _linear_warm(x, h)
        cls = type(f"Glm53Warm{base.__name__}", (base,), {"apply": apply, "_glm53_warm_base": base})
        _WRAPPED_QM[base] = cls
    qm.__class__ = cls
    layer._glm53_warm_handle = handle
    layer._glm53_warm_n = int(n)
    return True


def _vllm_fingerprints() -> dict:
    import integrate
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.models.glm5next.nvidia import model as gm

    fns = {"RowParallelLinear.forward": RowParallelLinear.forward,
           "Glm5NextDecoderLayer.forward": gm.Glm5NextDecoderLayer.forward,
           "Glm5NextMoE.forward": gm.Glm5NextMoE.forward}
    return {k: integrate.source_fingerprint(getattr(f, "_glm53_moeglue_orig", f)) for k, f in fns.items()}


def _moe_served_by_hook(moe) -> bool:
    """True when this MoE's routed experts use production's Exl3MoEMethod (whose apply calls the wrapped
    apply_exl3_experts, where the warm join is)."""
    prod = STATE.prodmod
    cls = getattr(prod, "Exl3MoEMethod", None)
    if cls is None:
        return False
    for sub in moe.experts.modules() if isinstance(getattr(moe, "experts", None), torch.nn.Module) else ():
        if isinstance(getattr(sub, "quant_method", None), cls):
            return True
    return False


def _warm_regions(dl, moe, letters: str, budget: int) -> list:
    """The tensors to warm for one MoE sublayer, in `letters` order, cut to `budget` bytes (a contiguous prefix of
    the last one that fits partially)."""
    sh = getattr(moe, "shared_experts", None)
    src = {"f": getattr(dl, "hc_ffn_fn", None),
           "r": getattr(getattr(moe, "gate", None), "weight", None),
           "g": getattr(getattr(sh, "gate_up_proj", None), "weight", None),
           "d": getattr(getattr(sh, "down_proj", None), "weight", None)}
    out, left = [], budget
    for c in letters:
        t = src.get(c)
        if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous() or t.numel() == 0:
            continue
        t = t.detach().reshape(-1)
        mis = (-t.data_ptr()) % 16                     # a view into a larger buffer: warm its 16 B-aligned interior
        if mis % t.element_size():
            continue
        t = t[mis // t.element_size():]
        nb = t.numel() * t.element_size()
        if nb > left:
            t = t[: (left // t.element_size()) // 16 * 16]
            nb = t.numel() * t.element_size()
        if nb >= 16:
            out.append(t)
            left -= nb
        if left < 16:
            break
    return out[:WARM_MAX_REGIONS]


def _tag_compile_cache() -> None:
    tag = f"warm:v{WARM_VERSION}:{CFG.warm_set}:{CFG.warm_mib:g}:{CFG.warm_blocks}:{CFG.warm_unroll}:{CFG.warm_max_m}"
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        ac = cfg.additional_config
        if not isinstance(ac, dict):
            raise TypeError(f"additional_config is {type(ac).__name__}, not a dict")
        ac["glm53_moeglue"] = tag
        STATE.tag = tag
        _log.info("glm53_moeglue warm: compile-cache tag additional_config[glm53_moeglue]=%s (config hash %s)", tag,
                  cfg.compute_hash()[:12])
    except Exception as exc:  # noqa: BLE001
        STATE.tag = None
        _log.warning("glm53_moeglue warm: could not tag the compile cache (%r): a graph compiled without warm could "
                     "be reused (then nothing forks; outputs unchanged)", exc)


def warm_post_load(model) -> dict:
    """Wire the o_proj of every decoder layer with a MoE MLP (after all weights are processed)."""
    out = {"wired": 0, "bytes": 0, "rejected": []}
    if not STATE.warm:
        return out
    fps = _vllm_fingerprints()
    bad = [f"{k} {v}" for k, v in fps.items() if v not in WARM_VERIFIED[k]]
    if bad:
        STATE.warm, STATE.warm_reason = False, "vLLM functions not the verified versions: " + "; ".join(bad)
        _log.warning("glm53_moeglue warm NOT wired: %s; production unchanged", STATE.warm_reason)
        return out
    _tag_compile_cache()
    budget = int(CFG.warm_mib * 2 ** 20)
    for name, dl in model.named_modules():
        if type(dl).__name__ != "Glm5NextDecoderLayer" or not getattr(dl, "_mlp_is_moe", False) or \
                getattr(dl, "is_mtp_layer", False) or not getattr(dl, "mhc", False):
            continue
        op = getattr(getattr(dl, "self_attn", None), "o_proj", None)
        moe = getattr(dl, "mlp", None)
        why = None
        if op is None or type(op).__name__ != "RowParallelLinear":
            why = "no RowParallelLinear self_attn.o_proj"
        elif getattr(op, "bias", None) is not None and not getattr(op, "skip_bias_add", False):
            why = "o_proj has a bias"
        elif not _moe_served_by_hook(moe):
            why = "MoE experts not served by Exl3MoEMethod (no join point)"
        if why is None:
            regions = _warm_regions(dl, moe, CFG.warm_set, budget)
            if not regions:
                why = "no warmable tensor"
            else:
                w = regions[0]
                d = w.device.index
                if d not in WARM.side:
                    WARM.side[d] = torch.cuda.Stream(device=w.device)
                    WARM.sink[d] = torch.zeros(4, dtype=torch.int32, device=w.device)
                n = int(getattr(op, "output_size_per_partition", 0) or 0)
                h = _WarmHandle(op.quant_method, op, getattr(type(op.quant_method), "_glm53_warm_base",
                                                               type(op.quant_method)), regions, n, name)
                WARM.handles.append(h)
                if n <= 0 or not _wrap_o_proj(op, len(WARM.handles) - 1, n):
                    WARM.handles.pop()
                    why = "o_proj quant method not wrappable"
                else:
                    out["wired"] += 1
                    out["bytes"] = h.nbytes
        if why is not None:
            out["rejected"].append(f"{name}: {why}")
            _log.warning("glm53_moeglue warm: %s not wired: %s", name, why)
    _log.info("glm53_moeglue warm wired %d MoE sublayers of %s (%.2f MiB each: %s; %d blocks, forks at <= %d tokens); "
              "%d rejected", out["wired"], type(model).__name__, out["bytes"] / 2 ** 20, CFG.warm_set,
              CFG.warm_blocks, CFG.warm_max_m, len(out["rejected"]))
    return out


def _hook_loader() -> bool:
    if STATE.loader_hooked:
        return True
    import vllm.model_executor.model_loader.base_loader as BL
    orig = BL.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        try:
            warm_post_load(model)
        except Exception as exc:  # noqa: BLE001 - never break model loading; unwired layers stay production's
            _log.warning("glm53_moeglue warm: post-load wiring failed (%r); layers wired before the failure fork "
                         "and join as designed, the rest are production's", exc)

    process_weights_after_loading._glm53_moeglue_orig = orig
    BL.process_weights_after_loading = process_weights_after_loading
    STATE.loader_hooked = True
    return True


# ---------------------------------------------------------------------------------------------------------------
# breakable CUDA graphs (module docstring): no warm outlives the graph segment it was forked in
class _Bk:
    cls = None       # vllm.compilation.breakable_cudagraph.BreakableCUDAGraphCapture once hooked
    fc = None        # (is_forward_context_available, get_forward_context, CUDAGraphMode.FULL)


def _bk_active() -> bool:
    """True inside a breakable-CUDA-graph segment that an eager break can end (vLLM's eager_break_during_capture rule)."""
    cls = _Bk.cls
    if cls is None:
        return False
    capobj = cls.current()
    if capobj is None or not getattr(capobj, "_capturing", False):
        return False
    fc = _Bk.fc
    if fc is not None:
        try:
            if fc[0]() and fc[1]().cudagraph_runtime_mode == fc[2]:
                return False
        except Exception:  # noqa: BLE001 - no usable forward context: the breaks happen
            pass
    return True


def _bk_join(capobj, at_break: bool) -> None:
    """Join a warm forked in the segment that is about to end into it, on the segment's capture stream."""
    s = getattr(capobj, "_glm53_origin", None)
    for d, p in list(WARM.pending.items()):
        if p != "cap":
            continue                                   # not forked in this capture (eager): warm_join's rules apply
        WARM.pending[d] = False
        try:
            (s if s is not None else torch.cuda.current_stream(d)).wait_stream(WARM.side[d])
            _count("warm_joined_at_break" if at_break else "warm_joined_at_capture_end")
        except Exception:  # noqa: BLE001 - the capture is already invalid; capture_end reports the real error
            _count("warm_dropped_at_segment_end")
            continue
        if at_break and not WARM.bk_skip:
            WARM.bk_skip = True
            _log.warning("glm53_moeglue warm: a warm was still pending at a breakable CUDA-graph break (joined there, "
                         "the capture stays valid); the warm is no longer forked inside breakable segments")


def _in_chain(fn, marker: str) -> bool:
    while fn is not None:
        if getattr(fn, marker, False):
            return True
        fn = getattr(fn, "_glm53_seg_orig", None)
    return False


def _bk_hook() -> str:
    """Wrap vLLM's BreakableCUDAGraphCapture.add_eager / _end_segment / _begin_segment (idempotent; the wrappers do
    nothing while no warm is pending). Returns "hooked", "absent" (this vLLM has no breakable capture) or why not."""
    try:
        from vllm.compilation import breakable_cudagraph as bk
    except Exception:  # noqa: BLE001
        return "absent"
    cls = getattr(bk, "BreakableCUDAGraphCapture", None)
    if cls is None:
        return "absent"
    for name in ("current", "add_eager", "_begin_segment", "_end_segment"):
        if not callable(getattr(cls, name, None)):
            return f"BreakableCUDAGraphCapture.{name} missing"
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available
        _Bk.fc = (is_forward_context_available, get_forward_context, CUDAGraphMode.FULL)
    except Exception:  # noqa: BLE001
        _Bk.fc = None
    if not _in_chain(cls.add_eager, "_glm53_warm_seg"):
        orig_add, orig_end, orig_begin = cls.add_eager, cls._end_segment, cls._begin_segment

        def add_eager(self, fn):
            if getattr(self, "_capturing", False) and any(p == "cap" for p in WARM.pending.values()):
                _bk_join(self, True)
            return orig_add(self, fn)

        def _end_segment(self):
            if getattr(self, "_capturing", False) and any(p == "cap" for p in WARM.pending.values()):
                _bk_join(self, False)
            return orig_end(self)

        def _begin_segment(self):
            self._glm53_origin = torch.cuda.current_stream()     # the stream capture_begin captures
            return orig_begin(self)

        for f, o in ((add_eager, orig_add), (_end_segment, orig_end), (_begin_segment, orig_begin)):
            f._glm53_warm_seg = True
            f._glm53_seg_orig = o
            f.__doc__ = o.__doc__
        cls.add_eager, cls._end_segment, cls._begin_segment = add_eager, _end_segment, _begin_segment
    _Bk.cls = cls
    return "hooked"


def _refuse(report, reason):
    report["reason"] = reason
    _log.warning("glm53_moeglue enabled (%s) but NOT installed: %s; production apply unchanged", ENV, reason)
    return report


def install(prodmod=None, *, force: bool | None = None, warm: bool | None = None, hook_loader: bool = True) -> dict:
    """Wrap production's apply_exl3_experts and register the per-layer self-test (glue), and / or prepare the warm
    wiring (post-load). No-op unless enabled (env or force / warm=True). Must run after integrate.install() (TF) and
    before the model's weights are loaded. report["installed"] = glue served; report["warm"] = warm armed."""
    report = {"installed": False, "reason": None, "warm": False, "warm_reason": None}
    glue_on = env_enabled() if force is None else bool(force)
    warm_on = env_warm() if warm is None else bool(warm)
    if not glue_on:
        report["reason"] = f"{ENV} not enabled"
    if not warm_on:
        report["warm_reason"] = f"{ENV_WARM} not enabled"
    if not glue_on and not warm_on:
        return report
    import tf_exl3_moe as tf

    CFG.prefetch = env_prefetch()
    if prodmod is None:
        import importlib
        prodmod = importlib.import_module(PROD_MODULE)
    fps = fingerprints(prodmod)
    report["fingerprints"] = fps
    bad = [f"{k} {v}" for k, v in fps.items() if v not in VERIFIED[k]]
    if bad:
        report["warm_reason"] = "production functions not the verified versions"
        return _refuse(report, "production functions not the verified versions: " + "; ".join(bad))
    for name in ("fused_moe_enabled", "pin_exl3_expert_map", "apply_exl3_fused_moe", "load_exllamav3_ext"):
        if not callable(getattr(prodmod, name, None)):
            report["warm_reason"] = f"production module lacks {name}"
            return _refuse(report, f"production module lacks {name}")
    try:
        E = tf.load_ext()
    except Exception as exc:  # noqa: BLE001
        report["warm_reason"] = f"tf_exl3_moe_ext not loadable: {exc!r}"
        return _refuse(report, f"tf_exl3_moe_ext not loadable: {exc!r}")
    STATE.prodmod, STATE.tfext = prodmod, E
    glue_ok = False
    if glue_on:
        why = None
        try:
            ext = prodmod.load_exllamav3_ext()
        except Exception as exc:  # noqa: BLE001
            ext, why = None, f"exllamav3_ext not loadable: {exc!r}"
        if why is None and not getattr(getattr(ext, "exl3_moe", None), "_tf_exl3_dispatch", False):
            why = "tf_exl3_moe's dispatcher is not installed (TF_EXL3_MOE off or refused)"
        if why is None and not getattr(getattr(prodmod, "apply_exl3_fused_moe", None), "_tf_exl3_apply_hook", False):
            why = "tf_exl3_moe's K2 apply hook is not installed (TF_EXL3_APPLY=0 or not verified)"
        if why is None and not hasattr(E, "moe_forward_glue"):
            why = "tf_exl3_moe_ext has no moe_forward_glue (extension older than this module)"
        if why is not None:
            _refuse(report, why)
        else:
            STATE.ext = ext
            glue_ok = True
    warm_ok = False
    if warm_on:
        why = warm_cfg_from_env()
        if why is None and not hasattr(E, "l2_warm"):
            why = "tf_exl3_moe_ext has no l2_warm (extension older than this module)"
        if why is None:
            WARM.bk_guard = _bk_hook()
            if WARM.bk_guard not in ("hooked", "absent"):
                why = f"breakable CUDA graphs cannot be guarded ({WARM.bk_guard})"
        if why is None and hook_loader:
            try:
                _hook_loader()
            except Exception as exc:  # noqa: BLE001
                why = f"model loader not hookable: {exc!r}"
        if why is not None:
            report["warm_reason"] = why
            _log.warning("glm53_moeglue warm enabled (%s) but NOT armed: %s; production unchanged", ENV_WARM, why)
        else:
            warm_ok = True
    if not glue_ok and not warm_ok:
        return report
    if glue_ok and _post_register not in tf.POST_REGISTER_HOOKS:
        tf.POST_REGISTER_HOOKS.append(_post_register)
    orig = prodmod.apply_exl3_experts
    if not hasattr(orig, "_glm53_moeglue_orig"):
        prodmod.apply_exl3_experts = _make_hook(orig)
    STATE.hooked = True
    STATE.enabled, STATE.reason = glue_ok, ("ok" if glue_ok else report["reason"])
    STATE.warm, STATE.warm_reason = warm_ok, ("armed" if warm_ok else report["warm_reason"])
    report.update(installed=glue_ok, warm=warm_ok)
    if glue_ok:
        report.update(reason="ok", prefetch=CFG.prefetch)
        _log.info("glm53_moeglue installed: apply_exl3_experts -> glue_prep | grouped g/u | gateup_epilogue | grouped "
                  "d | glue_finish (prefetch %s); layers are self-tested as TF registers them", CFG.prefetch)
    if warm_ok:
        report.update(warm_reason="armed")
        _log.info("glm53_moeglue warm armed: set %s, <= %g MiB per MoE sublayer, %d blocks, forks at <= %d tokens; "
                  "o_proj layers are wired after weight loading; breakable CUDA graphs: segment guard %s",
                  CFG.warm_set, CFG.warm_mib, CFG.warm_blocks, CFG.warm_max_m, WARM.bk_guard)
    return report


def uninstall(prodmod=None) -> dict:
    import tf_exl3_moe as tf

    rep = {"restored": False}
    if prodmod is None:
        prodmod = STATE.prodmod
    fn = getattr(prodmod, "apply_exl3_experts", None) if prodmod is not None else None
    if fn is not None and hasattr(fn, "_glm53_moeglue_orig"):
        prodmod.apply_exl3_experts = fn._glm53_moeglue_orig
        rep["restored"] = True
    if _post_register in tf.POST_REGISTER_HOOKS:
        tf.POST_REGISTER_HOOKS.remove(_post_register)
    warm_join()
    WARM.bk_skip = False
    STATE.enabled, STATE.reason = False, "uninstalled"
    STATE.warm, STATE.warm_reason, STATE.hooked = False, "uninstalled", False   # wrapped o_proj -> production apply
    VERDICT.clear()
    return rep


def summary() -> str:
    return f"glm53_moeglue: {dict(sorted(COUNTERS.items()))}"


def plugin_install() -> None:
    """integrate.plugin_register calls this after TF's install. Never raises into vLLM's plugin loader."""
    try:
        on, won = env_enabled(), env_warm()
        _log.info("glm53_moeglue plugin loaded (pid %d): %s=%r, %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
                  ENV_WARM, os.environ.get(ENV_WARM), "installing" if (on or won) else "off, production unchanged")
        if on or won:
            install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moeglue not installed (production apply unchanged): %r", exc)
