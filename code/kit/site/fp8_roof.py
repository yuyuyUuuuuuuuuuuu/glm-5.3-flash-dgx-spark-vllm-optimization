"""GLM53_DEC_FP8ROOF: decode-side work on production's FP8 linears beyond fp8_gemv.py (docs/DEC_FP8ROOF.md).

Two independent parts, both inert unless ``GLM53_DEC_FP8ROOF`` in {1, on, true, yes} (read once, at plugin load, in
every vLLM process) AND the fp8_gemv small-M path is installed (``GLM53_FP8_GEMV=1``: the hooks live in its
Glm53DenseFp8Method wrappers). Unset -> nothing is patched, no extension is imported, production is unchanged.

(1) Table (``GLM53_DEC_FP8ROOF_TABLE``, default on): fp8_gemv.TABLE gets entries for shapes it did not serve
    (never changes an existing entry, so every shape production serves today keeps its kernel and its bits):
      (4096, 20480)  the drafter fc (GLM53_DRAFT_FP8=layers,fc; production: Marlin, 398 us/step in the R15 trace)
    Served exactly like every other fp8_gemv shape (per-layer self-test vs Marlin at load or at the first eager
    call, Marlin on any decline).

(2) L2 prefetch (``GLM53_DEC_FP8ROOF_PF``, default ``all`` = t0..t5, or a comma list; ``off`` disables): between two
    bandwidth-bound kernels of a decode step production runs ~50-300 us of latency-bound work (all-reduce, mHC,
    norms, KDA recurrent, MLA attention) during which DRAM is nearly idle. Right after a known predecessor is
    enqueued, a side stream (forked by an event, so it starts when the predecessor finishes) streams the first
    bytes of the NEXT FP8 weight into L2 with plain loads whose values are discarded (kernels/fp8_roof.cu); the
    successor's GEMV then hits L2 for those bytes. Triggers (predecessor -> prefetched weight, default budget):
      t1  KDA in_proj(L)          -> KDA o_proj(L)                         12 MiB (window: f_b/g_b, conv, recurrent)
      t2  KDA / MLA o_proj(L)      -> shared-expert gate_up(L) + down(L)    all 12.6 MB   (window: all-reduce, mHC,
                                     (dense layers: mlp.gate_up(L))         12 MiB         router; the shared expert
                                                                                           then runs from L2 next
                                                                                           to the routed MoE)
      t3  routed MoE(L)            -> first FP8 linear(s) of layer L+1       12 MiB (window: epilogue, all-reduce, mHC)
          (dense layers:              (KDA in_proj, or MLA fused_qkv_a
          mlp.down_proj(L))           then the start of MLA q_b)
      t4  MLA q_b(L)               -> MLA o_proj(L)                          16 MiB (window: MLA attention)
    and two eager-only triggers on the (FP8, GLM53_LMHEAD_FP8) lm_head, which the drafter shares with the target; per
    step the eager calls are  drafter fc, drafter lm_head, [target forward: a CUDA graph, no Python], target lm_head:
      t5  target lm_head           -> drafter fc (GLM53_DRAFT_FP8=..fc)     16 MiB (window: logits all-gather, sampler,
          (2nd lm_head call since                                                   rejection, host step loop, ~200 us)
          the last fc call)
      t0  any other lm_head call   -> first FP8 linear of target layer 0   12 MiB (window: drafter logits, host step
          (the drafter's; or the                                                    loop, embedding all-reduce, ~2 ms)
          target's when there is
          no FP8 fc)
    t0 / t5 fork only outside CUDA-graph capture; their pending prefetch is joined by the next eager hook (fc, the
    target's layer-0 linear when the target runs eagerly, or the next lm_head call), never inside a capture.
    (``GLM53_DEC_FP8ROOF_PF_MIB`` = "t0=12,t1=12,t2=16,t3=12,t4=16,t5=16" overrides budgets; t2 caps the shared/dense
    bytes.)
    The fork happens only for decode-sized calls (rows <= ``GLM53_DEC_FP8ROOF_PF_MAX_M``, default 64), never
    under torch.compile (the drafter), and only when the successor is a registered layer of the same forward.
    Join ("late"): right AFTER the successor's own kernel is enqueued, its stream waits for the prefetch's
    completion event, so the successor never waits for the prefetch and every forked stream rejoins the capture
    origin before the forward ends (CUDA-graph capture requirement). Safety net: a pending prefetch whose target
    layer index is below the current one is joined at the next hook; pending prefetches left over from an
    earlier forward (a new forward is detected when the layer index goes backwards) or recorded in another
    capture state (eager vs capturing) are dropped, never joined across a capture boundary.
    The t3 trigger wraps production's Exl3MoEMethod.apply (routed experts; output returned unchanged); for the
    three dense layers its predecessor is the dense MLP's down_proj (an FP8 linear, hooked like the others).
    Breakable CUDA graphs (vLLM VLLM_USE_BREAKABLE_CUDAGRAPH=1, auto-enabled for Glm5Next*: production's PIECEWISE
    graphs, vllm/compilation/breakable_cudagraph.py): such a capture is a chain of separate CUDA graphs ("segments")
    cut at every @eager_break_during_capture op, which runs eagerly between them: the KDA core
    (Glm5NextLinearAttention._forward, between in_proj / f_b / g_b and o_proj), the sparse-MLA indexer and the MLA
    attention (both between q_b and o_proj). A stream forked in one segment must rejoin before that segment ends, or
    capture_end fails with "capturing stream has unjoined work" (the first deploy-r16 boot). So inside a breakable
    segment (and only there: eager calls, FULL graphs and a breakable capture in FULL runtime mode are unchanged):
      - t1 and t4 are not forked (BREAK_CROSS: their join site lies behind an eager break; tests/r16/
        test_r16_breakable.py shows each alone failing the capture without this);
      - guard: BreakableCUDAGraphCapture.add_eager / _end_segment are wrapped so that every prefetch forked in the
        closing segment is joined into it first (on the segment's capture stream). At a break that is a structure
        this module did not expect: the trigger is then no longer forked inside breakable segments (WARNING once per
        trigger); at the end of a capture (normal or raising) the join is silent, so a failed capture never leaves a
        stale fork behind and its real error is not masked by an unjoined-stream error.

Numerics: (2) changes no computed value (it only reads weights; outputs are bitwise identical with and without,
tests/test_fp8_roof.py R.8 / R.10 and every simulator run). (1) moves the drafter fc from Marlin to fp8_gemv: not
bitwise vs Marlin (rel_l2 1e-5..1.1e-4 on the real fc weight), the same error vs float64 as Marlin (R.5); it only
feeds the drafter, so it can change proposals / acceptance, never the target's distribution. Collectives are
untouched; every decision depends only on the tensor shapes, the layer structure, the call order and the
environment, i.e. is the same on every TP rank. Memory: no allocation (one side stream and CUDA events).
Revert: unset GLM53_DEC_FP8ROOF on both ranks and restart (or _PF=off / _TABLE=0 for one part).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).resolve().parent
ENV = "GLM53_DEC_FP8ROOF"
ENV_TABLE = "GLM53_DEC_FP8ROOF_TABLE"
ENV_PF = "GLM53_DEC_FP8ROOF_PF"
ENV_PF_MIB = "GLM53_DEC_FP8ROOF_PF_MIB"
ENV_PF_CTAS = "GLM53_DEC_FP8ROOF_PF_CTAS"
ENV_PF_MAX_M = "GLM53_DEC_FP8ROOF_PF_MAX_M"
ENV_PF_POL = "GLM53_DEC_FP8ROOF_PF_POL"
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
_TRUE = frozenset({"1", "on", "true", "yes"})
_FALSE = frozenset({"0", "off", "false", "no"})
_log = logging.getLogger("vllm.tf_fp8_roof")

# (1) fp8_gemv.TABLE additions: (Npad, K) -> {MB bucket: (warps, kw, u, evict_first)}. Paired CUDA-graph A/B vs
# production's Marlin on nodeC (tests/roof/bench_fc.py, docs/logs/dec_fp8roof/): both buckets use KW = 8.
EXTRA_TABLE: dict[tuple[int, int], dict[int, tuple[int, int, int, bool]]] = {
    (4096, 20480): {1: (16, 8, 2, True), 2: (8, 8, 2, True)},
}

# (2) prefetch plan
TRIGGERS = ("t0", "t1", "t2", "t3", "t4", "t5")
# triggers whose join site lies behind an eager break of vLLM's breakable CUDA graphs (never forked inside a segment)
BREAK_CROSS = {"t1": "KDA core Glm5NextLinearAttention._forward between in_proj and o_proj",
               "t4": "sparse-MLA indexer + MLA attention between q_b and o_proj"}
DEFAULT_MIB = {"t0": 12.0, "t1": 12.0, "t2": 16.0, "t3": 12.0, "t4": 16.0, "t5": 16.0}
EAGER_ROLES = frozenset({"lm_head", "fc"})     # (role, -1): eager-only anchors of t0 / t5
DEFAULT_CTAS = 8
DEFAULT_MAX_M = 64
# (group of Glm53DenseFp8Method, prefix suffix) -> role
ROLE_OF = (
    ("kda", ".self_attn.in_proj_qkvbfg_a", "kda_in"),
    ("kda", ".self_attn.o_proj", "kda_o"),
    ("mla", ".self_attn.fused_qkv_a_proj", "mla_qkv_a"),
    ("mla", ".self_attn.q_b_proj", "mla_q_b"),
    ("mla", ".self_attn.o_proj", "mla_o"),
    ("shared", ".mlp.shared_experts.gate_up_proj", "sh_gu"),
    ("shared", ".mlp.shared_experts.down_proj", "sh_dn"),
    ("dense", ".mlp.gate_up_proj", "dense_gu"),
    ("dense", ".mlp.down_proj", "dense_dn"),
)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index(prefix: str) -> Optional[int]:
    m = _LAYER_RE.search(prefix or "")
    return int(m.group(1)) if m else None


def role_of(group: str, prefix: str) -> Optional[tuple[str, int]]:
    """(role, layer index) of a production FP8 linear, or None (not part of the prefetch plan). The lm_head
    (glm53_runtime's Glm53DenseFp8Method("lm_head", "lm_head")) and the drafter fc ("draft", "model.fc") are
    ("lm_head", -1) / ("fc", -1)."""
    if group == "lm_head":
        return "lm_head", -1
    if group == "draft" and prefix and prefix.split(".")[-1] == "fc" and _layer_index(prefix) is None:
        return "fc", -1
    if not prefix or ".mtp" in prefix or "draft" in prefix or "visual" in prefix:
        return None
    L = _layer_index(prefix)
    if L is None:
        return None
    for g, suffix, role in ROLE_OF:
        if g == group and prefix.endswith(suffix):
            return role, L
    return None


class _State:
    def __init__(self) -> None:
        self.installed = False
        self.table = False
        self.pf = False
        self.triggers: frozenset = frozenset()
        self.mib = dict(DEFAULT_MIB)
        self.ctas = DEFAULT_CTAS
        self.max_m = DEFAULT_MAX_M
        self.pol = 0
        self.ext = None
        self.side: dict = {}             # device index -> side stream
        self.reg: dict = {}              # (role, L) -> weight tensor (Marlin int32, contiguous)
        self.pending: dict = {}          # (role, L) -> (event, capturing, target L, trigger)
        self.last_L = -1
        self.moe_cls = None
        self.moe_orig = None
        self.lm_count = 0                # eager lm_head calls since the last fc call (t0 / t5)
        self.disabled_reason: Optional[str] = "not installed"
        self.added_shapes: list = []
        self.bk_skip: set = set()        # triggers not forked inside a breakable CUDA-graph segment
        self.bk_learn = True             # a trigger joined at a break joins bk_skip
        self.bk_guard: Optional[str] = None   # "hooked" | "absent" (no breakable capture in this vLLM) | refusal


STATE = _State()
COUNTERS: dict[str, int] = {}
_LOGGED: set = set()


def _count(key: str, n: int = 1) -> None:
    COUNTERS[key] = COUNTERS.get(key, 0) + n


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def env_enabled(environ: dict | None = None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


def ext():
    """AOT module tf_fp8_roof_ext (setup.py) or, with TF_EXL3_JIT=1 (nodeC tests), a JIT build of the .cu."""
    if STATE.ext is not None:
        return STATE.ext
    try:
        import tf_fp8_roof_ext as m
    except ImportError as err:
        if os.environ.get("TF_EXL3_JIT", "0").strip().lower() not in _TRUE:
            raise ImportError(f"tf_fp8_roof_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load
        import fp8_gemv
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = fp8_gemv._include_shim()
        m = load(name="tf_fp8_roof_jit", sources=[str(HERE / "kernels" / "fp8_roof.cu")],
                 extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)
    STATE.ext = m
    return m


# ---------------------------------------------------------------------------------------------------------------
# env parsing
def _parse_pf(raw: Optional[str]) -> tuple[frozenset, Optional[str]]:
    v = (raw or "").strip().lower()
    if v in ("", "all") or v in _TRUE:
        return frozenset(TRIGGERS), None
    if v in _FALSE:
        return frozenset(), None
    items = {t.strip() for t in v.split(",") if t.strip()}
    bad = items - set(TRIGGERS)
    if bad:
        return frozenset(), f"{ENV_PF}={raw!r}: unknown trigger(s) {sorted(bad)} (expected all | off | {','.join(TRIGGERS)})"
    return frozenset(items), None


def _parse_mib(raw: Optional[str]) -> tuple[dict, Optional[str]]:
    out = dict(DEFAULT_MIB)
    v = (raw or "").strip()
    if not v:
        return out, None
    for item in v.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            return dict(DEFAULT_MIB), f"{ENV_PF_MIB}: {item!r} is not t<n>=<MiB>"
        k, val = (s.strip().lower() for s in item.split("=", 1))
        try:
            f = float(val)
        except ValueError:
            f = -1.0
        if k not in DEFAULT_MIB or not 0.0 <= f <= 64.0:
            return dict(DEFAULT_MIB), f"{ENV_PF_MIB}: {item!r} (trigger in {TRIGGERS}, 0 <= MiB <= 64)"
        out[k] = f
    return out, None


def _parse_int(name: str, default: int, lo: int, hi: int) -> tuple[int, Optional[str]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default, None
    try:
        v = int(raw)
    except ValueError:
        return default, f"{name}={raw!r} is not an integer"
    if not lo <= v <= hi:
        return default, f"{name}={raw!r} is not in {lo}..{hi}"
    return v, None


# ---------------------------------------------------------------------------------------------------------------
# registry (called from fp8_gemv's process_weights_after_loading wrapper, eager, at load)
def register(method, layer) -> None:
    """Record a production FP8 linear that is part of the prefetch plan (by its group + prefix)."""
    if not STATE.pf:
        return
    key = role_of(getattr(method, "group", ""), getattr(method, "prefix", ""))
    if key is None:
        return
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or not w.is_cuda or not w.is_contiguous() or w.data_ptr() % 16:
        _log_once("reg_bad", logging.WARNING, "tf_fp8_roof: %s weight is not a contiguous 16-byte aligned CUDA "
                  "tensor: not prefetched", getattr(method, "prefix", "?"))
        return
    STATE.reg[key] = w
    layer._glm53_roof_key = key
    dev = w.device.index if w.device.index is not None else torch.cuda.current_device()
    if dev not in STATE.side:
        STATE.side[dev] = torch.cuda.Stream(device=w.device)
    _count("registered")


def _nbytes(w: torch.Tensor) -> int:
    return w.numel() * w.element_size()


def plan(src_role: str, L: int) -> tuple[Optional[str], list]:
    """(trigger, [((role, L'), nbytes), ...]) prefetched after (src_role, L); the first entry is the join key."""
    mib = STATE.mib
    reg = STATE.reg

    def take(key, budget):
        w = reg.get(key)
        if w is None or budget <= 0:
            return None
        return key, min(_nbytes(w), int(budget)) // 16 * 16

    def first_of_layer(L1, budget):
        if ("kda_in", L1) in reg:
            return [take(("kda_in", L1), budget)]
        if ("mla_qkv_a", L1) in reg:           # MLA: all of fused_qkv_a (8.4 MB), then the start of q_b
            a = take(("mla_qkv_a", L1), budget)
            return [a] if a is None else [a, take(("mla_q_b", L1), budget - a[1])]
        return []

    out = []
    trig = None
    if src_role == "lm_head":                  # L = eager lm_head calls since the last fc call (after_linear)
        if L == 2 and ("fc", -1) in reg:
            trig = "t5"
            out = [take(("fc", -1), mib["t5"] * 2 ** 20)]
        else:
            trig = "t0"
            out = first_of_layer(0, mib["t0"] * 2 ** 20)
    elif src_role == "kda_in":
        trig = "t1"
        out = [take(("kda_o", L), mib["t1"] * 2 ** 20)]
    elif src_role in ("kda_o", "mla_o"):
        trig = "t2"
        budget = mib["t2"] * 2 ** 20
        if ("sh_gu", L) in reg:
            a = take(("sh_gu", L), budget)
            out = [a]
            if a is not None:
                out.append(take(("sh_dn", L), budget - a[1]))
        else:
            out = [take(("dense_gu", L), budget)]
    elif src_role in ("moe", "dense_dn"):
        trig = "t3"
        out = first_of_layer(L + 1, mib["t3"] * 2 ** 20)
    elif src_role == "mla_q_b":
        trig = "t4"
        out = [take(("mla_o", L), mib["t4"] * 2 ** 20)]
    out = [o for o in out if o is not None and o[1] > 0]
    if trig not in STATE.triggers:
        return trig, []
    return trig, out


# ---------------------------------------------------------------------------------------------------------------
# fork / join (called from the apply wrappers, right after the layer's own kernels are enqueued)
def _new_forward_check(L: int) -> None:
    if L < STATE.last_L and STATE.pending:
        _count("stale_dropped", len(STATE.pending))
        _log_once("stale_drop", logging.WARNING, "tf_fp8_roof: %d prefetch(es) of an earlier forward were never "
                  "joined (dropped; counted, not repeated)", len(STATE.pending))
        STATE.pending.clear()
    STATE.last_L = L


def _join(key, L: int) -> None:
    """Late join: the current stream (the successor's, its kernel already enqueued) waits for the prefetch."""
    if not STATE.pending:
        return
    cap = torch.cuda.is_current_stream_capturing()
    cur = torch.cuda.current_stream()
    for k in list(STATE.pending):
        ev, ev_cap, tL, _trig = STATE.pending[k]
        if k == key or tL < L:
            del STATE.pending[k]
            if ev_cap != cap:
                _count("dropped_capture_mismatch")
                continue
            cur.wait_event(ev)
            _count("joined" if k == key else "joined_late_safety")


def _fork(trig: str, targets: list) -> None:
    cur = torch.cuda.current_stream()
    dev = cur.device.index if cur.device.index is not None else torch.cuda.current_device()
    side = STATE.side.get(dev)
    if side is None:
        return
    cap = torch.cuda.is_current_stream_capturing()
    E = STATE.ext
    ev = torch.cuda.Event()
    ev.record(cur)
    side.wait_event(ev)
    done = torch.cuda.Event()
    try:
        with torch.cuda.stream(side):
            for key, nb in targets:
                E.l2_prefetch(STATE.reg[key], 0, nb, STATE.ctas, STATE.pol)
    finally:
        done.record(side)       # always: the side stream must be joinable even if a launch raised
        first = targets[0][0]   # ... and it is: pending even on a raise, so _disable() rejoins it (a capture in
        old = STATE.pending.pop(first, None)          # progress must end with every forked stream joined)
        if old is not None:      # same target forked twice without a join in between: join the older one now
            if old[1] == cap:
                cur.wait_event(old[0])
            _count("double_fork")
        STATE.pending[first] = (done, cap, first[1], trig)
    _count("fork_" + trig)
    _count("bytes_" + trig, sum(nb for _, nb in targets))


def _rows(x) -> int:
    try:
        return x.numel() // max(int(x.shape[-1]), 1)
    except Exception:  # noqa: BLE001
        return 1 << 30


def _disable(exc: BaseException) -> None:
    STATE.pf = False
    STATE.disabled_reason = f"prefetch raised {type(exc).__name__}: {str(exc)[:200]}"
    _log.warning("tf_fp8_roof: L2 prefetch disabled for the rest of this process: %s", STATE.disabled_reason)
    _rejoin_all()


def _rejoin_all() -> None:
    """After a failure (prefetch now off, so no later hook joins anything): the current stream waits for every pending
    prefetch of the current capture state. Without it a non-CUDA error raised in a hook during a CUDA-graph capture
    (e.g. a TORCH_CHECK of l2_prefetch) would leave forked side streams unjoined and fail the whole capture, i.e.
    vLLM's start-up; with it the capture ends cleanly and production's path runs without the prefetch."""
    try:
        cap = torch.cuda.is_current_stream_capturing()
        cur = torch.cuda.current_stream()
        for k in list(STATE.pending):
            ev, ev_cap, _tL, _trig = STATE.pending.pop(k)
            if ev_cap == cap:
                cur.wait_event(ev)
                _count("joined_on_disable")
            else:
                _count("dropped_capture_mismatch")
    except Exception:  # noqa: BLE001 - best effort; a CUDA / capture error is re-raised by the caller anyway
        pass


_MISSING = object()


def _key_of(layer, method):
    """The plan key of a layer: set by register() at load; else classified once from its method (the lm_head, which
    glm53_runtime converts by hand without process_weights_after_loading) and cached on the layer (None = no role)."""
    key = getattr(layer, "_glm53_roof_key", _MISSING)
    if key is _MISSING:
        key = None
        if method is not None:
            k = role_of(getattr(method, "group", "") or "", getattr(method, "prefix", "") or "")
            key = k if k is not None and k[0] == "lm_head" else None
        try:
            layer._glm53_roof_key = key
        except Exception:  # noqa: BLE001
            pass
    return key


def _join_eager(match) -> None:
    """Eager anchors (lm_head, fc): the current stream waits for every pending eager prefetch (late join: this call's
    kernel is already enqueued). Capture-time entries cannot be pending outside a capture (the capture would have
    failed); if one is, it is dropped, never waited on."""
    cur = None
    for k in list(STATE.pending):
        ev, ev_cap, _tL, _trig = STATE.pending.pop(k)
        if ev_cap:
            _count("dropped_capture_mismatch")
            continue
        if cur is None:
            cur = torch.cuda.current_stream()
        cur.wait_event(ev)
        _count("joined" if k == match else "joined_eager")


def _eager_anchor(role: str, x) -> None:
    """t0 / t5 bookkeeping at an eager lm_head or drafter fc call. Inside a CUDA-graph capture: nothing at all."""
    if torch.cuda.is_current_stream_capturing():
        return
    if role == "fc":
        _join_eager(("fc", -1))
        STATE.lm_count = 0
        return
    _join_eager(None)                        # lm_head: the previous eager prefetch (t0 or t5) is long done
    STATE.lm_count += 1
    if _rows(x) <= STATE.max_m:
        trig, targets = plan("lm_head", STATE.lm_count)
        if targets:
            _fork(trig, targets)


def after_linear(layer, x, method=None) -> None:
    """Hook: a production FP8 linear's kernels were just enqueued on the current stream."""
    if not STATE.pf:
        return
    key = _key_of(layer, method)
    if key is None:
        return
    role, L = key
    try:
        if role in EAGER_ROLES:
            _eager_anchor(role, x)
            return
        _join(key, L)
        _new_forward_check(L)
        if _rows(x) <= STATE.max_m:
            trig, targets = plan(role, L)
            if targets and not _bk_skipped(trig):
                _fork(trig, targets)
    except Exception as exc:  # noqa: BLE001
        _disable(exc)
        if "capture" in str(exc) or "CUDA" in str(exc):
            raise


def after_moe(layer, x) -> None:
    """Hook: production's routed-expert kernels of a MoE layer were just enqueued on the current stream."""
    if not STATE.pf or "t3" not in STATE.triggers:
        return
    L = getattr(layer, "_glm53_roof_L", None)
    if L is None:
        L = _layer_index(getattr(layer, "layer_name", "") or getattr(layer, "prefix", "") or "")
        L = -1 if L is None else L
        try:
            layer._glm53_roof_L = L
        except Exception:  # noqa: BLE001
            pass
    if L < 0:
        return
    try:
        _join(None, L)
        _new_forward_check(L)
        if _rows(x) <= STATE.max_m:
            trig, targets = plan("moe", L)
            if targets and not _bk_skipped(trig):
                _fork(trig, targets)
    except Exception as exc:  # noqa: BLE001
        _disable(exc)
        if "capture" in str(exc) or "CUDA" in str(exc):
            raise


def _make_moe_apply(orig):
    def apply(self, layer, *args, **kwargs):
        out = orig(self, layer, *args, **kwargs)
        if STATE.pf and not torch.compiler.is_compiling():
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, torch.Tensor):
                after_moe(layer, x)
        return out

    apply.__doc__ = getattr(orig, "__doc__", None)
    apply._tf_fp8_roof_hook = True
    apply._tf_fp8_roof_orig = orig
    return apply


# ---------------------------------------------------------------------------------------------------------------
# breakable CUDA graphs (module docstring): no prefetch outlives the graph segment it was forked in
class _Bk:
    cls = None       # vllm.compilation.breakable_cudagraph.BreakableCUDAGraphCapture once hooked
    fc = None        # (is_forward_context_available, get_forward_context, CUDAGraphMode.FULL)


def _bk_active() -> bool:
    """True inside a breakable-CUDA-graph segment that an eager break can end: vLLM's eager_break_during_capture rule
    (a BreakableCUDAGraphCapture of this thread is capturing and the forward context is not in FULL runtime mode)."""
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


def _bk_skipped(trig) -> bool:
    """A BREAK_CROSS (or learned) trigger inside a breakable segment: not forked (counted)."""
    if trig in STATE.bk_skip and _bk_active():
        _count("bk_skipped_" + str(trig))
        return True
    return False


def _bk_join(capobj, at_break: bool) -> None:
    """Join every prefetch forked in the segment that is about to end into it, on the segment's capture stream.
    at_break: called from add_eager (an eager break follows: a structure BREAK_CROSS did not name -> learn + WARNING);
    else from _end_segment, i.e. the end of the capture (normal or raising): silent, counted."""
    s = getattr(capobj, "_glm53_origin", None)
    if s is None:
        s = torch.cuda.current_stream()
    for k in list(STATE.pending):
        ev, ev_cap, _tL, trig = STATE.pending[k]
        if not ev_cap:
            continue                        # an eager fork (t0 / t5): not part of this capture, its own rules apply
        del STATE.pending[k]
        try:
            s.wait_event(ev)
            _count("joined_at_break" if at_break else "joined_at_capture_end")
        except Exception:  # noqa: BLE001 - the capture is already invalid; capture_end reports the real error
            _count("dropped_at_segment_end")
            continue
        if at_break:
            _count("joined_at_break_" + str(trig))
            if STATE.bk_learn and trig not in STATE.bk_skip:
                STATE.bk_skip.add(trig)
                _log.warning("tf_fp8_roof: prefetch %s was still pending at a breakable CUDA-graph break (joined there, "
                             "the capture stays valid); %s is no longer forked inside breakable segments", trig, trig)


def _in_chain(fn, marker: str) -> bool:
    while fn is not None:
        if getattr(fn, marker, False):
            return True
        fn = getattr(fn, "_glm53_seg_orig", None)
    return False


def _bk_hook() -> str:
    """Wrap vLLM's BreakableCUDAGraphCapture.add_eager / _end_segment / _begin_segment (idempotent; the wrappers do
    nothing while no prefetch is pending). Returns "hooked", "absent" (this vLLM has no breakable capture) or why not."""
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
    if not _in_chain(cls.add_eager, "_glm53_roof_seg"):
        orig_add, orig_end, orig_begin = cls.add_eager, cls._end_segment, cls._begin_segment

        def add_eager(self, fn):
            if STATE.pending and getattr(self, "_capturing", False):
                _bk_join(self, True)
            return orig_add(self, fn)

        def _end_segment(self):
            if STATE.pending and getattr(self, "_capturing", False):
                _bk_join(self, False)
            return orig_end(self)

        def _begin_segment(self):
            self._glm53_origin = torch.cuda.current_stream()     # the stream capture_begin captures
            return orig_begin(self)

        for f, o in ((add_eager, orig_add), (_end_segment, orig_end), (_begin_segment, orig_begin)):
            f._glm53_roof_seg = True
            f._glm53_seg_orig = o
            f.__doc__ = o.__doc__
        cls.add_eager, cls._end_segment, cls._begin_segment = add_eager, _end_segment, _begin_segment
    _Bk.cls = cls
    return "hooked"


# ---------------------------------------------------------------------------------------------------------------
def install(prodmod=None, *, force: bool | None = None) -> dict:
    """Enable the parts selected by the environment. Needs fp8_gemv installed first (its wrappers call the hooks).
    Idempotent; never raises (problems -> WARNING + report["reason"])."""
    import fp8_gemv as F
    rep: dict = {"installed": False, "reason": None}
    on = env_enabled() if force is None else bool(force)
    if not on:
        rep["reason"] = f"{ENV} not enabled"
        return rep
    if STATE.installed:
        rep.update(installed=True, reason="already installed", table=STATE.table, pf=STATE.pf)
        return rep
    try:
        if not F.STATE.enabled or F.STATE.cls is None:
            rep["reason"] = (f"{ENV} is set but the fp8_gemv small-M path is not installed (GLM53_FP8_GEMV unset or "
                             f"refused: {F.STATE.disabled_reason}): nothing to do")
            _log.warning("tf_fp8_roof: %s", rep["reason"])
            return rep
        # (1) table
        tv = os.environ.get(ENV_TABLE, "").strip().lower()
        STATE.table = tv not in _FALSE
        if STATE.table:
            for shape, cfgs in EXTRA_TABLE.items():
                if shape not in F.TABLE:
                    F.TABLE[shape] = dict(cfgs)
                    STATE.added_shapes.append(shape)
        # (2) prefetch
        trig, why = _parse_pf(os.environ.get(ENV_PF))
        mib, why2 = _parse_mib(os.environ.get(ENV_PF_MIB))
        ctas, why3 = _parse_int(ENV_PF_CTAS, DEFAULT_CTAS, 1, 256)
        max_m, why4 = _parse_int(ENV_PF_MAX_M, DEFAULT_MAX_M, 1, 256)
        pol, why5 = _parse_int(ENV_PF_POL, 0, 0, 1)
        problems = [w for w in (why, why2, why3, why4, why5) if w]
        if problems:
            trig = frozenset()
            _log.warning("tf_fp8_roof: L2 prefetch NOT enabled: %s", "; ".join(problems))
        pf_ok = bool(trig)
        if pf_ok:
            try:
                E = ext()
                if getattr(E, "VERSION", None) != 1:
                    raise ImportError(f"extension version {getattr(E, 'VERSION', None)!r} != 1")
            except Exception as exc:  # noqa: BLE001
                pf_ok = False
                _log.warning("tf_fp8_roof: L2 prefetch NOT enabled: extension not importable: %r", exc)
        if pf_ok:
            STATE.bk_guard = _bk_hook()
            if STATE.bk_guard not in ("hooked", "absent"):
                pf_ok = False
                _log.warning("tf_fp8_roof: L2 prefetch NOT enabled: breakable CUDA graphs cannot be guarded (%s)",
                             STATE.bk_guard)
        if pf_ok and "t3" in trig:
            try:
                if prodmod is None:
                    import importlib
                    prodmod = importlib.import_module(PROD_MODULE)
                cls = getattr(prodmod, "Exl3MoEMethod", None)
                if cls is None or not callable(getattr(cls, "apply", None)):
                    raise AttributeError("no Exl3MoEMethod.apply")
                if not getattr(cls.apply, "_tf_fp8_roof_hook", False):
                    STATE.moe_cls, STATE.moe_orig = cls, cls.apply
                    cls.apply = _make_moe_apply(cls.apply)
            except Exception as exc:  # noqa: BLE001
                trig = trig - {"t3"}
                _log.warning("tf_fp8_roof: trigger t3 (MoE -> next layer) NOT enabled: %r", exc)
        STATE.triggers, STATE.mib, STATE.ctas, STATE.max_m, STATE.pol = frozenset(trig), mib, ctas, max_m, pol
        STATE.bk_skip = set(BREAK_CROSS) & set(trig)
        STATE.pf = pf_ok and bool(trig)
        F.ROOF_HOOK = sys.modules[__name__]
        STATE.installed = True
        STATE.disabled_reason = None
    except Exception as exc:  # noqa: BLE001
        rep["reason"] = f"install raised {exc!r}"
        _log.warning("tf_fp8_roof: NOT installed: %s", rep["reason"])
        return rep
    rep.update(installed=True, reason="ok", table=STATE.table, added_shapes=list(STATE.added_shapes), pf=STATE.pf,
               triggers=sorted(STATE.triggers), mib=dict(STATE.mib), ctas=STATE.ctas, max_m=STATE.max_m, pol=STATE.pol,
               bk_guard=STATE.bk_guard, bk_skip=sorted(STATE.bk_skip))
    _log.info("tf_fp8_roof installed (%s): table additions %s; L2 prefetch %s", ENV,
              [f"{a}x{b}" for a, b in STATE.added_shapes] or "none",
              (f"triggers {sorted(STATE.triggers)}, MiB {STATE.mib}, {STATE.ctas} CTAs, rows <= {STATE.max_m}, "
               f"policy {'evict_last' if STATE.pol else 'normal'}; breakable CUDA graphs: segment guard "
               f"{STATE.bk_guard}, not forked inside a segment: {sorted(STATE.bk_skip) or 'none'}")
              if STATE.pf else "off")
    return rep


def uninstall() -> dict:
    import fp8_gemv as F
    rep = {"restored": False}
    if STATE.moe_cls is not None and getattr(STATE.moe_cls.apply, "_tf_fp8_roof_hook", False):
        STATE.moe_cls.apply = STATE.moe_cls.apply._tf_fp8_roof_orig
        rep["restored"] = True
    for shape in STATE.added_shapes:
        F.TABLE.pop(shape, None)
    if getattr(F, "ROOF_HOOK", None) is not None:
        F.ROOF_HOOK = None
    STATE.__init__()
    STATE.ext = None
    return rep


def summary() -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(COUNTERS.items())) or "no calls"


def plugin_register() -> None:
    """Called from integrate.plugin_register after fp8_gemv.plugin_register (every vLLM process); never raises."""
    try:
        on = env_enabled()
        _log.info("tf_fp8_roof plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
                  "installing" if on else "off, production unchanged")
        if on:
            install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_fp8_roof plugin install failed (production unchanged): %r", exc)
