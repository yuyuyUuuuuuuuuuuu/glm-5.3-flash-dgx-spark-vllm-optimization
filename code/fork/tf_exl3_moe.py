"""tf_exl3_moe — TensorFold's EXL3 routed-expert kernels behind the exact call contract of
``exllamav3_ext.exl3_moe`` (the production vLLM decode path), per docs/DESIGN.md §C.

Public surface (used by integrate.py only):

    plan(args)   -> Plan | None   pure host metadata, O(1), never raises; None = delegate to the original
    launch(p, args, orig)          enqueue the 6 kernels on the current stream (exception policy §C.5)
    preflight(layer, orig)         once per layer at load (§C.3): eligibility, pointer sanity, scratch,
                                   self-test against the original exl3_moe, registration
    exl3_moe_tf(*29 or 30 args)    strict positional entry for tests (raises if outside the TF domain)
    apply_fused(x2d, ids, weights, layer, inners, expert_map, limit) -> out | None
                                   K2 (docs/OPTIMIZATION.md): production's whole decode apply_exl3_fused_moe
                                   (routing prelude + exl3_moe) from the router ids; None = run production's
                                   own apply

Weights come exclusively from the positional pointer tables (args 13..21), zero copy. Scratch is persistent
and shared by all layers of a device (decode is sequential across layers, as production's own fused temps
assume). Nothing here allocates, synchronizes or reads the environment per call.
"""

from __future__ import annotations

import logging
import importlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"

_log = logging.getLogger("vllm.tf_exl3_moe")

ACT_SILU = 0                      # == docs/prod_exl3_reference.py:58 MOE_ACT_SILU, xl_exl3_moe_common.cuh:6
BITS = 4
SWIGLU_LIMIT_DEFAULT = 10.0       # == docs/prod_exl3_reference.py:54 (apply: swiglu_limit or this default)
# Tile configuration of the compiled kernels: (n tiles a block, warps a block, K splits) for gate/up and down,
# and (P threshold, down K splits) for tiny calls. These are overwritten from the loaded extension's own
# GATEUP_CFG / DOWN_CFG / DOWN_SMALL attributes by load_ext() (kernels/exl3.cpp is the single source of truth);
# the persistent scratch is sized by the extension's z_need_max, never from these numbers.
GATEUP_CFG = (4, 4, 4)
DOWN_CFG = (4, 4, 1)
DOWN_SMALL = (8, 2)
N_MAX_EXPERTS = 4096
_TRUE = frozenset({"1", "on", "true", "yes"})
_CUDA_ERR = re.compile(r"CUDA error|cudaError|illegal|capture")


# ---------------------------------------------------------------------------------------------------------
# configuration (read once: at import and by integrate.install(); never per call)

@dataclass
class Config:
    topk_max: int = 8             # TF_EXL3_TOPK_MAX (GLM num_experts_per_tok)
    max_pairs: int = 1024         # TF_EXL3_MAX_PAIRS
    tokens_lo: int = 1            # TF_EXL3_TOKENS (inclusive window of B served by TF; per-call policy only)
    tokens_hi: int = 1 << 30
    selftest: bool = True         # TF_EXL3_SELFTEST
    strict: bool = False          # TF_EXL3_STRICT: re-raise everything (tests)
    jit: bool = False             # TF_EXL3_JIT: allow the JIT build when the AOT module is missing (nodeC tests)
    apply: bool = True            # TF_EXL3_APPLY: K2, serve production's decode apply from the router ids
    invalid: list = field(default_factory=list)   # malformed TF_EXL3_* values (install() refuses to install)


CFG = Config()
_TOKENS_HI_INF = 1 << 30


def _parse_int(env, name: str, default: int, lo: int, hi: int, c: Config) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw.strip())
    except ValueError:
        c.invalid.append(f"{name}={raw!r} (not an integer)")
        return default
    if not lo <= v <= hi:
        c.invalid.append(f"{name}={raw!r} (must be in [{lo}, {hi}])")
        return default
    return v


def parse_tokens(raw: str) -> tuple[int, int]:
    """TF_EXL3_TOKENS -> inclusive (lo, hi). Forms: "lo:hi", "lo:" (no upper bound), ":hi" (= 1:hi) and a bare
    "N" (= 1:N, an upper bound: the E.5 bench's natural output). lo >= 1 and hi >= lo, else ValueError."""
    t = raw.strip()
    if ":" in t:
        lo_s, _, hi_s = t.partition(":")
        lo = int(lo_s) if lo_s.strip() else 1
        hi = int(hi_s) if hi_s.strip() else _TOKENS_HI_INF
    else:
        lo, hi = 1, int(t)
    if lo < 1 or hi < lo:
        raise ValueError(f"need 1 <= lo <= hi, got {lo}:{hi}")
    return lo, hi


def configure(environ: dict | None = None) -> Config:
    """(Re)read the TF_EXL3_* settings. Called at import and by integrate.install(). Never raises: a malformed
    value keeps that knob's default and is listed in CFG.invalid, and install() then refuses to install."""
    global CFG
    env = os.environ if environ is None else environ
    c = Config()
    c.topk_max = _parse_int(env, "TF_EXL3_TOPK_MAX", 8, 1, 64, c)
    c.max_pairs = _parse_int(env, "TF_EXL3_MAX_PAIRS", 1024, 1, 1 << 20, c)
    tok = env.get("TF_EXL3_TOKENS")
    if tok is not None and tok.strip():
        try:
            c.tokens_lo, c.tokens_hi = parse_tokens(tok)
        except ValueError as exc:
            c.invalid.append(f"TF_EXL3_TOKENS={tok!r} ({exc})")
    c.selftest = (env.get("TF_EXL3_SELFTEST", "1") or "1").strip().lower() in _TRUE
    c.strict = (env.get("TF_EXL3_STRICT", "0") or "0").strip().lower() in _TRUE
    c.jit = (env.get("TF_EXL3_JIT", "0") or "0").strip().lower() in _TRUE
    c.apply = (env.get("TF_EXL3_APPLY", "1") or "1").strip().lower() in _TRUE
    CFG = c
    return c


configure()


# ---------------------------------------------------------------------------------------------------------
# extension loading (A12): AOT module first; the JIT build only when TF_EXL3_JIT is set (nodeC tests)

_EXT: Any = None
_EXT_LOCK = threading.Lock()
EXT_SOURCE = None                 # "aot:<path>" | "jit:<path>"


def _cuda_include_shim(shim_dir: str | None = None) -> list[str]:
    """This image keeps cusparse.h etc. in pip's nvidia/cu13/include, whose own crt/ would shadow the
    toolkit's and break the __cudaLaunch stubs. Symlink only the top-level headers missing from the toolkit
    into a shim directory and -I that."""
    base = Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include")
    tk = Path("/usr/local/cuda/include")
    if not base.is_dir():
        return []
    shim = Path(shim_dir or os.environ.get("TF_EXL3_SHIM", "/tmp/tf_exl3_shim"))
    shim.mkdir(parents=True, exist_ok=True)
    for f in list(base.glob("*.h")) + list(base.glob("*.hpp")):
        if not (tk / f.name).exists():
            link = shim / f.name
            if not link.exists():
                try:
                    link.symlink_to(f)
                except OSError:
                    pass
    return [f"-I{shim}"]


def _sync_tile_cfg(m) -> None:
    """Take the tile configuration from the compiled extension (kernels/exl3.cpp), not from a Python copy."""
    global GATEUP_CFG, DOWN_CFG, DOWN_SMALL
    GATEUP_CFG = tuple(int(v) for v in m.GATEUP_CFG)
    DOWN_CFG = tuple(int(v) for v in m.DOWN_CFG)
    DOWN_SMALL = tuple(int(v) for v in m.DOWN_SMALL)


def load_ext(prefer: str | None = None):
    """The compiled kernels. prefer: None/"auto" (AOT, else JIT if allowed), "aot", "jit". The loaded module
    becomes the active one (_EXT, EXT_SOURCE and the tile configuration all follow it)."""
    global _EXT, EXT_SOURCE
    if _EXT is not None and prefer in (None, "auto"):
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None and prefer in (None, "auto"):
            return _EXT
        mode = prefer or os.environ.get("TF_EXL3_LOADER", "auto")
        err = None
        if mode in ("auto", "aot"):
            try:
                import tf_exl3_moe_ext as m  # AOT (setup.py build_ext / pip install)
                _sync_tile_cfg(m)
                _EXT, EXT_SOURCE = m, f"aot:{getattr(m, '__file__', '?')}"
                return _EXT
            except ImportError as e:
                err = e
                if mode == "aot":
                    raise
        if mode == "jit" or CFG.jit:
            from torch.utils.cpp_extension import load

            here = Path(__file__).resolve().parent / "kernels"
            inc = _cuda_include_shim()
            parity = os.environ.get("TF_PARITY", "1")
            m = load(
                name="tf_exl3_moe_ext_jit" if parity == "1" else f"tf_exl3_moe_ext_jit_p{parity}",
                sources=[str(here / "exl3.cpp"), str(here / "exl3.cu")],
                extra_cuda_cflags=["-O3", f"-DTF_PARITY={parity}", *inc],
                extra_cflags=["-O3", *inc],
                verbose=False,
            )
            _sync_tile_cfg(m)
            _EXT, EXT_SOURCE = m, f"jit:{getattr(m, '__file__', '?')}"
            return _EXT
        raise ImportError(f"tf_exl3_moe_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")


def use_ext(m) -> None:
    """Make an already-loaded extension module the active one (tests switching AOT / JIT / NATIVE builds):
    _EXT, EXT_SOURCE and the tile configuration always move together."""
    global _EXT, EXT_SOURCE
    _sync_tile_cfg(m)
    kind = "aot" if getattr(m, "__name__", "") == "tf_exl3_moe_ext" else "jit"
    _EXT, EXT_SOURCE = m, f"{kind}:{getattr(m, '__file__', '?')}"


# ---------------------------------------------------------------------------------------------------------
# process state

@dataclass
class _State:
    enabled: bool = True
    disabled_reason: str | None = None
    captured: bool = False        # a TF launch was recorded into a CUDA graph (scratch must then stay alive)
    apply_enabled: bool = True    # K2 path (off after a K2 self-test mismatch; the exl3_moe path is unaffected)
    apply_disabled_reason: str | None = None


STATE = _State()
# Why a call was not served by TF (the category of plan()'s / plan_apply()'s miss reason, "<category>: <detail>"):
#   b_gt_r        B > R (EXL3_TEMP_ROWS_FUSED): production's multi-launch prefill branch      } expected, by design
#   window        B outside TF_EXL3_TOKENS                                                     }
#   pairs         B * topk > P_cap (TF_EXL3_MAX_PAIRS) or > the route_ids limit               }
#   unregistered  the layer's pointer tables were not registered (rejected at pre-flight, or built before install)
#   tf_disabled   TF switched itself off (self-test mismatch, launch pre-check failure, uninstall)
#   k2_off        (K2 only) the apply path is off (TF_EXL3_APPLY=0 or a K2 self-test mismatch)
#   tf_error      a TF launch failed its pre-checks and fell back to production (TF is then disabled)
#   contract      anything else: a call production makes that TF does not accept (dtype, stride, table identity..)
DELEGATION_CATEGORIES = ("b_gt_r", "window", "pairs", "unregistered", "tf_disabled", "k2_off", "tf_error", "contract")
COUNTERS = {"tf_calls": 0, "delegated": 0, "fallback_errors": 0, "preflight_ok": 0, "preflight_rejected": 0,
            "tf_apply_calls": 0, "apply_delegated": 0, "graph_tf_calls": 0, "graph_delegated": 0,
            "prod_build_failed": 0, "tf_glue_calls": 0,
            **{f"delegated_{c}": 0 for c in DELEGATION_CATEGORIES},
            **{f"apply_delegated_{c}": 0 for c in DELEGATION_CATEGORIES}}
_LOGGED: set = set()


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


# ---------------------------------------------------------------------------------------------------------
# observability (docs/STATUS.md "Observability"): what served each MoE call, logged for production.
# Every exl3_moe-level call is served by TF (K2 or the dispatcher: COUNTERS tf_calls) or by production's exl3_moe
# (COUNTERS delegated, by category). Python runs only eagerly and while a CUDA graph is captured (a replay re-runs
# the captured kernels), so the capture-time outcome per batch size B is what the captured decode steps run.
# Logged: an INFO summary CAPTURE_QUIET_S after the last captured call (from a daemon thread: no CUDA calls, no
# effect on the serving path), an INFO summary when the call count reaches 10^3, 10^4, ..., and a WARNING (once per
# reason) when a call on a registered layer is delegated for an unexpected reason (contract / unregistered).

GRAPH_B: dict[str, set] = {"tf": set(), **{c: set() for c in DELEGATION_CATEGORIES}}
CAPTURE_QUIET_S = 10.0
_OBS = {"last_capture": 0.0, "watcher": None, "next_summary": 1000}
_OBS_LOCK = threading.Lock()


def _category(reason: str | None) -> str:
    c = (reason or "contract").split(":", 1)[0]
    return c if c in GRAPH_B else "contract"


def _ranges(bs) -> str:
    """{1,2,3,5,8,9} -> "1-3,5,8-9"."""
    out, run = [], []
    for b in sorted(bs):
        if run and b == run[-1] + 1:
            run.append(b)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [b]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(out) or "-"


def _rank() -> str:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return "?"


def summary() -> str:
    """One line: calls served by TF / delegated by category, and what the captured CUDA graphs contain."""
    c = COUNTERS
    total = c["tf_calls"] + c["delegated"]
    deleg = ", ".join(f"{k} {c['delegated_' + k]}" for k in DELEGATION_CATEGORIES if k != "k2_off")
    k2d = ", ".join(f"{k} {c['apply_delegated_' + k]}" for k in DELEGATION_CATEGORIES if c["apply_delegated_" + k])
    graph_prod = "; ".join(f"{k} B={_ranges(set(GRAPH_B[k]))}" for k in DELEGATION_CATEGORIES if GRAPH_B[k])
    return (f"rank {_rank()}: {total} MoE calls: TF {c['tf_calls']} (K2 {c['tf_apply_calls']}), production "
            f"{c['delegated']} ({deleg}); K2 handed to the exl3_moe path: {k2d or 'none'}; CUDA graph capture: TF "
            f"{c['graph_tf_calls']} calls, B={_ranges(set(GRAPH_B['tf']))}; production {c['graph_delegated']} calls"
            f"{' (' + graph_prod + ')' if graph_prod else ''}; production build_exl3_fused_state failures "
            f"{c['prod_build_failed']}")


def _capture_watcher() -> None:
    while True:
        time.sleep(min(1.0, CAPTURE_QUIET_S))
        with _OBS_LOCK:
            if time.monotonic() - _OBS["last_capture"] >= CAPTURE_QUIET_S:
                _OBS["watcher"] = None
                break
    try:
        _log.info("tf_exl3_moe: after CUDA graph capture: %s", summary())
    except Exception:  # noqa: BLE001 - a racing counter update must never kill anything
        pass


def note_call(served: bool, B: int, capturing: bool, reason: str | None = None, k2: bool = False) -> None:
    """Account one exl3_moe-level call (served by TF, or delegated with plan()'s reason). Never raises."""
    try:
        if served:
            if capturing:
                COUNTERS["graph_tf_calls"] += 1
                GRAPH_B["tf"].add(int(B))
        else:
            cat = _category(reason)
            COUNTERS["delegated_" + cat] += 1
            if capturing:
                COUNTERS["graph_delegated"] += 1
                GRAPH_B[cat].add(int(B))
            if cat == "contract" or (cat == "unregistered" and REG):
                _log_once(f"deleg:{reason}"[:100], logging.WARNING,
                          "tf_exl3_moe: a call (B=%s%s) was handed to production's exl3_moe: %s", B,
                          ", during CUDA graph capture" if capturing else "", reason)
        if capturing:
            with _OBS_LOCK:
                _OBS["last_capture"] = time.monotonic()
                if _OBS["watcher"] is None:
                    t = threading.Thread(target=_capture_watcher, name="tf_exl3_moe-capture-summary", daemon=True)
                    _OBS["watcher"] = t
                    t.start()
        total = COUNTERS["tf_calls"] + COUNTERS["delegated"]
        if total >= _OBS["next_summary"]:
            while _OBS["next_summary"] <= total:
                _OBS["next_summary"] *= 10
            _log.info("tf_exl3_moe: %s", summary())
    except Exception:  # noqa: BLE001
        pass


def note_apply_delegated(B, reason: str | None) -> None:
    """K2 handed a call to production's apply (which then reaches the exl3_moe dispatcher). Never raises."""
    try:
        cat = _category(reason)
        COUNTERS["apply_delegated_" + cat] += 1
        if cat == "contract" or (cat == "unregistered" and REG):
            _log_once(f"k2deleg:{reason}"[:100], logging.WARNING,
                      "tf_exl3_moe: K2 apply path handed a call (B=%s) to production's apply (the exl3_moe path "
                      "still sees it): %s", B, reason)
    except Exception:  # noqa: BLE001
        pass


def disable(reason: str) -> None:
    """Sticky process-wide disable: every later plan() returns None."""
    STATE.enabled = False
    STATE.disabled_reason = reason
    _log_once("disable:" + reason[:80], logging.WARNING, "tf_exl3_moe disabled: %s", reason)


def set_enabled(on: bool = True) -> None:
    STATE.enabled = bool(on)
    if on:
        STATE.disabled_reason = None
        STATE.apply_enabled = True
        STATE.apply_disabled_reason = None


def s_cap(P: int, n: int) -> int:
    """Static segment capacity (§B.1): min(P, n) + ceil(P / 16)."""
    return min(P, n) + (P + 15) // 16


class Scratch:
    """Persistent per-(device, K, N) buffers for up to P_cap pairs (§C.3 step 4)."""

    def __init__(self, device: torch.device, K: int, N: int, P_cap: int, ext=None) -> None:
        S = P_cap + (P_cap + 15) // 16                   # >= s_cap(P, n) for every P <= P_cap and n
        f16, i32 = torch.float16, torch.int32
        self.P_cap, self.K, self.N, self.device = P_cap, K, N, device
        self.xg = torch.zeros((P_cap, K), dtype=f16, device=device)
        self.xu = torch.zeros((P_cap, K), dtype=f16, device=device)
        self.xd = torch.zeros((P_cap, N), dtype=f16, device=device)
        # Z: the extension's own bound (max over P <= P_cap of what moe_forward checks), not Python constants
        z = int((ext or load_ext()).z_need_max(P_cap, K, N))
        self.z = torch.zeros((z,), dtype=torch.float32, device=device)
        self.pair_expert = torch.full((P_cap,), -1, dtype=i32, device=device)
        self.seg_expert = torch.zeros((S,), dtype=i32, device=device)
        self.seg_row0 = torch.zeros((S,), dtype=i32, device=device)
        self.seg_rows = torch.zeros((S,), dtype=i32, device=device)
        self.nseg = torch.zeros((1,), dtype=i32, device=device)
        # K2: the sorted pair tables production's prelude would have built (token_sorted, weight_sorted)
        self.ts = torch.zeros((P_cap,), dtype=torch.int64, device=device)
        self.ws = torch.zeros((P_cap,), dtype=f16, device=device)
        # moeglue (docs/DEC_MOEGLUE.md): sorted row of every pair in router order (glue_prep -> glue_finish), 4 B/pair
        self.inv = torch.zeros((P_cap,), dtype=i32, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.xg, self.xu, self.xd, self.z, self.pair_expert,
                                                            self.seg_expert, self.seg_row0, self.seg_rows, self.nseg,
                                                            self.ts, self.ws, self.inv))


_SCRATCH: dict[tuple, Scratch] = {}
_SCRATCH_KEEP: list[Scratch] = []  # outgrown scratch stays alive: a captured graph may still point at it


def _scratch_for(device: torch.device, K: int, N: int, P_cap: int, ext=None) -> Scratch:
    key = (device.index, K, N)
    sc = _SCRATCH.get(key)
    if sc is not None and sc.P_cap >= P_cap:
        return sc
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("tf_exl3_moe scratch allocation during CUDA graph capture")
    if sc is not None:
        _SCRATCH_KEEP.append(sc)
    sc = Scratch(device, K, N, P_cap, ext)
    _SCRATCH[key] = sc
    return sc


_PTR_KEYS = ("gate_trellis", "gate_suh", "gate_svh", "up_trellis", "up_suh", "up_svh",
             "down_trellis", "down_suh", "down_svh")


@dataclass
class LayerInfo:
    device: int
    K: int
    N: int
    n: int
    P_cap: int
    tables: tuple                 # the 9 validated pointer tables (identity-checked per call)
    scratch_key: tuple
    ok: bool = True
    selftest: dict = field(default_factory=dict)


REG: dict[tuple, LayerInfo] = {}


# ---------------------------------------------------------------------------------------------------------
# §C.4 per-call plan

class Plan:
    __slots__ = ("B", "K", "N", "n", "P", "R", "limit", "scratch")

    def __init__(self, B, K, N, n, P, R, limit, scratch):
        self.B, self.K, self.N, self.n, self.P, self.R, self.limit, self.scratch = B, K, N, n, P, R, limit, scratch


_Tensor = torch.Tensor
_F16, _F32, _I64 = torch.float16, torch.float32, torch.int64


def _miss(why: list | None, reason: str) -> None:
    """A plan miss: record "<category>: <detail>" when the caller asked why (DELEGATION_CATEGORIES)."""
    if why is not None:
        why.append(reason)
    return None


def plan(args: tuple, why: list | None = None) -> Plan | None:
    """None => the caller must run the original exl3_moe(*args) unchanged. Never raises.
    why (a list) receives the reason of a miss, "<category>: <detail>" (DELEGATION_CATEGORIES)."""
    try:
        return _plan(args, require_ok=True, why=why)
    except Exception as exc:  # noqa: BLE001 - any surprise means "not ours"
        return _miss(why, f"contract: plan raised {type(exc).__name__}: {str(exc)[:80]}")


def _plan(args: tuple, require_ok: bool, window: bool = True, why: list | None = None) -> Plan | None:
    """require_ok=False skips the process/layer enable state; window=False skips the TF_EXL3_TOKENS policy
    window. Both are for the load-time self-test only: the hard domain limits (contract, registry, B <= R,
    P <= P_cap, scratch capacity) always apply."""
    if not STATE.enabled and require_ok:
        return _miss(why, f"tf_disabled: {STATE.disabled_reason}")
    na = len(args)
    if na != 29 and na != 30:
        return _miss(why, f"contract: {na} arguments")
    act, kg, ku, kd = args[9], args[10], args[11], args[12]
    # scalars must be plain Python numbers (a tensor here would turn == into a device op)
    if type(act) is not int or act != ACT_SILU:
        return _miss(why, f"contract: act_function {act!r}")
    if type(kg) is not int or type(ku) is not int or type(kd) is not int or kg != BITS or ku != BITS or kd != BITS:
        return _miss(why, f"contract: K bits {kg!r}/{ku!r}/{kd!r}")
    gm, g1, um, u1, dm, d1 = args[22:28]
    for f in (gm, g1, um, u1, dm, d1):
        if type(f) is not bool and type(f) is not int:
            return _miss(why, "contract: mcg/mul1 flags not bool/int")
    if not (gm and not g1 and um and not u1 and dm and not d1):
        return _miss(why, "contract: codebook flags (need mcg, not mul1)")
    lim = args[28]
    if type(lim) is not float and type(lim) is not int:
        return _miss(why, f"contract: act_limit type {type(lim).__name__}")
    lim = float(lim)
    if not math.isfinite(lim):
        return _miss(why, "contract: act_limit not finite")
    if na == 30 and type(args[29]) is not int:
        return _miss(why, "contract: num_active not int")
    x, out, ec, ts, ws = args[0], args[1], args[2], args[3], args[4]
    gt = args[13]
    if not isinstance(x, _Tensor) or not isinstance(gt, _Tensor):
        return _miss(why, "contract: hidden_state / gate table not tensors")
    dev = x.get_device()
    if dev < 0:
        return _miss(why, "contract: hidden_state on CPU")
    info = REG.get((dev, gt.data_ptr()))
    if info is None or (require_ok and not info.ok):
        return _miss(why, "unregistered: this layer's pointer tables are not registered")
    tabs = info.tables
    for i in range(9):
        if args[13 + i] is not tabs[i]:           # exactly the tables validated at pre-flight
            return _miss(why, f"contract: pointer table {_PTR_KEYS[i]} is not the registered tensor")
    # hidden_state
    if x.dtype is not _F16 or x.dim() != 2:
        return _miss(why, f"contract: hidden_state {x.dtype} {x.dim()}-D")
    B, K = x.shape
    if K != info.K or B < 1:
        return _miss(why, f"contract: hidden_state shape {tuple(x.shape)} (K={info.K})")
    xs0, xs1 = x.stride()
    if xs1 != 1 or xs0 % 4 or x.data_ptr() % 16:
        return _miss(why, f"contract: hidden_state stride {x.stride()} / alignment {x.data_ptr() % 16}")
    # output_state
    if (not isinstance(out, _Tensor) or out.dtype is not _F32 or out.shape != x.shape or not out.is_contiguous()
            or out.get_device() != dev or out.data_ptr() % 16):
        return _miss(why, "contract: output_state (need fp32, x's shape, contiguous, 16-B aligned)")
    # routing
    if (not isinstance(ec, _Tensor) or ec.dtype is not _I64 or ec.dim() != 1 or ec.numel() != info.n + 1
            or not ec.is_contiguous() or ec.get_device() != dev):
        return _miss(why, "contract: expert_count")
    if (not isinstance(ts, _Tensor) or ts.dtype is not _I64 or ts.dim() != 1 or not ts.is_contiguous()
            or ts.get_device() != dev):
        return _miss(why, "contract: token_sorted")
    P = ts.numel()
    if P < 1 or P > info.P_cap:
        return _miss(why, f"pairs: P={P} outside 1..P_cap={info.P_cap}")
    if (not isinstance(ws, _Tensor) or ws.dtype is not _F16 or ws.dim() != 1 or ws.numel() != P
            or not ws.is_contiguous() or ws.get_device() != dev):
        return _miss(why, "contract: weight_sorted")
    # temps: shapes only (exl3_moe's own shape checks, so a malformed call is delegated to fail identically)
    tsg, tsu, tig, tiu = args[5], args[6], args[7], args[8]
    for t in (tsg, tsu, tig, tiu):
        if not isinstance(t, _Tensor) or t.dtype is not _F16 or t.dim() != 3:
            return _miss(why, "contract: temps (need fp16 3-D)")
    if tsg.shape[2] != K or tsu.shape != tsg.shape or tig.shape != tiu.shape:
        return _miss(why, "contract: temps shapes")
    if tig.shape[1] != tsg.shape[1] or tig.shape[2] != info.N:
        return _miss(why, "contract: temps shapes")
    R = tsg.shape[1]
    if B > R:                                          # single-launch decode domain only
        return _miss(why, f"b_gt_r: B={B} > R={R}")
    if window and (B < CFG.tokens_lo or B > CFG.tokens_hi):   # per-call policy (TF_EXL3_TOKENS), inclusive
        return _miss(why, f"window: B={B} outside TF_EXL3_TOKENS")
    sc = _SCRATCH.get(info.scratch_key)
    if sc is None or sc.P_cap < P:
        return _miss(why, f"pairs: P={P} > scratch capacity")
    return Plan(B, K, info.N, info.n, P, R, lim, sc)


# ---------------------------------------------------------------------------------------------------------
# §C.5 launch

def _launch_kernels(p: Plan, args: tuple) -> None:
    sc = p.scratch
    _EXT.moe_forward(args[0], args[1], args[2], args[3], args[4],
                     args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21],
                     sc.xg, sc.xu, sc.xd, sc.z, sc.pair_expert, sc.seg_expert, sc.seg_row0, sc.seg_rows, sc.nseg,
                     p.R, p.N, p.limit)


def launch(p: Plan, args: tuple, orig=None):
    """Enqueue the TF kernels for a planned call. Returns None like exl3_moe.

    All TORCH_CHECKs of moe_forward run before its first kernel is enqueued, so a check failure (a c10 error
    without a CUDA-error message) leaves output_state untouched: disable TF (sticky) and run orig(*args).
    CUDA errors and capture errors are re-raised (a CUDA error poisons the context for orig as well).
    """
    cap = False
    try:
        cap = torch.cuda.is_current_stream_capturing()
        if cap:
            STATE.captured = True
        _launch_kernels(p, args)
    except Exception as exc:
        COUNTERS["fallback_errors"] += 1
        if CFG.strict or orig is None:
            raise
        if isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) and not _CUDA_ERR.search(str(exc)):
            disable(f"launch pre-check failed: {type(exc).__name__}: {str(exc)[:200]}")
            COUNTERS["delegated"] += 1
            note_call(False, p.B, cap, f"tf_error: {type(exc).__name__}")
            return orig(*args)
        raise
    COUNTERS["tf_calls"] += 1
    note_call(True, p.B, cap)
    return None


def exl3_moe_tf(hidden_state, output_state, expert_count, token_sorted, weight_sorted,
                temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
                act_function, K_gate, K_up, K_down,
                gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
                up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
                down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
                gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
                act_limit, num_active=None, /) -> None:
    """Strict TF entry with exl3_moe's positional contract (num_active accepted and ignored). Raises
    ValueError when the call is outside the TF domain. Not exported as `exl3_moe` (integrate.py dispatches)."""
    args = (hidden_state, output_state, expert_count, token_sorted, weight_sorted,
            temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
            act_function, K_gate, K_up, K_down,
            gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
            down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
            gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1, act_limit)
    p = plan(args)
    if p is None:
        raise ValueError("tf_exl3_moe: call is outside the TF domain (plan() returned None)")
    launch(p, args, None)


# ---------------------------------------------------------------------------------------------------------
# K2 (docs/OPTIMIZATION.md): production's decode apply from the router ids
#
# Production's apply_exl3_fused_moe, decode branch (tokens <= cap; docs/prod_exl3_reference.py:1595-1655):
# map_topk_to_local, argsort, gathers, expert_count, zeros(out), x2d.contiguous().half(), side-effect attributes,
# then one exl3_moe call. apply_fused() does the same with one route_ids kernel in place of the prelude; it serves
# exactly the calls whose exl3_moe call plan() would serve, and returns None (-> production's own apply) otherwise.

_APPLY_X = (torch.bfloat16, torch.float16)
_APPLY_W = (torch.float32, torch.bfloat16, torch.float16)
_ROUTE_IDS_MAX_PAIRS = 1024


class ApplyPlan:
    __slots__ = ("info", "sc", "B", "R", "P", "limit", "tables")

    def __init__(self, info, sc, B, R, P, limit, tables):
        self.info, self.sc, self.B, self.R, self.P, self.limit, self.tables = info, sc, B, R, P, limit, tables


def plan_apply(x2d, ids, weights, layer, inners, expert_map, limit, why: list | None = None) -> ApplyPlan | None:
    """None => production's own apply_exl3_fused_moe must run. Never raises. why: as plan()."""
    try:
        return _plan_apply(x2d, ids, weights, layer, inners, expert_map, limit, require_ok=True, why=why)
    except Exception as exc:  # noqa: BLE001
        return _miss(why, f"contract: plan_apply raised {type(exc).__name__}: {str(exc)[:80]}")


def _plan_apply(x2d, ids, weights, layer, inners, expert_map, limit, require_ok: bool = True,
                window: bool = True, why: list | None = None, ids_dtypes: tuple = (_I64,)) -> ApplyPlan | None:
    if require_ok and not (STATE.enabled and STATE.apply_enabled and CFG.apply):
        return _miss(why, "k2_off: " + (STATE.disabled_reason or STATE.apply_disabled_reason or "TF_EXL3_APPLY=0"))
    if not isinstance(x2d, _Tensor) or x2d.dtype not in _APPLY_X or x2d.dim() != 2:
        return _miss(why, "contract: x (need a bf16/fp16 2-D tensor)")
    dev = x2d.get_device()
    if dev < 0:
        return _miss(why, "contract: x on CPU")
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not isinstance(ptrs, dict) or temps is None:
        return _miss(why, "unregistered: layer has no _exl3_ptrs / _exl3_fused_temps")
    gt = ptrs.get("gate_trellis")
    if not isinstance(gt, _Tensor):
        return _miss(why, "unregistered: no gate_trellis table")
    info = REG.get((dev, gt.data_ptr()))
    if info is None or (require_ok and not info.ok):
        return _miss(why, "unregistered: this layer's pointer tables are not registered")
    tabs = info.tables
    for i, k in enumerate(_PTR_KEYS):
        if ptrs.get(k) is not tabs[i]:                # exactly the tables validated at pre-flight
            return _miss(why, f"contract: pointer table {k} is not the registered tensor")
    if int(getattr(layer, "_exl3_k", BITS)) != BITS:
        return _miss(why, f"contract: layer._exl3_k={getattr(layer, '_exl3_k', None)}")
    if not isinstance(inners, (list, tuple)) or len(inners) != info.n:
        return _miss(why, "contract: inners")
    B, K = x2d.shape
    if K != info.K or B < 1:
        return _miss(why, f"contract: x shape {tuple(x2d.shape)} (K={info.K})")
    xs0, xs1 = x2d.stride()
    if xs1 != 1 or xs0 % 4 or x2d.data_ptr() % 8:
        return _miss(why, f"contract: x stride {x2d.stride()} / alignment {x2d.data_ptr() % 8}")
    # production's decode branch: tokens <= cap = temps[0].shape[1]; the temps exactly as plan() requires them
    if not isinstance(temps, (tuple, list)) or len(temps) != 4:
        return _miss(why, "contract: temps")
    for t in temps:
        if not isinstance(t, _Tensor) or t.dtype is not _F16 or t.dim() != 3:
            return _miss(why, "contract: temps (need fp16 3-D)")
    tsg, tsu, tig, tiu = temps
    if tsg.shape[2] != K or tsu.shape != tsg.shape or tig.shape != tiu.shape or tig.shape[1] != tsg.shape[1] \
            or tig.shape[2] != info.N:
        return _miss(why, "contract: temps shapes")
    R = int(tsg.shape[1])
    if B > R:
        return _miss(why, f"b_gt_r: B={B} > R={R}")
    if window and (B < CFG.tokens_lo or B > CFG.tokens_hi):
        return _miss(why, f"window: B={B} outside TF_EXL3_TOKENS")
    if (not isinstance(ids, _Tensor) or ids.dtype not in ids_dtypes or ids.dim() != 2 or ids.shape[0] != B
            or not ids.is_contiguous() or ids.get_device() != dev):
        return _miss(why, "contract: ids (need contiguous int64 [B, topk] on x's device)")
    P = B * int(ids.shape[1])
    if P < 1 or P > info.P_cap or P > _ROUTE_IDS_MAX_PAIRS:
        return _miss(why, f"pairs: P={P} outside 1..min(P_cap={info.P_cap}, {_ROUTE_IDS_MAX_PAIRS})")
    if (not isinstance(weights, _Tensor) or weights.dtype not in _APPLY_W or weights.shape != ids.shape
            or not weights.is_contiguous() or weights.get_device() != dev):
        return _miss(why, "contract: weights")
    if expert_map is not None:
        # production raises for a map on another device / of another dtype: let it
        if (not isinstance(expert_map, _Tensor) or expert_map.dtype is not _I64 or expert_map.dim() != 1
                or not expert_map.is_contiguous() or expert_map.get_device() != dev):
            return _miss(why, "contract: expert_map")
    if type(limit) is not float and type(limit) is not int:
        return _miss(why, f"contract: limit type {type(limit).__name__}")
    lim = float(limit)
    if not math.isfinite(lim):
        return _miss(why, "contract: limit not finite")
    sc = _SCRATCH.get(info.scratch_key)
    if sc is None or sc.P_cap < P:
        return _miss(why, f"pairs: P={P} > scratch capacity")
    return ApplyPlan(info, sc, B, R, P, lim, tabs)


def _launch_apply(ap: ApplyPlan, x2d, ids, weights, expert_map, out) -> None:
    sc, t = ap.sc, ap.tables
    _EXT.moe_forward_ids(x2d, out, ids, weights, expert_map, ap.info.n, *t, sc.xg, sc.xu, sc.xd, sc.z,
                         sc.pair_expert, sc.seg_expert, sc.seg_row0, sc.seg_rows, sc.nseg, sc.ts, sc.ws,
                         ap.R, ap.info.N, ap.limit)


def apply_fused(x2d, ids, weights, layer, inners, expert_map, limit, why: list | None = None):
    """K2: production's decode apply_exl3_fused_moe (same result, same side effects), or None when the call is not
    one TF serves (the caller then runs production's apply unchanged). Exception policy as launch().
    why (a list): receives the reason when None is returned (plan_apply's, or "tf_error: ...")."""
    ap = plan_apply(x2d, ids, weights, layer, inners, expert_map, limit, why)
    if ap is None:
        COUNTERS["apply_delegated"] += 1
        return None
    # production's decode-branch side effects (docs/prod_exl3_reference.py:1645-1646)
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"
    out = torch.zeros(ap.B, ap.info.K, dtype=torch.float32, device=x2d.device)
    cap = False
    try:
        cap = torch.cuda.is_current_stream_capturing()
        if cap:
            STATE.captured = True
        _launch_apply(ap, x2d, ids, weights, expert_map, out)
    except Exception as exc:
        COUNTERS["fallback_errors"] += 1
        if CFG.strict:
            raise
        if isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) and not _CUDA_ERR.search(str(exc)):
            disable(f"apply pre-check failed: {type(exc).__name__}: {str(exc)[:200]}")
            _miss(why, f"tf_error: {type(exc).__name__}")
            return None
        raise
    COUNTERS["tf_calls"] += 1
    COUNTERS["tf_apply_calls"] += 1
    note_call(True, ap.B, cap, k2=True)
    return out


def disable_apply(reason: str) -> None:
    STATE.apply_enabled = False
    STATE.apply_disabled_reason = reason
    _log_once("disable_apply:" + reason[:80], logging.WARNING, "tf_exl3_moe: apply path (K2) disabled: %s", reason)


# ---------------------------------------------------------------------------------------------------------
# moeglue (docs/DEC_MOEGLUE.md, GLM53_DEC_MOEGLUE; wiring in glm53_moeglue.py): production's whole decode
# apply_exl3_experts from the router's raw ids (int32 or int64) to the routed output in x's dtype, 5 launches. Serves
# exactly the calls K2 serves (the same plan, ids may also be int32; x must be bf16 or fp16, the output is x's dtype).

_GLUE_IDS = (torch.int32, torch.int64)
_GLUE_MAX_TOPK = 32


def plan_glue(x2d, ids, weights, layer, inners, expert_map, limit, why: list | None = None) -> ApplyPlan | None:
    """None => production's own apply_exl3_experts must run. Never raises."""
    try:
        ap = _plan_apply(x2d, ids, weights, layer, inners, expert_map, limit, require_ok=True, why=why,
                         ids_dtypes=_GLUE_IDS)
        if ap is not None and (int(ids.shape[1]) > _GLUE_MAX_TOPK or ap.info.K % 512):
            return _miss(why, f"contract: topk {int(ids.shape[1])} > {_GLUE_MAX_TOPK} or K % 512")
        return ap
    except Exception as exc:  # noqa: BLE001
        return _miss(why, f"contract: plan_glue raised {type(exc).__name__}: {str(exc)[:80]}")


def _launch_glue(ap: ApplyPlan, x2d, ids, weights, expert_map, out, prefetch: bool, parts: int = 3) -> None:
    sc, t = ap.sc, ap.tables
    _EXT.moe_forward_glue(x2d, out, ids, weights, expert_map, ap.info.n, *t, sc.xg, sc.xu, sc.xd, sc.z,
                          sc.pair_expert, sc.seg_expert, sc.seg_row0, sc.seg_rows, sc.nseg, sc.ts, sc.ws, sc.inv,
                          ap.R, ap.info.N, ap.limit, bool(prefetch), int(parts))


def apply_glue(x2d, ids, weights, layer, inners, expert_map, limit, prefetch: bool = False,
               why: list | None = None, mid_event=None):
    """moeglue: production's decode apply_exl3_experts result (x2d's dtype, [B, K]) with apply_exl3_fused_moe's side
    effects, or None when the call is not one TF serves (the caller runs production's apply unchanged). Exception
    policy as apply_fused(). mid_event (a torch.cuda.Event): recorded on the current stream right after the gate/up
    GEMV is enqueued (the launches are split in two there)."""
    ap = plan_glue(x2d, ids, weights, layer, inners, expert_map, limit, why)
    if ap is None:
        return None
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"
    out = torch.empty(ap.B, ap.info.K, dtype=x2d.dtype, device=x2d.device)
    cap = False
    try:
        cap = torch.cuda.is_current_stream_capturing()
        if cap:
            STATE.captured = True
        if mid_event is None:
            _launch_glue(ap, x2d, ids, weights, expert_map, out, prefetch)
        else:
            _launch_glue(ap, x2d, ids, weights, expert_map, out, prefetch, 1)
            mid_event.record()
            _launch_glue(ap, x2d, ids, weights, expert_map, out, prefetch, 2)
    except Exception as exc:
        COUNTERS["fallback_errors"] += 1
        if CFG.strict:
            raise
        if isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) and not _CUDA_ERR.search(str(exc)):
            _miss(why, f"tf_error: {type(exc).__name__}: {str(exc)[:120]}")
            return None
        raise
    COUNTERS["tf_calls"] += 1
    COUNTERS["tf_glue_calls"] += 1
    note_call(True, ap.B, cap, k2=True)
    return out


# Called once per layer after a successful registration (moeglue's per-layer self-test); never raises into _register.
POST_REGISTER_HOOKS: list = []


# ---------------------------------------------------------------------------------------------------------
# §C.3 load-time pre-flight

def _resolve_orig():
    import exllamav3_ext  # the module production reads (`fn = exllamav3_ext.exl3_moe`)

    fn = exllamav3_ext.exl3_moe
    return getattr(fn, "_tf_exl3_orig", fn)


def production_routing(ids: torch.Tensor, weights: torch.Tensor, n: int, expert_map: torch.Tensor | None = None):
    """(expert_count, token_sorted, weight_sorted) in the form exl3_moe consumes (see apply_exl3_fused_moe in the
    production module).

    The global->local id mapping is not re-implemented here: it is delegated at run time to the production
    module's own map_topk_to_local, so the self-test always sees exactly what production would produce.
    The sorted tables follow the exl3_moe contract (docs/ref/xl_exl3_moe.cu header): one entry per
    (token, slot) pair, ordered by local expert id; token_sorted holds the pair's token index,
    weight_sorted its fp16 routing weight; expert_count[e] counts pairs per local expert and the trailing
    bucket [n] counts pairs mapped to n (invalid or non-local), which exl3_moe ignores."""
    prod = importlib.import_module(_PROD_MODULE)
    local = prod.map_topk_to_local(ids, n, expert_map).reshape(-1)
    tokens, topk = ids.shape
    order = torch.argsort(local)
    pair = torch.arange(tokens * topk, device=ids.device, dtype=torch.long)
    token_sorted = torch.div(pair, topk, rounding_mode="floor").index_select(0, order)
    weight_sorted = weights.reshape(-1).to(torch.float16).index_select(0, order)
    expert_count = torch.zeros(n + 1, dtype=torch.long, device=ids.device)
    expert_count.index_add_(0, local.long(), torch.ones(local.numel(), dtype=torch.long, device=ids.device))
    return expert_count, token_sorted, weight_sorted


def compare(out_tf: torch.Tensor, out_ref: torch.Tensor) -> dict:
    """E.1 metrics: global rel_l2, worst per-row rel_l2 over rows with norm > 1e-3 * max row norm, and
    whether the non-finite positions are identical."""
    a, b = out_tf.double(), out_ref.double()
    fin_a, fin_b = torch.isfinite(a), torch.isfinite(b)
    finite_equal = bool(torch.equal(fin_a, fin_b))
    a = torch.where(fin_a & fin_b, a, torch.zeros_like(a))
    b = torch.where(fin_a & fin_b, b, torch.zeros_like(b))
    ref_norm = float(b.norm())
    diff = float((a - b).norm())
    rel = diff / ref_norm if ref_norm > 0 else (0.0 if diff == 0 else math.inf)
    rn = b.norm(dim=1)
    rd = (a - b).norm(dim=1)
    mx = float(rn.max()) if rn.numel() else 0.0
    rows = rn > 1e-3 * mx
    row_rel = float((rd[rows] / rn[rows]).max()) if bool(rows.any()) else 0.0
    zero_rows_ok = bool((rd[~rows] <= 1e-3 * max(mx, 1e-30)).all()) if bool((~rows).any()) else True
    return {"rel_l2": rel, "row_rel_max": row_rel, "finite_equal": finite_equal, "ref_norm": ref_norm,
            "small_rows_ok": zero_rows_ok}


E1_GLOBAL_TOL = 2.0e-3            # 4 * u16 (docs/DESIGN.md §E.1)
E1_ROW_TOL = 1.0e-2


def passes_e1(m: dict) -> bool:
    return (m["finite_equal"] and m["small_rows_ok"] and m["rel_l2"] <= E1_GLOBAL_TOL
            and m["row_rel_max"] <= E1_ROW_TOL)


def _regions_of_layer(layer) -> dict[str, tuple[int, int, int]]:
    """(base, nbytes, matrix_bytes) of the production stacked Parameters each pointer kind must point into."""
    K = int(layer._exl3_hidden_size)
    N = int(layer._exl3_intermediate_local)
    tile_bytes = 16 * BITS * 2                        # int16 [.., 16 * bits] per 16x16 tile = 128 B at 4 bits
    mat_gu = (K // 16) * (N // 16) * tile_bytes
    mat_d = (N // 16) * (K // 16) * tile_bytes

    def reg(t, mb):
        return int(t.data_ptr()), int(t.numel() * t.element_size()), int(mb)

    return {
        "gate_trellis": reg(layer.w13_trellis, mat_gu), "up_trellis": reg(layer.w13_trellis, mat_gu),
        "gate_suh": reg(layer.w13_suh, K * 2), "up_suh": reg(layer.w13_suh, K * 2),
        "gate_svh": reg(layer.w13_svh, N * 2), "up_svh": reg(layer.w13_svh, N * 2),
        "down_trellis": reg(layer.w2_trellis, mat_d),
        "down_suh": reg(layer.w2_suh, N * 2), "down_svh": reg(layer.w2_svh, K * 2),
    }


def production_limit(layer) -> float:
    """The act_limit production will pass for this layer: Exl3MoEMethod.apply uses
    `getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT` (docs/prod_exl3_reference.py:2340), and
    the method is the layer's quant_method."""
    moe = getattr(getattr(layer, "quant_method", None), "moe", None)
    try:
        lim = float(getattr(moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT)
    except (TypeError, ValueError):
        lim = SWIGLU_LIMIT_DEFAULT
    return lim if math.isfinite(lim) else SWIGLU_LIMIT_DEFAULT


def preflight(layer, orig=None) -> bool:
    """§C.3 for a production layer (after build_exl3_fused_state). Never raises; True = registered."""
    try:
        K = int(getattr(layer, "_exl3_hidden_size", 0) or 0)
        N = int(getattr(layer, "_exl3_intermediate_local", 0) or 0)
        bits = int(getattr(layer, "_exl3_bits", 0) or 0)
        k_words = int(getattr(layer, "_exl3_k_words", 0) or 0)
        ptrs = getattr(layer, "_exl3_ptrs", None)
        temps = getattr(layer, "_exl3_fused_temps", None)
        if bits != BITS or k_words != 16 * BITS:
            return _reject(layer, f"bits={bits} k_words={k_words}")
        if not isinstance(ptrs, dict) or any(k not in ptrs for k in _PTR_KEYS) or temps is None:
            return _reject(layer, "no _exl3_ptrs / _exl3_fused_temps")
        regions = _regions_of_layer(layer)
        return register(ptrs, temps, K, N, regions, orig=orig, name=type(layer).__name__,
                        limit=production_limit(layer))
    except Exception as exc:  # noqa: BLE001 - §C.3: never propagate into production's load path
        return _reject(layer, f"{type(exc).__name__}: {exc}")


def _reject(layer_or_name, reason: str) -> bool:
    COUNTERS["preflight_rejected"] += 1
    _log_once("reject:" + reason[:60], logging.WARNING, "tf_exl3_moe: layer not registered (%s)", reason)
    return False


def register(ptrs: dict, temps: tuple, K: int, N: int, regions: dict, *, orig=None, name: str = "layer",
             selftest: bool | None = None, limit: float = SWIGLU_LIMIT_DEFAULT) -> bool:
    """Validate one layer's pointer tables against the memory regions they must point into, allocate the
    shared scratch, run the self-test against the original exl3_moe, and register. Never raises.
    `limit` = the act_limit production passes for this layer (the self-test runs at it and at 0)."""
    try:
        return _register(ptrs, temps, K, N, regions, orig, name, CFG.selftest if selftest is None else selftest,
                         limit)
    except Exception as exc:  # noqa: BLE001
        return _reject(name, f"{type(exc).__name__}: {exc}")


class _SelfTestNotPlanned(RuntimeError):
    """A self-test call fell outside the hard TF domain of this layer (not a numerical mismatch)."""


def _register(ptrs, temps, K, N, regions, orig, name, selftest, limit=SWIGLU_LIMIT_DEFAULT) -> bool:
    if torch.cuda.is_current_stream_capturing():
        return _reject(name, "pre-flight during CUDA graph capture")
    if not STATE.enabled:
        return _reject(name, f"TF disabled: {STATE.disabled_reason}")
    if CFG.invalid:
        return _reject(name, f"invalid TF_EXL3_* settings: {'; '.join(CFG.invalid)}")
    try:
        ext = load_ext()                                             # step 1
    except Exception as exc:  # noqa: BLE001
        disable(f"extension unavailable: {exc!r}")
        return _reject(name, "extension unavailable")
    if ext.parity() != 1:
        _log_once("parity", logging.WARNING, "tf_exl3_moe: extension built with TF_PARITY=0 (NATIVE numerics)")
    # step 2: eligibility
    if K <= 0 or N <= 0 or K % 256 or N % 128:
        return _reject(name, f"shape K={K} N={N} (need K % 256 == 0, N % 128 == 0)")
    tables = tuple(ptrs[k] for k in _PTR_KEYS)
    t0 = tables[0]
    if not isinstance(t0, torch.Tensor) or not t0.is_cuda:
        return _reject(name, "pointer tables not CUDA tensors")
    n = int(t0.numel())
    if n < 1 or n > N_MAX_EXPERTS:
        return _reject(name, f"n={n}")
    dev = t0.device
    for k, t in zip(_PTR_KEYS, tables):
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or t.dim() != 1 or t.numel() != n
                or not t.is_contiguous() or t.device != dev):
            return _reject(name, f"pointer table {k} malformed")
    if not (isinstance(temps, (tuple, list)) and len(temps) == 4
            and all(isinstance(t, torch.Tensor) and t.dtype == torch.float16 and t.dim() == 3 for t in temps)):
        return _reject(name, "temps malformed (need 4 fp16 3-D tensors)")
    R = int(temps[0].shape[1])
    # exactly the temp shapes plan() accepts per call (else every call of this layer would be delegated)
    if (int(temps[0].shape[2]) != K or temps[1].shape != temps[0].shape or temps[3].shape != temps[2].shape
            or int(temps[2].shape[1]) != R or int(temps[2].shape[2]) != N or R < 1):
        return _reject(name, "temps shapes")
    # step 3: pointer sanity (one host copy of the 9 tables, allowed at load)
    host = torch.stack([t for t in tables]).cpu().tolist()
    for k, vals in zip(_PTR_KEYS, host):
        base, nbytes, mb = regions[k]
        for v in vals:
            if v == 0 or v % 16:
                return _reject(name, f"{k}: null or misaligned pointer")
            if not (base <= v <= base + nbytes - mb) or (v - base) % mb:
                return _reject(name, f"{k}: pointer outside its weight tensor")
    # step 4: scratch
    P_cap = min(R * CFG.topk_max, CFG.max_pairs)
    sc = _scratch_for(dev, K, N, P_cap, ext)
    info = LayerInfo(device=dev.index, K=K, N=N, n=n, P_cap=P_cap, tables=tables, scratch_key=(dev.index, K, N))
    key = (dev.index, t0.data_ptr())
    REG[key] = info
    # step 5: self-test against the original exl3_moe on this layer's real tables and temps
    if selftest:
        try:
            orig_fn = orig if orig is not None else _resolve_orig()
            stats = _selftest(info, tables, temps, orig_fn, limit)
            info.selftest = stats
        except _SelfTestNotPlanned as exc:
            # a structural limit of this layer, not a numerical mismatch: leave only this layer on the original
            REG.pop(key, None)
            return _reject(name, f"self-test call not planned: {exc}")
        except Exception as exc:  # noqa: BLE001
            REG.pop(key, None)
            disable(f"self-test raised: {type(exc).__name__}: {exc}")
            return _reject(name, "self-test raised")
        if not stats["ok"]:
            REG.pop(key, None)
            disable(f"self-test mismatch: {stats}")
            return _reject(name, "self-test mismatch")
    if selftest and CFG.apply and STATE.apply_enabled:
        try:
            k2 = _selftest_apply(info, tables, temps, limit)
        except Exception as exc:  # noqa: BLE001
            k2 = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        info.selftest["apply"] = k2
        if not k2["ok"]:
            disable_apply(f"K2 self-test mismatch: {k2}")
    for hook in POST_REGISTER_HOOKS:                                  # empty unless GLM53_DEC_MOEGLUE installed one
        try:
            hook(info, tables, temps, limit)
        except Exception as exc:  # noqa: BLE001 - a hook never demotes the layer
            _log_once("post_register:" + repr(exc)[:60], logging.WARNING,
                      "tf_exl3_moe: post-register hook raised %r (layer stays registered)", exc)
    COUNTERS["preflight_ok"] += 1
    _log_once("registered", logging.INFO, "tf_exl3_moe: layers registered (rank %s, pid %d; K=%d N=%d n=%d P_cap=%d, "
              "scratch %.1f MiB, %s)", _rank(), os.getpid(), K, N, n, P_cap, sc.nbytes() / 2**20, EXT_SOURCE)
    if info.selftest:
        st = info.selftest
        k2 = st.get("apply")
        k2s = ("K2 apply path: off" if k2 is None else
               f"K2 apply path self-test {'ok' if k2['ok'] else 'MISMATCH (K2 disabled)'} "
               f"({k2.get('cases', 0)} cases, bitwise on one route per token, rel_l2 max {k2.get('rel_l2_max', 0.0):.1e})")
        # one line per layer: the real-weight margin to the E.1 tolerance can be read from production logs
        _log.info("tf_exl3_moe: layer %d registered, self-test rel_l2 max %.2e (tol %.0e, margin %.1fx; row max %.2e) "
                  "over B=%s L=%s; %s", COUNTERS["preflight_ok"], st["rel_l2_max"], E1_GLOBAL_TOL,
                  E1_GLOBAL_TOL / max(st["rel_l2_max"], 1e-30), st["row_rel_max"], st["B"], st["L"], k2s)
    return True


def selftest_shapes(info: LayerInfo, R: int) -> tuple[int, tuple[int, ...]]:
    """(topk, batch sizes) of the self-test: B = 1 and up to 8, always inside this layer's hard domain
    (B <= R, B * topk <= P_cap), independent of the per-call TF_EXL3_TOKENS window."""
    topk = max(1, min(8, info.n, info.P_cap))
    b2 = min(8, R, info.P_cap // topk)
    return topk, ((1, b2) if b2 > 1 else (1,))


def _selftest(info: LayerInfo, tables: tuple, temps: tuple, orig, limit: float = SWIGLU_LIMIT_DEFAULT) -> dict:
    """Seeded synthetic x (fp16), production routing incl. one sentinel id, B in {1, up to 8},
    L in {production limit, 0}; E.1 criteria. L = 0 (no clamp) leaves every gate/up weight or scale error
    visible; the production limit is what the layer will actually run. (L = 1 is not used: it clamps most
    activations, which masks gate/up scale errors, and is the noisiest case vs exl3_moe.)"""
    dev = torch.device("cuda", info.device)
    g = torch.Generator(device="cpu").manual_seed(20260927)
    n = info.n
    R = int(temps[0].shape[1])
    topk, Bs = selftest_shapes(info, R)
    Ls = tuple(dict.fromkeys((float(limit), 0.0)))
    worst = {"ok": True, "cases": 0, "rel_l2_max": 0.0, "row_rel_max": 0.0, "B": Bs, "L": Ls, "per_case": []}
    for B in Bs:
        x = torch.randn(B, info.K, generator=g).to(torch.float16).to(dev)
        ids = torch.stack([torch.randperm(n, generator=g)[:topk] for _ in range(B)]).to(dev)
        if B > 1:
            ids[B - 1, topk - 1] = -1                                     # one invalid (sentinel) route
        w = torch.softmax(torch.randn(B, topk, generator=g), -1).to(dev)
        ec, ts, ws = production_routing(ids, w, n)
        for L in Ls:
            base = (x, None, ec, ts, ws, *temps, ACT_SILU, BITS, BITS, BITS, *tables,
                    True, False, True, False, True, False, float(L))
            out_ref = torch.zeros(B, info.K, dtype=torch.float32, device=dev)
            out_tf = torch.zeros_like(out_ref)
            a_ref = base[:1] + (out_ref,) + base[2:]
            a_tf = base[:1] + (out_tf,) + base[2:]
            p = _plan(a_tf, require_ok=False, window=False)
            if p is None:
                raise _SelfTestNotPlanned(f"B={B} P={ts.numel()} R={R} P_cap={info.P_cap}")
            orig(*a_ref)
            _launch_kernels(p, a_tf)
            torch.cuda.synchronize(dev)
            m = compare(out_tf, out_ref)
            worst["cases"] += 1
            worst["per_case"].append((B, L, m["rel_l2"]))
            worst["rel_l2_max"] = max(worst["rel_l2_max"], m["rel_l2"])
            worst["row_rel_max"] = max(worst["row_rel_max"], m["row_rel_max"])
            if not passes_e1(m):
                worst["ok"] = False
                worst["failed"] = {"B": B, "L": L, **m}
                return worst
    return worst


def _selftest_apply(info: LayerInfo, tables: tuple, temps: tuple, limit: float = SWIGLU_LIMIT_DEFAULT) -> dict:
    """K2 against the exl3_moe path (itself checked against production's exl3_moe just before) on this layer's
    tables: bf16 x through production's .half(), fp32 router weights, one sentinel id; (a) one route per token:
    bit-identical out, (b) topk routes per token: E.1. B in {1, up to 8} (as _selftest), L = production limit."""
    dev = torch.device("cuda", info.device)
    g = torch.Generator(device="cpu").manual_seed(20260928)
    n, R = info.n, int(temps[0].shape[1])
    topk, Bs = selftest_shapes(info, R)
    res = {"ok": True, "cases": 0, "rel_l2_max": 0.0}
    sc = _SCRATCH[info.scratch_key]
    for B in Bs:
        for k in (1, topk):
            if B * k > min(info.P_cap, _ROUTE_IDS_MAX_PAIRS):
                continue
            x = torch.randn(B, info.K, generator=g).to(torch.bfloat16).to(dev)
            ids = torch.stack([torch.randperm(n, generator=g)[:k] for _ in range(B)]).to(dev)
            if B > 1:
                ids[B - 1, k - 1] = -1                                  # one invalid (sentinel) route
            w = torch.softmax(torch.randn(B, k, generator=g), -1).to(dev)
            ec, ts, ws = production_routing(ids, w, n)
            out_ref = torch.zeros(B, info.K, dtype=torch.float32, device=dev)
            a = (x.half(), out_ref, ec, ts, ws, *temps, ACT_SILU, BITS, BITS, BITS, *tables,
                 True, False, True, False, True, False, float(limit))
            p = _plan(a, require_ok=False, window=False)
            if p is None:
                continue
            _launch_kernels(p, a)
            out = torch.zeros_like(out_ref)
            ap = ApplyPlan(info, sc, B, R, B * k, float(limit), tables)
            _launch_apply(ap, x, ids, w, None, out)
            torch.cuda.synchronize(dev)
            res["cases"] += 1
            if k == 1:
                if not torch.equal(out, out_ref):
                    res.update(ok=False, failed={"B": B, "topk": k, "bitwise": False})
                    return res
            else:
                m = compare(out, out_ref)
                res["rel_l2_max"] = max(res["rel_l2_max"], m["rel_l2"])
                if not passes_e1(m):
                    res.update(ok=False, failed={"B": B, "topk": k, **m})
                    return res
    return res


def reset() -> None:
    """Forget registrations (tests / uninstall). Scratch is freed only if no TF launch was ever captured
    into a CUDA graph; otherwise it stays alive because a replay would still write into it."""
    REG.clear()
    if not STATE.captured:
        _SCRATCH.clear()
        _SCRATCH_KEEP.clear()
