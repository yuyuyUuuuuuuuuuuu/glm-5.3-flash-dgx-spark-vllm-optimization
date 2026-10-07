"""GLM53_DEC_AR1SHOT: the small (decode) tensor-parallel all-reduce of a 2-rank group as ONE network hop
(docs/DEC_AR1SHOT.md).

Production's decode step runs ~102 NCCL all-reduces over the RoCE pair (2 per layer of the target, 2 per drafter
layer, the embedding). NCCL picks RING_LL for them; with 2 ranks a ring all-reduce is

    send own half-chunk -> recv peer's half, reduce, write, send the reduced half -> recv the peer's reduced half

i.e. TWO dependent network transfers (each through the sender's NCCL proxy thread, the NIC and the receiver's poll).
An all-gather of the same 2-rank group is ONE transfer: every rank sends its whole partial to the peer and receives
the peer's. With GLM53_DEC_AR1SHOT=1, a decode-size all-reduce of a 2-rank group becomes

    pynccl all_gather(partial) -> [2, n];  out = gathered[0] + gathered[1]      (one bf16 add kernel)

Bytes sent per rank are the same (ring: n/2 + n/2, one-shot: n); one network latency instead of two.

Output equivalence: with two ranks every element of NCCL's sum is ONE bf16 addition a + b (rounded to nearest even;
NCCL's FuncSum<bf16> = __hadd2 on sm_80+), computed on whichever rank owns the chunk and copied to the other.
IEEE addition is commutative and torch's bf16 add is float(a) + float(b) rounded once to bf16, which equals the
correctly rounded bf16 sum (the float sum of two bf16 values is exact or, when the exponents differ by > 16, below a
quarter ulp of the result, so the second rounding cannot see a midpoint). So the result is bit-identical to
production's all-reduce; tests/decode6/test_ar1shot.py checks it against the real NCCL all-reduce (2 ranks, NCCL over
loopback sockets on nodeC, random and adversarial bf16 incl. ties, cancellations, huge exponent gaps, subnormals,
+-0) and `GLM53_DEC_AR1SHOT=verify` checks it in production on the real RoCE path (both computed, production's served,
mismatching elements counted on the GPU).

Served calls: world size 2, the group has an enabled pynccl communicator, the input is a CUDA bf16 / fp16 / fp32
contiguous tensor with numel * itemsize <= GLM53_DEC_AR1SHOT_MAX_KB (default 512 KiB = 64 decode rows x 4096 x bf16);
everything else -> production's CudaCommunicator.all_reduce unchanged. The decision depends only on the tensor's
shape / dtype and the group, which are identical on both ranks, so both ranks always issue the same collective.

Both ranks must agree to serve (a rank serving the all-gather against a peer serving the all-reduce would hang):
the wrapper is installed whenever the env asks for it, and its first eligible eager call (the profile run, before any
graph capture) all-reduces a readiness flag over production's path; served only if every rank of the group is ready.
"""
from __future__ import annotations

import logging
import os
import threading

import torch

_log = logging.getLogger("vllm.glm53_ar1shot")
ENV = "GLM53_DEC_AR1SHOT"
ENV_MAX_KB = "GLM53_DEC_AR1SHOT_MAX_KB"
ENV_LOG = "GLM53_DEC_AR1SHOT_LOG"
VERSION = 1
_ON = frozenset({"1", "on", "true", "yes"})
_OFF = frozenset({"", "0", "off", "false", "no"})
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_LOCK = threading.Lock()
COUNTERS = {"captured": 0, "eager": 0, "fallback": 0, "verify_calls_captured": 0, "verify_calls_eager": 0,
            "agree_ok": 0, "agree_fail": 0}


class _State:
    mode = "off"          # off | on | verify
    max_bytes = 512 * 1024
    log_every = 2000      # eager calls between stats lines (the target's eager embedding all-reduce: ~1 per step)
    agreed = {}           # id(device communicator) -> bool
    proof_logged = False
    mism = None           # verify: device int64 [2] = (mismatching elements, compared elements)
    eager_seen = 0


ST = _State()


def env_mode(environ=None) -> str:
    env = os.environ if environ is None else environ
    v = (env.get(ENV) or "").strip().lower()
    if v in _OFF:
        return "off"
    if v in _ON:
        return "on"
    if v == "verify":
        return "verify"
    raise ValueError(f"{ENV} must be unset, 0, 1 or verify (got a different value)")


def _int_env(name, default, lo, hi):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    v = int(raw)
    if not lo <= v <= hi:
        raise ValueError(f"{name} out of range [{lo}, {hi}]")
    return v


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def eligible(dc, x: torch.Tensor) -> bool:
    if getattr(dc, "world_size", 0) != 2:
        return False
    pc = getattr(dc, "pynccl_comm", None)
    if pc is None or getattr(pc, "disabled", True) or getattr(pc, "world_size", 0) != 2:
        return False
    if not x.is_cuda or x.dtype not in _DTYPES or not x.is_contiguous() or x.numel() == 0:
        return False
    return x.numel() * x.element_size() <= ST.max_bytes


def one_shot(pc, x: torch.Tensor) -> torch.Tensor:
    """all_gather the two partials, add them (bf16 add = production's 2-rank NCCL sum, bitwise)."""
    n = x.numel()
    g = torch.empty((2, n), dtype=x.dtype, device=x.device)
    pc.all_gather(g, x.view(-1))
    out = torch.empty_like(x)
    torch.add(g[0], g[1], out=out.view(-1))
    return out


def _agree(dc, orig, ready: bool) -> bool:
    flag = torch.tensor([1.0 if ready else 0.0], dtype=torch.float32, device=torch.cuda.current_device())
    if ST.mode == "verify" and ST.mism is None:
        # verify's counter must exist BEFORE any capture: created lazily inside a capture, its zeros kernel would be
        # recorded into that graph and every replay of it would reset the count (review 2026-10-05, test G)
        ST.mism = torch.zeros(2, dtype=torch.int64, device=flag.device)
    tot = orig(dc, flag)
    ok = bool(round(float(tot.item())) == dc.world_size)
    with _LOCK:
        COUNTERS["agree_ok" if ok else "agree_fail"] += 1
    rank = getattr(dc, "rank_in_group", getattr(dc, "rank", -1))
    if ok:
        _log.info("glm53_ar1shot: rank %s/%d agreement: every rank ready -> one-shot all-reduce armed (mode %s, "
                  "<= %d KiB)", rank, dc.world_size, ST.mode, ST.max_bytes // 1024)
    else:
        _log.warning("glm53_ar1shot: rank %s/%d agreement: a rank is not ready -> production's all-reduce on every "
                     "rank of this group", rank, dc.world_size)
    return ok


def _stats(dc) -> None:
    rank = getattr(dc, "rank_in_group", -1)
    msg = ""
    if ST.mode == "verify" and ST.mism is not None:
        m = ST.mism.tolist()
        msg = f", verify: {m[0]} differing of {m[1]} compared elements"
    with _LOCK:
        c = dict(COUNTERS)
    _log.info("[glm53-ar1shot] rank %s mode %s: captured %d graph calls, eager %d, fallback %d%s", rank, ST.mode,
              c["captured"] + c["verify_calls_captured"], c["eager"] + c["verify_calls_eager"], c["fallback"], msg)
    if ST.mode == "verify" and ST.mism is not None and m[0] != 0:
        _log.warning("glm53_ar1shot: rank %s one-shot all-reduce differs from production's (%d elements)", rank, m[0])


def _wrap_all_reduce(orig):
    def all_reduce(self, input_):
        if ST.mode == "off" or not eligible(self, input_):
            return orig(self, input_)
        key = id(self)
        ok = ST.agreed.get(key)
        cap = _capturing()
        if ok is None:
            if cap:  # never agree inside a capture (the agreement syncs); both ranks reach this point identically
                with _LOCK:
                    COUNTERS["fallback"] += 1
                return orig(self, input_)
            ok = ST.agreed[key] = _agree(self, orig, True)
        if not ok:
            return orig(self, input_)
        pc = self.pynccl_comm
        if ST.mode == "verify":
            ref = orig(self, input_)
            out = one_shot(pc, input_)
            if ST.mism is None and not cap:
                ST.mism = torch.zeros(2, dtype=torch.int64, device=input_.device)
            if ST.mism is not None:   # (never allocated inside a capture: see _agree)
                d = (ref.view(-1).view(torch.int16 if ref.element_size() == 2 else torch.int32)
                     != out.view(-1).view(torch.int16 if out.element_size() == 2 else torch.int32))
                ST.mism[0].add_(d.sum())
                ST.mism[1].add_(d.numel())
            res = ref
            ck = "verify_calls_captured" if cap else "verify_calls_eager"
        else:
            res = one_shot(pc, input_)
            ck = "captured" if cap else "eager"
        with _LOCK:
            COUNTERS[ck] += 1
        if not cap:
            ST.eager_seen += 1
            if not ST.proof_logged and COUNTERS["captured"] + COUNTERS["verify_calls_captured"] > 0:
                ST.proof_logged = True
                _log.info("[glm53-ar1shot] rank %s serving confirmed (mode %s): %d graph-captured one-shot "
                          "all-reduces", getattr(self, "rank_in_group", -1), ST.mode,
                          COUNTERS["captured"] + COUNTERS["verify_calls_captured"])
            if ST.log_every and ST.eager_seen % ST.log_every == 64 % max(ST.log_every, 1):
                _stats(self)
        return res
    all_reduce._glm53_ar1shot = True
    return all_reduce


def _wrap_init(orig_init, orig_all_reduce):
    """Agree right where every rank builds the group's communicator: CudaCommunicator.__init__ is collective (pynccl's
    own warmup all-reduce runs inside it), all ranks construct the same groups in the same order, and it happens long
    before the profile run and any CUDA-graph capture - so the decision is taken eagerly and deterministically, not at
    'the first eligible eager all-reduce' (which, under full-graph capture, may only come with the first request)."""
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            if ST.mode == "off" or getattr(self, "world_size", 0) != 2:
                return
            pc = getattr(self, "pynccl_comm", None)
            if pc is None or getattr(pc, "disabled", True):
                return
            if id(self) not in ST.agreed and not _capturing():
                ST.agreed[id(self)] = _agree(self, orig_all_reduce, True)
                with _LOCK:
                    COUNTERS["agree_at_init"] = COUNTERS.get("agree_at_init", 0) + 1
        except Exception as exc:  # noqa: BLE001  (a failed agreement leaves the lazy path; both ranks run this code)
            _log.warning("glm53_ar1shot: init-time agreement skipped: %r", exc)
    __init__._glm53_ar1shot = True
    return __init__


def install_now(mode: str | None = None) -> dict:
    report = {"mode": "off", "reason": None}
    try:
        mode = env_mode() if mode is None else mode
        report["mode"] = mode
        if mode == "off":
            report["reason"] = "off"
            return report
        ST.max_bytes = _int_env(ENV_MAX_KB, 512, 1, 16384) * 1024
        ST.log_every = _int_env(ENV_LOG, 2000, 0, 10**9)
        from vllm.distributed.device_communicators import cuda_communicator as cc
        C = cc.CudaCommunicator
        if getattr(C.all_reduce, "_glm53_ar1shot", False):
            report["reason"] = "already installed"
        else:
            orig_ar = C.all_reduce
            C.all_reduce = _wrap_all_reduce(orig_ar)
            if not getattr(C.__init__, "_glm53_ar1shot", False):
                C.__init__ = _wrap_init(C.__init__, orig_ar)
        ST.mode = mode
        _log.info("glm53_ar1shot: hooked CudaCommunicator.all_reduce (mode %s, 2-rank groups, <= %d KiB, "
                  "dtypes bf16/fp16/fp32) PROOF mode=%s", mode, ST.max_bytes // 1024, mode)
    except Exception as exc:  # noqa: BLE001
        ST.mode = "off"
        report["mode"] = "off"
        report["reason"] = repr(exc)
        _log.warning("glm53_ar1shot: not installed (production all-reduce unchanged): %r", exc)
    return report


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_DEC_AR1SHOT is on."""
    raw = os.environ.get(ENV)
    try:
        mode = env_mode()
    except ValueError as exc:
        _log.warning("glm53_ar1shot plugin loaded (pid %d): %s -> off (%s)", os.getpid(), ENV, exc)
        return
    _log.info("glm53_ar1shot plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, raw,
              f"installing (mode {mode})" if mode != "off" else "off, production all-reduce unchanged")
    if mode != "off":
        install_now(mode)


def summary() -> dict:
    with _LOCK:
        c = dict(COUNTERS)
    c["mode"] = ST.mode
    c["max_bytes"] = ST.max_bytes
    return c
