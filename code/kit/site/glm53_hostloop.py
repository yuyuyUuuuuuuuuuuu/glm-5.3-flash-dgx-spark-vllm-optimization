"""GLM53_DEC_HOSTLOOP: take the per-step host<->GPU sync of the sparse-MLA planner off the decode critical path
(docs/DEC_HOSTLOOP.md).

Production (vLLM model runner V2, async scheduling, DFlash2 drafter) runs the host one step ahead of the GPU: while
the GPU executes step n (target forward, sampling, drafter), the worker thread already prepares step n+1. Exactly one
call per step blocks the host on the GPU: FlashInferMLASparseSM90Builder._kv_lens_host (the patched
flashinfer_mla_sparse_sm90.py production mounts) needs the exact per-row KV lengths on the host for flashinfer's
planner, and under async scheduling it gets them with ``cam.seq_lens[:num_reqs].cpu()``: a pageable D2H copy that
waits for EVERYTHING queued on the stream, i.e. for the end of step n's drafter. Everything the host still has to do
for step n+1 after that point (the flashinfer plan, the four GDN/KDA metadata builds, the eager embedding all-reduce,
the target graph launch) then runs while the GPU idles: ~1.9 ms/step on rank 0 in the R15 decode trace
(tail of dec_gaps.py: Memcpy->Memcpy, scatter/index chains, 'before the target's first kernels').

The values that copy returns do not depend on the drafter. seq_lens(n+1)[i] = num_computed_tokens.gpu[idx_i] +
query_len_i (_prepare_pos_seq_lens_kernel), and num_computed_tokens.gpu is written only by post_update (sampling of
step n, BEFORE the drafter) and by add_requests' staged writes (new requests, whose value the host knows exactly). So
with GLM53_DEC_HOSTLOOP=1:
  * GPUModelRunner.postprocess_sampled (after post_update of step n) queues a 4-byte-per-request D2H copy of
    num_computed_tokens.gpu into a pinned buffer and records an event: both land on the stream BEFORE the drafter;
  * add_requests remembers the request slots it (re)filled since that snapshot (their GPU value is the host value);
  * prepare_inputs stashes this step's batch layout (the stashed context is cleared when execute_model returns);
  * _kv_lens_host, when the metadata it is given is this step's (same seq_lens buffer, same query_start_loc), waits
    for the snapshot EVENT (ready when step n's sampling finished, ~6 ms before the drafter ends) instead of the whole
    stream, rebuilds seq_lens exactly on the host and runs production's own _kv_lens_host code on it (the .cpu() of
    a CPU tensor is a no-op): the per-row lengths handed to flashinfer's plan() are the same integers.
The host then prepares and launches step n+1 while the GPU still runs the drafter; the target starts right after it.

Safety:
  * anything unexpected (other metadata, a snapshot not taken, a stash that does not match, a value that is not
    <= the host's optimistic upper bound, adaptive verification, PCP, a vLLM whose relevant functions differ from
    the fingerprinted ones) -> production's own path for that call (its D2H sync), counted, never an error;
  * verification: the first GLM53_DEC_HOSTLOOP_VERIFY (default 64) fast-path calls, and then one in every
    GLM53_DEC_HOSTLOOP_VERIFY_EVERY (default 1024), ALSO run production's D2H copy and compare the whole seq_lens
    vector; any difference -> WARNING and the fast path is switched off for the life of the process;
  * decisions only change when the host waits, never a value or a collective: the plan inputs are identical integers
    on the fast path and on the fallback, so the two TP ranks stay in lockstep whatever each one decides;
  * nothing is captured into a CUDA graph (the snapshot is skipped while a capture is running); memory: one pinned
    int32 host vector of max_num_reqs entries, no GPU memory.

GLM53_DEC_HOSTLOOP_METER=<N> (independent of GLM53_DEC_HOSTLOOP; off when unset/0): per-rank host-loop meter from CUDA
events on the model runner's stream, no profiler needed. Every N decode steps one INFO line
"[glm53-hostloop] meter rank R: ..." with median / p90 / max of
  gap_ms   : GPU time from the end of step n's queued work (event after sample_tokens returned, i.e. after the
             drafter) to the start of step n+1's forward (event at step_timing.forward_start): the host loop the GPU
             waits for, including step n+1's small input-preparation kernels;
  step_ms  : forward start to forward start (the GPU step time on this rank);
  hwait_ms : host time spent inside _kv_lens_host (the sync on the production path, the event wait here);
  cJ_ms    : duration of the step's J-th EAGER TP collective (production decode has two: c0 = the target's embedding
             all-reduce at the start of the target forward, c1 = the target's logits all-gather; the other 102
             collectives run inside CUDA graphs) = wait for the other rank + transfer: the rank with the longer
             duration arrived first, the difference of the two ranks' values is the skew at that point.
Both ranks log it, so the host-loop gap and its difference between the ranks (skew at the target start) can be
measured in production on nodeA and nodeB alike.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import inspect
import logging
import os
import sys
import textwrap
import time

_log = logging.getLogger("vllm.glm53_hostloop")
ENV = "GLM53_DEC_HOSTLOOP"
ENV_VERIFY = "GLM53_DEC_HOSTLOOP_VERIFY"
ENV_VERIFY_EVERY = "GLM53_DEC_HOSTLOOP_VERIFY_EVERY"
ENV_METER = "GLM53_DEC_HOSTLOOP_METER"
ENV_WAKE = "GLM53_DEC_HOSTLOOP_WAKE"
ENV_WAKE_TICK = "GLM53_DEC_HOSTLOOP_WAKE_TICK_US"
ENV_WAKE_PIN = "GLM53_DEC_HOSTLOOP_WAKE_PIN"
_OFF = frozenset({"", "0", "off", "false", "no"})

MR_MODULE = "vllm.v1.worker.gpu.model_runner"
SM90_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"

# sha256(ast.dump(source))[:16] (integrate.source_fingerprint) of the production functions this relies on, read from
# the production image ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor with the launcher's
# vllm-patches/flashinfer_mla_sparse_sm90.py.patched mounted (tests/test_hostloop.py prints and checks them).
#   _kv_lens_host: async branch = cam.seq_lens[:num_reqs].cpu(), lens derived only from seq_lens and the host qsl
#   prepare_inputs: seq_lens from prepare_pos_seq_lens(num_computed_tokens.gpu, query_start_loc from the host qsl)
#   postprocess_sampled: post_update is the only num_computed_tokens.gpu writer on the decode path
#   add_requests / RequestState.add_request: new slots get the host value by staged write
#   execute_model: prepare_inputs -> ... -> model_state.prepare_attn (builders) -> forward_start -> forward
#   _prepare_pos_seq_lens_kernel: seq_len = num_computed + (qsl[i+1]-qsl[i]); padded entries 0
VERIFIED_FINGERPRINTS = {
    "FlashInferMLASparseSM90Builder._kv_lens_host": frozenset({"281328403e853281"}),
    "GPUModelRunner.prepare_inputs": frozenset({"bad94d805e6a9f3e"}),
    "GPUModelRunner.postprocess_sampled": frozenset({"a1a918b3a8845bbd"}),
    "GPUModelRunner.add_requests": frozenset({"6db20e629ce7ed1c"}),
    "GPUModelRunner.execute_model": frozenset({"5d4e89e96731eb6e"}),
    "RequestState.add_request": frozenset({"f441e1fd18f1ce03"}),
    "_prepare_pos_seq_lens_kernel": frozenset({"5e42c1095fe39147"}),
}

STATS = {"fast": 0, "verified": 0, "mismatch": 0, "fallback_no_ctx": 0, "fallback_mismatch_ctx": 0,
         "fallback_no_snapshot": 0, "fallback_bound": 0, "fallback_error": 0, "snapshots": 0,
         "snapshots_skipped_capture": 0}


class _State:
    def __init__(self) -> None:
        self.enabled = False            # fast path installed and not switched off
        self.disabled_reason: str | None = None
        self.snap_cpu = None            # pinned int32 [max_num_reqs]
        self.snap_np = None
        self.snap_event = None
        self.snap_valid = False
        self.fresh: set[int] = set()    # request slots (re)filled by add_requests since the last snapshot
        self.ctx = None                 # _Ctx of the step being prepared (cleared when execute_model returns)
        self.verify_first = 64
        self.verify_every = 1024
        self.logged: set[str] = set()


ST = _State()


class _Ctx:
    __slots__ = ("seq_lens_ptr", "num_reqs", "idx_np", "qsl_np", "nc_upper", "fresh")

    def __init__(self, seq_lens_ptr, num_reqs, idx_np, qsl_np, nc_upper, fresh) -> None:
        self.seq_lens_ptr, self.num_reqs, self.idx_np = seq_lens_ptr, num_reqs, idx_np
        self.qsl_np, self.nc_upper, self.fresh = qsl_np, nc_upper, fresh


class _CamView:
    """Production's CommonAttentionMetadata with seq_lens replaced by the exact host vector (read-only view)."""
    __slots__ = ("_cam", "seq_lens")

    def __init__(self, cam, seq_lens) -> None:
        object.__setattr__(self, "_cam", cam)
        object.__setattr__(self, "seq_lens", seq_lens)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_cam"), name)


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _OFF


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw, 10))
    except ValueError:
        _log.warning("[glm53-hostloop] %s=%r is not an integer; using %d", name, raw, default)
        return default


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in ST.logged:
        ST.logged.add(key)
        _log.log(level, msg, *a)


def _rank() -> str:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))


def source_fingerprint(fn) -> str | None:
    """Same definition as integrate.source_fingerprint (kept local: this module must import without the TF fork)."""
    fn = getattr(fn, "fn", fn)          # triton.JITFunction -> python function
    fn = inspect.unwrap(fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _disable(reason: str) -> None:
    if ST.enabled:
        ST.enabled = False
        ST.disabled_reason = reason
        ST.ctx = None
        _log.warning("[glm53-hostloop] rank %s: fast path switched OFF for this process (%s); production's "
                     "synchronous _kv_lens_host is used from now on. stats=%s", _rank(), reason, STATS)


# ------------------------------------------------------------------ model-runner side (snapshot, fresh slots, stash)
def _snapshot(runner) -> None:
    import torch
    if torch.cuda.is_current_stream_capturing():
        ST.snap_valid = False
        STATS["snapshots_skipped_capture"] += 1
        return
    nc = runner.req_states.num_computed_tokens.gpu
    if ST.snap_cpu is None or ST.snap_cpu.shape != nc.shape or ST.snap_cpu.dtype != nc.dtype:
        ST.snap_cpu = torch.empty(nc.shape, dtype=nc.dtype, pin_memory=True)
        ST.snap_np = ST.snap_cpu.numpy()
        ST.snap_event = torch.cuda.Event()
    ST.snap_cpu.copy_(nc, non_blocking=True)
    ST.snap_event.record()
    ST.snap_valid = True
    ST.fresh.clear()
    STATS["snapshots"] += 1


def _stash(runner, input_batch) -> None:
    import numpy as np
    if getattr(runner, "adaptive_verification", None) is not None or getattr(runner, "pcp_manager", None) is not None:
        _log_once("ctx-unsupported", logging.WARNING, "[glm53-hostloop] adaptive verification / PCP active: fast "
                  "path not used (production's sync path runs)")
        return
    n = int(input_batch.num_reqs)
    idx = np.asarray(input_batch.idx_mapping_np)[:n].astype(np.int64, copy=True)
    fresh = np.fromiter((int(i) in ST.fresh for i in idx), dtype=bool, count=n)
    ST.ctx = _Ctx(seq_lens_ptr=int(input_batch.seq_lens.data_ptr()), num_reqs=n, idx_np=idx,
                  qsl_np=np.asarray(input_batch.query_start_loc_np)[: n + 1].astype(np.int64, copy=True),
                  nc_upper=np.asarray(input_batch.num_computed_tokens_np)[:n].astype(np.int64, copy=True),
                  fresh=fresh)


def exact_seq_lens(ctx, cam):
    """seq_lens of this step on the host (torch int32 [cam.num_reqs], padding 0), or None (reason counted)."""
    import numpy as np
    import torch
    num_reqs = int(cam.num_reqs)
    n = ctx.num_reqs
    if int(cam.seq_lens.data_ptr()) != ctx.seq_lens_ptr or num_reqs < n:
        STATS["fallback_mismatch_ctx"] += 1
        return None
    qsl = cam.query_start_loc_cpu[: num_reqs + 1].numpy().astype(np.int64)
    if qsl.shape[0] != num_reqs + 1 or not np.array_equal(qsl[: n + 1], ctx.qsl_np) or np.any(qsl[n:] != qsl[n]):
        STATS["fallback_mismatch_ctx"] += 1
        return None
    if not bool(ctx.fresh.all()):
        if not ST.snap_valid or ST.snap_event is None:
            STATS["fallback_no_snapshot"] += 1
            return None
        ST.snap_event.synchronize()        # step n's post_update + copy: done before the drafter starts
        nc = np.where(ctx.fresh, ctx.nc_upper, ST.snap_np[ctx.idx_np].astype(np.int64))
    else:
        nc = ctx.nc_upper
    if np.any(nc < 0) or np.any(nc > ctx.nc_upper):
        STATS["fallback_bound"] += 1
        _log_once("bound", logging.WARNING, "[glm53-hostloop] snapshot value outside [0, host upper bound] "
                  "(nc=%s upper=%s fresh=%s): production path used for that step", nc.tolist(),
                  ctx.nc_upper.tolist(), ctx.fresh.tolist())
        return None
    seq = np.zeros(num_reqs, dtype=np.int32)
    seq[:n] = nc + (ctx.qsl_np[1:] - ctx.qsl_np[:-1])
    return torch.from_numpy(seq)


def _should_verify() -> bool:
    k = STATS["fast"]
    if k < ST.verify_first:
        return True
    return ST.verify_every > 0 and (k - ST.verify_first) % ST.verify_every == 0


def make_kv_lens_host(orig):
    def _kv_lens_host(self, cam):
        t0 = time.perf_counter()
        try:
            ctx = ST.ctx
            if (not ST.enabled or ctx is None or not getattr(self, "_async_scheduling", False)
                    or getattr(cam, "positions", None) is not None):
                STATS["fallback_no_ctx"] += 1
                return orig(self, cam)
            try:
                seq_host = exact_seq_lens(ctx, cam)
            except Exception as exc:  # noqa: BLE001
                STATS["fallback_error"] += 1
                _log_once("err", logging.WARNING, "[glm53-hostloop] exact seq_lens failed (%r): production path", exc)
                seq_host = None
            if seq_host is None:
                return orig(self, cam)
            if _should_verify():
                ref = cam.seq_lens[: int(cam.num_reqs)].cpu()          # production's own sync, for comparison
                STATS["verified"] += 1
                if ref.dtype != seq_host.dtype or not bool((ref == seq_host).all()):
                    STATS["mismatch"] += 1
                    _disable(f"seq_lens mismatch host={seq_host.tolist()} device={ref.tolist()}")
                    return orig(self, cam)
            STATS["fast"] += 1
            if STATS["fast"] == 1:
                _log.info("[glm53-hostloop] rank %s: first step planned from the post-sampling snapshot (no wait "
                          "for the drafter); seq_lens=%s", _rank(), seq_host.tolist())
            return orig(self, _CamView(cam, seq_host))
        finally:
            _METER.hwait += time.perf_counter() - t0

    _kv_lens_host._glm53_hostloop = True
    _kv_lens_host._glm53_orig = orig
    _kv_lens_host.__doc__ = getattr(orig, "__doc__", None)
    return _kv_lens_host


# ------------------------------------------------------------------ meter (CUDA events; optional, independent)
class _Meter:
    """CUDA-event meter (GLM53_DEC_HOSTLOOP_METER=N). One record per real decode step:
    fwd = event at forward start, end = event after sample_tokens (after the drafter), colls = (start, end) event
    pairs around every EAGER tensor-parallel all-reduce / all-gather of the step (production decode: the target's
    embedding all-reduce and the target logits all-gather; collectives inside CUDA graphs are not visible to Python). A collective's duration on a rank = its wait for the other rank + the transfer, so the
    rank with the longer duration arrived first: comparing the two ranks' coll medians gives the skew at the target
    start (c0), at the end of the target (c1) and at the end of the drafter (c2) without any profiler."""

    def __init__(self) -> None:
        self.every = 0
        self.pending = []           # (record of step n-1, record of step n)
        self.samples = []           # (gap_ms, step_ms, hwait_ms, [coll_ms...])
        self.cur = None
        self.real_step = False
        self.hwait = 0.0
        self.pool = []

    def _evt(self):
        import torch
        return self.pool.pop() if self.pool else torch.cuda.Event(enable_timing=True)

    def _free(self, rec) -> None:
        self.pool.append(rec["fwd"])
        if rec["end"] is not None:
            self.pool.append(rec["end"])
        for a, b in rec["colls"]:
            self.pool.extend((a, b))

    def _active(self) -> bool:
        import torch
        return bool(self.every) and self.real_step and not torch.cuda.is_current_stream_capturing()

    def forward_start(self) -> None:
        if not self._active():
            return
        fwd = self._evt()
        fwd.record()
        rec = {"fwd": fwd, "end": None, "colls": [], "hwait": self.hwait}
        self.hwait = 0.0
        if self.cur is not None and self.cur["end"] is not None:
            self.pending.append((self.cur, rec))
        # a step without step_end (no sample_tokens call) is dropped, not recycled: it may still be the second
        # element of a pending pair whose events must not be re-recorded before that pair is read
        self.cur = rec
        self._drain()

    def step_end(self) -> None:
        if not self._active() or self.cur is None or self.cur["end"] is not None:
            return
        end = self._evt()
        end.record()
        self.cur["end"] = end

    def coll(self, fn, *a, **k):
        if not self._active() or self.cur is None or self.cur["end"] is not None or len(self.cur["colls"]) >= 8:
            return fn(*a, **k)
        s = self._evt()
        s.record()
        out = fn(*a, **k)
        e = self._evt()
        e.record()
        self.cur["colls"].append((s, e))
        return out

    def _drain(self) -> None:
        keep = []
        for prev, rec in self.pending:
            if rec["fwd"].query():
                gap = prev["end"].elapsed_time(rec["fwd"])
                step = prev["fwd"].elapsed_time(rec["fwd"])
                colls = [a.elapsed_time(b) for a, b in prev["colls"]]
                self.samples.append((gap, step, rec["hwait"] * 1e3, colls))
                self._free(prev)
            else:
                keep.append((prev, rec))
        if len(keep) > 64:
            for prev, _ in keep[:-64]:
                self._free(prev)
            keep = keep[-64:]
        self.pending = keep
        if len(self.samples) >= self.every:
            self._report()

    def _report(self) -> None:
        import math
        s = self.samples
        self.samples = []

        def q(vals, p):
            v = sorted(x for x in vals if not math.isnan(x))
            return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
        parts = []
        for i, name in enumerate(("gap_ms", "step_ms", "hwait_ms")):
            vals = [x[i] for x in s]
            parts.append("%s med %.3f p90 %.3f max %.3f" % (name, q(vals, 0.5), q(vals, 0.9), q(vals, 1.0)))
        ncoll = max((len(x[3]) for x in s), default=0)
        for j in range(ncoll):
            vals = [x[3][j] for x in s if len(x[3]) > j]
            parts.append("c%d_ms med %.3f p90 %.3f (n=%d)" % (j, q(vals, 0.5), q(vals, 0.9), len(vals)))
        _log.info("[glm53-hostloop] meter rank %s (%d steps, hostloop=%s): %s; stats=%s", _rank(), len(s),
                  "on" if ST.enabled else ("off" if ST.disabled_reason is None else "switched-off"),
                  "; ".join(parts), STATS)


_METER = _Meter()


# ------------------------------------------------------------------ wake (GLM53_DEC_HOSTLOOP_WAKE; independent)
_SPIN_CODE = r"""
import ctypes, os, sys, time
cpu = int(sys.argv[1]); parent = int(sys.argv[2]); tick = float(sys.argv[3]) * 1e-6 if len(sys.argv) > 3 else 0.0
try:
    ctypes.CDLL(None, use_errno=True).prctl(1, 9, 0, 0, 0)      # PR_SET_PDEATHSIG = SIGKILL
except Exception:
    pass
if os.getppid() != parent:
    sys.exit(0)
os.sched_setaffinity(0, {cpu})
try:
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
except Exception:
    os.nice(19)
if tick > 0:
    while True:               # a short timer keeps the core in shallow idle (WFI) at a few % of one core
        time.sleep(tick)
while True:
    pass
"""


_TICK_CODE = r"""
import ctypes, os, sys, threading, time
parent = int(sys.argv[1]); tick = float(sys.argv[2]) * 1e-6; cpus = [int(c) for c in sys.argv[3].split(",")]
try:
    ctypes.CDLL(None, use_errno=True).prctl(1, 9, 0, 0, 0)      # PR_SET_PDEATHSIG = SIGKILL
except Exception:
    pass
if os.getppid() != parent:
    sys.exit(0)


def tick_on(cpu):
    os.sched_setaffinity(0, {cpu})                           # per thread on Linux
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except Exception:
        pass
    while True:              # a short timer keeps the core in shallow idle (WFI) at ~1% of one core
        time.sleep(tick)


for c in cpus:
    threading.Thread(target=tick_on, args=(c,), daemon=True).start()
while True:
    time.sleep(3600)
"""


class _Wake:
    """GLM53_DEC_HOSTLOOP_WAKE=<cpu list>|auto. NCCL captures a CUDA host node (hostStreamPlanCallback) in front of
    the network collectives of every CUDA graph; the first one of each graph launch wakes the CUDA driver's
    host-callback thread from a sleeping core. On GB10 the deepest CPU idle state (LPI-3) has a 433 us exit latency,
    and a cold host node measured 300-520 us median / 600-900 us p90 (tests/probe_hostnode4.py; 78 us with no idle
    core, 126 us with the two GPU-interrupt CPUs kept awake and the callback thread pinned to one of them). In the
    R15 decode trace the drafter graph's first all-reduce waits ~0.5 ms for it every step and the target graph's first
    one ~0.1-0.3 ms. This keeps the listed CPUs out of deep idle states with a SCHED_IDLE thread per CPU that wakes
    every GLM53_DEC_HOSTLOOP_WAKE_TICK_US (default 100 us; ~1% of a core each, one helper process; 0 = busy
    spinners, one process per CPU) and pins the host-callback thread to the first listed CPU. No numerics, no GPU memory,
    no collective change; the processes die with the worker (PR_SET_PDEATHSIG). 'auto' = the effective CPUs of all
    nvidia MSI-X vectors (/proc/irq/*/nvidia; /proc/interrupts is masked in containers, so an explicit list of the
    ACTIVE vectors' CPUs from the host's /proc/interrupts is better: nodeC = 12,0)."""

    def __init__(self) -> None:
        self.raw = ""
        self.cpus: list[int] = []
        self.started = False
        self.failed = False
        self.procs = []
        self.cb = None
        self.cb_tid = 0
        self.pinned_tid = 0
        self.stream = None
        self.steps = 0
        self.tick_us = 100
        self.pin = True
        self.cb_aff = None               # the callback thread's affinity before pinning (restored on failure)


_WAKE = _Wake()


def _parse_cpus(txt: str) -> list[int]:
    out = []
    for part in txt.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def wake_cpus(raw: str) -> list[int]:
    if raw.strip().lower() == "auto":
        cpus: list[int] = []
        for d in sorted(os.listdir("/proc/irq"), key=lambda x: int(x) if x.isdigit() else 1 << 30):
            if d.isdigit() and os.path.isdir(f"/proc/irq/{d}/nvidia"):
                try:
                    for c in _parse_cpus(open(f"/proc/irq/{d}/effective_affinity_list").read().strip()):
                        if c not in cpus:
                            cpus.append(c)
                except OSError:
                    pass
        return cpus
    cpus = []
    for c in _parse_cpus(raw):
        if c not in cpus:
            cpus.append(c)
    return cpus


def _cudart():
    import ctypes
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if "libcudart.so" in p:
            return ctypes.CDLL(p)
    raise RuntimeError("libcudart is not mapped")


def _wake_stop() -> None:
    for p in _WAKE.procs:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    _WAKE.procs = []


def _wake_start() -> None:
    import atexit
    import ctypes
    import subprocess
    import threading
    import torch
    W = _WAKE
    W.started = True
    avail = os.sched_getaffinity(0)
    W.cpus = [c for c in wake_cpus(W.raw) if c in avail] if W.raw else []
    if not W.cpus:
        raise RuntimeError(f"no usable CPU in {ENV_WAKE}={W.raw!r} (affinity {sorted(avail)})")
    io = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    if W.tick_us > 0:            # one process, one ticker thread per CPU (~10 MB RSS in total)
        W.procs.append(subprocess.Popen([sys.executable, "-S", "-c", _TICK_CODE, str(os.getpid()), str(W.tick_us),
                                         ",".join(str(c) for c in W.cpus)], **io))
    else:                        # busy spinners need one process per CPU (the GIL)
        for c in W.cpus:
            W.procs.append(subprocess.Popen([sys.executable, "-S", "-c", _SPIN_CODE, str(c), str(os.getpid())],
                                            **io))
    atexit.register(_wake_stop)
    # find the CUDA driver's host-callback thread: a no-op host function on a private stream nobody waits for
    rt = _cudart()
    rt.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    rt.cudaLaunchHostFunc.restype = ctypes.c_int

    def _probe(_):
        W.cb_tid = threading.get_native_id()
    W.cb = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_probe)
    W.stream = torch.cuda.Stream()
    rc = rt.cudaLaunchHostFunc(ctypes.c_void_p(W.stream.cuda_stream), ctypes.cast(W.cb, ctypes.c_void_p), None)
    if rc != 0:
        raise RuntimeError(f"cudaLaunchHostFunc returned {rc}")
    _log.info("[glm53-hostloop] rank %s: wake: SCHED_IDLE %s on cpu %s (pids %s); host-callback thread %s",
              _rank(), f"{W.tick_us} us tickers" if W.tick_us > 0 else "busy spinners", W.cpus,
              [p.pid for p in W.procs], f"will be pinned to cpu {W.cpus[0]}" if W.pin else "not pinned (WAKE_PIN=0)")


def _wake_step() -> None:
    """Called once per real decode step (prepare_inputs) while GLM53_DEC_HOSTLOOP_WAKE is set. Cheap."""
    import torch
    W = _WAKE
    if W.failed or torch.cuda.is_current_stream_capturing():
        return
    try:
        if not W.started:
            _wake_start()
            return
        W.steps += 1
        if W.pin and W.cb_tid and W.cb_tid != W.pinned_tid:
            W.cb_aff = os.sched_getaffinity(W.cb_tid)
            os.sched_setaffinity(W.cb_tid, {W.cpus[0]})
            W.pinned_tid = W.cb_tid
            _log.info("[glm53-hostloop] rank %s: wake: host-callback thread %d pinned to cpu %d", _rank(), W.cb_tid,
                      W.cpus[0])
        dead = [p.pid for p in W.procs if p.poll() is not None]
        if dead:
            raise RuntimeError(f"spinner(s) {dead} exited")
    except Exception as exc:  # noqa: BLE001
        W.failed = True
        _wake_stop()
        # without the tickers a callback thread pinned to one CPU can only be slower than production's unpinned one:
        # give it back the affinity it had before pinning
        if W.pinned_tid and W.cb_aff:
            try:
                os.sched_setaffinity(W.pinned_tid, W.cb_aff)
            except Exception:  # noqa: BLE001
                pass
            W.pinned_tid = 0
        _log.warning("[glm53-hostloop] rank %s: wake switched off (%r); callback thread unpinned, nothing else "
                     "changes", _rank(), exc)


# ------------------------------------------------------------------ patching
def _verify_sources(mr_mod, sm90_mod) -> tuple[bool, str, dict]:
    from vllm.v1.worker.gpu import input_batch as ib_mod
    from vllm.v1.worker.gpu import states as states_mod
    R = mr_mod.GPUModelRunner
    B = sm90_mod.FlashInferMLASparseSM90Builder
    fns = {
        "FlashInferMLASparseSM90Builder._kv_lens_host": getattr(B._kv_lens_host, "_glm53_orig", B._kv_lens_host),
        "GPUModelRunner.prepare_inputs": getattr(R.prepare_inputs, "_glm53_orig", R.prepare_inputs),
        "GPUModelRunner.postprocess_sampled": getattr(R.postprocess_sampled, "_glm53_orig", R.postprocess_sampled),
        "GPUModelRunner.add_requests": getattr(R.add_requests, "_glm53_orig", R.add_requests),
        "GPUModelRunner.execute_model": getattr(R.execute_model, "_glm53_orig", R.execute_model),
        "RequestState.add_request": states_mod.RequestState.add_request,
        "_prepare_pos_seq_lens_kernel": ib_mod._prepare_pos_seq_lens_kernel,
    }
    got = {k: source_fingerprint(f) for k, f in fns.items()}
    bad = {k: v for k, v in got.items() if v not in VERIFIED_FINGERPRINTS[k]}
    if bad:
        return False, "unverified vLLM source: " + ", ".join(f"{k}={v}" for k, v in sorted(bad.items())), got
    return True, "ok", got


def _wrap(cls, name, make):
    cur = getattr(cls, name)
    if getattr(cur, "_glm53_hostloop", False):
        return
    new = make(cur)
    new._glm53_hostloop = True
    new._glm53_orig = cur
    new.__name__ = getattr(cur, "__name__", name)
    new.__doc__ = getattr(cur, "__doc__", None)
    setattr(cls, name, new)


def patch_model_runner(mr_mod, fast: bool, meter: bool) -> None:
    R = mr_mod.GPUModelRunner

    if fast:
        def mk_post(orig):
            def postprocess_sampled(self, *a, **k):
                r = orig(self, *a, **k)
                if ST.enabled:
                    try:
                        _snapshot(self)
                    except Exception as exc:  # noqa: BLE001
                        _disable(f"snapshot failed: {exc!r}")
                return r
            return postprocess_sampled

        def mk_add(orig):
            def add_requests(self, scheduler_output, *a, **k):
                r = orig(self, scheduler_output, *a, **k)
                if ST.enabled:
                    try:
                        for nr in scheduler_output.scheduled_new_reqs:
                            i = self.req_states.req_id_to_index.get(nr.req_id)
                            if i is not None:
                                ST.fresh.add(int(i))
                    except Exception as exc:  # noqa: BLE001
                        _disable(f"add_requests bookkeeping failed: {exc!r}")
                return r
            return add_requests

        _wrap(R, "postprocess_sampled", mk_post)
        _wrap(R, "add_requests", mk_add)

    def mk_prep(orig):
        def prepare_inputs(self, *a, **k):
            ST.ctx = None
            ib = orig(self, *a, **k)
            _METER.real_step = True
            if _WAKE.raw:
                _wake_step()
            if ST.enabled:
                try:
                    _stash(self, ib)
                except Exception as exc:  # noqa: BLE001
                    ST.ctx = None
                    _disable(f"stash failed: {exc!r}")
            return ib
        return prepare_inputs

    def mk_exec(orig):
        def execute_model(self, *a, **k):
            _METER.real_step = False
            try:
                return orig(self, *a, **k)
            finally:
                ST.ctx = None
        return execute_model

    _wrap(R, "prepare_inputs", mk_prep)
    _wrap(R, "execute_model", mk_exec)

    if meter:
        def mk_sample(orig):
            def sample_tokens(self, *a, **k):
                r = orig(self, *a, **k)
                try:
                    _METER.step_end()
                except Exception as exc:  # noqa: BLE001
                    _METER.every = 0
                    _log.warning("[glm53-hostloop] meter off: %r", exc)
                return r
            return sample_tokens
        _wrap(R, "sample_tokens", mk_sample)
        from vllm.v1.worker.gpu import async_utils
        C = async_utils.StepTimingCollector

        def mk_fwd(orig):
            def forward_start(self, *a, **k):
                r = orig(self, *a, **k)
                try:
                    _METER.forward_start()
                except Exception as exc:  # noqa: BLE001
                    _METER.every = 0
                    _log.warning("[glm53-hostloop] meter off: %r", exc)
                return r
            return forward_start
        _wrap(C, "forward_start", mk_fwd)
        from vllm.distributed import parallel_state
        G = parallel_state.GroupCoordinator

        def mk_coll(orig):
            def coll(self, *a, **k):
                return _METER.coll(orig, self, *a, **k)
            return coll
        _wrap(G, "_all_reduce_out_place", mk_coll)
        _wrap(G, "_all_gather_out_place", mk_coll)


def install_now(mr_mod=None, sm90_mod=None, *, fast: bool | None = None, meter_every: int | None = None,
                wake: str | None = None) -> dict:
    """Patch the (imported) model runner / sm90 modules. Returns a report; never raises."""
    report = {"fast": False, "meter": 0, "wake": "", "reason": None, "fingerprints": None}
    try:
        fast = _on(ENV) if fast is None else fast
        meter_every = _int_env(ENV_METER, 0) if meter_every is None else meter_every
        wake = (os.environ.get(ENV_WAKE, "") if wake is None else wake).strip()
        if wake.lower() in _OFF:
            wake = ""
        if wake:
            try:
                if not wake_cpus(wake):
                    raise ValueError("empty cpu list")
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-hostloop] %s=%r unusable (%r); wake off", ENV_WAKE, wake, exc)
                wake = ""
        _WAKE.raw = wake
        _WAKE.tick_us = _int_env(ENV_WAKE_TICK, 100)
        # default 1 also for an EMPTY value: the launcher passes every listed knob, unset ones as "" (deploy-r16)
        _WAKE.pin = (os.environ.get(ENV_WAKE_PIN, "").strip().lower() or "1") not in _OFF
        report["wake"] = wake
        if not fast and not meter_every and not wake:
            report["reason"] = "off"
            return report
        if mr_mod is None:
            mr_mod = importlib.import_module(MR_MODULE)
        if fast and sm90_mod is None:
            sm90_mod = importlib.import_module(SM90_MODULE)
        if fast:
            ok, why, fps = _verify_sources(mr_mod, sm90_mod)
            report["fingerprints"] = fps
            if not ok:
                _log.warning("[glm53-hostloop] %s NOT installed: %s (production path unchanged)", ENV, why)
                report["reason"] = why
                fast = False
        ST.verify_first = _int_env(ENV_VERIFY, 64)
        ST.verify_every = _int_env(ENV_VERIFY_EVERY, 1024)
        _METER.every = meter_every
        if not fast and not meter_every and not wake:
            report["fast"] = False
            return report
        patch_model_runner(mr_mod, fast=fast, meter=bool(meter_every))
        if fast:
            B = sm90_mod.FlashInferMLASparseSM90Builder
            if not getattr(B._kv_lens_host, "_glm53_hostloop", False):
                B._kv_lens_host = make_kv_lens_host(B._kv_lens_host)
            ST.enabled = True
        report.update(fast=fast, meter=meter_every, reason=report["reason"] or "ok")
        _log.info("[glm53-hostloop] rank %s pid %d: fast path %s (verify first %d, then 1/%d), meter %s, wake %s",
                  _rank(), os.getpid(), "ON" if fast else "off", ST.verify_first, ST.verify_every,
                  f"every {meter_every} steps" if meter_every else "off", wake or "off")
    except Exception as exc:  # noqa: BLE001
        ST.enabled = False
        report["reason"] = f"install failed: {exc!r}"
        _log.warning("[glm53-hostloop] install failed (production path unchanged): %r", exc)
    return report


class _Finder(importlib.abc.MetaPathFinder):
    """Patch the V2 model runner when a worker process imports it (the API server / engine core never do)."""

    def find_spec(self, name, path, target=None):
        if name != MR_MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            install_now(mr_mod=module)
        spec.loader.exec_module = exec_module
        return spec


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless an env var is set. Never raises."""
    try:
        if not (_on(ENV) or _int_env(ENV_METER, 0) or os.environ.get(ENV_WAKE, "").strip().lower() not in _OFF):
            return
        if MR_MODULE in sys.modules:
            install_now(mr_mod=sys.modules[MR_MODULE])
        elif not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-hostloop] plugin install failed (production path unchanged): %r", exc)
