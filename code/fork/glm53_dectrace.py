"""GLM53_DEC_TRACE: per-step host timestamp tracer + GPU-timed eager collectives, for the decode gap and the rank
skew (docs/DEC_HOSTGAP.md §3). Measurement only — no numeric, no collective, no scheduling change.

Why: one decode step is ~70 ms; a torch.profiler trace explains its idle time (docs/DEC_HOSTLOOP.md §1) but is far
too heavy for production. This tracer stamps the per-step boundaries that are reachable from Python on BOTH ranks
into a bounded buffer flushed to one file per rank, and times the step's EAGER collectives on the GPU; the offline
script tests/align_dectrace.py pairs the two ranks' files step by step.

What is a shared clock and what is not (review 2026-09-30, docs/DEC_HOSTGAP.md §3.1). An eager collective's HOST
call returns as soon as NCCL has enqueued the kernel on the stream (vLLM pynccl.all_reduce -> ncclAllReduce(stream)):
it does NOT wait for the other rank. Measured on the production profile p6h (tests/review_eager_coll_host_vs_gpu.py,
docs/logs/hostgap/review_eager_coll_host_vs_gpu.log): the eager all-reduce's host call returns a median 127 us (max
6.6 ms) before its kernel ENDS, the eager all-gather's a median 58 ms before its kernel even STARTS (async
scheduling: the host is a step ahead). Host return times are therefore NOT a common clock, and nothing host-side is.
What is exact without any clock: a collective's GPU duration on each rank = its wait for the other rank + the
transfer, and both ranks finish it together (end-time residual 7.7 us median on the 01:00 two-rank profiler pair,
docs/logs/hostloop/skew_0100_pair.log), so per step  arrival skew (r1 later) = dur_r0 - dur_r1  and
wait_r = dur_r - min(dur_r0, dur_r1). "Arrival" = when that rank's GPU reached the collective's slot on the stream;
if a GPU reaches it before its host has launched the kernel (a host-bound step), the duration also contains that
host launch latency (a few us: the begin event is recorded right before the NCCL call).
The tracer records those durations with CUDA events (`gar` / `gag` records); the host events give each rank's own
host windows. Cross-rank HOST skew needs a clock offset from outside (the `#! clock` wall-clock pairs = NTP accuracy,
or an offset the operator supplies); the aligner never derives one from collective returns.

Record lines (one per event, host clock = time.perf_counter_ns of the process, n = the process's execute_model index):
  "n t ev"        host events:
    sched   execute_model entered (the schedule is in hand; with async scheduling it arrived earlier)
    prep    prepare_inputs returned (the per-step input build, incl. the planner's kv_lens path)
    fg_b/fg_e  the target's FULL cudagraph replay launched / returned (production decode: the target runs here)
    pw_b/pw_e  the target's PIECEWISE run (ModelCudaGraphManager.run_pw_graph) entered / returned. Production auto-
               enables the breakable cudagraph (VLLM_USE_BREAKABLE_CUDAGRAPH), whose replay does NOT call the model's
               forward either: without this pair a PIECEWISE step (small prefill chunk, a batch that is not a
               captured FULL shape) had no target event at all (verification 2026-09-30: 4 of 33 steps of the
               review's own engine trace)
    tgt_b/tgt_e  target model forward entered / returned (eager forwards: prefill beyond the capture sizes,
                 warm-up, the breakable capture itself; nested inside pw_b/pw_e when a PIECEWISE run calls it)
    sam_b/sam_e  sample_tokens entered / returned (sampling; the drafter's propose runs inside it)
    rs_b/rs_e    RejectionSampler.forward (a tree where verification goes through it; not DFlash2's MRv2 path)
    dr_b/dr_e    the drafter's propose (speculator.propose; the V1 runner's propose_draft_token_ids too)
    ar_b/ar_e    eager all-reduce call entered / returned (GroupCoordinator._all_reduce_out_place): HOST enqueue only
    ag_b/ag_e    eager all-gather call entered / returned (GroupCoordinator._all_gather_out_place): HOST enqueue only
    exec_e  execute_model returned
  "n t ev dur_ns" GPU records (GLM53_DEC_TRACE_GPU, default on): ev = gar / gag, t = the host ar_b / ag_b time of that
                  collective (its order in the step = its position), dur_ns = CUDA-event time from just before to just
                  after the collective on the stream NCCL uses (= that rank's wait + transfer). At most 8 per step,
                  never during CUDA-graph capture; resolved lazily when the events have completed (no sync).
  "#! rank R", "#! clock <perf_counter_ns> <time_ns>" (at open and every flush), "#! stopped ..."
Collectives captured inside CUDA graphs are invisible here (not reachable from Python): production decode has 2 eager
collectives per step (target embedding all-reduce, target logits all-gather) and 102 in-graph ones.

Knobs (read once at install; everything unset = inert, the wrappers cost one attribute test):
  GLM53_DEC_TRACE=<dir>          write <dir>/dectrace-rank<R>-pid<PID>.ndjson (renamed once the TP rank is known;
                                 the "#! rank" line inside is authoritative). File is truncated.
  GLM53_DEC_TRACE_RING=<n>       in-memory bound in records (default 4096): flushed when reached
  GLM53_DEC_TRACE_FLUSH=<n>      also flush every n execute_model calls (default 64)
  GLM53_DEC_TRACE_MAX_STEPS=<n>  stop (close the file, one INFO line) after n steps (default 20000), 0 = unlimited.
                                 Bounds the file, not only the memory.
  GLM53_DEC_TRACE_EVENTS=<csv>   keep only these event names ("-name" drops one; default all)
  GLM53_DEC_TRACE_GPU=0          no CUDA events (host events only); default on

Both ranks must get the same values. Install: integrate.plugin_register -> plugin_install (after glm53_hostloop,
whose fingerprint check must see the unwrapped model-runner methods: tests/test_dectrace.py D3), inert when unset;
a failed install only logs.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import json
import logging
import os
import socket
import sys
import time

_log = logging.getLogger(__name__)

ENV = "GLM53_DEC_TRACE"
ENV_RING = "GLM53_DEC_TRACE_RING"
ENV_FLUSH = "GLM53_DEC_TRACE_FLUSH"
ENV_MAX_STEPS = "GLM53_DEC_TRACE_MAX_STEPS"
ENV_EVENTS = "GLM53_DEC_TRACE_EVENTS"
ENV_GPU = "GLM53_DEC_TRACE_GPU"
OFF = ("", "0", "off", "false", "no")
MR_MODULE = "vllm.v1.worker.gpu.model_runner"
GPU_PER_STEP = 8          # GPU-timed eager collectives per step (production decode has 2)
GPU_PENDING_MAX = 512     # unresolved event pairs kept; older ones are dropped (never re-recorded while in flight)
GPU_DRAIN_AT = 64         # step_done resolves the completed pairs once this many are pending (any FLUSH cadence)

EVENTS = ("sched", "prep", "fg_b", "fg_e", "pw_b", "pw_e", "tgt_b", "tgt_e", "sam_b", "sam_e", "rs_b", "rs_e",
          "dr_b", "dr_e", "ar_b", "ar_e", "ag_b", "ag_e", "exec_e", "gar", "gag")

_clock = time.perf_counter_ns


class _Tracer:
    """Bounded record buffer -> one file per rank. A flush is one write() of the whole chunk."""

    __slots__ = ("enabled", "path", "fh", "buf", "cap", "flush_steps", "max_steps", "on",
                 "step", "last_flush", "n_written", "steps_flushed", "rank", "kept",
                 "model_seen", "spec_seen", "gpu", "gpend", "gpool", "gcount", "gdropped", "torch", "stream")

    def __init__(self) -> None:
        self.enabled = False
        self.on = False                     # hot-path flag: enabled AND the file is open
        self.path = ""
        self.fh = None
        self.buf: list[str] = []
        self.cap = 4096
        self.flush_steps = 64
        self.max_steps = 20000
        self.step = 0
        self.last_flush = 0
        self.n_written = 0
        self.steps_flushed = 0
        self.rank = -1
        self.kept = frozenset(EVENTS)
        self.model_seen = False
        self.spec_seen = False
        self.gpu = False                    # GPU timing of the eager collectives (CUDA events)
        self.gpend: list = []               # (step, t_host_begin, ev, ev_begin, ev_end), FIFO
        self.gpool: list = []
        self.gcount = 0                     # GPU-timed collectives in the current step
        self.gdropped = 0
        self.torch = None
        self.stream = None                  # () -> the stream NCCL enqueues on (vLLM's current_stream)

    # ---- file -------------------------------------------------------------
    def open_file(self, directory: str) -> bool:
        try:
            os.makedirs(directory, exist_ok=True)
            self.path = os.path.join(directory, f"dectrace-rank{self.rank}-pid{os.getpid()}.ndjson")
            fh = open(self.path, "w", buffering=1 << 16)
            hdr = json.dumps({
                "ver": 2, "pid": os.getpid(), "boot_epoch_ns": time.time_ns(), "boot_pc_ns": _clock(),
                "clock": "time.perf_counter_ns", "host": socket.gethostname(),
                "ring": self.cap, "flush_steps": self.flush_steps, "max_steps": self.max_steps,
                "gpu": self.gpu, "cmd": sys.argv[-1][:200] if sys.argv else "",
            }, ensure_ascii=True)
            fh.write("#!glm53-dectrace " + hdr + "\n")
            self.fh = fh
            self.on = self.enabled
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("[glm53-dectrace] cannot open a trace file under %r: %r; tracer off",
                         directory, exc)
            self.fh = None
            self.on = False
            return False

    def set_rank(self, rank: int) -> None:
        """Once the rank is knowable (parallel_state init): one '#! rank' line, and rename the file."""
        if rank is None or rank < 0 or self.rank == rank:
            return
        self.rank = rank
        if self.fh is not None:
            try:
                self.fh.write(f"#! rank {rank}\n")
            except Exception:  # noqa: BLE001
                pass
            new = os.path.join(os.path.dirname(self.path), f"dectrace-rank{rank}-pid{os.getpid()}.ndjson")
            try:
                if new != self.path:
                    os.rename(self.path, new)   # fine while open (POSIX); the fh keeps working
                    self.path = new
            except OSError:
                pass

    # ---- hot path ---------------------------------------------------------
    def add(self, ev: str) -> int:
        t = _clock()
        if ev in self.kept:
            self.buf.append("%d %d %s" % (self.step, t, ev))
        return t

    # ---- GPU timing of the eager collectives --------------------------------
    def gpu_begin(self):
        """Record the 'before' event on the collective's stream; None = not timed (off / capture / budget)."""
        if self.gcount >= GPU_PER_STEP:
            return None
        try:
            if self.torch.cuda.is_current_stream_capturing():
                return None
            s = self.stream()
            eb = self.gpool.pop() if self.gpool else self.torch.cuda.Event(enable_timing=True)
            ee = self.gpool.pop() if self.gpool else self.torch.cuda.Event(enable_timing=True)
            eb.record(s)
            self.gcount += 1
            return (eb, ee, s)
        except Exception as exc:  # noqa: BLE001
            self.gpu_off(exc)
            return None

    def gpu_end(self, g, ev: str, t_begin: int, step: int) -> None:
        eb, ee, s = g
        try:
            ee.record(s)
        except Exception as exc:  # noqa: BLE001
            self.gpu_off(exc)
            return
        self.gpend.append((step, t_begin, ev, eb, ee))
        if len(self.gpend) > GPU_PENDING_MAX:          # never re-record an event that may still be in flight: drop
            drop = len(self.gpend) - GPU_PENDING_MAX
            self.gpend = self.gpend[drop:]
            self.gdropped += drop

    def gpu_drain(self) -> None:
        """Resolve the completed event pairs (query only, never a sync) into 'n t gar|gag dur_ns' records."""
        if not self.gpend:
            return
        keep = []
        try:
            for rec in self.gpend:
                step, tb, ev, eb, ee = rec
                if ee.query():
                    dur_ns = int(eb.elapsed_time(ee) * 1e6)
                    if ev in self.kept:
                        self.buf.append("%d %d %s %d" % (step, tb, ev, dur_ns))
                    self.gpool.append(eb)
                    self.gpool.append(ee)
                else:
                    keep.append(rec)
        except Exception as exc:  # noqa: BLE001
            self.gpu_off(exc)
            keep = []
        self.gpend = keep

    def gpu_off(self, exc) -> None:
        if self.gpu:
            _log.warning("[glm53-dectrace] GPU timing of the eager collectives switched off (%r); host events "
                         "continue", exc)
        self.gpu = False
        self.gpend = []

    # ---- flush / stop -----------------------------------------------------
    def flush(self, force: bool = False) -> None:
        fh = self.fh
        if fh is None:
            self.buf.clear()
            return
        if self.gpu:
            self.gpu_drain()
        if self.buf:
            chunk = "\n".join(self.buf)
            n = len(self.buf)
            self.buf.clear()
            try:
                fh.write(chunk)
                fh.write("\n#! clock %d %d\n" % (_clock(), time.time_ns()))
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-dectrace] write failed (%r); tracer off", exc)
                self.fh = None
                self.on = False
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
                return
            self.n_written += n
            self.steps_flushed = self.step
            fh.flush()
        elif force and fh is not None:
            try:
                fh.flush()
            except Exception:  # noqa: BLE001
                pass

    def step_done(self) -> None:
        """execute_model exit: the flush cadence and the max-steps bound."""
        if self.gpu and len(self.gpend) >= GPU_DRAIN_AT:   # resolve completed pairs before the pending bound drops any
            self.gpu_drain()
        if len(self.buf) >= self.cap or (self.step - self.last_flush) >= self.flush_steps:
            self.flush()
            self.last_flush = self.step
        if self.max_steps and self.step >= self.max_steps:
            self.stop("max steps reached")

    def stop(self, reason: str) -> None:
        self.flush(True)
        fh, self.fh = self.fh, None
        self.on = False
        self.enabled = False
        if fh is not None:
            try:
                fh.write(f"#! stopped {reason} at step {self.step} after {self.n_written} records"
                         f"{f', {self.gdropped} GPU pairs dropped' if self.gdropped else ''}\n")
                fh.close()
            except Exception:  # noqa: BLE001
                pass
            _log.info("[glm53-dectrace] stopped (%s): %d steps, %d records -> %s", reason, self.step,
                      self.n_written, self.path)
        self.gpend = []


_T = _Tracer()


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in OFF


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _resolve_rank() -> int:
    """TP rank of this process, once it is knowable (-1 = not yet; retried per step)."""
    try:
        from vllm.distributed import parallel_state as ps
        ready = getattr(ps, "model_parallel_is_initialized", None) or \
            getattr(ps, "tensor_model_parallel_is_initialized", None)
        if ready is not None and ready():
            return int(ps.get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    for k in ("RANK", "LOCAL_RANK"):
        try:
            v = int(os.environ.get(k, "-1"))
            if v >= 0:
                return v
        except ValueError:
            pass
    return -1


def _wrap(cls, name, make):
    cur = getattr(cls, name, None)
    if cur is None or getattr(cur, "_glm53_dectrace", False):
        return False
    new = make(cur)
    new._glm53_dectrace = True
    new._glm53_dectrace_orig = cur
    new.__name__ = getattr(cur, "__name__", name)
    setattr(cls, name, new)
    return True


def _around(orig, begin: str, end: str):
    def wrapper(*a, **k):
        if not _T.on:
            return orig(*a, **k)
        _T.add(begin)
        try:
            return orig(*a, **k)
        finally:
            if _T.on:
                _T.add(end)
    return wrapper


def _coll(orig, begin: str, end: str, gev: str):
    """An eager collective: host enter/return + (GLM53_DEC_TRACE_GPU) its GPU duration from CUDA events."""
    def wrapper(*a, **k):
        if not _T.on:
            return orig(*a, **k)
        step = _T.step
        tb = _T.add(begin)
        g = _T.gpu_begin() if _T.gpu else None
        try:
            return orig(*a, **k)
        finally:
            if g is not None:
                _T.gpu_end(g, gev, tb, step)
            if _T.on:
                _T.add(end)
    return wrapper


def _after(orig, ev: str):
    def wrapper(*a, **k):
        r = orig(*a, **k)
        if _T.on:
            _T.add(ev)
        return r
    return wrapper


# ------------------------------------------------------------------ model-runner hooks
def patch_model_runner(mr_mod) -> int:
    R = mr_mod.GPUModelRunner
    n = 0

    def mk_exec(orig):
        def execute_model(self, *a, **k):
            if not _T.on:
                return orig(self, *a, **k)
            _step_guard(self)
            _T.add("sched")
            try:
                return orig(self, *a, **k)
            finally:
                if _T.on:
                    _T.add("exec_e")
                    _T.step_done()
        return execute_model

    if _wrap(R, "execute_model", mk_exec):
        n += 1
    if _wrap(R, "prepare_inputs", lambda orig: _after(orig, "prep")):
        n += 1
    if _wrap(R, "propose_draft_token_ids", lambda orig: _around(orig, "dr_b", "dr_e")):
        n += 1
    if _wrap(R, "sample_tokens", lambda orig: _around(orig, "sam_b", "sam_e")):
        n += 1
    if _wrap(R, "load_model", _make_load()):
        n += 1
    # the target's FULL-cudagraph replay: production decode never calls the model's forward (tgt_* stay silent there)
    M = getattr(mr_mod, "ModelCudaGraphManager", None)
    if M is not None and _wrap(M, "run_fullgraph", lambda orig: _around(orig, "fg_b", "fg_e")):
        n += 1
    # the target's PIECEWISE run: with the breakable cudagraph (auto-enabled in production) its replay does not call
    # the model's forward either. run_pw_graph is inherited from CudaGraphManager: setting it on ModelCudaGraphManager
    # wraps the target's manager only (the drafters' managers subclass CudaGraphManager directly and stay untouched)
    if M is not None and _wrap(M, "run_pw_graph", lambda orig: _around(orig, "pw_b", "pw_e")):
        n += 1
    return n       # the drafter itself is wrapped per instance, at the first step (see _step_guard)


def _make_load():
    def make(orig):
        def load_model(self, *a, **k):
            r = orig(self, *a, **k)
            if _T.on and _wrap_target_forward(self):
                _T.model_seen = True
            return r
        return load_model
    return make


def _step_guard(runner) -> None:
    """Once per execute_model enter: the step index, the one-time target-forward wrap and the rank lookup."""
    _T.step += 1
    _T.gcount = 0
    if _T.rank < 0:
        r = _resolve_rank()
        if r >= 0:
            _T.set_rank(r)
    if not _T.model_seen and getattr(runner, "model", None) is not None:
        _T.model_seen = _wrap_target_forward(runner)
    if not _T.spec_seen:
        sp = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
        if sp is not None and callable(getattr(sp, "propose", None)):
            _T.spec_seen = _wrap_speculator(sp)


def _wrap_speculator(sp) -> bool:
    """Wrap the drafter's propose (per instance: the class depends on the speculative method)."""
    if getattr(sp, "_glm53_dectrace_sp", False):
        return True
    try:
        sp._glm53_dectrace_sp = True
        sp.propose = _around(sp.propose, "dr_b", "dr_e")
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-dectrace] cannot wrap the drafter propose: %r", exc)
        return False


def _wrap_target_forward(runner) -> bool:
    """Wrap the target model's forward (module instance; the drafter's model is a different one).
    True once the instance carries the wrapper (already wrapped counts)."""
    try:
        m = runner.get_model()
    except Exception:  # noqa: BLE001
        return False
    if m is None:
        return False
    if getattr(m, "_glm53_dectrace_fwd", False):
        return True
    try:
        m._glm53_dectrace_fwd = True
        m.forward = _around(m.forward, "tgt_b", "tgt_e")
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-dectrace] cannot wrap the target model forward: %r", exc)
        return False


# ------------------------------------------------------------------ collective / sampler hooks
def patch_collectives(ps_mod, rej_mod=None) -> int:
    G = ps_mod.GroupCoordinator
    n = 0
    if _wrap(G, "_all_reduce_out_place", lambda orig: _coll(orig, "ar_b", "ar_e", "gar")):
        n += 1
    if _wrap(G, "_all_gather_out_place", lambda orig: _coll(orig, "ag_b", "ag_e", "gag")):
        n += 1
    if rej_mod is not None:
        R = getattr(rej_mod, "RejectionSampler", None)
        if R is not None and _wrap(R, "forward", lambda orig: _around(orig, "rs_b", "rs_e")):
            n += 1
    return n


def _setup_gpu() -> bool:
    """CUDA events for the eager collectives: needs torch with CUDA; records on vLLM's current stream."""
    if os.environ.get(ENV_GPU, "1").strip().lower() in OFF:
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        try:
            from vllm.utils.torch_utils import current_stream as _cs
        except Exception:  # noqa: BLE001
            _cs = torch.cuda.current_stream
        _T.torch = torch
        _T.stream = _cs
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-dectrace] no GPU timing (%r); host events only", exc)
        return False


# ------------------------------------------------------------------ install
def install_now(mr_mod=None, ps_mod=None, rej_mod=None, directory: str | None = None) -> dict:
    """Patch the (imported) modules and open this process's trace file. Returns a report; never raises."""
    report = {"on": False, "records": 0, "path": "", "hooks": 0, "gpu": False, "reason": None}
    try:
        directory = directory if directory is not None else os.environ.get(ENV, "").strip()
        if not directory or directory.lower() in OFF:
            report["reason"] = "off"
            return report
        _T.cap = max(64, _int_env(ENV_RING, 4096))
        _T.flush_steps = max(1, _int_env(ENV_FLUSH, 64))
        _T.max_steps = max(0, _int_env(ENV_MAX_STEPS, 20000))
        kept = set(EVENTS)
        for tok in os.environ.get(ENV_EVENTS, "").replace(";", ",").split(","):
            t = tok.strip()
            if not t:
                continue
            (kept.discard if t.startswith("-") else kept.add)(t[1:] if t.startswith("-") else t)
        _T.kept = frozenset(kept)
        _T.gpu = _setup_gpu()
        _T.enabled = True
        if not _T.open_file(directory):
            report["reason"] = "cannot open file"
            return report
        atexit.register(_T.stop, "exit")
        if mr_mod is None:
            mr_mod = importlib.import_module(MR_MODULE)
        if ps_mod is None:
            import vllm.distributed.parallel_state as ps_mod
        if rej_mod is None:
            try:
                import vllm.v1.sample.rejection_sampler as rej_mod
            except Exception:  # noqa: BLE001 - tree without that module: no rs_b/rs_e events
                rej_mod = None
        nh = patch_model_runner(mr_mod)
        nh += patch_collectives(ps_mod, rej_mod)
        report.update(on=_T.enabled, hooks=nh, path=_T.path, gpu=_T.gpu, reason=report["reason"] or "ok")
        _log.info("[glm53-dectrace] rank %s pid %d: tracing to %s (ring %d, flush every %d steps, "
                  "max steps %d, %d hooks, GPU-timed eager collectives %s)", _T.rank, os.getpid(), _T.path,
                  _T.cap, _T.flush_steps, _T.max_steps, nh, "on" if _T.gpu else "off")
    except Exception as exc:  # noqa: BLE001
        _T.on = False
        _T.enabled = False
        report["reason"] = f"install failed: {exc!r}"
        _log.warning("[glm53-dectrace] install failed (tracing off, nothing else touched): %r", exc)
    return report


class _Finder(importlib.abc.MetaPathFinder):
    """Patch the V2 model runner when a worker process imports it (API server / engine core never do)."""

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
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_DEC_TRACE is set."""
    try:
        if not _on(ENV):
            return
        if MR_MODULE in sys.modules:
            install_now(mr_mod=sys.modules[MR_MODULE])
        elif not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-dectrace] plugin install failed (tracing off): %r", exc)
