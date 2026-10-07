"""[dec-hostgap] GLM53_DEC_TRACE tracer + the two-rank aligner (docs/DEC_HOSTGAP.md §3).

Host (no torch, no GPU):
  T1 env unset -> install_now is inert (no hooks, no file);
  T2 installed -> every event carries the step index of ITS execute_model call (sched included: the first version
     stamped sched with the previous step), the target's FULL-graph replay gives fg_b/fg_e, its PIECEWISE run
     (breakable-cudagraph replay: no model forward call) gives pw_b/pw_e while a drafter's manager (a CudaGraphManager
     sibling class) stays unwrapped, the rank line is written
     and the file renamed, the target forward is wrapped once the runner has a model, no double wrapping;
  T3 the ring bounds the buffer and the file receives every chunk;
  T4 GLM53_DEC_TRACE_MAX_STEPS stops the tracer (file closed, "#! stopped" line);
  T5 GLM53_DEC_TRACE_EVENTS filters;
  T6 per-record overhead (well below 1 us);
  T7 the GPU records' bookkeeping with a fake CUDA (deterministic): one pair per eager collective, at most
     GPU_PER_STEP per step, none while the stream is capturing (host events still), resolved only once complete (no
     sync) with the exact duration, drained from step_done at GPU_DRAIN_AT pending whatever the FLUSH cadence, the
     pending list bounded (drops counted in the stop line), a failing event record switches GPU timing off without
     touching the collective's result or the host events, and a raising collective still propagates;
  A1 the aligner on a PHYSICALLY modelled two-rank pair (the host returns of the eager collectives are NOT
     synchronized, as measured on p6h; the GPU durations are wait + transfer with one shared end): per-position
     arrival skew / waits recovered exactly from the GPU records; no cross-rank host skew without an outside clock;
     --offset-us recovers the injected host skew exactly, --wallclock within the injected NTP error; negative
     control: an offset derived from the host returns (the first version's method) is off by the host skew.
Production image (tests/hostloop_gpu.sh python3 tests/test_dectrace.py --plugin --gpu):
  D1/D2 plugin wiring in fresh interpreters against the real vLLM tree: unset -> no hooks, no file; set -> hooks on
     GPUModelRunner / ModelCudaGraphManager.run_fullgraph / GroupCoordinator, one trace file;
  D3 with GLM53_DEC_HOSTLOOP=1 (production's state) and GLM53_DEC_TRACE through integrate.plugin_register: the
     hostloop fast path IS installed (its fingerprint check saw the unwrapped model-runner methods) and the tracer
     hooks sit on top; D2 also: plugin_register creates no CUDA context; D3n negative control: the tracer installed FIRST -> hostloop refuses (so D3 would catch an
     ordering change in integrate.py);
  G1 real CUDA: a collective stand-in that enqueues torch.cuda._sleep on vLLM's current stream: the gar duration
     equals the calibrated GPU time, the host call returns long before (enqueue semantics), a pending pair is not
     resolved (and the host does not block) until the GPU is done;
  G2 CUDA-graph capture: the wrapped collective captured inside torch.cuda.graph records no CUDA event, the capture
     succeeds and replays correctly; G3 GLM53_DEC_TRACE_GPU=0 -> host events only; G4 the host cost of a
     GPU-timed eager collective vs bare (paired, alternating).
Run: python3 tests/test_dectrace.py                                         (T1-T7, A1)
     tests/hostloop_gpu.sh python3 tests/test_dectrace.py --plugin --gpu   (also D1-D3, G1-G3)
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAILS: list[str] = []


def check(cond, msg) -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILS.append(msg)
    return bool(cond)


class tmpdir:
    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="dectrace-test.")

    def __enter__(self):
        return self.d

    def __exit__(self, *a):
        import shutil
        shutil.rmtree(self.d, ignore_errors=True)


ENV_NAMES = ("GLM53_DEC_TRACE", "GLM53_DEC_TRACE_RING", "GLM53_DEC_TRACE_FLUSH", "GLM53_DEC_TRACE_MAX_STEPS",
             "GLM53_DEC_TRACE_EVENTS", "GLM53_DEC_TRACE_GPU", "LOCAL_RANK", "RANK")


def fresh(env: dict):
    """Reload the tracer with a clean state, build stub modules, return (dt, mr, ps, rej, env backup)."""
    for m in [k for k in list(sys.modules) if k.startswith("glm53_dectrace")]:
        del sys.modules[m]
    old = {k: os.environ.get(k) for k in ENV_NAMES}
    for k in ENV_NAMES:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})

    dt = importlib.import_module("glm53_dectrace")

    class Drafter:
        pass

    class Sampler:
        def __call__(self, x):
            return x

    class RejectionSampler:
        def forward(self, x=0):
            return x + 1

        def __call__(self, x=0):
            return self.forward(x)

    class Proposer:
        def propose(self, x=0):
            return [x]

    class CudaGraphManager:
        def run_pw_graph(self, model, model_inputs):
            return "pw"                 # breakable-cudagraph replay: the segments replay, the model's forward is not called

    class ModelCudaGraphManager(CudaGraphManager):
        def run_fullgraph(self, desc=None):
            return "hidden"

    class DraftCudaGraphManager(CudaGraphManager):   # the drafters' managers: siblings of ModelCudaGraphManager
        pass

    class GPUModelRunner:
        def __init__(self):
            self.model = None
            self.rs = RejectionSampler()
            self.sampler = Sampler()
            self.speculator = Proposer()
            self._drafter = Drafter()
            self.cudagraph_manager = None
            self.pw = False             # with a cudagraph manager: True = PIECEWISE step, False = FULL step

        def get_model(self):
            return self.model

        def load_model(self):
            return "loaded"

        def prepare_inputs(self, x=0):
            return x + 1

        def execute_model(self, x=0):
            self.prepare_inputs(x)
            if self.cudagraph_manager is not None and self.pw:
                self.cudagraph_manager.run_pw_graph(self.model, {})   # PIECEWISE: breakable replay, no forward
            elif self.cudagraph_manager is not None:
                self.cudagraph_manager.run_fullgraph(x)     # FULL decode: graph replay, no model forward
            elif callable(self.model):
                self.model(x)          # nn.Module: __call__ -> the instance's (wrapped) forward
            return x + 1

        def propose_draft_token_ids(self, x=0):
            return [x]              # the V1 runner's path; wrapped at class level too

        def sample_tokens(self, x=0):
            return self.rs(x)          # the rejection sampler runs inside sample_tokens

    mr = types.ModuleType("mr_stub")
    mr.GPUModelRunner = GPUModelRunner
    mr.ModelCudaGraphManager = ModelCudaGraphManager
    mr.CudaGraphManager = CudaGraphManager
    mr.DraftCudaGraphManager = DraftCudaGraphManager

    class GroupCoordinator:
        def _all_reduce_out_place(self, t=0):
            return t + 1

        def _all_gather_out_place(self, t=0, dim=-1):
            return t + 1

    ps = types.ModuleType("ps_stub")
    ps.GroupCoordinator = GroupCoordinator

    rej = types.ModuleType("rej_stub")
    rej.RejectionSampler = RejectionSampler
    return dt, mr, ps, rej, old


def restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def trace_path(tmp: str) -> str:
    """The rank's trace file: the name carries the rank once it is known, so list the directory."""
    fs = sorted(glob.glob(os.path.join(tmp, "dectrace-*.ndjson")))
    assert len(fs) == 1, fs
    return fs[0]


def read_records(path: str):
    """(host records [(n, t, ev)], gpu records [(n, t, ev, dur_ns)], rank, stopped line)."""
    recs, gpu, rank, stopped = [], [], None, None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#!"):
            if line.startswith("#! rank"):
                rank = int(line.split()[2])
            elif line.startswith("#! stopped"):
                stopped = line
            continue
        p = line.split(" ")
        if len(p) == 3:
            recs.append((int(p[0]), int(p[1]), p[2]))
        else:
            gpu.append((int(p[0]), int(p[1]), p[2], int(p[3])))
    return recs, gpu, rank, stopped


class Sampler:
    def __call__(self, x):
        return x


def run_steps(runner, group, n):
    for i in range(n):
        runner.execute_model(i)
        runner.sample_tokens(i)
        runner.propose_draft_token_ids(i)
        runner.speculator.propose(i)
        group._all_reduce_out_place(i)
        group._all_gather_out_place(i)


STEP_EVENTS = ["sched", "prep", "exec_e", "sam_b", "rs_b", "rs_e", "sam_e", "dr_b", "dr_e", "dr_b", "dr_e",
               "ar_b", "ar_e", "ag_b", "ag_e"]


# --------------------------------------------------------------------------- T1
def t1():
    print("T1 env unset -> inert")
    dt, mr, ps, rej, old = fresh({})
    try:
        r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej, directory="")
        check(r["reason"] == "off" and r["on"] is False and r["path"] == "", f"report off: {r}")
        for cls, name in ((mr.GPUModelRunner, "execute_model"), (mr.GPUModelRunner, "prepare_inputs"),
                          (mr.ModelCudaGraphManager, "run_fullgraph"), (mr.ModelCudaGraphManager, "run_pw_graph"),
                          (ps.GroupCoordinator, "_all_reduce_out_place"), (rej.RejectionSampler, "forward")):
            check(not hasattr(getattr(cls, name), "_glm53_dectrace"), f"{name} untouched")
        check(dt._T.fh is None and not dt._T.enabled, "no file opened, tracer disabled")
        r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej, directory="0")
        check(r["reason"] == "off", f"GLM53_DEC_TRACE=0 is off too: {r}")
    finally:
        restore(old)


# --------------------------------------------------------------------------- T2
def t2():
    print("T2 installed -> every event in its own step, fg on FULL replay, rank, forward wrap")
    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_FLUSH": "1", "LOCAL_RANK": "1"})
        try:
            r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            # 5 runner hooks + run_fullgraph + run_pw_graph + AR, AG, rejection sampler
            check(r["on"] and r["hooks"] == 10, f"report: {r}")
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()
            run_steps(runner, group, 2)
            dt._T.flush(True)
            path = trace_path(tmp)
            recs, _, rank, _ = read_records(path)
            got = [(n, e) for n, _, e in recs]
            want = [(1, e) for e in STEP_EVENTS] + [(2, e) for e in STEP_EVENTS]
            check(got == want, f"(step, event) over 2 steps, sched in its own step: {got}")
            ts = [t for _, t, _ in recs]
            check(ts == sorted(ts), "timestamps non-decreasing")
            check(rank == 1, f"rank line: {rank}")
            check(os.path.basename(path) == f"dectrace-rank1-pid{os.getpid()}.ndjson", f"path renamed: {path}")

            runner.cudagraph_manager = mr.ModelCudaGraphManager()       # FULL decode: replay, no forward call
            runner.execute_model(3)
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            got = [(n, e) for n, _, e in recs if n == 3]
            check(got == [(3, "sched"), (3, "prep"), (3, "fg_b"), (3, "fg_e"), (3, "exec_e")],
                  f"FULL replay -> fg_b/fg_e inside the step: {got}")

            class Model:
                def forward(self, x):
                    return x

                def __call__(self, x):
                    return self.forward(x)

            runner.cudagraph_manager = None
            runner.model = Model()
            runner.execute_model(9)
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            got = [(n, e) for n, _, e in recs if n == 4]
            check(got == [(4, "sched"), (4, "prep"), (4, "tgt_b"), (4, "tgt_e"), (4, "exec_e")],
                  f"eager forward wrapped on first use: {got}")
            check(not getattr(runner._drafter, "_glm53_dectrace_fwd", False), "drafter object untouched")
            check(getattr(runner.speculator, "_glm53_dectrace_sp", False), "speculator.propose wrapped")
            check(hasattr(runner.get_model(), "_glm53_dectrace_fwd"), "target model marked")
            # the wrappers are idempotent: installing twice must not double-wrap
            dt.patch_model_runner(mr)
            dt.patch_collectives(ps, rej)
            runner.execute_model(10)
            group._all_reduce_out_place(1)
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            got = [e for n, _, e in recs if n == 5]
            check(got == ["sched", "prep", "tgt_b", "tgt_e", "exec_e", "ar_b", "ar_e"],
                  f"no double wrapping: {got}")
            check(group._all_reduce_out_place(41) == 42 and runner.execute_model(1) == 2, "results pass through")
            # PIECEWISE step (production: breakable-cudagraph replay, the model's forward is NOT called)
            check(mr.DraftCudaGraphManager().run_pw_graph(None, {}) == "pw", "a drafter manager's run_pw_graph works")
            runner.cudagraph_manager, runner.pw = mr.ModelCudaGraphManager(), True
            runner.execute_model(11)
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            got = [(n, e) for n, _, e in recs if n == 7]
            check(got == [(7, "sched"), (7, "prep"), (7, "pw_b"), (7, "pw_e"), (7, "exec_e")],
                  f"PIECEWISE run -> pw_b/pw_e inside the step (no forward call there): {got}")
            check(not any(e in ("pw_b", "pw_e") for n, _, e in recs if n != 7),
                  "the drafter manager's run_pw_graph (a CudaGraphManager sibling) records nothing")
            check(not getattr(mr.CudaGraphManager.run_pw_graph, "_glm53_dectrace", False)
                  and getattr(mr.ModelCudaGraphManager.run_pw_graph, "_glm53_dectrace", False),
                  "run_pw_graph wrapped on ModelCudaGraphManager only, the base class untouched")
        finally:
            restore(old)
            dt._T.stop("test end")


# --------------------------------------------------------------------------- T3/T4/T5
def t3():
    print("T3 ring bound + every chunk flushed")
    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_RING": "64",
                                      "GLM53_DEC_TRACE_FLUSH": "1000000"})
        try:
            dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            runner = mr.GPUModelRunner()
            runner.sampler = Sampler()
            peaks = []
            for i in range(30):
                runner.execute_model(i)
                peaks.append(len(dt._T.buf))
            check(max(peaks) <= 64 + 16, f"buffer stayed bounded (peak {max(peaks)})")
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            check(sum(1 for _, _, e in recs if e == "sched") == 30, f"30 scheds: {len(recs)} records")
        finally:
            restore(old)
            dt._T.stop("test end")


def t4():
    print("T4 max steps stops the tracer")
    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_MAX_STEPS": "2",
                                      "GLM53_DEC_TRACE_FLUSH": "1"})
        try:
            r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            runner = mr.GPUModelRunner()
            for i in range(5):
                runner.execute_model(i)
            check(not dt._T.enabled and dt._T.fh is None, "tracer stopped")
            recs, _, _, stopped = read_records(trace_path(tmp))
            check(max(n for n, _, _ in recs) == 2, f"last recorded step 2 ({max(n for n, _, _ in recs)})")
            check(stopped is not None and "max steps reached" in stopped, f"stop line written: {stopped}")
            check(runner.execute_model(7) == 8, "the runner keeps working after the stop")
        finally:
            restore(old)


def t5():
    print("T5 event filter")
    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_FLUSH": "1",
                                      "GLM53_DEC_TRACE_EVENTS": "-ar_b,-ar_e,-sam_e"})
        try:
            dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()
            runner.execute_model(1)
            group._all_reduce_out_place(2)
            runner.sample_tokens(3)
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            evs = [e for _, _, e in recs]
            check("ar_b" not in evs and "ar_e" not in evs, f"ar filtered: {evs}")
            check("sam_b" in evs and "sam_e" not in evs, "sam_b kept, sam_e dropped")
            check("sched" in evs, "default events kept")
        finally:
            restore(old)
            dt._T.stop("test end")


# --------------------------------------------------------------------------- T6
def t6():
    print("T6 per-record overhead")
    dt, mr, ps, rej, old = fresh({})
    try:
        dt.patch_model_runner(mr)
        dt.patch_collectives(ps)
        dt._T.enabled = True
        dt._T.on = True
        dt._T.kept = frozenset(dt.EVENTS)
        dt._T.step = 5
        n = 200000
        t0 = time.perf_counter_ns()
        for i in range(n):
            dt._T.add("ar_b")
        per = (time.perf_counter_ns() - t0) / n
        dt._T.buf.clear()
        print(f"  {per:.0f} ns per record")
        check(per < 1000, f"record cost {per:.0f} ns < 1 us")
        runner = mr.GPUModelRunner()
        t0 = time.perf_counter_ns()
        for i in range(20000):
            runner.prepare_inputs(i)
        per2 = (time.perf_counter_ns() - t0) / 20000
        dt._T.on = False
        bare = mr.GPUModelRunner.prepare_inputs._glm53_dectrace_orig
        t0 = time.perf_counter_ns()
        for i in range(20000):
            bare(runner, i)
        per3 = (time.perf_counter_ns() - t0) / 20000
        print(f"  wrapped call {per2:.0f} ns vs unpatched {per3:.0f} ns (delta {per2 - per3:.0f} ns)")
        check(per2 - per3 < 1000, f"wrapper cost {per2 - per3:.0f} ns < 1 us")
    finally:
        restore(old)
        dt._T.on = False
        dt._T.buf.clear()


# --------------------------------------------------------------------------- T7
class FakeStream:
    def __init__(self):
        self.clock = 0              # the "GPU" clock in ns, advanced by the fake collective
        self.inflight = []

    def complete(self):
        for e in self.inflight:
            e.done = True
        self.inflight.clear()


class FakeEvent:
    created = 0
    fail_record = False

    def __init__(self, enable_timing=True):
        FakeEvent.created += 1
        self.t, self.done = None, False

    def record(self, s):
        if FakeEvent.fail_record:
            raise RuntimeError("injected cudaEventRecord failure")
        self.t, self.done = s.clock, False
        s.inflight.append(self)

    def query(self):
        return self.done

    def elapsed_time(self, other):
        assert self.done and other.done, "elapsed_time on an incomplete event (a real one would raise)"
        return (other.t - self.t) / 1e6          # ms, like torch


class FakeCuda:
    Event = FakeEvent

    def __init__(self):
        self.capturing = False

    def is_current_stream_capturing(self):
        return self.capturing


def t7():
    print("T7 GPU records with a fake CUDA: budget, capture, lazy resolution, drain, bounds, failure")
    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_FLUSH": "1000000",
                                      "GLM53_DEC_TRACE_GPU": "0"})
        try:
            stream, cuda = FakeStream(), FakeCuda()

            def fake_ar(self, t=0, dur_ns=250_000):
                stream.clock += dur_ns          # the kernel: this rank's wait + transfer
                if t == "raise":
                    raise ValueError("collective failed")
                return t + 1

            ps.GroupCoordinator._all_reduce_out_place = fake_ar
            r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            check(r["gpu"] is False, "GLM53_DEC_TRACE_GPU=0 -> no GPU timing at install")
            dt._T.gpu, dt._T.torch, dt._T.stream = True, types.SimpleNamespace(cuda=cuda), (lambda: stream)
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()

            runner.execute_model(0)                      # step 1
            for i in range(10):
                check(group._all_reduce_out_place(i) == i + 1, "result passes through") if i == 0 else \
                    group._all_reduce_out_place(i)
            check(len(dt._T.gpend) == dt.GPU_PER_STEP, f"per-step budget: {len(dt._T.gpend)} pending of 10 calls")
            dt._T.flush(True)
            _, gpu, _, _ = read_records(trace_path(tmp))
            check(gpu == [], "incomplete pairs are not resolved (no sync, no record)")
            stream.complete()
            dt._T.flush(True)
            recs, gpu, _, _ = read_records(trace_path(tmp))
            check(len(gpu) == dt.GPU_PER_STEP and all(g[2] == "gar" and g[3] == 250_000 and g[0] == 1 for g in gpu),
                  f"completed pairs -> 'n t gar dur_ns' with the exact duration: {gpu[:2]}")
            arb = [t for n, t, e in recs if e == "ar_b" and n == 1]
            check([g[1] for g in gpu] == arb[:dt.GPU_PER_STEP], "gar t = the host ar_b time of that collective")
            check(len(dt._T.gpool) == 2 * dt.GPU_PER_STEP, "resolved events go back to the pool")

            runner.execute_model(0)                      # step 2: capture
            created = FakeEvent.created
            cuda.capturing = True
            group._all_reduce_out_place(5)
            cuda.capturing = False
            check(dt._T.gpend == [] and FakeEvent.created == created, "no event while the stream is capturing")
            dt._T.flush(True)
            recs, _, _, _ = read_records(trace_path(tmp))
            check([e for n, _, e in recs if n == 2 and e.startswith("ar")] == ["ar_b", "ar_e"],
                  "host events still recorded during capture")

            dropped0 = dt._T.gdropped
            for s in range(40):                          # FLUSH is huge: step_done must drain at GPU_DRAIN_AT
                runner.execute_model(s)
                group._all_reduce_out_place(1)
                group._all_reduce_out_place(2)
                stream.complete()
            check(len(dt._T.gpend) < dt.GPU_DRAIN_AT and dt._T.gdropped == dropped0,
                  f"drained from step_done: {len(dt._T.gpend)} pending, {dt._T.gdropped - dropped0} dropped")
            check(sum(1 for x in dt._T.buf if x.endswith(" 250000") and " gar " in x) >= dt.GPU_DRAIN_AT,
                  "the drained pairs are records in the buffer")

            for s in range(100):                         # a GPU that never completes: the pending list is bounded
                runner.execute_model(s)
                for _ in range(8):
                    group._all_reduce_out_place(1)
            check(len(dt._T.gpend) <= dt.GPU_PENDING_MAX and dt._T.gdropped > 0,
                  f"pending bounded at {dt.GPU_PENDING_MAX} ({len(dt._T.gpend)}), {dt._T.gdropped} dropped")
            stream.complete()

            runner.execute_model(0)
            try:
                group._all_reduce_out_place("raise")
                check(False, "a raising collective propagates")
            except ValueError:
                check(True, "a raising collective propagates (its pair still recorded)")
            FakeEvent.fail_record = True
            runner.execute_model(0)
            ok = group._all_reduce_out_place(3) == 4
            FakeEvent.fail_record = False
            check(ok and dt._T.gpu is False and dt._T.gpend == [], "a failing event record -> GPU timing off, "
                  "the collective's result intact")
            n_before = len(dt._T.buf)
            runner.execute_model(0)
            group._all_reduce_out_place(3)
            check(len(dt._T.buf) > n_before and dt._T.on, "host events continue after GPU timing switched off")
            dt._T.stop("test end")
            _, _, _, stopped = read_records(trace_path(tmp))
            check(stopped is not None and "GPU pairs dropped" in stopped, f"drops reported: {stopped}")
        finally:
            restore(old)
            dt._T.stop("test end")


# --------------------------------------------------------------------------- A1
AR_SKEWS = (140e3, 220e3, 300e3, 380e3, 460e3)     # rank 1's GPU later at the eager AR (ns): median/mean 300 us


def synth_pair(tmp: str, n: int = 80, off_ns: float = 3.2e6, ntp1_ns: float = 40e3, host_skew_ns: float = 250e3,
               tails=(1.7e6, 1.9e6), ag_skew_ns: float = -100e3):
    """Two rank files modelled on what the hardware does (not on what the aligner assumes):
    wall time W; rank r's perf clock = W + c_r (c0 = 0, c1 = off_ns); rank r's time_ns = W + e_r (e1 = NTP error).
    Host: rank 1's host is host_skew_ns later at sched; each rank's host issues the eager AR in its tail and the call
    returns 55 us later (enqueue only). GPU: rank 1 reaches the AR AR_SKEWS[k] later than rank 0; both finish at
    max(arrivals) + 20 us, so dur_r = end - arrival_r. The eager AG: rank 0 later by 100 us (ag_skew_ns < 0)."""
    paths = []
    for r in (0, 1):
        c, e = (0.0, 0.0) if r == 0 else (off_ns, ntp1_ns)
        p = os.path.join(tmp, f"dectrace-rank{r}-pid{100 + r}.ndjson")
        with open(p, "w") as f:
            f.write("#!glm53-dectrace " + json.dumps({"ver": 2, "pid": 100 + r, "host": f"h{r}",
                                                      "boot_pc_ns": int(1e12 + c), "boot_epoch_ns": int(1e12 + e)})
                    + "\n#! rank %d\n" % r)
            for s in range(1, n + 1):
                T = 1e12 + s * 70e6
                d_ar = AR_SKEWS[(s * 7) % 5]
                sched = T + (host_skew_ns if r == 1 else 0.0)
                prep = sched + 300e3
                ar_b = prep + 200e3
                ar_e = ar_b + 55e3                               # host return: NOT when the peer arrived
                fg_b = prep + tails[r]
                fg_e = fg_b + 150e3
                exec_e = fg_e + 50e3
                sam_b = exec_e + 30e3
                ag_b = sam_b + 20e3
                ag_e = ag_b + 60e3
                dr_b, dr_e = ag_e + 50e3, ag_e + 6.05e6
                sam_e = dr_e + 100e3
                a0 = T + 5e6                                     # GPU arrivals (wall)
                a = (a0, a0 + d_ar)
                end = max(a) + 20e3
                b0 = T + 60e6
                b = (b0, b0 + ag_skew_ns)
                end2 = max(b) + 15e3
                rows = [(sched, "sched"), (prep, "prep"), (ar_b, "ar_b"), (ar_e, "ar_e"), (fg_b, "fg_b"),
                        (fg_e, "fg_e"), (exec_e, "exec_e"), (sam_b, "sam_b"), (ag_b, "ag_b"), (ag_e, "ag_e"),
                        (dr_b, "dr_b"), (dr_e, "dr_e"), (sam_e, "sam_e")]
                for t, ev in rows:
                    f.write("%d %d %s\n" % (s, int(t + c), ev))
                f.write("%d %d gar %d\n" % (s, int(ar_b + c), int(end - a[r])))
                f.write("%d %d gag %d\n" % (s, int(ag_b + c), int(end2 - b[r])))
                if s % 10 == 0:
                    f.write("#! clock %d %d\n" % (int(T + c), int(T + e)))
            f.write(f"#! stopped exit at step {n} after {n * 15} records\n")
        paths.append(p)
    return paths


def aligner(*args):
    r = subprocess.run([sys.executable, str(REPO / "tests" / "align_dectrace.py"), *args], capture_output=True,
                       text=True)
    s = json.loads(next((x[8:] for x in r.stdout.splitlines() if x.startswith("SUMMARY ")), "{}"))
    return r, s


def t8():
    print("A1 aligner on a physically modelled two-rank pair")
    with tmpdir() as tmp:
        p0, p1 = synth_pair(tmp)
        r, s = aligner(p0, p1)
        print("\n".join("    " + x for x in r.stdout.splitlines()[:16]))
        check(r.returncode == 0, f"aligner exit 0 ({r.stderr[-300:]})")
        check(s.get("matched") == 80 and s.get("gpu_pairs") == 160, f"80 steps, 160 GPU pairs: {s.get('matched')}, "
              f"{s.get('gpu_pairs')}")
        ar0 = s.get("gpu_positions", {}).get("AR0", {})
        check(ar0.get("skew_med_us") == 300.0, f"AR0 arrival skew median exactly 300 us: {ar0}")
        check(abs(ar0.get("wait_r0_mean_us", 0) - 300.0) < 1e-6 and ar0.get("wait_r1_mean_us") == 0.0,
              "AR0: rank 0 waits the skew (mean 300 us), rank 1 not at all")
        ag0 = s.get("gpu_positions", {}).get("AG0", {})
        check(ag0.get("skew_med_us") == -100.0 and ag0.get("wait_r1_mean_us") == 100.0,
              f"AG0: rank 0 later by 100 us, rank 1 waits 100 us: {ag0}")
        w = s.get("windows_us", {})
        check(w.get("prep->FULL graph launch") == [1700.0, 1900.0], f"host tails per rank: {w.get('prep->FULL graph launch')}")
        check(s.get("host_skew_us") == {} and s.get("host_offset_ms") is None,
              "no cross-rank host skew without an outside clock")
        r, s = aligner(p0, p1, "--offset-us", "3200")
        hs = s.get("host_skew_us", {})
        check(hs.get("sched") == 250.0 and hs.get("fg_b") == 450.0,
              f"--offset-us: host skew recovered exactly (sched 250, fg_b 450): {hs}")
        r, s = aligner(p0, p1, "--wallclock")
        hs = s.get("host_skew_us", {})
        check(abs(hs.get("sched", 1e9) - 250.0) <= 40.5, f"--wallclock: sched skew within the 40 us NTP error: {hs}")
        # negative control: the first version's clock = the median difference of the eager collectives' host returns
        ret = {}
        for rank, p in ((0, p0), (1, p1)):
            for line in open(p):
                q = line.split()
                if len(q) == 3 and q[2] == "ar_e":
                    ret.setdefault(int(q[0]), {})[rank] = int(q[1])
        est = statistics.median(v[1] - v[0] for v in ret.values() if len(v) == 2)
        err_us = (est - 3.2e6) / 1e3
        print(f"    host-return offset estimate {est / 1e6:.3f} ms vs true 3.200 ms: error {err_us:+.1f} us")
        check(abs(err_us) >= 200, f"negative control: host returns misplace the clock by the host skew ({err_us:+.0f} us)")


# --------------------------------------------------------------------------- D1-D3 (production image)
CHILD = r'''
import os, sys, glob, logging, importlib, json
sys.path.insert(0, "/w")
logging.basicConfig(level=logging.WARNING)
mode = os.environ["DT_MODE"]
import torch
if mode == "plugin":
    import integrate
    integrate.plugin_register()
    cuda_after_plugin = torch.cuda.is_initialized()     # before any model-runner import (API server / engine core)
    import vllm.v1.worker.gpu.model_runner as MR
else:                                  # reversed order on purpose: the tracer wraps first, then hostloop installs
    import glm53_dectrace, glm53_hostloop
    import vllm.v1.worker.gpu.model_runner as MR
    cuda_after_plugin = None
    glm53_dectrace.install_now(mr_mod=MR)
    rep = glm53_hostloop.install_now(mr_mod=MR)
import vllm.distributed.parallel_state as PS
res = dict(
    exec_=getattr(MR.GPUModelRunner.execute_model, "_glm53_dectrace", False),
    prep=getattr(MR.GPUModelRunner.prepare_inputs, "_glm53_dectrace", False),
    sam=getattr(MR.GPUModelRunner.sample_tokens, "_glm53_dectrace", False),
    load=getattr(MR.GPUModelRunner.load_model, "_glm53_dectrace", False),
    fg=getattr(MR.ModelCudaGraphManager.run_fullgraph, "_glm53_dectrace", False),
    pw=getattr(MR.ModelCudaGraphManager.run_pw_graph, "_glm53_dectrace", False),
    pw_base=getattr(importlib.import_module("vllm.v1.worker.gpu.cudagraph_utils").CudaGraphManager.run_pw_graph,
                    "_glm53_dectrace", False),
    ar=getattr(PS.GroupCoordinator._all_reduce_out_place, "_glm53_dectrace", False),
    ag=getattr(PS.GroupCoordinator._all_gather_out_place, "_glm53_dectrace", False),
    files=sorted(os.path.basename(p) for p in glob.glob(os.environ.get("DT_OUT", "/nonexistent") + "/*")),
    cuda_after_plugin=cuda_after_plugin,
    cuda_after_import=torch.cuda.is_initialized(),
)
if "glm53_hostloop" in sys.modules:
    H = sys.modules["glm53_hostloop"]
    sm = importlib.import_module("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90")
    res["hl_fast"] = bool(H.ST.enabled)
    res["hl_klh"] = getattr(sm.FlashInferMLASparseSM90Builder._kv_lens_host, "_glm53_hostloop", False)
    res["hl_post"] = getattr(getattr(MR.GPUModelRunner.postprocess_sampled, "_glm53_dectrace_orig",
                                     MR.GPUModelRunner.postprocess_sampled), "_glm53_hostloop", False)
    if mode != "plugin":
        res["hl_reason"] = rep.get("reason")
print("RESULT", json.dumps(res))
'''


def child(env: dict, tmp: str):
    e = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
    e.update(env)
    e["DT_OUT"] = tmp
    e.setdefault("DT_MODE", "plugin")
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=e)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    warn = [x for x in p.stderr.splitlines() if "WARNING" in x or "NOT installed" in x]
    return (json.loads(line[0][7:]) if line else {"error": p.stderr[-600:]}), warn


HOOKS = ("exec_", "prep", "sam", "load", "fg", "pw", "ar", "ag")


def t9():
    if "--plugin" not in sys.argv:
        print("D1-D3 plugin wiring: NOT RUN here (pass --plugin inside the production image: tests/hostloop_gpu.sh)")
        return
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("D1-D3 plugin wiring: NOT RUN (vLLM not importable here)")
        return
    print("D1-D3 plugin wiring (fresh interpreters, real vLLM tree)")
    with tmpdir() as tmp:
        r0, w0 = child({}, tmp)
        print("  D1 unset:", r0)
        check("error" not in r0 and not any(r0.get(k) for k in HOOKS) and r0.get("files") == [],
              "D1 unset -> no hooks, no file")
    with tmpdir() as tmp:
        r1, w1 = child({"GLM53_DEC_TRACE": tmp}, tmp)
        print("  D2 on:", r1)
        check(all(r1.get(k) for k in HOOKS), "D2 hooks on GPUModelRunner / run_fullgraph / run_pw_graph / "
              "GroupCoordinator")
        check(r1.get("pw_base") is False, "D2 run_pw_graph wrapped on ModelCudaGraphManager only (the base "
              "CudaGraphManager the drafters' managers inherit from is untouched)")
        check(len([f for f in r1.get("files", []) if f.startswith("dectrace-")]) == 1, f"D2 one trace file: {r1}")
        check(r1.get("cuda_after_plugin") is False and r0.get("cuda_after_plugin") is False,
              "D2 plugin_register creates no CUDA context (a process that never imports the model runner stays clean)")
        check(r1.get("cuda_after_import") == r0.get("cuda_after_import"),
              f"D2 after the model-runner import the CUDA state is the same as without the tracer "
              f"({r1.get('cuda_after_import')} vs {r0.get('cuda_after_import')}: the import itself initializes CUDA)")
    with tmpdir() as tmp:
        r2, w2 = child({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_HOSTLOOP": "1"}, tmp)
        print("  D3 hostloop + trace via plugin_register:", r2, w2[:3])
        check(r2.get("hl_fast") is True and r2.get("hl_klh") is True and r2.get("hl_post") is True,
              "D3 the hostloop fast path is installed next to the tracer (production's state)")
        check(all(r2.get(k) for k in HOOKS), "D3 tracer hooks installed on top")
        check(not any("NOT installed" in x for x in w2), f"D3 no refusal line: {w2[:2]}")
    with tmpdir() as tmp:
        r3, w3 = child({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_HOSTLOOP": "1", "DT_MODE": "reversed"}, tmp)
        print("  D3n tracer first, then hostloop:", {k: r3.get(k) for k in ("hl_fast", "hl_reason")})
        check(r3.get("hl_fast") is False and "unverified vLLM source" in str(r3.get("hl_reason")),
              "D3n negative control: with the tracer wrapped first hostloop refuses (D3 guards the order)")


# --------------------------------------------------------------------------- G1-G3 (production image, GPU)
def t10():
    if "--gpu" not in sys.argv:
        print("G1-G3 real CUDA: NOT RUN here (pass --gpu inside the production image)")
        return
    try:
        import torch
        from vllm.utils.torch_utils import current_stream
        assert torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        print(f"G1-G3 real CUDA: NOT RUN ({exc!r})")
        return
    print("G1-G3 real CUDA events around a collective stand-in on vLLM's current stream")
    CYC = int(os.environ.get("DT_SLEEP_CYCLES", "4000000"))

    def calib(cycles, n=7):
        v = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s = current_stream()
            a.record(s)
            torch.cuda._sleep(cycles)
            b.record(s)
            b.synchronize()
            v.append(a.elapsed_time(b) * 1e3)
        return statistics.median(v)

    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_FLUSH": "1000"})
        try:
            def sleeper(self, t, cycles=CYC):
                torch.cuda._sleep(cycles)            # enqueued on the current stream like ncclAllReduce(stream)
                return t.add_(1)

            ps.GroupCoordinator._all_reduce_out_place = sleeper
            x = torch.zeros(4, device="cuda")
            cal = calib(CYC)
            print(f"  calibrated stand-in kernel: {cal:.1f} us")
            r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            check(r["gpu"] is True, f"GPU timing on: {r}")
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()
            host = []
            for i in range(20):
                runner.execute_model(i)
                t0 = time.perf_counter_ns()
                group._all_reduce_out_place(x)
                host.append((time.perf_counter_ns() - t0) / 1e3)
            torch.cuda.synchronize()
            dt._T.flush(True)
            recs, gpu, _, _ = read_records(trace_path(tmp))
            durs = [g[3] / 1e3 for g in gpu if g[2] == "gar"]
            md = statistics.median(durs) if durs else float("nan")
            print(f"  gar durations med {md:.1f} us (n={len(durs)}), host call med {statistics.median(host):.1f} us")
            check(len(durs) == 20 and abs(md - cal) <= max(0.05 * cal, 10.0),
                  f"G1 gar duration == the GPU time of the collective ({md:.1f} vs {cal:.1f} us)")
            check(statistics.median(host) < 0.25 * cal, "G1 the host call returns long before the GPU finishes "
                  "(host return is not a completion time)")
            check(torch.equal(x, torch.full_like(x, 20)), "G1 results intact")
            runner.execute_model(0)
            group._all_reduce_out_place(x, cycles=CYC * 40)     # a long "wait" still running on the GPU
            pend = len(dt._T.gpend)
            t1 = time.perf_counter_ns()
            dt._T.gpu_drain()
            t_drain = (time.perf_counter_ns() - t1) / 1e3
            check(pend >= 1 and len(dt._T.gpend) == pend, f"G1 a pair still on the GPU is not resolved ({pend} "
                  f"pending after drain)")
            check(t_drain < 1000, f"G1 drain does not block ({t_drain:.0f} us while the GPU sleeps "
                  f"~{cal * 41 / 1e3:.0f} ms)")
            torch.cuda.synchronize()
            dt._T.gpu_drain()
            check(len(dt._T.gpend) == 0, "G1 resolved once the GPU finished")

            # G2 capture: the same wrapped collective inside torch.cuda.graph
            y = torch.zeros(4, device="cuda")
            runner.execute_model(0)
            g = torch.cuda.CUDAGraph()
            pend0, count0 = len(dt._T.gpend), dt._T.gcount
            nbuf0 = len(dt._T.buf)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                group._all_reduce_out_place(y)
            check(len(dt._T.gpend) == pend0 and dt._T.gcount == count0, "G2 no CUDA event recorded during capture")
            check(any(b.endswith(" ar_b") for b in dt._T.buf[nbuf0:]), "G2 host events recorded during capture")
            check(torch.equal(y, torch.zeros_like(y)), "G2 capture did not execute the kernel")
            g.replay()
            g.replay()
            torch.cuda.synchronize()
            check(torch.equal(y, torch.full_like(y, 2)), f"G2 two replays -> 2: {y.tolist()}")
            check(dt._T.gpu is True, "G2 GPU timing still on after the capture")
            dt._T.stop("test end")
        finally:
            restore(old)
            dt._T.stop("test end")

    with tmpdir() as tmp:           # G4 host cost of one GPU-timed collective vs host events only (paired)
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_FLUSH": "64"})
        try:
            ps.GroupCoordinator._all_reduce_out_place = lambda self, t: t.add_(1)
            dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()
            z = torch.zeros(4, device="cuda")
            bare = ps.GroupCoordinator._all_reduce_out_place._glm53_dectrace_orig
            arms = {"bare": [], "host": [], "gpu": []}
            for rnd in range(40):
                for arm in (("bare", "host", "gpu") if rnd % 2 == 0 else ("gpu", "host", "bare")):
                    runner.execute_model(0)
                    dt._T.gpu = arm == "gpu"
                    for _ in range(8):
                        t0 = time.perf_counter_ns()
                        if arm == "bare":
                            bare(group, z)
                        else:
                            group._all_reduce_out_place(z)
                        arms[arm].append((time.perf_counter_ns() - t0) / 1e3)
                torch.cuda.synchronize()
            m = {k: statistics.median(v) for k, v in arms.items()}
            print(f"  G4 host time per eager collective (median of {len(arms['gpu'])}): bare {m['bare']:.2f} us, "
                  f"+host events {m['host']:.2f} us, +GPU timing {m['gpu']:.2f} us")
            check(m["gpu"] - m["bare"] < 25.0, f"G4 tracer cost per eager collective {m['gpu'] - m['bare']:.1f} us "
                  f"(x2 per decode step) < 25 us")
        finally:
            restore(old)
            dt._T.stop("test end")

    with tmpdir() as tmp:
        dt, mr, ps, rej, old = fresh({"GLM53_DEC_TRACE": tmp, "GLM53_DEC_TRACE_GPU": "0", "GLM53_DEC_TRACE_FLUSH": "1"})
        try:
            r = dt.install_now(mr_mod=mr, ps_mod=ps, rej_mod=rej)
            runner, group = mr.GPUModelRunner(), ps.GroupCoordinator()
            runner.execute_model(0)
            group._all_reduce_out_place(1)
            dt._T.flush(True)
            recs, gpu, _, _ = read_records(trace_path(tmp))
            check(r["gpu"] is False and gpu == [] and any(e == "ar_b" for _, _, e in recs),
                  "G3 GLM53_DEC_TRACE_GPU=0 -> host events only")
        finally:
            restore(old)
            dt._T.stop("test end")


def main():
    for f in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
        f()
    print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
