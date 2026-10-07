"""GLM53_MLA_PLAN_PIN: page-locked staging for production's sparse-MLA FA2 plan (docs/MLA_PLAN_PIN.md).

The defect (2026-10-05 production profile, $TF_EXL3_ASSETS/prof-1005/ana/, not shipped)
-------------------------------------------------------------------------------------
Production's FLASHINFER_MLA_SPARSE_SM90 backend (the launcher-mounted ``flashinfer_mla_sparse_sm90.py.patched``,
``_SM90State.plan``) replans flashinfer's BatchMLAPagedAttentionWrapper once per metadata build from three cached CPU
staging tensors (``_qo_cpu`` / ``_kv_cpu``: max_tokens + 1 int32, ``_lens_cpu``: max_tokens int32) allocated
PAGEABLE. With ``use_cuda_graph=True`` the wrapper copies them into its fixed device buffers with
``copy_(non_blocking=True)``. At production's max-num-batched-tokens 16384 the indptr copies are 16385 x 4 = 65540 B,
4 B over CUDA's 64 KiB limit for asynchronous pageable H2D copies: each of those two calls synchronizes the host with
the stream, i.e. ``plan()`` blocks until the GPU has drained everything queued before it - the previous step's
DFlash2 drafter graph (~5.7 ms every step on nodeA) - and the GPU then idles while the host finishes the step's
remaining host work (nodeC MRv2 rig: drafter->forward gap 1.385 ms vs 0.049 ms pinned; plan() host 2.99 vs 0.17 ms).

The fix (GLM53_MLA_PLAN_PIN=1; unset/empty/0 = production byte for byte, nothing installed)
---------------------------------------------------------------------------------------------
``_SM90State.plan`` is replaced (source fingerprint checked against production's; a different source -> not
installed, production path) by the same statements with PAGE-LOCKED staging in a 2-slot ring:

* each slot owns pinned qo / kv / lens staging AND its own page-locked copy of the wrapper's int workspace
  (flashinfer's MLAPlan writes the work schedule into ``_pin_memory_int_workspace_buffer`` on the host and then
  ``cudaMemcpyAsync``s it, without a sync; today the pageable indptr copy's implicit stream sync is what keeps the
  NEXT plan from overwriting that host buffer before the previous copy ran - with pinned staging that implicit sync
  is gone, so the buffer must be ring-owned too), and one CUDA event recorded on the current stream after the
  wrapper's plan (after its last async H2D copy);
* before a slot is rewritten its event must have completed (``event.synchronize()`` only if it has not; counted as a
  ring wait): a slot's host bytes are never touched while an async copy from them can still be pending - safe with
  any number of plans per step, two plans can be in flight;
* values are computed exactly as production computes them (clamp / mul / fill / slice-assign); the device buffers
  the captured graphs read (qo/kv indptr, kv_len_arr, the int workspace) receive the same bytes in the same stream
  order, only without the host stall. ``kv_indices`` (the device top-k buffer) is untouched, as in production.

Runs outside CUDA-graph capture only (production's RuntimeError inside a capture is kept). The first plan of each
process self-checks (one event sync, outside capture): the wrapper's three device plan buffers == the slot's host
staging; a mismatch -> WARNING and production's pageable path for the rest of the process.

Cost: +3 pinned staging tensors x 2 slots (max_tokens 16384: 6 x ~64 KiB) and +8 MiB page-locked (the second slot's
int workspace; slot 0 reuses the wrapper's own), per process with an _SM90State (every TP rank).

Logs: "glm53_mla_planpin plugin loaded ... -> off ..." / "... -> installing (mode on)" (every process),
"glm53_mla_planpin: patched _SM90State.plan ..." (at the backend import), "glm53_mla_planpin: rank R self-test: device
plan buffers == pinned staging ..." (first plan), "[glm53-mla-planpin] rank R serving confirmed (mode on): N plans
through the pinned ring ..." (at 64 plans) and a stats line every 100000 plans.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import os
import sys
import textwrap

_log = logging.getLogger("vllm.glm53_mla_planpin")
ENV = "GLM53_MLA_PLAN_PIN"
TAG = "[glm53-mla-planpin]"
SM90_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
FI_MODULE = "flashinfer.mla"
SLOTS = 2
PROOF_AT = 64
STATS_EVERY = 100000
_ON = frozenset({"1"})
_OFF = frozenset({"", "0"})

# sha256(ast.dump(source))[:16] (as glm53_mla_exactlens.source_fingerprint) of the functions this patch replaces or
# relies on, read in the production image with the launcher's vllm-patches/flashinfer_mla_sparse_sm90.py.patched
# (sha256 526399d8, identical on nodeC / nodeA / nodeB's worker copy) and flashinfer 0.6.18 (fi618, mla/_core.py
# sha256 58d06e61, identical on all three) mounted; tests/r16z8p/test_planpin.py prints and checks them.
VERIFIED = {
    "_SM90State.plan": frozenset({"5c9836599374a0ec"}),
    "BatchMLAPagedAttentionWrapper.plan": frozenset({"bcadfb24d6959d4a"}),
}


class _State:
    mode = "off"
    installed = False
    enabled = False       # False after a failed self-test / ring allocation: production's plan from then on
    reason = ""
    plans = 0
    waits = 0
    selftest = None
    rank = None


ST = _State()


def env_mode(environ=None) -> str:
    env = os.environ if environ is None else environ
    v = (env.get(ENV) or "").strip()
    if v in _OFF:
        return "off"
    if v in _ON:
        return "on"
    raise ValueError(f"{ENV} must be unset, empty, 0 or 1 (got a different value)")


def source_fingerprint(fn) -> str | None:
    fn = getattr(fn, "fn", fn)
    fn = inspect.unwrap(fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _rank():
    if ST.rank is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            ST.rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            ST.rank = "?"
    return ST.rank


class _Slot:
    __slots__ = ("qo", "kv", "lens", "ws", "event", "recorded")


class _Ring:
    """SLOTS x (pinned qo/kv/lens staging + a page-locked int workspace + a CUDA event)."""

    def __init__(self, state, wrapper):
        import torch
        mt = int(state.max_tokens)
        self.arange = torch.arange(mt + 1, dtype=torch.int32)   # host-only source of the clamp (never copied)
        self.slots = []
        ws0 = wrapper._pin_memory_int_workspace_buffer
        if not ws0.is_pinned():
            raise RuntimeError("the wrapper's int workspace is not page-locked")
        for i in range(SLOTS):
            s = _Slot()
            s.qo = torch.empty(mt + 1, dtype=torch.int32, pin_memory=True)
            s.kv = torch.empty(mt + 1, dtype=torch.int32, pin_memory=True)
            s.lens = torch.full((mt,), state.topk_width, dtype=torch.int32).pin_memory()
            s.ws = ws0 if i == 0 else torch.zeros(ws0.shape, dtype=ws0.dtype, pin_memory=True)
            s.event = torch.cuda.Event()
            s.recorded = False
            self.slots.append(s)
        self.next = 0
        self.extra_bytes = sum(t.numel() * t.element_size() for s in self.slots for t in (s.qo, s.kv, s.lens)) + \
            (SLOTS - 1) * ws0.numel() * ws0.element_size()


_RINGS: list = []   # (state, ring) of every _SM90State that planned through the ring (one per process in production)


def _fall_back() -> None:
    """Switch the process back to production's plan AS PRODUCTION RUNS IT: wait for every async copy still reading a
    ring slot, give each wrapper back its own page-locked int workspace (slot 0's), and drop the pinned staging from
    the production attribute names (``_arange_cpu = None`` -> production's plan re-allocates its PAGEABLE staging on
    the next call). Without this, production's plan would keep staging in the last slot's PINNED tensors with the shared
    int workspace and no ring - the unprotected "naive pinning" schedule race of docs/MLA_PLAN_PIN.md 2.1."""
    ST.enabled = False
    for state, ring in _RINGS:
        for sl in ring.slots:
            try:
                if sl.recorded:
                    sl.event.synchronize()
            except Exception:  # noqa: BLE001
                pass
        try:
            state.wrapper._pin_memory_int_workspace_buffer = ring.slots[0].ws
        except Exception:  # noqa: BLE001
            pass
        state._arange_cpu = state._qo_cpu = state._kv_cpu = state._lens_cpu = None


def make_plan(orig_plan):
    def plan(self, num_tokens: int, kv_lens) -> None:
        import torch
        if not ST.enabled:
            return orig_plan(self, num_tokens, kv_lens)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInferMLASparseSM90 plan() called inside CUDA graph "
                "capture; lengths must be planned host-side before capture."
            )
        w = self.wrapper
        ring = getattr(self, "_glm53_planpin_ring", None)
        if ring is None:
            try:
                ring = _Ring(self, w)
            except Exception as exc:  # noqa: BLE001
                _fall_back()
                _log.warning("glm53_mla_planpin: rank %s ring allocation failed (%r): production's pageable plan "
                             "staging from now on", _rank(), exc)
                return orig_plan(self, num_tokens, kv_lens)
            self._glm53_planpin_ring = ring
            _RINGS.append((self, ring))
        si = ring.next
        ring.next = (si + 1) % SLOTS
        sl = ring.slots[si]
        if sl.recorded and not sl.event.query():
            ST.waits += 1
            sl.event.synchronize()      # the async copies from this slot's host bytes (2 plans ago) have not run yet
        # production's staging arithmetic, byte for byte, into the slot's page-locked tensors
        torch.clamp(ring.arange, max=num_tokens, out=sl.qo)
        torch.mul(sl.qo, self.topk_width, out=sl.kv)
        sl.lens.fill_(self.topk_width)
        sl.lens[:num_tokens] = kv_lens.to(torch.int32)
        # flashinfer's MLAPlan writes the work schedule into this host buffer and cudaMemcpyAsync's it (no sync)
        w._pin_memory_int_workspace_buffer = sl.ws
        w.plan(
            sl.qo,
            sl.kv,
            self.kv_indices,
            sl.lens,
            self.num_heads,
            self.kv_lora_rank,  # head_dim_ckv
            self.qk_rope_head_dim,  # 0 (NoPE) or 64 (rope MLA)
            1,  # page_size: top-k slots are the page table
            False,  # causal: encoded by the indexer's selection
            self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=self.kv_dtype,
        )
        sl.event.record()               # current stream: after the wrapper's last async H2D copy from this slot
        sl.recorded = True
        # production's attribute names keep pointing at the staging of the latest plan (read-only users, tests)
        self._arange_cpu, self._qo_cpu, self._kv_cpu, self._lens_cpu = ring.arange, sl.qo, sl.kv, sl.lens
        ST.plans += 1
        if ST.selftest is None:
            _self_test(self, w, sl, ring)
        elif ST.plans == PROOF_AT:
            _log.info("[glm53-mla-planpin] rank %s serving confirmed (mode on): %d plans through the pinned ring, "
                      "ring waits %d", _rank(), ST.plans, ST.waits)
        elif ST.plans % STATS_EVERY == 0:
            _log.info("[glm53-mla-planpin] rank %s mode on: %d plans through the pinned ring, ring waits %d", _rank(),
                      ST.plans, ST.waits)

    plan._glm53_planpin = True
    plan._glm53_orig = orig_plan
    plan.__doc__ = getattr(orig_plan, "__doc__", None)
    return plan


def _self_test(state, w, sl, ring) -> None:
    import torch
    try:
        sl.event.synchronize()
        n = int(state.max_tokens)
        same = [torch.equal(w._qo_indptr_buf[: n + 1].cpu(), sl.qo), torch.equal(w._kv_indptr_buf[: n + 1].cpu(), sl.kv),
                torch.equal(w._kv_len_arr_buf[:n].cpu(), sl.lens)]
        n_same = sum(same)
        ok = n_same == 3
        detail = f"{n_same}/3 device plan buffers equal"
    except Exception as exc:  # noqa: BLE001
        ok, n_same, detail = False, 0, f"raised {exc!r}"
    ST.selftest = ok
    if ok:
        _log.info("glm53_mla_planpin: rank %s self-test: %d/3 device plan buffers == pinned staging (max_tokens %d: "
                  "indptr %d B, lens %d B per slot page-locked, %d slots, +%.1f MiB pinned) PROOF mode=on", _rank(),
                  n_same, state.max_tokens, 4 * (state.max_tokens + 1), 4 * state.max_tokens, SLOTS,
                  ring.extra_bytes / 2**20)
    else:
        _fall_back()
        _log.warning("glm53_mla_planpin: rank %s self-test FAILED (%s): production's pageable plan staging from now on",
                     _rank(), detail)


def install_now(sm90_mod) -> bool:
    try:
        cls = sm90_mod._SM90State
        if getattr(cls.plan, "_glm53_planpin", False):
            return True
        fi = importlib.import_module(FI_MODULE)
        fps = {
            "_SM90State.plan": source_fingerprint(cls.plan),
            "BatchMLAPagedAttentionWrapper.plan": source_fingerprint(fi.BatchMLAPagedAttentionWrapper.plan),
        }
        bad = {k: v for k, v in fps.items() if v not in VERIFIED[k]}
        if bad:
            ST.reason = "unverified source: " + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
            _log.warning("glm53_mla_planpin: NOT installed: %s (production's pageable plan staging unchanged)",
                         ST.reason)
            return False
        cls.plan = make_plan(cls.plan)
        ST.installed = ST.enabled = True
        _log.info("glm53_mla_planpin: patched _SM90State.plan (pid %d): page-locked staging, %d-slot ring with a "
                  "per-slot CUDA event and int workspace (sources verified: %s) PROOF mode=on", os.getpid(), SLOTS,
                  ", ".join(f"{k} {v}" for k, v in sorted(fps.items())))
        return True
    except Exception as exc:  # noqa: BLE001
        ST.reason = f"install failed: {exc!r}"
        _log.warning("glm53_mla_planpin: NOT installed: install failed (production's pageable plan staging "
                     "unchanged): %r", exc)
        return False


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != SM90_MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            install_now(module)
        spec.loader.exec_module = exec_module
        return spec


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_MLA_PLAN_PIN=1."""
    raw = os.environ.get(ENV)
    try:
        mode = env_mode()
    except ValueError as exc:
        _log.warning("glm53_mla_planpin plugin loaded (pid %d): -> off (%s)", os.getpid(), exc)
        return
    _log.info("glm53_mla_planpin plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, raw,
              "installing (mode on)" if mode == "on" else "off, production's pageable plan staging unchanged")
    if mode == "off":
        return
    ST.mode = mode
    try:
        if SM90_MODULE in sys.modules:
            install_now(sys.modules[SM90_MODULE])
        elif not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_mla_planpin: NOT installed: plugin install failed (production path unchanged): %r", exc)


def summary() -> dict:
    return {"mode": ST.mode, "installed": ST.installed, "enabled": ST.enabled, "plans": ST.plans, "waits": ST.waits,
            "selftest": ST.selftest, "reason": ST.reason}
