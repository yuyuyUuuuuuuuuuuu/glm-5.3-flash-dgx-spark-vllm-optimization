"""GLM53_MLA_EXACT_LENS: plan production's sparse-MLA FA2 with the EXACT per-row key counts (decode over-read fix).

The defect (docs/MLA_PREFILL.md section 6, found 2026-09-28, never fixed; docs/MLA_EXACT_LENS.md)
------------------------------------------------------------------------------------------------
Production's FLASHINFER_MLA_SPARSE_SM90 backend (the launcher-mounted ``flashinfer_mla_sparse_sm90.py.patched``) plans
FlashInfer's FA2 on the host, once per step, with ``lens = ctx if ctx <= index_topk else index_topk + ctx % index_kpool``
(``FlashInferMLASparseSM90Builder._kv_lens_host``). The GLM5Next kpool indexer, however, keeps ``select_k - 1`` = 511
pools (``pool_ids[:, : select_k - 1]``, sparse_attn_indexer_kpool.py ~:581 / ~:875) plus the trailing incomplete pool,
so the real number of selected keys of a row with ``ctx >= index_topk`` (2048) is ``index_topk - index_kpool +
ctx % index_kpool`` = 2044 + ctx % 4, i.e. **4 fewer** than planned. FA2 reads ``kv_indices[row * W + j]`` for every
``j < planned`` (W = 2048, the top-k buffer width), so every such row ALSO attends

  * ``4 - ctx % 4`` copies of global KV slot 0 (the -1 tail of the converted row is clamped to slot 0 in
    ``forward_mqa``), and
  * when ``ctx % 4 != 0``, the first ``ctx % 4`` keys of the NEXT row (the next query's first selected pool: for a
    spec-verify step mostly the same request's next position), or for the last row of the step stale slot ids an
    earlier call left in the process-wide ``kv_indices`` buffer.

With ``GLM53_MLA_PREFILL=1`` (production) prefill runs the exact kernel (device-side valid counts), so only DECODE has
the extra keys: decode and prefill attend different key sets -> the decode-vs-prefill consistency KL. It also explains
the 2026-10-01 production A/B of GLM53_KPOOL_DROP_LOWEST (dvp 0.0075 -> 0.0109): that patch orders each row's pools by
score, so the next row's first pool - the one the over-read duplicates - became the next query's TOP-scored pool.

The fix (GLM53_MLA_EXACT_LENS=1; unset/0 = production byte for byte)
--------------------------------------------------------------------
``FlashInferMLASparseSM90Builder.build`` is replaced by the same three statements (fingerprint-checked against the
production source; a different source -> not installed, production path) with one change: the planned lengths go
through ``exact_lens`` before ``plan()``:

  rows with planned ``lens < index_topk``    -> unchanged (ctx < 2048: every pool + the tail is selected, valid = ctx)
  rows with planned ``lens >= index_topk``   -> ``lens - index_kpool`` (= 2044 + ctx % 4, the kpool path)
     EXCEPT rows of a prefill request in a SHORT-prefill step (the indexer's own predicate: every prefill request's
     seq_len <= index_topk, sparse_attn_indexer_kpool.py ~:443): those rows take the identity top-k (all ``ctx``
     positions), and a planned ``lens >= index_topk`` there means ctx == index_topk exactly: valid = 2048, unchanged.
  The decode/prefill split is vLLM's own ``split_decodes_and_prefills`` with the indexer builder's arguments
  (decode_threshold = 1 + num_speculative_tokens, require_uniform = not (flattening or varlen), short extends as
  decodes), host values only (query_start_loc_cpu, seq_lens_cpu_upper_bound which is exact for prefill rows).

The corrected lengths are never larger than production's, so the change can only REMOVE reads (no new address can
be touched; the clamp to slot 0 in forward_mqa stays). It runs in the builder, outside CUDA-graph capture, host-side
only (no device sync is added; ``_kv_lens_host`` - with or without GLM53_DEC_HOSTLOOP's wrapper - is called exactly
as production calls it). Decode numerics change (by design: the extra keys disappear); prefill on the exact kernel
does not change at all; prefill on FA2 (eager chunks < GLM53_MLA_PREFILL_MIN_TOKENS rows, or GLM53_MLA_PREFILL unset)
loses the same extra keys.

Self-test (once per process, at the first ``build`` on a CUDA device, outside capture): the image's REAL ops at
production's index_topk / index_kpool - ``torch.ops._C.persistent_topk`` (decode), ``top_k_per_row_prefill``
(prefill), the ``pool_ids[:, : select_k - 1]`` truncation, ``expand_pools_and_append_tail`` and the backend's
``triton_convert_req_index_to_global_index(return_valid_counts=True)`` - over contexts around every boundary
(< 2048, == 2048, 2049..2051, multiples of 4, long) must give EXACTLY ``exact_lens``'s numbers, and the identity
(short-prefill) rows ``ctx``. Any mismatch or error -> WARNING, the fix stays off for the process (production's
lengths), never an exception into vLLM.

Logs: "[glm53-mla-exactlens] installed ..." (plugin), "[glm53-mla-exactlens] self-test passed ..." / "... FAILED ...",
then "N builds corrected" at 1, 10, 1000, 100000 builds.
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

_log = logging.getLogger("vllm.glm53_mla_exactlens")
ENV = "GLM53_MLA_EXACT_LENS"
TAG = "[glm53-mla-exactlens]"
_OFF = frozenset({"", "0", "off", "false", "no"})
SM90_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"

# sha256(ast.dump(source))[:16] (integrate.source_fingerprint) of the production functions, read from the production
# image with the launcher's vllm-patches/flashinfer_mla_sparse_sm90.py.patched mounted (tests/mla_exactlens/
# test_exactlens_unit.py prints and checks them). _kv_lens_host is the same function GLM53_DEC_HOSTLOOP pins.
VERIFIED = {
    "FlashInferMLASparseSM90Builder.build": frozenset({"819d7860e255a9f3"}),
    "FlashInferMLASparseSM90Builder._kv_lens_host": frozenset({"281328403e853281"}),
}


class _State:
    installed = False
    enabled = False
    selftest = None          # None = not yet run, True/False = result
    selftest_detail = ""
    builds = 0
    corrected_rows = 0
    reason = ""


STATE = _State()


def _on() -> bool:
    return os.environ.get(ENV, "").strip().lower() not in _OFF


def source_fingerprint(fn) -> str | None:
    fn = getattr(fn, "fn", fn)
    fn = inspect.unwrap(fn)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the arithmetic (pure, unit-tested)
def exact_from_planned(lens, short_row, topk: int, kpool: int):
    """Production's planned lens (torch int32 [rows]) -> exact valid counts. short_row: bool tensor [rows] (rows of a
    prefill request in a short-prefill step) or None (no such row)."""
    import torch
    if kpool <= 1:
        return lens
    big = lens >= topk
    if short_row is not None:
        big = big & ~short_row
    return torch.where(big, lens - kpool, lens).to(lens.dtype)


def valid_reference(ctx, short_row, topk: int, kpool: int):
    """What the indexer + expand + convert produce, from the token context of each row (tests only)."""
    import torch
    sel = (topk // kpool - 1) * kpool + ctx % kpool
    v = torch.where(ctx < topk, ctx, sel)
    if short_row is not None:
        v = torch.where(short_row, ctx, v)
    return v


def _decode_threshold(builder) -> int:
    th = getattr(builder, "_glm53_exl_dth", None)
    if th is None:
        n_spec = 0
        try:
            sc = builder.vllm_config.speculative_config
            n_spec = int(sc.num_speculative_tokens) if sc is not None else 0
        except Exception:  # noqa: BLE001
            n_spec = 0
        th = n_spec + 1
        builder._glm53_exl_dth = th
    return th


def short_prefill_rows(builder, cam, num_rows: int):
    """Bool tensor [num_rows]: rows of prefill requests in a step where the indexer takes its short-prefill path
    (every prefill request's seq_len <= index_topk); None when no row can be one (the common case: no prefill)."""
    import torch
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills
    topk = int(builder._index_topk)
    next_n = _decode_threshold(builder)
    try:
        from vllm.platforms import current_platform
        sm100 = bool(current_platform.is_device_capability_family(100))
    except Exception:  # noqa: BLE001
        sm100 = False
    use_flattening = (not sm100) and next_n not in (1, 2)
    try:
        from vllm.v1.attention.backends.mla.indexer import _supports_varlen_paged_mqa_logits
        varlen = bool(_supports_varlen_paged_mqa_logits())
    except Exception:  # noqa: BLE001
        varlen = False
    nd, npf, ndt, _ = split_decodes_and_prefills(cam, decode_threshold=next_n,
                                                  require_uniform=not (use_flattening or varlen),
                                                  treat_short_extends_as_decodes=True)
    if npf == 0:
        return None
    ub = cam.seq_lens_cpu_upper_bound
    num_reqs = int(cam.num_reqs)
    if ub is None or int(ub[nd:num_reqs].max()) > topk:
        return None                      # not a short-prefill step: every row takes the kpool path
    out = torch.zeros(num_rows, dtype=torch.bool)
    out[int(ndt):] = True
    return out


def exact_lens(builder, cam, num_rows: int, lens):
    topk = int(builder._index_topk)
    kpool = max(int(builder._index_kpool), 1)
    if kpool <= 1 or num_rows == 0 or not bool((lens >= topk).any()):
        return lens
    short = short_prefill_rows(builder, cam, num_rows)
    new = exact_from_planned(lens, short, topk, kpool)
    STATE.corrected_rows += int((new != lens).sum())
    return new


# ------------------------------------------------------------------ self-test against the image's real ops
def self_test(topk: int = 2048, kpool: int = 4, device: str = "cuda") -> tuple[bool, str]:
    import torch
    import vllm  # noqa: F401  registers torch.ops._C.*
    from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail
    from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
    select_k = topk // kpool
    ctxs = [1, 7, 1500, topk - 5, topk - 4, topk - 1, topk, topk + 1, topk + 2, topk + 3, topk + 4, topk + 5,
            topk + 7, 3001, 6000, 6003, 14401, 100000, 100003]
    rows = len(ctxs)
    ctx = torch.tensor(ctxs, dtype=torch.int32, device=device)
    max_pools = max(ctxs) // kpool + 1
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    logits = torch.randn((rows, max_pools), generator=g, device=device, dtype=torch.float32)
    pools = (ctx // kpool).to(torch.int32)
    block = 64
    nblk = (max(ctxs) + block) // block + 1
    block_table = torch.arange(nblk, dtype=torch.int32, device=device).unsqueeze(0).contiguous()
    req = torch.zeros(rows, dtype=torch.int32, device=device)
    width = topk

    def valid_of(pool_topk):
        buf = torch.full((rows, width), -1, dtype=torch.int32, device=device)
        exp = expand_pools_and_append_tail(pool_topk.to(torch.int64)[:, : select_k - 1], ctx, kpool)
        buf[:, : exp.shape[-1]] = exp
        _, vc = triton_convert_req_index_to_global_index(req, block_table, buf, BLOCK_SIZE=block,
                                                         NUM_TOPK_TOKENS=width, return_valid_counts=True)
        return vc.to(torch.int64).cpu()

    # decode: persistent_topk over the pool-granular lengths (production ~:815)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    dst = torch.full((rows, select_k), -1, dtype=torch.int32, device=device)
    torch.ops._C.persistent_topk(logits, pools, dst, ws, select_k, max_pools)
    v_dec = valid_of(dst)
    # prefill: top_k_per_row_prefill over [0, pools) (production ~:559)
    dst2 = torch.full((rows, select_k), -1, dtype=torch.int32, device=device)
    ks = torch.zeros(rows, dtype=torch.int32, device=device)
    torch.ops._C.top_k_per_row_prefill(logits, ks, pools, dst2, rows, logits.stride(0), logits.stride(1), select_k)
    v_pf = valid_of(dst2)
    # short prefill (identity rows, ctx <= topk only; production ~:470)
    short_ctx = [c for c in ctxs if c <= topk]
    ar = torch.arange(width, dtype=torch.int32, device=device)
    sc = torch.tensor(short_ctx, dtype=torch.int32, device=device)
    sbuf = ar[None, :].repeat(len(short_ctx), 1)
    sbuf[ar[None, :] > (sc - 1)[:, None]] = -1
    _, vs = triton_convert_req_index_to_global_index(torch.zeros(len(short_ctx), dtype=torch.int32, device=device),
                                                     block_table, sbuf.contiguous(), BLOCK_SIZE=block,
                                                     NUM_TOPK_TOKENS=width, return_valid_counts=True)
    v_short = vs.to(torch.int64).cpu()

    ctx_cpu = torch.tensor(ctxs, dtype=torch.int64)
    planned = torch.where(ctx_cpu <= topk, ctx_cpu, topk + ctx_cpu % kpool).to(torch.int32)   # production's formula
    want_kpool = exact_from_planned(planned, None, topk, kpool).to(torch.int64)
    sc_cpu = torch.tensor(short_ctx, dtype=torch.int64)
    planned_s = torch.where(sc_cpu <= topk, sc_cpu, topk + sc_cpu % kpool).to(torch.int32)
    want_short = exact_from_planned(planned_s, torch.ones(len(short_ctx), dtype=torch.bool), topk, kpool).to(torch.int64)
    bad = []
    if not torch.equal(v_dec, want_kpool):
        bad.append(f"decode valid {v_dec.tolist()} != exact {want_kpool.tolist()}")
    if not torch.equal(v_pf, want_kpool):
        bad.append(f"prefill valid {v_pf.tolist()} != exact {want_kpool.tolist()}")
    if not torch.equal(v_short, want_short):
        bad.append(f"short-prefill valid {v_short.tolist()} != exact {want_short.tolist()}")
    if bool((want_kpool > planned.to(torch.int64)).any()):
        bad.append("an exact length exceeds production's planned length")
    over = int((planned.to(torch.int64) - v_dec).clamp_min(0).sum())
    if bad:
        return False, "; ".join(bad)
    return True, (f"{rows} decode + {rows} prefill + {len(short_ctx)} short-prefill rows on the real ops == exact_lens; "
                  f"production's plan over-reads {over} keys on these rows")


# ------------------------------------------------------------------ the patched build
def make_build(cls, sm90_mod):
    parent_build = cls.__mro__[1].build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = parent_build(self, common_prefix_len, common_attn_metadata, fast_build)
        state = sm90_mod._SM90_STATE
        if state is not None:
            num_rows, kv_lens = self._kv_lens_host(common_attn_metadata)
            if STATE.enabled:
                try:
                    if STATE.selftest is None:
                        _run_self_test(self)
                    if STATE.selftest:
                        kv_lens = exact_lens(self, common_attn_metadata, num_rows, kv_lens)
                        STATE.builds += 1
                        if STATE.builds in (1, 10, 1000, 100000):
                            _log.info("%s %d builds planned with exact lengths (%d rows corrected so far)", TAG,
                                      STATE.builds, STATE.corrected_rows)
                except Exception as exc:  # noqa: BLE001
                    STATE.enabled = False
                    _log.warning("%s exact_lens failed (%r): production's lengths from now on", TAG, exc)
            state.plan(num_rows, kv_lens)
        return metadata

    build._glm53_exactlens = True
    build._glm53_orig = cls.build
    build.__doc__ = getattr(cls.build, "__doc__", None)
    return build


def _run_self_test(builder) -> None:
    import torch
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        ok, detail = self_test(int(builder._index_topk), max(int(builder._index_kpool), 1),
                               device=str(getattr(builder, "device", "cuda")))
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"raised {exc!r}"
    STATE.selftest, STATE.selftest_detail = ok, detail
    if ok:
        _log.info("%s self-test passed: %s", TAG, detail)
    else:
        STATE.enabled = False
        _log.warning("%s self-test FAILED (%s): production's planned lengths stay", TAG, detail)


def install_now(sm90_mod) -> bool:
    try:
        cls = sm90_mod.FlashInferMLASparseSM90Builder
        if getattr(cls.build, "_glm53_exactlens", False):
            return True
        fps = {
            "FlashInferMLASparseSM90Builder.build": source_fingerprint(cls.build),
            "FlashInferMLASparseSM90Builder._kv_lens_host": source_fingerprint(
                getattr(cls._kv_lens_host, "_glm53_orig", cls._kv_lens_host)),
        }
        bad = {k: v for k, v in fps.items() if v not in VERIFIED[k]}
        if bad:
            STATE.reason = "unverified source: " + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
            _log.warning("%s NOT installed: %s (production path unchanged)", TAG, STATE.reason)
            return False
        cls.build = make_build(cls, sm90_mod)
        STATE.installed = STATE.enabled = True
        _log.info("%s installed (pid %d): FA2 sparse-MLA plan uses the exact selected-key counts (2044 + ctx %% 4 "
                  "for ctx >= index_topk; short-prefill rows unchanged); self-test at the first build", TAG, os.getpid())
        return True
    except Exception as exc:  # noqa: BLE001
        STATE.reason = f"install failed: {exc!r}"
        _log.warning("%s install failed (production path unchanged): %r", TAG, exc)
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
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_MLA_EXACT_LENS is on."""
    try:
        if not _on():
            return
        if SM90_MODULE in sys.modules:
            install_now(sys.modules[SM90_MODULE])
        elif not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _log.warning("%s plugin install failed (production path unchanged): %r", TAG, exc)
