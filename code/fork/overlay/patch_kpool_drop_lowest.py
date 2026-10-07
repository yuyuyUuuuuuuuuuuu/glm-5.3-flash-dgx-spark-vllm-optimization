#!/usr/bin/env python3
"""[glm53-kpool-drop-lowest] The kpool indexer must drop the LOWEST-scored pool, not an arbitrary one.

The defect
----------
The GLM-5.3-Flash sparse indexer pools every ``index_kpool`` (4) consecutive keys into one fp8 entry, so the
sparse-attention top-k selects ``select_k = topk_tokens // index_kpool`` POOLS (512 at topk_tokens 2048) and
``expand_pools_and_append_tail`` then expands them back to tokens, appending the request's trailing incomplete
pool as the last one: the history region needs ``select_k - 1`` pools + the tail = ``select_k`` pools =
``topk_tokens`` tokens. The pool ids come from

  decode : ``torch.ops._C.persistent_topk``                       (sparse_attn_indexer_kpool.py ~:815)
  prefill: ``torch.ops._C.top_k_per_row_prefill``                 (~:559)

Both return the top-k SET deterministically, but their ORDER is not deterministic (which of the selected pools
lands in which column varies per run; tests/probe_persistent_topk_det2.py, branch fa2plan). The callers convert
without sorting (~:849 / ~:571) and then keep ``pool_ids[:, : select_k - 1]`` (~:875 decode, ~:581 prefill) -
i.e. they drop whichever pool happens to sit in the LAST column, an arbitrary pool per run, instead of the
lowest-scored one. A wrongly dropped pool is one of the ``select_k`` best candidates of the sparse MLA, so the
attended token set differs run to run (quality noise, and the decode-vs-prefill consistency probe drifts).

The fix (GLM53_KPOOL_DROP_LOWEST=1; unset/empty/0 = production byte-identical)
-----------------------------------------------------------------------------
Keep the ``select_k - 1`` HIGHEST-scored valid pools, by score, deterministically:

  scores  = logits.gather(1, pool_ids.clamp_min(0) [+ cu_seqlen_ks per row in prefill: the prefill op returns
            ks-relative ids over chunk-global logits columns])   the logits the top-k op itself ranked
  invalid (-1 fill, a row with fewer valid pools than select_k) score -inf, so a valid pool is NEVER dropped
  while a -1 is kept; if the row has fewer than ``keep`` valid pools the -1s stay (the expand kernel maps
  pid < 0 to -1 tokens, kpool_compress.py ``hist_out = tl.where(pid >= 0, hist_val, -1)``) - unchanged semantics.
  key     = (order_preserving_uint32(score) - 2**31) * 2**32 + (2**32 - 1 - pool_id)
            the classic float bit trick: a non-negative float keeps its fp32 bits with the sign bit set, a
            negative one is bit-inverted, so the unsigned 32-bit results order like the float (incl. +-0/+inf;
            NaN would sort high, but a selected pool's logit is a real number). Subtracting 2**31 turns them
            signed, which the int64 key needs: an unsigned high half shifted up would wrap into the sign bit and
            put every positive score below every negative one under torch's SIGNED descending sort, and reading
            the uints as int32 instead would break the order at 2**31. The high half is multiplied by 2**32 and
            the low half ADDED (not '|'), so negative highs stay exact; the low half tie-breaks equal scores by
            the LOWER pool id, which makes every key distinct: the result is a function of the SET
            {pool ids, scores} alone, not of the op's column order.
  torch.sort(key, dim=-1, descending=True) -> first keep columns -> gather the pool ids

No host sync (gather / where / sort, no .item(), no nonzero, no boolean-mask indexing) and CUDA-graph safe: the
shapes are [rows, select_k] and [rows, keep] - rows and select_k are static per captured batch shape, nothing is
allocated at replay time that capture did not allocate (the temporaries join the graph's private pool, like
``pool_topk`` / the persistent-topk workspace already do). Decode runs inside FULL graphs, so the patched
statement must not change any shape: it returns exactly the ``select_k - 1`` columns the truncated slice had.

opt-decodekit (2026-10-03): by default the installed helper runs ONE Triton kernel that drops the lowest-scored
pool and keeps the op's column order (the score-sorted order raised the decode-vs-prefill KL 0.007 -> 0.0105 on the
handoff mini; the torch helper cost ~0.65 ms per decode step). The kept SET is deterministic; the column order is the
op's, as in stock production. GLM53_KPOOL_DROP_LOWEST_ORDER=fusedsort|sorted restores the sorted order (see HELPER).

Determinism note (sorted orders): torch.sort along the last dim is a segmented radix sort (deterministic for a fixed input), and
the keys are unique, so the kept ids AND their order are bitwise reproducible; duplicated pool ids in one row
(the op never emits them, but if it did) share a key and produce the same VALUE either way, so the output tensor
is still bitwise identical.

Scope: ONLY the two ``expand_pools_and_append_tail`` call sites that truncate to ``select_k - 1``. The third
kpool path (prefill with ``positions is None``, ``expand_pools_to_tokens`` with the full ``select_k`` columns
and a validity mask) is untouched - it drops nothing. The eager-indexer (non-kpool) file is untouched.

Wiring: registered in overlay/patch_tf_bundle.py (runs after patch_kpool_tail_ring.py, which owns a different
span of this file), forwarded to BOTH ranks by the r16l start.sh (tools/deploy16/make_start_sh.py --stage r16l),
operator switch `tools/deploy16/env_r16.sh on|off kpooldown`, boot_checks line, docs/KPOOL_DROP_LOWEST.md.

Idempotency / fail-closed: pristine -> patched; patched (marker + the exact patched text) -> no-op ("already
present"); anything else (partial marker, drifted anchor, a span another overlay moved) -> SystemExit, which
stops the container start. Pre-preflighted before anything is written; atomic replace; pyc cleared.
Env override for tests: GLM53_SPARSE_INDEXER_KPOOL_PY (same name as patch_kpool_tail_ring.py uses).
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
INDEXER_PY = Path(
    os.environ.get("GLM53_SPARSE_INDEXER_KPOOL_PY", SITE / "model_executor/layers/sparse_attn_indexer_kpool.py")
)
ENV = "GLM53_KPOOL_DROP_LOWEST"
TAG = "[glm53-kpool-drop-lowest]"
MARK = "[glm53-kpool-drop-lowest]"

# ---- the helper, inserted before `def sparse_attn_indexer_kpool(` (module level, the only consumer scope)
HELPER_ANCHOR = "@eager_break_during_capture\ndef sparse_attn_indexer_kpool("
HELPER = '''# [glm53-kpool-drop-lowest] Keep the `keep` (= select_k - 1) HIGHEST-scored pools of every row and drop the
# lowest-scored one, deterministically. torch.ops._C.persistent_topk / top_k_per_row_prefill return the top-k SET
# in a nondeterministic ORDER, so the previous `pool_ids[:, :select_k - 1]` dropped an arbitrary pool per run. The
# scores are the SAME logits the top-k op ranked, gathered on device; invalid (-1) fills score -inf, so a valid
# pool is never dropped while a -1 is kept (a row with fewer valid pools than `keep` keeps its -1s, which the
# expand kernel maps to -1 as before). Equal scores tie-break by the LOWER pool id: every sort key is distinct,
# so the kept ids and their order are bitwise reproducible across runs. Fixed shapes [rows, select_k] / [rows,
# keep], no host sync, no data-dependent shape or allocation: safe inside FULL decode CUDA graphs.
# [opt-decodekit] GLM53_KPOOL_DROP_LOWEST_ORDER (nodeC experiments only; start.sh does not forward it):
#   unset/"fused" = ONE Triton kernel per call, order-PRESERVING: the op's column order with the lowest-scored column
#                   removed (bitwise == the torch "stock" variant below). ~2.3 us per call = the cost of the stock
#                   `.to(int64)[:, :keep]` it replaces (the r16l torch helper: ~58 us per call = ~0.65 ms per decode
#                   step over the 11 MLA indexers; tests/kpool_drop_lowest_bench.py). Default because the score-SORTED
#                   order is what raised the decode-vs-prefill KL (handoff mini with GLM53_MLA_PREFILL=1: stock
#                   0.0057/0.0071/0.0084, sorted 0.0103/0.0109, order-preserving 0.0081; docs/OPT_DECODEKIT.md 3).
#   "fusedsort"   = one Triton kernel, BITWISE == the r16l torch helper (score order), ~3.9 us per call
#   "sorted"      = the r16l torch helper;  "stock" = the torch order-preserving drop
# The kept SET is identical in all four. Unexpected layouts (non-fp32 logits, non-unit column stride, keep !=
# select_k - 1) fall back to the torch code of the selected order.
_GLM53_KPOOL_DL_ORDER = __import__("os").environ.get("GLM53_KPOOL_DROP_LOWEST_ORDER", "").strip() or "fused"
_GLM53_KPOOL_DL_STOCK_ORDER = _GLM53_KPOOL_DL_ORDER in ("stock", "fused")  # fused falls back to stock
_GLM53_KPOOL_DL_FUSED = _GLM53_KPOOL_DL_ORDER in ("fused", "fusedsort")
if _GLM53_KPOOL_DL_FUSED:
    from vllm.triton_utils import tl as _dl_tl, triton as _dl_triton

    @_dl_triton.jit
    def _glm53_kpool_drop_lowest_kernel(
        logits_ptr, logits_stride, ids_ptr, ids_stride, off_ptr, out_ptr, out_stride,
        SELECT_K: _dl_tl.constexpr, BLOCK: _dl_tl.constexpr, HAS_OFF: _dl_tl.constexpr, SORTED: _dl_tl.constexpr,
    ):
        # one program per row, int64 out [rows, SELECT_K - 1]
        r = _dl_tl.program_id(0).to(_dl_tl.int64)
        c = _dl_tl.arange(0, BLOCK)
        m = c < SELECT_K
        km = c < SELECT_K - 1
        ids = _dl_tl.load(ids_ptr + r * ids_stride + c, mask=m, other=-1).to(_dl_tl.int64)
        idc = _dl_tl.maximum(ids, 0)
        col = idc
        if HAS_OFF:
            col = col + _dl_tl.load(off_ptr + r).to(_dl_tl.int64)
        sc = _dl_tl.load(logits_ptr + r * logits_stride + col, mask=m & (ids >= 0), other=float("-inf"))
        # the torch helper's key: order-preserving fp32 bits (signed high half: -0.0 < +0.0 like the helper), then
        # the LOWER id first. Low half = 2**32 - 2 - id (id + 1 in [0, 2**32 - 1]: a -1 fill gets its own value
        # 2**32 - 1 and the id is recovered from the key alone; among equal scores a -1 sorts before id 0 - the torch
        # helper ties them, which needs a selected VALID pool with a -inf logit: never). Padding columns sort last.
        bits = sc.to(_dl_tl.int32, bitcast=True).to(_dl_tl.int64) & 0xFFFFFFFF
        neg = (bits >> 31) == 1
        high = _dl_tl.where(neg, (~bits) & 0xFFFFFFFF, bits | 0x80000000) - 0x80000000
        key = high * 4294967296 + (4294967294 - ids)
        if SORTED:  # == the r16l helper: the keep highest keys, in key order
            key = _dl_tl.where(m, key, -9223372036854775807 - 1)
            ks = _dl_tl.sort(key, 0, descending=True)
            v = 4294967294 - (ks & 0xFFFFFFFF)
            _dl_tl.store(out_ptr + r * out_stride + c, v, mask=km)
        else:  # == the torch "stock" variant: drop the FIRST column holding the minimum key, keep the op's order
            key = _dl_tl.where(m, key, 9223372036854775807)
            kmin = _dl_tl.min(key, 0)
            drop = _dl_tl.min(_dl_tl.where(key == kmin, c, BLOCK), 0)
            src = _dl_tl.where(c < drop, c, c + 1)
            v = _dl_tl.load(ids_ptr + r * ids_stride + src, mask=km, other=-1).to(_dl_tl.int64)
            _dl_tl.store(out_ptr + r * out_stride + c, v, mask=km)


def _kpool_keep_highest_pools(
    logits: torch.Tensor,  # [rows, num_pools] fp32: the logits the top-k op ranked
    pool_ids: torch.Tensor,  # [rows, select_k] int, the selected pools (-1 = no pool)
    keep: int,
    col_off: torch.Tensor | None = None,  # [rows] logits column of pool 0 (prefill: cu_seqlen_ks); None = 0
) -> torch.Tensor:  # [rows, keep] int64: pool_ids, the `keep` highest scores first
    if (_GLM53_KPOOL_DL_FUSED and keep == pool_ids.shape[1] - 1 and pool_ids.shape[0] > 0
            and logits.stride(1) == 1 and pool_ids.stride(1) == 1 and logits.dtype == torch.float32):
        rows, sel = pool_ids.shape
        out = torch.empty((rows, keep), dtype=torch.int64, device=pool_ids.device)
        _glm53_kpool_drop_lowest_kernel[(rows,)](
            logits, logits.stride(0), pool_ids, pool_ids.stride(0),
            col_off if col_off is not None else pool_ids, out, out.stride(0),
            SELECT_K=sel, BLOCK=_dl_triton.next_power_of_2(sel), HAS_OFF=col_off is not None,
            SORTED=_GLM53_KPOOL_DL_ORDER == "fusedsort",
        )
        return out
    ids = pool_ids.to(torch.int64)
    # top_k_per_row_prefill returns ids RELATIVE to the row's cu_seqlen_ks (request-local pools), while the
    # prefill logits columns are chunk-global: the op ranked logits[r, ks[r] + id]. Decode ids are columns.
    cols = ids.clamp_min(0)
    if col_off is not None:
        cols = cols + col_off.to(torch.int64).unsqueeze(1)
    scores = logits.gather(1, cols)
    scores = torch.where(ids >= 0, scores, torch.full_like(scores, float("-inf")))
    # float -> order-preserving int64 key: a non-negative float keeps its fp32 bits with the sign bit set, a
    # negative one is bit-inverted, so the unsigned 32-bit results order like the floats; subtracting 2**31
    # (NOT reading them as int32, which would break the order at 2**31) makes them signed, and the signed high
    # half * 2**32 + low half stays inside int64 (an unsigned high half shifted up would wrap the sign bit and
    # misorder positives below negatives under torch's signed sort).
    bits = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    neg = (bits >> 31) == 1
    high = torch.where(neg, (~bits) & 0xFFFFFFFF, bits | 0x80000000) - 0x80000000
    key = high * 4294967296 + (0xFFFFFFFF - ids.clamp_min(0))  # ties: the LOWER pool id sorts first
    if _GLM53_KPOOL_DL_STOCK_ORDER:  # [opt-decodekit experiment] drop the lowest, keep the op's column order
        drop = key.argmin(dim=1, keepdim=True)
        ar = torch.arange(keep, device=ids.device, dtype=torch.int64).unsqueeze(0)
        return ids.gather(1, ar + (ar >= drop).to(torch.int64))
    _, order = torch.sort(key, dim=1, descending=True)
    return ids.gather(1, order[:, :keep])


'''

# ---- prefill (~:571-582): only the `positions is not None` branch truncates; the expand_pools_to_tokens branch
# keeps all select_k columns behind a validity mask and must stay untouched.
PF_ANCHOR = """            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
"""
PF_PATCHED = PF_ANCHOR + """                    # [glm53-kpool-drop-lowest] GLM53_KPOOL_DROP_LOWEST=1: the top-k SET of
                    # top_k_per_row_prefill is deterministic, its ORDER is not, so the old last-column
                    # truncation dropped an arbitrary pool per run. Drop the LOWEST-scored pool
                    # instead (deterministic; -1 fills are dropped first).
                    pool_keep = _kpool_keep_highest_pools(
                        logits, pool_topk, select_k - 1, chunk.cu_seqlen_ks
                    )
"""
PF_OLD = """                    expanded = expand_pools_and_append_tail(
                        pool_ids[:, : select_k - 1], q_seq, index_kpool
                    )
"""
PF_NEW = """                    expanded = expand_pools_and_append_tail(pool_keep, q_seq, index_kpool)
"""

# ---- decode (~:849-875), inside FULL CUDA graphs: same replacement, `logits` is this call's own scores
DEC_ANCHOR = """            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
"""
DEC_PATCHED = DEC_ANCHOR + """            # [glm53-kpool-drop-lowest] GLM53_KPOOL_DROP_LOWEST=1: persistent_topk returns the top-k
            # SET deterministically, its ORDER is not, so the old last-column truncation dropped an
            # arbitrary pool per run. Drop the LOWEST-scored pool instead (deterministic; -1 fills
            # are dropped first). Fixed shapes, no host sync: FULL-graph safe.
            pool_keep = _kpool_keep_highest_pools(logits, pool_topk, select_k - 1)
"""
DEC_OLD = """            out = expand_pools_and_append_tail(
                pool_ids[:, : select_k - 1], dec_seq, index_kpool
            )
"""
DEC_NEW = """            out = expand_pools_and_append_tail(pool_keep, dec_seq, index_kpool)
"""


def prepare(source: str) -> tuple[str, str]:
    n_marks = source.count(MARK)
    if n_marks:
        if n_marks != 3:  # the helper's banner + one comment line in each of the two call sites
            raise ValueError(f"unexpected {MARK} count {n_marks} (3 expected)")
        if source.count(PF_PATCHED) == 1 and source.count(DEC_PATCHED) == 1 and HELPER in source:
            return source, "already present"
        raise ValueError("partial/inconsistent drop-lowest patch (marker present but the patched text differs)")
    if source.count(PF_ANCHOR) != 1 or source.count(PF_OLD) != 1:
        raise ValueError(
            f"prefill anchor drifted (pre={source.count(PF_ANCHOR)}, old={source.count(PF_OLD)}; expected vLLM "
            "487ecf187 as the image ships it, with or without the tail-ring/seed-stride overlays)"
        )
    if source.count(DEC_ANCHOR) != 1 or source.count(DEC_OLD) != 1:
        raise ValueError(f"decode anchor drifted (dec={source.count(DEC_ANCHOR)}, old={source.count(DEC_OLD)})")
    if source.count(HELPER_ANCHOR) != 1:
        raise ValueError(f"helper anchor drifted (helper={source.count(HELPER_ANCHOR)})")
    patched = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    patched = patched.replace(PF_ANCHOR, PF_PATCHED, 1).replace(PF_OLD, PF_NEW, 1)
    patched = patched.replace(DEC_ANCHOR, DEC_PATCHED, 1).replace(DEC_OLD, DEC_NEW, 1)
    if (
        patched.count(MARK) != 3
        or patched.count("pool_ids[:, : select_k - 1]") != 0
        or patched.count("pool_keep = _kpool_keep_highest_pools(logits, pool_topk, select_k - 1)") != 1
        or patched.count("logits, pool_topk, select_k - 1, chunk.cu_seqlen_ks") != 1
        or patched.count("expand_pools_and_append_tail(pool_keep") != 2
        or source.count("pool_keep")
    ):
        raise ValueError("post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.{MARK.strip('[]')}.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob("sparse_attn_indexer_kpool*.pyc"):
            pyc.unlink(missing_ok=True)


def main() -> int:
    val = os.environ.get(ENV, "")
    if not val.strip():
        print(f"{MARK} {ENV} unset -> skipped (stock)")
        return 0
    if val.strip() not in ("0", "1"):
        raise SystemExit(f"{MARK} {ENV} must be 0 or 1, got {val!r}")
    if val.strip() == "0":
        print(f"{MARK} {ENV}=0: stock, files untouched")
        return 0
    if not INDEXER_PY.is_file():
        raise SystemExit(f"{MARK} missing {INDEXER_PY}")
    source = INDEXER_PY.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed for sparse_attn_indexer_kpool.py: {exc}") from exc
    compile(patched, str(INDEXER_PY), "exec")
    if patched != source:
        replace_file(INDEXER_PY, patched)
        clear_pyc(INDEXER_PY)
    print(f"{MARK} sparse_attn_indexer_kpool.py: {action} ({ENV}=1: the kpool indexer keeps the "
          f"select_k-1 highest-scored pools, dropping the lowest, instead of an arbitrary one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
