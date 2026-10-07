#!/usr/bin/env python3
"""Backport vLLM #58454 (kpool tail ring sized for speculative decoding) onto vLLM 487ecf187.

The bug
-------
The GLM-5.3-Flash sparse indexer pools every ``index_kpool`` (4) consecutive keys into one fp8 entry. The keys of
the pool in progress live in a per-request tail block (``KpoolTailSpec``, one block per request) that is used as a
ring addressed by ``pos % ring``. On vLLM 487ecf187 the ring is exactly ``index_kpool`` slots.

A spec-decode verify step runs ``1 + num_spec`` tokens (8 in production: DFlash2, num_speculative_tokens=7) through
``_kpool_decode_update_batched_kernel`` before acceptance is known. Every one of them is stashed at ``pos % 4``, and
a token with ``pos % 4 == 3`` compresses its pool from the ring. When a pool-completing draft is rejected, the
drafts stashed behind it have already overwritten ring slots that hold the pool's committed (accepted) keys; the
redo of that pool on the next step then compresses wrong keys. The corrupted entry is a pooled indexer key of the
model's own output, i.e. a wrong candidate in sparse-attention top-k selection once the context exceeds
``index_topk`` (2048) tokens.

The fix (upstream https://github.com/vllm-project/vllm/pull/58454, merged 2617fe93)
------------------------------------------------------------------------------------
``ring = kpool * next_power_of_2(cdiv(kpool + num_spec, kpool))`` (16 here), the spec's ``block_size`` and
``sliding_window`` are the ring, and the kernels address the ring with ``pos % RING`` / ``slot // RING`` instead of
``POOL_SIZE``. The tail slot mapping (``compute_kpool_tail_slot_mapping``, and the ``[glm53-kpool-tail-slotmap]``
clamp of the generic V1 kernel) already use the spec's block_size, so they follow unchanged.

What this overlay edits (three files, all preflighted before any is written)
---------------------------------------------------------------------------
1. ``models/glm5next/nvidia/attention.py`` -- ``Glm5NextTailCache.get_kv_cache_spec`` returns the ring as block_size
   / sliding_window (upstream's formula and block-size assert; this image keeps its [1 head x 2*head_dim] packing).
2. ``models/glm5next/nvidia/ops/kpool_compress.py`` -- decode kernel + wrapper: ``RING`` constexpr exactly as
   upstream. The decode region between its two section banners is pinned by sha256 (pristine and patched).
3. ``model_executor/layers/sparse_attn_indexer_kpool.py`` -- the prefill seed call passes ``tail_kv_cache.shape[2]``
   (the ring) as the seed kernel's ``kpool`` argument.

Why the seed is fixed at its call site and not inside ``_kpool_tail_seed_kernel`` like upstream: this fork already
ships ``patch_kpool_tail_seed_stride.py`` (verbatim upstream MiaAI-Lab #264 / vLLM #57477), which owns that kernel's
text and re-verifies it byte for byte on every run (a second run over an edited kernel exits non-zero and stops the
container). Leaving that region byte-identical keeps both overlays idempotent in either order of re-runs. The
stride-fixed seed kernel uses its ``KPOOL`` argument for the block (``slot // KPOOL``), the row (``slot % KPOOL``),
the score half (``KPOOL_HEAD = tail.stride(1)``) and the tail-membership look-ahead (``i + KPOOL``); passing the ring
makes the first three exactly upstream's ``RING`` addressing, and the look-ahead seeds each request's last ``ring``
(16) prefill tokens instead of the last ``kpool`` (4). The decode kernel only ever reads the in-progress pool's
slots, which are among the last ``kpool`` tokens, so the ring contents it reads are identical to upstream's (the
tests check this against upstream's form of the kernel).

Preconditions (fail closed otherwise)
-------------------------------------
* The seed kernel must already address the padded tail stride (vLLM #57477: ``patch_kpool_tail_seed_stride.py``,
  ``GLM53_KPOOL_SEED_STRIDE=1``, which ``patch_tf_bundle.py`` runs before this overlay). With the pre-#57477 dense
  addressing no ring size is correct.
* The value of ``GLM53_KPOOL_RING`` must be exactly ``1`` when this runs (``patch_tf_bundle.py`` skips it when the
  variable is empty or unset).
* Every anchor must match vLLM 487ecf187 (the image's files, with or without the launcher's runtime overlays -- none
  of them edits these three spans).

Installation states per file: pristine -> patched; patched (marker + exact patched text) -> no-op ("already
present"); anything else (partial marker, drifted anchor) -> SystemExit. Atomic replace, pyc cleared.

Env overrides for tests: GLM53_GLM5NEXT_ATTENTION_PY, GLM53_KPOOL_COMPRESS_PY (same name as the seed-stride overlay),
GLM53_SPARSE_INDEXER_KPOOL_PY.
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
ATTN_PY = Path(os.environ.get("GLM53_GLM5NEXT_ATTENTION_PY", SITE / "models/glm5next/nvidia/attention.py"))
KPOOL_PY = Path(os.environ.get("GLM53_KPOOL_COMPRESS_PY", SITE / "models/glm5next/nvidia/ops/kpool_compress.py"))
INDEXER_PY = Path(
    os.environ.get("GLM53_SPARSE_INDEXER_KPOOL_PY", SITE / "model_executor/layers/sparse_attn_indexer_kpool.py")
)
ENV = "GLM53_KPOOL_RING"
TAG = "[glm53-kpool-ring]"


def ring_slots(kpool: int, num_spec: int) -> int:
    """Upstream #58454's ring size: kpool * next_power_of_2(cdiv(kpool + num_spec, kpool))."""
    if kpool < 1 or num_spec < 0:
        raise ValueError("kpool >= 1 and num_spec >= 0 required")
    pools = -(-(kpool + num_spec) // kpool)
    return kpool * (1 if pools < 1 else 1 << (pools - 1).bit_length())


# ---------------------------------------------------------------------------------------------------------------------
# 1. attention.py: Glm5NextTailCache.get_kv_cache_spec
# ---------------------------------------------------------------------------------------------------------------------
ATTN_MARK = "        # [glm53-kpool-ring] vLLM #58454: drafts are stashed before acceptance.\n"
ATTN_ANCHOR = """    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # K + gate score packed into head_size (== 2*head_dim), head_size_v=0:
        # KpoolTailBackend.get_kv_cache_shape only consumes head_size and splits
        # it into [2, kpool, head_dim] (K | score halves), so the connectors'
        # non-MLA K/V half-split transfers K and score as separate halves.
        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=1,
            head_size=2 * self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self._index_kpool,
        )
"""
ATTN_PATCHED = """    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # K + gate score packed into head_size (== 2*head_dim), head_size_v=0:
        # KpoolTailBackend.get_kv_cache_shape only consumes head_size and splits
        # it into [2, ring, head_dim] (K | score halves), so the connectors'
        # non-MLA K/V half-split transfers K and score as separate halves.
        # [glm53-kpool-ring] vLLM #58454: drafts are stashed before acceptance.
        # With a one-pool ring, the drafts behind a rejected pool-completing
        # draft overwrite the keys read by its redo, so the ring holds the
        # in-progress pool plus every draft of one verify step.
        from vllm.utils.math_utils import cdiv, next_power_of_2

        num_spec = vllm_config.num_speculative_tokens
        span = self._index_kpool + num_spec
        ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))
        # ring must divide the attention block size (a multiple of 128; of 256
        # with the kpool paged-MQA indexer on SM12x).
        assert self.cache_config.block_size % ring == 0, (
            f"Glm5NextTailCache: cache_config.block_size "
            f"({self.cache_config.block_size}) must be a multiple of the "
            f"tail ring ({ring})"
        )
        logger.info_once(
            "[glm53-kpool-ring] indexer tail ring: %d slots per request "
            "(index_kpool=%d, num_speculative_tokens=%d, block_size=%d)",
            ring,
            self._index_kpool,
            num_spec,
            self.cache_config.block_size,
        )
        return KpoolTailSpec(
            block_size=ring,
            num_kv_heads=1,
            head_size=2 * self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=ring,
        )
"""

# ---------------------------------------------------------------------------------------------------------------------
# 2. kpool_compress.py: decode kernel + wrapper (region between the two section banners, pinned by sha256)
# ---------------------------------------------------------------------------------------------------------------------
REGION_HEAD = "# kpool_decode_update_and_maybe_write_cache_batched : decode step\n"
REGION_TAIL = "# Pool-level topk helpers: select pools -> expand to tokens -> append tail\n"
KPOOL_MARK = "        # [glm53-kpool-ring] vLLM #58454: the tail ring holds RING >= POOL_SIZE\n"
# (old, new, count) -- applied in order inside the region only.
KPOOL_HUNKS: tuple[tuple[str, str, int], ...] = (
    (
        "    POOL_SIZE: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n",
        "    POOL_SIZE: tl.constexpr,\n    RING: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n",
        1,
    ),
    (
        "    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE\n"
        "    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one\n"
        "    completion can occur per request per call, but the ordered loop is correct\n"
        "    for any NEXT_N.\n",
        "    programs are independent (distinct tail blocks). RING >= POOL_SIZE.\n",
        1,
    ),
    (
        "        slot = safe_pos % POOL_SIZE\n        phys_slot = safe_pos % POOL_SIZE\n",
        "        slot = safe_pos % POOL_SIZE\n"
        + KPOOL_MARK
        + "        # slots, so the drafts stashed behind a rejected pool-completing draft\n"
        "        # cannot overwrite the committed keys that the pool's redo reads.\n"
        "        phys_slot = safe_pos % RING\n",
        1,
    ),
    (
        "        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE\n",
        "        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING\n",
        1,
    ),
    (
        "                phys = (pool_logical_start + pool_slot) % POOL_SIZE\n",
        "                phys = (pool_logical_start + pool_slot) % RING\n",
        2,
    ),
    (
        "        tail_kv_cache: paged tail cache ``[num_blocks, 2, pool_size, head_dim]``\n"
        "            bf16 (K at half 0, gate score at half 1).\n",
        "        tail_kv_cache: paged tail cache ``[num_blocks, 2, ring, head_dim]``\n"
        "            bf16 (K at half 0, gate score at half 1); ``ring`` is a multiple of\n"
        "            ``pool_size`` (vLLM #58454).\n",
        1,
    ),
    (
        "    assert tail_kv_cache.shape[2] == pool_size\n",
        "    ring = tail_kv_cache.shape[2]\n    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)\n",
        1,
    ),
    (
        "        POOL_SIZE=pool_size,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n",
        "        POOL_SIZE=pool_size,\n        RING=ring,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n",
        1,
    ),
)
# sha256 of the text between REGION_HEAD (inclusive) and REGION_TAIL (exclusive).
REGION_PRISTINE_SHA = "663be3a67efea2f07a5b87d382e20d8e177995e0b361545be9d8c123f9664bb2"
REGION_PATCHED_SHA = "299fa3809af37c0410ddd095f8bc18f5725e299173ddf4a865d96fbfd3894d35"

# The #57477 precondition: the seed kernel addresses the padded tail stride (patch_kpool_tail_seed_stride.py).
SEED_FIXED_BASE = "    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n"
SEED_FIXED_SCORE = "    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)\n"
SEED_DENSE_BASE = "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n"
SEED_LAUNCH_STRIDES = "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n        KPOOL_HEAD=tail_kv_cache.stride(1),\n        HEAD_DIM=head_dim,\n        KPOOL=kpool,\n"

# ---------------------------------------------------------------------------------------------------------------------
# 3. sparse_attn_indexer_kpool.py: the prefill seed call
# ---------------------------------------------------------------------------------------------------------------------
IDX_MARK = "                        # [glm53-kpool-ring] vLLM #58454: the tail block is a\n"
IDX_ANCHOR = """                        kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
"""
IDX_PATCHED = (
    IDX_MARK
    + """                        # ring of tail_kv_cache.shape[2] (>= index_kpool) slots
                        # addressed by pos % ring. The seed kernel's kpool
                        # argument is that ring (block = slot // ring, row =
                        # slot % ring), so it seeds each request's last `ring`
                        # tokens, a superset of the in-progress pool.
                        kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            tail_kv_cache.shape[2],
                            head_dim,
                        )
"""
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_region(source: str) -> tuple[str, str, str]:
    if source.count(REGION_HEAD) != 1 or source.count(REGION_TAIL) != 1:
        raise ValueError(
            f"kpool_compress decode-region banners drifted (head={source.count(REGION_HEAD)}, "
            f"tail={source.count(REGION_TAIL)})"
        )
    a = source.index(REGION_HEAD)
    b = source.index(REGION_TAIL)
    if b <= a:
        raise ValueError("kpool_compress decode-region banners out of order")
    return source[:a], source[a:b], source[b:]


def patch_region(region: str) -> str:
    out = region
    for old, new, n in KPOOL_HUNKS:
        if out.count(old) != n:
            raise ValueError(f"decode-region hunk count {out.count(old)} != {n}: {old.strip()[:70]!r}")
        out = out.replace(old, new)
    return out


def seed_is_stride_fixed(source: str) -> bool:
    return (
        source.count(SEED_FIXED_BASE) == 1
        and source.count(SEED_FIXED_SCORE) == 1
        and SEED_DENSE_BASE not in source
        and source.count(SEED_LAUNCH_STRIDES) == 1
    )


def prepare_attention(source: str) -> tuple[str, str]:
    marks = source.count(ATTN_MARK)
    if marks:
        if marks == 1 and source.count(ATTN_PATCHED) == 1 and ATTN_ANCHOR not in source:
            return source, "already present"
        raise ValueError(f"partial/inconsistent tail-ring spec patch in attention.py (marker={marks})")
    if source.count(ATTN_ANCHOR) != 1:
        raise ValueError(
            f"Glm5NextTailCache.get_kv_cache_spec anchor drifted (anchor={source.count(ATTN_ANCHOR)}; "
            "expected vLLM 487ecf187)"
        )
    patched = source.replace(ATTN_ANCHOR, ATTN_PATCHED, 1)
    if patched.count(ATTN_MARK) != 1 or patched.count(ATTN_PATCHED) != 1:
        raise ValueError("attention.py post-patch verification failed")
    return patched, "patched"


def prepare_kpool(source: str) -> tuple[str, str]:
    if not seed_is_stride_fixed(source):
        raise ValueError(
            "the tail seed kernel does not address the padded tail stride (vLLM #57477). "
            "Apply patch_kpool_tail_seed_stride.py first (GLM53_KPOOL_SEED_STRIDE=1): "
            "no ring size is correct with the dense seed addressing"
        )
    head, region, tail = split_region(source)
    marks = source.count(KPOOL_MARK)
    sha = _sha(region)
    if marks:
        if marks == 1 and sha == REGION_PATCHED_SHA:
            return source, "already present"
        raise ValueError(f"partial/inconsistent tail-ring decode patch (marker={marks}, region sha {sha[:16]})")
    if sha != REGION_PRISTINE_SHA:
        raise ValueError(
            f"kpool_compress decode region drifted (sha {sha[:16]}, pinned {REGION_PRISTINE_SHA[:16]}; "
            "expected vLLM 487ecf187)"
        )
    new_region = patch_region(region)
    if _sha(new_region) != REGION_PATCHED_SHA or new_region.count(KPOOL_MARK) != 1:
        raise ValueError("kpool_compress post-patch verification failed")
    return head + new_region + tail, "patched"


def prepare_indexer(source: str) -> tuple[str, str]:
    marks = source.count(IDX_MARK)
    if marks:
        if marks == 1 and source.count(IDX_PATCHED) == 1 and IDX_ANCHOR not in source:
            return source, "already present"
        raise ValueError(f"partial/inconsistent tail-ring seed-call patch (marker={marks})")
    if source.count(IDX_ANCHOR) != 1:
        raise ValueError(
            f"sparse_attn_indexer_kpool seed-call anchor drifted (anchor={source.count(IDX_ANCHOR)}; "
            "expected vLLM 487ecf187)"
        )
    patched = source.replace(IDX_ANCHOR, IDX_PATCHED, 1)
    if patched.count(IDX_MARK) != 1 or patched.count(IDX_PATCHED) != 1:
        raise ValueError("sparse_attn_indexer_kpool post-patch verification failed")
    return patched, "patched"


TARGETS = (
    # (label, path getter, prepare, pyc glob)
    ("kpool_compress.py", lambda: KPOOL_PY, prepare_kpool, "kpool_compress*.pyc"),
    ("sparse_attn_indexer_kpool.py", lambda: INDEXER_PY, prepare_indexer, "sparse_attn_indexer_kpool*.pyc"),
    ("attention.py", lambda: ATTN_PY, prepare_attention, "attention*.pyc"),
)


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-ring.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path, pattern: str) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(pattern):
            pyc.unlink(missing_ok=True)


def main() -> int:
    val = os.environ.get(ENV, "")
    if val.strip() != "1":
        raise SystemExit(f"{TAG} {ENV} must be exactly 1 to install (unset/empty = stock), got {val!r}")
    plans = []
    for label, get_path, prepare, pyc in TARGETS:
        path = get_path()
        if not path.is_file():
            raise SystemExit(f"{TAG} missing {path}")
        source = path.read_text()
        try:
            patched, action = prepare(source)
        except ValueError as exc:
            raise SystemExit(f"{TAG} preflight failed for {label}: {exc}") from exc
        compile(patched, str(path), "exec")
        plans.append((label, path, source, patched, action, pyc))
    for label, path, source, patched, action, pyc in plans:
        if patched != source:
            replace_file(path, patched)
            clear_pyc(path, pyc)
    print(f"{TAG} " + "; ".join(f"{label}: {action}" for label, _, _, _, action, _ in plans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
