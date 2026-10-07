#!/usr/bin/env python3
"""Give the kpool tail builder its token positions in the V2 hybrid runner (per-request tail rings).

The bug (vLLM 487ecf187 as shipped in the production image, V2 model runner)
---------------------------------------------------------------------------
GLM-5.3-Flash's sparse indexer keeps the keys of the pool in progress in a per-request tail ring (``KpoolTailSpec``,
group 1: ONE block per request; ``KpoolTailManager`` only ever writes column 0 of the request's block-table row).
The tail slots must therefore be ``own_block * ring + pos % ring``. The V2 runner computes every group's slot mapping
with the generic kernel ``_compute_slot_mappings_kernel`` (``v1/worker/gpu/block_table.py``):
``block_table[req, pos // ring] * ring + pos % ring`` -- for ``pos >= ring`` that reads a column
``KpoolTailManager`` never wrote. ``KpoolTailMetadataBuilder.build`` replaces it with the circular per-request form
(``compute_kpool_tail_slot_mapping``) ONLY when ``common_attn_metadata.positions`` is set, and the hybrid model state
(``MambaHybridModelState.prepare_attn``, ``v1/worker/gpu/model_states/mamba_hybrid.py``) is the one ``prepare_attn``
that does not pass ``positions`` to ``build_attn_metadata`` (``DefaultModelState`` does). The replacement is dead
code for GLM-5.3-Flash, so (measured on the nodeC mini rig, tests/handoff/run_engine.py HANDOFF_TAIL_PROBE=1):

* every decode tail write and every prefill seed past the first ring of a request lands in tail block 0, a ring
  SHARED by all running requests (zero-initialised column) -- concurrent requests overwrite each other's
  in-progress pool keys, so pools completed while another request runs are compressed from foreign keys;
* columns that once held a block id (the boot shape-warmup dummy requests stage 7-11 tail blocks per row; rows are
  never cleared past the request's own entries) send the FIRST chunk's seed of every later request in that row to
  those stale block ids. The tail co-owns the indexer allocation (patch_kpool_tail_seed_stride / #57477 addresses
  it with the padded stride), so ``tail block b`` is the first 2048 bytes of indexer page b: 16 pooled indexer keys
  (64 token positions) of whatever request or CACHED PREFIX owns block b get overwritten with raw tail bytes. A later
  prefix hit then reads the corrupted pooled keys -> wrong sparse top-k -> wrong logits (the decode4 review's
  "hit at 32256 + short suffix" inconsistency: A0's block holding positions 27648..27711 was overwritten by the
  next request's first chunk).

The V1 clamp (launcher patch_kpool_tail_slotmap.py, ``v1/worker/block_table.py``) is not on the V2 path, and a clamp
would not help there anyway: the V2 row is ``cdiv(max_model_len, ring)`` wide, so clamping reads the last (zero)
column, not column 0.

The fix
-------
Pass ``positions=input_batch.positions`` from ``MambaHybridModelState.prepare_attn`` to ``build_attn_metadata``,
exactly as ``DefaultModelState.prepare_attn`` does. The only metadata builder that reads
``CommonAttentionMetadata.positions`` on this model is ``KpoolTailMetadataBuilder`` (grep of the image:
``v1/attention/backends/mla/indexer.py`` and the CPU backend), which then emits the circular per-request slots its
docstring and the tail kernels were written for. Every other builder ignores the field, so no other group changes.

Value 2 (prefixhit-adv, graph-safe form)
---------------------------------------
Value 1 alone is NOT effective in FULL CUDA graphs (production: FULL_AND_PIECEWISE, every uniform spec-verify decode
step is a FULL replay). ``compute_kpool_tail_slot_mapping`` returns ``slot_mapping.clone()``: a NEW tensor built
OUTSIDE the graph (V2 builds metadata before capture and before every replay), so a FULL graph keeps reading the
capture-time clone's address (freed after capture, reused by the allocator) while the per-step clone it builds is
never read. Value 2 also patches ``KpoolTailMetadataBuilder.build`` (``v1/attention/backends/mla/indexer.py``) to copy
the circular slots INTO the persistent slot-mapping buffer the graph was captured with, for the real tokens only
(``query_start_loc_cpu[num_reqs]``; FULL padding keeps PAD_SLOT_ID instead of being mapped into the last request's
ring).

Installation states: pristine -> patched; patched (marker + exact patched text) -> no-op; anything else (drifted
anchor, partial marker) -> SystemExit. Atomic replace, pyc cleared. ``patch_tf_bundle.py`` only runs this file when
``GLM53_KPOOL_TAIL_POSITIONS`` is non-empty; ``0`` returns without touching anything; any value other than 0/1 is a
boot failure.

Env override for tests: GLM53_MAMBA_HYBRID_PY.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
TARGET = Path(
    os.environ.get("GLM53_MAMBA_HYBRID_PY", SITEPKG / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
)
ENV = "GLM53_KPOOL_TAIL_POSITIONS"
INDEXER = Path(os.environ.get("GLM53_INDEXER_PY", SITEPKG / "vllm/v1/attention/backends/mla/indexer.py"))
IMARK = "# [glm53-kpool-tail-inplace]"
IANCHOR = """            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
"""
IPATCHED = """            # [glm53-kpool-tail-inplace] write the circular slots INTO the persistent
            # slot-mapping buffer (FULL CUDA graphs replay its capture-time address; a
            # fresh tensor would never be read there), real tokens only (padding keeps
            # PAD_SLOT_ID).
            _n_real = min(
                int(common_attn_metadata.query_start_loc_cpu[common_attn_metadata.num_reqs]),
                common_attn_metadata.num_actual_tokens,
            )
            if _n_real > 0:
                _circ = compute_kpool_tail_slot_mapping(
                    slot_mapping,
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.query_start_loc,
                    positions,
                    _n_real,
                    common_attn_metadata.num_reqs,
                    self.kv_cache_spec.block_size,
                )
                slot_mapping[:_n_real].copy_(_circ[:_n_real])
"""


def apply_indexer(src: str) -> tuple[str, str]:
    n = src.count(IMARK)
    if n:
        if n == 1 and src.count(IPATCHED) == 1 and src.count(IANCHOR) == 0:
            return src, "present"
        raise SystemExit(f"{TAG} {INDEXER}: partial or drifted in-place installation (marker x{n})")
    if src.count(IANCHOR) != 1:
        raise SystemExit(f"{TAG} {INDEXER}: anchor found {src.count(IANCHOR)} times (expected 1); vLLM drifted")
    return src.replace(IANCHOR, IPATCHED), "patched"
TAG = "[glm53-kpool-tail-positions]"
MARK = "# [glm53-kpool-tail-positions]"

ANCHOR = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
"""
PATCHED = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # [glm53-kpool-tail-positions] token positions for KpoolTailMetadataBuilder
            # (circular per-request tail slots), as DefaultModelState passes them.
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
"""


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".glm53tmp")
    mode = stat.S_IMODE(path.stat().st_mode)
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    pyc = path.parent / "__pycache__"
    if pyc.is_dir():
        for f in pyc.glob(path.stem + ".*.pyc"):
            f.unlink()


def apply(src: str) -> tuple[str, str]:
    """Return (new_text, state) with state in {"patched", "present"}; SystemExit on any other form."""
    n_mark = src.count(MARK)
    if n_mark:
        if n_mark == 1 and src.count(PATCHED) == 1 and src.count(ANCHOR) == 0:
            return src, "present"
        raise SystemExit(f"{TAG} {TARGET}: partial or drifted installation (marker x{n_mark})")
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{TAG} {TARGET}: anchor found {src.count(ANCHOR)} times (expected 1); vLLM drifted")
    if "positions=" in src.split("def prepare_attn", 1)[-1].split("def postprocess_state", 1)[0]:
        raise SystemExit(f"{TAG} {TARGET}: prepare_attn already passes positions (unexpected source)")
    return src.replace(ANCHOR, PATCHED), "patched"


def main(argv=None) -> int:
    val = os.environ.get(ENV, "").strip()
    if val == "0":
        print(f"{TAG} {ENV}=0 -> stock (nothing read or written)")
        return 0
    if val not in ("1", "2"):
        raise SystemExit(f"{TAG} {ENV}={val!r}: expected 0, 1 or 2")
    if val == "2":
        inew, istate = apply_indexer(INDEXER.read_text())
        if istate == "patched":
            _atomic_write(INDEXER, inew)
        print(f"{TAG} {INDEXER}: {istate} (in-place persistent tail slots)")
    src = TARGET.read_text()
    new, state = apply(src)
    if state == "present":
        print(f"{TAG} {TARGET}: already present")
        return 0
    _atomic_write(TARGET, new)
    print(f"{TAG} {TARGET}: patched (MambaHybridModelState.prepare_attn passes positions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
