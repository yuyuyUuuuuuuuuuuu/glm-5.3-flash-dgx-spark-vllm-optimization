#!/usr/bin/env python3
"""Seed a resumed request's Mamba/KDA running column with the MAMBA block size (V2 runner, align mode).

The bug (vLLM 487ecf187, ``v1/worker/gpu/model_states/mamba_hybrid.py``)
-----------------------------------------------------------------------
``MambaHybridModelState.add_request`` seeds the running state column of a request that starts with computed
tokens (a prefix-cache hit, or a resumed request) as ``(num_computed_tokens - 1) // cache_config.block_size``. The
align pre-copy (``preprocess_mamba_align_fused_kernel`` + ``precopy_mamba_align_fused_kernel``) then copies the
state from that column into the step's destination column, which it derives with ``MambaSpec.block_size``.

``cache_config.block_size`` is the Mamba block only by accident. The engine core recomputes it as the minimum
block size over the prefix-caching groups (``v1/engine/core.py`` ``_initialize_kv_caches``), and GLM-5.3-Flash's
DFlash2 drafter group drags it to 576 while the Mamba groups keep 4608 (``cache_config.mamba_block_size``, which
``MambaSpec`` is built from). Whether the worker sees the recomputed value depends on the executor:

* multiprocess executor (production: ``--distributed-executor-backend mp``): the workers were spawned before the
  recompute and keep their own config copy (4608) -> the seed is right;
* in-process executor (``UniProcExecutor``: TP=1 without ``mp``, e.g. the nodeC handoff rig, or any future change
  that propagates the recomputed value to workers): the worker shares the mutated config (576) -> a hit at 27648
  seeds column 47 instead of 5, the pre-copy reads a column past the request's row (the null block -> zeros, or a
  stale/foreign block id -> another request's state), and the KDA layers prefill the suffix from a ZERO or FOREIGN
  recurrent/conv state while ``has_initial_state`` says the state is valid. Measured on the mini rig
  (tests/handoff/run_engine.py HANDOFF_PH_TRACE=1): every prefix-hit request's KDA layers read rec == 0 / garbage,
  fresh prefills read the checkpoint bit for bit; hit-vs-fresh KL 0.45-1.8 for short suffixes, decaying with the
  suffix length (the KDA state "forgets"), nondeterministic hit-vs-hit when the stray column holds a live block id.

The fix
-------
Use ``cache_config.mamba_block_size`` (the value ``MambaSpec.block_size`` and the pre-copy kernel use; never
recomputed), falling back to ``cache_config.block_size`` only if it is unset. In the production (mp) worker both
are 4608, so the computed column is byte-identical there; the fix removes the executor dependence.

Installation states: pristine -> patched; patched -> no-op; anything else -> SystemExit. Atomic replace, pyc
cleared. ``patch_tf_bundle.py`` runs this file only when ``GLM53_MAMBA_ALIGN_SEED`` is non-empty; ``0`` returns
without touching anything; values other than 0/1 fail boot.

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
ENV = "GLM53_MAMBA_ALIGN_SEED"
TAG = "[glm53-mamba-align-seed]"
MARK = "# [glm53-mamba-align-seed]"

ANCHOR = """            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
"""
PATCHED = """            # Seed the running state block from the resumed/prefilled position.
            # [glm53-mamba-align-seed] in MAMBA blocks (MambaSpec.block_size ==
            # cache_config.mamba_block_size); cache_config.block_size is recomputed by
            # the engine core as the min over prefix-caching groups (576 with DFlash2).
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1)
                // (self.cache_config.mamba_block_size or self.cache_config.block_size)
            )
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
    n_mark = src.count(MARK)
    if n_mark:
        if n_mark == 1 and src.count(PATCHED) == 1 and src.count(ANCHOR) == 0:
            return src, "present"
        raise SystemExit(f"{TAG} {TARGET}: partial or drifted installation (marker x{n_mark})")
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"{TAG} {TARGET}: anchor found {src.count(ANCHOR)} times (expected 1); vLLM drifted")
    return src.replace(ANCHOR, PATCHED), "patched"


def main(argv=None) -> int:
    val = os.environ.get(ENV, "").strip()
    if val == "0":
        print(f"{TAG} {ENV}=0 -> stock (nothing read or written)")
        return 0
    if val != "1":
        raise SystemExit(f"{TAG} {ENV}={val!r}: expected 0 or 1")
    new, state = apply(TARGET.read_text())
    if state == "present":
        print(f"{TAG} {TARGET}: already present")
        return 0
    _atomic_write(TARGET, new)
    print(f"{TAG} {TARGET}: patched (add_request seeds in mamba blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
