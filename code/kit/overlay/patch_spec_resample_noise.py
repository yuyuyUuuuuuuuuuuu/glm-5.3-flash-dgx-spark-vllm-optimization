#!/usr/bin/env python3
"""[glm53-resample-noise] Independent Gumbel noise for the residual resample of a
rejected probabilistic draft (idempotent, fail closed, env-gated at runtime).

Bug (image vLLM, model runner V2, draft_sample_method="probabilistic"):
  * the draft token x is argmax(scores/T + G(k0, id)) with k0 = randint(seed, pos)
    (DFlash2 walk: dflash2/speculator.py:89-98, key position = sample_pos - 1 = the
    verification row's pos; base drafters: speculator.py:340-348 "add 1 ... to match
    the Gumbel noise used for draft and target sampling");
  * after a rejection the residual token y is argmax(log max(p - q, 0) + G(k0, id))
    with the SAME k0 (rejection_sampler_utils.py _resample_kernel ->
    gumbel.py gumbel_block_argmax: gumbel_seed = tl.randint(seed, pos)).
  Conditioning on x = argmax(log q + G) truncates every other token's noise, so y is
  not distributed as the residual and the output is not p (measured on the real
  kernels by tests/drafter/test_spec_resample_noise.py).

Fix: for a rejected, valid draft with draft logits present (the only case where the
draft consumed k0), key the residual noise by lane 1 of Philox(seed, pos) instead of
lane 0. Lane 0 (acceptance u = tl.rand(seed, pos), draft key k0, bonus-token and
greedy paths, placeholder rows) is untouched, so draft tokens, acceptance decisions,
bonus tokens and greedy outputs stay bit-identical; only the token emitted at a
rejection changes. Upstream vLLM fixed the same bug the other way round (#54282,
merged 2026-08-29: salt the DRAFT stream by 1 << 30); either side restores exactness
of rejection_sample_method="standard" (the production setting).

NOT covered: rejection_sample_method="block". In block mode every valid row draws
u = tl.rand(seed, pos) and its draft consumed randint(seed, pos), including the rows
after the rejection point, and the outcome depends on them; the next step's rows reuse
exactly those (seed, pos) keys, so consecutive steps are coupled and the output is not
the target distribution (measured: tests/drafter/test_spec_resample_noise.py E4). This
is a property of stock block mode that neither this patch nor #54282 changes; the
patched rejection_sample logs a one-time warning when block verification is used.

Runtime switch (read once when vLLM imports the module):
  GLM53_SPEC_RESAMPLE_INDEPENDENT=1 | 0 = stock numerics (the independent branch is a
  tl.constexpr, compiled out). There is NO default: an unset or invalid value fails
  closed, both here (container start) and at import. Every TP rank samples locally and
  feeds its own tokens to the next step, so a rank that silently fell back to a default
  while the other rank had an explicit value would desynchronize TP=2 at the first
  rejected sampled draft. The launcher must pass the same value to every rank.

Usage: python3 patch_spec_resample_noise.py [--preflight]
  GLM53_SITE overrides the vLLM package root (tests patch a copy, never the image).
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

MARK = "[glm53-resample-noise]"
ENV_NAME = "GLM53_SPEC_RESAMPLE_INDEPENDENT"
SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
TARGET = SITE / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
# sha256 of the stock file in ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
# (informational: the anchors below are what is enforced).
STOCK_SHA256 = "659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a"

# ---- anchor 1: module-level switch --------------------------------------------------
A1_OLD = """import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax, tl_rand32
"""
A1_NEW = """import os

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_block_argmax,
    tl_rand32,
    tl_rand64,
)


def _glm53_resample_independent() -> bool:
    # [glm53-resample-noise] see /opt/glm53/patch_spec_resample_noise.py. No default:
    # each TP rank samples locally, so every rank must be given the same explicit value.
    raw = os.environ.get("GLM53_SPEC_RESAMPLE_INDEPENDENT")
    if raw is None or raw.strip() not in ("0", "1"):
        raise ValueError(
            "GLM53_SPEC_RESAMPLE_INDEPENDENT must be set to 0 or 1 on every rank "
            "(got: %r)" % (raw,)
        )
    return raw.strip() == "1"


_GLM53_RESAMPLE_INDEPENDENT = _glm53_resample_independent()
_GLM53_BLOCK_WARNED = False


def _glm53_warn_block_verification() -> None:
    # [glm53-resample-noise] block verification is not exact in this image (see the patch).
    global _GLM53_BLOCK_WARNED
    if _GLM53_BLOCK_WARNED:
        return
    _GLM53_BLOCK_WARNED = True
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "[glm53-resample-noise] rejection_sample_method='block' is not exact for sampled "
        "requests in this image, with or without the independent residual noise: rows "
        "after the rejection point consume the (seed, pos) keys that the next step "
        "reuses. Use rejection_sample_method='standard'."
    )


@triton.jit
def _glm53_residual_gumbel_argmax(
    residual_logits,
    block,
    mask,
    seed,
    pos,
    USE_FP64: tl.constexpr,
):
    # [glm53-resample-noise] Same Gumbel-max as gumbel_block_argmax (temperature
    # already applied), keyed by lane 1 of Philox(seed, pos). Lane 0 of that block
    # is the draft's Gumbel key and the acceptance draw; lane 1 is used nowhere else.
    _, residual_key, _, _ = tl.randint4x(seed, pos)
    logits = residual_logits
    if USE_FP64:
        logits = logits.to(tl.float64)
        u = tl_rand64(residual_key, block, includes_zero=False)
        gumbel_noise = -tl.log(-tl.log(u))
    else:
        u = tl_rand32(residual_key, block, includes_zero=False)
        gumbel_noise = -tl.log(-tldevice.log1p(-u))
    logits = tl.where(mask, logits + gumbel_noise, float("-inf"))
    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx
"""

# ---- anchor 2: kernel signature (unique: followed by the resample body) -------------
A2_OLD = """    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
"""
A2_NEW = """    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
    INDEPENDENT_RESAMPLE: tl.constexpr = False,  # [glm53-resample-noise]
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
"""

# ---- anchor 3: the noise draw of the resample ---------------------------------------
_STOCK_CALL = """gumbel_block_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        None,  # logits_cache_ptr
        0,  # logits_cache_stride
        None,  # logits_cache_col_ptr
        vocab_size,
        APPLY_TEMPERATURE=False,
        USE_FP64=USE_FP64,
    )
"""
A3_OLD = "    # Resample the rejected/bonus token.\n    value, idx = " + _STOCK_CALL
A3_NEW = (
    "    # Resample the rejected/bonus token.\n"
    "    # [glm53-resample-noise] A rejected probabilistic draft consumed the Gumbel key\n"
    "    # of this (seed, pos); resampling its residual with the same noise biases the\n"
    "    # output. Bonus, greedy and placeholder rows keep the stock draw.\n"
    "    if INDEPENDENT_RESAMPLE and HAS_DRAFT_LOGITS:\n"
    "        if is_bonus or not is_valid_rejected_draft:\n"
    "            value, idx = "
    + _STOCK_CALL.replace("\n    ", "\n            ").replace("\n            )\n", "\n            )\n")
    + "        else:\n"
    "            value, idx = _glm53_residual_gumbel_argmax(\n"
    "                residual_logits,\n"
    "                block,\n"
    "                mask,\n"
    "                tl.load(seed_ptr + req_state_idx),\n"
    "                tl.load(pos_ptr + resample_token_idx),\n"
    "                USE_FP64=USE_FP64,\n"
    "            )\n"
    "    else:\n"
    "        value, idx = "
    + _STOCK_CALL.replace("\n    ", "\n        ")
)

# ---- anchor 4: the launch passes the switch ------------------------------------------
A4_OLD = """        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
    )

    # Insert the resampled tokens into the output sampled.
"""
A4_NEW = """        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        INDEPENDENT_RESAMPLE=_GLM53_RESAMPLE_INDEPENDENT,  # [glm53-resample-noise]
    )
    if use_block_verification:  # [glm53-resample-noise]
        _glm53_warn_block_verification()

    # Insert the resampled tokens into the output sampled.
"""

EDITS = (("module switch", A1_OLD, A1_NEW), ("kernel signature", A2_OLD, A2_NEW),
         ("resample noise", A3_OLD, A3_NEW), ("kernel launch", A4_OLD, A4_NEW))


def parse_env(raw: str | None) -> bool:
    # No default (see the module docstring): unset fails closed like any invalid value.
    if raw is None or raw.strip() not in ("0", "1"):
        raise ValueError(f"{ENV_NAME} must be set to 0 or 1 on every rank (got: {raw!r})")
    return raw.strip() == "1"


def prepare(source: str) -> tuple[str, str]:
    """Return (patched source, action). Every anchor must be in exactly one state."""
    states = []
    for label, old, new in EDITS:
        n_old, n_new = source.count(old), source.count(new)
        if n_old == 1 and n_new == 0:
            states.append("stock")
        elif n_old == 0 and n_new == 1:
            states.append("patched")
        else:
            raise ValueError(f"{label}: anchor ambiguous or drifted (stock={n_old}, patched={n_new})")
    if all(s == "patched" for s in states):
        return source, "already present"
    if not all(s == "stock" for s in states):
        raise ValueError(f"partially patched source: {dict(zip((e[0] for e in EDITS), states))}")
    if MARK in source:
        raise ValueError(f"{MARK} present outside the known edits")
    out = source
    for _label, old, new in EDITS:
        out = out.replace(old, new, 1)
    return out, "patched"


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-resample-noise.tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{target.stem}*.pyc"):
            pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    unknown = [a for a in argv[1:] if a != "--preflight"]
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")
    preflight = "--preflight" in argv[1:]
    if not TARGET.is_file():
        raise SystemExit(f"{MARK} missing {TARGET}")
    try:
        on = parse_env(os.environ.get(ENV_NAME))
        source = TARGET.read_text()
        patched, action = prepare(source)
        compile(patched, str(TARGET), "exec")
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed: {exc}") from exc
    sha = hashlib.sha256(source.encode()).hexdigest()
    stock = "stock sha256 match" if sha == STOCK_SHA256 else f"sha256 {sha[:12]} (not the pinned stock file; anchors matched)"
    mode = "independent residual noise" if on else "stock noise (GLM53_SPEC_RESAMPLE_INDEPENDENT=0)"
    if preflight:
        print(f"{MARK} {TARGET.name}: preflight OK ({action}; {stock}; runtime {mode})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{MARK} {TARGET.name}: {action}; {stock}; runtime {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
