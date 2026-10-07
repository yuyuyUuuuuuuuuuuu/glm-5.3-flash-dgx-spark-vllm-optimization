#!/usr/bin/env python3
"""[glm53-block-keys] Row-keyed randomness for block verification (idempotent, fail closed).

Why (image vLLM, model runner V2, DFlash2, draft_sample_method="probabilistic"):
  Every random draw of a speculative step is keyed by Philox(seed, pos) with pos = the
  verification row's position: the draft token of draft index k (dflash2/speculator.py
  _selector_walk_kernel, key randint(seed, sample_pos - 1) = P + k), the acceptance draw
  u_k = rand(seed, P + k) (rejection_sampler_utils.py _rejection_kernel) and the resample /
  bonus token (_resample_kernel). The next step starts at P' = P + num_sampled and keys its
  rows by the same (seed, pos) values.
  * standard verification stops at the first rejection, so the emitted tokens depend only on
    rows <= the rejection point; the rows the next step reuses were never looked at. Exact.
  * block verification (Sun et al. 2024) draws u for EVERY valid row and keeps the LAST row
    that passes, so "tau = k" also says "rows k+1..n-1 all failed": it conditions u_{k+1..}
    upward and conditions the drafts d_{k+2..} on failing. The next step reuses exactly those
    keys (its u_0 IS the old u_{k+1}; its draft 0 uses the old draft k+1's noise vector), so
    consecutive steps are coupled and the output is not the target distribution
    (tests/drafter/test_spec_resample_noise.py E4; tests/blockverify/test_block_exactness.py).

Fix (block mode only): key every draw of local row j of a step by (seed, pos, j), i.e. put j
in the high word of the 64-bit Philox counter (Triton randint4x(seed, offset) uses
offset_lo = pos, offset_hi = offset >> 32; stock offsets are < 2**32 so their hi word is 0).
Row j of the step that starts at P is keyed (P + j, j); two different steps of one request
start at different P, so no key is ever reused, and row 0 (j = 0) keeps the stock key.
Within a step the structure is the stock one: draft index j and u_j share lane 0 of the row
key, the bonus token uses lane 0, and the residual of a rejected draft uses lane 1 (the same
lane choice as patch_spec_resample_noise.py, here unconditional: block mode is not exact
within a step without it).
  rejection_sampler_utils.py  _rejection_kernel u draw; _resample_kernel noise; launches pass
                              BLOCK_KEYS=use_block_verification (constexpr: standard mode
                              compiles the stock code, bit-identical)
                              _compute_local_residual_mass_kernel (block mode only) skips the bonus
                              row: with adaptive K (n < 7) the stock kernel reads one element past
                              draft_sampled for the last request (memcheck); the value was unused
  dflash2/speculator.py       _selector_walk_kernel key; BLOCK_KEYS is True iff the
                              speculative config says rejection_sample_method == "block"
Both switches come from the same speculative config, so drafter and verifier always agree,
and every TP rank derives them from its own (identical) --speculative-config.

Requires patch_spec_resample_noise.py applied first (its marker must be present) and keeps that patch's four
patched anchors contiguous, so re-running it on a block-keys file is a no-op. Its "block is not exact" warning is
suppressed (the flag it checks is set) and a one-time "[glm53-block-keys]" info line is logged instead. Only the DFlash2
drafter is keyed: block mode with another drafter (MTP/EAGLE: speculator.py sample_draft)
stays coupled across steps; the launcher refuses block unless SPEC_METHOD=dflash.

Usage: python3 patch_spec_block_keys.py [--preflight]
  GLM53_REJECTION_METHOD=block applies the patch; =standard leaves every file untouched (and
  prints so); anything else (incl. unset) fails closed. GLM53_SITE overrides the vLLM package
  root (tests patch a copy, never the image).
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

MARK = "[glm53-block-keys]"
ENV_NAME = "GLM53_REJECTION_METHOD"
SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
RSU = SITE / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
SPEC = SITE / "v1/worker/gpu/spec_decode/dflash2/speculator.py"
# sha256 of the image files (informational; the anchors are what is enforced)
RSU_RESAMPLE_PATCHED_FROM = "659e82c2ce1249a6e614f7adf62e3824329a7b79e7a4caa55cfadf254068a98a"
SPEC_STOCK_SHA256 = "d2f6662a4a27856c3331a598a12a44808c366a6317184441138aeeb99963ce48"
RESAMPLE_MARK = "[glm53-resample-noise]"

# =============================================================== rejection_sampler_utils.py
# Every edit keeps the four patched anchors of patch_spec_resample_noise.py intact (contiguous), so re-running that
# patch on a block-keys file reports "already present" instead of failing (containers re-run the overlay chain).
# ---- R1: helpers after the resample patch's helpers, before the first stock kernel -----------------------
R1_OLD = """

@triton.jit
def _compute_max_and_sumexp(logits):
"""
R1_NEW = """

# [glm53-block-keys] With the row-keyed draws below, block verification is exact across steps
# (/opt/glm53/tf/overlay/patch_spec_block_keys.py), so the resample patch's "block is not exact"
# warning no longer applies: mark it as already emitted.
_GLM53_BLOCK_WARNED = True
_GLM53_BLOCK_KEYS_LOGGED = False


def _glm53_log_block_keys() -> None:
    # [glm53-block-keys] one-time marker: block verification runs with row-keyed randomness.
    global _GLM53_BLOCK_KEYS_LOGGED
    if _GLM53_BLOCK_KEYS_LOGGED:
        return
    _GLM53_BLOCK_KEYS_LOGGED = True
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "[glm53-block-keys] rejection_sample_method='block': acceptance, resample and bonus "
        "draws keyed by (seed, pos, row); residual on lane 1"
    )


@triton.jit
def _glm53_block_keyed_gumbel_argmax(
    logits,
    block,
    mask,
    seed,
    key_pos,
    residual_lane,
    temp,
    USE_FP64: tl.constexpr,
):
    # [glm53-block-keys] Gumbel-max (temperature already applied) keyed by Philox(seed, key_pos),
    # key_pos = pos + (row << 32): lane 1 for the residual of a rejected draft (the draft and u of
    # this row used lane 0), lane 0 for bonus and placeholder rows. temp == 0: plain argmax.
    k0, k1, _, _ = tl.randint4x(seed, key_pos)
    key = tl.where(residual_lane, k1, k0)
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        if USE_FP64:
            u = tl_rand64(key, block, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(key, block, includes_zero=False)
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))
    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _compute_max_and_sumexp(logits):
"""

# ---- R2: _rejection_kernel signature (unique: SYNTHETIC_MODE precedes it only here) ---------
R2_OLD = """    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
"""
R2_NEW = """    SYNTHETIC_MODE: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
    BLOCK_KEYS: tl.constexpr = False,  # [glm53-block-keys]
):
"""

# ---- R3: the acceptance draw ------------------------------------------------------------------
R3_OLD = """        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
"""
R3_NEW = """        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            if BLOCK_KEYS:  # [glm53-block-keys] Philox counter hi word = row index
                pos = pos + (tl.cast(i, tl.int64) << 32)
            u = tl_rand32(seed, pos, includes_zero=False)
"""

# ---- R4: _resample_kernel signature: before USE_FP64 (the resample patch's anchor starts there) ---
R4_OLD = """    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
"""
R4_NEW = """    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    BLOCK_KEYS: tl.constexpr,  # [glm53-block-keys] (no default: it precedes USE_FP64)
    USE_FP64: tl.constexpr,
"""

# ---- R5: the resample noise: after the resample patch's branches, block mode redraws row-keyed ---
R5_OLD = """    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr
"""
R5_NEW = """    if BLOCK_KEYS:  # [glm53-block-keys] replaces the draw above: row-keyed, residual on lane 1
        value, idx = _glm53_block_keyed_gumbel_argmax(
            residual_logits,
            block,
            mask,
            tl.load(seed_ptr + req_state_idx),
            tl.load(pos_ptr + resample_token_idx) + (tl.cast(resample_idx, tl.int64) << 32),
            (not is_bonus) and is_valid_rejected_draft,
            temp,
            USE_FP64=USE_FP64,
        )
""" + R5_OLD

# ---- R6: _rejection_kernel launch + marker -------------------------------------------------------
R6_OLD = """        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )
"""
R6_NEW = """        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        BLOCK_KEYS=use_block_verification,  # [glm53-block-keys]
        num_warps=1,
    )
    if use_block_verification:  # [glm53-block-keys]
        _glm53_log_block_keys()
"""

# ---- R7: _resample_kernel launch ------------------------------------------------------------------
R7_OLD = """        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
"""
R7_NEW = """        HAS_DRAFT_LOGITS=has_draft_logits,
        BLOCK_KEYS=use_block_verification,  # [glm53-block-keys]
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
"""

# ---- R8: _compute_local_residual_mass_kernel skips the bonus row (stock OOB read with adaptive K) ---------
# With n < num_speculative_steps verified drafts the bonus row's local position is n (< 7), so the stock early
# return misses it and it reads draft_sampled[logit_idx + 1]: the next request's row 0, or one element past the end
# of draft_sampled for the last request (compute-sanitizer memcheck: "Invalid __global__ read ... 1 bytes after the
# nearest allocation", tests/blockverify/sanitize_block.py). Its residual mass is never read. The last row of a
# request is the one whose next row starts a request (local position 0) or does not exist.
R8_OLD = """    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
"""
R8_NEW = """    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    # [glm53-block-keys] the bonus row of a request verified with n < num_speculative_steps drafts
    # (adaptive K) is not caught below; skip it (unused, and its draft_sampled[logit_idx + 1] read is
    # out of bounds for the last request). The grid's first axis is num_logits (no integer argument:
    # Triton would specialize it on the batch shape and recompile at run time).
    has_next_row = logit_idx + 1 < tl.num_programs(0)
    next_local_pos = tl.load(expanded_local_pos_ptr + logit_idx + 1, mask=has_next_row, other=0)
    if next_local_pos == 0:
        return
    if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
"""

RSU_EDITS = (("rsu helpers", R1_OLD, R1_NEW), ("rsu rejection signature", R2_OLD, R2_NEW),
             ("rsu acceptance draw", R3_OLD, R3_NEW), ("rsu resample signature", R4_OLD, R4_NEW),
             ("rsu resample noise", R5_OLD, R5_NEW), ("rsu rejection launch", R6_OLD, R6_NEW),
             ("rsu resample launch", R7_OLD, R7_NEW), ("rsu residual-mass bonus row", R8_OLD, R8_NEW))

# =============================================================== dflash2/speculator.py
S1_OLD = """    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
"""
S1_NEW = """    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_KEYS: tl.constexpr = False,  # [glm53-block-keys]
):
    row = tl.program_id(0)
"""
S2_OLD = """        position = tl.load(sample_pos_ptr + flat) - 1
"""
S2_NEW = """        position = tl.load(sample_pos_ptr + flat) - 1
        if BLOCK_KEYS:  # [glm53-block-keys] Philox counter hi word = draft index (= row index)
            position = position + (tl.cast(step, tl.int64) << 32)
"""
S3_OLD = """        self.selector_top_k = int(draft_config["selector_top_k"])
"""
S3_NEW = """        self.selector_top_k = int(draft_config["selector_top_k"])
        # [glm53-block-keys] same switch as the verifier (rejection_sampler.py sets
        # use_block_verification from this config field)
        self._glm53_block_keys = (
            getattr(self.speculative_config, "rejection_sample_method", "standard") == "block"
        )
        if self._glm53_block_keys:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "[glm53-block-keys] DFlash2 draft keys: (seed, pos, draft index) for block verification"
            )
"""
S4_OLD = """            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids"""
S4_NEW = """            USE_FP64=self.use_fp64_gumbel,
            BLOCK_KEYS=self._glm53_block_keys,  # [glm53-block-keys]
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids"""
SPEC_EDITS = (("spec walk signature", S1_OLD, S1_NEW), ("spec walk key", S2_OLD, S2_NEW),
              ("spec init switch", S3_OLD, S3_NEW), ("spec walk launch", S4_OLD, S4_NEW))


def parse_env(raw: str | None) -> str:
    val = (raw or "").strip()
    if val not in ("standard", "block"):
        raise ValueError(f"{ENV_NAME} must be 'standard' or 'block' (got: {raw!r})")
    return val


def prepare(source: str, edits, label: str) -> tuple[str, str]:
    """Return (patched source, action). Every anchor must be in exactly one state."""
    states = []
    for name, old, new in edits:
        n_old, n_new = source.count(old), source.count(new)
        if n_new == 1 and (n_old == 0 or (old in new and n_old == new.count(old))):
            states.append("patched")
        elif n_old == 1 and n_new == 0:
            states.append("stock")
        else:
            raise ValueError(f"{name}: anchor ambiguous or drifted (stock={n_old}, patched={n_new})")
    if all(s == "patched" for s in states):
        return source, "already present"
    if not all(s == "stock" for s in states):
        raise ValueError(f"{label}: partially patched source: {dict(zip((e[0] for e in edits), states))}")
    if MARK in source:
        raise ValueError(f"{label}: {MARK} present outside the known edits")
    out = source
    for _name, old, new in edits:
        out = out.replace(old, new, 1)
    return out, "patched"


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-block-keys.tmp")
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
    try:
        method = parse_env(os.environ.get(ENV_NAME))
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed: {exc}") from exc
    if method == "standard":
        print(f"{MARK} {ENV_NAME}=standard: not needed, files untouched (stock standard verification)")
        return 0
    for target in (RSU, SPEC):
        if not target.is_file():
            raise SystemExit(f"{MARK} missing {target}")
    try:
        rsu_src, spec_src = RSU.read_text(), SPEC.read_text()
        if RESAMPLE_MARK not in rsu_src:
            raise ValueError(
                f"{RSU.name} lacks {RESAMPLE_MARK}: run patch_spec_resample_noise.py first "
                "(GLM53_SPEC_RESAMPLE_INDEPENDENT=1)"
            )
        rsu_new, rsu_action = prepare(rsu_src, RSU_EDITS, RSU.name)
        spec_new, spec_action = prepare(spec_src, SPEC_EDITS, SPEC.name)
        compile(rsu_new, str(RSU), "exec")
        compile(spec_new, str(SPEC), "exec")
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed: {exc}") from exc
    sha = hashlib.sha256(spec_src.encode()).hexdigest()
    spec_note = "stock sha256 match" if sha == SPEC_STOCK_SHA256 else f"sha256 {sha[:12]} (anchors matched)"
    if preflight:
        print(f"{MARK} preflight OK ({RSU.name}: {rsu_action}; {SPEC.name}: {spec_action}, {spec_note})")
        return 0
    # both files validated before either is written
    if rsu_new != rsu_src:
        replace_file(RSU, rsu_new)
        clear_pyc(RSU)
    if spec_new != spec_src:
        replace_file(SPEC, spec_new)
        clear_pyc(SPEC)
    print(f"{MARK} {RSU.name}: {rsu_action}; dflash2/{SPEC.name}: {spec_action} ({spec_note}); "
          f"block verification with row-keyed randomness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
