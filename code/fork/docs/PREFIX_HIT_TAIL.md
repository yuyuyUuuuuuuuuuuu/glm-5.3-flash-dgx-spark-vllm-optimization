# Prefix hit + short prefill inconsistency: verdict, root causes, fixes (branch `prefixhit`, 2026-10-04)

Question (decode4 review, DEC_KDA_LAZY.md 7.1): on the nodeC mini rig, a prefix hit at the 32256 checkpoint followed
by a short prefill (4-23 tokens) disagreed badly with a fresh prefill of the same tokens (KL ~0.8, top-1 0-1 of
12-20). Is this a production bug?

## Verdict

**The reported inconsistency is an artifact of the nodeC rig's single-process engine, not a production defect.**
Production (multiprocess workers) serves prefix hits consistently with fresh prefills (measured on production, below).
On the way, a **second, real production defect** was found in the kpool tail-ring slot mapping (independent of the
executor), with a fix behind an OFF-default switch.

## 1. The mini-rig inconsistency = KDA state not restored on a hit, only with an in-process worker

* `MambaHybridModelState.add_request` (V2 runner, align mode) seeds the running state column of a request that starts
  with computed tokens as `(num_computed_tokens - 1) // cache_config.block_size`; the align pre-copy then copies that
  column into the step's destination column (computed with `MambaSpec.block_size` = 4608).
* The engine core recomputes `cache_config.block_size` as the minimum over the prefix-caching groups
  (`v1/engine/core.py::_initialize_kv_caches`): the DFlash2 drafter group makes it **576**. Mamba keeps 4608
  (`cache_config.mamba_block_size`).
* The rig runs `VLLM_ENABLE_V1_MULTIPROCESSING=0` + TP=1 = `UniProcExecutor`: the worker shares the engine core's
  config object and sees 576. A hit at 27648 seeds column 47 instead of 5, the pre-copy reads past the request's row
  (null block -> zeros, or a stale id -> another request's state), and the KDA layers prefill the suffix from a zero /
  foreign state with `has_initial_state=True`.
  Evidence (`docs/logs/prefixhit/R11_kda_state_read_by_forward.txt`, per-layer trace in the forward): every hit reads
  `rec == 0` (or garbage), every fresh prefill reads the checkpoint bit for bit; the checkpoint itself, MLA KV and
  indexer K of the prefix are identical between the two (dump compare).
* This explains every symptom: short suffixes worst (the KDA state "forgets" over long suffixes, 27648+4.6k looked
  fine), checkpoint-independent (27648, 32256, 36864 all affected), decode- and prefill-saved checkpoints alike,
  hit-vs-hit nondeterminism (the stray column holds whatever block id is left there), unchanged by CUDA graphs
  (`R10eager`).
* Production runs `--distributed-executor-backend mp`: workers are spawned **before** the recompute and keep 4608, so
  the seed is right. Reproduced on the rig with `HANDOFF_MP=1` (worker in its own process), **stock code**:
  hit-vs-fresh KL 0.0006-0.07, top-1 9/9, same as fresh-vs-fresh 0.0002-0.017 (`R13mp`).
* With the seed fixed in-process (`GLM53_MAMBA_ALIGN_SEED=1`, `R12seed`): KL 0.45-1.8 -> 0.0006-0.05, top-1 15/17
  (the 2 misses are near-ties where fresh vs fresh2 also flip), hit vs hit2 deterministic again.

| rig run (A0 37000 tok) | hit 27648 +12 | +64 | hit 32256 +4 | +12 | +600 | +3744 | hit 36864 +36 |
|---|---|---|---|---|---|---|---|
| R1 stock, in-process | 0.470 | 0.150 | 0.656 | 1.096 | 0.471 | 0.512 | 0.455 |
| R12 seed fix, in-process | 0.028 | 0.051 | 0.020 | 0.0006 | 0.0016 | 0.015 | 0.0013 |
| fresh vs fresh2 (noise) | 0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
(first-token KL over top-20, fresh vs hit; full tables `docs/logs/prefixhit/minirig_hit_vs_fresh.txt`)

Consequence for the rig: every earlier nodeC result that compared prefix-hit outputs (e.g. the decode4 review's
"hit path outputs vary run-to-run even with lazy OFF") was measured with this artifact. Run hit experiments with
`HANDOFF_MP=1` or `GLM53_MAMBA_ALIGN_SEED=1`.

Fix (robustness, production-neutral): `overlay/patch_mamba_align_seed.py`, switch **`GLM53_MAMBA_ALIGN_SEED=1`**:
seed with `cache_config.mamba_block_size or cache_config.block_size`. In the production worker both are 4608, so the
column is identical there; it only removes the executor dependence (a future change that propagates the recomputed
value to workers, or a uniproc deployment, would otherwise silently break every prefix hit).

## 2. Production check (real model, nodeA, after AB_RUNNING was gone, under the prodbench lock, concurrency 1)

`tests/prefixhit/prod_prefixhit3.py`: text prefix + an open question at the end so the next token is uncertain
(entropy 1.5-3.0 nats); warm request over T[:37864]; then hit (cached 36864, verified by `cached_tokens`),
fresh, fresh2 (unique `cache_salt`, cached 0), hit2; greedy, top-20 logprobs.

| suffix | KL fresh/fresh2 (noise) | KL fresh/hit | KL hit/hit2 | top-1 fresh=hit |
|---|---|---|---|---|
| 20 | 0.160 | 0.100 | 0.104 | yes |
| 64 (3 positions) | 0.083 / 0.149 / 0.014 | 0.044 / 0.082 / 0.011 | 0.049 / 0.117 / 0.044 | yes x3 |
| 600 | 0.076 | 0.075 | 0.067 | yes |

Hit vs fresh is inside production's own run-to-run noise floor; no hit-specific error. (`prod_prefixhit.py`, confident
positions p_top ~1.0 at suffix 4/16/64/600: top-1 4/4, logprob deltas of the tail tokens equal to the noise floor.)
Short suffixes (< 16) are not served from the 36864 checkpoint in production at all: the kpool replay floor needs
>= one ring (16 tokens with GLM53_KPOOL_RING=1) of fresh tokens, so suffix 4 got cached 0 / 32256.
Load: 2 runs, 9 + 13 requests of ~37k tokens, ~3 min of prefill in total, never more than 1 running request of ours.
(A prompt_logprobs variant was started and aborted within seconds: prompt logprobs over 37k positions is too heavy
for production; server stayed healthy, all requests 200.)

## 3. Real production defect: kpool tail ring slot mapping (V2 runner)

`KpoolTailSpec` (group 1) is a one-block-per-request ring; `KpoolTailManager` writes only column 0 of the request's
block-table row. Slots must be `own_block * ring + pos % ring`.

* The V2 runner computes all slot mappings with `_compute_slot_mappings_kernel`: `block_table[req, pos // ring]`.
  For `pos >= ring` this reads columns nobody wrote.
* `KpoolTailMetadataBuilder.build` would replace it with `compute_kpool_tail_slot_mapping` (the circular form) only
  when `common_attn_metadata.positions` is set - and `MambaHybridModelState.prepare_attn` is the only `prepare_attn`
  that does not pass `positions`. The replacement is dead code for GLM-5.3-Flash. The launcher's V1 clamp
  (`patch_kpool_tail_slotmap.py`) edits `v1/worker/block_table.py`, which the V2 runner does not use (and a clamp
  would read the last, zero column of the 10240/262144-wide V2 row, not column 0).
* Measured (`docs/logs/prefixhit/R7_tail_slots_stock.txt`, `HANDOFF_TAIL_PROBE=1`): every decode tail write and every
  prefill seed past the first ring goes to **tail block 0** (zero column) - one ring shared by all running requests.
  Columns that once held ids (boot shape-warmup dummies stage 7-11 tail blocks per row; rows are never cleared) send
  a later request's **first-chunk** seed to those stale ids. The tail co-owns the indexer allocation (padded stride,
  `patch_kpool_tail_seed_stride`), so tail block b = the first 2048 bytes of indexer page b = **16 pooled indexer keys
  (64 positions) of whatever request or cached prefix owns block b**. Observed: A0's cached block 49 (positions
  27648-27711) overwritten by the next request's first chunk (`R4_block49_tail_row_history.txt`); a live request's
  first 16 pooled keys (positions 0-63, i.e. the start of the prompt) corrupted with NaN/garbage scales during a
  2-request run (`conc_pooled_keys.txt`, C1base req 1: 10-12 pools per layer with rel err > 0.25 or NaN).
* Effect: wrong pooled keys -> wrong sparse top-k candidates (the indexer picks or drops those 64 positions) once the
  context exceeds index_topk. Which blocks get hit depends on the stale ids per block-table row; the shared ring adds
  cross-request mixing of in-progress pools under concurrency. Not visible in the single-request hit-vs-fresh probes
  above (no stale id happened to land on the probed prefix).

With production's ring (`GLM53_KPOOL_RING=1`, 16 slots) the stale-id damage is larger: 2-request run, stock
(C3ringbase): req 1 has 38-47 corrupted prompt pools per MLA layer (NaN / rel err > 0.25); with the fix (C4ringfix):
0/1750 (`conc_pooled_keys_ring16.txt`; solo run C5: prompt 0/1750, generated pools rel err <= 0.036).

Open item (not caused by the tail mapping, not fixed here): under 2-request concurrency the DECODE-written pooled keys
of the generated region deviate from a solo re-prefill by up to 0.6 relative norm in MLA layers 2 and 5 at 4-11 of
23-24 generated pools - identically with and without the fix and with ring 4 or 16 - while the solo run stays
<= 0.036. Worth a separate look (multi-request spec-verify path), it is outside this question.

**Superseded (prefixhit-adv 4f35f68, docs/logs/prefixhit-adv/conc_decode_tail_value1_vs_value2.txt): value 1 is NOT
effective under FULL CUDA graphs (production: FULL_AND_PIECEWISE) - use `GLM53_KPOOL_TAIL_POSITIONS=2`, which also
writes the circular slots into the persistent slot-mapping buffer the graphs replay. The r16z7 kit refuses 1. With 2,
the "open item" above (concurrent decode-written pooled keys deviating up to 0.6) is gone as well (0/235 vs stock
13/235, docs/logs/r16z7/engine_smoke_pooled_keys.txt).** Original text:

Fix: `overlay/patch_kpool_tail_positions.py`, switch **`GLM53_KPOOL_TAIL_POSITIONS=1`**: `prepare_attn` passes
`positions=input_batch.positions` (as `DefaultModelState` does). Only `KpoolTailMetadataBuilder` reads the field on
this model, so no other group changes. With it: tail slots = own block (`R9fix_tail_slots_fixed.txt`), the write
probe shows no writes outside the request's own block, the prompt-region corruption is gone (C2fix req 1: 0/1750).

## Switches, tests, files

| switch | default | file | effect |
|---|---|---|---|
| `GLM53_MAMBA_ALIGN_SEED` | unset (stock) | overlay/patch_mamba_align_seed.py | seed in mamba blocks (production-neutral) |
| `GLM53_KPOOL_TAIL_POSITIONS` | unset (stock) | overlay/patch_kpool_tail_positions.py | per-request tail rings |

Both run from `patch_tf_bundle.py` only when the variable is non-empty; `0` = stock; other values fail boot.

* `tests/prefixhit/test_prefixhit_patches.py` (CPU, no torch): stock seed = column 47 with block_size 576 (bug) and 5
  with 4608 (prod worker); patched = 5/6/7/-1; positions patch adds exactly `positions=`; idempotent; composes in
  either order; `=0` untouched. ALL OK on the image's pristine file.
* Engine (nodeC mini rig, `tests/prefixhit/go.sh`, harness options in `tests/handoff/run_engine.py`:
  `HANDOFF_B_LENS/_REPEAT/_GEN` paired fresh/hit/fresh2/hit2 per absolute length, `HANDOFF_PH_DUMP`/`_TRACE` state
  dump + per-layer trace of the final prefill step, `HANDOFF_TAIL_PROBE` tail-slot and indexer-page write probe,
  `HANDOFF_CONC_TEST` pooled keys written vs a solo re-prefill, `HANDOFF_MP` worker in its own process,
  `HANDOFF_EAGER`, `HANDOFF_NO_SPEC`): fails before (R1), passes after (R12seed); stock with mp passes (R13mp).
* Production probes: `tests/prefixhit/prod_prefixhit.py`, `prod_prefixhit3.py` (run on nodeA under the lock).

## What the operator should do

1. Nothing urgent for prefix hits: production's hit path is consistent (section 2). Do not act on the rig's
   "short-prefill after hit" finding.
2. (superseded: use `=2`, the r16z7 kit, docs/DEPLOY_R16Z7.md) Consider `GLM53_KPOOL_TAIL_POSITIONS=1` in the next kit (needs the kit's start.sh to forward the name to both
   ranks; one A/B against the current kit: quality probes with >= 2 concurrent long requests, plus speed - the change
   adds one small index_select per step in the tail metadata). It fixes silent corruption of pooled indexer keys.
3. `GLM53_MAMBA_ALIGN_SEED=1` is optional hardening (identical behaviour in today's mp workers).
4. For nodeC rig experiments with prefix hits, set `HANDOFF_MP=1` or `GLM53_MAMBA_ALIGN_SEED=1`.
