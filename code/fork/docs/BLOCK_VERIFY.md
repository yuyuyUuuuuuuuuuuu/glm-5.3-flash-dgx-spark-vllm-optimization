# BLOCK_VERIFY — block verification for DFlash2 sampled decoding (`GLM53_REJECTION_METHOD`)

Status 2026-09-29, branch `blockverify` (from `deploy-r16` b859af4). Everything below was read, built and measured on
nodeC (one GB10, production image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, every GPU run
under `/tmp/tf-gpu-bench.lock` via `tests/r16/gpu.sh`, < 8 GiB). Production (nodeA/nodeB) was not accessed; nothing was
deployed. Every number is in a committed log under `docs/logs/blockverify/`.

## 0. Verdict

* The image's block verification (Sun et al. 2024, `rejection_sample_method="block"`) is **not exact** in production's
  configuration, with or without today's overlays. Measured on the real kernels (§4): the emitted token-sequence
  distribution differs from the target's with chi-square p = 0 (TV 0.019 on 5-token sequences, 1.02 M samples, the
  production-like model; p = 1e-8 at the production vocab). Two causes, both about Philox keys, not about the algorithm:
  1. **within a step** (the bug `patch_spec_resample_noise.py` already fixes for standard): the residual resample of the
     rejected row reuses the Gumbel noise that drew that row's draft. The overlay's fix is *not* conditional on the
     method, so with `GLM53_SPEC_RESAMPLE_INDEPENDENT=1` it already covers block mode (the existing
     `tests/drafter/test_spec_resample_noise.py` E4 measured block + resample fix exact within one step).
  2. **across steps** (block only): block verification draws `u_i = rand(seed, P + i)` for *every* verified row and keeps
     the last row that passes, so "τ = k" tells that rows k+1…n−1 failed. The next step starts at `P' = P + τ + 1` and
     keys its rows (acceptance draws *and* DFlash2 draft noise) by exactly those `(seed, pos)` values, so consecutive
     steps are coupled. Standard verification stops at the first rejection and never looks at those rows: exact.
* **Fix** (`overlay/patch_spec_block_keys.py`, block mode only): row `j` of a step keys every draw by `(seed, pos, j)` —
  `j` goes into the high word of the 64-bit Philox counter (`randint4x(seed, offset)` uses `offset_lo = pos`,
  `offset_hi = offset >> 32`). Two steps of one request start at different `P`, so no key is ever reused; row 0 keeps
  the stock key; standard mode compiles the stock code (constexpr) and is bit-identical. With it, block verification
  is exact on every configuration tested (§4), and the accepted lengths follow Sun et al.'s acceptance law exactly.
* **Expected gain** (§5; synthetic distributions, stated as such): +1.0 … +2.1 % tokens/step at n = 4, +1.5 … +2.9 %
  at n = 5, +2.1 … +4.3 % at n = 7 for LLM-like entropy mixtures calibrated to production's per-position acceptance
  (0.70 / 0.48 / 0.34 / 0.21), up to +7.5 / +9.8 / +15.2 % for a flat, prose-like mixture. Verification cost:
  +18 … +28 µs per step at 1–2 requests (≈ 0.03 % of a ~105 ms step). Only the production A/B (§7) can say what
  production gets; adaptive K will also pick longer blocks when acceptance rises.
* **Switch**: `GLM53_REJECTION_METHOD=standard|block`, unset/empty = standard = today's `--speculative-config` byte for
  byte. `block` also runs the new bundle overlay on both ranks. Ship as an opt-in A/B candidate; the default stays
  standard until the production bench (§7) shows a tok/s gain.

## 1. What the image implements (vLLM 0.1.dev20051+g487ecf187, model runner V2)

Config: `config/speculative.py:81` `RejectionSampleMethod = "standard" | "synthetic" | "block"`, field at :220;
`draft_sample_method` (:291) `"probabilistic"` is what start.sh sets (DFlash2 caches its full fp32 draft logits).
`RejectionSampler.__init__` (`v1/worker/gpu/spec_decode/rejection_sampler.py:97`) turns `"block"` into
`use_block_verification`; `_verify` (:134) runs the real `Sampler.apply_sampling_params` (logit bias, penalties over
prompt + output + the draft prefix of each row, bad words, thinking budget, temperature, min_p, top-k/top-p) and then
`rejection_sample` (`rejection_sampler_utils.py:922`). Per request, rows `0..n` hold (last token, d_1..d_n); row `i`
verifies `d_{i+1}`; `pos` of row i = `P + i`.

Block mode adds two kernels and a branch (`rejection_sampler_utils.py`):
* `_compute_cumulative_log_p_kernel` (:306): `P_i = min(P_{i-1} · p_i(d_i)/q_i(d_i), 1)` in log space (Sun et al.'s
  `p_i`); placeholders (-1) carry the last value.
* `_compute_local_residual_mass_kernel` (:387): per vocab block, `Σ_x max(P_i · M_b(x | d≤i+1) − M_s(x | d≤i+1), 0)` from
  row i+1's processed target and draft step i+1's cached draft distribution.
* `_rejection_kernel` block branch (:595): `h_i = r_i / (r_i + 1 − P_i)` (i < n−1), `h = P` on the last verified row
  or before a placeholder; `accepted_length = i + 1 if u_i <= h_i` for every valid row (:626), i.e. the LAST passing row
  wins. Drafts are stored for every row; the resample row is `τ`.
* `_resample_kernel` (:696): τ = n → bonus row sampled from `M_b` (target logits); τ < n → residual
  `max(P_τ · M_b − M_s, 0)` (`log_p_tau` shift, :792), Gumbel-max keyed by `(seed, pos_τ)`.
* Greedy requests (temperature 0) never enter the block branch: `is_greedy` is tested first (argmax match), and the
  greedy bonus row is an argmax. At temperature 0 block == standard bit for bit (§4, `test_greedy_identity.py`).

Randomness (all Philox, `tl.randint(seed, offset)` = lane 0 of `randint4x`): draft index k of the DFlash2 walk
(`dflash2/speculator.py:89`, `position = sample_pos − 1 = P + k`) uses `k0 = randint(seed, P + k)` as the key of its
Gumbel noise over the 16 candidates; `u_k = rand(seed, P + k)` (`rejection_sampler_utils.py:569`) is the same 32-bit
`k0` as a uniform; the resample/bonus row τ draws Gumbel noise keyed by `randint(seed, P + τ)` (`gumbel.py:139`;
`patch_spec_resample_noise.py=1` moves the residual of a rejected draft to lane 1). Seeds are per request (random 64-bit
when the client sends none, `sample/states.py:53`), identical on both TP ranks.

## 2. Interactions with production's overlays and features

| item | standard (today) | block | block + `patch_spec_block_keys.py` |
|---|---|---|---|
| `patch_spec_resample_noise.py` (=1) | needed; exact | needed (same residual/draft noise sharing at row τ; the patch is method-independent and covers it); still biased across steps; the patch logs a one-time "block is not exact" warning | the block path always draws the residual on lane 1 of the row key; the resample patch's four anchors stay contiguous (re-running it is a no-op) and its warning is marked emitted; a one-time `[glm53-block-keys]` info line instead |
| adaptive K {4,5,7} (`patch_adaptive_k.py`) | n+1 rows, `num_speculative_steps` stays 7 | works: the last verified row uses `h = P_n` (Sun et al.'s γ = n); unverified drafts n..6 are never observed. Stock bug: the bonus row's local pos n < 7, so `_compute_local_logits_stats_kernel` (harmless, in bounds) and `_compute_local_residual_mass_kernel` also process it; the latter reads `draft_sampled[logit_idx + 1]`, one element past the end for the last request (memcheck-confirmed, §3); the value is unused | the residual-mass kernel skips the bonus row (memcheck clean); exact at n = 4, 5, 7 (§4). Higher acceptance raises the EMA, so adaptive K will choose n = 5/7 more often |
| DFlash2 proposer (`patch_dflash2.py`, image file) | probabilistic pairwise-selector walk over top-16 candidates; the cached draft logits are the realized conditional scores = exactly the `M_s(· ∣ d≤i)` block verification needs | unchanged | the walk's key gets the draft index in the counter's high word iff the speculative config says `block` (`DFlash2Speculator._glm53_block_keys`); same code otherwise |
| DFlash2 CUDA graphs | the walk + draft-logit cache run inside the drafter's FULL graph (`dflash/speculator.py:143` captures `_generate_draft`) | same | `BLOCK_KEYS` is a constexpr fixed at import (from the config) → captured like the stock kernel; graph replay == eager bit for bit (§4) |
| rejection sampler CUDA graphs | not captured: `GPUModelRunner.sample_tokens → sample → rejection_sampler` runs eagerly (`model_runner.py:1464`) | same | same |
| penalties / top_p / top_k / min_p / bad words / thinking budget | applied per row by `apply_sampling_params` (draft prefix included) = the target the verifier uses | same inputs; block only needs `M_b(· ∣ d≤i)` per row, which these rows are | exact with top_p 0.95 + repetition_penalty 1.05 / 1.5 (§4) |
| structured output | grammar bitmask per row; invalid drafts are -1 or get p = 0 | p = 0 at an invalid draft makes `P = 0` from there on, so `h = 0` and no row at or after it can be accepted; rows after it (bitmask not advanced) feed only an `r = 0` term | same; adaptive K keeps structured requests at full length |
| logprobs | computed from the same rows for the emitted tokens | unchanged (only num_sampled differs) | unchanged |
| greedy / mixed batches | argmax path | identical to standard | identical to standard (the salted u of a greedy row is unused) |
| TP = 2 | every rank samples locally from identical inputs | same; the method comes from each rank's `--speculative-config` | drafter and verifier switch come from that same config field; the overlay runs on both ranks via `patch_tf_bundle.py`; deterministic |

Side findings (not changed here, both methods): (a) the image's top-p is two different algorithms depending on the verify
batch's row count — `apply_top_k_top_p` uses the Triton Qrita pivot search for ≥ 8 rows and the sort-based PyTorch path
below 8 (one request at n = 4/5 has 5/6 rows). On LLM-like rows at V = 154880 their masks differ on most rows and the
Triton path keeps 94.2 … 95.7 % of the mass where the sort-based one keeps ≥ 95 %
(`probe_topp_paths.log`); the verifier is exact with respect to whichever processed distribution it computed. (b) the
PyTorch sort path costs ≈ 0.45 ms more per step than Triton at one request (`bench_verify_cost.log`: n = 4/5 854/954 µs
vs n = 7 432 µs).

## 3. The fix (`overlay/patch_spec_block_keys.py`, run by `patch_tf_bundle.py` when `GLM53_REJECTION_METHOD` is set)

Row j of the step that starts at P draws everything from `Philox(seed, counter = (P + j) | j << 32)`:
* `_rejection_kernel`: `u_j = rand(seed, pos + (j << 32))` when `BLOCK_KEYS` (constexpr = `use_block_verification`).
* `_resample_kernel`: after the stock/resample-patch draw, block mode redraws with `_glm53_block_keyed_gumbel_argmax`
  keyed by `(seed, pos_τ + (τ << 32))`: lane 1 for the residual of a rejected draft, lane 0 for the bonus / placeholder
  rows, plain argmax at temperature 0 (the first draw is dead code the compiler may drop).
* DFlash2 `_selector_walk_kernel`: `position = sample_pos − 1 + (k << 32)` for draft index k when `BLOCK_KEYS`.
* `_compute_local_residual_mass_kernel` (launched in block mode only) returns early for the last row of a request.
  Stock skips the bonus row only when its local position is ≥ `num_speculative_steps` (n = 7); with adaptive K
  (n = 4/5) it processes the bonus row and reads `draft_sampled[logit_idx + 1]`, which for the last request of the
  batch is one element past the end of the tensor — compute-sanitizer memcheck: "Invalid __global__ read of size 4
  bytes … 1 bytes after the nearest allocation of size 40 bytes" (`sanitize_block.log`, stock block, n = 4). The
  value is never used (only rows ≤ n−1 feed `h`), so outputs are unchanged; with the fix memcheck reports 0 errors
  (`sanitize_block_keys.log`). The bound of "next row" is the grid (`tl.num_programs(0)` = num_logits), not an
  integer argument: the first version passed `num_logits`, which Triton specializes (== 1, % 16 == 0), so the kernel
  recompiled at run time when the batch shape changed class (3 variants, +209 ms and +138 ms first-call stalls on
  nodeC, `review/probe_jit_specialization.log`; stock and the grid bound compile 1).
Within a step the structure is the stock one (draft j and u_j share lane 0 of the row key, the residual is on lane 1),
so within-step exactness is the resample patch's; across steps, step s's emitted tokens depend only on keys
`{(P_s + j, j)}` and step s+1 uses `{(P_{s+1} + j, j)}` with `P_{s+1} > P_s`: disjoint. The unverified drafts j ≥ n and
the bonus row share lane 0 of `(P+n, n)`, but those drafts are never observed. Only the DFlash2 walk is keyed: block with
MTP/EAGLE drafters (`speculator.py sample_draft`) would stay coupled, so the launcher refuses block unless
`SPEC_METHOD=dflash`.

Fail-closed like the other overlays: both files preflighted before either is written, every anchor must be in exactly
one state, `GLM53_REJECTION_METHOD` must be exactly `standard` or `block` (standard: nothing written), block requires the
resample-noise patch's marker, idempotent, and the chain (resample-noise → block-keys) can be re-run
(`tests/blockverify/test_patch_block_keys.py`).

## 4. Exactness evidence (real kernels, production overlays applied to copies of the image files)

`tests/blockverify/test_block_exactness.py` (run: `tests/blockverify/run_exactness.sh`) drives, per step and for R
requests at once, the real DFlash2 walk + draft-logit cache (7 drafts), the real `RejectionSampler._verify` with the
real V2 `Sampler` (`apply_sampling_params` incl. repetition penalty over prompt + output + draft prefix and top-p) and
`rejection_sample`, verifying the first n ∈ {4, 5, 7} drafts (adaptive K shapes), then commits the emitted tokens
(output counts for the penalty) and starts the next step at `P + num_sampled` with the same per-request seed — i.e.
multi-step chains exactly as production keys them. Target: A active tokens spread over the vocab (other logits a fixed
N(−16, 1.5) tail that top-p removes), next-token logits a random function of the last two tokens, made history-dependent
by the penalty. Draft: noisy copy, context-dependent, on 16 candidates. Ground truth: the processed distribution of every
state computed by the same `apply_sampling_params`, then the exact probability of every length-L sequence (A^L cells).
Every verified row's processed logits were checked against that table (0 mismatches everywhere). Tests: `seq` = the first
L emitted tokens vs the exact sequence distribution; `step2` = the first token of step 2 given (state after step 1,
step-1 num_sampled), the cross-step test; `law` = realized τ minus E[τ | drafts] of the method's rule computed from the
rows the verifier saw (Sun et al.'s h_i for block), z-score.

<!-- gen:EXACT_TABLE -->
| config / n | variant | N sequences | seq chi2/dof, p | TV | step2 chi2/dof, p | law: mean(τ − E[τ ∣ X]), z | tokens/step |
|---|---|---|---|---|---|---|---|
| prod/n4 | std-prod | 1024000 | 190/186, 0.397 | 0.0035 | 76/82, 0.67 | -0.00029, -0.31 | 2.0738 |
| prod/n4 | blk-prod | 1024000 | 2597/186, 0 | 0.0192 | 1486/70, 2.64e-264 | -0.00441, -5.51 | 2.1590 |
| prod/n4 | blk-fix | 1024000 | 167/186, 0.832 | 0.0033 | 71/70, 0.458 | -0.00027, -0.33 | 2.1674 |
| prod/n5 | std-prod | 1024000 | 181/186, 0.594 | 0.0032 | 83/82, 0.441 | -0.00125, -1.21 | 2.1025 |
| prod/n5 | blk-prod | 1024000 | 2771/186, 0 | 0.0200 | 1498/70, 8.46e-267 | -0.00518, -6.09 | 2.2098 |
| prod/n5 | blk-fix | 1024000 | 188/186, 0.438 | 0.0037 | 91/70, 0.0501 | +0.00156, +1.84 | 2.2211 |
| prod/n7 | std-prod | 1024000 | 207/186, 0.14 | 0.0034 | 71/82, 0.799 | +0.00141, +1.25 | 2.1228 |
| prod/n7 | blk-prod | 1024000 | 2576/186, 0 | 0.0187 | 1212/70, 8.99e-208 | -0.00385, -4.24 | 2.2660 |
| prod/n7 | blk-fix | 1024000 | 182/186, 0.57 | 0.0033 | 83/70, 0.142 | +0.00026, +0.29 | 2.2777 |
| stress/n5 | std-prod | 614400 | 60/55, 0.297 | 0.0027 | 34/26, 0.139 | -0.00098, -0.94 | 2.1920 |
| stress/n5 | blk-prod | 614400 | 1325/55, 5.7e-241 | 0.0184 | 594/23, 7.59e-111 | -0.00527, -5.83 | 2.2459 |
| stress/n5 | blk-fix | 614400 | 63/55, 0.215 | 0.0031 | 16/23, 0.84 | +0.00031, +0.35 | 2.2515 |
| textbook/n4 | std-prod | 512000 | 753/716, 0.161 | 0.0113 | 71/74, 0.584 | +0.00282, +2.54 | 2.3058 |
| textbook/n4 | blk-prod | 512000 | 3668/716, 0 | 0.0322 | 529/58, 3.7e-77 | -0.00175, -1.67 | 2.3140 |
| textbook/n4 | blk-fix | 512000 | 674/716, 0.868 | 0.0109 | 44/58, 0.909 | +0.00017, +0.17 | 2.5028 |
| textbook/n7 | std-prod | 512000 | 734/716, 0.316 | 0.0122 | 79/64, 0.0987 | +0.00121, +0.83 | 2.4581 |
| textbook/n7 | blk-prod | 512000 | 1899/716, 6.87e-108 | 0.0203 | 138/42, 3.21e-12 | -0.00991, -7.32 | 2.5048 |
| textbook/n7 | blk-fix | 512000 | 673/716, 0.875 | 0.0111 | 45/42, 0.35 | +0.00177, +1.32 | 2.8851 |
| vocab/n7 | std-prod | 51200 | 151/173, 0.879 | 0.0159 | 72/72, 0.462 | +0.00152, +0.46 | 2.1346 |
| vocab/n7 | blk-prod | 51200 | 295/173, 2.07e-08 | 0.0254 | 130/59, 2.79e-07 | -0.00056, -0.21 | 2.2852 |
| vocab/n7 | blk-fix | 51200 | 183/173, 0.284 | 0.0156 | 60/60, 0.491 | -0.00547, -2.03 | 2.2903 |
<!-- /gen:EXACT_TABLE -->

The residual-mass bonus-row skip (§3, last bullet) was added after a first full run: every statistic of every variant
above is identical with and without it (`preR8_vs_final.txt`, 21/21; the pre-fix logs are in `preR8/`).

Bitwise checks (same seeds and positions, 20 launches × R requests, every emitted token and every step's num_sampled):
`std-fix` == `std-prod` (the block-keys modules are inert in standard mode), `blk-fix-cg` == `blk-fix` and `std-prod-cg`
== `std-prod` (draft kernels replayed from a captured CUDA graph == eager) — all PASS in `exact_*.log`.
`test_greedy_identity.py`: greedy requests bit-identical under standard, block and block + keys; greedy drafts
independent of `BLOCK_KEYS`; sampled requests do change (`greedy_identity.log`).

Configs: `prod` = production sampling (T 1.0, top_p 0.95, repetition_penalty 1.05), V = 9000 (2 rejection vocab blocks,
9 resample blocks), A = 4, L = 5, R = 1024 × 1000 launches; `stress` = T 0.8, top_p 0.9, repetition_penalty 1.5, n = 5;
`textbook` = context-independent p = (.1, .5, .4), q = (.5, .4, .1), no sampling params, A = 3, L = 6 (the E4 case
of `test_spec_resample_noise.py` as a multi-step chain); `vocab` = prod params at the production vocab V = 154880.

## 5. Expected gain (tokens/step), and its basis

Basis, honestly: production's distributions cannot be sampled on nodeC (rules: no production access, GPU < 8 GiB), so
the gain is estimated on synthetic draft/target chains. `tests/blockverify/estimate_gain.py` (host CPU) draws a fresh
target and draft at every position of a chain (V = 4096, target: temperature 1.0 + sort-based top_p 0.95; draft: the
target's logits + Gaussian noise, restricted to its own top-16 as DFlash2), X ~ q, and computes E[τ | X] exactly for
both rules (Rao-Blackwellized; the block formula is the one the kernels were shown to follow, `law` column above).
The noise per position is calibrated so that the **standard** conditional acceptance is production's
(0.70 / 0.686 / 0.708 / 0.618 → unconditional 0.70 / 0.48 / 0.34 / 0.21 as in the vLLM SpecDecoding lines;
0.60 assumed for positions 5–7, rarely verified under adaptive K). Families differ in how entropy is spread:
`mixed` (45 % near-deterministic / 35 % moderate / 20 % open positions), `runs` (same, sticky along the chain),
`mixed-sharp` (over-confident draft, β = 1.6), `peaky` (code-like), `flat` (prose-like, open distributions).
6000 chains per family (`estimate_gain.log`).

<!-- gen:GAIN_TABLE -->
| family | n | tokens/step standard | tokens/step block | gain % (SE) | standard unconditional acceptance per position |
|---|---|---|---|---|---|
| mixed | 4 | 2.728 | 2.784 | +2.05 (0.20) | 0.69, 0.48, 0.34, 0.21 |
| mixed | 5 | 2.853 | 2.936 | +2.89 (0.26) | 0.69, 0.48, 0.34, 0.21, 0.12 |
| mixed | 7 | 2.974 | 3.100 | +4.25 (0.34) | 0.69, 0.48, 0.34, 0.21, 0.12, 0.07, 0.05 |
| runs | 4 | 2.963 | 3.016 | +1.81 (0.19) | 0.68, 0.53, 0.43, 0.32 |
| runs | 5 | 3.212 | 3.285 | +2.28 (0.21) | 0.68, 0.53, 0.43, 0.32, 0.25 |
| runs | 7 | 3.554 | 3.662 | +3.05 (0.26) | 0.68, 0.53, 0.43, 0.32, 0.25, 0.19, 0.15 |
| mixed-sharp | 4 | 2.703 | 2.741 | +1.43 (0.18) | 0.68, 0.48, 0.34, 0.21 |
| mixed-sharp | 5 | 2.826 | 2.878 | +1.83 (0.22) | 0.68, 0.48, 0.34, 0.21, 0.12 |
| mixed-sharp | 7 | 2.946 | 3.032 | +2.92 (0.29) | 0.68, 0.48, 0.34, 0.21, 0.12, 0.07, 0.05 |
| peaky | 4 | 2.700 | 2.726 | +0.99 (0.13) | 0.69, 0.47, 0.33, 0.20 |
| peaky | 5 | 2.819 | 2.862 | +1.52 (0.17) | 0.69, 0.47, 0.33, 0.20, 0.12 |
| peaky | 7 | 2.930 | 2.992 | +2.10 (0.23) | 0.69, 0.47, 0.33, 0.20, 0.12, 0.07, 0.04 |
| flat | 4 | 2.714 | 2.918 | +7.52 (0.35) | 0.70, 0.48, 0.34, 0.21 |
| flat | 5 | 2.835 | 3.114 | +9.84 (0.44) | 0.70, 0.48, 0.34, 0.21, 0.12 |
| flat | 7 | 2.951 | 3.399 | +15.17 (0.60) | 0.70, 0.48, 0.34, 0.21, 0.12, 0.07, 0.04 |
<!-- /gen:GAIN_TABLE -->

The same Rao-Blackwellized comparison on the real-kernel harness (both rules evaluated on the SAME drafts and rows,
those of the blk-fix run's first 100 launches) next to the measured tokens/step of the two runs. They agree except for
`stress`, where the two methods reach different context mixes (repetition penalty 1.5 makes acceptance strongly
state-dependent, and block moves through the sequence in longer steps); the same-drafts column is the clean
comparison:

<!-- gen:RB_TABLE -->
| config / n | E[tau] standard (same drafts) | E[tau] block (same drafts) | tokens/step gain % (SE) | measured tokens/step std -> blk-fix |
|---|---|---|---|---|
| prod/n4 | 1.0746 | 1.1678 | +4.49 (0.04) | 2.0738 -> 2.1674 (+4.52 %) |
| prod/n5 | 1.1020 | 1.2231 | +5.76 (0.04) | 2.1025 -> 2.2211 (+5.64 %) |
| prod/n7 | 1.1202 | 1.2759 | +7.34 (0.05) | 2.1228 -> 2.2777 (+7.30 %) |
| stress/n5 | 1.1676 | 1.2513 | +3.86 (0.03) | 2.1920 -> 2.2515 (+2.71 %) |
| textbook/n4 | 1.3029 | 1.5002 | +8.57 (0.05) | 2.3058 -> 2.5028 (+8.54 %) |
| textbook/n7 | 1.4612 | 1.8867 | +17.29 (0.08) | 2.4581 -> 2.8851 (+17.37 %) |
| vocab/n7 | 1.1345 | 1.2956 | +7.55 (0.15) | 2.1346 -> 2.2903 (+7.30 %) |
<!-- /gen:RB_TABLE -->

Reading: block's gain grows with the verified length and with the entropy of the target. For production's measured
acceptance profile the plausible range is **+1 … +3 % tokens/step at n = 4–5 (typical under adaptive K) and +2 … +4 % at
n = 7**, more on open prose. Cost on the real kernels at V = 154880 (`bench_verify_cost.log`, `_verify` incl. top-p):
+18 … +28 µs per step at 1–2 requests, +118 µs at 4 requests × n = 7; the DFlash2 walk with row keys costs the same
(9.1 vs 9.2 µs). Against a ~105 ms decode step that is ≤ 0.1 %. Not modelled: adaptive K choosing longer blocks when the
EMA rises (more verified rows per step; block gains more at larger n, the verifier forward costs more) and any
correlation structure of real drafts. The production A/B below measures all of it.

## 6. The switch (launcher)

`launcher/start.sh` = the deploy-r16 kit's start.sh (production's current, sha256 b721de12…) +
`launcher/start.sh.blockverify.patch` (generated and checked by `tools/blockverify/make_start_sh.py`):
* both inner scripts (head and worker) build `"rejection_sample_method": (GLM53_REJECTION_METHOD or "standard")` in the
  DFlash2 `--speculative-config`; any other value raises inside the container before `vllm serve` (the inner script's
  `set -euo pipefail` aborts on the failing `$(python3 …)`);
* forwarded to both ranks like `GLM53_SPEC_RESAMPLE_INDEPENDENT`: head `-e GLM53_REJECTION_METHOD="${…:-}"` right after
  it, worker `serve_env_names` entry (so `GLM53_EXTRA_ENV` cannot shadow it);
* `validate_numeric_config`: unset/empty = standard; else exactly `standard|block`; `block` also needs
  `SPEC_METHOD=dflash`, `GLM53_SPEC_RESAMPLE_INDEPENDENT=1`, `overlay/tf/overlay/patch_spec_block_keys.py` and an
  `overlay/patch_tf_bundle.py` that runs it (review 2026-09-29: with only start.sh and the overlay file installed, the
  kit's bundle script never applied the overlay and both ranks would have verified with stock, biased block mode)
  (refused before anything is stopped).
* `overlay/patch_tf_bundle.py` (= `launcher/overlay/patch_tf_bundle.py`) runs `patch_spec_block_keys.py` right after the
  resample-noise patch when `GLM53_REJECTION_METHOD` is non-empty (standard: prints, touches nothing).
Default: with the variable unset the spec JSON is byte-identical to today's, the bundle skips the new overlay, and the
containers get one extra empty `-e`. `bash -n` passes; `tests/blockverify/test_launcher_switch.sh` checks all of it.

## 7. How to A/B in production (operator, nodeA; nothing here was run there)

1. Install (both ranks get it from nodeA on the next start): `overlay/patch_spec_block_keys.py` →
   `~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/overlay/tf/overlay/`; `launcher/overlay/patch_tf_bundle.py` → `overlay/`;
   `launcher/start.sh` → `start.sh` (or `patch -p1 < start.sh.blockverify.patch` on the kit's b721de12). Back up the three
   files first. With `GLM53_REJECTION_METHOD` unset this is today's behaviour.
2. Baseline: `tools/wait_idle.sh` (kit) → restart as usual → `python3 tools/blockverify/bench_sampled.py --label standard
   --out ~/bs_standard.json` (idle-gated itself; 3 prompts × 8 seeds × 256 tokens, T 1.0 / top_p 0.95, one request at
   a time; drops any request that overlapped another client).
3. Block: add `GLM53_REJECTION_METHOD=block` to `.env`, wait idle, restart. Boot checks on BOTH containers' logs
   (head and worker): `[glm53-block-keys] rejection_sampler_utils.py: patched; dflash2/speculator.py: patched` and
   `[glm53-tf-bundle] patch_spec_block_keys.py: applied (GLM53_REJECTION_METHOD=block)` (overlay step),
   `"rejection_sample_method":"block"` in the head's `launching: vllm serve …` line (the worker prints no argument
   list; its drafter marker below shows its config), `[glm53-block-keys] DFlash2 draft keys: (seed, pos, draft index)
   for block verification` (drafter init, every rank), and `[glm53-block-keys] rejection_sample_method='block':
   acceptance, resample and bonus draws keyed by (seed, pos, row)` (already at boot: vLLM's JIT warmup,
   `enable_jit_warmup` default on, calls rejection_sample in block mode; else at the first spec-decode step);
   no `block is not exact` warning anywhere. Then `bench_sampled.py --label block --out ~/bs_block.json`.
4. Compare: `bench_sampled.py --compare ~/bs_standard.json ~/bs_block.json` (accepted/step, tok/s, ms/step per prompt,
   ± 1.96 SE; draft tokens/step shows adaptive K's response). Keep block only if tok/s improves beyond the noise; the
   vLLM SpecDecoding log lines under real traffic give the per-position view over a longer window.
   **Sample size (review 2026-09-29).** The defaults (3 prompts × 8 seeds × 256 tokens) give ≈ 2,250 verify steps per
   arm. With production's acceptance profile the accepted count per step has SD ≈ 1.5 (n = 4: mean 1.73, E[X²] 5.31),
   so the SE of the A−B difference in tokens/step is ≈ 1.7 % and the 95 % interval ≈ ±3.3 % — as wide as the expected
   +1 … +3 %: the default A/B will most likely be inconclusive. Use `--seeds 64 --max-tokens 512` (≈ 16× the steps,
   interval ≈ ±0.8 %). Decide on accepted tokens/step (the mechanism; independent of the machine state) and ms/step
   (the cost); tok/s across a restart also moves with the boot-time memory state, so run A, B, A and check that the
   two A runs agree.
5. Revert: remove the `.env` line (or set it to `standard`), wait idle, restart → identical to step 1. Full revert: restore
   the three backed-up files.

## 8. Risks

* New code on the sampling path (the Philox key of the block path and the DFlash2 walk). Standard mode compiles the
  stock kernels (constexpr) and is bit-identical (measured); block mode is the only thing that changes.
* Both ranks must run the same method and the same overlay: guaranteed by construction (one env forwarded to both, the
  launcher copies `overlay/tf` to nodeB on every start, the drafter switch is read from the same config field), and
  checked by the boot markers. A rank without the overlay would sample differently and desynchronize TP=2.
* Stock block mode + adaptive K (n < 7) reads one int32 past the end of `draft_sampled` in
  `_compute_local_residual_mass_kernel` (memcheck, §3). With PyTorch's caching allocator that read normally lands in
  the same 512-byte block (production batches are far from the multiple-of-128 row counts that could put it at a
  segment end), and the value is unused — but it is an out-of-bounds read in the path block mode enables. The overlay
  skips that row; standard mode never launches the kernel.
* Adaptive K's EMA and set {4,5,7} were tuned with standard verification; block raises acceptance, so the chosen n
  shifts up. The A/B measures the net effect; re-tuning `GLM53_ADAPTIVE_K_SET` may be worth it.
* The gain estimate is synthetic (§5). The exactness proof covers the kernels with the synthetic distributions and
  production's sampling parameters, both vocab sizes, n = 4/5/7, eager and CUDA-graph drafting, greedy/mixed batches;
  it does not cover a grammar mask explicitly (it is one more per-row target mask, like top-p).

## 9. Files and tests

| file | what |
|---|---|
| `overlay/patch_spec_block_keys.py` | the fix (bundle overlay, `GLM53_REJECTION_METHOD`) |
| `overlay/patch_tf_bundle.py`, `launcher/overlay/patch_tf_bundle.py` | runs it after the resample-noise patch |
| `launcher/start.sh`, `launcher/start.sh.blockverify.patch` | the switch (kit start.sh b721de12 + patch) |
| `tools/blockverify/make_start_sh.py` | generates/validates the launcher edit |
| `tools/blockverify/bench_sampled.py` | production sampled-decoding bench (idle-gated, fixed seeds, /metrics deltas) |
| `tools/blockverify/exact_table.py` | the §4 tables from `exact_*.json` |
| `tests/blockverify/test_block_exactness.py`, `run_exactness.sh` | §4 (GPU) |
| `tests/blockverify/test_greedy_identity.py` | greedy bit-identity, drafts independent of `BLOCK_KEYS` (GPU) |
| `tests/blockverify/bench_verify_cost.py`, `probe_topp_paths.py`, `sanitize_block.py` | cost, top-p paths, memcheck (GPU) |
| `tests/blockverify/estimate_gain.py` | §5 (host CPU) |
| `tests/blockverify/test_patch_block_keys.py` | overlay: fail-closed, idempotent, re-runnable chain, bundle wiring (host) |
| `tests/blockverify/test_launcher_switch.sh` | launcher: reproducible, both ranks, JSON default, validation (host) |
| `tests/blockverify/test_bench_sampled.py` | bench against a fake vLLM server: idle gate, deltas, contamination, compare (host) |
| `tests/blockverify/test_block_mutants.py` | power check of the exactness harness: biased mutants must fail it (GPU, review) |
| `tests/blockverify/probe_jit_specialization.py` | compiled-variant count of the block kernels across batch shapes and dtypes (GPU, review) |

## 10. Adversarial review (2026-09-29)

Logs in `docs/logs/blockverify/review/`. What was re-run or added, and what changed:
* **The harness can fail** (`test_block_mutants.py`, prod config, 1.02 M sequences per run; control and mutants): the
  unmutated overlay passes with a fresh seed base (7000; std and blk-fix, n = 4/5/7; the committed n = 5 step2 p = 0.050
  was chance, 0.573 with fresh seeds); every biased mutant is flagged — drafter key stock + verifier keyed (seq p = 0,
  step2 p = 1e-211), verifier u stock + drafter keyed (seq p = 1e-61, law z = −4.7), keyed residual on lane 0 (seq p = 0,
  TV 0.032), `h → min(1.03 h, 1)` (law z = +13.0, seq p = 1e-23), residual-mass skip of the last draft row (harness
  assertion: a row whose residual is empty gets chosen and a token with no target mass is emitted). The committed prod run reproduces bit for
  bit (`rerun_exact_prod.json`, only timings differ).
* **Launcher guard added**: block now also requires the blockverify `patch_tf_bundle.py`. With only start.sh and the
  overlay file installed the kit's bundle script never runs the overlay, and both ranks would verify with stock block
  mode (biased); `test_launcher_switch.sh` has the refusal case and fails on the unguarded launcher.
* **Run-time recompile removed**: `num_logits` argument → `tl.num_programs(0)` (§3; `probe_jit_specialization*.log`:
  3 compiled variants before, 1 after, as stock). Re-run after the change: all four exactness configs (`exact_*.json`
  equal to the previous run in every statistic, only timings differ; all bitwise checks PASS), `greedy_identity.log`
  byte-identical, memcheck (host CUDA 13.0 `compute-sanitizer` mounted read-only, the image has none;
  `review/sanitize_*.log`): stock block 259 errors, block + keys 0, standard 0. Every (target, draft) dtype pair of
  the image's JIT warmup compiles and runs in block mode with the overlay (bf16/fp32 × bf16/fp32).
* **Checked, no change**: the launcher switch reaches both ranks identically and defaults to standard (28 → 30 checks
  pass); the drafter switch reads `DraftModelSpeculator.speculative_config` (set before `DFlash2Speculator.__init__` reads it); nothing in
  the kit (site plugins, overlays) replaces the DFlash2 walk at run time; `patch_dflash2.py` is not run by start.sh
  (the image already carries the stock drafter), so it cannot overwrite the keyed walk; `patch_adaptive_k.py` edits only
  scheduler.py / cudagraph_utils.py; the bench never prints or stores the key and refuses to send anything while busy.
* **Bench power**: see §7 — the default sample size cannot resolve the expected gain.

Revert of this branch = do not install it; production is unchanged until step 1 of §7.
