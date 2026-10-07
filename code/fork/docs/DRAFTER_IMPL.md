# DRAFTER_IMPL — what was built for DRAFTER_PLAN P0, D1 and D2a

GLM-5.3-Flash EXL3 4bpw, vLLM (Mia recipe), TP=2, DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`, adaptive-K {4,5,7},
`draft_sample_method="probabilistic"`, `rejection_sample_method="standard"`.

- Status (2026-09-27): built and verified **on nodeC only**. Production (nodeA/nodeB) was not accessed and nothing is installed there.
- Revised the same day after a review (9 findings, all confirmed). What changed and why is in "Review findings and fixes" at the end; the sections below already include the fixes.
- Every number below is copied from a committed log under `docs/logs/drafter/`.
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, never modified. The patches were applied to copies of image files inside `--rm` containers.
- Weights (public Hugging Face; used locally for measurement only):
  - `incoai/GLM-5.3-Flash-DFlash2` @ dc77ff1c99eeb2df044ee3d4f0094eb033fee410, licence CC BY-NC-ND 4.0, sha256 `b33c0347…e410b` (matches the HF LFS oid). Stored at `${HOME}/models/GLM-5.3-Flash-DFlash2-dc77ff1c`. No derived weights are stored or published.
  - `lm_head.weight` of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ 9eaebb7c, Range-fetched into `${HOME}/models/GLM-5.3-Flash-EXL3-TR3-4bpw-partial/lm_head`.
- Path conventions:
  - `V` = `/usr/local/lib/python3.12/dist-packages/vllm` in the image.
  - `G` = `V/v1/worker/gpu/spec_decode`.
  - "Launcher snapshot" = the 09-24 copy of the MiaAI launcher tree saved earlier on nodeC. The production launcher `a15d696` was not read.

## Files

| File | Purpose |
|---|---|
| `overlay/patch_spec_resample_noise.py` | A: independent Gumbel noise for the residual resample. Edits `G/rejection_sampler_utils.py`. `GLM53_SPEC_RESAMPLE_INDEPENDENT` = 1 or 0, **no default**: unset fails closed on that rank. |
| `overlay/patch_drafter_fp8.py` | B: drafter linears on the target's FP8 Marlin path. Edits `V/model_executor/models/qwen3_dflash.py`. `GLM53_DRAFT_FP8` = off (default), 1/layers, fc, or layers,fc. The value is written into the patched file, so vLLM's compile caches see it. |
| `overlay/patch_drafter_lmhead_fp8.py` | C: a drafter-only FP8 copy of the shared lm_head, used for candidates only. Edits `G/dflash/utils.py` and `V/model_executor/models/qwen3_dflash2.py`. `GLM53_DRAFT_LMHEAD_FP8` = 0 (default) or 1. |
| `tests/drafter/test_spec_resample_noise.py` | A1, A3, A4, A5 (two consecutive steps, standard vs block) on the real Triton kernels. 46 checks. |
| `tests/drafter/bench_drafter_fp8.py` | B1 and C1: timing (median with the spread of its reps; lm_head in 3 trials) and accuracy on real weights. |
| `tests/drafter/test_drafter_fp8_build.py` | B3 and C3: the real drafter built by vLLM's own loader, three builds; B vs vLLM's compile-cache key and loader check. 53 checks. |
| `tests/drafter/review_counterfactual.py` | Host-side, CPU only: the pre-review patches (commit a4d51fa) fail the two new install checks (B's mode in the file text, A's missing default), the current ones pass. |
| `tests/drafter/run_drafter_tests.sh` | Runs all four; the GPU tests one container at a time. Last run: all PASSED. |
| `tests/gpu_run.sh` | Now accepts `GPU_RUN_RO=dir1:dir2` for read-only weight mounts. |

All three patches follow the style of Mia's `patch_spinwait.py`:
- idempotent;
- every anchor must be in exactly one state (all stock or all patched), otherwise they fail closed (the C patch checks both of its files before writing either);
- they `compile()` the result and write atomically;
- `--preflight` writes nothing;
- the env value is validated at install time and again at runtime (B also checks that the runtime value equals the mode written into its file);
- each file ends with `    sys.exit(main())` and carries a unique identity string (`[glm53-resample-noise]`, `[glm53-draft-fp8]`, `[glm53-draft-lmhead-fp8]`), as `validate_overlay_artifacts` expects.

---

## A. Sampling-bias fix (DRAFTER_PLAN P0)

### A1. The bias, measured on production's real kernels

**Test:** `tests/drafter/test_spec_resample_noise.py`, log `docs/logs/drafter/test_spec_resample_noise.log`.

Each launch runs production's code in production order:
1. `G/dflash2/speculator.py::_selector_walk_kernel` draws the draft, keyed by `randint(seed, sample_pos-1)`.
2. `_cache_draft_logits_kernel` writes the fp32 draft-logits cache (realized scores on 16 candidates, -inf elsewhere).
3. `G/rejection_sampler_utils.py::rejection_sample` verifies.
4. `V/v1/worker/gpu/sample/gumbel.py::gumbel_sample` gives the non-speculative sample at the same (seed, pos).

Parameters: real vocab V = 154880, temperature 1.0, distinct random int64 seeds, random positions.

**E1:** p = (.1, .5, .4) and q = (.5, .4, .1) on 3 tokens (-inf elsewhere), one draft token, 500 launches × 2048 requests.

| Variant | Output, N = 1,024,000 (95 % CI) | χ² vs p (dof 2) | Accept (theory Σmin(p,q) = 0.6) | Bonus vs (.2,.3,.5) |
|---|---|---|---|---|
| STOCK (production: draft and resample share noise) | (0.1002, 0.4473, 0.4524) ± (0.0006, 0.0010, 0.0010) | 12727.7, p = 0 | 0.60019 | (0.2001, 0.2996, 0.5004), p = 0.743 |
| STOCK, draft walk keyed by an independent seed | (0.0997, 0.5004, 0.4000) ± (0.0006, 0.0010, 0.0009) | 1.3, p = 0.518 | 0.60074 | p = 0.236 |
| PATCHED (`GLM53_SPEC_RESAMPLE_INDEPENDENT=1`) | (0.1002, 0.4996, 0.4002) ± (0.0006, 0.0010, 0.0009) | 1.1, p = 0.581 | 0.60019 | (0.2001, 0.2996, 0.5004), p = 0.743 |

- The stock kernels reproduce the numpy prediction (.100, .448, .453).
- With the draft noise independent, the unmodified verifier is exact. So the bias is the shared noise, not the verifier.

**E2, production shape:**
- 7 drafts, top-16 pairwise selector.
- Target over the full vocab: 16-token head of N(0, 1.5) logits plus a tail at -11 ± 0.5 (about 2–3 % of the mass).
- 600 launches × 256 requests.
- Measured per verified row: among requests that reach row k, the emitted token's distribution against p_k (17 bins).

| Row | n = 7: STOCK TV / χ² p | n = 7: PATCHED TV / χ² p | n = 4: STOCK TV / χ² p | n = 4: PATCHED TV / χ² p |
|---|---|---|---|---|
| r0 | 0.0112 / 6e-69 | 0.0016 / 0.94 | 0.0124 / 9.6e-80 | 0.0030 / 0.37 |
| r1 | 0.0198 / 3.4e-86 | 0.0038 / 0.25 | 0.0190 / 6.1e-73 | 0.0053 / 0.19 |
| r2 | 0.0208 / 7.3e-47 | 0.0053 / 0.37 | 0.0201 / 7.4e-43 | 0.0037 / 0.87 |
| r3 | 0.0234 / 1.7e-48 | 0.0056 / 0.48 | 0.0235 / 6.9e-47 | 0.0056 / 0.46 |
| r4 | 0.0342 / 4.4e-88 | 0.0050 / 0.64 | **bonus** 0.0049 / 0.31 | **bonus** 0.0049 / 0.31 |
| r5 | 0.0178 / 8.2e-15 | 0.0086 / 0.66 | | |
| r6 | 0.0176 / 7.7e-11 | 0.0062 / 0.90 | | |
| r7 | **bonus** 0.0098 / 0.58 | **bonus** 0.0098 / 0.58 | | |

- The same run at 4× the samples (`test_spec_resample_noise_4x.log`, 614,400 requests per shape): STOCK χ² p ≈ 0 on every draft row. PATCHED minimum p is 0.014 (n = 7) and 0.041 (n = 4).
- **The size of the bias on real GLM distributions is UNKNOWN.** These are synthetic distributions; the per-token TV here is 0.011–0.034.

### A2. Why the draft noise is aligned to the verification position

**The code aligns it on purpose:**
- `G/speculator.py:340`: "We must add 1 to the positions to match the Gumbel noise used for draft and target sampling."
- `G/dflash/speculator.py:261-265`: "sample_pos is the predicted token's position Q; verification keys Gumbel by the predecessor (Q-1)".
- The DFlash2 walk uses `sample_pos - 1` (`G/dflash2/speculator.py:89`).
- The residual resample, the bonus token and the ordinary sampler all key `gumbel_seed = tl.randint(seed, pos)` (`gumbel.py:138-139`) on the row whose `pos` is Q-1.

**What it gives:**
- The draft for position Q uses exactly the noise vector the non-speculative sampler would use for Q.
- If q = p, the draft is the seeded non-speculative sample and is always accepted. Speculative and non-speculative runs of a seeded request then emit the same tokens.
- For q ≠ p it is only a coupling: the draft is argmax(log q + G), the non-speculative sample is argmax(log p + G), and they agree only when q and p pick the same token under that noise. Acceptance does not make them agree (table below).

**Nothing in verification depends on it:**
- Acceptance is the probability-ratio test `log p > log u + log q` against the cached q (`rejection_sampler_utils.py:662`). u = `tl_rand32(seed, pos)` is a scalar (`:569`), not the noise vector.
- The bonus token is keyed by its own row.
  - With adaptive-K, the bonus row's key P+n is also the key of the unverified draft d_{n+1}. But d_{n+1} is not part of the acceptance event, so the bonus stays unbiased.
  - Measured: STOCK bonus row p = 0.58 (n = 7, bonus at r7) and 0.31 (n = 4, bonus at r4 while 7 were drafted).
- Upstream vLLM reached the same conclusion. PR #54282 ("decouple the draft's Gumbel noise from the target's", merged 2026-08-29, commit fe755c8) salts the **draft** stream offset by `1 << 30`. It notes that the acceptance test is a ratio test against the cached draft distribution, not a Gumbel coupling.

**What the alignment buys:** agreement of the emitted token with the non-speculative sample at the same (seed, pos), E1.

| Variant | All requests | Accepted requests (emitted = draft) | Rejected requests (emitted = residual) |
|---|---|---|---|
| STOCK | 0.6683 (partly the bias itself) | 0.7464 | 0.5510 |
| PATCHED | 0.6191 | 0.7464 | 0.4279 |
| Independent draft noise, as upstream #54282 | 0.5055 | 0.4170 | 0.6387 |

- Salting the resample side, as done here, keeps the draft on the non-speculative noise vector. That is all it keeps.
- **An accepted draft is not the seeded non-speculative sample**: in E1, 74.6 % of accepted drafts equal it (41.7 % with independent draft noise). The test asserts that this is below 0.99.
- Only q = p makes seeded speculative and non-speculative runs agree token for token. Neither A nor #54282 gives token-level reproducibility across the two modes; this is a statistical coupling, not a preserved property.

### A3. The patch (`overlay/patch_spec_resample_noise.py`)

**Change:**
- In `_resample_kernel`, only for a *rejected, valid* draft *with draft logits present*, the residual Gumbel noise is keyed by **lane 1 of `Philox(seed, pos)`** (`tl.randint4x(seed, pos)[1]`) instead of lane 0.
- Lane 0 is the draft's key and the acceptance draw. Lane 1 is used nowhere else.
- The math is otherwise the same as `gumbel_block_argmax`: fp32 `log1p` tail form, or the fp64 path.
- Bonus rows, placeholder rows, greedy requests and the one-hot (greedy-draft) branch keep the stock call.
- `GLM53_SPEC_RESAMPLE_INDEPENDENT=0` compiles the new branch out (`tl.constexpr`).
- The value is read once at import. **There is no default**: unset, or anything other than 0 or 1, raises, both in the patch script at container start and again at import.
  - Why: each TP rank samples on its own and feeds its own sampled tokens into the next step (V2 model runner: `last_sampled=self.req_states.last_sampled_tokens`, `V/v1/worker/gpu/model_runner.py:747`; the only sampled-token broadcast, `:1845`, is for PP).
  - Two ranks in different modes emit different residual tokens at the first rejected sampled draft, then run different token ids through the all-reduced layers. Nothing raises.
  - With the former in-file default of 1, a worker whose container never received the variable (a missing name in the launcher's worker env list) ran mode 1 whatever the head ran, so a rollback to 0 reached the head only. Now that worker does not start.
  - Not detected in-process: two containers given *different explicit* values. The launcher builds both ranks' values from one shell variable, so this needs a hand edit. Compare the start lines of both ranks after every restart (install step 4). A cross-rank collective check was not added: the only startup call site (`spec_decode_rejection_warmup`) swallows exceptions, and a collective in the sampler path risks a hang.
- With `rejection_sample_method="block"`, the patched `rejection_sample` logs a one-time warning that block verification is not exact (A5). Numerics are unchanged by the warning.

**Unchanged, bit for bit (same seeds, same logits):**

| Property | Evidence |
|---|---|
| Acceptance decisions / accepted lengths | 0 mismatches in 1,024,000 (E1), 153,600 (E2 n=7), 153,600 (E2 n=4), 12,800 (mixed batch) |
| Bonus tokens | 0 mismatches over 614,597 accepted requests (E1) |
| Greedy (temperature 0) requests | every emitted token identical: 0 diffs over 12,800 requests in a half-greedy batch |
| `GLM53_SPEC_RESAMPLE_INDEPENDENT=0` | identical to the stock file everywhere: 0 mismatches in E1, E2 and the mixed batch |
| Output at rejections only | the emitted token changed in 0.1236 of E1 requests |

- Cost: `rejection_sample`, 8 rows per request, stock → patched: B=1 66.4 → 66.4 µs; B=8 348.6 → 344.4 µs (noise level).
- Fail-closed checks (in the test): a drifted anchor, a partially patched file, an invalid env value and an unset env value all stop the patch with the file untouched; a second run is a no-op; importing the patched module with the variable unset raises.
- The stock file's sha256 `659e82c2…a98a` is printed ("stock sha256 match"), for information only. The anchors are what is enforced.

**Relation to upstream:**
- #54282 fixes the same bug from the draft side. Either side restores exactness of **standard** verification (the production setting); both together are still exact. Neither makes block verification exact (A5).
- When the image moves to a vLLM that contains fe755c8, drop this overlay. If that release changed the anchored text, the patch stops the container at start (fail closed), and it must be removed from the overlay order.

### A4. Alternative: `draft_sample_method="greedy"` (vLLM's default; not implemented)

- **Exactness:** with argmax drafts, q is one-hot and the draft does not depend on the noise, so the stock verifier is exact. Measured on the real kernels (E3, one-hot branch, same synthetic model): every row unbiased (min χ² p 0.098 / 0.117 at 4× samples).
- **Acceptance trade-off:**
  - Per sampled position, greedy drafts are accepted with probability p(argmax q) ≤ max p.
  - Probabilistic drafts are accepted with Σ min(p, q) = 1 − TV(p, q).
  - The two agree when the target is confident; greedy loses when the target is spread (prose at temperature 1.0 / top_p 0.95).
  - E1 example: greedy 0.1 vs probabilistic 0.6.
  - E3 synthetic model, mean accepted drafts per step: greedy 0.304 vs probabilistic 2.275 (n = 7), and 0.307 vs 1.880 (n = 4).
  - These are synthetic numbers. **Real acceptance on sampled traffic is UNKNOWN.** The production bench runs only at temperature 0, where both methods draft the same argmax.
- **Other effects:**
  - Greedy traffic is unchanged: at temperature 0 the walk is an argmax either way.
  - It removes the fp32 draft-logits cache: max_num_seqs 8 × 7 × 154880 × 4 B = 34.7 MB. It also removes the per-step cache kernel. Both are tiny.
- **Recommendation:** keep probabilistic drafts with patch A.

### A5. Block verification is not exact, with or without A (review finding F1)

**Mechanism** (`G/rejection_sampler_utils.py` in the image):
- Standard verification stops at the first rejection: `verifying = accepted` (:663). It draws u = `tl_rand32(seed, pos)` (:569) only while `verifying`.
- Block verification (`rejection_sample_method="block"`, branch at :595) clears `verifying` only for invalid drafts (:565). Every valid row draws its u, and its h depends on the next draft token (`next_draft_token`, :600). The step ends at the last row with u ≤ h (`accepted_length = tl.where(u <= h, i + 1, accepted_length)`, :626).
- Ending at τ < n−1 therefore conditions the rows after τ: u > h there, and their drafts entered h.
- The next step's row j has pos = P + τ + 1 + j. It reuses exactly those (seed, pos) keys: the same u, and the same draft Gumbel key (the draft for position Q is keyed by Q−1, `G/dflash/speculator.py:261-265`).
- So consecutive steps are coupled. A re-keys only the residual draw inside one step and cannot remove this. Upstream #54282 (draft-side salt) does not either: u is still reused.

**Measured (E4, real kernels):** 3 drafts; target p = (.1, .5, .4) on every row, q = (.5, .4, .1) for every draft, so every emitted token should be p, independent of the past. Two consecutive steps per request with the same seed; step 2 starts at P + num_sampled of step 1, as in production. 300 launches × 512 requests.

| Variant | Step 1, per row: min χ² p | Step 2 first token (all): TV / χ² p | Step-2 length vs step-1 length: χ² p |
|---|---|---|---|
| standard + A | 0.035 | 0.0013 / 0.55 | 0.46 |
| block, stock | 0 (TV 0.050 on r0) | 0.0635 / 0 | 0 |
| block + A | 0.158 | 0.0091 / 6.4e-32 | 0 |
| block + A, step 2 with a fresh seed (control) | 0.158 | 0.0006 / 0.72 | 0.66 |

- Standard + A is exact within a step and across steps.
- Block + A is exact within one step but not across steps. Given that step 1 ended after 1 or 2 tokens, step 2's first token is (0.111, 0.497, 0.392) and (0.121, 0.494, 0.385) instead of (0.1, 0.5, 0.4) (χ² p 1.2e-18 and 6.1e-34).
- With a fresh seed for step 2, block + A is exact, so the remaining bias is the key reuse.

**Consequence:**
- Block verification is not a quality-neutral lever in this image, whatever the order of installation. DRAFTER_PLAN §3 row 7 and §7 are corrected.
- Making it exact needs fresh per-step randomness for both u and the draft keys (for example keys salted by the step's first position). That touches the drafter as well as the verifier and is not implemented.
- The patched `rejection_sample` logs once: `[glm53-resample-noise] rejection_sample_method='block' is not exact for sampled requests in this image, …`. The test checks that it fires exactly once in block mode and never in standard mode.

---

## B. Drafter decoder layers → FP8 (DRAFTER_PLAN D1)

### How the drafter gets its quant method today (verified with the real config classes)

- `DFlashQwen3Model.__init__` sets `self.quant_config = get_draft_quant_config(vllm_config)` (`qwen3_dflash.py:372`). That is the quant config of the **draft** `ModelConfig`.
- The launcher's speculative config has no `quantization` key (launcher snapshot, `start.sh` head script), and the DFlash2 `config.json` has no `quantization_config`.
- Built from `EngineArgs` with the production speculative config and the real GLM-5.3-Flash config, the result is:
  > `target: quantization=exl3 quant_config=Exl3Config; draft: quantization=None; get_draft_quant_config -> None`
- So every drafter linear gets `UnquantizedLinearMethod` (BF16) and **never reaches `Exl3Config.get_quant_method`**.
- The `"draft" in prefix` exclusion at `exl3.py:1965` is therefore not what keeps the drafter BF16.
  - The drafter's prefixes contain no "draft": `model.layers.45.mlp.down_proj … model.layers.49.self_attn.qkv_proj` (start_layer_id = the target's 45 layers) and `model.fc`.
  - They would even match GLM53_DENSE_FP8's `dense` suffixes if the drafter ever received an Exl3Config.

### Measured on nodeC (`tests/drafter/bench_drafter_fp8.py`, log `bench_drafter_fp8.log`)

Setup:
- Real DFlash2 weights, TP=2 rank-0 shards: qkv 3072×4096, o 4096×2048, gate_up 12288×4096, down 4096×6144, per layer, × 5.
- BF16 path: `UnquantizedLinearMethod.apply` (→ `F.linear`).
- FP8 path: production's `exl3.Glm53DenseFp8Method` (`process_weights_after_loading` + `apply`).
- Timing: CUDA-graph replay, cold weights. Each figure is the median of 15 reps; [min–max] of the reps in brackets.

| B (M = 8B rows) | BF16 5 layers | FP8 5 layers | Saved per step per rank |
|---|---|---|---|
| 1 (8) | 4.316 [4.287–4.370] ms (223 GB/s) | 2.097 [2.084–2.139] ms (230 GB/s) | **2.219 ms** |
| 2 (16) | 4.362 [4.320–4.629] ms | 2.125 [2.096–2.156] ms | **2.237 ms** |
| 4 (32) | 4.360 [4.332–4.434] ms | 2.127 [2.115–2.152] ms | **2.233 ms** |
| 8 (64) | 4.530 [4.483–4.626] ms (213 GB/s) | 2.266 [2.250–2.353] ms (213 GB/s) | **2.264 ms** |

- Eager (no graph), B=1: 4.360 → 2.148 ms (saved 2.212 ms).
- Per kind at B=1, each figure the **sum of that kind's 5 matrices** (one per layer): qkv 0.571→0.274, o 0.393→0.196, gate_up 2.246→1.082, down 1.115→0.554 ms. One matrix is a fifth of that, e.g. one gate_up ≈ 0.45 → 0.22 ms.
- **fc 20480→4096** (replicated; in production it runs **eagerly**, outside the drafter graph, on the target's T·B tokens). Timed three ways, BF16 → FP8 in µs (saved):

| T, B (M) | Graph replay | Eager, back-to-back | Eager, one call with a host sync around it |
|---|---|---|---|
| 5, 1 (5) | 753.2 → 372.8 (+380.4) | 763.6 → 375.6 (+388.1) | 793.0 → 396.2 (+396.7) |
| 5, 8 (40) | 788.1 → 380.9 (+407.3) | 788.5 → 378.9 (+409.5) | 802.4 → 397.4 (+405.0) |
| 8, 1 (8) | 758.3 → 373.1 (+385.2) | 754.1 → 373.2 (+380.9) | 766.1 → 392.7 (+373.4) |
| 8, 8 (64) | 800.9 → 386.8 (+414.1) | 794.6 → 389.3 (+405.3) | 808.8 → 405.2 (+403.6) |

  - The synced column includes the host dispatch of the Marlin wrapper: FP8 costs 16–21 µs more there than back-to-back.
  - Eager saving: **0.37–0.41 ms/step**, the same as under graph replay within the spread.
- Accuracy on the real weights (relative output error, M = 8):

| Matrix | Median / max | With 1/256 outlier channels ×30 |
|---|---|---|
| qkv | 2.649e-2 / 2.660e-2 | |
| o | 2.649e-2 / 2.657e-2 | |
| gate_up | 2.652e-2 / 2.656e-2 | |
| down | 2.661e-2 / 2.667e-2 | ≤ 2.680e-2 |
| fc | 2.600e-2 | |

  - For comparison, production has already accepted kda at 2.79e-2 / 2.84e-2 and dense at 2.28e-2 / 2.30e-2 (`docs/logs/fp8_groups_check.log`).
  - Marlin vs the dequantized reference: ≤ 1.67e-3.
- Memory: 0.965 → 0.482 GB per rank.

### The patch (`overlay/patch_drafter_fp8.py`)

- It wraps the call at `qwen3_dflash.py:372`. With `GLM53_DRAFT_FP8` set, the draft quant config becomes `Glm53DraftFp8Config`. Its `get_quant_method` returns:
  - `exl3.Glm53DenseFp8Method("draft")` for `model.layers.*.self_attn.{qkv_proj,o_proj}` and `model.layers.*.mlp.{gate_up_proj,down_proj}`, and for `model.fc` when the value is `layers,fc`;
  - `UnquantizedLinearMethod` for every other linear (the conv `kernel_projection`s, the selector's `hidden_projection`);
  - `None` for non-linear layers. Attention then behaves exactly as with no quant config.
- The launcher overlay's `Glm53DenseFp8Method(group, prefix)` signature (TP=3 Marlin gate) is detected with `inspect` and given the prefix.
- `model.fc` is a `ReplicatedLinear` without TP partition sizes, so the config sets them (they equal the full sizes) before Marlin reads them. This was found by the build test, where it failed first.
- It fails closed:
  - if the drafter already has a quant config;
  - if the context-K/V fusion (`_build_context_kv_buffers`, which slices `qkv_proj.weight`) ever runs after the FP8 repack. Normally it runs at the end of `load_weights`, before `process_weights_after_loading`, and keeps a BF16 copy of the K/V rows.
- The target is untouched. Default `off` returns the stock config object unchanged.
- It writes the canonical mode (`off`, `layers`, `fc` or `layers,fc`) into the patched file as `_GLM53_DRAFT_FP8_INSTALLED`, and the drafter build fails closed if the runtime `GLM53_DRAFT_FP8` differs from it. A file patched for another mode gets only that line rewritten (`re-patched (mode off -> layers)`). Reason: the compile caches, next subsection.

### B and vLLM's compile caches (review finding, fixed)

**Production compiles the drafter:**
- `ENFORCE_EAGER` defaults to 0 (launcher snapshot `start.sh:277`).
- `DFlashQwen3Model` is `@support_torch_compile` (`qwen3_dflash.py:348`); DFlash2's model subclasses it and inherits its `forward`.
- torch is 2.13, so `VLLM_USE_AOT_COMPILE` defaults to 1 (`V/envs.py:353-362`).
- The cache lives on a persistent host volume (`start.sh:2148` head, `:2107` worker). Nothing in the launcher sets `VLLM_DISABLE_COMPILE_CACHE` or `VLLM_FORCE_AOT_LOAD`.

**The cache key never sees `GLM53_DRAFT_FP8`:**
- The AOT key is sha256 of `aot_compile_hash_factors(vllm_config)` plus `_model_hash_key(forward)` (`V/compilation/decorators.py:539-547`).
  - The first is `envs.compile_factors()` (known `VLLM_*` variables only) plus `vllm_config.compute_hash()`, where the draft quantization is None in every mode.
  - The second is the vLLM version, the forward's qualname and its first line.
- Measured in the build test: the key is `31451ce1baef…` for `off`, `1` and `layers,fc` alike.
- A loaded artifact runs with guards disabled (`evaluate_guards` defaults to False, `V/config/compilation.py:368`; `decorators.py:305`).

**What the loader does check** is the text of every source file torch traced: the root frame and every inlined frame, with the full module text (the root is appended at `torch/_dynamo/output_graph.py:754`; recorded by `aot_compile.py:442-444` through `package.py:171-190`). `_verify_source_unchanged` (`decorators.py:265`) re-hashes those files; on a difference, the load fails and vLLM compiles again (`:307-324`, log `Compiling model again due to a load failure from …, reason: Source code has changed since the last compilation.`). The non-AOT VllmBackend cache hashes the same traced files (`V/compilation/backends.py:1036-1063`).

**Before the fix:** the patched file was byte-identical for every value. The pre-review patch script (commit a4d51fa) gives one sha256, `a6006ddcca19`, for `off`, `1`, `fc` and `layers,fc` (`tests/drafter/review_counterfactual.py`, log `review_counterfactual.log`). A graph traced for one weight layout would then be loaded for the other: the BF16 graph over Marlin-packed int32 weights after enabling B, or the FP8 graph (reading `weight_scale`/`workspace`) over BF16 layers after the env-only rollback. That end-to-end failure was inferred, not run.

**After the fix** (build test):

| Check | Result |
|---|---|
| Patched text for `off`, `1`, `fc`, `layers,fc` | 4 distinct sha256 (`fac4daf8e75e`, `ed07b7d1906f`, `e9aad1a4b353`, `74cf14506c0b`) |
| Sources torch records for the drafter's forward (real `SourceInfo.add_code` on `DFlashQwen3Model.forward` and `DFlash2Qwen3DecoderLayer.forward`) | `qwen3_dflash`, `qwen3_dflash2` |
| vLLM's `_verify_source_unchanged`, artifact recorded under `off`, file re-patched to `layers` | raises "Source code has changed" → recompile |
| Same, file back to `off` | accepts |
| Runtime `GLM53_DRAFT_FP8` ≠ the written mode (4 combinations) | drafter build raises |

- **Cost:** the first start after installing B, every toggle and every rollback recompiles the drafter once at start. The artifacts of different modes share one key, so switching back and forth recompiles each time.
- **C and A are not affected:**
  - `compute_candidates` is not inside a compiled module: `@support_torch_compile` covers `DFlashQwen3Model` and `CandidateSelector` only, and `compute_candidates` is called eagerly from `G/dflash2/speculator.py:234`. `GLM53_DRAFT_LMHEAD_FP8` never enters a compiled graph. Installing C changes the text of `qwen3_dflash2.py`, which forces one recompile at the first start.
  - A is a Triton kernel; its switch is a `tl.constexpr`, which is part of Triton's own cache key.
- **Not verified on nodeC:** a full AOT compile, save and reload of the real drafter.

### Verified with vLLM's own classes (`tests/drafter/test_drafter_fp8_build.py`, log `test_drafter_fp8_build.log`, 53/53)

Setup:
- The real drafter is built by `load_dflash_model` → `get_model` → `DefaultModelLoader` → `process_weights_after_loading`.
- The `VllmConfig` comes from `EngineArgs` with the production speculative config.
- One GB10, TP=1, patched copies imported under the canonical module names.

| Check | Result |
|---|---|
| Stock build | 32 linears, all `UnquantizedLinearMethod` |
| `GLM53_DRAFT_FP8=1` | exactly the 20 projections are `Glm53DenseFp8Method` and processed; fc and the other linears stay BF16 |
| `layers,fc` | 21 FP8 (projections + `model.fc`); fc output rel 2.594e-2 |
| Module outputs vs the stock build (rel) | qkv 2.673e-2, o 2.691e-2, mlp 4.229e-2 (two FP8 matmuls in series); fc bit-identical while it stays BF16 |
| Fused context K/V weight | bitwise equal to the checkpoint K/V rows in both builds |
| Linear weight bytes | stock 2.033 GiB → FP8 1.135 GiB at TP=1, i.e. **0.482 GB per rank at TP=2** |
| CUDA-graph capture and replay of FP8 layer-0 mlp / qkv_proj | bitwise equal to eager |
| `torch.compile(fullgraph=True)` of the same modules | traces the Marlin op; rel vs eager 0.00e+00 |
| Launcher overlay's `(group, prefix)` signature (stub subclass) | works |

### Not verified on nodeC (production only)

- TP=2 sharding through the real classes. The patch is TP-agnostic, and the shapes are covered by the bench.
- vLLM's own compile backend and FULL-graph capture of the whole drafter. Only single modules were covered, with plain `torch.compile` and `torch.cuda.graph`. The compile-cache key and vLLM's loader check were run on the real drafter code (previous subsection); an actual AOT compile/save/reload was not.
- `LOAD_FORMAT=instanttensor`. It is the same `DefaultModelLoader.load_model` flow with another weights iterator (`model_loader/__init__.py:53`, `default_loader.py:272`). On nodeC it staged the checkpoint in non-PyTorch GPU buffers (the process reached 12.3 GiB), which exceeds the 8 GiB test cap, so the build test uses safetensors.
- **Acceptance rate** with FP8 drafter weights.
- Real step time.

---

## C. Drafter-only FP8 lm_head copy (DRAFTER_PLAN D2a)

### Measured (`bench_drafter_fp8.py`)

Setup: the target's BF16 lm_head vocab half per rank, 77440 × 4096, drafter M = 7B rows.

Each figure is the median of 3 independent trials (new graph, new input), each trial the median of 15 reps.

| B (M) | BF16 | FP8 | Saved per step per rank | BF16 lowest rep over the 3 trials |
|---|---|---|---|---|
| 1 (7) | 2.751 ms (231 GB/s) | 1.390 ms | **1.361 ms** | 2.730 ms |
| 2 (14) | 2.761 ms | 1.389 ms | **1.373 ms** | 2.741 ms |
| 3 (21) | 2.971 ms (214 GB/s) | 1.390 ms | **1.580 ms** | 2.943 ms |
| 4 (28) | 2.984 ms (213 GB/s) | 1.400 ms | **1.583 ms** | 2.964 ms |
| 5 (35) | 2.824 ms | 1.415 ms | **1.409 ms** | 2.782 ms |
| 6 (42) | 2.807 ms | 1.424 ms | **1.382 ms** | 2.785 ms |
| 7 (49) | 2.817 ms | 1.431 ms | **1.386 ms** | 2.790 ms |
| 8 (56) | 2.827 ms (224 GB/s) | 1.440 ms | **1.387 ms** | 2.798 ms |

- FP8 is flat, 1.39–1.44 ms.
- At B = 3–4 (M = 21, 28) the BF16 GEMM itself is ~0.2 ms slower. This is repeatable: 3 of 3 trials, lowest reps 2.943–2.964 ms against 2.730–2.798 ms at the other M; M = 28 was already the outlier in the pre-review log (2.968 ms, commit a4d51fa). It is a shape-specific BF16 kernel choice, not an FP8 benefit.
- Budget with **1.36–1.41 ms** (median over B = 1..8: 1.386 ms). The 1.58 ms holds at B = 3–4 only.

- **Memory cost per rank: 302.6 MiB.** Packed weight 302.5 MiB, scales 151.2 KiB, workspace 0.2 KiB.
- Candidate quality is indicative only (synthetic hidden states, full gathered vocab):

| Hidden states | Median max-prob | Top-1 agreement | Top-16 set overlap | Identical top-16 sets | Softmax TV |
|---|---|---|---|---|---|
| Random directions | 0.040 | 0.9329 | 0.9496 | 0.3403 | 0.0307 |
| Peaked | 1.000 | 0.9995 | 0.9895 | 0.8367 | 0.0004 |

- Worthwhile: ~1.39 ms/step (1.58 at B = 3–4), about 60 % of the D1 saving, for 0.3 GiB per rank.

### The patch (`overlay/patch_drafter_lmhead_fp8.py`)

- After `load_dflash_model` points the drafter's `lm_head` at the target's BF16 `ParallelLMHead` (`G/dflash/utils.py:84`), `GLM53_DRAFT_LMHEAD_FP8=1` attaches `dflash_model.glm53_candidate_head`: a per-rank FP8 copy of that shard.
  - The quantization is the same as `Glm53DenseFp8Method`, computed in 8192-row chunks, so the fp32 transient is about 128 MiB instead of 2.5 GiB.
  - It uses the production class's `apply`.
- `compute_candidates` (`qwen3_dflash2.py:291`) uses the copy.
- The target keeps its BF16 lm_head (the same object) for verification, sampling and logprobs.
- The copy is a plain attribute, not a registered submodule, so loader passes, `state_dict` and reloads never see it. Its apply adapter is not a `QuantizeMethodBase`.
- It fails closed:
  - for a drafter without `compute_candidates`;
  - if `compute_candidates` does not consume the copy (`qwen3_dflash2.py` replaced after the patch);
  - for an fp32 `head_dtype`;
  - for any env value other than 0 or 1.
- The copy is taken once at load. Do not combine it with in-place target-weight reloads (RL refit / sleep-mode reload).

### Verified (build test, same run as B)

| Check | Result |
|---|---|
| Candidate head attached; drafter `lm_head` is still the target's object | PASS |
| Target lm_head bytes | bitwise equal to the checkpoint afterwards |
| Verification-side `compute_logits` | bitwise equal to `F.linear(h, BF16 lm_head)` |
| `compute_candidates` consumes the copy | its top-16 equals the copy's top-16 |
| Top-16 overlap with the BF16 head (random hidden) | 0.953; top-1 agreement 0.911 |
| Copy's packed Marlin weight, scales and `apply` | bitwise equal to the production `Glm53DenseFp8Method` on the same rows |
| Copy size | 605.3 MiB at TP=1 (full vocab), i.e. 302.6 MiB per rank at TP=2 |

**Not verified on nodeC:** TP=2 (the copy is per-rank and the gather is unchanged, but the path was not run); acceptance; step time; memory headroom on production.

**Memory:** the copy is +302.6 MiB per rank of **non-KV** memory, allocated at model load, with no headroom check in the patch.
- The KV pool is pinned in production: whenever `ENFORCE_EAGER` != 1, the launcher appends `--kv-cache-memory ${KV_CACHE_BYTES:-9484754862}` (`configure_kv_cache_memory`, snapshot `start.sh:306-316`), and vLLM then uses that size without profiling (`V/v1/worker/gpu_worker.py:475-494`). The KV-capacity line cannot shrink, so it is no gate (the install gate is replaced, step 4).
- Snapshot defaults: `GPU_MEM_UTIL` 0.85 (`start.sh:197`), KV 9,484,754,862 B = 8.83 GiB. The ops note's 0.87 / 15 GiB may be production `.env` overrides (UNKNOWN; not read).
- Enable C together with B: net −0.15 GiB per rank. C alone adds +0.30 GiB of non-KV memory per rank on the unified-memory GB10.

---

## Expected effect per speculative step (nodeC cold-weight replay, per rank; UNVERIFIED on production)

| B | D1 layers | D2a lm_head copy | D1 + D2a | + fc (`layers,fc`), eager with host sync |
|---|---|---|---|---|
| 1 | 2.219 ms | 1.361 ms | 3.581 ms | ≈ 3.95–3.98 ms (fc +0.373 at T=8, +0.397 at T=5) |
| 8 | 2.264 ms | 1.387 ms | 3.650 ms | ≈ 4.05 ms (fc +0.404 at T=8, +0.405 at T=5) |

- Against the corrected 92.5–137.2 ms step (DRAFTER_PLAN §1), D1 + D2a is ≈ +2.7–4.0 % tok/s. This is arithmetic only; it assumes the drafter is on the critical path, which it is (sequential with verify).
- Net memory per rank: D1 −0.482 GB and D2a +0.317 GB, so −0.165 GB; −0.25 GB with fc.
- **Quality:**
  - Greedy outputs stay exact (verification is unchanged), apart from the T-dependent near-tie flips that already exist.
  - Sampled outputs stay exact only with A installed, and only with `rejection_sample_method="standard"` (A5). Without A, any drafter change shifts the bias (DRAFTER_PLAN §2). **Install A first, or together with B/C.**

---

## Production install (operator; every step needs explicit owner approval)

Line numbers below are from the launcher snapshot (09-24). Re-check them against `a15d696` before editing.

1. Copy the three `overlay/patch_*.py` files into the launcher's `overlay/` directory.
2. In `start.sh`, for each file, mirror what `patch_dense_fp8.py` has:

   | What | Snapshot line | Values |
   |---|---|---|
   | `*_PATCH_HOST` variables | ~233 | `RESAMPLE_NOISE_PATCH_HOST`, `DRAFT_FP8_PATCH_HOST`, `DRAFT_LMHEAD_FP8_PATCH_HOST` |
   | `validate_overlay_artifacts` entries | ~761 | `"$RESAMPLE_NOISE_PATCH_HOST\|[glm53-resample-noise]\|$main_guard"`, `"$DRAFT_FP8_PATCH_HOST\|[glm53-draft-fp8]\|$main_guard"`, `"$DRAFT_LMHEAD_FP8_PATCH_HOST\|[glm53-draft-lmhead-fp8]\|$main_guard"` |
   | Preflight existence checks | ~1131 | |
   | Worker `scp` to `/tmp/` | ~1885 | |
   | Worker `-v '/tmp/patch_X.py:/opt/glm53/patch_X.py:ro'` | ~2126 | |
   | Head `-v "$X_PATCH_HOST:/opt/glm53/patch_X.py:ro"` | ~2167 | |
   | `GLM53_OVERLAY_ORDER` | ~1632 | append after `patch_dense_fp8.py`: `patch_spec_resample_noise.py`, `patch_drafter_fp8.py`, `patch_drafter_lmhead_fp8.py` |
   | Env defaults | ~412 | `GLM53_SPEC_RESAMPLE_INDEPENDENT="${GLM53_SPEC_RESAMPLE_INDEPENDENT:-1}"`, `GLM53_DRAFT_FP8="${GLM53_DRAFT_FP8:-off}"`, `GLM53_DRAFT_LMHEAD_FP8="${GLM53_DRAFT_LMHEAD_FP8:-0}"` |
   | Worker env list | ~2041 | add the three names |
   | Head `-e` | ~2226 | add the three names |

   - The patch files run at every container start, inside the container, on the image's files. That is why they check anchors instead of assuming a stock tree.
   - The worker gets only the names in its env list (`serve_env+=" -e $v='${!v:-}'"`).
     - A missing name for A stops the worker container at start: patch A has no in-file default (`must be set to 0 or 1 on every rank`). This is deliberate (A3).
     - A missing name for B or C leaves that rank at `off`/`0`. The ranks stay in sync, because the drafter's all-reduce and all-gather give both ranks the same tensors, but the change is then applied on one rank only. Step 4 catches it.
   - Only `patch_dflash2.py` (Dockerfile build time, already in the image) touches one of these files (`qwen3_dflash.py`), and not at these anchors.
3. Restart both nodes, since TP=2 needs both: stop → 25 s → `SKIP_BUILD=1 ./start.sh start` (ops note, DRAFTER_PLAN §7).
   - The first start after installing B or C, and every start after changing `GLM53_DRAFT_FP8`, recompiles the drafter. Expect `Compiling model again due to a load failure from …, reason: Source code has changed since the last compilation.` once per rank, and a longer start.
4. After **every** restart (install, every toggle, every rollback), check on **both** ranks. The three start lines must be byte-identical between head and worker:
   - `[glm53-resample-noise] rejection_sampler_utils.py: patched; stock sha256 match; runtime independent residual noise` (or `runtime stock noise (GLM53_SPEC_RESAMPLE_INDEPENDENT=0)`)
   - `[glm53-draft-fp8] qwen3_dflash.py: patched; stock sha256 match; mode layers (FP8 groups)` (or `mode off (stock BF16 drafter)`)
   - `[glm53-draft-lmhead-fp8] utils.py: patched (stock sha256); qwen3_dflash2.py: patched (stock sha256); runtime …`
   - Compare them with `docker logs <container> 2>&1 | grep -E '^\[glm53-(resample-noise|draft-fp8|draft-lmhead-fp8)\]'` on each node. Nothing else compares the ranks' modes (A3).
   - When B/C are on, at model load: `[glm53-draft-fp8] drafter linears -> FP8 e4m3 per output channel (Marlin): layers` and `[glm53-draft-lmhead-fp8] drafter candidate head: FP8 copy of lm_head shard (77440, 4096) (+302.6 MiB this rank)`.
   - **Memory gate**, per rank, against the baseline start. The KV-capacity line is not a gate: the KV pool is pinned by `--kv-cache-memory` (section C, Memory), so it cannot shrink.
     - `Model loading took X GiB` (it includes the drafter): expected change B −0.45 GiB, C +0.30 GiB, B + C −0.15 GiB.
     - `Graph capturing finished in … secs, took Y GiB`: unchanged.
     - Host `MemAvailable` on both nodes after warm-up and after a long-prefill soak at the maximum concurrency: no lower than baseline minus the load delta, and above the launcher's own preflight headroom.
     - Any shortfall: roll back C first.
5. Suggested order (DRAFTER_PLAN §7 protocol: ABA, n ≥ 10 per workload, permutation test):
   1. **A alone.** Temperature-0 bench unchanged (A cannot change greedy). Add a temperature 1.0 / top_p 0.95 bench to confirm sampled acceptance is unchanged.
   2. **A + B (`GLM53_DRAFT_FP8=1`) + C (`GLM53_DRAFT_LMHEAD_FP8=1`)** as one A/B.
      - Gate: greedy output identity on 20 fixed prompts, allowing only near-tie flips at the baseline-vs-baseline rate.
      - Measure acceptance (`pos[]`) and tok/s.
   3. Optionally `GLM53_DRAFT_FP8=layers,fc`.

## Rollback

| Part | Rollback | Note |
|---|---|---|
| A | `GLM53_SPEC_RESAMPLE_INDEPENDENT=0` on **both** ranks and restart; check the start line on both ranks | bit-identical to stock (verified). A rank that did not get the variable does not start (no default) |
| B | `GLM53_DRAFT_FP8=off` and restart | the patch rewrites the mode line, the drafter is recompiled once (compile caches are invalidated by the file text), and the patched code returns the stock config object |
| C | `GLM53_DRAFT_LMHEAD_FP8=0` and restart | the attach is a no-op; `compute_candidates` falls back to `self.lm_head` |

- Removing an entry from `GLM53_OVERLAY_ORDER` (and its mount) also reverts that part, because each start patches the image's untouched files. For B and C, the file text returns to stock, so the drafter is recompiled once.
- A patch that fails closed at start (anchor drift after an image update) stops the container before vLLM loads. Remove it from the order list to recover.

## What only production can verify

1. Sampled-traffic acceptance with A. By construction it is unchanged, since accept decisions are bit-identical for identical inputs. What changes is the emitted token at rejections: 12.4 % of E1 requests, a workload-dependent fraction in production.
2. The size of the removed bias on real GLM distributions and prompts. It is synthetic here: per-token TV 0.011–0.034.
3. Acceptance with an FP8 drafter (B) and with FP8 candidates (C). The per-matrix error is the same class as production's dense/kda FP8; the candidate top-16 overlap is 0.95–0.99 on synthetic states.
4. Real step-time savings at TP=2 across nodes, under vLLM's compile + FULL graph and adaptive-K. Also whether the drafter is graphed at all (DRAFTER_PLAN D3).
5. Memory headroom for the +302.6 MiB per rank copy (the memory gate in install step 4; the KV line is pinned and cannot show it).
6. The instanttensor loader path. (The launcher overlay's `exl3.py` is now verified on nodeC with the real file: see "Launcher overlay exl3.py" below.)
7. The fraction of production traffic that is sampled. The checkpoint `generation_config` defaults to temperature 1.0 / top_p 0.95 (DRAFTER_PLAN §2).
8. The drafter's recompile after a mode change under vLLM's AOT cache (the `Source code has changed` line), and a clean start with `ENFORCE_EAGER=0` after each toggle and rollback.

## Launcher overlay exl3.py (2026-09-27 night)

Production does not run the image's `quantization/exl3.py`: the launcher (df864b5) mounts its `overlay/exl3.py` at
`/opt/glm53/exl3.py` and `patch_dense_fp8.py` copies it over the image's file at every container start (copy:
`docs/ref/prod_live/overlay_exl3.py`; mechanism reproduced from the 09-24 launcher snapshot, see below). What matters here:
`Glm53DenseFp8Method(group, prefix)` instead of `(group)`, `process_weights_after_loading` asks vLLM for the TP world size
(a TP=3-only KDA gate) and `apply` has a KDA large-M BF16 branch (only for `group == "kda"` with
`GLM53_KDA_BF16_LARGE_M=1`). B and C were already signature-aware; the two tests that built the class with one argument
now build it with the module's own shape, and the bench initialises a single-rank TP group.

| Check | Result |
|---|---|
| `tests/drafter/run_drafter_tests.sh` with the overlay file bound over the image's (`GPU_RUN_BIND`) | ALL PASSED (`docs/logs/drafter_live/`). The build test reports the overlay's `Glm53DenseFp8Method(group, prefix)` signature; B's config passes the prefix (checked on a real `ReplicatedLinear`); the real drafter builds put exactly the 20 projections (21 with fc) through the overlay's `process_weights_after_loading` (TP world size 1) with the same output errors as the image run (qkv 2.673e-2, o 2.691e-2, mlp 4.229e-2); C's copy is bitwise equal to the overlay class's packing and `apply`. Bench: same savings within run-to-run spread (decoder layers 2.229 ms at B = 1, lm_head copy 1.388 ms median over B = 1..8) |
| Same suite against the image file after these test changes | ALL PASSED (`docs/logs/drafter_image/`; the numbers above still come from `docs/logs/drafter/`) |
| Launcher runtime overlay chain (20 patches of the 09-24 snapshot, whose `exl3.py` is byte-identical to production's) then A, B, C (`tests/drafter/chain_launcher_overlays.sh`, CPU only) | every patch exits 0; A/B/C print "stock sha256 match" (no launcher overlay touches their files); second run "already present"; the installed `quantization/exl3.py` is the overlay (sha256 `849e2588…`); the patched modules import and B's config returns the overlay's `Glm53DenseFp8Method("draft", "model.layers.45.mlp.down_proj")` (`docs/logs/prod_live/chain_launcher_overlays.log`) |

Still to read on production (docs/PRODUCTION_PLAN.md Phase 0): df864b5's own `patch_dense_fp8.py` and `GLM53_OVERLAY_ORDER`.

## Reproduce on nodeC

`tests/drafter/run_drafter_tests.sh`. It checks MemAvailable ≥ 40 GB first (`gpu_run.sh`), runs one container at a time, keeps each test ≤ 8 GiB of PyTorch memory (peak 5.71 GiB), and writes to `docs/logs/drafter/`. It takes about 7 minutes.

`test_spec_resample_noise_4x.log` is the earlier 4×-sample run of E2 (`E2_LAUNCHES=2400`, `E1_LAUNCHES=20`), made before the review; the E1/E2 code it ran is unchanged.

## Review findings and fixes (2026-09-27)

All nine findings were re-checked against the image and the launcher snapshot. All are real.

| ID | Finding | Verdict | Fix | Test that would have caught it |
|---|---|---|---|---|
| F1 (major) | Block verification stays biased with A; the docs gated it only on P0 | Real. Measured: block + A exact within a step, biased across steps (step-2 first token χ² p 6.4e-32; step-2 length vs step-1 length p 0); control with a fresh step-2 seed exact | A5 added. DRAFTER_PLAN §3 row 7 and §7 corrected. Patch docstring limits the claim to standard verification. The patched sampler warns once in block mode | E4 in `test_spec_resample_noise.py` (two consecutive steps, standard vs block, fresh-seed control; warning fires once, only in block mode) |
| F2 (minor) | "An accepted draft is still the seeded non-speculative sample" is false | Real. E1: 74.6 % of accepted drafts equal it | A2 rewritten: a coupling of the noise vector, not token equality; table split by accepted/rejected | E1 asserts agreement among accepted < 0.99 |
| F3 (minor) and A-default-on (minor) | A's per-process switch had an in-file default of 1 and no cross-rank check; a missed worker passthrough turns a rollback into a split TP configuration | Real | No default: unset fails closed in the patch script and at import. Install step 4: compare the start lines of both ranks after every restart, including rollbacks. Two explicit different values remain undetectable in-process (A3 says why) | The patch script and the import with the variable unset must fail (resample test). The pre-review patch succeeded with it unset (`review_counterfactual.log`) |
| B-aot-cache (major) | vLLM's AOT compile cache key ignores `GLM53_DRAFT_FP8`, so a toggle or the env-only rollback reuses a graph traced for the other weight layout | Real (key identical for off/1/layers,fc, measured; guards disabled; loader checks only source text, which was identical) | B writes its mode into `qwen3_dflash.py`; a mode change changes the traced source, so vLLM's loader check forces a recompile; runtime/value mismatch fails closed. Rollback table and install step 3 state the one recompile | Build test: 4 modes → 4 distinct texts (the pre-review patch: 1, `review_counterfactual.log`); real `_verify_source_unchanged` on torch's `SourceInfo` of the drafter rejects an artifact recorded under another mode |
| C-memory (minor) | The install gate "KV-capacity line must not shrink" is vacuous: KV is pinned by `--kv-cache-memory` | Real (`start.sh:306-316`, `gpu_worker.py:475-494`) | Gate replaced by per-rank `Model loading took` / `Graph capturing … took` deltas and host MemAvailable after a soak. Recommend C only with B. The "0.87 / 15 GiB" statement is corrected to the snapshot defaults (0.85, 8.83 GiB) with the ops-note values marked UNKNOWN | Documentation only; the patch cannot see non-KV headroom on unified memory |
| M1 (minor) | The top of "1.36–1.56 ms/step" rested on one unreplicated BF16 point | Real (no spread, one run). Re-measured in 3 trials for every B = 1..8: the BF16 slowdown at M = 21 and 28 repeats in every trial, so 1.58 ms is real but holds at B = 3–4 only | C table covers B = 1..8 with the lowest rep; budget with 1.36–1.41 ms (median 1.386 ms) | The bench prints [min–max] per timing and 3 independent trials for the lm_head |
| M2 (minor) | The fc saving was measured under graph replay, but the fc runs eagerly | Real. Eager saving measured: 0.37–0.41 ms/step, the same as graph replay within the spread; the Marlin wrapper's host dispatch adds 16–21 µs per call | The fc is also timed eagerly, back-to-back and with a host sync around each call; the "+ fc" column uses the synced eager numbers | The bench's fc section |
| M3 (minor) | "Per-matrix breakdown" was the sum over 5 layers | Real | Relabelled "per kind, each the sum of its 5 matrices", with the single-matrix figure | The bench prints the per-kind label |

