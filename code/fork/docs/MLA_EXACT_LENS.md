# MLA_EXACT_LENS — plan production's sparse-MLA FA2 with the exact selected-key counts (`GLM53_MLA_EXACT_LENS`)

Status 2026-10-02, branch `opt-decodekit` (from `r16z2rev` 60b53bf). Built and measured on nodeC only (production image,
the production SM90 mounts, GPU jobs under `/tmp/tf-gpu-bench.lock`). Production (nodeA/nodeB) was read only (files,
`docker logs`); nothing was installed there. **Default OFF: unset = production byte for byte.**

## 1. The defect (known since 2026-09-28, docs/MLA_PREFILL.md section 6, never fixed)

Production's `FLASHINFER_MLA_SPARSE_SM90` backend (the launcher-mounted `flashinfer_mla_sparse_sm90.py.patched`,
md5 `210d7d7f…`, identical on nodeC/nodeA/nodeB) plans FlashInfer FA2 on the host every step with

    lens = ctx if ctx <= index_topk else index_topk + ctx % index_kpool        # _kv_lens_host, :371

The GLM5Next kpool indexer keeps `select_k - 1` = 511 pools (`pool_ids[:, : select_k - 1]`,
`sparse_attn_indexer_kpool.py` :581 prefill / :875 decode) plus the trailing incomplete pool, so a row with
`ctx >= 2048` has **2044 + ctx % 4** selected keys — 4 fewer than planned. FA2 reads `kv_indices[row * 2048 + j]` for every
planned `j`, so every such row also attends:

* `4 - ctx % 4` copies of KV slot 0 (`forward_mqa` clamps the -1 tail of the converted row to slot 0), and
* for `ctx % 4 != 0` the first `ctx % 4` keys of the NEXT row's (compacted) selection; the LAST row of a step reads
  whatever an earlier, larger call left in the process-wide `kv_indices` buffer (another request's slots when that
  call was another request's prefill chunk or a bigger batch).

Measured with the image's real ops (`tests/mla_exactlens/test_exactlens_unit.py` U2/U3,
`${HOME}/tf-exl3-assets/opt-decodekit/test_exactlens_unit.log`): `persistent_topk` / `top_k_per_row_prefill` →
the stock `[:, :511]` → `expand_pools_and_append_tail` → `triton_convert_req_index_to_global_index(return_valid_counts)`
gives exactly `ctx` below 2048 and `2044 + ctx % 4` from 2048 on, for decode and prefill rows; the short-prefill path
(every prefill request <= 2048 tokens, identity top-k) gives `ctx` (2048 at ctx == 2048). Production's plan is 4 keys
too long on **every** row with ctx >= 2048 (52 over-read keys on the self-test's 19 rows).

With `GLM53_MLA_PREFILL=1` (production) prefill runs the exact kernel on device-side valid counts, so only **decode**
attends the extra keys: decode and prefill attend different key sets.

### Hypothesis (REJECTED by measurement, section 3): the over-read explains the drop-lowest dvp regression

Production measured decode-vs-prefill KL 0.0075 -> 0.0109 with `GLM53_KPOOL_DROP_LOWEST=1` (2026-10-01). That patch
emits each row's pools in score order, so the next row's first pool (what the over-read duplicates) becomes the next
query's top pool. The handoff A/B (arms dloff/dlon) does NOT support this: drop-lowest raises the mini's dvp KL with
or without the exact lengths (0.0103 vs 0.0105). The cause of that regression is still open (candidate: the
reordered key order changes FA2's accumulation order; prefill and decode then round differently - untested).

## 2. The fix (`GLM53_MLA_EXACT_LENS=1`)

`glm53_mla_exactlens.py` replaces `FlashInferMLASparseSM90Builder.build` by the same three statements
(source fingerprint `819d7860e255a9f3` required, `_kv_lens_host` `281328403e853281` = the one GLM53_DEC_HOSTLOOP pins;
anything else -> not installed, WARNING, production path) with one change: the planned lengths go through
`exact_lens()` before `plan()`:

* `lens < index_topk` -> unchanged;
* `lens >= index_topk` -> `lens - index_kpool`, except rows of a prefill request in a short-prefill step (vLLM's own
  `split_decodes_and_prefills` with the indexer builder's arguments; the indexer's predicate: every prefill request's
  `seq_len <= index_topk`, from `seq_lens_cpu_upper_bound`, exact for prefill rows).

Never larger than production's lengths (only removes reads). Host-only, outside capture, no device sync added;
`_kv_lens_host` is called exactly as before, so the hostloop fast path is unaffected (logged: the hostloop line
`first step planned from the post-sampling snapshot` in the same runs). Cost: **10.8 µs per decode build** (1 or 8
requests), 63 µs for a mixed step with a 13,824-token chunk (`tests/mla_exactlens/bench_host.py`, CPU container,
`bench_host.log`), off the critical path under the hostloop.

Self-test: at the first `build` on the device (outside capture) the module reruns U3 (real ops at the model's
index_topk / index_kpool); a mismatch keeps production's lengths for the process (`self-test FAILED`, WARNING).

## 3. Measured on nodeC (handoff mini engine, the real production-image engine)

`tests/handoff/run.sh` (kit r16z2rev, production's decode features: MLA prefill, hostloop, kpool ring, quickwins,
smallops dconv; prompts 6000 + 3001 tokens, 96 greedy tokens each, KL(decode || fresh prefill) at 96 positions of
the 6000-token request; raw production top-k ops). Logs `${HOME}/tf-exl3-assets/opt-decodekit/exl_ab/<arm>/`.

| arm | config | KL(decode \|\| prefill) mean | p50 | p95 | max | top-1 agree | last decode row read past the step |
|---|---|---|---|---|---|---|---|
| off1 | production plan | 0.00566 | 0.00028 | 0.0701 | 0.102 | 94/96 | 142 of 190 steps (0 foreign slots) |
| on1 | GLM53_MLA_EXACT_LENS=1 | 0.00597 | 0.00025 | 0.0642 | 0.102 | 93/96 | 0 of 190 |
| dloff | + GLM53_KPOOL_DROP_LOWEST=1 | 0.01027 | 0.00030 | 0.0802 | 0.104 | 94/96 | 142 of 190 |
| dlon | + DROP_LOWEST + EXACT_LENS | 0.01051 | 0.00031 | 0.0903 | 0.103 | 91/96 | 0 of 190 |

Reading: the exact lengths remove every over-read (last column) and change nothing measurable in the consistency
KL. The mean is carried by 5-8 positions per arm with KL 0.05-0.10 (near-tie top tokens) whose POSITIONS move between
arms (per-position table in the transcript; e.g. pos 28/31/48 drop in on1, pos 38/50/65/85 rise): at n=96 on one
prompt the noise floor is about +-0.003, larger than any effect of 4 extra keys of ~2048. Verdict: a correctness
cleanup with no measured quality gain; ship only together with something else that needs a restart.

## 4. kpool "drops an arbitrary pool" (memory note 2026-09-30) — measured: it does not, practically

`tests/kpool_lastcol_probe.py` / `tests/kpool_topk_inplace_probe.py` (real ops, pristine host copies of the scores,
`${HOME}/tf-exl3-assets/opt-decodekit/{lastcol_probe,inplace_probe}.log`):

* `top_k_per_row_prefill` (prefill): the last column is the **lowest-scored** selected pool in every row of every
  case (600 … 65,536 pools; gaussian, recency-skewed and heavily tied scores) — the stock `[:, :511]` already drops the
  lowest pool (ties: one of the tied lowest).
* `persistent_topk` (decode): the last column is usually in the bottom 3 % (rank >= 495 of 512 in the in-place
  probe, 600 … 32,768 pools), but NOT always: `lastcol_probe.log` at 16,384 pools (64k context) saw rank 380
  (gaussian scores) and rank 275 (recency-skewed scores) in the last column; at 65,536 pools it was exactly the lowest.
  Neither op modifies its input.

So the memory note is right in kind (decode drops a non-lowest pool, at long context sometimes a mid-ranked one) but
the effect is one of 512 pools (4 of ~2048 keys). The production dvp regression of the drop-lowest patch is NOT
explained by the FA2 over-read either (section 3: dlon == dloff). Neither the drop-lowest patch nor the exact lengths
showed a measurable quality gain on the handoff mini; neither is recommended on its own.

## 5. Wiring (default off everywhere)

* `overlay/patch_mla_exactlens.py` (bundle overlay, `GLM53_MLA_EXACT_LENS` in `patch_tf_bundle.py` PATCHES): =1 copies
  `overlay/glm53_mla_exactlens.py` into site-packages and arms the site `integrate.py` (a marked block after the
  glm53_hostloop step); =0 removes/disarms (stock bytes); preflights integrate.py before writing anything.
  The kit's `site/` stays the previous kit's (the module is NOT a py_module; the repo-root integrate.py is unchanged).
* `tools/deploy16/make_start_sh.py --stage r16z2x` = r16z2 + XL1-XL4 (validation empty/0/1 + overlay files + bundle
  registration; head `-e`; worker `serve_env_names`; note) — `launcher/start.sh` (f5669ae8…) and
  `launcher/start.sh.exactlens.patch`; `make_kit.sh` builds `KIT_STAGE=r16z2x` by default.
  r16z2 (0962bae7) still regenerates byte for byte.
* `env.r16` `#switch GLM53_MLA_EXACT_LENS 0|1`; `tools/env_r16.sh on|off exactlens` (on refuses, .env unchanged,
  without the r16z2x start.sh / overlay files / bundle registration); `revert_r16.sh` removes the line (it reads every
  `#switch`); an older kit's revert leaves the line, inert under that kit's start.sh (it does not forward it).
* `boot_checks.sh`: head == worker env; =1 needs `patch_mla_exactlens.py: applied`, `integrate.py: arm`,
  `installed (pid`, `self-test passed` on BOTH ranks; never `NOT installed` / `install failed` / `self-test FAILED` /
  `exact_lens failed` / `not loaded` (a rank serving production's plan next to an exact one = BAD); unset/0 need the
  skip/stock line and never the installed lines.
* Tests: `tests/mla_exactlens/test_exactlens_unit.py` (GPU, real ops), `test_patch_mla_exactlens.py` (host),
  `test_start_stage.sh` (host), `consistency_kl.py` (summarizer), `bench_host.py`.

NOT done: `test_boot_checks.sh` / `test_kit_scripts.sh` / `kit_chain.sh` / `off_equals_prev.sh` /
`check_boot_strings.py` rows for the new switch (the boot_checks rows follow the kpooldown rows line for line); no
kit was built. The branch `r16z3` (GLM, in progress) will need the same XL edits on its stage.

## 6. Production A/B (proposed; owner decision)

Switch-only after the kit is applied: `tools/env_r16.sh on exactlens` + idle-gated restart of both ranks;
boot_checks; then the usual probes. Expected: decode-vs-prefill KL down or unchanged (stock top-k order), long-context
KL unchanged (prefill is not touched with GLM53_MLA_PREFILL=1), decode ms/step unchanged (4 fewer keys of ~2048 per
row; +11 µs host per step off the critical path). Back off: `tools/env_r16.sh off exactlens` + restart.
