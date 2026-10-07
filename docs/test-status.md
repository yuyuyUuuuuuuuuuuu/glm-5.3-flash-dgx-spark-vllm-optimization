# Test status

What `code/fork/tests/run_all.sh` gives, step by step, and which non-PASS results a reader should expect on their own
box. Procedure and inputs: `REPRODUCE.md` section 13.

**Runs behind this table** (2026-10-07, one GB10 node, the serving image by digest, `SKIP_BUILD=1 R16=on`):

- **this tree**: a clean copy of this repository with all 12 extensions freshly built by `code/build/build_all.sh`;
  every input of section 13 present. Result: `SUMMARY: 53 PASS, 0 FAIL, 0 SKIP`.
- **393c2a5**: the original fork at commit `393c2a5` with production's `.so` files
  (`code/kit/PRODUCTION_BINARIES.sha256`), whose `run_all.sh` had no precondition-clash exclusions yet. Result: 49 PASS,
  4 FAIL (the four marked `strict`, which are the four `R16_STRICT=1` failures of this tree, failing on the same
  assertions).
- **no inputs**: this tree with `TF_EXL3_MODELS` and `TF_EXL3_ASSETS` pointing at empty directories, for the
  asset-dependent steps plus `test_fp8_large_m` and `check_paths`: every asset-dependent step printed
  `--- <step>: SKIP (needs ...)` and was counted apart (`SUMMARY: 3 PASS, 0 FAIL, 11 SKIP`).

Columns: *this tree* and *393c2a5* are our results; *your box* is what to expect after `tests/derive_assets.sh` (all
derivable inputs) but without the inputs that are not published or not derivable (section 11 / 13). A SKIP is not a
PASS; it means the step did not run.

| # | step | this tree | 393c2a5 | your box | note |
|---|---|---|---|---|---|
| 1 | `verify` | PASS | PASS | PASS |  |
| 2 | `prod_module` | PASS | PASS | PASS |  |
| 3 | `test_layout_equiv` | PASS | PASS | PASS |  |
| 4 | `test_units` | PASS | PASS | PASS |  |
| 5 | `test_stage_vs_xl` | PASS | PASS | PASS |  |
| 6 | `test_e2e_vs_xl` | PASS | PASS | PASS |  |
| 7 | `test_e2e_vs_f64` | PASS | PASS | PASS |  |
| 8 | `test_graph` | PASS | PASS | PASS |  |
| 9 | `test_integrate` | PASS | PASS | PASS |  |
| 10 | `test_apply_fused` | PASS | PASS | PASS |  |
| 11 | `test_prefill_cap` | PASS | PASS | PASS |  |
| 12 | `test_variants` | PASS | PASS | PASS |  |
| 13 | `test_fp8_gemv` | PASS | PASS | PASS |  |
| 14 | `test_fp8_integrate` | PASS | PASS | PASS |  |
| 15 | `test_fp8_large_m` | PASS | PASS | PASS, 160 of 285 checks without the TR3 samples (not published) |  |
| 16 | `test_fp8_large_m_integrate` | PASS | PASS | PASS |  |
| 17 | `test_bf16_gemv` | PASS | PASS | PASS |  |
| 18 | `test_gemv_install` | PASS | PASS | SKIP until `derive_assets.sh` (NVFP4 + DFlash2) |  |
| 19 | `test_glm53_runtime` | PASS | PASS | PASS |  |
| 20 | `native_kernel` | PASS | PASS | PASS |  |
| 21 | `bench_ab_decode` | PASS | PASS | PASS |  |
| 22 | `bench_fp8_gemv` | PASS | PASS | PASS |  |
| 23 | `bench_fp8_large_m` | PASS | PASS | PASS |  |
| 24 | `test_fp8_roof` | PASS | PASS | SKIP until `derive_assets.sh` (DFlash2) |  |
| 25 | `test_moeglue` | PASS | PASS | PASS |  |
| 26 | `test_moeglue_warm` | PASS | PASS | PASS |  |
| 27 | `rv_moeglue_adversarial` | PASS | PASS | PASS |  |
| 28 | `test_smallops_kernels` | PASS | PASS | SKIP until `derive_assets.sh` (NVFP4 + DFlash2) |  |
| 29 | `test_smallops_install` | PASS | FAIL (strict) | SKIP until `derive_assets.sh` (NVFP4 + DFlash2) | run with `GLM53_DEC_SMALLOPS_KINDS` left out (precondition clash); `R16_STRICT=1`: FAIL by design, same assertion as 393c2a5 |
| 30 | `test_smallops_tc` | PASS | PASS | SKIP until `derive_assets.sh` (NVFP4) |  |
| 31 | `review_smallops_adversarial` | PASS | FAIL (strict) | SKIP until `derive_assets.sh` (NVFP4 + DFlash2) | run with `GLM53_DEC_SMALLOPS_KINDS` left out (precondition clash); `R16_STRICT=1`: FAIL by design, same assertion as 393c2a5 |
| 32 | `test_kpool_ring_gpu` | PASS | PASS | PASS |  |
| 33 | `r16_decode_combo` | PASS | PASS | PASS |  |
| 34 | `r16_breakable` | PASS | PASS | PASS |  |
| 35 | `test_quickwins` | PASS | FAIL (strict) | PASS after `derive_assets.sh fi618` | run with `GLM53_MLA_PREFILL` left out (precondition clash); `R16_STRICT=1`: FAIL by design, same assertion as 393c2a5 |
| 36 | `test_mla_prefill` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 37 | `test_mla_integrate` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 38 | `review_mla_adv` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 39 | `review_mla_adv2` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 40 | `review_mla_plugin` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 41 | `bench_mla_prefill` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 42 | `test_hostloop` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 43 | `test_hostloop_plugin` | PASS | FAIL (strict) | PASS after `derive_assets.sh fi618` | run with `GLM53_DEC_MOEGLUE_WARM` left out (precondition clash); `R16_STRICT=1`: FAIL by design, same assertion as 393c2a5 |
| 44 | `test_hostloop_wake` | PASS | PASS | PASS after `derive_assets.sh fi618` | timing test; 1 FAIL in 9 runs over both trees, see below |
| 45 | `test_prof_diag` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 46 | `review_hostloop_async` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 47 | `r16_prefill_combo` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 48 | `r16_plugins` | PASS | PASS | SKIP: deploy-r15 bundle (not published) |  |
| 49 | `r16_loo_census` | PASS | PASS | PASS after `derive_assets.sh fi618` |  |
| 50 | `handoff` | PASS | PASS | PASS after `derive_assets.sh fi618` + `derive_assets.sh mini launcher models` | one earlier run was cut by an outside SIGTERM (not a test result), see below |
| 51 | `test_patch_kpool_tail_ring` | PASS | PASS | SKIP until `derive_assets.sh` (vllm-src) |  |
| 52 | `test_consistency_probe_mock` | PASS | PASS | PASS |  |
| 53 | `check_paths` | PASS | new (replaces `check_docs`: PASS) | PASS |  |

**Totals on your box.** With every `derive_assets.sh` step run (including `fi618`, which copies production's exact
flashinfer 0.6.18 overlay out of a public image): 52 PASS, 1 SKIP (`r16_plugins`, which needs the unpublished deploy-r15
bundle). Without the `fi618` step: 37 PASS, 16 SKIP (the 15 section-C steps and `handoff`). Without `derive_assets.sh`:
30 PASS, 23 SKIP.
A FAIL on your box is not expected from any step; if `test_hostloop_wake` fails, read the note below before anything
else.

## Non-PASS results we saw, and their class

Classes: **(a)** environmental or pre-existing: the same on `393c2a5` / production's `.so` files, or caused by the
host; **(b)** packing defect: only in this repository's copy (fixed); **(c)** reproducibility: a fresh build differs
from production's binaries.

| what | class | cause |
|---|---|---|
| `test_hostloop_wake` FAIL `W3: wake did not reduce the latency` (1 of 9 runs: 5 on this tree, 4 on 393c2a5) | (a) | W3 compares the median cold graph-host-node latency with and without the wake tickers; in the failing run the CPU was not idling deeply (median without wake 92 us, against 213-339 us in the 8 passing runs), so there was nothing for wake to remove (128 us). Host timing, not code: rerun it on an idle box |
| `handoff` stopped with `Terminated` at the start of its third run (`mla_defect`) | not a test result | a SIGTERM from outside, 8.6 min into the step, most likely the end of the session that had started the suite in the foreground (`tests/handoff/run.sh`'s own limit is `GPU_TIMEOUT`, 5400 s). Started detached (`setsid nohup`), all three runs passed in 13.4 min |
| `test_smallops_install`, `review_smallops_adversarial`, `test_quickwins`, `test_hostloop_plugin` FAIL under `R16_STRICT=1` | (a), by design | each asserts a precondition that one ship flag overrides on purpose (section 13); identical assertions fail on 393c2a5. The default `R16=on` runs each with that one knob left out |
| `code/fork/docs/ref/prod_live/inner_start.masked.sh` failed `bash -n` | (b), fixed | the secret masker had taken five non-secret `${VAR:-default}` expansions (`MAX_NUM_BATCHED_TOKENS`, `LONG_PREFILL_TOKEN_THRESHOLD`, `MTP_TOKENS`, `MM_IMAGE_TOKENS` twice) for token values; restored from the launcher's `start.sh`. Reference copy only, no test reads it |
| `REPRODUCE.md` said `test_fp8_large_m` keeps its drafter checks without the TR3 samples | (b), fixed | the real drafter-fc check is part of the real-sample set; without the samples it is 160 of 285 checks whether the drafter is mounted or not |
| fresh `.so` bytes != `code/kit/PRODUCTION_BINARIES.sha256` | (c), metadata only | 9 extensions differ in 1-3 bytes, all inside nvcc's temporary file name `tmpxft_<pid>_...`; `glm53_mhc_fused_ext` (25 bytes) and `glm53_moe_e4m3_ext` (26) also in the build-id and recorded source mtime; `_flashkda_fp32_C` (443,114 bytes) in the build-id, nvcc's per-file anonymous-namespace hash and its compressed fatbin. For those three, `.text` is byte-identical and `cuobjdump -sass` is identical line for line. Test outcomes are the same step for step |

## What this table does not cover

- Two-node behaviour (TP=2, NCCL) and the serving benchmarks: section 10, not `run_all.sh`.
- `fi618/` built from flashinfer source at `61a6c651` instead of the copied binaries: not checked.
- The one-off scripts under `code/fork/tests/` that `run_all.sh` does not call (review and bench scripts of single
  branches); several of them compare against deploy kits or branch worktrees under `TF_EXL3_KITS` that are not
  published.
