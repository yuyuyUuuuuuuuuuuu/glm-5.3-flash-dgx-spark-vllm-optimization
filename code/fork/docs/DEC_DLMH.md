# DEC_DLMH — the drafter's candidate head without reading the whole FP8 lm_head (`GLM53_DEC_DLMH`)

Branch `decode5-dlmh` (from `decode4-kdalazyfix` c61e3e1), 2026-10-05. nodeC only (one GB10, production image, every GPU
job under `flock /tmp/tf-gpu-bench.lock` through `tests/gpu_run.sh` / `tests/handoff/run.sh`); production read-only
(container list, log grep, start.sh sha256). Logs: `${HOME}/tf-exl3-assets/decode5/logs/` and
`.../decode5/engine/` (copied into `docs/logs/dlmh/`).

## 0. Result in one paragraph

Every decode step the DFlash2 drafter runs the full 154880-row FP8 lm_head (317 MB per TP rank, 1.34 ms in the R15
trace, plus an 86 µs all-gather of the bf16 logits) only to take the top-16 candidates of its 7 query rows.
`GLM53_DEC_DLMH=1` ranks the vocabulary with a 4-bit coarse copy (164 MB per rank), recomputes the **exact** FP8 logits
of the coarse top-128 column octets per row with production's own fp8_gemv arithmetic (read in place), all-gathers
7 x 2048 int32 instead of 7 x 77440 bf16 and runs production's `torch.topk` on a -inf tensor holding only those exact
values. Candidates and unary logits are byte-identical to production whenever the coarse top-C contains every element
>= the 16th exact value (0 / 6272 rows differ on the real drafter's hidden states; 1 / 3052 rows in a real-engine run
of the near-uniform mini drafter, an exact tie at the 16th value). nodeC, per call: **1414 -> 920 µs at 1 request
(-495 µs), 1437 -> 1069 µs at 2 requests (-369 µs)**, plus ~60 µs less all-gather in production (TP=2): projected
**-0.55 ms/step on all four workloads at batch 1** (the drafter runs once per step whatever K is), -0.43 ms at 2
requests. A miss can only change the drafter's proposal; the target distribution, the acceptance rule and the outputs
are untouched (speculative decoding stays lossless).

## 1. Where the step goes now (nodeC, fresh, rows M = 3 / 5 / 6 / 8)

Bandwidth-bound parts measured in isolation on nodeC (cold weights, CUDA graphs; `logs/breakdown_*.log`):

| component (per step, one rank) | M=3 | M=5 | M=6 | M=8 | source |
|---|---|---|---|---|---|
| MoE layer x42 (router + routed experts corr40 + shared expert on aux + window stand-in) | 24.7 | 33.2 | 35.9 | 40.4 | `tests/dlmh/moe_rows_bench.py`: 588 / 791 / 855 / 962 µs per layer |
| target FP8 GEMVs (KDA in/o/f_b/g_b x34, MLA x11, dense x3, lm_head) | 14.6 | 14.7 | 14.7 | 14.8 | `tests/bench_fp8_gemv.py --m 3,5,6,7,8` (fp8_gemv TABLE configs, 240-243 GB/s) |
| drafter (5 layers FP8 + fc + lm_head at 7 rows + bf16 kernel_projection + ctx KV) | 4.3 | 4.3 | 4.3 | 4.3 | same bench (drafter shapes) + R15 trace for the bf16 parts |
| sum of the bandwidth-bound parts | 43.6 | 52.2 | 54.9 | 59.5 | |
| production ms/step (bench_round, KDA_LAZY v2) | – | prose 72.1 (mean M ~5.8), ja 67.7 | coding 82.0 | structured 80.0 | operator, 2026-10-04 |

* The MoE grows 2.3-2.7 ms per verified row at M 5-8 on nodeC (4.3 ms/row at 3->5). Production's own slope is
  4.5-5.3 ms per K (`SPEC_VTRIM.md` 1): about 2 ms/row more than nodeC's corr40 model - real routing touches more
  distinct experts per added row than corr40. Fewer rows (decode4 track B, `GLM53_SPEC_VTRIM`) remains the only lever
  on that part.
* Everything else in production's step (~18-20 ms at M~5.8) is latency-bound: 102 all-reduces over RoCE
  (~22 µs floor + rank skew each), ~2100 kernels with ~1500 under 15 µs (mHC, router, rot_in, epilogues, KDA
  recurrent, MLA indexer chain), host gaps. The 09-29 production A/B of the L2 prefetch (`r16-rollout` memory note)
  showed these windows are not idle: every byte read during them slowed the all-reduce (p50 31 -> 50 µs), mHC and the
  router as much as it saved. **Only removing bytes transfers to production reliably** — which is what this branch
  does in the one place where whole-weight reads are avoidable without changing a single target value.

## 2. Mechanism (`glm53_dlmh.py`, `kernels/dlmh_gemv.cu`)

Per drafter call (T = 7 x requests rows, inside the drafter's FULL CUDA graph), per TP rank (vocab shard 77440 rows):

1. **coarse** (`_coarse_oct_kernel`, Triton): int4 codes of the rank's FP8 lm_head (e4m3 x per-channel scale, rounded
   to symmetric int4 in groups of 128 along K, fp16 group scales; 158.6 + 5.0 MB, built once from the Marlin FP8
   weight at the drafter's first eager call), x [T, 4096] -> per row the max coarse logit of every column **octet**
   {64 t + r + 8 i : i = 0..7}: in the Marlin FP8 layout these 8 columns are one contiguous 128-byte line per k-tile.
   696 µs (235 GB/s).
2. `torch.topk(octet scores, C = 128)` per row (38 µs).
3. **exact** (`dlmh_gemv_kernel`): fp8_gemv_kernel copied verbatim (`tests/dlmh/check_kernel_copy.py`) except that a
   virtual 64-column n-tile reads its 8 octets in place from the original weight and each output column takes its
   source column's scale. Same KW (k-tiles kw, kw + KW, ...), same mma per 16-k tile, same fixed-order split-K sum,
   scale and bf16 rounding -> every value is bit-identical to the full head's at that vocabulary id (U3b).
   7 x 128 octets x 32 KB = 29 MB, ~150 µs (random 128-B lines: 196 GB/s).
4. `_post_kernel`: (global id, bf16 bits) of the row's own 1024 columns -> int32 [T, 2048]; all-gather over TP
   (7 KB per rank instead of 1.08 MB).
5. production's `torch.topk` on a [T, 154880] bf16 tensor that is -inf except at the gathered ids.

Why the result is production's: torch.topk's selection depends only on the values and indices of the elements >= its
k-th value (radix select; among equal values the lower index; then a sort of the k selected) — U4 tests it on
tie-heavy bf16 tensors (ties at the 16th value are frequent: 28-43 % of real drafter rows have more than 16 elements
>= the 16th value). So the result is identical iff every such element is among the rescored columns of its rank.
The coarse top-C octets contain every column whose coarse logit ranks within the top C of its rank; the measured worst
need was 61 (`tests/dlmh/recall_probe.py`, column-level, int4 g128) and 0 / 6272 rows differ at C = 96, 128, 192
(6 / 6272 at C = 64; U5).

Served calls: T a multiple of 7 with T <= 16 (1-2 requests; production's own lm_head call runs fp8_gemv there —
`GLM53_FP8_GEMV_MAX_M=16` — and the coarse kernel's BLOCK_M is 16). Any other row count, a first call during
capture, a non-FP8 head, a soft cap / scale on the candidate logits, a vocab layout other than vocab-parallel
all-gather -> production's `compute_candidates` unchanged for that call (or the process).

## 3. Exactness evidence (nodeC)

`tests/dlmh/test_dlmh.py` (real lm_head, production-converted FP8 shards as `glm53_runtime.convert_lm_head_fp8` builds
them, real DFlash2 drafter hidden states; `logs/test_dlmh_r5.log`): **ALL OK**

| test | result |
|---|---|
| U1 Marlin FP8 unpack == e4m3 x scale | 0 / 317,194,240 elements differ |
| U2 / U2b coarse kernel vs torch reference; fused octet max | rel_l2 1.5e-7; octet max identical |
| U3 / U3b gathered / in-place octet fp8_gemv == the full head's columns (T = 1, 5, 7, 8, 14, 16; 8..1024 octets with duplicates; production configs and the 4-warp rescoring blocks) | 0 mismatches (bitwise) |
| U4 masked torch.topk == full torch.topk on tie-heavy bf16 tensors | 0 / 300 |
| U5 two-stage == production candidates + unary logits, TP=2 emulated (two 77440-row shards, rank-major all-gather), 7 sets x 448 drafter rows (near-uniform L0 to confident L0.9, random directions), T = 7 and 14 | C = 64: 6 / 6272 rows differ; **C = 96 / 128 / 192: 0 / 6272** |
| U6 inside a CUDA graph, replayed with new inputs | 0 / 12 replays differ |
| U7 the installed DFlash2Qwen3ForCausalLM.compute_candidates (setup + self-test + on + verify) vs the stock method | identical; verify counters 0 mismatched rows; T = 21 -> production path |
| U8 (`tests/dlmh/test_dlmh_tp2.py`) two TP ranks = two processes (collectives over gloo: NCCL refuses two ranks on one GPU), real 77440-row shards, the stock head all-gathering bf16 logits: A both ranks serve, hooked == stock (peaked / random / mixed rows, T = 7, 14; T = 21 production path); B rank 1's coarse build raises -> the setup agreement turns the head off on BOTH ranks, every call == stock, no hang | A OK (served 6 / 6 on both ranks), B OK (`logs/test_dlmh_tp2.log`) |

Real engine (`tests/handoff/run.sh`, production image + launcher overlay chain, DFlash2 drafter, EXL3-MoE mini target,
block verification, KDA_LAZY on, temperature 1.0, 4 requests in 2 batches, 128 tokens; the mini's lm_head is given
the 77440-row fp8_gemv config in every arm, `HANDOFF_FP8_FULLVOCAB=1`): boot logs the coarse build and the
byte-equality self-test; `GLM53_DEC_DLMH=verify` counted **1 differing row of 3052**; the built-in diagnosis
(`engine/runs/verify/container.log`): the 16th value 1.2422 was shared by 4 vocabulary ids (the mini drafter is
near-uniform: its top logit 1.52), one of them had coarse octet rank 164 > C. The arms off / on / off2 (outputs vs the
A/A band) are in section 6.

The enable decision is TP-wide: each rank's setup result and then its self-test result are all-reduced before any
call is served (a rank that cannot build the coarse copy or fails the self-test turns the head off on every rank:
the two-stage head all-gathers a different tensor than production's, so a split decision would mismatch the
collective).

## 4. Speed (nodeC)

`tests/dlmh/bench_step.py` (one rank's part, real lm_head shard and drafter rows, CUDA graphs, 15 interleaved rounds;
`logs/bench_step_r3.log`):

| rows | production (fp8_gemv lm_head + topk) | two-stage C = 128 | saving | two-stage C = 96 |
|---|---|---|---|---|
| T = 7 (1 request) | 1414.5 µs | 919.9 µs | **-494.6 µs** | 918.0 µs |
| T = 14 (2 requests) | 1437.3 µs | 1068.8 µs | **-368.5 µs** | 1044.4 µs |

Not in this number (TP=1 node): the all-gather, [T, 77440] bf16 (1.08 MB, 86 µs in R15) -> [T, 2048] int32 (7 KB,
at the ~22 µs latency floor). Projection for production at batch 1: **-0.55 ms/step on structured, prose, coding and
ja alike** (prose 72.1 -> ~71.5); at 2 requests -0.43 ms. Kernel breakdown (`logs/prof`): coarse 696-740, rescoring
149, octet topk 38-43, final topk ~65 (production pays the same), the rest < 30 µs.

Memory per rank: +163.6 MiB (coarse copy) + small buffers; the setup line prints it (166.5 MiB at TP=2). Production
sizes its KV pool with kv_cache_memory_bytes, so this comes out of the host headroom (nodeB: 3 GB available on
2026-10-05), not out of KV.

## 5. Knobs, boot strings, revert

`GLM53_DEC_DLMH` = unset/0 (production) | 1 | verify; `_C` (octets per row, multiple of 8 in 16..256, default 128);
`_GROUP` (32 | 64 | 128, default 128); `_LOG` (stats period in drafter steps, default 2000, first after 64).
`verify` runs both heads (production's served) and costs ~+0.9 ms/step: a measurement mode, not a speed mode.

Boot (both ranks): `glm53_dlmh plugin loaded (pid N): GLM53_DEC_DLMH='1' -> installing (mode on, C 128, g 128)`,
`glm53_dlmh: hooked DFlash2 compute_candidates`, `glm53_dlmh: rank R/2 coarse candidate head built (...)`,
`glm53_dlmh: rank R self-test: candidates and unary logits byte-equal to production's (T in [7, 14]) PROOF mode=on`.
Never: `glm53_dlmh: setup failed`, `self-test: two-stage candidates differ`, `glm53_dlmh: not wired`. Stats:
`[glm53-dlmh] rank R mode on C 128 g 128: steps N, served graph-calls ... (counters as of step M)` (M = N - 32) and once
per rank `[glm53-dlmh] rank R serving confirmed (mode on)` when served graph-calls > 0 (the A/B PROOF string). Off: `-> off, production's candidate head
unchanged`. Revert: `tools/env_r16.sh off dlmh` + restart (the module is inert without the variable).

## 6. Engine A/B (mini rig)

Same rig and config as section 3 (temperature 1.0, block verification, KDA_LAZY on; the rig is not deterministic across
processes, so the A/A pair off / off2 sets the band). `logs/engine_compare.txt`, first differing token per request
(4 requests, 128 tokens):

| pair | req 0 | req 1 | req 2 | req 3 |
|---|---|---|---|---|
| off vs off2 (A/A) | 19 | 3 | identical | 122 |
| off vs on (GLM53_DEC_DLMH=1) | 19 | 5 | identical | identical |
| off vs verify (production candidates served) | 60 | 5 | identical | identical |

`on` served the two-stage head in 303 of the 350 drafter steps the stats line covers (the rest: the boot's eager
calls and batches outside T in {7, 14}); its divergences are inside the A/A band; max |dlogprob| before the first
difference 0.18 (A/A 0.22).

## 7. Not done / risks

* TP=2 itself is not runnable on nodeC: the all-gather of the packs is emulated (rank-major concatenation, U5/U6);
  the real collective is production's `tensor_model_parallel_all_gather`, the same op the stock head uses.
* Recall is measured on real drafter weights with optimized (probe) context features and on a mini-engine run, not on
  production traffic: run `GLM53_DEC_DLMH=verify` for an hour in production to count differing rows before `1`, or
  accept that a miss only perturbs one of 16 candidates (proposal-only).
* Row counts 21-64 (3-8 concurrent requests) stay on production's path (production's own head runs Marlin there).

## 8. Measured but not built (proposal-changing, needs a production acceptance A/B)

* Drafter MLP at W4A16 (Marlin GPTQ uint4b8, g128) instead of FP8 (`tests/dlmh/bench_w4_drafter.py`,
  `logs/bench_w4_drafter.log`, 8 rows, cold): gate_up 215.1 -> 113.3 µs, down 106.7 -> 58.2 µs = **-0.75 ms/step**
  (5 layers), on every workload. Changes the drafter's numerics (the precedent `GLM53_DRAFT_FP8` was decided by a
  production acceptance A/B, offline agreement proxies were misleading); not built here.
