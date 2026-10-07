# OPT_DECODE — where production's decode step goes now, what is left to remove, what this branch does

Branch `opt-decode` (from `r16z2rev` 60b53bf), 2026-10-03, nodeC only; production read only (non-secret `.env` knobs,
`SpecDecoding metrics` / adaptive-K log lines). Logs in `docs/logs/opt_decode/`.

## 1. Production's decode today (r16z2 + FLASHKDA, MHC_SP, MOE_E4M3, W8A8 subset: prefill features)

Decode-relevant knobs read from nodeA's `.env`: DFlash2 + `GLM53_ADAPTIVE_K_SET=4,5,7`, `GLM53_REJECTION_METHOD=block`,
`GLM53_DRAFT_FP8=layers,fc`, `GLM53_DEC_HOSTLOOP=1` + `_WAKE=auto`, `GLM53_DEC_SMALLOPS_KINDS=dconv`,
`GLM53_KPOOL_RING=1`, `GLM53_KDA_STRIDED_QKV=1`, FP8 GEMV / BF16 GEMV / dedup router, TF EXL3 MoE.
**Not** on: `GLM53_DEC_FP8ROOF`, `GLM53_DEC_MOEGLUE_WARM` (L2 prefetch: no gain in production, 09-29).

Production's last 24 h (`prod_spec_metrics_24h_summary.txt`, all traffic): 187,862 verify steps, **3.61 tokens per
step** (2.61 accepted drafts + 1), 4.84 drafts per step; adaptive K chose 4 / 5 / 7 in 50 / 30 / 20 % of steps;
P(accepted >= i) = 0.80 / 0.62 / 0.47 / 0.36 / 0.19 / 0.09 / 0.08 (i = 1..7, positions 5-7 exist only at K >= 5 / 7).
Step time from the owner's A/Bs: structured (K = 7, M = 8) 82.2-82.7 ms, prose ~72.5 ms.

## 2. Anatomy

The newest kernel-level profile is R15's (`decode-prof/p6h.json.gz`, rank 0, prose, all 39 steps at M = 5, median
69.1 ms, `tests/optdec`-independent re-analysis with `${HOME}/tf-exl3-assets/opt-decode/moe_vs_m.py`):

| category (ms / step) | R15 M=5 | what changed since | removable? |
|---|---|---|---|
| routed MoE (TF grouped, 84 calls) | 29.75 | e4m3 / fold / tokgather are prefill-only | only fewer bytes: bytes = unique experts of the M rows (~26 us per expert per layer). Per verified row: TF kernel with corr40 routing (which reproduces production's in-situ 706-708 us per layer at M = 5) 706 -> 884 us per layer for M 5 -> 8 = **~2.5 ms per row per step**; the 0922 production K sweep (old kernels, every per-row cost) gave ~8 ms per row -> **verify width** is the lever (section 4) |
| FP8 GEMV (dense, KDA, MLA, shared, drafter, 2 lm_heads) | 23.90 | – | at 205-246 GB/s; isolated already at the roofline (DEC_FP8ROOF R.1); L2 prefetch failed in production (NIC shares DRAM) |
| NCCL (102 AR + 2 AG) | 4.44 | WAKE (-0.9 ms) | ~half is rank skew; count is structural (2 per layer) |
| other small kernels (mHC, rot_in, epilogues, conv, top-k, ...) | 7.07 | dconv | many 2-25 us kernels; prior rounds took the cheap ones |
| cutlass bf16 (MLA W_UK/W_UV, indexer wq_b, drafter ctx) | 2.11 | – | <= ~0.2 ms (BF16_GEMV.md: cuBLAS already 206-228 GB/s for most) |
| bf16 GEMV (router etc.) | 1.34 | dedup | small |
| KDA recurrent kernel | 1.06 | strided qkv (-4 copies/layer) | **+ hidden write-back of the per-row states: 1.4-2.4 ms (measured, section 3)** |
| drafter fc (Marlin) | 0.40 | – | – |
| host gap drafter -> target | ~2 | hostloop + WAKE removed most | – |

New measurement on nodeC (`probe_kda_state_writes.log`): the per-row fp32 state stores of the KDA verify (2 MiB per row
per layer, 34 layers) cost **1.57 / 2.42 / 2.44 ms per step at M = 5 / 6 / 8** in a bandwidth-bound step — not visible
in the kernel's own 30 us, paid by the GEMVs that run next.

## 3. Implemented: `GLM53_DEC_KDA_LAZY` (docs/DEC_KDA_LAZY.md)

One state store per layer per step instead of M: the verify saves the rows' raw inputs, a single eager commit after
sampling recomputes the accepted prefix with production's arithmetic and stores the one state the next step reads
(+ the align prefix-cache column). Bitwise: outputs and every committed state == production's kernel (8 unit tests,
real kernels; real engine 620 / 655 committed states byte-equal, 0 mismatches), with a built-in self-check and repair
fallback. **Paired nodeC step benches: -1.39 (M=5), -1.54 (M=6), -2.28 (M=8), -2.30 (M=5 x 2 requests), -4.44
(M=8 x 2) ms per step**, ~-1.6 ms per step single-stream at production's K mix (~2 %), more with concurrency.
Default off; memory +51.7 MiB per rank.

## 4. Analysed, not implemented: per-step verify trimming (the biggest lever left; acceptance-dependent)

Why: a verified row costs mostly its routed experts (8 picks, partly shared with the other rows of the request); the
dense GEMVs do not grow with M; with DEC_KDA_LAZY the KDA state stores do not grow with M either. Today 4.84 drafts
are verified per step for 2.61 accepted. An oracle that verifies exactly the accepted prefix would save ~2.2 rows x
2.5-8 ms = 5.5-18 ms of ~75 ms (an upper bound of +8-30 % tokens/s, the range being the per-row cost estimate:
corr40 model vs the 0922 production sweep); a confidence policy gets a fraction of that.

Mechanism (no new CUDA graphs, no host sync, lossless):
1. after the DFlash2 walk, a GPU rule picks per request n* <= K;
2. rows n*+1..K are dead: in the TF MoE route kernel their expert ids become -1 (the sentinel group — no expert bytes
   read; `route_ids_kernel` already maps ids < 0 to the sentinel), and the lazy KDA verify skips their scratch;
3. the rejection sampler sees drafts -1 for them (only in the `rejection_sample` call, not in `apply_sampling_params`):
   vLLM's placeholder semantics are exactly "a proposal of length n*" in standard AND block verification (block: the
   row before a placeholder uses h = P, the residual after it is the target distribution — the stop-symbol drafter
   argument: Sun et al.'s rule with M_s = delta_stop gives h = P_n and residual = target);
4. live rows never read dead rows (causal attention / recurrence / per-token MoE), dead rows behave like rejected rows.

Exactness condition (easy to get wrong): the decision to verify row i may depend on d_1..d_{i-1} and on the drafter's
*distribution* q_i, **not on the sampled d_i itself** (cutting row i because q_i(d_i) is small biases the proposal at
row i and breaks losslessness). "Stop after a low-confidence token" is allowed (q_i(d_i) decides rows > i).

What is missing: the calibration between DFlash2's confidences (realized selector scores) and production acceptance
at temperature 1.0 — not measurable on nodeC (no full target), not in the logs (aggregates only). **This branch adds
the collector**: `GLM53_DEC_VTRIM_STATS=<N>` (`glm53_vtrim_stats.py`, default off, outputs untouched) wraps
`RejectionSampler.__call__` and records per request and verify step 16 floats — n, a, qmax_1..7, qd_1..7 (probabilities
and counts only, no token ids) — into a GPU ring written every N verify calls to
`/root/.cache/vllm/glm53_vtrim_stats.npy` (the bind-mounted cache dir), plus one INFO line; cost while on 0.35-0.46 ms
per verify step (`bench_vtrim_stats.log`).
`tests/optdec/vtrim_policy_from_stats.py records.npy` then evaluates the exact-safe rule family (survival product of
calibrated P(accept | qmax_i, qd_{i-1}), fit on one half, measured on the other) against the oracle for per-row costs
2.5 / 5 / 8 ms. Verified: `test_vtrim_stats.log` (records == CPU reference, 200 random batches), the engine run
`handoff_stats_t1.log` (temperature 1.0, 276 records) and `vtrim_policy_rig_pipeline_test.txt` (the pipeline; the mini
model's numbers mean nothing: synthetic acceptance does not depend on q, so no rule gains there).
Next step (owner's go, one restart): run production a few hours with `GLM53_DEC_VTRIM_STATS=2000` (the kit must forward
the three `GLM53_DEC_VTRIM_STATS*` names like any knob), copy the .npy, run the evaluator; build the trimming rule
only if it shows a gain worth it. Adaptive K's EMA would see the trimmed acceptance (feed it the pre-trim view or fix
K = 7 and let the rule decide).

## 5. Negative results / rejected (this round)

| idea | result |
|---|---|
| commit with the state tables as runtime int64 addresses + runtime slot stride | 1-ulp differences vs production (different register layout of the state tile); fixed with anchor + 16-aligned offsets + constexpr stride (bitwise) |
| commit by re-launching production's own kernel per layer | exact, but 34 launches per step (~0.7-1 ms latency) — kept only as the self-check / repair path |
| store the state at a guessed row in the verify, commit only on a wrong guess | worse expected traffic (2 MiB always + 4 MiB x P(wrong ~0.65)) than the commit (4 MiB) |
| fold the commit into the next verify (no re-read) | ~0.3 ms more per step, but the state is not materialized between steps (align copies, non-spec paths read it) — deferred |
| cross-process bitwise comparison of engine runs | impossible: off vs off differs (logprobs up to 0.15, tokens in 2 of 3 requests); the in-process self-check is the oracle |
| KDA in_proj "99 blocks on 48 SMs" wave quantization | not a lever: 246 GB/s isolated (DEC_FP8ROOF R.1) |
| drafter at 4 bit / drafter vocab pruning / distributed top-16 | lossless for the target but acceptance-dependent; 0.1-1.8 ms; needs production A/B — not started |
| top-p for < 8 rows on the sort path (+0.45 ms/step, BLOCK_VERIFY §2) | an exact replacement must reproduce the torch scan's rounding at the boundary; not bitwise -> not done |
