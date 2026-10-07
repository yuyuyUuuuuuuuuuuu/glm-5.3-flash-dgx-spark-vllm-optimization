# SPEC_VTRIM - speculation policy for low-acceptance text (decode4 track B)

Branch `decode4-b-spec` (from `r16z6rev` de1a3d3), 2026-10-04. Production read-only (light probes under
`flock ${HOME}/tf-exl3-assets-prodbench.lock`, concurrency 1, idle-checked), GPU work on nodeC only.
Everything below is reproducible from the files in `docs/logs/spec_vtrim/` (`tests/vtrim/policy/analyze.py` prints
sections 1-4 from the saved production traces without GPU or production access).

## 0. Result in one paragraph

The CPU-side speculation policy (which K in {4,5,7} adaptive-K verifies) is already at its optimum: no K set, EMA
alpha/margin, K floor (2/3), K=6 graph, or removing the two-step decision lag moves any of the four workloads by more
than ~2-3 % in an exact position replay of production's own acceptance (section 3). The one lever with a large ceiling is
**per-step** verify length: an oracle that knew each block's acceptance length and skipped the routed experts of the
rows it does not need would gain prose +19 %, coding +13 %, ja +24 %, structured 0 (section 4). The drafter's own
confidence is the only signal available at that moment, so this branch ships **GLM53_SPEC_VTRIM** (default off):
`shadow` measures how good DFlash2's confidence is in production (and projects every threshold of a grid), `on` trims.
It is exact at every temperature (section 6). How much of the ceiling DFlash2's confidence reaches can only be measured
in production; a target-confidence proxy reaches +1.5..+6.6 % (section 4), the oracle +12..+24 %.

## 1. Measurements (production, read-only)

* **Per-step traces** (`tests/vtrim/policy/trace_stream.py`, SSE chunks with `continuous_usage_stats`: one chunk =
  one engine step, tokens = 1 + accepted, gap = step time): 4 workloads x 5 runs, base (`traces/ada_a.json`) and pinned
  to K=7 through a no-op structured-output regex (`[\s\S]*`; adaptive-K keeps structured requests at full length;
  `traces/pin7_a.json`; the grammar costs nothing measurable: structured 82.5 vs 82.3 ms/step).
* **Adaptive-K decides two steps late**: simulating production's EMA on the observed accepted counts, only lag 2 is
  consistent with every step (0 / 996 steps accepting more than the simulated K; lag 0: 33, lag 1: 32, lag 3: 19).
  K_t is computed from acceptances up to step t-3 (async scheduling with the batch queue).
* **Step time vs K** (inferred K, lag 2; median ms): prose 71.2 / 75.7 / 84.4 at K 4 / 5 / 7, ja 68.3 / 71.1 / 80.1,
  coding 78.6 / 88.8 at 5 / 7, structured 83.4 at 7. Fit: `ms = C0[w] + 4.46 K`, C0 = structured 52.6, prose 53.7,
  coding 57.1, ja 50.1 (residual sd 3.5 ms).
* **Uncensored acceptance at every position** (`tests/vtrim/policy/probe_positions.py`): for a 200-token temp-0
  reference, `prompt + ref[:p]` is sent with max_tokens 10; the prefill emits ref[p] (checked) and the first verify step
  runs at K=7 (adaptive-K's MIN_STEPS), so its emitted count gives L(p) = accepted drafts of a fresh block at p. 13
  references (the 4 bench prompts + 3 more prose / ja / coding prompts each), 2,587 probes, ~0.45 s each. A replay of
  production's EMA along L(p) reproduces the real adaptive trace of the same text step for step (coding: the first 19
  steps identical), so the replay below is the exact dynamics, including the fact that a truncated block is mostly
  recovered by the next, fresh block (why K=4 accepts about as much per step as K=7 on prose).
* **Repetition penalty is not the cause** of low prose/ja acceptance: the server default `repetition_penalty 1.05`
  (the drafter does not know it) vs an explicit 1.0: prose 2.13 vs 2.12, ja 1.81 vs 1.79 acc/step (`traces/rp10_a.json`).
* **Base is not deterministic at temp 0** (prose: 10 distinct texts in 10 runs, coding 7, ja 9; structured 1): "identical
  outputs" can only be judged against base's own spread (MoE atomics order).

## 2. Dead rows cost nothing in the MoE (nodeC)

`tests/vtrim/bench_dead_rows.py`: production's decode MoE layer (router, grouped top-k, production's
`apply_exl3_experts` through the TF K2 apply, shared expert on the aux stream; 42 layers per CUDA graph, corr40
routing), T = 8 rows with the last d routed ids set to -1 vs T = 8-d live rows, interleaved rounds
(`docs/logs/spec_vtrim/bench_dead_rows.log`):

| rows | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| live rows, us / layer | 452 | 585 | 693 | 784 | 850 | 906 | 956 |
| 8 rows with 8-T dead, us / layer | 457 | 590 | 697 | 784 | 850 | 912 | 956 |

Dead rows cost +0..5 us per layer over removing them (the -1 sentinel group reads no expert bytes); a row costs 2.1
(8->7) to 4.5 (4->3) ms per step over 42 layers. Live rows are equal to the no-dead run within the kernel's own atomics
noise (rel 3e-5..9e-4, the same as two identical runs).

## 3. CPU policies (position replay, 13 references; tok/s vs production's EMA {4,5,7})

| policy | structured | prose | coding | ja |
|---|---|---|---|---|
| EMA {4,5,7} (production) | 0 | 0 | 0 | 0 |
| fixed K = 3 / 4 / 7 | -36.5 / -25.6 / 0 % | +0.8 / +1.9 / -11.7 % | -7.9 / -4.3 / -3.4 % | +2.8 / 0 / -12.9 % |
| EMA {3,4,5,7} | 0 | -0.5 | -0.4 | +2.6 |
| EMA {2,4,7} | 0 | -1.3 | -0.9 | +3.0 |
| EMA {4,5,6,7} / {4,7} / {5,7} | 0 | +0.1 / +0.2 / -4.4 | -0.5 / +0.5 / +0.3 | +0.1 / +0.3 / -3.9 |
| EMA alpha / margin variants | 0 | -1.1..+0.6 | -1.4..+0.2 | 0..+0.4 |
| EMA without the 2-step lag | 0 | +0.4 | -1.8 | 0 |

The only CPU change above noise is adding K=3 for Japanese (+2.6..3.0 %), driven by the harder ja references
(ja_sleep E[L] 1.21); on the bench's own ja text it is -1.6 %, and prose/coding lose slightly. Not recommended.
DFlash2's block is 8 tokens, structured accepts 6.96 of 7: nothing to gain there from policy.

## 4. Per-step trimming: ceiling and a proxy

Trimming inside the EMA's K, dead rows cost no routed-expert bytes (section 2 table), position replay:

| rule | structured | prose | coding | ja |
|---|---|---|---|---|
| oracle n* = L (knows the block's acceptance) | 0 | +19.1 % | +12.7 % | +24.3 % |
| target-confidence proxy, tau 0.3 / 0.5 / 0.7 | 0 | +1.5 / +4.0 / -0.8 % | +0.7 / +1.1 / -0.4 % | +1.5 / +4.8 / +6.6 % |

The proxy uses the TARGET's top-1 probability at each drafted position (from `prompt_logprobs`, `traces/tlp.json`); it
is a weak predictor here (the target is >= 0.99 sure at most positions where the drafter still fails: acceptance
0.81-0.88 in that bin for prose/ja). DFlash2's selector distribution is the drafter's own uncertainty and may be
better or worse; that is what `shadow` measures.

## 5. GLM53_SPEC_VTRIM (glm53_spec_vtrim.py, site py_module)

Rule: q_i = softmax(realized selector scores of step i / T) over DFlash2's 16 candidates (T = the request's
temperature, 1 for greedy); S_i = qd_1 ... qd_{i-1} x qmax_i; n* = the longest prefix with S_i >= TAU (>= MIN).
Rows of drafts i > n* are dead. Hooks (installed after weight load, before graph capture):

| hook | where | what |
|---|---|---|
| drafter | `DFlash2Speculator._sample_path` (inside the drafter FULL graph) | one Triton kernel: n* per request state (+ the 8-threshold grid in shadow) |
| propose | `DFlash2Speculator.propose` | batches with structured-output requests: n* = no trim |
| rejection | `rejection_sampler.rejection_sample` (eager) | on: drafts of dead rows -> -1 (vLLM's placeholder) in this call only; stats (both modes) |
| combine (on) | `model_runner.combine_sampled_and_draft_tokens` (eager, step preparation) | LIVE[token] = 0 for dead rows of the step |
| moe (on) | `exl3.apply_exl3_fused_moe` (inside the target graph, decode sizes <= 64 tokens) | routed ids of dead rows -> -1 |

Stats line every GLM53_SPEC_VTRIM_LOG verify steps (first after 64), both ranks:
`[glm53-spec-vtrim] rank R mode M tau T min N: verify steps S, request-steps Q, drafts D, (would-be) dead X (x%),
accepted A (a/request-step)[, accepted drafts the trim would cut Y (y%)], trimmed steps Z, structured-skips W, nsum H`
(nsum = a checksum of n*: equal on both ranks). Rank 0 also writes `<VLLM_CACHE_ROOT>/glm53_spec_vtrim_hist.json`
(request-steps by K, min(n*, K), accepted; in shadow also for every grid threshold 0.05..0.7), which
`tests/vtrim/project_hist.py` turns into a projected tok/s per threshold (calibrated against the exact replay: cut
accepted drafts are not recovered, because the EMA then lowers K; section 4's proxy matched within 0.5 point except
prose at tau 0.7). `tests/vtrim/shadow_bench.sh` runs the 4 bench workloads under shadow and prints that projection per
workload and threshold.

Safety: off = nothing installed (production byte for byte; `integrate.py` only imports the module). Mode on refuses to
start (raises in the plugin loader / at weight load) when the hooks cannot be installed: a rank that does not trim
while the other does would accept different drafts. Both ranks compute n* from the replicated drafter outputs (the
same values the draft tokens come from). Batches with grammar requests are never trimmed. Memory: max_num_seqs int32 +
max_num_batched_tokens bytes + 4.5 KiB (shadow grid +35 KiB) per rank. Cost of mode on when nothing is dead (the
masked_fill per MoE layer in the decode graph, `tests/vtrim/bench_hook_cost.py`, nodeC rig): +2.6 / +4.7 us per layer =
+0.11 / +0.20 ms per step at 5 / 8 rows (structured, which is never trimmed, pays ~0.25 %); shadow has no MoE hook. EMA interplay: adaptive-K sees the trimmed
acceptance (K may drop); included in the replay numbers.

## 6. Exactness (nodeC)

* `tests/vtrim/test_vtrim_exactness.py` = tests/blockverify/test_block_exactness.py (production's real DFlash2 walk +
  draft-logits cache + RejectionSampler._verify with production's sampling params, adaptive-K n in {4,5,7}, cross-step
  chi-square vs the exact sequence distribution) with n* from `nstar_launch` on the walk's realized scores and the
  production hook's masking: std-prod-vt, blk-fix-vt (TAU 0.3), blk-fix-vt9 (TAU 0.9, 84-100 % dead rows): exact in
  every config (prod n=4/5/7: seq p 0.04-0.93, step2 p 0.13-0.99, 1,024,000 sequences each); an ILLEGAL rule that cuts
  at the first draft with q(d_i) < 0.5 (uses d_i) is detected (p = 0, TV 0.027-0.24): the harness has power against
  exactly the mistake the rule avoids. Logs `docs/logs/spec_vtrim/exact_{prod,textbook,stress}.log`.
* `tests/vtrim/test_spec_vtrim_unit.py`: kernel == float64 reference (14,400 + grid cases), n* independent of the
  draft at position n*+1, LIVE kernel == reference (200 layouts), MoE/rejection hooks, env parsing (`unit.log`).
* `tests/vtrim/test_spec_vtrim_unit.py` also checks the fused stats kernel against a torch reference (300 batches).
* Real engine (`tests/vtrim/run_engine_ab.sh`: the handoff harness = production image + the launcher overlay chain + this
  bundle, the EXL3-MoE handoff mini model (10 layers, 32 routed experts) + the real DFlash2 drafter, adaptive K {4,5,7},
  greedy, prompts 700/1500/3001, 96 tokens; `docs/logs/spec_vtrim/engine_ab{,2}.txt` + `engine_ab/`): every mode boots,
  captures its graphs and serves 285 verify steps; on installs all five hooks, on100 (TAU 1.0) runs every step with
  100 % dead rows, on30 with 98-99.6 % (the mini drafter is never accepted and never confident: its own n* agrees);
  synthetic acceptance (rates 0.95..0.5) + on30 runs 99.7 % dead rows that would have been accepted and falls back to
  the target's greedy text. Greedy outputs: the engine is not deterministic across processes (off_a vs off_b first
  differ at tokens 26 / 18 / never; shadow30 30 / 17 / never; on30 64 / 46 / 64; on100b 26 / 17 / 64 of 64), so
  cross-process equality cannot be the test - the divergence points are of the same kind as off vs off, at exact or
  near logit ties of the degenerate mini model (e.g. request 2 position 1: two tokens at -4.4298). One on100 run showed
  a 5e-4 different PREFILL logprob of request 2; the repeat (on100b) and a base run with other generation lengths
  (off_g64) reproduce the base value exactly: cross-process noise, not a dependence on the previous request. The
  shadow histogram file is written by rank 0 into the bind-mounted cache and `project_hist.py` reads it
  (`engine_ab/shadow30_hist.json`).

## 7. Deploy / A/B (operator)

Kit stage `r16z6sv` (= r16z6 + SV1-SV4: GLM53_SPEC_VTRIM / _TAU / _MIN / _LOG forwarded to both ranks and validated;
`tools/deploy16/make_start_sh.py --stage r16z6sv`), env.r16 `#knob` lines, `tools/env_r16.sh on|off specvtrim` /
`specvtrim-shadow`, boot_checks rows (plugin line on every rank; on/shadow need the install line with every hook).
Order: (1) shadow restart + `shadow_bench.sh` (doubles as a base arm; outputs untouched) -> projected tok/s per workload
and threshold; (2) only if a threshold projects a gain on prose/ja without hurting coding: `on` at that TAU, A/B with
arm_env_ab.sh on all 4 workloads + klh dvptf (must stay at base ~0.0115: trimming does not touch live rows).
