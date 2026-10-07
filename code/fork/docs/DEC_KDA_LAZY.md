# DEC_KDA_LAZY — one KDA recurrent state per verify step instead of one per row (`GLM53_DEC_KDA_LAZY`)

Status 2026-10-03, branch `opt-decode` (from `r16z2rev` 60b53bf). Built and measured on nodeC only (one GB10,
production image, `tests/gpu_run.sh` under `flock /tmp/tf-gpu-bench.lock`). Production (nodeA/nodeB) was read only
(the non-secret `.env` knobs and the `SpecDecoding metrics` / adaptive-K log lines). Logs: `docs/logs/opt_decode/`.

## 1. The cost

Every DFlash2 verify step runs `fused_recurrent_kda` once per KDA layer (34 of 45 layers, both ranks) over the
M = K + 1 rows of each request. Because the next step starts from the state after the last *accepted* row
(`num_accepted_tokens - 1`), which is only known after sampling, production stores the fp32 state after **every** row
(32 heads x 128 x 128 x 4 B = **2 MiB per row per layer per rank**): 10 / 12 / 16 MiB per layer at M = 5 / 6 / 8,
340 - 544 MiB per step per request, of which one 2 MiB state per layer is read again. The kernel itself is short
(~30 us, the stores land in L2), but every dirty line is written back to DRAM later, by whatever bandwidth-bound GEMV
runs next (DEC_FP8ROOF R.6 measured +35 us on an in_proj after 8 MiB of fresh writes).

Measured on nodeC (`tests/optdec/probe_kda_state_writes.py`, the image's kernel, a 34-layer step with GEMV
stand-ins for in_proj / o_proj / MoE in one CUDA graph): removing the per-row stores (index table with NULL slots)
saves 1.57 / 2.42 / 2.44 ms per step at M = 5 / 6 / 8 (`probe_kda_state_writes.log`).

## 2. What GLM53_DEC_KDA_LAZY=1 does (`glm53_kda_lazy.py`)

* **verify** (inside the target's decode CUDA graphs): `_kda_lazy_verify_kernel` is production's strided
  `fused_recurrent_gated_delta_rule_fwd_kernel` (the r16 `patch_kda_strided_qkv.py` text) specialised to the
  spec-verify call, every arithmetic line in production's order. It stores the output exactly like production and
  **no state**; instead each row's raw inputs (k, v, g logits, beta logits as loaded: 24.6 KiB per row per layer) go
  to a per-layer scratch, with the row count, the initial state's slot id and the row slot ids (`meta`, flag = 1).
* **commit** (once per step, eager, right after sampling): `MambaHybridModelState.postprocess_state` is wrapped; before
  its own work (the `num_accepted` scatter and the align post-copy) one launch of `_kda_lazy_commit_kernel` over all
  lazy layers recomputes, from the untouched initial state and the saved rows, the first A = max(num_sampled, 1)
  rows and stores the state after row A - 1 into column A - 1, and, when this step makes the mamba "align" prefix
  cache copy a block-aligned state, the state of that column too (the predicate and column of
  `postprocess_mamba_fused_kernel`). Then every flag is cleared. Commit traffic: 2 MiB read + 2 MiB write per layer
  per request (vs M x 2 MiB written).
* After the commit, every column the rest of vLLM reads (the next verify's initial column, the align pre-copy and
  post-copy columns) holds production's bytes. Columns of rejected rows (and accepted rows nobody reads) keep stale
  bytes instead of production's never-read ones.
* Only pure spec-verify batches go lazy (every FULL decode graph). Mixed batches (prefill / plain decode rows, eager
  breakable segments), any dtype / shape / stride class it was not validated for, a first call inside a capture ->
  production's function with exactly the arguments received, and that layer's pending flags are cleared. The commit
  kernel addresses every layer's state from one anchor tensor + 16-aligned element offsets with production's
  constexpr slot stride: these address facts decide the register layout of the state tile and with it the rounding
  of every `tl.sum`; with a runtime stride the commit differed from production by 1 ulp (found by T1, fixed).
* **Self-check** (`GLM53_DEC_KDA_LAZY_VERIFY`, default the first 64 commits, then one in
  `GLM53_DEC_KDA_LAZY_VERIFY_EVERY`, default 1024; 0 = never): on a checked step, per layer, production's own kernel
  recomputes the state from the saved rows into a temp buffer and the commit kernel runs in DRY mode (same code,
  store target = a temp buffer); they are compared byte for byte; the cache then gets production's bytes for that
  step whatever the result. A difference switches every later commit to production's kernel per layer (repair mode:
  exact, slower) for the life of the process — captured graphs keep needing a commit, so the commit is what falls
  back. A checked step costs a host sync (like hostloop's VERIFY steps).
* Memory per rank: scratch 3 x 34 layers x max_num_seqs (8) x 8 rows x 32 x 128 x 2 B + tables = **51.7 MiB**;
  a checked step allocates ~36 MiB transient per layer, sequentially.
* `GLM53_DEC_KDA_LAZY` unset/empty: the module only logs that it is off; nothing is patched.

## 3. Exactness (all on the real kernels; the image's `fused_recurrent.py` / `kda.py` with the strided backport bound
over it = production since r16k)

`tests/optdec/test_kda_lazy.py` via `tests/optdec/run_kda_lazy_tests.sh`: **ALL OK** (`test_kda_lazy.log`)

| test | what | result |
|---|---|---|
| T1 | 120 cases: N 1..8 sequences x T 1..8 rows x strided/contiguous q/k/v x random initial column 1..8 and A 0..T | output == production bitwise; the verify writes no state byte; commit column A-1 == production bitwise; nothing else touched |
| T2 | 100 steps of chains, K in {4,5,7} per step, random acceptance, slot tables moving (align migration) | every output and every next-initial state bitwise |
| T3 | 72 requests with a block boundary inside the accepted rows (bias column) + ALIGN_ALL | bias / all accepted columns bitwise |
| T4 | lazy verify captured in a CUDA graph, 8 replays with new inputs, eager commit | bitwise |
| T5 | wrapper outside a pure spec batch | production's output and per-row states; flags cleared; a later commit writes nothing |
| T6 | scalar num_sampled 0 / 1 / 3 | bitwise (0 treated as 1, like postprocess_state) |
| T7 | self-check on correct commits (18 checked steps) | passes, no repair |
| T8 | corrupted fast commit (gate table +0.25) | detected on the first check, that step committed by production's kernel, repair mode exact |
| T9 | gate lower bound -3 / -7.5 (the commit uses the bound the lazy calls saw; layers with different bounds keep production's path) | bitwise |

Real engine (`tests/handoff/run.sh`, production-composed container with the r16z2 kit: V2 runner, breakable +
FULL CUDA graphs, mamba align prefix caching with 4608-token blocks, FLASHINFER_MLA_SPARSE_SM90, DFlash2 adaptive K
{4,5,7}; mini model; `GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1`):
* `lazy_syn` (vLLM's synthetic acceptance 0.9..0.5 so many rows are accepted; prompts 4590/3001 concurrently, then 9200,
  4600: decode crosses the 4608 / 9216 align boundaries): 102 commits, all self-checked, **620 committed states ==
  production's kernel byte for byte (550 with A > 1), 0 mismatches**.
* `lazy_def` (default first-64 check, then every 16th, 400 tokens per request): 232 commits, 157 on the fast path
  between checks, 75 checks, 0 mismatches, generation normal.
* Result-level comparison across processes is not bitwise even for off vs off (`handoff_result_compare.txt`: off vs
  off2 differ in logprobs by up to 0.15 and in tokens for 2 of 3 requests; off vs lazy: tokens identical, logprob
  differences inside the off-vs-off band) — the engine's own cross-process nondeterminism, which is why the in-process
  self-check is the integration oracle.

## 4. Speed (nodeC, one GB10 = one TP rank's shapes; `bench_kda_lazy_step.log`)

`tests/optdec/bench_kda_lazy_step.py`: one step = 34 KDA layers x (in_proj stand-in 51.5 MB -> KDA recurrent ->
o_proj stand-in 16.8 MB -> MoE stand-in 177 MB) in one CUDA graph (production's FULL decode graph), lazy mode adds the
eager commit after the replay; A drawn per step from production's per-position acceptance; paired, alternating order.

| M (rows) | requests | production ms | lazy ms (graph + commit) | paired saving, median [min, max] |
|---|---|---|---|---|
| 5 | 1 | 51.87 | 50.61 (49.98 + 0.64) | **1.39** [0.42, 4.22] ms |
| 6 | 1 | 52.59 | 50.78 (50.15 + 0.64) | **1.54** [0.21, 4.50] ms |
| 8 | 1 | 53.33 | 50.80 (50.15 + 0.64) | **2.28** [0.86, 3.87] ms |
| 5 | 2 | 53.76 | 51.38 (50.09 + 1.30) | **2.30** [0.14, 3.77] ms |
| 8 | 2 | 57.31 | 52.80 (51.52 + 1.28) | **4.44** [3.06, 6.14] ms |

(An earlier session of the same bench, before the self-check code, gave 1.42 / 1.48 / 1.76 (M=5), 2.24 (M=6), 2.71
(M=8), 2.71 (M=5 x 2) ms: same picture, nodeC shared.) With production's 24 h adaptive-K mix (K=4 50 %, 5 30 %, 7 20 %,
`prod_spec_metrics_24h_summary.txt`) the single-stream expectation is **~1.6 ms per verify step (~2 % of ~75 ms;
structured / K=7 ~2.8 %)**, and it grows with concurrent requests (the stores grow with every request).

Not proven here: the TP=2 / RoCE environment. This is a *fewer bytes* change (≈200-400 MiB less DRAM traffic per step
and request), not a "use idle windows" one, so the fp8roof lesson (NIC contention ate the gain) should not apply, but
only the production A/B can say.

## 5. Production A/B (needs the owner's go; nothing was deployed)

1. Kit: the knob must reach both ranks. In the kit's `start.sh` add `GLM53_DEC_KDA_LAZY GLM53_DEC_KDA_LAZY_VERIFY
   GLM53_DEC_KDA_LAZY_VERIFY_EVERY` to the knob loop (next to `GLM53_DEC_SMALLOPS_KINDS`, line ~2506 of the r16z2
   start.sh) and an `-e NAME="${NAME:-}"` line per name for both containers (next to `GLM53_DEC_SMALLOPS_KINDS`,
   ~2747); `env_r16.sh`: a feature `kdalazy` with those names; the bundle `site/` gets `glm53_kda_lazy.py`
   (setup.py py_modules) and the new `integrate.py`.
2. Boot check: both ranks log `glm53_kda_lazy on Glm5NextForCausalLM: 34 KDA layers, 8 seqs x 8 rows, 32 heads x 128;
   scratch 51.7 MiB` and, within the first decode steps, `self-check 1: fast commit == production's kernel byte for
   byte`, later `self-check 64: ...`; never `self-check: ... differ` (repair mode = exact but no gain).
3. Bench (A = today, B = + `GLM53_DEC_KDA_LAZY=1`): the usual bench_round per workload (ms/step; acceptance must be
   unchanged: the change is bitwise), `kpool_decode_consistency.py` (decode-vs-prefill KL must stay ~0.007-0.008,
   the state handoff class of 0929), quality_long.
4. Revert: remove the `.env` line and restart (unset = nothing patched).

## 6. Not done / follow-ups

* Fold the commit into the next verify kernel's prologue (no re-read of the initial state, no extra launch: another
  ~0.3 ms per step per request) — needs every reader between steps (align copies, non-spec paths) handled; not worth
  the risk before the A/B of this version.
* The 2 MiB states are fp32 by vLLM's choice (`kda_state_dtype` -> float32); a bf16 state would halve everything but
  changes numerics (not lossless) — not proposed.

## 7. v2 (branch `decode4-kdalazyfix`, 2026-10-04): the production corruption and its fix

**What production showed** (r16z6rev A/B 2026-10-04 15:01, `env-ab/z6kl/run.log`): dvptf 0.0115 -> 0.206, kpool
decode-vs-prefill KL 0.022 -> 0.415 from the first 128 generated tokens, coding acc/step 4.13 -> 2.73, temp-0 outputs
all different, one Japanese output with doubled characters; prefill KL unchanged (0.0097 -> 0.0099). The self-check
logged "byte for byte" and never "differ".

**Root cause.** v1 registered layers by `recurrent_state.data_ptr()`. GLM-5-Next's hybrid KV layout
(`vllm/v1/core/kv_cache_utils.py` `_glm5_next_tensor_layout`) makes ONE tensor per MLA layer, co-owned by that MLA
layer and one KDA layer of EACH mamba group (disjoint block ids). Production: 34 KDA layers in groups of 11 -> every
recurrent-state view is shared by 3-4 KDA layers (same data_ptr, same stride). The first co-owner registered and went
lazy; every later co-owner found the slot's registered A_log different ("a_log / g_bias tensors changed"), took
production's path, and the wrapper then **cleared the first co-owner's pending flags** (`ST.meta[s, :, 0].zero_()`).
The commit skipped that layer, so ~11 of 34 KDA layers never stored their state during decode (each verify restarted
from a stale column: the model "forgets" the last tokens -> repeated characters). The self-check could not see it: it
compares the commit with production's kernel over the same saved rows, and a layer without flags is not compared
(`checked_states` stayed 0 while the line said "byte for byte"). Neither the unit tests (one tensor per layer) nor the
handoff mini model (5 KDA + 5 MLA = a single mamba group, no sharing) had a shared tensor.

**Fix (v2).** Layers are keyed by `(recurrent_state.data_ptr(), A_log.data_ptr())` (`layer_key`); sharing layers get
their own scratch slot and table row; their states live in the same tensor at disjoint slots, which the commit already
addresses per layer (anchor + offset + slot). Self-check: a checked commit with no pending state no longer counts as a
proof (`checked_empty`); the proof line names live layers / states:
`glm53_kda_lazy: self-check 1: fast commit == production's kernel byte for byte (v2 per-layer keys: 34 of 34 layers live x 1 sequences, 34 states)`;
a checked commit where only some registered layers were pending logs a one-time WARNING `only N of M registered KDA
layers had a pending lazy verify`. Boot line ends with `; v2 per-layer keys`.

**Proof.**
* `tests/optdec/test_kda_lazy_shared.py` (new; real kernels, the real wrapper + commit, 6 KDA layers on 2 shared state
  tensors in 3 groups with page padding, K in {4,5,7}, partial acceptance A in 1..K+1, N = 1/2/4/8, fast commit and
  self-check modes): **v1 FAIL** (2 of 6 layers registered, 600 / 1800 next-initial states and 152 outputs differ,
  self-check "passes" with 0 checked states) -> **v2 ALL OK, bitwise** (`docs/logs/decode4/test_shared_*.log`).
  T1-T9 ALL OK after the fix (`test_kda_lazy_T1T9_after_fix.log`).
* Real engine (`tests/handoff/build_mini_2g.py`: 10 KDA + 5 DSA = 2 mamba groups, every state tensor shared by 2 KDA
  layers; synthetic acceptance 0.9..0.4 so most steps accept several rows; prompts 3001/1200, 64 tokens, 48 fresh
  prefills B_j): decode-vs-prefill KL p50 / mean / top-1: off 0.00047 / 0.131 / 48/48, off rerun 0.00044 / 0.080 /
  48/48, **v1 0.456 / 1.275 / 41/48**, **v2 0.00047 / 0.039 / 48/48**, v2 rerun 0.00040 / 0.058 / 48/48. Tokens: v2 and
  the off rerun diverge from off at the same position (11, |dlogprob| 0.066 = cross-process noise); v1 diverges with
  |dlogprob| 0.30-0.46. v1's log shows `production path for a KDA call (a_log / g_bias tensors changed)` and
  `registered 5` of 10 (`handoff_mini2g_*`).

**Speed (nodeC, one GB10; `tests/optdec/bench_kda_lazy_side.py`, 34-layer step + a 1.1 GB DFlash2 drafter stand-in,
2 steps per sample, paired, 40 rounds):** v2 (34 lazy layers) saves 1.12 / 1.48 / 2.16 ms per step at M = 5 / 6 / 8
and 2.04 ms at M = 5 x 2 requests. The v1-in-production emulation (11 lazy layers) saves 0.34 / 0.40 / 0.64 ms on
nodeC, but production measured +0.41 ms (structured, M = 8) and -0.42 ms (prose) for v1: production paid ~1 ms more
than the nodeC model predicts for 11 layers. So the production gain of v2 is **not established**: nodeC says -1.1 to
-2.2 ms/step, the v1 calibration says it may be ~1 ms less. Only the production A/B can tell.
A side-stream commit overlapped with the drafter was measured and rejected (lazy vs side: +0.02 / +0.04 / -0.02 ms:
the drafter is bandwidth-bound, nothing to overlap).
* Same engine rig with production's decode knobs on top (GLM53_DEC_HOSTLOOP=1 fast path ON, SMALLOPS dconv, KPOOL_RING,
  PREFILL_QUICKWINS=all, MLA_PREFILL=1; `handoff_mini2g_prodknobs.txt`): KL p50 / mean / top-1 off 0.00074 / 0.045 /
  48/48, **v2 0.00041 / 0.063 / 48/48**, **v1 0.451 / 1.281 / 41/48**; v2 log `10 of 10 layers live`, 0 mismatches.
  Runner: `docs/logs/decode4/handoff_go.sh`, KL tool `tests/handoff/consistency_kl.py`.

**A/B recipe (operator).** Kit `tf-exl3-deploy16.r16z6rev-kl2` = r16z6rev + this site/glm53_kda_lazy.py (prev-r16 =
r16z6rev with its production start.sh variant 1686a3b2; apply_r16 dry run on a production mirror: replaces launcher/start.sh (the
plain one, as r16z6rev's own swap did) + site/glm53_kda_lazy.py, adds nothing to .env, start.sh validates). Judge on
ALL FOUR bench workloads (structured never exercises partial acceptance) + klh dvptf; PROOF
`v2 per-layer keys: 34 of 34 layers live`; NEVER `glm53_kda_lazy: self-check:`; also grep both ranks for
`production path for a KDA call (a_log` and `pending lazy verify` (must be absent).

### 7.1 Adversarial review (decode4 review, 2026-10-04)

* Re-ran T1-T9 and S1-S3 on nodeC: ALL OK. New `tests/optdec/test_kda_lazy_adv.py`: production shape (34 KDA layers on
  11 shared state tensors, 4 mamba groups 11/11/11/1), 10 requests drawn into random batches (N 1..8, random row order,
  join/leave), K in {4,5,7} per step, partial acceptance, align bias columns, 1 in 5 steps mixed (production path):
  outputs, column A-1 and 2958 align-bias columns bitwise == production, 34/34 registered; again with the self-check on
  every commit: 0 mismatches. `docs/logs/decode4/review/test_kda_lazy_adv.log`.
* Host cost of the eager commit (34 layers, align on, nodeC Grace): 20.7 us p50 per step - cannot explain the ~1 ms
  v1 production discrepancy (`review/hostcost.log`).
* Engine (mini2g, FAST commit only: `_VERIFY=0 _VERIFY_EVERY=0`; the original runs used `_VERIFY=4/_EVERY=16`, so
  1 in 16 steps had production's bytes), 2 concurrent decodes, both crossing a mamba block boundary during decode
  (32256 retention point and 4608), B_j prefills with A0's salt (prefix hits) paired with fresh prefills of the same
  tokens (`HANDOFF_B_SALT=A HANDOFF_B_PAIR=1`): hit-vs-fresh at the prefill-saved 27648 checkpoint off 0.046 / v2
  0.036 mean KL (top-1 18/20 both); decode-vs-prefill j<20 v2 p50 0.0012, top-1 20/20. Hit-path outputs vary
  run-to-run even with lazy OFF (off == off2 == v2-checked, but off != offC (both OFF) and offC == v2-fast): process
  nondeterminism of the hit path, not the lazy commit.
* NOT a lazy defect, found on the way (OFF, mini rig): a prefix hit at the 32256 checkpoint followed by a SHORT prefill
  (4-23 tokens) is badly inconsistent with a fresh prefill of the same tokens - KL ~0.8, top-1 0-1 of 12-20 - both for
  the checkpoint saved during decode (offP) and the one saved during prefill (offQ, `HANDOFF_B_BASE=32260`); a hit at
  27648 followed by ~4.6k tokens is fine. Unverified on production; see `review/engine_prefix_hits.txt`.
