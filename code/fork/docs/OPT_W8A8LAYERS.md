# OPT_W8A8LAYERS — where the W8A8 dvp damage sits, and a layer filter (branch opt-w8a8layers, from r16z2rev 60b53bf)

2026-10-03, nodeC only (every GPU run through `tests/handoff/run.sh` under `/tmp/tf-gpu-bench.lock`, 40 GiB guard).
nodeA/nodeB untouched. Raw outputs: `${HOME}/tf-exl3-assets/opt-w8a8layers/runs/` (per-position `dvp.jsonl`);
copies of the small ones (arm summaries, driver logs, per-run `summary.txt`) in `docs/logs/opt_w8a8layers/`.

Question: production's W8A8 "all" arm (A/B 2026-10-02, MoE e4m3 on) misses only the dvp gate (0.0242 vs <= ~0.023;
base 0.0195, sub 0.0181). Can a few sensitive layers be excluded to recover dvp while keeping most of all's speed?

## 0. Verdict

* **The dvp damage is concentrated by PROJECTION, not by layer.** `mla.o_proj` (11 GEMMs in production, 106 of the
  551 ms/chunk) carries **63-80 %** of the all-arm's dvp excess on both rigs; excluding it everywhere leaves
  **27-35 %** (4 independent measurements: 33/35 % on rig P, 32/27 % on rig O). KDA projections, `kda.f_b/g_b`,
  `mla.q_b` and `shared.*` are at or near the noise floor; `mla.fused_qkv_a` (13-23 %) and the dense MLP of the last
  dense layer (rig P only, ~29 %) are the next ones.
* **Per-layer concentration exists but is not a property of the layer's weights**: on the original mini 3 of 10 layers
  carry ~90 % (L8 48 %, L5 28 %, L2 17 %); swapping attention weights between layers (rig P) moves the sensitivity to
  the new positions only partly and spreads it (L2 20, L3 16, L5 12, L7 19, L8 22 %). What is common to both rigs:
  the MLA layers' `mla.o_proj` in the middle-to-late MLA layers, never the last MLA layer, plus the third (last dense)
  layer. A "top-k layers" choice cannot be read off the 10-layer mini for production's 45 layers; the
  projection-level cut is the robust one, a depth-band cut of `mla.o_proj` is the speed-leaning bet.
* Built: **`GLM53_DENSE_W8A8_SKIP_LAYERS`** (fp8_w8a8.py; `<layers>[:<projection|group>]` items, validated, applied after
  ONLY, unset = r16z2rev bit for bit - unit tests and a real-engine bitwise check), so production can exclude
  projections per layer band (e.g. `19-39:mla.o_proj`, `2:dense`) - which ONLY cannot express. **Kit work not done**
  (start.sh forwarding to both ranks, env.r16 knob, boot_checks), see section 5.
* Proposed production arms (section 4): **A1** `SKIP_LAYERS=0-44:mla.o_proj` (~2,675 tok/s, dvp est 0.0208-0.0221),
  **A2** `SKIP_LAYERS=19-39:mla.o_proj` (~2,709, dvp est 0.0217-0.0227), **A4** `SKIP_LAYERS=0-44:mla.o_proj,
  0-44:mla.fused_qkv_a_proj,2:dense` (~2,664, dvp est 0.0195-0.0210). Production's dvp probe is a single free-running
  sample per arm; repeat base twice.

## 1. Measurement: teacher-forced decode-vs-prefill on identical trajectories (`tests/w8a8layers/dvp_driver.py`)

The production probe compares the decode path's logits A_j (prompt prefilled, then one token per step) with a fresh
prefill B_j of prompt + generated tokens. W8A8 serves only prefill GEMMs (M >= 512), so in A it touches only the
prompt's state while in B it touches every scored position. The driver reproduces exactly that, but with every arm on
the SAME token trajectories (the opt-decodekit-rev refuter showed free-running arms land on different texts):

* trajectories: 4 real-text windows (prompt 6,000 tokens, the next 256 real tokens forced) + 1 greedy trajectory
  (the W8A8-off arm's own greedy continuation of a 5th prompt, generated once, then forced for every arm);
  1,275 scored positions per arm (k = 1..255; k = 0 is a prefill row on both sides);
* A: the real decode path (one token per step, CUDA-graph replays), the sampler's token overwritten by the trajectory
  token after the fp32 log-softmax of the full vocabulary is captured; B: one prefill of prompt + trajectory[:255]
  (prompt_logprobs capture); metric **dvp = KL(A || B)** per position, plus pf = KL(B_off || B_arm) and
  da = KL(A_off || A_arm);
* all arms in ONE engine process (one boot with `GLM53_DENSE_W8A8=1`, every layer self-tested at load): an arm is a
  set of (layer, projection) pairs, the driver swaps `fp8_w8a8.selected` (consulted on every eager call) for membership
  and counts the served pairs through `try_w8a8` (0 SERVED-MISMATCH in 12 runs);
* the reported number is the PAIRED excess over the off arm (same positions), with a 90 % bootstrap interval over
  positions (resampled within each trajectory) - `tests/w8a8layers/summarize.py`;
* engine: the production image composed like production (kit r16z2rev chain + this worktree's bundle), production's
  prefill config (`GLM53_MOE_E4M3=1 GLM53_PREFILL_FUSED_CAP=1 GLM53_KDA_FLASHKDA=1 GLM53_MLA_PREFILL=1
  GLM53_PREFILL_QUICKWINS=all`), MNBT 13,824, fp8 KV, deterministic prefill top-k (stable sort), no drafter
  (decode = M=1 steps; production verifies M=K+1 <= 8, also never W8A8).

Rigs (both at TP=2 rank-0 shapes, 5 KDA + 5 MLA, 3 dense-MLP + 7 MoE layers with the shared expert):
* **O** = `GLM-5.3-Flash-handoff-moe-mini-tp2r0` (attention sources KDA0, KDA1, KDA10, DSA11, KDA12, DSA45, KDA13, DSA11,
  DSA45, DSA11);
* **P** = the same file with the attention tensors of layers 3<->5, 9<->8, 2<->6 swapped (header-only edit, built by
  this branch; `${HOME}/models/GLM-5.3-Flash-handoff-moe-mini-tp2r0-perm`, attention sources KDA0, KDA1, KDA13,
  DSA45, KDA12, DSA11, KDA10, DSA11, DSA11, DSA45). Purpose: separate "which weights" from "which position".

### Noise floor
| | rig O | rig P |
|---|---|---|
| dvp of off (level) | 0.00178-0.00201 | 0.00077-0.00088 |
| off repeated in the same process (excess, 7 / 6 repeats) | -0.00008 .. +0.00019 (<= 6 % of all) | -0.00000 .. +0.00003 (<= 1 %) |
| all repeated (excess) | 0.00278 / 0.00298 / 0.00317 / 0.00311 / 0.00309 / 0.00291 / 0.00293 | 0.00209 / 0.00207 / 0.00216 / 0.00216 / 0.00215 |
| pf of off repeats (prefill side) | <= 0.00003 | <= 0.00004 |

The decode side is not bit-reproducible between repeats (da of off#2 ~0.0005-0.0009 on O, much smaller on P); the
prefill side is (pf ~0). Rig O's per-trajectory swings (+-0.0005) make single-pair arms below ~0.0004 unresolvable
there; rig P resolves ~0.00005.

## 2. Sensitivity tables (dvp excess as a share of the all arm's excess, same run)

Per layer (one arm per layer, every projection of it): runA (O), runP (P)

| mini layer | O: attention / MLP | O share | P: attention | P share |
|---|---|---|---|---|
| 0 | KDA0 / dense | -1 % | KDA0 | 2 % |
| 1 | KDA1 / dense | -1 % | KDA1 | -4 % |
| 2 | KDA10 / dense (last dense layer) | **17 %** | KDA13 | **20 %** |
| 3 | DSA11 / shared | -7 % | DSA45 | **16 %** |
| 4 | KDA12 / shared | -2 % | KDA12 | -1 % |
| 5 | DSA45 / shared | **28 %** | DSA11 | **12 %** |
| 6 | KDA13 / shared | 4 % | KDA10 | 1 % |
| 7 | DSA11 / shared | 3 % | DSA11 | **19 %** |
| 8 | DSA45 / shared | **48 %** | DSA11 | **22 %** |
| 9 | DSA11 / shared (last MLA) | -6 % | DSA45 | 3 % |

Per projection type, all layers (runB1 O, runP2 P): `mla.o_proj` **80 % / 63 %**, `mla.fused_qkv_a_proj` 13 / 23 %,
`dense.down_proj` -3 / 19 %, `dense.gate_up_proj` -7 / 10 %, `kda.o_proj` -11 / 7 %, `kda.in_proj_qkvbfg_a` -6 / 5 %,
`mla.q_b_proj` -1 / 3 %, `shared.gate_up_proj` 12 / 0 %, `shared.down_proj` 7 / 0 %, `kda.f_b+g_b` -3 / -1 %
(O's single-projection values within +-0.0005 are noise). On P the layer-2 dense MLP alone is 10 % (gate_up) + 18 %
(down); layers 0/1 dense are ~0 on both rigs.

`mla.o_proj` per MLA layer (runB1/B3 O, runP3 P): L3 3 / 8 %, L5 24 / 12 %, L7 3 / 20 %, L8 46 / 23 %, L9 0 / 5 %.

Exclusion arms (remaining share of the all arm's excess; O = runB1/B2/B3, P = runP2/P3):

| arm | O | P | production cost (ms/chunk kept of 551) |
|---|---|---|---|
| sub (in_proj + mla.o) | 74 % | 67 % | 359 |
| all minus layers 0-2 | 77 % | 72 % | 491 |
| all minus layer 2 | 76 % | 78 % | (prod layer 2: 531) |
| all minus shared.* | 97 % | 92 % | 494 |
| all minus mla.fused_qkv_a, dense.* | 84 % | 75 % | 514 |
| all minus mla.fused_qkv_a, layer-2 dense | 81 % | 71 % | 535 |
| all minus the 2 worst MLA layers' mla.o (O: L5,L8; P: L7,L8) | 46 % | 60 % | (A2: 493) |
| **all minus mla.o_proj** | **32 %, 27 %** | **33 %, 35 %** | **445** |
| all minus mla.o, shared.* | 22 % | 34 % | 388 |
| all minus mla.o, layer 2 | 15 % | 23 % | |
| all minus mla.o, dense.* | 9 % | 28 % | 414 |
| all minus mla.o, mla.fused_qkv_a, dense.* | -2 % | 15 % | 408 |
| (O) all minus layers 2, 5, 8 | 10 %, 5 % | | |

Reading: errors that enter the residual stream at the scored token itself (attention output `mla.o_proj`, MLP down)
show up in dvp; errors of projections that feed the cached state (KDA in_proj, MLA q/kv) are mostly carried into
decode through the prompt's state (they appear in pf/da but cancel between A and B). That is why the KDA projections,
which dominate the saving, are almost free for dvp.

## 3. The filter: `GLM53_DENSE_W8A8_SKIP_LAYERS` (overlay/fp8_w8a8.py, repo-root copy identical)

* Syntax: comma list of `<layers>[:<name>]`; `<layers>` = `N` or `A-B` (model layer index from the vLLM prefix
  `...layers.N...`, 0 <= A <= B <= 4095); `<name>` = a `PROJ_NAMES` entry, `kda.f_b_proj` / `kda.g_b_proj` (served when
  ONLY is unset, not ONLY-selectable), or a whole group `kda | mla | dense | shared`; no name = every projection of those
  layers. Applied after `GLM53_DENSE_W8A8_ONLY`. A prefix without a layer index is never excluded.
* Unset/empty = `CFG.skip None` = `selected()` returns exactly r16z2rev's value, the load path calls exactly the same
  self-tests, no new counter, no new log line, report dict unchanged.
* Malformed value -> `_problem()` refuses the whole W8A8 install (production's path, WARNING names the bad item), like
  ONLY. Excluded layers are not self-tested at load and counted (`skip_layers_at_load=N` in the summary); the install
  log gains one INFO line `tf_fp8_w8a8 layer filter GLM53_DENSE_W8A8_SKIP_LAYERS: excluded ...` only when set.

Tests:
* `tests/w8a8layers/test_skip_layers.py` **195/195** on the host (torch stub) and **195/195 in the production image**
  (`docs/logs/opt_w8a8layers/test_skip_layers_{host,image}.log`): parsing (good/bad values incl. ranges, groups,
  f_b/g_b, malformed items), `selected()` == the verbatim r16z2rev function over production's 45-layer layout x 5 ONLY
  settings when unset, served-GEMM counts on production's layout (all 259, sub 45, `0-2` 241, `0-44:mla.o_proj` 248,
  each single layer removes exactly its own projections, every (layer, name) item exactly one GEMM), the load path
  (self-tests and the skip counter), the per-call path (excluded -> production's apply, kept -> try_w8a8), the log text.
* Real engine (rig O, production config `ONLY=kda.in_proj_qkvbfg_a,mla.o_proj`, SKIP unset): B logits of the off,
  sub and repeat arms **bitwise identical** (sha256 of the fp32 rows) between the r16z2rev module and this branch's
  (runs idBase / idNew; the in-process `all` arm's second trajectory differs in 1 of 3 boots in both modules - the
  occasional prefill nondeterminism also seen as pf 0.00001-0.00004 on off repeats).
* Real engine with `SKIP_LAYERS=0-2,3-9:mla.o_proj` (run skipEnv): load summary `selftest_passed=32,
  skip_layers_at_load=23` (55 - 23), served pairs 32, and B logits **bitwise equal** to the in-process emulation
  `all^L0..2^P:mla.o_proj` on both trajectories. `SKIP_LAYERS=0-2:bogus` (run skipBad): WARNING + not installed.

## 4. Production A/B proposal (base = production today: e4m3 MoE + `ONLY=kda.in_proj_qkvbfg_a,mla.o_proj`)

Speed: `tests/w8a8layers/arm_cost.py` (per-call savings at M=13,824 from the r16z2rev table over production's 45-layer
layout, time per 32k request interpolated between the measured sub/all arms). Quality: production all-excess
(dvp +0.0047, long KL +0.0020 over base) x the remaining share, two ways - **direct** (the rigs' remaining share) and
**count-scaled** (rig P's per-GEMM sensitivities x production's GEMM counts: mla.o 45 %, mla.fused_qkv_a 17 %,
kda.o 15 %, in_proj 11 %, dense 9 %, q_b 2 %, shared/f_b/g_b ~0 of all's excess; the 10-layer rigs have 5 MLA layers of
10, production 11 of 45). Long KL uses the pf ratio (all minus mla.o: 0.52-0.57).

| arm | env (with `GLM53_DENSE_W8A8=1`, ONLY unset) | GEMMs | ms/chunk | est. tok/s (32k) | est. dvp | est. long KL |
|---|---|---|---|---|---|---|
| base today (sub) | `ONLY=kda.in_proj_qkvbfg_a,mla.o_proj` | 45 | 359 | 2,620 (measured) | 0.0181 (measured) | 0.0140 (measured) |
| all | - | 259 | 551 | 2,752 (measured) | 0.0242 (measured) | 0.0154 (measured) |
| **A1** | `GLM53_DENSE_W8A8_SKIP_LAYERS=0-44:mla.o_proj` | 248 | 445 | **~2,675** | 0.0208 (direct) .. 0.0221 (scaled) | ~0.0145 |
| **A2** | `GLM53_DENSE_W8A8_SKIP_LAYERS=19-39:mla.o_proj` (6 of 11 MLA layers: 19,23,27,31,35,39) | 253 | 493 | **~2,709** | 0.0217 .. 0.0227 | ~0.0148 |
| **A4** | `GLM53_DENSE_W8A8_SKIP_LAYERS=0-44:mla.o_proj,0-44:mla.fused_qkv_a_proj,2:dense` | 235 | 429 | **~2,664** | 0.0195 .. 0.0210 | ~0.0143 |

* A1 is the robust choice (4/4 rig measurements -65..-73 %; the count-scaled estimate still passes). Equivalent without
  the new knob: `ONLY=` every name except `mla.o_proj` (leaves kda.f_b/g_b on production's path: 68 GEMMs x ~0.03 ms,
  0 dvp on both rigs).
* A2 bets on the depth pattern both rigs share (the middle-to-late MLA layers' `mla.o_proj` hurt, the last one and the
  first ones much less); it needs the new knob. If A2 passes, `15-39` (2,702) / `23-39` are the next points.
* A4 is the safe fallback if A1 misses (adds the two next contributors; `2:dense` = only production's last dense layer,
  10 ms, needs the new knob).
* Not worth an A/B: excluding `shared.*` (-3..-8 % only), layers 0-2 as a block (-23..-28 % for -60 ms), `in_proj`.
* Run base twice (the probe is one free-running sample: base 0.0195 vs sub 0.0181 already differ by 0.0014 although
  the rigs put sub at +67-74 % of all's excess - production's dvp noise is likely of that order).

## 5. Not done / caveats

* **Kit**: `GLM53_DENSE_W8A8_SKIP_LAYERS` must be forwarded by start.sh to BOTH ranks (a filter on one rank only = a
  half-served pair, same rule as ONLY) - add it next to `GLM53_DENSE_W8A8_ONLY` in make_start_sh's head `-e` list,
  worker `serve_env_names`, the validation block (`grep -qF GLM53_DENSE_W8A8_SKIP_LAYERS overlay/fp8_w8a8.py` + a
  syntax check), env.r16 `#knob`, boot_checks "equal on both ranks" + the `skip_layers_at_load` / `selftest_passed`
  count in the load summary. Not done here because the next kit stage name belongs to the r16z3+ branches.
* The rigs are 10-layer minis with repeated real layers (layers 0, 1, 10-13, 45 of the real model exist on nodeC) and
  1/4 of the routed experts; absolute KLs are not production's (dvp_off 0.0008-0.002 vs production 0.0195 with
  e4m3), only shares transfer, and even those depend on the layer composition (rig O vs P per-layer tables differ).
* Decode here is M=1 without the drafter; production decodes M=K+1 verify steps (never W8A8 either).
* The production numbers in section 4 are extrapolations; the A/B decides.
