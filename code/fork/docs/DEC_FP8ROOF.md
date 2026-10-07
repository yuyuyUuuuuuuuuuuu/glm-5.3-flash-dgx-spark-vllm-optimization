# DEC_FP8ROOF — decode FP8 GEMVs past the DRAM roofline (`GLM53_DEC_FP8ROOF`)

Status 2026-09-28, branch `dec-fp8roof` (from `deploy-r15` 45047ef). Built and measured on nodeC only (one GB10,
production image, `tests/gpu_run.sh`, every GPU run under `flock /tmp/tf-gpu-bench.lock`). Production (nodeA/nodeB)
was not accessed. Every number below is in a log under `docs/logs/dec_fp8roof/` (commands: `COMMANDS.txt` there).

## 1. Why tiling is not the lever: in isolation every FP8 shape is already at the roofline

The stream started from "KDA in_proj (12608 × 4096) runs at 205 GB/s because 99 blocks on 48 SMs quantize badly".
Measured, that is not what limits it:

| shape (N × K), M = 5 | production config | isolated, cold weights, clean L2 (`docs/logs/fp8_gemv/bench_final.log`) | in situ, R15 trace (rank 0) |
|---|---|---|---|
| KDA in_proj 12576 × 4096 (Npad 12608), grid 99 | (8,4,2,·) | **209.5 µs = 246 GB/s** | 251 µs = 205 GB/s |
| KDA o_proj 4096 × 4096 | (16,8,2,T) | 69.5 µs = 241 GB/s | ~80 µs |
| MLA o_proj 4096 × 8192 | (8,8,2,·) | 144.1 µs = 233 GB/s | 146–156 µs |
| shared gate_up / MLA qkv_a 2048 × 4096 | (16,16,2,T) | 36.0 µs = 233 GB/s | 50–60 / 41 µs |
| shared down 4096 × 1024 | (16,8,2,T) | 18.7 µs = 225 GB/s | 26–31 µs |
| MLA q_b 8192 × 1536 | (16,4,2,T) | 52.9 µs = 238 GB/s | 56–59 µs |

With 99 blocks the in_proj kernel reaches 246 GB/s of a ~250 GB/s read ceiling: there is no wave-quantization loss
to recover. The in-situ gap comes from the environment of the call, reproduced on nodeC:

- **Dirty L2 lines.** `explore/probe_d{0,8,16}.log`: the same in_proj call after 0 / 8 / 16 MiB of freshly written
  lines (the KDA recurrent kernel writes per-token states right before) takes 220.7 / 255.5 / 271.8 µs with plain
  loads; the evict_first configs are more robust (219.3 / 221.2 / 232.4 µs), and production's config for M = 5..8
  already is one of them. `explore/sim_d*.log`: at the level of a whole layer the eviction policy of in_proj changes
  nothing (the write-backs are paid by the next kernel instead).
- **Ramp after a latency-bound window**, and the shared expert running next to the routed experts on the aux stream.
- Other tilings of in_proj under the same conditions ((8,2,2), (8,1,2), (8,8,2), (16,4,2,T), (16,16,2,T),
  `explore/probe1.log`, `probe_d*.log`) are all slower than or equal to production's; split-K / stream-K cannot help
  a kernel that is already at 98 % of the DRAM ceiling when it runs alone.

(The prefetch kernel itself, R.6: a 4096 × 4096 GEMV at M = 5 takes 75.2 µs on cold weights and 16.5 µs right after
its whole weight was prefetched.)

What *is* left: ~16 ms/step of non-bandwidth time (all-reduce, mHC, norms, KDA recurrent, MLA attention, router,
sampler, host loop) during which DRAM is nearly idle, directly before bandwidth-bound GEMVs whose weights are known in
advance. `GLM53_DEC_FP8ROOF` streams the start of the **next** weight into L2 (24 MiB on GB10) during those windows, so
the GEMV reads that part from L2 — faster than the DRAM roofline allows.

## 2. What it does (`fp8_roof.py`, `kernels/fp8_roof.cu`, hooks in `fp8_gemv.py`)

Inert unless `GLM53_DEC_FP8ROOF=1` (read once at plugin load, every vLLM process) **and** the fp8_gemv small-M path
is installed (`GLM53_FP8_GEMV=1`, production has it). Installed from `integrate.plugin_register` right after
`fp8_gemv.plugin_register`. Two independent parts:

**(1) Table** (`GLM53_DEC_FP8ROOF_TABLE`, default on): `fp8_gemv.TABLE[(4096, 20480)]` = the drafter fc
(`GLM53_DRAFT_FP8=layers,fc`; production: Marlin, 398 µs/step in R15), MB 1 → (16,8,2,T), MB 2 → (8,8,2,T). No existing
entry changes. Served like every fp8_gemv shape (per-layer self-test vs Marlin, Marlin on any decline).

**(2) L2 prefetch** (`GLM53_DEC_FP8ROOF_PF`, default `all`; a comma list of triggers; `off`). A tiny kernel
(`l2_prefetch`, 8 CTAs × 256 threads, `ld.global.cg` 16-byte loads whose values are discarded; no workspace, no
stores) runs on a side stream forked by an event right after a known predecessor's kernels are enqueued, so it starts
when the predecessor finishes and runs through the latency-bound window:

| trigger | predecessor (hook) | prefetched (default budget) | window in production (R15 trace, median) |
|---|---|---|---|
| t1 | KDA in_proj(L) | KDA o_proj(L), 12 MiB of 16.8 MB | f_b, g_b, conv, recurrent: 57 µs |
| t2 | KDA / MLA o_proj(L) (dense: → mlp.gate_up) | shared gate_up(L) + down(L), all 12.6 MB (dense: 16 MiB) | AR, mHC, router: 61–63 µs |
| t3 | routed MoE(L) (`Exl3MoEMethod.apply`), dense down_proj(L) | layer L+1's first linear: KDA in_proj 12 MiB, or MLA fused_qkv_a (8.4 MB) + 4 MiB of q_b | epilogue, AR, mHC: 56–58 µs |
| t4 | MLA q_b(L) | MLA o_proj(L), 16 MiB of 33.5 MB | MLA attention: 295 µs |
| t5 | target lm_head (eager, 2nd lm_head call since the last fc) | drafter fc, 16 MiB of 84 MB | all-gather, sampler, rejection, host: 197 µs |
| t0 | any other lm_head call (the drafter's) | target layer 0's in_proj, 12 MiB | drafter logits, host step loop, embedding AR: ~2 ms |

Join: **late** — right after the successor's own kernel is enqueued, its stream waits for the prefetch's completion
event, so the successor never waits for the prefetch and every forked stream rejoins before a graph capture ends
(vLLM's breakable PIECEWISE graphs end a graph at every eager break: t1 / t4 are not forked there, section 12).
t1–t4 fork during capture and eager alike (they are part of the target's decode CUDA graphs); t0 / t5 are eager-only
(the lm_head and the fc run outside the graphs) and never fork, join or count inside a capture. Safety nets: a pending
prefetch whose target layer is below the current one is joined; pending prefetches of an earlier forward or of the
other capture state (eager vs capturing) are dropped, never joined across a capture boundary; rows > 64
(`GLM53_DEC_FP8ROOF_PF_MAX_M`) never fork; torch.compile'd calls (the drafter's layers) never fork. Knobs:
`_PF_MIB` ("t0=12,t1=12,t2=16,t3=12,t4=16,t5=16"), `_PF_CTAS` (8), `_PF_POL` (0 normal, 1 evict_last). Any bad knob →
WARNING, prefetch off, table still on. Any exception in a hook → WARNING once, prefetch off for the process, and every
pending prefetch of the current capture state is rejoined into the current stream first (`_rejoin_all`), so a non-CUDA
error raised during a CUDA-graph capture (e.g. a TORCH_CHECK of `l2_prefetch`) leaves the capture valid instead of
failing it with `cudaErrorStreamCaptureUnjoined` (review fix, R.12; a CUDA / capture error is still re-raised: it
cannot be recovered inside a capture). Decisions depend only on shapes, layer
structure and call order → identical on both TP ranks; no collective is touched.

## 3. Measured (nodeC, paired, CUDA graphs)

### 3.1 Per call, inside the decode graph (`sim_final_m5.log`)

`tests/roof/sim_step.py` builds production's decode step through the REAL integration (the launcher overlay's
`Glm53DenseFp8Method` at every production shape / prefix / config, fp8_gemv wrappers, the roof hooks; a stand-in
`Exl3MoEMethod` wrapped the same way) and replaces the latency-bound work by stand-ins with the R15 trace's median
durations: attention pre-block (AR 25 + mHC 12 + 6 µs), KDA conv 7 / recurrent 24 µs + 10 MiB of per-token state
writes / norm 3, MLA attention 150 µs + 16 MiB of reads, MoE block AR + mHC + router (2.4 MB) + topk + rot_in, routed
experts = 3 × 51 MB evict_first FP8 GEMVs, shared expert on an aux stream (vLLM runner order). 19 layers (3 dense,
then MLA / KDA / KDA / KDA), M = 5. Per-call kernel times from CUPTI over 6 paired rounds:

| role | calls/step prod | MB | R15 in situ µs | sim off µs (GB/s) | sim on µs (GB/s) | saving µs/call |
|---|---|---|---|---|---|---|
| KDA in_proj | 34 | 51.6 | 251 | 251.5 (205) | **181.8 (284)** | 69.7 |
| KDA o_proj | 34 | 16.8 | ~80 | 73.7 (228) | **45.4 (370)** | 28.3 |
| MLA fused_qkv_a | 11 | 8.4 | 41 | 36.1 (232) | **9.0 (930)** | 27.1 |
| MLA q_b | 11 | 12.6 | 56–59 | 52.8 (238) | **39.3 (320)** | 13.5 |
| MLA o_proj | 11 | 33.6 | 146–156 | 154.9 (217) | **94.6 (355)** | 60.3 |
| dense gate_up | 3 | 50.3 | 212–220 | 212.5 (237) | **167.8 (300)** | 44.7 |
| dense down | 3 | 25.2 | 113–116 | 106.0 (237) | 106.3 (237) | — (not prefetched) |
| shared gate_up | 42 | 8.4 | 50–60 | 40.1 (209) | **15.7 (535)** | 24.4 |
| shared down | 42 | 4.2 | 26–31 | 56.1 (75)¹ | **6.9 (604)** | 49.2¹ (prod ≈ 20) |
| f_b / g_b | 68 | 0.5 | 4 | 4.0 | 5.7 / 4.7 | −1.7 / −0.7 |

¹ the simulator's shared down overlaps its routed stand-in on the other stream; production runs it between the routed
gate/up and down kernels (26–31 µs), so its production saving is ≈ 20 µs, not 49.

"Off" reproduces production's in-situ numbers (in_proj 251.5 vs 251 µs). The prefetch kernels: 76 per forward
(19 layers), 5.3 ms of side-stream time, all inside DRAM-idle windows or overlapping the start of the successor.

### 3.2 Per forward (`sim_final_m5.log`, 15 paired rounds, alternating order)

| config | ms / forward (19 layers) | saving | paired ratio median [min, max] |
|---|---|---|---|
| off (production) | 22.451 | — | 1 |
| **all** (this branch) | **20.477** | **1.973 ms = 8.8 % = 104 µs/layer** | 0.912 [0.904, 0.932] |
| v1 (first session's plan: no dense-down t3, no q_b bytes) | 20.563 | 1.888 ms | 0.916 [0.910, 0.928] |

Outputs of every configuration are bitwise equal to off.

Per trigger (`sim_ablation_m5.log`, 15 paired rounds; a first attempt ran while another GPU user was active,
spread 14–18 %, and was repeated): t1 alone 0.533 ms (ratio 0.977), t2 0.620 (0.969), t3 1.090 (0.956), t4 0.167
(0.990), all 2.088 ms (0.910) — nearly additive (sum 2.41). Robustness (`sim_m{1,8,16}_d10.log`,
`sim_m5_d{0,16}.log`, 13 rounds each), paired ratio all/off, median [min, max]:

| M (rows) | 1 | 5 | 8 | 16 |
|---|---|---|---|---|
| D = 10 MiB state writes | 0.914 [0.816, 0.971]² | 0.912 [0.904, 0.932] | 0.912 [0.896, 0.935] | 0.920 [0.906, 0.935] |

| dirty state writes per KDA layer (M = 5) | 0 MiB | 10 MiB | 16 MiB |
|---|---|---|---|
| paired ratio | 0.918 [0.888, 0.932] | 0.912 [0.904, 0.932] | 0.920 [0.904, 0.931] |

² that run's off spread was 24 % (another GPU user); the median is still in line.

Re-tune on this simulator (`sim_tune_m5.log`): t3 = 16 MiB 0.909, t1 = 16 MiB 0.916, t1/t3/t4 = 16/16/24 0.915,
12 CTAs 0.918, 6 CTAs 0.914, evict_last 0.907, defaults 0.911 — all within the run's noise (spread 3–7 %), so the
defaults stay (8 CTAs, normal priority, t0 12 / t1 12 / t2 16 / t3 12 / t4 16 / t5 16 MiB; the first session's CTA
sweep on the earlier simulator: 8 > 6, 12 > 4, 32, `explore/step6.log`).

### 3.3 Eager anchors t0 / t5 (`sim_loop_m5.log`)

`tests/roof/sim_loop.py` runs production's per-step eager loop around a captured target graph: drafter fc (real
integration, fp8_gemv) → 5 drafter layers (plain GEMVs) → drafter lm_head (77440 × 4096 FP8, built like
`glm53_runtime.convert_lm_head_fp8`, i.e. never `register()`ed) → all-gather / top-k / 1 ms host gap → target graph
(embedding AR, layer-0 in_proj, 8 × 50 MB GEMVs) → target lm_head → all-gather + sampler. CUPTI per call, M = 5:

| call | off µs (GB/s) | t0 + t5 µs (GB/s) | saving |
|---|---|---|---|
| drafter fc 4096 × 20480 (t5) | 360.7 (233) | **316.0 (265)** | 44.6 µs |
| target layer-0 in_proj (t0) | 209.2 (247) | **167.5 (308)** | 41.7 µs |
| drafter / target lm_head | 1329.9 / 1343.6 | 1344.9 / 1337.8 | noise |

The per-step wall time of this eager loop is too noisy on a shared nodeC to resolve 0.09 ms (spread 15–30 %); the
per-call CUPTI numbers above are the measurement.

### 3.4 Drafter fc table entry (`bench_fc.log`, 2 cold copies, 9 paired rounds)

| M | production Marlin µs (GB/s) | fp8_gemv µs (GB/s) | Marlin / new, per round |
|---|---|---|---|
| 1 | 377.4 (222) | 354.1 (237) | 1.070 |
| 5 | 380.8 (220) | 353.7 (237) | 1.076 |
| 6 | 385.1 (218) | 359.2 (234) | 1.074 |
| 8 | 380.5 (220) | 353.1 (238) | 1.065 |
| 16 | 389.0 (216) | 359.5 (233) | 1.074 |

## 4. Projected saving per decode step (R15 call counts, rank 0)

FP8 kernel time saved per step = Σ calls × saving/call (3.1; shared down at its production ≈ 20 µs; layer 0's in_proj
via t0, 3.3; fc: table ≈ 27 µs (3.4) + t5 45 µs):
in_proj 33 × 69.7 + 41.7 = 2.34 ms, KDA o 34 × 28.3 = 0.96, MLA qkv_a 11 × 27.1 = 0.30, q_b 11 × 13.5 = 0.15, MLA o
11 × 60.3 = 0.66, dense gate_up 3 × 44.7 = 0.13, shared gate_up 42 × 24.4 = 1.02, shared down 42 × 20 = 0.84, fc 0.07,
f_b / g_b 68 × −1.2 = −0.08 → **≈ 6.4 ms/step of FP8 kernel time**. In the simulator the forward got 58 % of its
kernel-time saving back as wall time (1.97 of 3.37 ms: a prefetch longer than its window overlaps the start of its
successor, and part of every window is spent in kernels that touch DRAM too). Applying that ratio: **≈ 3.7 ms/step
(R15 69.1 → ~65.4 ms, −5.4 %; bench_decode prose 73.6 ms/step, 40.6 tok/s → ~42.7 tok/s)**. Scaling the simulator's
104–110 µs/layer to 45 layers instead gives 4.7–4.9 ms. Stated range **3–4.7 ms/step**; only a production A/B can pin
it (section 8).

## 5. Numerics

- **Prefetch (t0–t5): bitwise identical.** It only reads weights. Checked: every simulator configuration's forward
  output == off (bitwise), `tests/test_fp8_roof.py` R.8 (eager, CUDA graph at M = 1, 5, 8, 16, 64, replay vs eager,
  every trigger subset) and R.10 (eager anchors), R.6 (the prefetched weight is bitwise unchanged).
- **Drafter fc on fp8_gemv: not bitwise vs Marlin** (as for every shape fp8_gemv serves). Real DFlash2 `fc.weight`,
  heavy-tailed activations, M = 1..16 (R.5): new vs Marlin rel_l2 1.3e-5 … 1.1e-4, max-abs 0.125 … 2 (bitwise equal
  on 99.74–99.87 % of outputs); vs float64 both kernels have rel_l2 1.62e-3 … 1.66e-3 (bf16 output rounding) and
  err/allowed ≤ 0.997; correctly rounded 99.78–99.87 % (new) vs 99.71–99.83 % (Marlin). It feeds only the drafter:
  it can move proposals / acceptance, never the target's distribution. `GLM53_DEC_FP8ROOF_TABLE=0` keeps Marlin.

## 6. Memory

No allocation: one side stream per device and CUDA events (per fork). Registry = references to existing weights.
Each captured decode graph gains ~145 prefetch kernel nodes (45 layers) + event edges. **Extra GPU memory per rank:
0 MiB** (R.8 / R.10 peak CUDA allocation is the test's own tensors).

## 7. Enable (both ranks) / revert

Ship in the TF bundle (`overlay/tf/site/`): `fp8_roof.py`, `tf_fp8_roof_ext.cpython-312-aarch64-linux-gnu.so`
(`python3 setup.py build_ext --inplace` in the image; `tests/run_all.sh` build step), and this branch's
`fp8_gemv.py` and `integrate.py`. Then set on **both** ranks:

    GLM53_DEC_FP8ROOF=1

(the defaults are the tuned ones; `GLM53_FP8_GEMV=1` must stay on). Either add `GLM53_DEC_FP8ROOF` to the head's
`docker run -e` list and to the worker's `serve_env_names` loop in `start.sh` (next to `GLM53_FP8_GEMV`), or pass it
through `GLM53_EXTRA_ENV="GLM53_DEC_FP8ROOF=1"` (forwarded to both ranks). Log lines to check on both ranks:
`tf_fp8_roof installed (GLM53_DEC_FP8ROOF): table additions ['4096x20480']; L2 prefetch triggers ['t0', 't1', 't2',
't3', 't4', 't5'] …` and no `tf_fp8_roof: L2 prefetch disabled` / `were never joined` WARNING.
Partial: `GLM53_DEC_FP8ROOF_PF=off` (table only), `GLM53_DEC_FP8ROOF_TABLE=0` (prefetch only).
Revert: unset `GLM53_DEC_FP8ROOF` on both ranks and restart (nothing is patched when it is unset: R.1).

## 8. What production must measure, risks

- A/B decode (prose, coding, structured) `GLM53_DEC_FP8ROOF=1` vs unset, both ranks; a torch.profiler step like R15:
  per-call in_proj / o_proj / shared gate_up / down / MLA o / qkv_a / q_b, and the `l2_prefetch_kernel` launches per
  step (forks in the target graph: 34 t1 + 45 t2 + 44 t3 + 11 t4, 145 kernels since a shared-expert t2 and an MLA t3
  launch two; plus t0 and t5 eagerly).
- The simulator's windows are stand-ins (spin kernels for AR / mHC / recurrent …). Production's AR goes over RoCE and
  ~2.1 ms/step of it is waiting for the other rank; both ranks prefetch identically, so the skew should not move, but
  that is exactly what the A/B must confirm.
- If a window is shorter than its prefetch the tail overlaps the successor (late join): it then only competes with
  the successor's own weight stream (no stall; worst case ≈ neutral, simulator: never slower per call except the
  4 µs f_b / g_b calls, −1.7 / −0.7 µs).
- A CUDA / capture error inside a fork cannot be recovered inside the capture (the hook re-raises it); every input of
  the launch is validated at install / register time, and R.8b / R.10 capture at M = 1..64. A non-CUDA error during a
  capture turns the prefetch off and rejoins every pending side stream (R.12: the capture succeeds and replays
  bitwise; before the review fix it failed with `cudaErrorStreamCaptureUnjoined`, i.e. vLLM's start-up).
- The KL probe is not needed for the prefetch (bitwise); the fc table entry changes drafter numerics only
  (acceptance-rate check).

## 9. Not done here

- The drafter's own FP8 layers (qkv / o / gate_up / down, ~2.1 ms/step) are torch.compile'd into piecewise graphs;
  a fork in one piece joined in the next would fail the capture, so they get no prefetch (would need the split points
  of the drafter's compiled graph; windows AR + conv ≈ 30 µs → ≤ ~0.3 ms/step).
- PDL (programmatic dependent launch) for the small GEMVs: subsumed — the prefetch starts the weight stream a whole
  window earlier than PDL could.

## 10. Tests and tools

- `tests/test_fp8_roof.py` (overlay + real drafter fc mounted; 110 checks): R.1 inert, R.2 needs fp8_gemv, R.3 env
  parsing, R.4 install, R.5 drafter fc numerics, R.6 prefetch kernel, R.7 roles, R.8 mini model (eager / graphs /
  subsets / stale), R.9 torch.compile, R.10 eager anchors, R.11 uninstall, R.12 launch error inside a capture (review).
  Regression with this branch (AOT build, overlay bound, `reg_*.log`; before the review fix): `test_fp8_roof` 101/101, `test_fp8_gemv` 711/711, `test_fp8_integrate` 383/383,
  `test_fp8_large_m_integrate` 152/152, `test_integrate` 75/75. `tests/run_all.sh` builds and imports
  `tf_fp8_roof_ext` and runs `test_fp8_roof` when `GPU_RUN_BIND` is set.
- `tests/r16/test_r16_breakable.py` + `tests/r16/bk_rig.py` (section 12): the triggers under vLLM's breakable
  PIECEWISE capture on vLLM's own Glm5Next forwards; `R16_BK_EXPECT=before` reproduces the first R16 boot's failure with
  the 0501c3e modules; `R16_BK_TIMING=1` adds PIECEWISE vs FULL timing. Step `r16_breakable` of `tests/run_all.sh`.
- `tests/roof/sim_step.py` (decode-step simulator, `ROOF_SIM_PROFILE` for per-call CUPTI, `ROOF_SIM_STANDIN=spin|chase|mix`
  for what the latency-bound windows are made of, section 11), `sim_loop.py` (eager loop),
  `bench_fc.py`, `probe_copies.py`, `probe_prefetch.py`, `sim_layer.py`, `sim_prefetch.py` (exploration).

## 11. Adversarial review (2026-09-28, nodeC; logs `docs/logs/dec_fp8roof/review/`, commands in its `COMMANDS.txt`)

**Reproduced.** AOT build + `test_fp8_roof` 101/101, `test_fp8_gemv` 711/711, `test_fp8_integrate` 383/383,
`test_fp8_large_m_integrate` 152/152, `test_integrate` 75/75 (`reg_*.log`). Simulator, M = 5, spin stand-ins
(`sim_orig_m5.log`): off 22.520 → all 20.696 ms per 19-layer forward, saving 1.825 ms = 96 µs/layer, paired ratio
0.917 [0.904, 0.923], outputs bitwise equal; per call in_proj 252.6 → 178.5, KDA o 73.6 → 46.0, qkv_a 35.6 → 8.8,
q_b 52.7 → 39.3, MLA o 153.1 → 96.3, dense gate_up 210.2 → 166.4, shared gate_up 41.2 → 16.2 µs (all within a few µs
of section 3.1). Drafter fc table (`bench_fc.log`, 9 rounds): Marlin / new 1.04–1.08 (M = 1 / 5 / 8 / 16: 375 → 358,
372 → 358, 384 → 360, 374 → 376 µs with the MB 2 config (8,8,2,T), 361 µs with (16,8,2,T)); single rounds dip below 1
on this shared node, so treat it as ≈ 15–25 µs/step.

**Where the simulator is optimistic: its windows are `clock64` spins, blind to DRAM contention.** Run alone vs next to
a 12 MiB prefetch started at the same time (`rev_micro.log`): the 2.4 MB router GEMV 8.0 → 20.1 µs, a 0.5 MB FP8 GEMV
(f_b-sized) 6.0 → 9.1 µs, 4 L2-resident rms_norm kernels 24.4 → 28.1 µs. Latency-bound work that runs while a prefetch
streams gets slower. `ROOF_SIM_STANDIN=chase` (now an option of `tests/roof/sim_step.py`) replaces every spin by
dependent loads over 256 MiB (DRAM-latency-bound, slowed by the prefetch: the pessimistic end), `mix` = half/half:

| M (rows), stand-ins | spin | mix | chase |
|---|---|---|---|
| 1 | — | — | 78.0 µs/layer, ratio 0.935 [0.911, 0.955] |
| 5 | 96.0, 0.917 [0.904, 0.923] | 83.0, 0.930 [0.923, 0.944] | 71.5, 0.941 [0.925, 0.976] |
| 8 | (section 3.2: 0.912) | — | 64.8, 0.948 [0.933, 0.954] |
| 32 (4 requests × 8; FP8 linears on Marlin) | 64.9, 0.954 [0.934, 0.960] | — | 36.8, 0.975 [0.959, 0.992] |

Still a gain in every configuration: the largest paired ratio of any round is 0.992 (M = 32, chase). Long-context MLA window
(`sim_a64/a128.log`, A = 64 / 128 MiB of reads in the attention window, ~ an indexer K cache at 0.5 / 1 M tokens):
all 102.1 / 86.4 µs/layer, without t4 95.0 / 91.4: t4 turns neutral-to-slightly-negative at ~1 M-token contexts
(≤ ~0.06 ms/step over 11 MLA layers; `GLM53_DEC_FP8ROOF_PF=t0,t1,t2,t3,t5` drops it if long-context A/B shows it).

**Projection (R15 call counts, single request, 45 layers):** 45 × 65–96 µs = **2.9–4.3 ms/step**, central (M = 5,
mix) ≈ 3.7 ms, conservative (M = 5, chase) ≈ 3.2 ms; with 4 concurrent requests (M = 32) 1.7–2.9 ms. t0 / t5
(`sim_loop_m5.log`): per call fc 357 → 315 µs, layer-0 in_proj 212 → 169 µs, but the loop's wall time moved 22 µs,
ratio 0.995 [0.984, 1.011]: not resolved; their Python costs +19 µs of host time per lm_head call (fork + join,
`rev_micro.log` (2): 25.7 → 45.0 µs/call), normally hidden behind the 1.35 ms lm_head kernel. Not modelled here and
the first thing the production A/B must show: NCCL all-reduce over RoCE next to a running prefetch, and the
cross-rank skew (~2.1 ms/step of AR waiting in R15).

**Numerics, independently (`rev_micro.log` (3)).** Drafter fc on the real DFlash2 `fc.weight`, M = 1, 3, 5, …, 15, 16:
Gaussian, massive-activation (12 channels at ~3000), non-contiguous (a column slice, row stride 20736) and half-zero
inputs: every call served by the new kernel, err/allowed vs float64 ≤ 1.000 (Marlin: up to 1.020), rel_l2 vs float64
1.63–1.68e-3 for both, new vs Marlin rel_l2 ≤ 1.5e-4, correctly rounded as often as Marlin (±0.1 %). Prefetch outputs
bitwise equal in every simulator run above.

**Fixed (this commit): a non-CUDA error in a hook during a CUDA-graph capture failed the capture.** `_fork` recorded
the failed fork's completion event but never made it pending, and `_disable` turned the prefetch off without joining
the pending side streams, so the capture ended with `cudaErrorStreamCaptureUnjoined` (= vLLM start-up failure on that
rank) instead of falling back. R.12 injects a `RuntimeError` at the 1st / 7th / 15th launch of a capture: before the fix
2 of 3 captures fail (`r12_orig.log`, 106/108), after it all succeed, prefetch off, replay bitwise == eager
(`r12_fixed.log`, 110/110). No change on the normal path (same operations, same order).

## 12. Breakable CUDA graphs: the first deploy-r16 boot (branch `sidestream`, 2026-09-29; logs `docs/logs/sidestream/`)

**What happened.** The first R16 boot (nodeA / nodeB) died in `Capturing CUDA graphs (PIECEWISE): 0/23`:
`breakable_cudagraph.py:383 _capture -> ... Glm5NextDecoderLayer.forward -> kda.py forward -> self._forward ->
breakable_cudagraph.py:115 wrapper -> add_eager -> _end_segment -> capture_end`: `CUDA error: capturing stream has
unjoined work`, then PyTorch's `markCaptureEnd called with no captures in progress` (the capture's `__exit__` ending the
same segment again). Production has run the R16 bundle since without `GLM53_DEC_FP8ROOF` and `GLM53_DEC_MOEGLUE_WARM`.

**Why nodeC did not see it.** vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for `Glm5Next*`
(`vllm/config/vllm.py`, boot line `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1`). Its PIECEWISE graphs are then not
torch.compile pieces but one `BreakableCUDAGraphCapture` per batch size: a chain of separate CUDA graphs ("segments")
cut at every `@eager_break_during_capture` op, which runs eagerly between two graphs at capture and at every replay.
In this model: the KDA core `Glm5NextLinearAttention._forward` (after in_proj, f_b, g_b; before o_norm, o_proj), the
sparse-MLA indexer `sparse_attn_indexer_kpool` and the MLA attention `unified_mla_attention_with_output` (both after
q_b, before o_proj). A stream forked in a segment must rejoin before that segment's `capture_end`. Every nodeC test
(R.8, `tests/roof/sim_step.py`, `r16_decode_combo`) captured the whole forward in ONE `torch.cuda.graph`, which is the
FULL regime, where no break exists.

**Which triggers cross a break** (`tests/r16/test_r16_breakable.py`, `breakable_before.log` 44/44: the kit's 0501c3e
modules bound over the tree, PIECEWISE M = 8 through vLLM's own `BreakableCUDAGraphWrapper`, one trigger at a time):
t1 (in_proj → KDA o_proj, across the KDA core) fails with production's exact traceback and the pending fork
`('kda_o', 0)`, also with production's own (quickwins-recompiled, decorated) `_forward`, and the same with quickwins off
(vLLM's stock Glm5NextModel / DecoderLayer / KDA `_forward`, production's state since 2026-09-29):
`breakable_before_traceback.log` is that case's full output: `breakable_cudagraph.py:383 _capture` → `model.py forward`
→ `model.py:478 forward` (`self.self_attn`) → `kda.py:363 forward` → `breakable_cudagraph.py:115 wrapper` → `:203
add_eager` → `:189 _end_segment` → `capture_end`: `torch.AcceleratorError: CUDA error: capturing stream has unjoined
work`, then during `__exit__` `:169` → `:189` → `RuntimeError: num_active_captures_ > 0 INTERNAL ASSERT FAILED ...
markCaptureEnd called with no captures in progress` (production's boot, with quickwins on, showed the recompiled
DecoderLayer frame instead of `model.py:478`); t4 (q_b → MLA o_proj, across
indexer + attention) fails with `('mla_o', 3)` pending; t2 (o_proj → shared expert / dense gate_up: all-reduce, mHC,
router between them) and t3 (routed MoE / dense down → the next layer's first linear: the KDA in_proj runs BEFORE the
KDA core, the MLA fused_qkv_a before the indexer) capture and replay bitwise; the FULL capture of the same stack with
every trigger passes. t0 / t5 never fork inside a capture. So the prefetch OF the KDA in_proj (t3) is not behind a
break and is kept; what crosses is the prefetch forked AT the in_proj (t1, for the o_proj) and t4.

**Fix** (`fp8_roof.py`; eager calls, FULL graphs and a breakable capture in FULL runtime mode are unchanged):
- inside a breakable segment (vLLM's own rule: a `BreakableCUDAGraphCapture` of this thread is capturing and the
  forward context is not FULL) t1 and t4 (`BREAK_CROSS`) are not forked (counters `bk_skipped_t1` / `_t4`);
- guard: `BreakableCUDAGraphCapture.add_eager` / `_end_segment` / `_begin_segment` are wrapped (idempotent; the same
  kind of wrapper as moeglue warm's, both in the chain; inert while nothing is pending). A prefetch still pending when a
  segment ends is joined into it on the segment's capture stream. At a break that means a structure `BREAK_CROSS` does
  not name: WARNING once, and that trigger is no longer forked inside breakable segments (`joined_at_break_<t>`); at the
  end of a capture (normal or raising) it is silent (`joined_at_capture_end`). No capture can fail on an unjoined
  prefetch, and one that fails for another reason neither leaves a stale fork behind nor has its error replaced by an
  unjoined-stream error (without the guard, `breakable_before.log` inject cases: the injected error ends as
  `capturing stream has unjoined work` and the fork stays pending). `_rejoin_all` / the stale-fork rules of R.12 are
  unchanged (`test_fp8_roof` 110/110).
- the guard not installable (a vLLM without these methods) → `L2 prefetch NOT enabled: breakable CUDA graphs cannot be
  guarded` (WARNING, prefetch off, table on);
- install line: `...; breakable CUDA graphs: segment guard hooked, not forked inside a segment: ['t1', 't4']`.

Proof (`breakable_after.log`, 44/44; the same file in `run_on/` and `run_off/` via `tests/run_all.sh`): PIECEWISE M =
1 … 64 capture with t2 8 / t3 7 / warm 5 forks and t1 / t4 skipped, nothing joined at a break or the capture end,
replays bitwise equal to eager with both features off (fresh inputs, counters frozen); FULL M = 1 … 64 after them
(production's order) with every trigger, bitwise; production's own KDA `_forward`; A.9 the same with quickwins off
(stock forwards: PIECEWISE M 1 / 8 / 64, FULL M 8 / 64, warm / t1 / t4 alone, the stock KDA `_forward`); the guard with
the static skip emptied (t1 / t4 joined at the first break, learned, WARNING, bitwise); a Python error injected
mid-capture with forks / a warm pending (the real error surfaces, nothing pending afterwards, the next capture replays
bitwise). Plugin census (`r16_plugins`, `r16_loo_census`): the prefetch counts as installed only with `bk_guard
hooked` and `bk_skip ['t1', 't4']`, in every leave-one-out / one-alone state, in production's state of 2026-09-29 plus
the update (smallops + fp8roof + moeglue warm) and in fp8roof + moeglue alone.

**Which graphs production replays.** From the image's vLLM (`vllm/v1/worker/gpu/cudagraph_utils.py`, V2 model runner)
and production's launcher: `cudagraph_mode` = FULL_AND_PIECEWISE (the default; breakable keeps PIECEWISE); 23 capture
sizes = start.sh `configure_capture_sizes` for DFLASH_TOKENS 7, GLM53_ADAPTIVE_K_SET 4,5,7 and MAX_NUM_SEQS 8
({1, 2, 4, 8, 16, 24, 32} ∪ {5, 6, 8} × 1…8: exactly the boot's "0/23"; with 4 seqs it would be 14); the launcher's
`patch_adaptive_k.py` adds uniform-decode FULL graphs for query lengths 5, 6, 8 → 24 FULL graphs (≤ 8 requests, ≤ 64
tokens). Adaptive-K keeps decode / verify batches uniform, so every decode step dispatches to a FULL graph: one CUDA
graph per step with every trigger, unchanged by this fix. The 23 breakable PIECEWISE graphs serve mixed / non-uniform
batches of ≤ 64 tokens (e.g. the last chunk of a prefill next to decodes); larger batches run eagerly (rows > 64: no
fork). Boot lines to confirm on nodeA: `[glm53-adaptive-k] uniform decode graph query lens: [5, 6, 8]`,
`Capturing CUDA graphs (PIECEWISE): ... 23/23`, then `Capturing CUDA graphs (FULL): ... 24/24`.

**What remains.** Decode (FULL graphs): everything of sections 3–4 (3–4.7 ms/step projected), untouched: the fix
changes nothing there (same forks, bitwise, A.4). PIECEWISE steps (mixed batches ≤ 64 tokens) lose t1 and t4; t2, t3
and the warm stay. Measured on the breakable rig (`breakable_timing.log`: `test_r16_breakable.py --case
after:timing:noqw`, quickwins off as in production; 8 production-shaped layers = 6 KDA + 2 MLA, 3 dense + 5 MoE with
real EXL3 experts topk 8, clock spins at the R15 trace's medians for the latency-bound windows; per-forward replay time,
41 interleaved paired rounds, saving vs `off` of the same regime; null repeat `off2` −10 … +53 µs, p10–p90 of the paired
ratios ≈ ±0.6 %):

| µs saved per 8-layer forward | FULL M=5 | PW M=5 | FULL M=8 | PW M=8 | FULL M=16 | PW M=16 |
|---|---|---|---|---|---|---|
| fp8roof, all triggers (PIECEWISE: t1 / t4 not forked) | **586** (0.947) | **433** (0.962) | **574** (0.956) | **424** (0.968) | **404** (0.970) | **343** (0.974) |
| fp8roof without t1 / t4 (`roof_t23`) | 464 | 453 | 467 | 399 | 351 | 313 |
| t1 alone (not forked in PIECEWISE) | 187 | −10 | 156 | −23 | 51 | 8 |
| t2 alone | 231 | 171 | 178 | 141 | 97 | 98 |
| t3 alone | 323 | 315 | 290 | 289 | 239 | 218 |
| t4 alone (not forked in PIECEWISE) | −47 | −47 | −26 | −10 | −66 | −25 |
| fp8roof + moeglue warm | 618 | 469 | 591 | 390 | 409 | 316 |
| rejected alternative: t1 / t4 forked, joined at the break | – | 53 | – | −27 | – | −168 |

- What PIECEWISE gives up = FULL `roof` − FULL `roof_t23` = **122 / 107 / 53 µs** per forward at M = 5 / 8 / 16, all of
  it t1 (≈ 31 / 26 / 9 µs per KDA layer; t4 shows no gain on this rig, the simulator of section 3.2 gave it ≈ 42 µs per
  MLA layer). PIECEWISE keeps **74 % / 74 % / 85 %** of the FULL prefetch gain (PIECEWISE `roof` ≈ its `roof_t23`, as
  it must: t1 / t4 do not fork there). Scaled to production's 34 KDA + 11 MLA layers: a PIECEWISE step loses ≈ 0.9–1.05
  ms (t1; up to ≈ 1.5 ms if t4 is worth the simulator's 42 µs per MLA layer) of the prefetch gain it would have had.
- Joining t1 / t4 at the break instead of not forking them (a hook that joins every pending fork before
  `_end_segment`) is worse than skipping: 53 / −27 / −168 µs instead of 433 / 424 / 343. The eager break then waits for
  the side stream's L2 reads on the critical path. Skipping is the kept policy; the guard's join-at-break only covers an
  unexpected structure (WARNING, then that trigger is skipped too).
- Production decode steps replay FULL graphs (above): no loss. Only steps with a prefill chunk and ≤ 64 tokens in total
  replay a PIECEWISE graph (e.g. a follow-up turn whose uncached suffix is short); steps of > 64 tokens run eagerly and
  never forked (rows > 64).
