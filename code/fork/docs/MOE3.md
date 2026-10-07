# MOE3 — quality-preserving speedups for the routed-MoE prefill (branch moe3, from moe2 8396cca)

2026-10-02, nodeC only (one GB10, production image, every GPU run through `tests/gpu_run.sh` / `tests/handoff/run.sh`
under `flock /tmp/tf-gpu-bench.lock`). nodeA/nodeB untouched; no kit built or installed. Evidence (logs):
`${HOME}/tf-exl3-assets/moe3/` (copies of the small ones in `docs/logs/moe3/`).
Weights: the real GLM-5.3-Flash EXL3-TR3-4bpw layer-10 routed experts, TP=2 rank-0 shard (the only complete MoE layer
on nodeC), production module of the image with prefill cap 1 installed (as deployed), production's own arguments.

## 0. Verdict

| candidate | measured | quality | status |
|---|---|---|---|
| **(c) production arithmetic, faster** — `GLM53_MOE_FUSED16=1` | per MoE layer call at 13,824 tokens: **-7.1 .. -7.7 ms** (apply level 63.6 -> 56.6 ms; grouped part 61.8 -> 54.2); 12,288: -4.9; 8,192: -2.0; <= 4,289: ~0 | **h13 / h2 bit-identical to production**; output differs only in the order of the fp32 atomic adds (production's own nondeterminism class): bf16 output 0.0045 % of elements one bf16 ulp apart | **built, recommended A/B arm** |
| (a) layer-selective e4m3 — `GLM53_MOE_E4M3_LAYERS` | -15.7 ms per selected layer vs (c) (e4m3_d16: -11.2) | per-layer sensitivity is NOT measurable on nodeC (one real MoE layer); uniform-linear estimate from the production A/B: +0.00020 long-KL, +0.00025 dvp-KL per layer (d16 ~x0.67) -> the dvp gate (+0.0024 left) allows ~9 layers (d16 ~14) | knob built; layer choice needs a production group A/B (section 4) |
| (b) token-selective (fp16 for the last N tokens) | not built | cannot help either gate: quality_long scores EVERY prompt position, and the dvp probe's B side is a fresh prefill of prompt + all 1536 generated tokens (every scored position is a prefill position) | rejected by the measurement design |
| (c') the "atomics / DRAM floor" claim (moe2) | down-only kernel 24.4 ms with the fp32 scatter vs 18.1 without (p16b); production's fm_down 22.5 | — | **confirmed**: the out RMW (110,592 rows x 16 KB, ~3.6 GB/layer-call) is the down's floor; the 128-row fused e4m3 schedule does not transfer to fp16 operands (section 2.3) |

Projection (32k request = 2 x 13,824 + 4,289 tokens, from ~2,110 tok/s = 15.14 s): (c) -42 x 7.3 ms x 2 chunks
(+ ~0 for the tail) = **-0.61 s -> ~2,195 tok/s (+4 %) at zero KL cost** [E]. With (a) on 9 layers (e4m3) or 14 layers
(e4m3 + DOWN=f16) on top: a further ~-0.32 / -0.36 s -> ~2,245 tok/s [E], dvp KL then at ~0.0099 (the gate) under the
uniform-sensitivity assumption — measure before choosing (section 4).

## 1. What production's routed-MoE prefill spends (real layer-10 experts, T = 13,824, ms per call)

`tests/moe3/bench_p16.py` / dev probes: production fm_gather 5.3, fm_gateup 30.8, fm_down 22.5 (sum 58.6) + tables 0.2;
apply level 63.6 (+ thin kernel, zero-fill, casts, routing). Removal probes on the ported kernels:

| | ms | note |
|---|---|---|
| gate/up as production (ported, its double exp) | 31.0-31.6 | == production's 30.8 (faithful port) |
| ... with `exp_prod` (bit-identical, section 2.1) | 27.3-27.9 | the FP64 exp costs ~3.6 ms (more when not hidden by the second CTA: 9.5 ms in a 1-CTA kernel) |
| ... without the trellis decode (garbage weights) | 22.0 | decode ~6 ms; mma floor 1.855 TFLOP / 107.8 TFLOPS = 17.2 |
| down (p16b, down items only) / without the fp32 red.v4 | 24.4 / 18.1 | the scatter is ~6 ms exposed; mma floor 8.6 |
| gather | 5.3 | DRAM: 906 MB of fp16 rows written |

## 2. What was built (kernels/moe_e4m3.cu, same extension `glm53_moe_e4m3_ext`; overlay/glm53_moe_fused16.py)

### 2.1 `exp_prod` — the SiLU's `(float) exp((double) g)` bit for bit in fp32
exllamav3's `fm_gateup_kernel` computes the sigmoid with the double-precision exp (to be immune to `--use_fast_math`);
GB10's FP64 rate makes that ~10 % of the gate/up kernel. `exp_prod(a)`: a = (64 n + j) ln2/64 + r (3-part Cody-Waite,
exact first step by FMA), 2^n * T[j] * e^r in float-float (64-entry hi/lo table, degree-5 polynomial, relative error
< 2^-46); y = RN(y + yl) is returned unless the true value could lie on the other side of a rounding midpoint
(|half-ulp - |yl|| <= 2^-42 y) - those cases (1.4e-6 of all floats) and a outside (-87, 88) / NaN take the double
statement itself. **Exhaustively verified: every one of the 2,237,530,114 floats in [-87, 88] gives the same bits as
`(float) exp((double) a)` (`tests/moe3/exp_check.py`, `test_fused16.py` E).**

### 2.2 exllamav3's fat kernels ported (`pgu`, `pdn`, `gather16`)
Statement-for-statement ports of `fm_gather_kernel` / `fm_gateup_kernel` / `fm_down_kernel`
(`docs/ref/mia_exl3-fat-kernel/exl3_fat_moe.cu`) with this repo's primitives (TensorFold trellis decode, which yields the
same m16n8k16 B fragments, `mma.sync.m16n8k16.f32.f16.f16.f32`, the same k order, the same epilogues incl. the
fp16-rounded `__hmul2` suh products, `__fdiv_rn`, the fp16 route weight) + `exp_prod`. `gather16` visits the rows token
by token (x read once instead of once per routed expert; same arithmetic). `pgu` / `pdn` take a segment chunk so the
down of chunk c can run on a side stream next to the gate/up of chunk c+1 (`sep` schedule).

### 2.3 `p16b` — one persistent launch for gather + gate/up + down
2 CTAs x 8 warps per SM (exllamav3's CTA shape: one CTA's epilogue overlaps the other's mainloop), 64-row segments
(production's own `build_grouped_fat_tables`, tile 64), three item kinds from a global ticket counter: G(seg) gather,
U(seg, nb) gate/up of 128 intermediate columns (waits `gathered[seg]`), D(seg, nb) down of 256 hidden columns + red.v4
(waits `ready[seg]` = 8 U done). Order: rounds of [G(r), U(r - LG, 0..7), D(r - LG - LD, 0..15)] (LG 8, LD 24; the
result is insensitive to 4..64 / 8..160). An item only waits for smaller tickets and every CTA is resident.
Tried and dropped (all bit-identical, slower): the e4m3 kernel's 1-CTA 16-warp 128-row fused schedule on fp16
operands (57.2 ms with the double exp, 48.9 with exp_prod - fp16 mma needs 2x the instructions and 2x the A bytes per
FLOP, and one CTA per SM cannot hide its epilogue); a 64x32-per-warp tiling of it (59.0: decode 11.5 ms); decoding both
k16 sub-steps up front (-0.3 ms, noise).

### 2.4 Schedule choice (`SCHED`), measured (`bench_p16.log`, grouped part, real / collapsed routing)

| T | production | p16b | sep (2 chunks) | sep (1) | served by |
|---|---|---|---|---|---|
| 13,824 | 61.83 / 60.88 | **54.25 / 53.74** | 56.80 / 55.77 | 58.61 / 57.38 | p16b (T >= 11,264) |
| 13,856 (+32 decode tokens) | 61.99 / 60.99 | **54.16 / 53.73** | 57.04 / 56.04 | 58.64 / 57.74 | p16b |
| 12,288 | 54.10 / 53.34 | **49.23 / 49.23** | 51.12 / 50.59 | 52.19 / 51.36 | p16b |
| 8,192 | 37.44 / 36.79 | 36.10 / 34.63 | **35.40** / 35.07 | 35.68 / 34.95 | sep |
| 4,289 | 20.81 / 20.17 | 23.65 / 21.21 | 20.85 / 19.91 | **20.64 / 19.63** | sep |
| 2,048 / 1,791 | 13.26 / 12.25 | 17.06 / 16.26 | 13.33 / 12.53 | 13.02 / 12.39 | production (< 4,096) |

p16b is slower than production below ~10k tokens (fewer segments: the dataflow waits and the expert-major gather
dominate); `sep` gains little there. Apply level (`bench_apply.log`): 13,824 real 63.63 -> 56.56, collapsed 63.52 ->
55.97; 13,856 63.94 -> 56.62; 4,289 21.89 -> 21.86; 1,791 unchanged.

## 3. The switch (`GLM53_MOE_FUSED16`), its gate and tests

- `overlay/glm53_moe_fused16.py` wraps production's module-global `apply_exl3_grouped_fat` (called by name from
  `apply_exl3_fused_moe`); routing, the thin kernel, zero-fill, casts, tables and scratch stay production's.
  Unset / "" / 0: nothing installed; 1: on; anything else: WARNING, nothing installed. Installs only if
  `apply_exl3_grouped_fat` / `build_grouped_fat_tables` / `_grouped_scratch` carry glm53_prefill_cap's verified AST
  fingerprints. Exposes production's function as `_tf_exl3_orig` (later fingerprint checks see production's).
- Per layer, at model load (`Exl3MoEMethod.process_weights_after_loading` wrapped): shape checks + a 2,304-token
  synthetic grouped call on the layer's real experts through production's kernels and both schedules; served only if
  h2 is bit-identical and out within 1e-6 rel-L2 (measured ~3e-9). Summary line `glm53_moe_fused16 summary: N/M layers
  served, ...`. Pass-through: T < 4,096, CUDA-graph capture, non-fp16 input.
- Memory: production's grouped scratch (h13 / h2) + a 32 KB sync buffer + per call O(rows) tables. A side stream for
  `sep`. A device fault (illegal address, the p16b watchdog `__trap`) is sticky (as GLM53_MOE_E4M3; revert = unset +
  restart).
- Kit: `overlay/patch_moe_fused16.py` (+ `patch_tf_bundle.py` entry before e4m3's; its integrate.py block goes before
  glm53_moe_e4m3's so that one stays last). NOT done (kit integrator): start.sh forwarding of `GLM53_MOE_FUSED16` to
  both ranks, env.r16 switch, boot_checks lines, make_kit shipping `glm53_moe_fused16.py`.

Tests (logs in `docs/logs/moe3/`):
- `tests/moe3/test_fused16.py` **161/161**: E exhaustive exp_prod; B every schedule (p16b, sep with 1/2/4 chunks) at
  T 13,824 / 13,856 / 8,192 / 4,289 / 2,048 / 300, real + collapsed routing, x40 activations with inf/NaN rows, top-4,
  expert_map with half the experts non-local: h13 and h2 **0 differing values** everywhere, out rel 2e-9..7e-8 vs
  production's own run-to-run 2e-9..8e-9; H the hook: bf16 apply output vs unhooked production: every differing
  element within one bf16 ulp (+1e-5 x row rms), 0.0045 % of elements at 13,824; served / passed counts; capture
  pass-through; load-time self-test; knob values; fingerprint refusal; no double wrap; uninstall.
- `tests/moe3/test_patch_fused16.py` 22/22 (host), `tests/moe3/test_layers.py` 18/18.
- Regression of the e4m3 path on the rebuilt .so: test_moe_e4m3 56/56, test_down16 45/45, test_wiring 37/37,
  review_adv 16/16; test_patch_moe_e4m3 17/17.
- Real-engine run on the handoff MoE mini at TP=2 rank-0 shapes (section 5).

## 4. Layer-selective e4m3 (`GLM53_MOE_E4M3_LAYERS`) and how to choose the layers

`GLM53_MOE_E4M3_LAYERS=3-13,20` (comma list / ranges of model layer indices, read from RoutedExperts.layer_name):
only those layers take the e4m3 path (and only they run its load-time self-test); every other layer stays on
production's path (with `GLM53_MOE_FUSED16=1`: P16). Unset = every layer (unchanged behaviour). The summary line gains
`; GLM53_MOE_E4M3_LAYERS: K indices selected, U layers unselected` only when a list is set (boot_checks' 42/42 regex
then needs the selected count - kit work).

Calibration (production A/B, all 42 layers): long-KL +0.0083 (0.0049 -> 0.0132), dvp +0.0106 (0.0076 -> 0.0182).
Uniform-linear per layer: +0.000198 / +0.000252 (e4m3); DOWN=f16 removes ~1/3 of the added variance (moe2) ->
~+0.000133 / +0.000169. Gates left: long <= 0.010 (+0.0051), dvp <= 0.010 (**+0.0024, binding**).

| arm | layers | speed vs FUSED16 per 13,824 chunk | est. long-KL | est. dvp |
|---|---|---|---|---|
| e4m3 | 9 | -0.14 s | 0.0067 | 0.0099 |
| e4m3 + DOWN=f16 | 14 | -0.16 s | 0.0068 | 0.0100 |

Per-layer sensitivity cannot be measured on nodeC (one real MoE layer; the mini repeats layer 10's experts). To pick
layers, the cheapest measurement is a group A/B on production with FUSED16=1 as the base: 4 arms
`GLM53_MOE_E4M3_LAYERS=3-12 / 13-22 / 23-33 / 34-44`, quality_long + the dvp probe each; check additivity against the
all-layer numbers, then take the least sensitive groups up to the dvp budget.

## 5. Real-engine check (handoff MoE mini, TP=2 rank-0 shapes)

`tests/moe3/kl_chain.sh` on `GLM-5.3-Flash-handoff-moe-mini-tp2r0` (tests/moe3/build_mini_moe_tp2.py: 10 real layers,
MoE layers 3..9 = 32 real layer-10 EXL3 experts cut to the TP=2 rank-0 shard), the real vLLM engine composed like
production (kit r16x chain + this worktree's bundle), GLM53_PREFILL_FUSED_CAP=1, prompts 14,000 / 6,000 / 3,001 tokens,
1,434 scored positions, KL vs the `off` arm (logs `docs/logs/moe3/kl_*.log`). The mini's distribution is nearly flat
(NLL 13.8), so absolute KLs are not production's; the RELATIVE numbers are the point.

| arm | KL mean | excess over A/A | p95 | top-1 |
|---|---|---|---|---|
| off2 (A/A: production's own nondeterminism) | 0.00030 | - | 0.00061 | 97.7 % |
| **GLM53_MOE_FUSED16=1** (7/7 self-tests passed at load, p16b + sep served) | 0.00037 | +0.00007 (noise) | 0.00068 | 97.6 % |
| e4m3, all 7 MoE layers (= production since 10-02) | 0.00194 | +0.00164 | 0.00347 | 93.6 % |
| e4m3 + DOWN=f16, all layers | 0.00130 | +0.00100 | 0.00254 | 94.1 % |
| e4m3, DOWN=f16 on layers 3-5 only (`GLM53_MOE_E4M3_DOWN_LAYERS=3-5`) | 0.00132 | +0.00102 | 0.00248 | 93.5 % |
| e4m3, DOWN=f16 on layer 3 only | 0.00156 | +0.00126 | 0.00320 | 93.0 % |
| e4m3 on layers 3-5 only (6-9 FUSED16) | 0.00185 | +0.00155 | 0.00334 | 93.4 % |
| e4m3 on layers 6-9 only (3-5 FUSED16) | 0.00051 | +0.00021 | 0.00076 | 96.3 % |
| e4m3 on layer 3 only / layer 9 only | 0.00114 / 0.00039 | +0.00084 / +0.00009 | | |

Findings: (1) FUSED16 in a real engine is at the A/A noise floor, as the bit-level tests say. (2) The e4m3 damage is
NOT uniform over depth on the mini: the first MoE layers carry ~90 % of it (3-5: +0.00155 of +0.00164; layer 3 alone
half), roughly additive (3-5 + 6-9 = +0.00176 vs all +0.00164). (3) DOWN=f16 only on the sensitive early layers recovers
as much as DOWN=f16 everywhere, at 3/7 of the cost. Caveat: the mini repeats layer 10's experts at every depth and is
only 10 layers deep; whether production's 42 MoE layers show the same early-layer concentration must be measured
(group A/B, section 4), but it is the hypothesis to test first.

Quality-recovery exchange rate on the mini (excess KL removed per ms of a 13,824-token chunk, from section 2.4 /
bench_apply: e4m3 -> e4m3_d16 +4.5 ms per layer, e4m3 -> FUSED16 +15.7 ms per layer): DOWN=f16 on 3-5 4.6e-5 /ms,
FUSED16 on 3-5 3.0e-5 /ms (but the largest absolute recovery: -87 %), DOWN=f16 everywhere 2.0e-5 /ms.

## 5b. `GLM53_MOE_E4M3_DOWN_LAYERS` (per-layer f16 down)

Same list syntax as GLM53_MOE_E4M3_LAYERS; when set, exactly those layers run the fp16 down projection (fused variant
16, their load-time self-test against the f16-down spec), every other served layer the e4m3 down. Unset:
GLM53_MOE_E4M3_DOWN decides for all layers (unchanged). Verified in the real engine on the mini (self-tests passed,
the KL rows above).

## 6. Production recommendation (after the owner accepted e4m3-level quality, 2026-10-02)

Production now runs GLM53_MOE_E4M3=1 on all layers (long KL 0.0134, dvp 0.0195). What this branch adds is a cheap
way to BUY BACK quality budget (to stack W8A8 within it), cheapest first, per 32k request (2 x 13,824 + 4,289):
- `GLM53_MOE_E4M3_DOWN_LAYERS=3-12` (f16 down on the first 10 MoE layers): +~0.10 s (~-0.7 % tok/s) [E]; on the mini
  the early-layer f16 down removed 38 % of the e4m3 excess.
- `GLM53_MOE_FUSED16=1 GLM53_MOE_E4M3_LAYERS=13-44` (first 10 MoE layers exact): +~0.36 s (~-2.4 % tok/s) [E]; on the
  mini the equivalent split removed 87 % of the excess.
- Before either: one group A/B to confirm the depth concentration in the real model (base = current production;
  arms `GLM53_MOE_FUSED16=1 GLM53_MOE_E4M3_LAYERS=13-44` / `3-12,23-44` / `3-22,34-44` / `3-33`, each = one decile
  group of layers exact; quality_long + dvp each).
- If e4m3 is ever turned off again: `GLM53_MOE_FUSED16=1` alone = production numerics, -0.61 s per 32k request.
Kit work for any of these: forward GLM53_MOE_FUSED16 / GLM53_MOE_E4M3_LAYERS / GLM53_MOE_E4M3_DOWN_LAYERS to both ranks,
ship glm53_moe_fused16.py, boot_checks lines (the e4m3 summary gains a suffix when a layer list is set).

## 7. Not run / limits

- Nothing on nodeA/nodeB; no two-rank run (one rank's shard on one GB10); no kit, no boot of the real model.
- One real MoE layer (layer 10); the speed of other layers' experts is assumed equal (same shapes, same routing law).
- The production chunks' real routing is replaced by the calibrated synthetic routing kinds (real / collapsed).
- (a)'s quality numbers are extrapolations from the production A/B, not measurements.
