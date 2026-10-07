# KDA_FLASHKDA2 — why "quality short vs R15" doubled with FlashKDA, and the precision build that fixes the kernel

Status 2026-10-02, branch `fkda2` (from `fkda` b4567bf = kit r16n, which production runs with `GLM53_KDA_FLASHKDA=1`).
nodeC only; production (nodeA/nodeB) was not touched. No kit was built or installed.

## 1. Is the regression real? Yes.

`tools/prodcheck/quality_probe.py --compare` ("quality short vs R15": 6 fixed ~115-240-token texts,
`prompt_logprobs=20`, KL(A||B) over the union of the two top-20 sets, unseen tokens floored at 1e-9), same R15
reference file every time, n = 962 positions:

| run (A/B log) | kit | mean | p95 | max |
|---|---|---|---|---|
| r16k-ab/A.log 2026-09-30 | r16k A | 0.0085 | 0.0273 | 0.645 |
| r16k-ab/B.log 2026-09-30 | r16k B | 0.0078 | 0.0225 | 0.643 |
| r16k-ab/r16l.log 2026-10-01 | r16l (FlashKDA off) | 0.0084 | 0.0254 | 0.643 |
| r16n-ab/run.log 2026-10-01 | r16n, FlashKDA on | **0.0174** | **0.0571** | **2.56** |

Three earlier runs on three kits/days spread by ±5 % (0.0078-0.0085, p95 0.022-0.027, max always 0.64); FlashKDA
moved all three statistics by 2-4x. Not noise. (The decode half of the probe — identical 2-5/20 across all kits — does
not discriminate anything.)

## 2. Root cause: FlashKDA 17a037d is less precise than the Triton chain inside a call (not only at short lengths)

Rig: `tests/fkda2/short_accuracy.py` — per-position output error and final-state error vs an **fp64 reference of the
exact KDA recurrence**, for the production Triton chain (`chunk_kda_with_fused_gate`) and a FlashKDA build.
Inputs: **real KDA inputs** captured from the handoff mini model (real GLM-5.3-Flash KDA weights, production-composed
container, `tests/fkda2/dump_kda_inputs.py`: every `chunk_prefill` call of the 6 quality-probe texts + a 160- and a
1500-token real-text prompt × 5 KDA layers = 40 calls, real A_log/dt_bias) plus synthetic long-regime inputs.
"floor" = the fp64 output rounded to bf16 (both kernels emit bf16).

Output rel-RMS vs fp64, token-weighted per position bin (all 44 cases):

| position in sequence | bf16 floor | Triton | FlashKDA 17a037d (shipped r16n) | **fkda2 build** |
|---|---|---|---|---|
| 0-16 | 1.62e-3 | 3.62e-3 | 4.93e-3 (1.36x) | 3.72e-3 (1.03x) |
| 16-64 | 1.65e-3 | 3.79e-3 | 5.05e-3 (1.33x) | 3.34e-3 (0.88x) |
| 64-256 | 1.65e-3 | 3.83e-3 | 4.78e-3 (1.25x) | 3.16e-3 (0.83x) |
| 256-1024 | 1.65e-3 | 4.23e-3 | 5.18e-3 (1.22x) | 3.39e-3 (0.80x) |
| 1024+ | 1.65e-3 | 4.23e-3 | 5.15e-3 (1.22x) | 3.39e-3 (0.80x) |

Real-input means (40 calls): output Triton 3.71e-3 / shipped FlashKDA 4.63e-3 (**1.25x**) / fkda2 3.13e-3 (**0.84x**);
final state Triton 2.12e-3 / shipped 3.15e-3 (**1.49x**) / fkda2 2.17e-3 (1.02x). Worst single token: shipped
1.27e-2, fkda2 8.7e-3. Chained 8 × 13,824-token chunks (long-memory regime, `tests/fkda/chained_chunks_vs_fp64.py`):
state vs fp64 shipped 5.6e-3 = 1.36x Triton → **fkda2 3.58e-3 = 0.86-0.88x Triton, flat over 110,592 tokens**.

So: **the Triton chain was closer to exact than shipped FlashKDA** (by ~25 % on outputs, ~50 % on states), uniformly
over positions — the excess is not specific to short prompts; the short probe is just where the KL metric sees it.
The kernel's intra-tile arithmetic rounds to bf16 repeatedly where the Triton chain keeps fp32
(`fkda-evidence/flashkda-src/csrc/smxx/`):

| site | stock 17a037d | roundings |
|---|---|---|
| K1 decay pass | `q = bf16(q·rnorm)`, `e = bf16(exp2(gcum))`, `qd = q * e * bf16(scale)` (bf16 operators), same for `kd`, `k_inv = k*bf16(exp2(-g))`, `k_restored = k*inv*bf16(exp(gtot))` | 3-4 per MMA operand |
| K2 phase 3 | `u = (v - bf16(k@S)) * bf16(sigmoid(beta))` in bf16 operators | 4 |
| K2 phase 2/4 | `out = bf16(q@S) + bf16(Mqk@U)`, the add in bf16 | 3 on every output |
| K1/K2 | gate and beta via `tanh.approx.f32` | (measured: no effect) |

Attribution, one switch at a time (`tests/fkda2/patch_flashkda_precision.py`, real-input mean output error 4.63e-3
stock): `FKDA2_FP32_DECAY` → 3.81e-3 (the largest share; also state 3.15→2.46e-3), `FKDA2_FP32_U` → 4.36e-3,
`FKDA2_FP32_OUT` → 4.33e-3, `FKDA2_EXACT_SIGMOID` → 4.63e-3 (none: dropped, default OFF), all three → 3.13e-3.
Untouched (inherent to the tensor-core design, and Triton has the same class): bf16 MMA operands (Mqk, INV, U, the
bf16 view of the fp32 state), the bf16 round-trip of the initial/final state at the call boundary
(`tests/fkda/state_carry_precision.py`: 2.3e-3 vs the 1.7e-3 round-trip floor, unchanged).

### What a production KL vs R15 can and cannot show

R15's KDA prefill IS the Triton chain, so "KL vs R15" measures distance from Triton's rounding pattern, not
distance from exact. Measured on the same real inputs (relative to |ref|): shipped FlashKDA vs Triton 5.40e-3,
**fkda2 vs Triton 4.65e-3**, an exact kernel (fp64 rounded to bf16) vs Triton 4.02e-3. If the KDA part of the probe's
KL scales with the square of that distance, r16n's +0.009 becomes ≈ +0.0067 with fkda2 (mean ≈ 0.015), and **even an
exact kernel would leave ≈ +0.005**. The probe cannot return to 0.0084 with any kernel other than the Triton chain
itself; it is the wrong yardstick for "more accurate". In addition its unseen-token floor (1e-9) makes a single
top-20 membership flip at a high-entropy position cost ~p·20 nats (that is how one position reaches 2.56).

The nodeC mini model cannot settle the end-to-end question either: its logits are high-entropy (top logprob ≈ -4)
and bf16-quantized, so **any** KDA perturbation saturates the probe metric — `tests/fkda2/e2e_kl.py` +
`compare_e2e.py` (prompt logprobs of the 6 probe texts, KDA prefill = Triton / exact fp64 / shipped / fkda2, all else
identical; Triton vs Triton = 0 exactly): exact‖Triton 0.147, shipped‖Triton 0.146, fkda2‖Triton 0.139,
exact‖shipped 0.149, exact‖fkda2 0.151 — all the same. Only production can measure the real-model effect.

## 3. The fix (branch `fkda2`)

* `tests/fkda2/patch_flashkda_precision.py`: 9 exact-anchor edits to FlashKDA 17a037d `csrc/smxx/{utils,fwd_kernel1,
  fwd_kernel2}.cuh`, each behind a compile-time switch (default: FP32_DECAY/FP32_U/FP32_OUT on, EXACT_SIGMOID off):
  decayed q/k/k_inv/k_restored computed in fp32 and rounded to bf16 once; `u = (v - kS)·beta` from the fp32
  accumulator with fp32 beta, rounded once; `out` accumulates `Mqk@U` onto the fp32 `q@S` accumulator, rounded once.
  Same tiling, same MMAs, same register/smem budget class.
* `tests/fkda2/build_variant.sh <name> [-D...]`: the fkda build (registration-shim rename, sm_121a, CPU-only in the
  production image) of a patched copy. `v_fix` (default switches) → `overlay/_flashkda_fp32_C.abi3.so`
  sha256 `dd1788c24f10af2c…` (4,657,048 B). Reproducibility: `v_stock` (all switches 0) is **bit-identical** to the
  shipped c286213f… build on single and varlen calls (`tests/fkda2/check_bitwise.py`), so every difference is the patch.
* `overlay/glm53_flashkda.py`: pins `EXT_SHA256` (another `.so` → RuntimeError at boot, fail-closed; harness override
  `STATE["allow_any_ext"]`), and the RESERVED boot line now ends with
  `; extension sha256 dd1788c24f10af2c = fkda2 precision build (fp32 decay/u/out)` (the boot_checks need-string is a
  prefix and still matches). `tests/fkda2/wrapper_pin.py`: pinned build configures, the stock build is refused, the
  override configures and says so — ALL OK.

Speed (`tests/fkda/bench_prefill_on_off.py`, interleaved ship/fix/ship/fix + the installed overlay, T = 13,824,
per layer): op alone 6.35-6.38 (stock) vs 6.31-6.42 ms (fkda2); wrapper + kda_conv (production's path) 6.47-6.51 vs
6.47-6.50 ms; **saving per 13,824-token chunk 548-562 vs 548-558 ms** — unchanged within noise. 4,608 / 1,791 the same.

Existing rigs re-run on the fkda2 build: `wrapper_unit.py` ALL OK; `wrapper_short_varlen.py` ALL OK (per-row vs
Triton: out ≤ 6.7e-3, state ≤ 5.4e-3, was 7.8e-3 / 6.5e-3); `state_carry_precision.py` unchanged;
`chained_chunks_vs_fp64.py` ratio 0.86-0.88 (was 1.35-1.38).
Real-weight handoff mini A/B (`tests/handoff/run.sh`, r16n kit staging, QUICKWINS=all + MLA_PREFILL=1 both arms,
`--consistency 8`): control IDENTICAL; boot line shows the fkda2 sha; **greedy 48/48 identical** qw_off2 vs qw_on2
(qw_off2 also token-identical to the fkda review's qw_off); decode-vs-fresh-prefill consistency KL mean 0.0153
(fkda2) vs 0.0155 (Triton arm), top-1 8/8 vs 7/8; owned KDA state max|Δ| vs the Triton arm 2.0e-4..2.1e-3 (fkda
qw_on: 2.2e-4..2.3e-3). on_long2 (30,000 tokens, 3 chunks): control IDENTICAL, state max|Δ| vs Triton 2.5e-4..6.5e-3
(fkda: 2.4e-4..4.5e-3) — note this is distance from Triton, which is itself 1.15x further from exact than fkda2.

Kit-script compatibility (against a throw-away copy of the r16n kit with only `overlay/glm53_flashkda.py` +
`overlay/_flashkda_fp32_C.abi3.so` replaced and its MANIFEST re-hashed, deleted afterwards — not a kit):
`check_boot_strings.py` 139 markers, 0 not printable; `test_boot_checks.sh` 71 ok, ALL OK.

Logs: `docs/logs/fkda2/` (copies) and `${HOME}/tf-exl3-assets/fkda2/` (raw: `acc/`, `bench/`, `units/`,
`handoff/`, `e2e/`, `builds/`, the real-input dumps `handoff/dump/kda_calls/` 1.5 GB, sources `src/`).

## 4. What a production A/B must check (an r16o kit = r16n with the two overlay files replaced)

* boot line ends `extension sha256 dd1788c24f10af2c = fkda2 precision build`; nothing else changes (decode path,
  buffers, kda_conv extension are identical).
* "quality short vs R15": expect it to DROP but not to 0.0084 (estimate ≈ 0.015 ± noise; the probe measures distance
  from the Triton rounding pattern). Better yardsticks: decode-vs-prefill KL (r16n 0.00712), long-context KL
  (0.0047), and ideally a probe without the 1e-9 floor (e.g. KL over the intersection, or min-logprob fill).
* speed: real-text prefill / random-24k prefill unchanged vs r16n (expect ±1 %).
* not covered on nodeC: 800k-token sessions (the chained 110k-token test is flat), TP=2 rank symmetry (same kernel on
  both ranks), and the fact that nothing on nodeC can measure the real model's sensitivity.

## 5. Reproduce (nodeC)

```
tests/fkda2/build_variant.sh v_fix                                  # CPU; -DFKDA2_X=0 switches for variants
HANDOFF_KIT=${HOME}/tf-exl3-deploy16.r16n tests/handoff/run.sh <out> dump \
  HANDOFF_DRIVER=/w/tests/fkda2/dump_kda_inputs.py GLM53_PREFILL_QUICKWINS=all GLM53_MLA_PREFILL=1 \
  GLM53_KDA_FLASHKDA=1 -- --prompts 1,1,1,1,1,1,160,1500 --gen 2 --shadow 0
FKDA_SCRATCH=<dir holding handoff/dump> FKDA_SRC=<dir with real_A_log.npy> flock /tmp/tf-gpu-bench.lock \
  tests/fkda/gpu_run.sh -c "python3 /w/tests/fkda2/short_accuracy.py --calls '/fkda/handoff/dump/kda_calls/*.pt' \
  --first-call 120 --synthetic 16,64,160,1500 --ext-dir /w/overlay --out /fkda/acc/x.json"
flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh -c "python3 /w/tests/fkda2/check_bitwise.py --a <dir> --b <dir>"
FKDA_USER=root flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh -c \
  "python3 /w/tests/fkda2/wrapper_pin.py --other <dir with the stock .so>"
flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh -c "python3 /w/tests/fkda/bench_prefill_on_off.py [--ext-dir d]"
```
