# PF3000 — kill-tests for the FP8 levers of the prefill-3000 plan (steps 3 and 4)

Status 2026-09-30, branch `pffp8` (from r16j = production). Node1 only (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, every GPU run `tests/gpu_run.sh` under
`flock /tmp/tf-gpu-bench.lock`, peak device budget ≤ 8 GiB per process, host MemAvailable ≥ 91 GB on every
run; tests/gpu_run.sh additionally caps each container's torch CUDA allocations at 40 GiB and --memory 64g
since the 09-30 15:13 host hang). Node2/nodeB were not accessed; nothing was installed anywhere;
measurement only.

Input: `${HOME}/tf-exl3-assets/prefill3000/PLAN.md` steps 3-4 and the critique (`critic.md` items 4-6).
Scripts: `tests/pf3000/` (`bench_test_c.py`, `resident_copy.py`, `ceil_e4m3.cu`, `bench_test_d_numerics.py`).
Logs: `docs/logs/pf3000/` (`test_c_bench.log`, `test_c_resident_copy.log`, `test_d_ceilings_e4m3.log`,
`test_d_numerics.log`, `probe_cutlass.log`).

## Verdicts at a glance

| test | threshold | measured | verdict |
|---|---|---|---|
| C: KDA in_proj W8A8 (`cutlass_scaled_mm`, M=13824, 2048-row pieces, per-token quant included) | ≤ 11 ms / 13,824 rows | **9.57 ms** (piece GEMMs 8.72 ms at 163 TFLOPS + quant 0.72 ms) | **PASS** (x1.78 vs production's current 17.0 ms) |
| C: layout (i) resident standard-layout fp8 copy | ≤ 0.5 GiB/rank | **3.50 GiB/rank** (= ~467k tokens of the 2.00M pool at 8,039 B/token) | **FAIL** |
| C: layout (ii) per-call re-layout | ≤ 10 ms/chunk | **106.9 ms/chunk** via existing kernels; fp8-direct floor 32.7 ms/chunk | **FAIL** |
| C: per-GEMM relative error (W8A8 vs fp32 reference) | < 3 % | 2.42e-2 … 2.65e-2 on every shape, both activation distributions | **PASS** |
| **C overall (plan step 3)** | | | **KILL as written** — the speed and numerics pass, but neither weight layout fits the budget (critic item 4 was right) |
| D: e4m3 MoE microkernel (trellis decode + `cvt.rn.satfinite.e4m3x2` + `mma.sync m16n8k32`, registers, 16 warps/SM, 128-row tiles) | ≥ 140 TFLOPS | **160.5 TFLOPS** (bf16 mix 97.2, e4m3 peak 215.5) | **PASS** |
| D: exit-B numerics on real GLM MoE-FFN weights | ≤ 1.2 x EXL3-native error | **x1.14-1.17** in the routed (Hadamard-rotated) domain, both weight variants, 3 shapes x 2 activation distributions; EXL3-native 6.19-6.70e-2 rel_l2 | **PASS** (x1.14 margin, exactly the plan's estimate) |

## 1. What production actually runs today (the comparison target)

Per-rank dense FP8 shapes and per-13,824-token-chunk call counts (from `DEC_FP8ROOF.md` §3.1 and the
`fp8_gemv` install line; KDA in_proj composition from `vllm/models/glm5next/nvidia/kda.py:196-219`:
q/k/v local 4096 each + b local 32 + f_a/g_a replicated 128 each = 12,576):

| shape (N x K) | calls/chunk | Marlin repack GiB/rank |
|---|---|---|
| KDA in_proj 12576 x 4096 (Npad 12608) | 34 | 1.63 |
| KDA o_proj 4096 x 4096 | 34 | 0.53 |
| MLA fused_qkv_a 2048 x 4096 | 11 | 0.086 |
| MLA q_b 8192 x 1536 | 11 | 0.13 |
| MLA o_proj 4096 x 8192 | 11 | 0.34 |
| shared gate_up 2048 x 4096 | 42 | 0.33 |
| shared down 4096 x 1024 | 42 | 0.16 |
| dense gate_up 12288 x 4096 | 3 | 0.14 |
| dense down 4096 x 6144 | 3 | 0.07 |
| drafter fc 4096 x 20480 | 1 | 0.078 |
| **total** | 192 | **3.51** |

Production's current path (env_nonsecret.txt: `GLM53_DENSE_FP8=dense,kda,mla,shared`, `GLM53_FP8_GEMV=1`,
`GLM53_FP8_GEMV_MAX_M=16`, `GLM53_FP8_LARGE_M=1`; `GLM53_KDA_BF16_LARGE_M` unset): the weights live **only**
as the Marlin repack (`prepare_fp8_layer_for_marlin`, int32 [K/16, 4·Npad], permuted, e4m3 x per-channel
bf16 scale); at prefill the large-M shapes are served by the exact-BF16-dequant + TileLang W8A16 path, the
rest by `marlin_gemm`. All measured through the real `Glm53DenseFp8Method.apply` of the image's own module
(`vllm/model_executor/layers/quantization/exl3.py`, sha256 2656a699… = docs/prod_exl3_reference.py).

## 2. TEST C — W8A8 dense GEMMs (`tests/pf3000/bench_test_c.py`, `test_c_bench.log`)

Method: interleaved rounds, median of 5 (order rotated per round), 3 reps per cell, M = 13,824 in
2,048-row pieces (6 x 2048 + 768). The W8A8 arm uses the image's own `torch.ops._C.cutlass_scaled_mm`
(sm_121 fp8, verified `cutlass_scaled_mm_supports_fp8 = True` in `probe_cutlass.log`) with per-token e4m3
activation quantization by the image's own `dynamic_per_token_scaled_fp8_quant`
(`ops.scaled_fp8_quant(use_per_token_if_dynamic=True)`; 0.13-4.2 ms per call depending on K —
`kda.in_proj` 0.715 ms for the full M). Weight operand: standard-layout fp8 [N, K] row-major (`.t()` view
to the kernel) built with production's own quantization formula (per-output-channel amax/448); the
un-permuted values were verified **100.00 % bitwise identical** to production's Marlin payload for every
shape (the dequant of the Marlin repack equals my fp8 tensor byte for byte).

ms per call at M = 13,824 (production current = its own path selection; W8A8 = pieces incl. quant):

| shape | production current | Marlin only | W8A8 pieces | W8A8 single call | speedup | GEMM-only TFLOPS |
|---|---|---|---|---|---|---|
| KDA in_proj | 17.0 | 54.9 | **9.57** | 27.9 | 1.78x | 163 |
| KDA o_proj | 5.87 | 5.87 | 3.66 | 2.73 | 1.60x | 151 |
| MLA qkv_a | 2.85 | 2.85 | 2.32 | 1.32 | 1.23x | 155 |
| MLA q_b | 4.83 | 4.83 | 3.24 | 2.26 | 1.49x | 116 |
| MLA o_proj | 17.3 | 17.3 | 7.20 | 12.3 | 2.41x | 154 |
| shared gate_up | 2.85 | 2.85 | 2.41 | 1.32 | 1.18x | 156 |
| shared down | 1.63 | 1.63 | 1.43 | 0.91 | 1.14x | 97 |
| dense gate_up | 16.4 | 48.7 | 9.38 | 27.2 | 1.75x | 167 |
| dense down | 8.79 | 8.79 | 5.10 | 7.58 | 1.72x | 163 |
| drafter fc | 29.9 | 52.7 | 23.4 | 30.9 | 1.28x | 122 |
| **chunk total** | **1,348** | 2,753 | **818** | 1,446 | **+530 ms/chunk** | |

- The plan's step-3 speed assumption is confirmed and beaten: piece W8A8 at 148-163 TFLOPS (plan assumed
  169 peak; the measured dense ceiling on this GPU is 166-167 TFLOPS GEMM-only at the biggest shapes).
  The pieces matter enormously (a single M=13824 call is 2.9x slower on in_proj).
- Numerics (real weights where the partial checkpoint has them — KDA in_proj/o_proj, shared gate_up/down,
  dense gate_up/down; synthetic otherwise; activations Gaussian sigma 0.02 and the x30-outlier-channel
  recipe of `FP8_LARGE_M.md`): W8A8 rel_l2 vs the fp32 reference **2.42e-2 … 2.65e-2** on every shape and
  both distributions (production's own path: 1.66e-3, as expected for the exact-e4m3 class). The
  W8A8-vs-production difference is 2.43e-2 … 2.65e-2. Under the 3 % kill line.

**Why step 3 is nevertheless a KILL as written — the weight layout.**
- Option (i), resident standard-layout fp8 copies for all 192 per-chunk calls: **3.50 GiB/rank**
  (`resident_copy.py` actually allocates every layer's copy; host MemAvailable fell by 3.0 GB) — 7x the
  0.5 GiB/rank budget, and it buys ~467k tokens off the 2.00M KV pool (8,039 B/token). Even in_proj alone
  is 1.63 GiB.
- Option (ii), per-call re-layout (what production's TileLang path already does once per call, plus the
  bf16→fp8 cast cutlass needs): **106.9 ms/chunk** total (in_proj 1.55 ms, dense gate_up 1.58 ms, fc
  2.54 ms per call). A dedicated fp8→fp8 repack kernel has a hard floor of 2·N·K bytes at the measured
  230 GB/s DRAM bandwidth: **32.7 ms/chunk** — still > 3x the 10 ms/chunk budget.
- Conclusion: the speed and numerics of W8A8 dense are real (worth ~+530 ms of the -550 ms the plan
  budgets for step 3), but the Marlin-only weight storage makes it unaffordable in both stated layouts.
  What the kill test therefore leaves open (for the plan owner): a *standard-layout* store at load time
  (weights re-packed once, Marlin kept only for decode M ≤ 64 — that is exactly option (i) at 3.50 GiB,
  or a split where only the ≥ 42-call shapes are copied: in_proj + shared = 1.96 GiB, still over), or a
  fused dequant-GEMM kernel that consumes the Marlin layout directly (what TileLang already does — the
  measured 106.9 ms is its cost, and the W8A8 arm cannot reach it without dropping to ~4 ms/chunk of
  re-layout). Not measured here (NOT RUN): a custom Marlin-layout direct W8A8 kernel; the 0.5 GiB/rank
  and 10 ms/chunk budgets are the plan's own numbers, not re-derived.

## 3. TEST D — e4m3 MoE ceiling microkernel (`tests/pf3000/ceil_e4m3.cu`, `test_d_ceilings_e4m3.log`)

Extends the E4 probe (`tf-exl3-fork-wt-e4/tools/e4/ceil.cu`, whose bf16 numbers this first reproduces:
bf16 mma m16n8k16 107.8 TFLOPS, decode+mma mix 86.5 (64-row) / 96.3 (128-row) TFLOPS) with the exit-B
mainloop: per warp, decode the two 16x16 trellis tiles of one n8 half (k16 each) into fp16 fragments in
registers, `cvt.rn.satfinite.e4m3x2.f16x2` x 4 + pack -> one k32 e4m3 B fragment, then `mma.sync
m16n8k32 f32.e4m3.e4m3.f32` per 16-row block. Decode-to-FLOP ratio is identical to the bf16 mix
(8192·MB FLOP per decode; MB = 8 for a 128-row tile). Routed-expert shapes per rank (moe_intermediate
2048, TP=2): gate/up N 2048 x K 4096, down N 4096 x K 1024 — the probe is shape-free (operands never
leave registers); the shape enters only through the tile counts.

Results (`test_d_ceilings_e4m3.log`, nvcc -O3 -arch=sm_121a, median of 5 rounds of the best launch; the
bf16 rows reproduce the E4 probe's `ceilings.log` within noise, which validates the harness):

| kernel | TFLOPS |
|---|---|
| mma.sync m16n8k16 f16/f32 (bf16 reference peak, 8 or 16 warps) | 107.8 |
| **mma.sync m16n8k32 e4m3/f32 (peak)** | **215.5** (= 2.0x bf16, as assumed) |
| trellis decode only (16w x 1 blk/SM) | 5.06 Gtiles/s (ceilings.log: 4.98) |
| bf16 mix, 64-row tiles (1 decode -> 8 mma k16) | 86.5 (8w) / 88.8 (16w) |
| bf16 mix, 128-row tiles (1 decode -> 16 mma k16) | 96.3 (8w) / 97.2 (16w) |
| **exit-B mix (B side cvt in registers), 64-row tiles** | 111.0 (8w) / 116.3 (16w) |
| **exit-B mix (B side cvt in registers), 128-row tiles** | 136.8 (8w) / **160.5 (16w)** |
| worst case, A and B both re-converted per k32 (A is fp8 from the gather in the real kernel) | 117.7 |

- **Verdict: PASS at the specified configuration** (16 warps/SM, 128-row tiles): **160.5 TFLOPS >= 140**
  — inside the plan's 140-170 structure-ceiling band, x1.65 over the bf16 mix (97.2). Below 140 only at
  128 rows with 8 warps (136.8, 2 CTAs/SM) or at 64-row tiles — the same tile-efficiency story as E4
  (partial tiles pay a full decode), which pushes toward 128-row tiles + the 64-row tail launch, exactly
  the shared work the plan already lists for exit A.
- If the A operand also had to be converted in registers every k step (it should not: exit B's A rows
  come out of the gather already e4m3), the mix drops to 117.7 TFLOPS — a KILL for that variant. The A
  operand must come from the fp8 gather, not from a re-rounding in the mainloop.

## 4. TEST D — exit-B numerics on real weights (`tests/pf3000/bench_test_d_numerics.py`)

## 4. TEST D — exit-B numerics on real weights (`tests/pf3000/bench_test_d_numerics.py`)

- **Real EXL3 routed-expert weights: NOT RUN.** The partial checkpoint's index lists all 42 MoE layers x
  288 experts (trellis/mcg/suh/svh), but the 120 safetensors shards are not on this node (the directory
  holds only bf16_samples/ + lm_head/, 3.0 GB; the script prints the missing shard name).
- Substitution (state as such): real GLM-5.3-Flash **MoE-FFN** weights from `bf16_samples`
  (shared_experts of MoE layer 10, dense MLP layer 1), raw and Hadamard-rotated (the routed path's own
  128-point transform, via the repo's reference `kernels/exl3_format_ref.py`); EXL3's 4-bpw error
  modelled as relative-RMS noise 6.25 % (RD bound) / 6.70 % (QTIP-class) — the same rate-distortion model
  the moe-kernels reader used. Activations: Gaussian and x30-outlier channels, 2,048 rows per call.
  Paths: today = fp16 operands, fp32 accumulate; exit B = per-row e4m3 activations x e4m3 weights
  (direct satfinite cvt, and a favourable per-128-column-block variant), fp32 accumulate.
- Results (`test_d_numerics.log`, 48 cells, fp16 epilogue on both paths, fp32 accumulate):

| weights / domain | EXL3-native rel_l2 | exit B, direct satfinite cvt | exit B, per-128-block weights | ratio (rotated) |
|---|---|---|---|---|
| shared gate/up 2048x4096, Hadamard-rotated | 6.23-6.70e-2 | x1.144-1.165 | x1.140-1.161 | **pass** |
| shared down 4096x1024, Hadamard-rotated | 6.19-6.70e-2 | x1.144-1.166 | x1.141-1.161 | **pass** |
| dense MLP l1 12288x4096, Hadamard-rotated | 6.25-6.70e-2 | x1.145-1.166 | x1.141-1.161 | **pass** |
| same three shapes, raw (unrotated) | 6.19-6.70e-2 | **x1.18-1.30 (KILL at rho=0.0625 on gate/up and down)** | x1.13-1.16 (pass) | mixed |

- **Verdict: PASS for exit B as designed.** In the routed domain (where the experts actually operate, and
  where the rotation lifts the decoded weights into e4m3's normal range) the output error is
  **x1.140-1.166** the EXL3-native error — the plan's x1.14 estimate confirmed, under the 1.2 kill line,
  with Gaussian and x30-outlier activations alike, and for either weight-scaling variant. Two design
  constraints fall out of the raw-domain rows: (1) the in-register `cvt.rn.satfinite.e4m3x2` must not be
  applied to unscaled decoded weights outside the rotated domain (values near/below e4m3's minimum normal
  2^-6 lose relative precision; x1.30 at gate/up rho=0.0625); the rotation the experts already apply (or a
  per-128-block scale) is required, not optional. (2) The measured increment (x1.14-1.17) is against the
  EXL3-native error being 6.2-6.7 %, so the absolute exit-B GEMM error is 7.1-7.8 % per routed GEMM.
- Caveats: the EXL3 4-bpw error is a rate-distortion model, not real EXL3 weights (section 5); the fc2
  per-128-column activation scale of the plan's Y8 is on top of this (it would add a small activation
  error; here the A side is per-row e4m3, the finer of the two).

## 5. What was NOT RUN

1. Real EXL3 routed-expert weights (Test D numerics): shards absent from nodeC — substitution documented
   above. A faithful re-run needs the TR3-4bpw checkpoint's MoE shards fetched to nodeC read-only.
2. Real captured activations for Test C (the handoff-mini model): no activation dump exists on nodeC and
   wiring the handoff engine to harvest one was out of scope; Test C replays production's documented
   synthetic recipes instead (Gaussian, x30 outlier channels) on real weights where available.
3. A custom W8A8 kernel reading the Marlin layout directly (the only layout variant that could still save
   step 3), and any residency/re-layout pipelining.
4. Any production-side A/B: nodeA/nodeB untouched, no server was started, nothing installed.
5. `mma.sync m16n8k64` (mxf4/nvf4) peaks — out of scope for exit B (e4m3 only).

## 6. Adversarial review (2026-09-30)

**Test C — the layout KILL does not follow.**
- The "0.5 GiB/rank" and "10 ms/chunk" budgets appear nowhere in PLAN.md (its critique only estimates
  ~3.2 GiB resident / ~30 ms/chunk re-layout); the doc's claim that they are "the plan's own numbers" is false.
- On the measured numbers, per-call re-layout (0 GiB resident, transient <= 0.2 GiB) still nets
  1,348 - (818 + 106.9) = **-423 ms/chunk/rank**; with an fp8->fp8 repack at the DRAM floor
  1,348 - (818 + 32.7) = **-497 ms**. Production's own TileLang large-M path already pays a per-call
  Marlin->bf16 dequant inside the 1,348. Corrected verdict: **speed PASS, layout PASS via option (ii)**
  (net criterion printed by `bench_test_c.py` from now on; log not re-run, arithmetic from the committed log).
- Minor: for several shapes the single call beats the 2048-row pieces (shared gate_up 1.32 vs 2.41 ms,
  qkv_a 1.32 vs 2.32); a per-shape choice would lower 818 further. The 3 % numerics line passes by
  construction (e4m3 activation rounding alone ~2.6 %); it is not evidence of quality (section 9 KL gates).

**Test D speed — defect in k_mixf8, conclusion survives.**
- `k_mixf8` decodes k32 x n16 per iteration but feeds only one n8 half (`tb0/tb1` dead, removed by the
  compiler); it is not the mainloop the plan describes and its FLOP/decode is not the bf16 mix's.
  Added `k_mixf8full` (both n8 halves, 2*MB mma per 2 decodes) and a variant reading A fragments from
  shared memory every k32 step (`test_d_ceilings_e4m3_review.log`): 128-row x 16w **175.1 TFLOPS**
  (registers A), **166.4 TFLOPS** (A from smem); 4-row-block 117.6; 128-row x 8w 161.8.
- Still NOT modelled (all versions): the global/cp.async load of trellis words and of the gathered A rows,
  the gather itself, fp32 epilogue/finalize, per-expert tile-count tails, occupancy under a real register
  budget (d[8][2][4] + staging). The 140 line is a structure ceiling, not a kernel prediction; the
  plan's own 31-43 ms/layer adds those costs separately.
- mixf8ab (117.7) is not a meaningful "A re-conversion" model (A built from B decodes); ignore it.

**Test D numerics — apples-to-apples but model-bound.**
- Same fp64 reference for both paths: OK. But the EXL3 error is synthetic Gaussian noise at an assumed rho;
  the ratio is essentially sqrt(rho^2 + e_fp8^2)/rho with e_fp8 ~3.6 %, so the verdict is decided by the
  assumption: ratio <= 1.2 needs rho >= ~5.4 %. Real routed-expert EXL3 error was not measured.
- "direct cvt" converts weights at their raw bf16 magnitude (std ~1e-2, partly e4m3 subnormal); real trellis
  output is unit-scale codebook values with suh/svh applied outside, so the "unrotated x1.18-1.30" result is
  partly an artefact of scale, not of rotation. Activations are Gaussian + channel outliers, not captured rows.
- Verdict: PASS stands only as "not killed by the model"; needs real EXL3 experts + captured activations.
