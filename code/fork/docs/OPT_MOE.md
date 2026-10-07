# OPT_MOE: routed-MoE prefill, faster and checked (branch opt-moe, base moe3 ebac57e)

Scope: the e4m3 routed-MoE prefill that production runs since 2026-10-02 (`GLM53_MOE_E4M3=1`, every MoE layer,
e4m3 down, fp32 accumulator) and its glue. All numbers are nodeC (GB10), the production image, the real layer-10
experts of the TP=2 rank-0 shard (288 experts, K 4096, N_loc 1024), Gumbel-top-8 "real" routing
(`tests/prefill_cap_common.py`), CUDA events, median of 5-7 rounds, unless stated otherwise. Logs:
`${HOME}/tf-exl3-assets/opt-moe/` (copied to `docs/logs/optmoe/`).

## 1. What landed (ACC / FOLD opt-in; TOKGATHER default-on, numerically transparent)

| knob | what | 13,824-token layer call | 4,289 | quality |
|---|---|---|---|---|
| `GLM53_MOE_E4M3_ACC=bf16` | the down epilogue rounds each weighted contribution to bf16 and adds it with `red.add.noftz.v4.bf16x2` into the bf16 output apply returns (gather2 zeroes it): half the scatter bytes, half the zeroing, no `.to(bf16)` pass | **-7.0 ms** (41.47 -> 34.49, `bench_acc`) | -2.8 ms | rel-L2 3.9e-3 vs the fp32 accumulator (the cast alone 1.7e-3); distance-to-production variance x1.003; handoff-mini KL vs off 0.00197 / 0.00198 (two runs) vs the fp32-accumulator arm's 0.00206 / 0.00205; bf16-arm A/A 0.00068 = its distance to the fp32 arm (0.00069) |
| `GLM53_MOE_E4M3_TOKGATHER` (default on, `=0` off) | every expert of a layer has the same w13 suh (checked per layer at load; true for all 7 mini layers and the real layer 10), so the gathered e4m3 gate/up input does not depend on the expert: `gather_tok` writes ONE row per token (57 MB instead of 453 MB at 13,824) and the fused gate/up jobs read A rows through the item's row tokens (staged in shared memory, cp.async) | **-1.24 ms** bf16 acc (34.14 -> 32.90), -1.81 fp32 acc (`test_tg`) | -0.70 / -0.84 ms | the same bytes (gather_tok row == gather2 row of every pair, scales bitwise) and the same arithmetic; real engine: logprobs IDENTICAL to a second run of the per-pair path (KL vs e4m3 2.86e-5 = e4m3x's A/A exactly, KL vs off 0.0020487 = e4m3x's) |
| `GLM53_MOE_E4M3_FOLD_SHARED=1` (needs ACC=bf16) | the routed sum is accumulated straight into the shared experts' bf16 output (vLLM runs them first for prefill-sized batches) and vLLM's `shared_output + fused_output` add is skipped for that call | **-2.05 ms** (36.03 -> 33.98 incl. the add, `test_fold`) | -0.5 ms | folded 4.0e-3 vs S + routed(fp32) where the unfolded bf16 path is 4.3e-3; handoff-mini real engine: 28/28 served calls folded, KL vs off 0.00212 (e4m3 arms 0.00197-0.00206), vs the bf16 arm 0.00083 (its A/A 0.00068) |

Per 32k request (2 x 13,824 + 4,289 tokens, 42 MoE layers, both ranks in parallel): ACC=bf16 saves
42 x (2 x 7.0 + 2.8) = 0.71 s, FOLD another 42 x (2 x 2.05 + 0.5) = 0.19 s, TOKGATHER 42 x (2 x 1.24 + 0.70) =
0.13 s, against 31,937 / 2,620 = 12.19 s -> ~11.16 s, **~2,860 tok/s predicted** [E: kernel-level savings times
layer count; the gate itself must confirm]. With all three on: run() at 13,824 = 32.9 ms - 1.5 (the add it absorbs)
vs production's 41.5 + 1.5 (its add).

### 1.1 GLM53_MOE_E4M3_ACC=bf16 (commit 84bafe8)

Kernel: `me_fused_kernel` template OB (0 = fp32 `red.v4.f32` as before; 2 = bf16, 8 values per op: the two rows
of a round exchange halves between lane pairs; 1 = 4 values per op, kept for A/B). Module: `run()` allocates a bf16
output and gather2 zeroes it; any call where the bf16 accumulator does not apply (topk not 2/4/8, a non-bf16 x, the
chunked schedule) silently keeps the fp32 accumulator (more precise, not less). The load-time self-test runs both
accumulators and refuses the layer if the bf16 one is off by more than 8e-3 from the fp32 one.
Tests: `tests/optmoe/test_acc.py` 87/87 (T 13,824 / 13,856 / 4,289 / 2,048 / 300, real + collapsed routing, x8
activations, NaN-prefilled out fully overwritten, half the experts non-local, T=1/17, one expert holding every row,
the hook incl. refused values); `identity.py` vs the r16z2 kit build 6/6 (the default path is unchanged within the
fp32 atomics-order class).
Non-determinism: bf16 atomics make the run-to-run result order-dependent at bf16 resolution (A/A KL 6.8e-4 on the
mini vs 2.9e-5 for the fp32 accumulator); production's fp32 atomics were already order-dependent, only less.

### 1.2 GLM53_MOE_E4M3_FOLD_SHARED=1

No vLLM file is patched. `install()` adds two runtime wrappers on `vllm...fused_moe.runner.moe_runner`:
`MoERunner.forward` arms the fold for that forward only when the runner's result is exactly
`shared_output + fused_output` (routed_scaling_factor 1.0, no routed output transform, fused output not pre-reduced,
not sequence-parallel, no zero-expert router, no DBO) and `_unpack` drops the shared half of a folded call's
(shared, fused) tuple, so `forward` takes its `result = fused_output` branch. The served call (`_serve`) folds only
when the shared experts' pending output is a contiguous bf16 [T, 4096] on x's device (prefill order NO_OVERLAP; in the
aux-stream order the shared output does not exist yet and nothing is folded) and the bf16 accumulator applies.
Fail-safe: the wrappers refuse to install if the image's `MoERunner.forward` / `_apply_quant_method` source no longer
contains the four lines the fold relies on (`FOLD_FINGERPRINT`); the e4m3 path then runs unfolded (WARNING).
Tests: `tests/optmoe/test_fold.py` 39/39 (numerics at 13,824 / 4,289 / 2,048; ignored for fp32 accumulator, wrong
shape, non-contiguous, fp16 buffer; the runner protocol on a stand-in moe_runner with vLLM's lines: folded result is
the shared buffer itself, no add; not folded with scaling 2.5, an output transform, pre-reduced output, sequence
parallel, no pending shared output, a decode-sized call; context cleared; uninstall restores; the image's real
moe_runner: fingerprints present, wrappers install/uninstall; refused values; timing).

### 1.3 GLM53_MOE_E4M3_TOKGATHER (default on)

`tok_gather_ok(layer)`: `_exl3_shared_w13_suh` (production's gate == up check) and every expert's gate suh equal to
expert 0's (one `torch.equal`, cached on the layer, counted in the summary line "token gather N/M served layers").
`gather_tok` = gather2's TOKM mode (one row per token with expert 0's suh; zeroes the out rows of its tokens). Fused
variants 2048 (+ 16 for the f16 down): the item's 128 row tokens are staged in a static shared table (`g_srt[2][128]`,
current / next item; the next item's by a 4-byte cp.async at the loop head that joins the mainloop's first copy
group) and `fload_a` reads A row r from `a8[g_srt[pb][r]]`; the gate/up epilogue's row scale from `asc[g_srt[pb][r]]`.
What it cost to get there (the kernel is register-capped at 128/thread, so the mainloop is sensitive to anything that
changes the allocation): a `__ldg(row_token)` per A chunk made the fused kernel +1.9 ms (net gain ~0), a Job field
holding the shared-table pointer +1.5 ms, the buffer index passed as an argument (final) +1.0 ms against the gather's
-2.3 ms. Tests: `tests/optmoe/test_tg.py` (qualification incl. a perturbed suh / gate != up; bytes bitwise at T 13,824 /
4,289 / 300 / 17 / 1; NaN-prefilled out zeroed; fused output == per-pair within the atomics-order class for variants
0 and 16, fp32 and bf16, real + collapsed routing, half the experts non-local, the fold; hook: unset on, 0 off, invalid
refused; timing).

## 2. Where the fused kernel spends its time (so the next step attacks the right thing)

Removal probes (debug build `-DME_DEBUG_VARIANTS`, variants 1001-1164, results garbage, timing only), bf16
accumulator, T = 13,824, base 30.4 ms for the fused kernel alone:

| probe | ms | reading |
|---|---|---|
| no trellis (B) loads | 29.1 | weights re-read ~3.5x per call (1,008 segments x 6 MB) but that traffic is NOT the limit |
| every segment on expert 0 (weights L2-resident) | 29.6 | same conclusion |
| no A + no B loads | 28.4 | memory traffic as a whole ~2 ms |
| no down reds (bf16) | 29.1 | scatter-add ~1.2 ms (fp32: ~2.4) |
| no mma | 25.7 | |
| no trellis decode | 26.2 | |
| no decode, no mma | 24.3 | |
| no B loads, no decode | 23.0 | |
| skeleton (no mma / decode / A / B / ldmatrix) | 13.1 | loop + barriers 4.9, epilogues + actq + ticket 8.9 (clock64 per phase) |

clock64 phases of the real kernel (DBG 64, `prof_fused.py`, per SM): mainloop gate/up 17.4, mainloop down 8.8,
gate/up epilogue 1.9, actq 0.8, down epilogue 3.5, ticket 0.8 ms (sum 33.2 with the probe's barriers).
mma.sync peak on GB10 (`tests/optmoe/micro/mma_rate.cu`): m16n8k32 e4m3 -> f32 = **1,027 MAC/clk/SM = 215 TFLOPS**
(f16 accumulate the same; m16n8k16 f16 half of it). The mainloop runs 7,140 cycles per 128-row x 256-col x k128
stage against a ~3,500-4,100-cycle mma floor: the kernel is at ~42 % of tensor peak and is issue/overlap-bound
(trellis decode ALU, 32 `ldmatrix.x4` per warp per stage = 256 KB of shared-memory reads per SM per stage, one
barrier per stage), plus ~7 ms of epilogue during which the tensor pipe idles (1 CTA of 16 warps per SM, 128 regs).

What would move it (not done here, each is a kernel restructure):
- epilogue overlap: the tensor pipe is idle for ~7 of 30 ms. Needs warp specialization or two CTAs per SM; the
  register file (512 x 128) and shared memory (pipe 64 KB + staging 17 KB of 100 KB) are full, and a 2-CTA split of
  the gate/up item breaks the 128-column Hadamard block of the epilogue (gate and up of the same 128 columns must
  meet in one CTA).
- ldmatrix traffic: the 128x16 warp tile reuses an A fragment for only 2 mma; a 64x32 tile halves A traffic but
  doubles the per-warp trellis decode unless the decoded B is shared through shared memory.

## 3. Real-engine checks (handoff MoE mini, TP=2 rank-0 shapes, `tests/optmoe/kl_chain.sh`, kit r16z2)

| arm | config | KL vs off | KL vs e4m3 | KL vs e4m3b | top-1 vs off |
|---|---|---|---|---|---|
| e4m3 | production (fp32 acc) | 0.00206 | - | - | 0.933 |
| e4m3x | e4m3 again (A/A) | 0.00205 | 0.0000286 | - | 0.935 |
| e4m3b | + ACC=bf16 | 0.00197 | 0.00069 | - | 0.935 |
| e4m3bx | + ACC=bf16 again (A/A) | 0.00198 | 0.00076 | 0.00068 | 0.942 |
| e4m3bf | + ACC=bf16 + FOLD_SHARED=1 (folded 28 of 28 served calls) | 0.00212 | 0.00094 | 0.00083 | 0.937 |
| e4m3t | production + TOKGATHER (7/7 layers) | 0.0020487 | 0.0000286 | - | 0.935 |
| e4m3bft | + ACC=bf16 + FOLD_SHARED=1 + TOKGATHER (28/28 folded, 7/7 TG) | 0.00199 | 0.00094 | 0.00085 | 0.939 |

The arms above e4m3t ran before TOKGATHER existed (per-pair gather). e4m3t's numbers equal e4m3x's to every printed
digit: the token gather reproduces the per-pair path exactly in the real engine.

## 4. Negative results (do not retry)

- MP=1, the epilogue's per-row metadata (token, router weight, row scale) and per-item vectors prefetched into
  shared memory with cp.async at item start: 31.55 vs 30.44 ms (slower). Debug builds only.
- EB=2 epilogue rounds (32 rows per round, half the barriers): 30.89 ms (4 stages) / 30.70 (3 stages) vs 30.60.
- 3 vs 4 pipeline stages: no difference (the mainloop is not load-latency-bound).
- Weight-traffic / L2 scheduling ideas (co-scheduling the segments of one expert): the all-weights-in-L2 probe
  bounds the gain at ~1 ms.
- Atomics-free down (TMA bulk reduce, or per-row outputs + a combine pass): bounded by the 1.2 ms the bf16 reds
  cost in total; a combine pass alone costs more.
- Shared-expert overlap on a side stream: the persistent fused kernel holds every SM (1 CTA x 512 threads, ~98 KB
  smem per SM); concurrent kernels only fill its tail. Folding the shared output into the accumulator (1.2) is what
  remains of that idea.
- Ticket prefetch (TKP: claim the item-after-next's ticket during the current mainloop): 30.93 vs 30.57 ms at
  13,824, 13.99 vs 13.55 at 4,289 (slower; fp32 path 36.27 vs 35.66). Debug builds only.
- lag (gate/up segments ahead of the downs): 10 / 12 / 14 / 16 / 20 = 30.21 / 30.19 / 30.11 / 30.12 / 30.19 ms at
  13,824 (flat; 6 = 33.07, worse). 12 stays.

## 5. Production recommendation

0. TOKGATHER is on by default once a kit ships this branch's module + extension (no env needed; `=0` restores the
   per-pair gather). Boot check: the summary line's "token gather 42/42 served layers" (per rank).
1. `GLM53_MOE_E4M3_ACC=bf16` with the current `GLM53_MOE_E4M3=1`: -0.71 s per 32k request (~+6 %), quality inside
   the e4m3 arms' own spread on the real engine. Kit work: ship this branch's `overlay/glm53_moe_e4m3.py` + the
   extension built from this `kernels/moe_e4m3.cu`, pass the env through start.sh (validate it like the other
   GLM53_MOE_E4M3_* values) and boot-check the summary line's "GLM53_MOE_E4M3_ACC=bf16 (bf16 accumulator)".
2. `GLM53_MOE_E4M3_FOLD_SHARED=1` on top: another -0.19 s per 32k request; check the boot log has no
   "GLM53_MOE_E4M3_FOLD_SHARED=1 NOT active" WARNING and the counters show folded == served.
3. Before the gate: the long-context KL and decode-vs-prefill KL on the full model (the mini cannot see long-range
   effects); expected cost is small (bf16 rounding of the routed sum, ~0.3 % of the e4m3 path's added variance).

Regression on the extension built from this branch's final source (TOKGATHER on by default, so every test below ran
the token gather wherever the real layer qualifies): test_tg 49/49, test_acc 87/87, test_fold 39/39, test_moe_e4m3
56/56, test_down16 45/45, test_wiring 37/37 (image module) + 37/37 (live production module), review_adv 16/16,
test_patch_moe_e4m3 17/17, identity vs the r16z2 kit build 6/6 (default path within the fp32 atomics-order class:
13,824 real rel-L2 2.1e-9, bf16 elements differing 1.6e-7 vs run-to-run 1.1e-7).
