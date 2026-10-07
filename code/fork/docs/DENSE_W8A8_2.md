# DENSE_W8A8_2 — adversarial review + further work on GLM53_DENSE_W8A8 (branch w8a82, from w8a8 58647dd)

2026-10-02, nodeC only (one GB10, production image, `tests/gpu_run.sh` / `tests/handoff/run.sh` under
`flock /tmp/tf-gpu-bench.lock`). nodeA/nodeB untouched; no kit built or installed. Evidence:
`${HOME}/tf-exl3-assets/w8a82/` (probe_gemm.log, lt_probe.log, cutlass_bench_*.log, prof_*.log, test_w8a82.log,
fp8_w8a8_unit.log, kl/*).

## A. Verification

### A1 Speed — NOT REFUTED (the saving is real), two accounting corrections
Per 13,824-token chunk per rank through the real hooked apply, production overlay exl3.py, same process
(`tests/w8a82/prof_shipped.py`, now incl. the KDA f_b/g_b projections):

| M | production | w8a8 (cutlass_scaled_mm pieces) | w8a82 (custom GEMM) |
|---|---|---|---|
| 13824 | 1347.9-1366.3 | 845.4 (-521) | **796.5 (-551)** |
| 4289 (32k tail) | 446.3-449.7 | 298.1 (-148) | **281.4 (-168)** |
| 1791 | 206.8 | 140.6 (-66) | 140.1 (-68) |

- The repack runs on EVERY call (no cache, by design): 39.9 ms/chunk (w8a8 v1) -> 32.6 ms (w8a82 v2).
- Per-token quant is **128-130 ms/chunk, not 90.7** (`bench_fp8_w8a8.py:111-137` timed `x[:2048]` x ceil(M/2048);
  the real call quantizes the full M; DRAM-bound at ~222 GB/s, 3 B/element).
- The doc's shape table omits **KDA f_b_proj / g_b_proj (4096x128, 68 calls/chunk)**; they ARE served (group kda;
  the mini-engine stats show them). Measured 0.645 vs 0.69 ms: a small win, not a defect.
- Thresholds: every M >= 512 is served and wins (512: in_proj ~1.0x, the rest 1.0-1.1x). Mixed prefill+decode
  batches are just a larger M (served; see A4 for the numerics consequence). CUDA-graph capture always declines.

### A2 Quality — the risk; measured on real activations
Real-engine A/B on the handoff mini (10 real GLM-5.3 layers, production quant path), `tests/w8a82/kl_driver.py`:
fp32 log-softmax at 996 prompt positions (4 x 4000-token real-text prompts), deterministic top-k:

| arm | KL vs prod-FP8 (off) | KL vs BF16 dense | top-1 vs off |
|---|---|---|---|
| off2 (A/A) | **0.0000** (bitwise) | | 100 % |
| prod FP8 weights (off) | — | **0.00445** | |
| W8A8 all projections (on, on2 = custom GEMM: identical) | **0.00427** (p95 0.0136) | 0.00801 | 92.0 % |
| W8A8 only kda.in_proj + mla.o_proj (onsub) | **0.00132** (p95 0.0044) | 0.00499 | 96.0 % |

Per-layer on real activations (first served call, 4096 rows): r = added activation-rounding variance / the variance
the e4m3 WEIGHTS already add vs BF16: median **0.97 (kda), 1.05 (mla), 1.15 (dense)**; gate_up 1.3-1.45, down_proj
0.29-0.55 (outlier-heavy inputs, amax/rms 40-70, put their energy in the large elements). Per-token scaling is
adequate: e4m3-subnormal energy share <= 3e-7, worst-token rel error <= 6.7e-2. **W8A8 adds as much KL as the whole
dense-FP8 weight quantization did** (0.00427 vs 0.00445 at the logits). No free error reduction exists: the 2.6 %
is the e4m3 mantissa (scale choice irrelevant at this subnormal share); hi+lo activations double the GEMM.

Production sizing: R2 (FP8 weights for mla+shared = 117 of 259 projection instances/forward) cost +0.013 nats/tok over
noise. With r ~ 1, W8A8 on all 259 instances ~ +0.02-0.03; the in_proj+mla.o subset (45 instances, 31 % of the
all-arm KL on the mini) ~ +0.006-0.009. For scale: MoE e4m3 (+0.0083 long-context KL vs R15, 0.0049 -> 0.0132)
FAILED its gates. **All-projections W8A8 is ~2-3x that increment and will fail the same gates; the subset is about
the same size as MoE e4m3. Stacking both is additive.**

### A3 Memory — the KV claim is wrong for production
Production pins the pool (`--kv-cache-memory`, KV_CACHE_BYTES=16106127360, env_nonsecret.txt:13, start.sh:337-343):
the KV pool does NOT shrink (the "2.003M -> 1.961M" line of DENSE_W8A8.md:49 assumes utilization sizing). The cost is
UMA headroom: persistent scratch at production shapes = one fp8 [maxN, K] per K in {4096, 8192, 6144, 1536, 1024,
128} = 121.6 MiB (+1 MiB GEMM workspace in w8a82), not 322 MiB (the bench also allocated the drafter-shaped
K=20480 buffer); subset = 81 MiB. Transients (fp8 activation <= 113 MB at K=8192) are comparable to the
large-M path's 128 MiB dequant it replaces, and are inside vLLM's profile run.

### A4 Failure modes
- Decline/fallback correct (unit 127/127 re-run; new 336/336). CUDA errors propagate like production's.
- Decode/verify tokens co-batched with a prefill chunk (M >= 512) get W8A8 numerics (`fp8_w8a8.py:try_w8a8`,
  58647dd:312): concurrent load moves generated-token logits and DFlash acceptance, not just prompts.
- Prefill W8A8 vs decode Marlin = a systematic decode-vs-prefill mismatch (MoE e4m3 moved that probe 0.0076 -> 0.0182).
- Latent: one repack scratch per (device, K) with no stream awareness (`_scratch`, 58647dd:153): two concurrent
  streams would race (not the case in production).

### Verdict: GO for a production A/B of the SUBSET only; all-projections NO-GO unless the gates are relaxed
Arms (same kit, W8A8 alone, MOE_E4M3 off): (a) `GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj`
(primary); (b) all projections (measurement only, expected to fail). KL gates (same as MoE e4m3, with A/A):
1. quality_long OFF twice -> A/A (expect <= 0.005).
2. ON vs OFF: KL mean <= A/A + 0.004 (abs <= 0.009), p95 <= 0.030, top-1 >= 98.9 %; vs R15 <= 0.009.
3. decode-vs-prefill KL <= 0.010 (current 0.0076), top-1 >= 98 %.
4. short prefill KL <= baseline + 0.003.
5. concurrent run (2-4 requests, one prefilling) + quality_short greedy: no regression.
6. If W8A8 and MoE e4m3 are ever stacked, they share ONE budget: summed increment <= 0.005 (A/A-relative).
Speed gate: 13.8k chunk -0.36 s (subset) / -0.55 s (all) per rank +-15 %; boot: every selected layer
"self-test passed: ... custom GEMM == cutlass_scaled_mm".

## B. Further work (commits on w8a82)

B1 **Custom CUTLASS 4.x SM120 GEMM** (`kernels/fp8_w8a8.cu` fp8_w8a8_gemm, ext VERSION 2): persistent TMA
warp-specialized kernels (coop 128x128x128 / pingpong 128x128x128 / coop 128x256x64 / coop 256x128x64) with the
EXACT ScaledEpilogue arithmetic of cutlass_scaled_mm, per-shape tile/swizzle/raster (`fp8_w8a8.py` GEMM_TABLE,
GEMM_TABLE_SMALL for M < 3072), ONE call over the whole M (the image kernel's single call was 2.9-3.4x slower at
M=13824 on the big weights - raster/L2 - hence w8a8's pieces). Output **bitwise equal** to cutlass_scaled_mm at every
shape, M in {512..13856} and every cfg (336/336), and checked per shape at load (fail-closed to cutlass_scaled_mm).
GEMM-only chunk: 630 -> 574 ms (bench), in-situ -49 ms at 13.8k, -17 ms at 4289. `GLM53_DENSE_W8A8_GEMM=cutlass_mm`
restores w8a8's path. cuBLASLt OUTER_VEC scaling is NOT supported on sm_121 (status 15, lt_probe.log).
B2 **Repack v2** (smem-staged, coalesced uint4 loads): byte-identical, 39.9 -> 32.6 ms/chunk (copy floor ~29.5).
B3 **`GLM53_DENSE_W8A8_ONLY`** projection filter (quality/speed dial; invalid names refuse install).
Error: unchanged by B1/B2 (bitwise). Total vs w8a8 at 13.8k: ~-56 ms/chunk/rank; 32k request ~-0.14 s.

Expected end-to-end (32k = 2 x 13,824 + 4,289, from ~2,110 tok/s): all projections -1.27 s -> ~2,300 tok/s;
subset -0.83 s -> ~2,230 tok/s; on top of MoE e4m3 (~2,450): ~2,710 / ~2,620 (quality budget permitting).

Not done (kit integrator): rebuild tf_fp8_w8a8_ext with setup.py (CUTLASS include path added), forward
GLM53_DENSE_W8A8_ONLY / GLM53_DENSE_W8A8_GEMM to both ranks, optional boot never-line "custom GEMM output differs".
Rejected/not worth it: resident repacked copy (3.5 GiB/rank of UMA for 33 ms), quant fusion into mHC/o_norm producers
(~45-60 ms possible but needs model.py edits on top of the MHC_SP/quickwins fingerprints; with MHC_SP an fp8
all-gather for KDA in_proj could also save ~46 ms NCCL - a separate, invasive project).
