# MOE2 — adversarial review + further work on GLM53_MOE_E4M3 (branch moe2, from combo d5600d0)

2026-10-02, nodeC only (one GB10, production image, `tests/gpu_run.sh` under `flock /tmp/tf-gpu-bench.lock`).
nodeA/nodeB untouched; no kit built or installed. Logs: `docs/logs/moe2/` (copies in `${HOME}/tf-exl3-assets/moe2/`).

## A. Verification

### A1 Speed — NOT REFUTED
Real layer-10 experts (TP=2 rank-0 shard), per `apply_exl3_experts` call, production = prefill cap 1 as deployed.

| T (real routing) | prod ms | e4m3 (combo) | delta |
|---|---|---|---|
| 13824 / 13856 (chunk + 32 decode tokens, mixed) | 63.6 / 64.1 | 41.6 / 41.9 | -22 |
| 4289 / 4321 (32k tail, mixed) | 21.3 / 21.7 | 17.4 / 17.5 | -4 |
| 384 / 300 / 260 (just above the 256 cap: small prefill + decode) | 8.8 / 8.7 / 8.3 | 8.1 / 8.0 / 7.6 | -0.6..-0.7 |

Collapsed routing same picture (`bench_adv_shapes.log`). Sustained 160-call bursts (6.5 s e4m3 / 10 s production): no
throttling, first-10 vs last-40 medians equal, x1.52 (`bench_sustained.log`). Faster than production at every size above
the cap, so the D1 gate (any call > 256 tokens) never costs speed.

### A2 Quality — the open risk, now quantified against EXL3's own noise
`tests/moe2/exl3_vs_e4m3_noise.py`: the SHARED expert of layers 3/10/23/43 is stored in BF16 and has exactly a routed
expert's shapes; it is quantized with the image's own exllamav3 0.0.43 (K=4, mcg, LDLQ, gate/up sharing H and su as in
the checkpoint) and the FFN output compared on held-out activations: BF16 ref vs production arithmetic (EXL3 4bpw) vs
the e4m3 spec (TP=2 per-rank down scale). r = added noise variance / EXL3's noise variance; dbpw = 0.5*log2(1+r) =
the bitrate drop of the routed experts that costs the same (D ~ 2^-2R).

| activations | EXL3 4bpw vs BF16 | e4m3 adds (r, dbpw) | e4m3 + fp16 down (r, dbpw) |
|---|---|---|---|
| Gaussian x real LN profile | 8.0-11.6 % | 0.29-0.31, 0.18-0.19 | 0.17-0.19, 0.12-0.13 |
| + token-norm spread + x30 outliers | 9.3-13.2 % | 0.22-0.25, 0.15-0.16 | 0.14-0.16, 0.10 |
| power-law spectrum (k^-1) | 5.7-7.9 % | **0.58-0.69, 0.33-0.37** | **0.35-0.43, 0.22-0.25** |

EXL3's LDLQ shapes its error away from the dominant activation directions; e4m3 rounding is input-agnostic (2.66 %
per operand). So the more anisotropic the real activations (they are), the LARGER e4m3 is relative to the noise the
quant already has: plausibly r ~ 0.6-1, i.e. the routed experts behave like ~3.6-3.7 bpw instead of 4.0. "6.5 % per
layer" is not small: it is the same order as the 4bpw quantization noise itself.
Error-reduction options measured (same harness): per-32 activation scales r 0.66 -> 0.60 (needs block-scaled mma,
power-of-2 scales only: not worth it); codebook prescale scan: best non-power-of-2 prescale 2.630 % vs 2.669 %
(`codebook_prescale_scan.py`, nothing); 2-term (hi+lo e4m3) activations halve the activation part but double the
mma of that GEMM (gate/up: +~10 ms, not worth it); **fp16 down** removes 2 of the 4 roundings for ~+4 ms — built (B1).

KL estimate (assumption-laden): GLM-4.7 EXL3 public table (mratsim): KL 4bpw 0.137 vs 8bpw 0.076 => ~0.06 of
quantization KL at 4 bpw, ~0.6-0.8 of it from the experts. KL(ON || OFF) ~ r x that ~ **0.01-0.03 (e4m3), 0.006-0.02
(fp16 down)** full-vocab; the production probes' top-5 KL reads lower. Against the probe band of every accepted kit
(quality_long vs R15: KL mean 0.0047-0.0049, p95 0.015-0.017, top-1 99.21-99.24 %), e4m3 may well be visible.

### A3 Failure modes (code read + tests re-run: test_moe_e4m3 56/56, test_wiring 37/37 image + live overlay,
review_adv 16/16 on the moe2 build)
- Boot self-test / fallback / summary / decode bound: as documented; production bound 4 x (7+1) = 32 <= 256 (start.sh
  defaults MAX_NUM_SEQS 4, DFLASH_TOKENS 7). OK.
- `kernels/moe_e4m3.cu:944` watchdog `__trap` and any device fault = sticky error, both ranks die, no automatic revert
  (recovery = `env_r16.sh off moee4m3` + restart). The A/B needs an operator standing by / a revert script.
- `overlay/glm53_moe_e4m3.py:227` one `_SYNC` ticket/flag buffer per device: two MoE calls running concurrently on
  different streams (vLLM DBO / micro-batching) would corrupt each other (hang -> trap). Not the case in production
  (no DBO); latent if that is ever enabled.
- `overlay/glm53_moe_e4m3.py:492` (combo :456) D1: decode/verify tokens co-batched with a prefill chunk get e4m3
  arithmetic too -> under concurrent load, generated tokens (and DFlash acceptance) see the e4m3 noise, not only prompts.
  The single-request probes do not exercise this.
- combo `overlay/glm53_moe_e4m3.py:184` `x2d.contiguous().half()`: an extra full pass over x per call (perf, fixed B2).
- Memory: none beyond production's (B2 removes the 113 MB fp16 copy per call).

### Verdict: GO for a production A/B (speed is real, failure handling adequate), with the quality gate below
deciding; NO-GO for enabling without it. Run BOTH arms (e4m3, and e4m3 + `GLM53_MOE_E4M3_DOWN=f16`).

KL gates (same kit, same texts; A/A first):
1. `quality_long.py` OFF twice -> A/A noise (expect <= 0.005).
2. ON vs OFF (`--compare`): **GO if KL mean <= A/A + 0.005 (absolute <= 0.010), p95 <= 0.035, top-1 agree >= 98.8 %**;
   ON vs R15: KL mean <= 0.010. Any arm above -> NO-GO for that arm.
3. kpool decode-vs-prefill probe (prefill e4m3, decode not: it measures exactly the mismatch): all-positions KL mean
   <= 0.010 (current 0.0076, R15 0.0062), top-1 >= 98.0 %.
4. A concurrent run (2-4 requests, one prefilling while others decode) + quality_short greedy texts: no regression.
5. boot_checks 42/42 served on both ranks; prefill gate: 13.8k chunk -0.89 s (e4m3) / -0.74 s (f16 down) +-15 %.

## B. Further work (commits on moe2)

B1 `GLM53_MOE_E4M3_DOWN=f16` (fused kernel variant 16, `DN16` template): the gate/up epilogue applies the down input
transform per 128-column item (H(fp16(act) * suh_d) * r -> fp16, in place of a16; no actq), down = mma m16n8k16 on the
fp16 trellis decode (production's down arithmetic). Off (unset / "" / "e4m3") = variant 0, code unchanged (DN16=0
instantiation). Any other value: install refuses. Needs kit forwarding of the new env to both ranks (not done here).
B2 gather reads bf16 x directly (rounded to fp16 in-kernel = `x.half()`): a8 bytes + scales bit-identical (tested incl.
fp16-subnormal / overflow inputs); default output == previous build within 1e-8 (fp32-atomics order).

Before/after per call (`bench_moe2_v2.log`, interleaved medians, real routing):

| T | prod | e4m3 combo | e4m3 moe2 | e4m3 + f16 down |
|---|---|---|---|---|
| 13824 | 63.88 | 42.28 | **41.27** (x1.548) | 45.13 (x1.415) |
| 4289 | 21.49 | 17.40 | 17.11 | 18.18 |
| 1791 / 512 / 300 | 12.93 / 9.34 / 8.57 | 11.74 / 8.62 / 7.95 | 11.56 / 8.75 / 7.91 | 11.82 / 8.89 / 8.34 |

Error vs production per layer (`test_down16.log`): e4m3 6.51 % -> f16 down 5.34 % (variance x0.67), kernel vs its spec
2e-4. Tried / rejected: lag 4/8/16/24 for variant 16 (12 is fine), separate bf16 contribution rows + reduce kernel
instead of fp32 atomics (exposed reduce ~4 ms > atomics' exposed ~6 ms minus zeroing/convert savings: no gain on
paper), shared-expert overlap (the persistent kernel holds every SM's register file: nothing to overlap with).
Per 13,824-token chunk (42 layers, scaled x0.936 to production's traced per-layer time): e4m3 moe2 **-0.89 s** (combo
-0.85), f16 down **-0.74 s**. 32k request (2 chunks + 4,289 tail) from today's ~2,110 tok/s: e4m3 ~-1.95 s ->
~2,420 tok/s; f16 down ~-1.60 s -> ~2,360 tok/s [E].
