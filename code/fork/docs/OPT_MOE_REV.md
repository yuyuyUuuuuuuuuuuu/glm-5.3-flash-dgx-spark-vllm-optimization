# OPT_MOE_REV: adversarial review of branch opt-moe (d0b8544)

All on nodeC (GB10), the production image, real layer-10 experts (TP=2 rank-0 shard), the fresh extension built from
d0b8544's `kernels/moe_e4m3.cu` (`tools/moee4m3/build.py`; the .so is not in git: a kit must build it). Logs in
`docs/logs/optmoe_rev/` (copied from `${HOME}/tf-exl3-assets/opt-moe-rev/`).

## Re-runs of the author's tests
test_tg 49/49, test_acc 87/87, test_fold 39/39 on the fresh build. Timing reproduced (test_tg): 13,824 tokens
run_f32 39.82 / run_f32_tg 38.31 / run_bf16 33.99 / run_bf16_tg 32.72 ms; 4,289: 17.03 / 16.30 / 14.63 / 13.92.

## Composed A/B (tests/optmoe_rev/bench_e2e.py)
Each arm = fresh shared output S (copy, same in every arm) + routed call + what vLLM does to combine. Median of 15
per-call CUDA-event samples, arms interleaved; "cold" = 128 MiB write between calls (outside the timed region).

| T | cond | prod (f32, per-pair, cast, add) | f32+TG | ACC | ACC+TG | ACC+FOLD | ACC+TG+FOLD |
|---|---|---|---|---|---|---|---|
| 13,824 | warm | 44.39 | -1.31 | -7.40 | -8.27 | -9.12 | **-10.72** |
| 13,824 | cold | 44.05 | -1.19 | -6.85 | -7.97 | -8.88 | **-10.44** |
| 4,289 | warm | 17.98 | -0.58 | -2.59 | -3.04 | -3.12 | **-3.79** |
| 4,289 | cold | 18.55 | -0.96 | -3.11 | -3.54 | -3.61 | **-3.98** |

Error vs the production arm (bf16 result after the add): f32+TG **0 (bitwise)**, ACC / ACC+TG 4.28e-3, with FOLD
4.72e-3; production's own run-to-run 0 at that resolution. Request projection (kernel-level, NOT a gate measurement):
42 x (2 x 10.5 + 3.9) ms = 1.05 s of 12.19 s -> ~2,860 tok/s, as the author predicted.

## Default path unchanged (tests/optmoe_rev/bench_prodso.py)
fp32 / per-pair / variants 0 and 16 with this branch's .so vs the r16z2 kit's .so, two processes each: 41.01/41.69 vs
40.98/41.68 ms (13,824), 17.59/17.51 vs 17.76/17.64 (4,289); bf16 outputs differ in 0-5 of 17-57 M elements, the
same as two processes of one build (fp32 atomics order).

## TOKGATHER stress (tests/optmoe_rev/stress_tg.py) and racecheck
12 token counts (257 ... 16,384, item boundaries 2047/2048/2049, odd sizes) x real/zipf/collapsed routing x 3 reps:
36/36; worst fp32 TG vs per-pair rel-L2 1.6e-8 (atomics-order class), bf16-accumulator excess 2.3e-4. Host
compute-sanitizer 13.0 racecheck on T=300 TG calls (f32, bf16, bf16+DN16): no hazard reported (the tool's summary
line did not print inside the container, so this is weak evidence). All 43 checkpoint MoE layers (3..45) have one
w13 suh across their 288 experts (gate == up), so TG qualifies on every production layer.

## Real engine (handoff MoE mini, kit r16z2, this branch's module + build)
| arm | KL vs off | vs e4m3 (author's saved run) | vs e4m3b | top-1 vs off |
|---|---|---|---|---|
| e4m3x: production config, TG on by default | 0.0020569 | **0.0 (bitwise identical logprobs)** | - | 0.933 |
| e4m3bft: ACC=bf16 + FOLD + TG | 0.00184 | 0.00076 | 0.00065 | 0.940 |
Author's e4m3bft: 0.00199 / 0.00094 / 0.00085: the difference is the bf16 accumulator's run-to-run noise (author's
bf16 A/A 0.00068). folded 28 == served 28; the new log line fired in vLLM's profile run.

## Findings and fixes on this branch
1. FOLD was unobservable in production: `folded` lives only in in-process STATS, nothing is logged, so the author's
   boot check "folded == served" could not be implemented, and a FOLD that never fires (runner ineligible, aux-stream
   order, shape) would silently lose its 2 ms/layer. Fix: INFO "glm53_moe_e4m3 fold: first served call folded" once
   (fires in the profile run = boot-checkable) and WARNING "... was NOT folded: <why>" once.
2. FOLD under torch.compile: the wrappers keep Python state across the opaque moe_forward_shared custom op. Production
   runs CompilationMode.NONE (enforce_eager=False only for decode cudagraphs; checked in the head log), so this is
   fine today; with a compiled forward the state would be read at trace time. Fix: both wrappers pass through when
   `torch.compiler.is_compiling()`. tests/optmoe_rev/test_rev.py 9/9.

## Kit integration needed (none of this exists in r16z2)
- Ship overlay/glm53_moe_e4m3.py + an extension BUILT from this kernels/moe_e4m3.cu (not in git).
- start.sh: forward GLM53_MOE_E4M3_ACC, GLM53_MOE_E4M3_FOLD_SHARED, GLM53_MOE_E4M3_TOKGATHER to BOTH ranks (head -e and
  worker serve_env_names), validate (ACC ""/f32/bf16; FOLD ""/0/1 and 1 only with ACC=bf16; TG ""/0/1) and refuse when
  the overlay module does not read them (as for GLM53_MOE_E4M3_DOWN). Without forwarding, TOKGATHER=0 (its only env
  revert) cannot reach the containers: TG is default-on, which breaks the kit rule "switch unset = previous kit bytes";
  either make the kit set it explicitly or accept it as numerically transparent (bitwise in the engine test above).
- boot_checks: head == worker for the three values; need "accumulator: bf16 (GLM53_MOE_E4M3_ACC=bf16)", "token gather
  42/42 served layers", "glm53_moe_e4m3 fold: first served call folded"; never "FOLD_SHARED=1 NOT active", never
  "was NOT folded". A rank mismatch is not a paired-collective hazard (each rank's routed partial sum is independent),
  only a numerics difference.
- Quality gate before an A/B: full-model long-context KL and decode-vs-prefill KL with ACC=bf16 (+FOLD). The bf16
  accumulator makes prefill run-to-run non-deterministic at bf16 resolution (mini A/A KL 6.8e-4 vs 2.9e-5).
