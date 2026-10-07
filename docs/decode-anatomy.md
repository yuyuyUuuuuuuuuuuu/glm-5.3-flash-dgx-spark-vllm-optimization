# Decode anatomy: where a step goes

The model runs one speculative decode step per output token batch. In a step, the DFlash2 drafter
proposes up to K tokens (K is adaptive, from the set {4,5,7}). The target model then verifies M = K+1
rows in one forward pass. tok/s = accepted tokens per step / step time. This page covers the step time.

Hardware context: each GB10 reads DRAM at 250-265 GB/s in practice (273 GB/s spec). TP=2 means each
rank reads half of the dense weights and its share of the routed experts on every step. The slower
node sets the pace.

## Bytes per step and the floor

| quantity | value | source |
|---|---|---|
| routed experts per rank | 6 MiB per expert per layer, 42 MoE layers, 288 experts, top-8 | checkpoint headers |
| routed bytes at T=1 | 8 x 42 x 6 MiB = 1.97 GiB per rank | same |
| active weights per token (09-22 estimate) | 11.02 GB, i.e. 5.51 GB per node, a floor of 20.2 ms at 273 GB/s | operator notes |
| verifier pass with speculation off (09-22) | 55.4 ms (18 tok/s), 36% of peak bandwidth | operator notes |
| weights-only floor of a full speculative step (09-28, R15) | about 52 ms against a 69 ms step | profile |
| routed MoE per layer, 10-05 | 768 / 869 / 1,028 us at M = 5 / 6 / 8 = the bytes of 29.3 / 33.1 / 39.2 distinct experts at 240 GB/s (of 40 / 48 / 64 slots) | 2-rank profile |

Speculation makes the step mostly a function of **distinct experts touched**. More verify rows touch
more experts. That is why ms/step rises about 2-4.7 ms per extra row (`data/decode_step_vs_rows.csv`),
and why trimming rows that will be rejected (VTRIM) saves time on low-acceptance workloads.

On 09-13 an estimate from routing statistics gave about 57.5 distinct experts per layer at k=7 (8 rows),
with a per-layer floor of 1,464 us against 1,434 us measured, so the MoE kernel was at its floor. The
10-05 figure is derived from kernel time at 240 GB/s instead, and the two methods disagree on the expert
count. Read both as "the routed MoE kernel runs at the bandwidth floor for the bytes it touches."
With speculation off, the same kernel was latency-bound: 7.9 experts, an 8.5 ms floor against 73 ms
measured.

## Four snapshots

### 2026-09-13, before our kernel work (fixed k=7, 124 ms/step, prose)
| component | ms/step | share |
|---|---|---|
| EXL3 routed MoE (`exl3_moe_kernel<4,256>`, 1,428 calls x 1,434 us) | 57.2 | 46.1% |
| bf16 wmma GEMMs (dense, attention projections) | 34.0 + 12.8 | 27.4% + 10.3% |
| NCCL all-reduce | 6.8 | 5.5% |
| KDA recurrent kernel | - | 1.5% |

A single verify pass without speculation was 72.7-73.1 ms (1.62 ms/layer). The bf16 GEMV ran at
246.9-249.2 GB/s, so the Ampere-named cutlass kernels were not the problem. The dense weights were simply
large in bf16, which led to FP8 for those weights (09-22, 09-28).

### 2026-09-28 R7 (grouped MoE fork + FP8 groups, 78 ms/step, prose, rank 0, GPU busy 96%)
- MoE layers about 42 ms, of which 33.6 ms is the grouped kernel (bandwidth-bound).
- KDA about 15 ms (FP8 Marlin about 350 us/layer).
- MLA about 7 ms.
- lm_head 2.65 ms (1.38 ms after lm_head FP8 in R9b).
- Drafter about 5.8 ms.
- Sampling about 1.3 ms.
- NCCL all-reduce about 25 us x 102 calls.

### 2026-09-28 R15 (69 ms median step, prose, rank 0, 39 steps, 2,103 kernels/step)
| category | launches/step | ms/step | of which < 15 us |
|---|---|---|---|
| routed MoE (grouped) | 84 | 29.36 | 0 |
| FP8 GEMV | 281 | 23.92 | 68 launches, 0.29 ms |
| NCCL all-reduce | 102 | 4.25 | (about 2.1 ms is waiting for the other rank) |
| bf16 GEMM | 126 | 3.55 | 42, 0.44 ms |
| MoE glue | 260 | 2.22 | 216, 1.20 ms |
| other small | 842 | 1.75 | 839, 1.68 ms |
| mHC | 185 | 1.68 | 181, 1.56 ms |
| KDA small | 102 | 1.42 | 68, 0.34 ms |
| MLA attention / indexer | 118 | 0.99 | 96, 0.30 ms |
| GPU idle | - | 2.99 | 1.04 ms of it in 1,592 gaps < 10 us |

- The FP8 GEMV for the KDA input projection ran at 205 GB/s: 99 blocks over 48 SMs gives a wave split.
  Small shapes ran at 150-170 GB/s.
- Context length barely matters. Stream gap p50 was 73-75 ms at 2k, 40k and 100k, with no stalls.
  Acceptance does not collapse above 500k (3.7 tokens/step).

### 2026-10-05, both ranks profiled together (74.2 ms profiled / 72.6 unprofiled, prose, mean M = 5.85)

Exposed time per step, rank 0:

| phase | span (ms) | notes |
|---|---|---|
| drafter (D) | 5.09 | exposed bandwidth 3.99, NCCL 0.69 |
| pre-target host work (PRE) | 2.95 | **2.59 ms idle**. A 65,540-byte pageable copy in the MLA attention plan, 4 bytes over the 64 KiB async-copy limit, blocked the host until the drafter graph ended. Fixed by pinned buffers (PLAN_PIN) |
| target verify pass (T) | 63.18 | exposed bandwidth 54.45, NCCL 3.17, small kernels 5.14 |
| post (sampling, commit) | 2.96 | |

| category (all phases) | ms/step |
|---|---|
| routed MoE | 34.66 |
| FP8 GEMV (dense, KDA, MLA, shared, lm_head; flat in M) | 22.46 |
| NCCL all-reduce + all-gather | 3.95 + 0.16 |
| bf16 GEMM | 2.83 (+0.48 small) |
| mHC small kernels | 1.80 |
| KDA small kernels | 1.05 |
| MoE glue | 0.81 |
| MLA attention / indexer | 0.78 |
| other small | 0.78 |
| KDA lazy state commit | 0.66 |
| drafter fc | 0.41 |
| idle | 3.23 |

By M (verify rows), rank 0:

| M | wall | bandwidth | MoE | FP8 GEMV | NCCL | small | idle |
|---|---|---|---|---|---|---|---|
| 5 | 70.42 | 57.63 | 31.34 | 22.36 | 3.90 | 5.63 | 3.27 |
| 6 | 75.09 | 61.92 | 35.58 | 22.45 | 4.10 | 5.84 | 3.23 |
| 8 | 83.11 | 69.04 | 42.34 | 22.78 | 4.76 | 6.16 | 3.14 |

- A node C model of only the bandwidth-bound kernels gives 43.6 / 52.2 / 54.9 / 59.5 ms for
  M = 3 / 5 / 6 / 8. The remaining 18-20 ms is latency:
  - 102 all-reduces per step and rank skew
  - about 1,500 of about 2,100 kernels per step run under 15 us
  - host gaps
- Kernel bandwidth: the lm_head FP8 GEMV runs at 236-242 GB/s and the KDA input projection at 215 GB/s,
  but (4096 x 2048) shapes reach only 163 GB/s and (4096 x 128) shapes 19 GB/s.

## What moved the step (prose ms/step, bench_decode)

| date | ms/step | main reason |
|---|---|---|
| 09-13 | 124 | fixed k=7, stock kernels (profiler figure) |
| 09-27 B2 | 96.65 | adaptive K {4,5,7}, dense FP8 for dense/KDA, upstream stack |
| 09-28 R1 | 84.76 | grouped EXL3 MoE decode kernel |
| 09-28 R3 | 77.20 | FP8 for MLA/shared, FP8 drafter |
| 09-28 R12 | 74.22 | FP8/BF16 GEMV for small M, router de-duplication |
| 09-28 R15 | 73.56 | R13-R15: kpool seed-stride fix, two RoCE rails, restart (small decode effect each) |
| 10-04 | 72.3 | KDA lazy state write-back (one fp32 write per step instead of per row) |
| 10-05 | 71.3-71.6 | pinned MLA plan buffers, one-hop all-reduce, drafter lm_head coarse-to-exact |
| 10-05 | **67.82** | verify-length trimming (tau 0.3) |

## What is left
- The bandwidth-bound part (routed MoE plus FP8 GEMV, about 57-69 ms) is at or near its floor for the
  bytes it reads. It can only shrink by:
  - reading fewer bytes. The operator ruled out lower-precision weights after 10-05.
  - touching fewer distinct experts, through fewer verify rows (VTRIM) or better drafter acceptance.
- About 13 ms per step is latency: NCCL about 4, small kernels about 5.8, idle about 3. The candidates
  are small-kernel fusion (an mHC boundary megakernel measured about 0.8 ms/step on node C and is still
  pending) and fewer host gaps.
- Communication is not the bottleneck: 18.5 MB per step, 0.8% link utilization. The fastest all-reduce
  backends are unavailable at world size 2 on GB10.
