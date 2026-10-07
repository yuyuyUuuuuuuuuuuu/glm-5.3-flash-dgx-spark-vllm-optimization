# OPT_MOE2: a leaner fused mainloop for the e4m3 routed MoE (branch opt-moe2, base opt-moe-rev b0f2850)

All numbers: nodeC (GB10, 48 SMs, ~2.18 GHz under load), the production image, the real layer-10 experts of the TP=2
rank-0 shard (288 experts, K 4096, N_loc 1024), Gumbel-top-8 "real" routing, CUDA events, medians. The GPU was shared
with other jobs during these runs (absolute ms drift by ~±1 ms between runs; every A/B below is interleaved
in one process). Logs: `docs/logs/optmoe2/` (scratch copies in `${HOME}/tf-exl3-assets/opt-moe2/`).

## 1. What landed: `GLM53_MOE_E4M3_MAINLOOP=1` (opt-in, default off = the shipped kernel byte for byte)

Fused variants + 8192 (8192 / 8208 / 10240 / 10256 = variants 0 / 16 / 2048 / 2064 with the new mainloop, both
accumulators). Two changes, both instruction-count reductions of the same mainloop (the same smem layout, the same
fragments, the same mma sequence per accumulator):

1. **Lean trellis decode** (`decode_tile_lean`, `mcg2_lean`): the AND mask 0x8FFF8FFF and the XOR 0x3B603B60 have equal
   16-bit halves, so they commute with the half-word permutation; they are applied after the PRMT as ONE `lop3`
   (`(a & imm) ^ c`, the XOR constant held in a register), and the byte-aligned fields `(v >> 8) & 0xffff` are one
   PRMT. The same integer function (bit-identical fragments), ~80 fewer instructions per warp per stage.
2. **A copy address table** (`at_fill` / `at_load`): with 128 rows x 8 chunks per stage and 512 threads, thread t
   copies chunks t and t + 512 of every stage of a job, so its two source addresses (row / token via g_srt / 64-bit
   address arithmetic) are computed ONCE per job (at its first stage load: the prologue, or the first `ld_nxt` stage,
   when g_srt[pb ^ 1] is already visible) into a per-thread table in shared memory (2 buffers like g_srt, 16 KB behind
   the epilogue staging); each later stage is one 8-byte shared load + one add per chunk. Same bytes, same places.
   Dynamic shared memory 98,816 B per CTA + static 1,040 B (TG off) / 2,064 B (TG on: g_srt, the production
   default) = 99,856 / **100,880 B**, under GB10's 101,376 B per-block limit (headroom 496 B with TG; refuter
   correction, opt-moe2-rev: the earlier "+1,040 static" holds only for the TG-off variants).

### Measured (ms per layer call; before = shipped TG variant, after = MAINLOOP=1)

| measurement | T | before | after | delta |
|---|---|---|---|---|
| fused kernel, bf16 acc (`bench_parts`, 9 rounds) | 13,824 | 31.89 | 29.49 | **-2.40** |
| fused kernel, fp32 acc | 13,824 | 35.90 | 34.23 | -1.67 |
| fused kernel, bf16 acc | 4,289 | 13.61 | 13.17 | **-0.44** |
| fused kernel, fp32 acc | 4,289 | 15.86 | 15.65 | -0.21 |
| run(), bf16 acc, TG (`test_ms` timing) | 13,824 | 33.41 | 30.85 | -2.56 |
| run(), fp32 acc, TG | 13,824 | 38.46 | 36.86 | -1.60 |
| run(), bf16 acc, TG | 4,289 | 14.30 | 13.76 | -0.54 |
| composed ACC+TG+FOLD incl. S copy (`bench_e2e_ms`, N=31), warm / cold | 13,824 | 33.75 / 33.87 | 31.26 / 31.36 | **-2.49 / -2.51** |
| composed production default (fp32 acc, TG, cast, add), warm / cold | 13,824 | 42.99 / 43.29 | 41.23 / 41.48 | -1.76 / -1.81 |
| composed ACC+TG+FOLD, warm / cold | 4,289 | 14.31 / 14.45 | 14.28 / 14.00 | -0.03 / -0.45 |
| composed production default, warm / cold | 4,289 | 18.06 / 17.94 | 17.54 / 17.57 | -0.52 / -0.37 |

The composed 4,289 ACC+TG+FOLD warm pair is inside the noise of that arm (the first N=15 run gave -0.41 warm); the
kernel-level -0.3..-0.5 ms is the better estimate there.

clock64 phases (debug build, DBG 64, `prof_fused.py`, ms of a 48-SM machine), TG + bf16 acc:

| T | variant | wall | mainloop gu | mainloop dn | epi gu | actq | epi dn | ticket |
|---|---|---|---|---|---|---|---|---|
| 13,824 | shipped (3164) | 34.05 | 17.84 | 8.91 | 1.99 | 0.82 | 3.61 | 0.86 |
| 13,824 | MAINLOOP=1 (4460544) | 30.36 | **15.27** | **7.65** | 1.99 | 0.82 | 3.68 | 0.94 |
| 4,289 | shipped | 14.56 | 7.77 | 3.88 | 0.74 | 0.32 | 1.39 | 0.45 |
| 4,289 | MAINLOOP=1 | 14.24 | 7.43 | 3.70 | 0.78 | 0.34 | 1.48 | 0.49 |

All of the gain is in the mainloop (-14 % at 13,824), the epilogues are unchanged. Reading: the mainloop is
**issue-bound** - the lean decode alone removed ~80 of ~830 instructions per warp per stage and the time fell by the
same ~5 %; the address table removed another ~60 and gave another ~1 ms. Per 32k request (2 x 13,824 + 4,289 tokens,
42 MoE layers): 42 x (2 x 2.5 + 0.4) = **~0.23 s** of ~11.2 s with ACC+TG+FOLD [E: kernel-level x layer count, not a
gate measurement].

### Numerics (bitwise where the arithmetic is deterministic)

- `tests/optmoe2/test_ms.py` 166/166: for T 13,824 / 4,289 / 2,049 / 300 / 17 / 1, real + collapsed routing (one
  expert holding every row), half the experts non-local, x variants 0 / 16 (f16 down) x TG on / off x fp32 / bf16
  accumulator (72 combinations): **a16, a8d and dsc bit-identical** to the shipped mainloop for every computed row;
  the output vs the shipped one within the shipped run-to-run spread (13,824 fp32: 2.18e-9 vs A/A 2.25e-9; bf16
  2.09e-4 vs A/A 2.16e-4; all 72 lines in `test_ms2.log`), tokens with no local expert exactly 0, finite.
- Composed bench: the production-default arm with MAINLOOP=1 is **bitwise identical** to production at 13,824
  (rel 0.00e+00; 3.7e-8 at 4,289 = fp32 atomics order, production's own run-to-run 1.2e-6); ACC+TG+FOLD 4.71e-3 vs
  production, as without (4.72e-3).
- Knob off: the SASS of all 24 kernels of the shipped build (every fused / gather / gate-up / down / P16 instance) is
  instruction-for-instruction identical to the opt-moe-rev build (the fused kernels gained one template argument,
  MS = 0, nothing else).

### Regression (extension built from this branch, `docs/logs/optmoe2/def_*` = knob off, `ms_*` = every fused call of
the test forced to + 8192 by `tests/optmoe2/run_with_ms.py`)

| test | knob off | MAINLOOP forced |
|---|---|---|
| test_moe_e4m3 | 56/56 | 56/56 |
| test_acc | 87/87 | 87/87 |
| test_fold | 39/39 | 39/39 |
| test_tg | 49/49 | 49/49 |
| test_down16 | 45/45 | 45/45 |
| test_wiring (image module) | 37/37 | 37/37 |
| test_wiring (live production overlay df864b5) | 37/37 | 37/37 |
| review_adv | 16/16 | 16/16 |

## 2. Negative results (measured, do not retry as-is)

13,824 / 4,289, fused kernel, TG + bf16 acc, vs the shipped 31.8-32.3 / 13.6-13.7 of the same run:

| idea | 13,824 | 4,289 | note |
|---|---|---|---|
| B (trellis words) through shared memory with A (3 x 32 KB stages, staging aliased into the last-consumed stage), A fragments double-buffered, decode of step j+1 between step j's mma (smem-B "MS 1") | 32.50 | 15.93 | the restructure that frees 16 registers; +0.7 / +2.3 ms |
| same, 2 stages | 33.68 | 15.98 | prefetch depth matters once B rides with A |
| same, decode not pipelined | 33.77 | 15.99 | pipelining is worth ~1.3 ms inside this layout, not enough |
| same, single A fragment buffer | 33.04 | 15.74 | double-buffering A (ldmatrix latency chain) is NOT the limit |
| same + lean decode | 32.26 | 15.58 | |
| same + epilogue metadata fetched once per item into smem | 32.83 | 15.92 | metadata latency is not the epilogue's cost (also MP in OPT_MOE.md) |
| B words as a 2-slot register ring per k32 step (RS), shipped stage layout | 33.19 | 14.75 | |
| RS + lean decode | 31.09-31.29 | 13.91-13.96 | = lean decode alone at 13,824, worse at 4,289 (2-step lead too short for short tiles) |
| RS ring of 4 + lean | 31.32 | 13.68 | |
| lean + AT + compile-time B strides (branch on job kind) | 31.48 | 13.80 | the branch costs more than the 8 IMADs it saves |
| lean + AT + unpredicated full-tile mainloop (mbs == 8 specialization) | 30.03 | 13.16 | = lean + AT (30.06 / 13.21): the per-mb predicates (PLOP3 from uniform predicates) are cheap; not shipped (2x mainloop code) |
| AT without lean decode | - | - | 88 B of register spills at 128 registers; not measured |

Why the bigger restructures in the task list were not built (each fails a hard constraint that the measurements above
make concrete):
- **Warp specialization (producer decode warps -> smem -> consumer mma warps)**: decoded e4m3 B for the CTA's 256
  columns is 32 KB per k128 stage; with A's pipeline that exceeds the 99 KB of shared memory unless stages shrink, and
  the smem-B experiment above shows that moving B into the copy pipeline already costs +0.7..+2.3 ms. Producer warps
  also need registers the consumers (64 accumulators + fragments, 128-register cap at 512 threads; setmaxnreg does
  not exist on sm_121) cannot give up.
- **2 CTAs per SM / epilogue of tile i over the mainloop of tile i+1**: both need a second live accumulator set
  (128 x 256 fp32 per tile = 128 KB, or the registers of a second CTA). With the register file full, a second CTA
  halves the tile; a 64-row tile doubles the trellis decode per MAC (the decode is the issue-slot bottleneck shown in
  1.), and a 128x128 tile splits the 128-column Hadamard block of the gate/up epilogue across CTAs. An fp16/bf16
  accumulator would free the registers but changes the numerics (excluded).
- **64x32 warp tile with decoded B shared through smem**: halves ldmatrix traffic, but the single-buffer vs
  double-buffer A experiment shows the ldmatrix chain is not what limits the mainloop, and the B exchange adds a
  named barrier per k32 step plus 16 KB of smem.

## 3. Integration notes (nothing here is in any kit yet)

- Ship this branch's `overlay/glm53_moe_e4m3.py` + an extension BUILT from this `kernels/moe_e4m3.cu`
  (`tools/moee4m3/build.py`; the .so is not in git). Knob off, the new extension's kernels are the opt-moe-rev
  kernels instruction for instruction, so shipping it with MAINLOOP unset changes nothing.
- start.sh: forward `GLM53_MOE_E4M3_MAINLOOP` to BOTH ranks (head -e and the worker's serve env), validate ""/0/1 and
  refuse when the overlay module does not read it (as for the other GLM53_MOE_E4M3_* values). A rank mismatch is only
  a speed difference (the arithmetic is the same).
- Boot checks: install line contains "mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)", the summary line
  "GLM53_MOE_E4M3_MAINLOOP=1 (lean mainloop, fused variants + 8192)"; the per-layer load self-test runs the served
  variant (+ 8192), so a device that cannot give 98,816 B of dynamic shared memory fails the self-test and the layer
  falls back to production's path (logged), it does not crash serving.
- Combine with ACC=bf16 + FOLD + TG (the opt-moe recommendation): composed 13,824-token call 33.75 -> 31.26 ms.
  No new quality gate is needed for MAINLOOP itself (bit-identical intermediates, output in the atomics-order class);
  the ACC/FOLD gates of OPT_MOE_REV.md still apply to those knobs.

## 4. Files

- `kernels/moe_e4m3.cu`: `mcg2_lean` / `decode_tile_lean`, `at_fill` / `at_load`, `fmainloop` / `fprologue` template
  flags LEAN / AT (default 0 = unchanged), `me_fused_kernel` template MS (0 = shipped; 8 | 128 = MAINLOOP=1); the
  negative-result variants (smem-B `fmainloop_ms`, RS `fmainloop_rs`, BK, FULL, metadata) stay in the source, their
  launch cases only in `-DME_DEBUG_VARIANTS` builds (codes listed at the launch table).
- `overlay/glm53_moe_e4m3.py`: `GLM53_MOE_E4M3_MAINLOOP` (`ms_mode`, `MS`, sched key "ms", install / summary lines).
- `tests/optmoe2/test_ms.py` (numerics + timing), `tests/optmoe2/bench_e2e_ms.py` (composed A/B),
  `tests/optmoe2/run_with_ms.py` (run any test with the knob forced), `tests/optmoe/prof_fused.py` (accepts 4460544).

## 5. Refuter review (branch opt-moe2-rev; logs `docs/logs/optmoe2rev/`)

Independent re-check of every claim above, own builds of b0f2850 / b5ddd55 (plain `tools/moee4m3/build.py`).

| claim | verdict | evidence |
|---|---|---|
| lean decode is the same integer function | **confirmed, exhaustive** | CPU (`tests/optmoe2rev/dec_exhaustive_cpu.c`): `(v>>8)&0xffff == prmt(v,0,0x4421)` for all 2^32 v, mcg2 vs lean lo/hi words for all 2^32 (s0, s1); GPU (`dec_exhaustive_gpu.cu`, the device functions extracted verbatim): mcg2 incl. hadd2 0 / 2^32, decode_tile 0 / 2 x 2^32 w; negative control (mask off by one bit) 1.38e9 mismatches |
| knob off = opt-moe-rev SASS | **confirmed** | 24 / 24 kernels instruction AND encoding identical (`sass_compare.py`); the 8 new kernels are the MS = 136 instances |
| address table correct, no races | **confirmed** | the table is per-thread private (thread t writes and reads only `tab[2t], tab[2t+1]`), so it needs no barrier; `adv_ms.py` 376/376: tile edges (1/15/16/17/127/128/129/255/256/257 rows, first + last expert), T = 1/2/3, one local expert, half non-local, invalid ids, grid 1/2/3/full x lag 1/2/12/64, all 8 variants, intermediate buffers POISONED (0xFF) before every call (test_ms compares against the previous call's buffers, which would hide a skipped tile); grid = 1 (deterministic atomics) gives a **bitwise identical output** in every case; compute-sanitizer racecheck (grid 1 and 4) 0 hazards, synccheck 0, memcheck 0 (racecheck positive control on sm_121: 64 errors) |
| speed | **confirmed** | `bench_ab_ms.py`, same process, A B A' interleaved, 2 routings each: 13,824 bf16 warm -2.52 / -2.69 ms (15/15 rounds), cold (L2 flushed) -2.64 / -2.74 (30/30), A/A control -0.01..-0.03; fp32 -1.66..-1.82; 4,289 bf16 -0.49..-0.52, fp32 -0.25..-0.26 |
| composed ACC+TG+FOLD (`bench_e2e_ms.py`, re-run) | **confirmed** | 13,824: 33.60 -> 30.97 warm (-2.64), 33.68 -> 31.59 cold (-2.09); 4,289: -0.53 warm, -0.26 cold. Note: the production-default arm was 1.66e-7 vs production in this run (the 0.00 above was a lucky atomics order; grid = 1 in `adv_ms.py` is the bitwise proof) |
| test_ms 166/166 | reproduced | own build, 166/166 (`test_ms_rev.log`) |
| initcheck | inconclusive (tool limit) | with the kernel filter only the fused kernel is instrumented, so every read of a buffer written by another kernel (seg_tables, gather_tok, row_token) is reported (1.3 M, `san_initcheck_summary.log`); unfiltered runs take > 1 h. The poisoned-buffer bitwise runs cover the same question for the intermediates |
| self-test runs + 8192 | **confirmed** (192-token synthetic call, not production T) | launched variants [10240, 10240] (fp32 then bf16 accumulator) |
| smem-short fallback "does not crash serving" | **refuted, fixed** | a build asking for +8 KB more (`-DME_AT_SM_EXTRA=8192`): the self-test raises and the layer falls back, but `cudaFuncSetAttribute`'s error stays recorded as the thread's last CUDA error and the NEXT unrelated torch op raised `CUDA error: invalid argument` (`fallback_ms.py`, unfixed build). Fixed in `set_smem` (consume the error before throwing): the fallback then serves production's path and torch keeps working. Latent on GB10 (TG variants use 100,880 of 101,376 B), but only 496 B of headroom |
| static smem "+1,040" | corrected | 2,064 B for the TG (production default) variants |

Minor: the non-TG e4m3 MAINLOOP variants (8192, f32 / bf16) have 6 / 9 local-memory spill instructions (STACK 8);
production serves TG (10240), which has none.

Regression on the opt-moe2-rev build (kernels SASS = b5ddd55 for all 32, only the host `set_smem` changed):
test_moe_e4m3 56/56, run_with_ms test_tg 49/49, run_with_ms test_acc 87/87, adv_ms 376/376, fallback_ms (serve 4/4,
fallback with the fix 4/4; the unfixed build FAILS the fallback check).
