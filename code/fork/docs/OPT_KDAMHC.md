# OPT_KDAMHC — KDA / mHC / NCCL / per-chunk glue review and cuts (branch `opt-kdamhc`, from `r16z2rev` 60b53bf)

Status 2026-10-03. nodeC only (one GB10); nodeA/nodeB read only (files, docker logs). Nothing deployed, no kit built.
Evidence: `docs/logs/opt_kdamhc/` (copies) and `${HOME}/tf-exl3-assets/opt-kdamhc/` (raw).

"Mini" below = the handoff mini engine (`tests/handoff/run.sh`, production image, production-composed container, 10
layers = 5 KDA + 5 DSA/MLA with per-rank TP=2 head counts, TP=1, one 13,824-token prefill step, production flags
`GLM53_PREFILL_QUICKWINS=all GLM53_MLA_PREFILL=1 GLM53_KDA_FLASHKDA=1 GLM53_DENSE_W8A8=1
GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj`), GPU kernel sum from torch.profiler
(`tests/opt_kdamhc/prof_driver.py`). Per-chunk production estimates scale per-call deltas by the production call
counts (90 mHC calls, 34 KDA, 11 MLA layers per 13,824-token chunk, ANATOMY).

## 1. Results (implemented and tested)

| # | item | switch | numerics | measured | per 13,824 chunk (est.) |
|---|---|---|---|---|---|
| 1 | **idx_gate quick win switched itself off at boot in production** (nodeA log 2026-10-02 08:07:51 UTC: `idx_gate result differs ... at M=16384; idx_gate turned off`, M=16384 = the dummy profile run): now a per-M verdict, NaN-aware | none (bug fix in `glm53_prefill_quickwins.py`) | bitwise (each served M still checked once) | mini: 5 x simt sgemm 2.1 ms + x.float() 1.4 ms -> 5 x qw_gate 0.66 ms; kernel sum 1067.5 -> 1053.6 ms; handoff `qw_all` ALL PASSED incl. `idx_gate_fast ran` | **-31 ms** (11 MLA layers; ANATOMY measured 41.6 ms spent on the slow path) |
| 2 | **fused mHC post + prenorm GEMM** (`kernels/mhc_post_prenorm.cu` cfg 9 = BM16/HT64 + register prefetch of the next chunk, `glm53_mhc_fused.py`, `overlay/patch_mhc_fused.py`) | `GLM53_MHC_FUSED=1` | residual_cur bitwise production's; the 24 logits in the decode branch's fp32 arithmetic (layer_input == decode branch's in 99.97 % of elements; production's tf32 prefill: 77 %) | whole op 4.996 -> 4.214 ms at the SP shard 6,912 rows (-0.78), 9.918 -> 8.377 at 13,824 (-1.54); post+GEMM 3.51 -> 2.61 ms at 6,912 (195 GB/s); prefill KL vs base on the mini 8.3e-4 (cfg 1; W8A8 subset: 1.3e-3 there) | **-70 ms** with MHC_SP (90 x 0.78), -139 ms without |
| 3 | **SP fp8 all-gather of the W8A8 KDA in_proj input** | `GLM53_SP_FP8AG=1` (needs MHC_SP + DENSE_W8A8) | bitwise in_proj output | quant 0.80 -> 0.43 ms per KDA layer; bf16 AG 56.6 MB -> fp8 28.3 MB + scales (2-process loopback 20.5-22.2 -> 11.3-12.5 ms incl. quant) | up to **-46 ms** NCCL (34 x ~1.35 ms at production wire rate) + -13 ms quant |
| 4 | `GLM53_MHC_SP2_K` knob (1 = odd-T SP only, no side-stream pipelining) | `GLM53_MHC_SP2=1 GLM53_MHC_SP2_K=1` | as SP2 | — | the **odd-T half of SP2 without any overlap risk** (see 2.2) |
| 5 | SP_FP8AG per-layer verdict agreed across ranks (MIN all-reduce, waits for the SP2 side stream) | — | — | test N1b/N2b | correctness (avoids a collective-size hang) |
| 6 | **MLA prefill writes back only the kv_indices rows an FA2 call can read** | default with `GLM53_MLA_PREFILL=1`; `GLM53_MLA_PREFILL_KV_ROWS=all` restores | every entry any FA2 call can read beyond its own rows byte-identical (test) | removes clamp 0.95 + DtoD 1.0 ms per MLA layer at 13,824 rows (production trace 2026-09-30) | **-21 ms** |
| 7 | fkda3 (direct output, fp32 state carried exactly) on the mini | `GLM53_KDA_FLASHKDA_V=3` (already in the r16z2 kit) | prefill KL vs base 1.6e-3 on the mini (state precision fix + no copy) | DtoD -1.1 ms per KDA layer, but its recurrence kernel +0.3 ms: step wall -1.9 ms over 5 KDA layers (`e2e_mini.log` noopt -> f3only) | **-13 ms** (34 x 0.4) |

**End to end on the mini** (production code = 60b53bf, the r16z2 kit's site/overlay, vs this branch with
`GLM53_KDA_FLASHKDA_V=3 GLM53_MHC_FUSED=1`; both with production's flags incl. `GLM53_KDA_STRIDED_QKV=1
GLM53_KPOOL_RING=1`, production top-k; untraced step wall, median of 7; `e2e_mini.log`):

| step | production | opt-kdamhc | delta |
|---|---|---|---|
| 13,824 tokens | 0.9839 s (14,050 tok/s) | 0.9204 s (15,019 tok/s) | **-63.5 ms (+6.9 %)** |
| 4,289 tokens (the 32k tail) | 0.3131 s (13,700 tok/s) | 0.2994 s (14,324 tok/s) | -13.7 ms (+4.6 %) |

Per feature at 13,824 rows (same harness): production 0.9839 s | + #1 + #6 (worktree, no switch) 0.9575 (-26.4 ms
for 5 MLA layers) | + fkda3 0.9556 (-1.9) | + #2 instead of fkda3 0.9277 (-29.8 for 20 mHC calls at 13,824 rows,
TP-sized) | all 0.9204. The mini has 10 of the 45 layers, dense MLPs instead of MoE and no TP, so the percentages do
not transfer; scaled by production's per-chunk call counts (1: 11 x 2.8, 2: 90 x 0.78 under MHC_SP, 6: 11 x 1.9,
7: 34 x 0.4) the four items save **~135 ms per 13,824-token chunk** and ~45 ms on the 4,289 tail (no SP for odd T
today, no idx_gate below 10,240 rows): at the 32k gate (31,937 tokens = 2 x 13,824 + 4,289, 12.19 s at 2,620 tok/s)
**~-0.31 s -> ~2,690 tok/s (+2.6 %)**, before SP_FP8AG (up to -60 ms per chunk if its all-gather is exposed) and
SP2's odd-T part (~-0.11 s per 32k request) -> up to ~2,740 (+4.6 %).

Memory: the transient peak of a 13,824-row step is unchanged (2,594 -> 2,602 MiB). The first cut of #2 checked its
first call against a full-size reference (453 MB at 13,824 rows + GEMM temporaries), which left torch's reserved pool
1.17 GiB larger after warm-up (10.13 -> 11.30 GiB, isolated to #2 by the f3only/fusedonly arms); the check now uses
the first 1,024 rows and the reserved pool is production's 10.13 GiB again (`fused2`). Quality on the mini (dvp_driver, 4 x 3,000-token real-text prompts, 96 greedy
tokens, vs the production arm): prefill KL 1.03e-3 (top-1 96.0 %), decode-vs-prefill KL 1.1e-5 (production arm 4.1e-5,
its repeat 1.6e-4: noise floor) - `dvp_mini.log` (optall). The production long-KL / dvp probe is the remaining gate.

### 1.1 Mini engine profile chain (13,824-token step, GPU kernel sum)

| config | kernel sum | note |
|---|---|---|
| production (worktree before #1) | 1067.5 ms | `prof_prod.log` |
| + #1 idx_gate fix | 1053.6 | `prof_qwfix.log` |
| + fkda3 + #2 (JIT hook) | 1036.9 | `prof_all.log` |
| + fkda3 + #2 through the real bundle (`GLM53_MHC_FUSED=1`, AOT .so) | 1034.9 | `prof_kitfused.log` |

These four profiles ran kl_driver's deterministic sort top-k in place of production's `topKPerRowPrefill` (found
2026-10-03, see 3.1), +~20 ms per MLA layer in every arm: the deltas hold, the absolute MLA glue is not production's.

## 2. Production readiness of the untested-on-TP=2 features

### 2.1 fkda3

FlashKDA has no collective: TP=2 only halves the heads per rank, and the mini already runs the per-rank shapes (H=32).
On the mini: engine boots with the fkda3 build, prefill KL 1.6e-3 vs the production build (the exact fp32 state
hand-off changes the numerics by design, KDA_FLASHKDA3.md A.1), decode-vs-prefill KL indistinguishable from the
mini's noise floor (base 4.1e-5, repeat 1.6e-4, fk3 5.1e-5). Nothing TP-specific is left untested; the production
long-context KL / dvp probe is the remaining gate.

### 2.2 SP2

Code review (patched model.py helpers): one communicator, the side stream serializes its NCCL ops, every compute-stream
collective in the SP2 window is preceded by `_sp2_sync()` or covered by the pending events, so no two NCCL kernels of
the TP communicator can run concurrently; the only exception found was the SP_FP8AG verdict all-reduce, fixed (#5).
What nodeC cannot show is the overlap on RoCE (loopback shares one GPU: the peer's spinning NCCL kernels time-slice
the mHC). Production fact (nodeA log): **SP is OFF for every odd-T prefill step today** (`sequence-parallel mHC prefill
off (T=2979)`, `(T=4175)`, `(T=1133)`, `(T=3939)` ...), including the 32k gate's 4,289-token tail. Recommendation:
A/B `GLM53_MHC_SP2=1 GLM53_MHC_SP2_K=1` first (odd-T SP only: the tail chunk's mHC 276 -> ~136 ms, ~-0.11 s per
32k request, no side stream), then K=2. Kit TODO (not done here, a make_start_sh stage): start.sh must forward
`GLM53_MHC_SP2_K`, `GLM53_SP_FP8AG` and `GLM53_MHC_FUSED` to BOTH ranks and boot_checks must compare them across ranks
(SP2_K and SP_FP8AG change collective counts/sizes: a mismatch is a hang).

## 3. Measurement findings

### 3.1 The mini profiles used a non-production top-k

`tests/opt_kdamhc/prof_driver.py` imported `tests/w8a82/kl_driver.py` and called `install_topk()`, which replaces
`torch.ops._C.top_k_per_row_prefill` with a deterministic stable full sort (Compare/where/neg/copies + radix sort:
~20 ms per MLA layer at 13,824 rows, +294 ms at 18,432 where torch switches to a segmented sort) — production runs
`topKPerRowPrefill` (0.85 ms per MLA layer, production trace). Found with `--stack-of sort` (Python stacks of the
aten ops). `--det-topk 0` is now the default for profiling; KL drivers keep the deterministic version on purpose.

### 3.2 Chunk size (MNBT 16,384 -> 18,432: the 32k gate in 2 steps instead of 3)

The scheduler aligns prefill chunks to the 4,608-token mamba block, so MNBT 16,384 gives 13,824-token chunks (2,560
tokens of the budget never used) and the 32k gate runs 3 steps (13,824 + 13,824 + 4,289). MNBT 18,432 gives
18,432 + 13,505 = 2 steps. Facts: KV capacity 2,003,436 -> 1,982,993 tokens (tests/kpoolring chain_check accounting;
the 1M context stays), production's step model has a fixed cost F ≈ 273-286 ms per step (ANATOMY linear fit; ~80 GiB
of weights per rank are re-streamed every step) -> ~-0.27 s per 32k request (+2.2 %) and ~-2 x F per 128k request.
Mini measurement (production top-k, `chunk_size_mini.log`): 13,824 rows 0.940 s (14,701 tok/s, transient peak
2.60 GiB above the post-warm-up baseline), 18,432 rows at MNBT 18,432 1.275 s (14,453 tok/s, -1.7 % per token,
transient 3.47 GiB = **+0.87 GiB**), 4,289 rows 0.307 s. Every GEMM / attention / mHC family scales within 1-2 % of
linear; the per-token loss is mostly the idx_gate quick win being refused at M=18,432 (cuBLAS reduces differently
there, so the bitwise check fails; +17 ms). Net for production ≈ -F + ~1-2 % of the per-token time ≈ -0.1..-0.15 s
per 32k request, for +0.9 GiB (more with MoE workspaces) of transient memory on nodes with 2-4 GiB available, +33 %
step time for concurrently decoding requests (MIXED_PREFILL=0) and a ~1 % smaller KV pool. **Not recommended now**
(kept as a measured negative/weak result); revisit only if activation memory is freed elsewhere.

## 4. Negative results (do not retry)

* vLLM's own decode-branch fused kernel `mhc_fused_tilelang` on prefill shapes: slower at every tile_n/n_splits
  (6.25-22.8 ms vs production's 5.07 ms at 6,912 rows; `bench_mhc_fused.log`).
* Fused post+prenorm with BM=64 tiles (fewer fn L2 re-reads): slower than BM=16/32 (wave quantization on 48 SMs).
* Overlapping a DRAM-bound mHC sub-chunk with a side-stream copy (NCCL stand-in): hides only 6-9 % (both streams
  share DRAM); GEMMs hide 48-78 % of the same copy (`overlap_probe.log`) — overlap comm with GEMMs, not with mHC.
* Larger chunks do not improve per-token kernel efficiency on the mini (13,824 vs 18,432: every GEMM/attention/mHC
  family within 1-2 % of linear; -1.7 % tok/s overall) — the MNBT gain is only the per-step fixed cost (3.2).

## 5. Ideas not implemented (with estimates)

* Overlap the attention/MoE output reduce-scatter with the output projection GEMM (split o_proj by the SP2 sub-chunk
  rows, RS sub-chunk 0 while the GEMM computes sub-chunk 1): ~-1.35 ms per KDA/MLA attention phase -> ~-60 ms per
  chunk if RoCE overlap behaves like nodeC's GEMM-vs-copy probe; TP=2 only, unmeasurable on nodeC.
* Fuse KDA's causal conv into FlashKDA's loads (K1/K2 read q/k/v straight from the strided in_proj output):
  ~-1.7 ms per KDA layer (~-58 ms per chunk); CuTe kernel surgery.
* Fuse layer_norm_gated into FlashKDA K2's epilogue: ~-1 ms per KDA layer (~-34 ms per chunk) if the V-split allows a
  per-head reduction.
* Skip the `full_like(-1)` fill in the MLA index conversion (the exact kernel reads only the valid prefix): -0.57 ms
  per MLA layer (~-6 ms per chunk).
