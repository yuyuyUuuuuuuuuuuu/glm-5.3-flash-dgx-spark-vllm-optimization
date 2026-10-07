# MHC_SP2: pipelined sequence-parallel mHC prefill (branch mhc2, on top of r16x GLM53_MHC_SP; default OFF)

`GLM53_MHC_SP2=1` (needs `GLM53_MHC_SP=1` on the same rank, else the overlay refuses with SystemExit) runs
`overlay/patch_mhc_sp2.py` after `patch_mhc_sp.py` in the bundle. Unset/empty -> the bundle skips it, `0` -> prints
and touches nothing: the composed tree is r16x's byte for byte. Decode is byte-identical in every state.

## 1. Where the mHC time goes under r16x SP (nodeC, production kernels, `tests/mhc2/bench_mhc_kernels.py`)

Per call of the production fused post->pre (eager; log `docs/logs/mhc2/b1_kernels.log`):

| rows | mhc_post | tf32 prenorm GEMM | pre_big_fuse_with_norm | sum (us) | per 13,824-chunk (90 calls + 6 means) |
|---:|---:|---:|---:|---:|---:|
| 13,824 (TP) | 4,719 (216 GB/s) | 2,146 (212 GB/s) | 3,130 (181 GB/s) | 9,995 | 920 ms |
| 6,912 (SP shard) | 2,333 (218) | 1,134 (200) | 1,547 (183) | 5,014 | 462 ms |
| 3,456 (SP2 sub-chunk) | 1,160 (220) | 558 (204) | 751 (189) | 2,468 | 455 ms (x2 sub-chunks) |
| 4,289 (TP tail) | 1,446 | 685 | 942 | 3,073 | |
| 2,146 (SP tail shard) | 717 | 352 | 441 | 1,510 | |

Every kernel is DRAM-bound at 180-220 GB/s of ~240: there is no launch overhead left to remove (host gap 4-34 us per
call) and splitting the shard into 2 sub-chunks costs nothing (455 vs 462 ms). The only remaining kernel-level cut is
fusing the tf32 prenorm GEMM into mhc_post (saves one 4H re-read of the residual: <= ~100 ms/chunk), which cannot be
bitwise (a new fp32 summation in place of deep_gemm's tf32 split) and was NOT done. What is left exposed is the
collective time: per phase a reduce-scatter and an all-gather of 56.6 MB (~2.7 ms each on the RoCE pair, ANATOMY 4),
in series with the mHC.

## 2. What SP2 does

1. **Pipelined phases.** Rank r's shard is k=2 interleaved blocks (r and r+2 of 4 blocks of B = ceil(T/4) rows), so
   the AG of sub-chunk j yields global rows [2jB, 2(j+1)B) in order and the RS of those rows yields rank r's sub-chunk
   j. Per phase: `RS0 | RS1 || mHC(sub0) | AG0 || mHC(sub1) | AG1` with the collectives on a high-priority side
   stream (the TP group's PyNcclCommunicator, explicit stream; ordering by CUDA events, one communicator, never two
   NCCL ops in flight); across the layer boundary the MLP's RS is handed to the next layer's attention mHC as pending
   events. Exposed comm per phase: one half-RS + one half-AG instead of a full RS + AG.
   The mHC runs through `_sp2_post_pre_into`, a mirror of `mhc_fused_post_pre_tilelang` (same kernels, same args)
   writing into row slices of preallocated outputs (no concat copies).
   Pipelining only when every sub-chunk has >= 1,537 rows (compute_num_split == 1 there, so every row is bitwise the
   r16x SP / TP result): T >= 6,145. Smaller steps use r16x's single-shard SP.
2. **Odd T.** r16x required T % tp == 0; the stock `sp_shard`/`sp_reduce_scatter` already pad with zero rows and every
   gather is sliced back to T, so the parity gate is dropped: the request's tail chunk (4,289 at the 32k gate) and odd
   mixed steps get the halved mHC too (shard 2,145 rows: still split_k 1, bitwise).
3. Fingerprints: quickwins VERIFIED (mhc_aux/mhc_mean) and moeglue WARM_VERIFIED gain the SP2 forwards' fingerprints
   on the lines patch_mhc_sp.py extended. `patch_mhc_sp.py` (mhc2 copy) accepts an SP2-extended tree on a second
   bundle pass (verified through `patch_mhc_sp2.prepare`, fail-closed).

## 3. Verification (nodeC)

* `tests/mhc2/test_mhc_sp2.py` ALL OK (`docs/logs/mhc2/test_mhc_sp2.log`): patch mechanics (0 untouched, refuses
  without GLM53_MHC_SP=1 and on a non-SP model.py, idempotent, second bundle pass no-op); live fingerprints;
  quickwins transplant + moeglue warm accept; mirror == production op bitwise on 3,456 / 1,728 / 1,538-row
  sub-chunks; 2-rank emulation of the real patched forwards (production mHC ops, cross-token attention core):
  T = 6,400 (k=2), 6,403 (k=2, padded last block), 3,075 (odd, single shard), 4,096 -> every layer's state, aux and
  final hidden states BITWISE == TP on both ranks; collective counts == plan; decode T=8 TP; capture gate False.
  T = 3,000 (r16x even) and 3,001 (SP2 odd): shard < 1,537 rows -> split-K fp32-order differences of the SAME size
  (rel-L2 1.19e-2 after 4 random-weight layers in both cases; the stub network amplifies a per-op ~1e-5 difference).
* `tests/mhc2/nccl_sp2_pipeline.py` (`docs/logs/mhc2/nccl_sp2_pipeline.log`): real NCCL (vLLM PyNcclCommunicator,
  2 processes on the one GB10, socket loopback), real streams, helpers exec'd from the patched model.py, T = 13,824:
  serial (r16x) vs pipelined outputs BITWISE equal on both ranks; timing see section 4.

## 4. Gain

Measured on nodeC (logs `docs/logs/mhc2/nccl_sp2_pipeline.log`, `contention_probe.log`):

| probe | result |
|---|---|
| real NCCL, 2 processes on ONE GB10, socket loopback (RS 56.6 MB = 25.6 ms, ~9x production's) | per phase serial 95.73 ms -> pipelined 94.20 ms: **-1.53 ms/phase** (of 5.0 ms of mHC that could hide); serial == pipelined bitwise |
| same, one mHC sub-chunk with an RS slice in flight | 2.75 -> 9.40 ms (x3.4): confounded - the peer process's spinning NCCL kernels time-slice the same GPU |
| single process, mHC sub-chunk + side-stream copy kernel (NCCL footprint stand-in) | 8+ CTAs streaming at 190-215 GB/s: combined time == serial sum (DRAM is the shared bottleneck); 1-3 CTAs (32-93 GB/s alone): the copy is starved while mHC runs (mHC x1.02-1.05, combined 5.8-6.0 vs 6.0-6.2 ms serial) |

Model for production (each rank owns its GPU; NCCL is wire-limited at 21 GB/s and needs ~60-105 GB/s of DRAM for the
host-staged no-GDR copies of a 28.3 MB slice): an overlap can save at most the comm time minus the DRAM time its
copies add to the DRAM-bound mHC: 1.35 ms - (85..140 MB / 235 GB/s) = 0.75-1.0 ms per overlap, two overlaps per
phase, ~84 pipelined phases per 13,824-chunk (layer 0's attention phase and the 5 phases after mhc_aux reuse use the
synchronous layout-aware gather) -> **-0.13..-0.17 s per 13,824-token chunk** (upper bound without any contention
-0.23 s; lower bound 0 if NCCL's 8 LL CTAs are starved of DRAM by the mHC the way nodeC's 1-3-CTA copy is - nodeC
cannot tell which). The odd-T part is independent of that: the 4,289-token tail chunk's mHC 276 -> 136 ms, minus 90
zero-pad copies of the RS input (~0.3 ms each) = **-0.11 s per 32k request**.

32k gate (31,937 tokens = 2 x 13,824 + 4,289; ~15.9 s at ~2,010 tok/s with FlashKDA): MHC_SP alone ~-0.84 s;
SP2 on top -0.37..-0.45 s (pipelining 2 x 0.13..0.17 + tail 0.11) -> roughly +2.5-3 % prefill tok/s over MHC_SP, IF the
pipelining is not starved on RoCE. A/B it as its own arm AFTER MHC_SP alone passed; if the pipelined arm does not beat
MHC_SP alone by >= ~1 %, keep only the odd-T part (set `MHC_SP2_K = 1` in the patch: plain SP with odd T).

## 5. Risks / what the production A/B must check

* Same D-A hazard as MHC_SP, stronger: GLM53_MHC_SP2 must be equal on both ranks (a k=2 rank issues 2 RS + 2 AG per
  phase, a k=1 rank 1+1: collective mismatch = hang or corruption). boot_checks needs a `[mhcsp2-pair]` gate like
  `[mhcsp-pair]` (both ranks `[glm53-mhc-sp2] ... model.py: patched;`, env equal) - NOT in the r16x kit tools yet,
  and start.sh must forward GLM53_MHC_SP2 to both ranks (head -e, worker serve_env_names, bool validation, requires
  GLM53_MHC_SP=1) - a make_start_sh stage is needed; neither was built here.
* Concurrency on the comm stream is unmeasured on RoCE: whether the NCCL proxy's host-staged copies and the 8 LL
  channel CTAs slow the DRAM-bound mHC more than nodeC's loopback shows. Watch: per-chunk step time (prefill tok/s at
  the 32k/128k gate), and a torch.profiler trace for NCCL kernels on a second stream overlapping mhc_* kernels.
* Memory: per phase one extra [T,H] gather buffer is alive during the pipeline (as in r16x) plus the RS output; no
  growth vs r16x beyond allocator slack (record_stream defers reuse of the comm-read inputs to the side stream).
* Bitwise vs TP only at sub-chunk/shard >= 1,537 rows; 1,024 <= T < 3,074 keeps r16x's split-K tolerance (odd T
  now also enters it). Short-prompt quality (1-3k token prompts) should be part of the KL/quality gate.
