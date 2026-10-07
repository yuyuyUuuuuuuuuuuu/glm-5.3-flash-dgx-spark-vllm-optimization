# Production rollout result — 2026-09-27/28 (nodeA head + nodeB worker, TP=2)

All numbers: launcher's own tests/bench_decode.py (TTFT excluded), n=10 per workload, temperature 0, measured on nodeA
with no concurrent request (num_requests_running=0). "ms/step" = decode_ms_per_draft_step_median.

| stage | change | prose tok/s (ms/step) | structured | coding | quality vs previous stage |
|---|---|---|---|---|---|
| B2 | before rollout (13 h / 14 d uptime, compaction off) | 31.53 (96.7) | 73.51 (108.3) | 39.45 (104.8) | - |
| R0 | both nodes rebooted, KV 1M | 31.88 (95.1) | 75.01 (106.1) | 40.76 (102.7) | cross-restart noise |
| R1 | (1) TF EXL3 MoE fork + (A) independent resample noise | **35.57 (84.8)** | **84.86 (93.8)** | **46.25 (91.1)** | prefill top-1 99.48-99.79% (noise 99.69-99.90) |
| R2 | (2) GLM53_DENSE_FP8 += mla,shared | **38.35 (80.9)** | **87.85 (90.6)** | **48.01 (87.4)** | prefill KL +~0.013 nats/tok over noise; top-1 99.58% |
| R3 | (3) drafter FP8 layers + drafter FP8 lm_head copy | **40.04 (77.2)** | **91.37 (87.1)** | **51.14 (86.1)** | at noise (KL 0.0066); acceptance unchanged |

R0 -> R3: prose +25.6 %, structured +21.8 %, coding +25.5 % (ms/step -18.8 / -17.9 / -16.1 %).

## Engagement evidence (per rank, both head and worker)
- fork: "tf_exl3_moe installed", 42/42 MoE layers "registered, self-test rel_l2 max 7.7-8.3e-4 (tol 2e-3, margin 2.2-2.8x)"
  on the REAL checkpoint (G3 resolved); every decode CUDA-graph capture size B=1..64 captured on the TF path (TF 1974 / prod 0).
- resample fix: "[glm53-resample-noise] ... patched; stock sha256 match" on both ranks.
- FP8: "dense fp8 groups: dense,kda,mla,shared"; drafter: "[glm53-draft-fp8] drafter linears -> FP8", "[glm53-draft-lmhead-fp8] ...".
- Incident during rollout: first R1 start passed the new env only to the worker (the head docker run lists env vars
  explicitly). Caught by the per-rank check; fixed by adding the 5 variables after GLM53_DENSE_FP8 in the head docker run.

## Launcher changes on nodeA (~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), all additive, backups kept
- start.sh: +17/-1 bundle wiring (tools/prodcheck/edit_start_sh.py) + head env passthrough (5 lines). Backups:
  start.sh.bak-tf-20260927-233117, start.sh.bak-tf-headenv-234249.
- overlay/patch_tf_bundle.py + overlay/tf/ (fork wheel contents + 3 overlay patches).
- .env: TF_EXL3_MOE=1, GLM53_SPEC_RESAMPLE_INDEPENDENT=1, GLM53_DENSE_FP8=dense,kda,mla,shared, GLM53_DRAFT_FP8=1,
  GLM53_DRAFT_LMHEAD_FP8=1, WORKER_GID=3 (moved 4->3 after nodeB reboot). Backups .env.bak-tf-*.
- Rollback: empty/remove those env lines (each feature independently; all inert when empty) and restart with
  ~/tf-exl3-deploy/restart2.sh (stop -> 25 s -> GID auto-fix -> SKIP_BUILD=1 start). Full revert: restore the backups.

## Node-level findings
- GB10 read bandwidth (spec 273 GB/s) decays with uptime and recovers on reboot: nodeB 238 -> 264 GB/s (14 d -> fresh),
  nodeA 250-256 -> 260 GB/s (13 h -> fresh). TP=2 runs at the slower node's pace. Not explained by fragmentation
  (compact_memory: no recovery), power capping (slow node had 0 ms capping), TLB (4 KiB random = sequential), or temperature
  (flat over 57-69 C). Root cause open; hourly state + decode-latency trend now logged on both nodes
  (~/tf-exl3-deploy/state/summary.csv, cron :17).
- vm.compaction_proactiveness=0 persisted on both nodes (/etc/sysctl.d/99-glm53-compaction.conf): stops the pgmigrate_fail storm;
  no measurable decode-speed effect in a quiet A/B (the earlier "+10 %" was confounded by a concurrent request).
- RoCE GID index can move across reboots; restart2.sh now auto-detects HEAD_GID/WORKER_GID (tools/prodcheck/fix_gids.sh).

## 2026-09-28 (R6-R9b): KV back to 2M, memory, bandwidth decay, lm_head FP8

Same bench (launcher tests/bench_decode.py, n=10, no concurrent request). ms/step = decode_ms_per_draft_step_median.

| stage | change (cumulative) | prose ms/step (tok/s) | structured | coding | nodeA / nodeB MemAvailable |
|---|---|---|---|---|---|
| R5-now | R5 re-measured 09:33 (KV 1M) | 77.61 (38.6) | 86.65 (91.86) | 83.21 (47.37) | 4.1 / 6.5 GiB |
| R6 | KV 2M + memprep before start + host memory hygiene + SIGUSR2 profiler | 76.99 (39.24) | 86.51 (92.01) | 84.09 (50.18) | 1.4 / 0.9 GiB, **3.5 GB swapped out on nodeA** |
| R7 | + NCCL_NCHANNELS=8 | 77.59 (39.72) | 86.46 (92.07) | 82.75 (48.02) | 4.0 / 3.9 GiB, no swap |
| R8 | + drafter fc FP8 (GLM53_DRAFT_FP8=layers,fc) + GLM53_SPINWAIT_MS=16 | 76.78 (39.61) | 86.39 (92.14) | 84.07 (47.89) | 2.4 / 4.0 GiB |
| R9b | + target lm_head FP8 (GLM53_LMHEAD_FP8=1), drafter FP8 lm_head copy off | **75.98 (40.05)** | **85.25 (93.38)** | 85.58 (58.72) | **4.5 / 4.5 GiB** |

Prefill (unique random prompts, max_tokens 1): R6 983 / 1050 tok/s at 6k / 24k tokens, R7 973 / 1055, R8 967 (6k): unchanged.

Quality (tools/prodcheck quality_probe.py, prefill top-20 KL nats/token): noise R8a-R8b 0.0078, R9a-R9b 0.0071; lm_head FP8
R8a-R9a 0.0113, R8b-R9b 0.0087 -> +~0.0026 over noise (the adopted mla,shared FP8 was +0.013). R3 vs R8 0.0083 (= noise).

Memory findings (glm53_runtime [glm53-mem] lines, per rank): the worker's 3.6 GiB of pinned host memory was 384 NCCL connection buffers of
9,633,792 B (Simple 4 MiB + LL 512 KiB + LL128 4.69 MiB) under NCCL's automatic channel count; NCCL_NCHANNELS=8 -> 0.53 GiB, decode and
prefill unchanged. malloc_trim returns ~0.6 GiB of heap after graph capture. lm_head FP8 frees 605 MiB BF16 (holds 303 MiB) and makes the
drafter's 303 MiB FP8 copy unnecessary.

Bandwidth decay (the "serious" item):
- GB10 cudaMalloc = 64 KiB chunks taken one at a time from the Linux buddy allocator, smallest free block first, GFP_KERNEL (unmovable)
  (open-gpu-kernel-modules 580.173.02: nv_alloc_system_pages; GB10B is PDB_PROP_GPU_ZERO_FB). Nothing can re-arrange them later.
- nodeC (11 d uptime), tests/bw_pagesize.py: 1 GiB chunks carved from >= 8 MiB free blocks read at 260-265 GB/s, chunks from scattered
  small blocks at 231-249 GB/s (the per-chunk buddyinfo deltas show which orders each chunk consumed). Earlier "random 4 KiB = sequential"
  did not rule this out (both patterns need one translation per 4 KiB).
- Fix: tools/memprep/memprep.py (no root): MADV_HUGEPAGE balloon up to MemAvailable-4 GiB (page-cache reclaim + direct compaction), then
  freed; run by start.sh (GLM53_MEMPREP=1) on both nodes right after the old containers are removed (6-8 s). After it: >= 8 MiB free
  blocks 85 -> 94-99 GiB, < 2 MiB fragments 11.7 -> 5.6 GB. Fresh allocations still vary per chunk (244-266 GB/s): scattered 32 MiB
  blocks do not reproduce a fresh boot's address-ordered allocation, so a periodic reboot keeps a residual ~2 % (B2 14 d -> R0 fresh).
- Watch: ~/tf-exl3-deploy/state/summary2.csv hourly (free >= 2/8 MiB, idle-only decode probe ms/step).

Step anatomy (R7 profile, rank 0, prose, 78 ms/step, GPU busy 96 %): MoE layers ~42 ms (TF grouped kernel 33.6 ms, bandwidth-bound),
KDA ~15 ms (FP8 Marlin ~350 us/layer), MLA ~7 ms, lm_head 2.65 ms (BF16; 1.38 ms after R9b), drafter ~5.8 ms, sampling ~1.3 ms,
NCCL all-reduce ~25 us x 102. docs/logs/prof-R7/rank0-kernel-table.txt.

## 2026-09-28 (R10-R12): NCCL channel A/B, E3 prefill, decode kernels

| stage | change | prose ms/step (tok/s) | structured | coding | prefill real text 8.5k / 15.3k tok/s | random 24k |
|---|---|---|---|---|---|---|
| R9b | (see above) | 75.98 (40.05) | 85.25 (93.38) | 85.58 (58.72) | - | - |
| R10 | NCCL_NCHANNELS 8 -> 4 | 75.23 (39.84) | 84.91 (93.75) | 85.72 (59.81) | 910 / 1030 | 1023 |
| R11 | NCCL back to 8; EXL3_FAT_GROUPED=1 + EXL3_TEMP_ROWS_FUSED=256 (E3 grouped kernels for prefill fat experts only) | 75.88 (40.64) | 85.32 (93.3) | 82.97 (50.27) | 968 / **1206** | **1290** |
| R12 | GLM53_FP8_GEMV=1 (MAX_M=16) + GLM53_BF16_GEMV=1 (router dedup) | **74.22 (40.59)** | **84.36 (94.35)** | **80.93 (52.88)** | 971 / 1214 | - |

- NCCL 4 channels: decode -0.4 %, but 24k prefill -3 % (large all-reduces) -> kept 8.
- E3 (image's exl3_fat_moe grouped kernels, previously 0 because it "did not fire in decode"): long-prompt prefill +17 % (real
  text) / +26 % (random tokens; routing collapses on random text, so real text is the number to quote). Decode unchanged (fused cap
  kept at 256, so the TF fork's decode domain is unchanged). +0.5 GiB scratch per rank.
- fp8_gemv.py (Marlin-layout FP8 small-M GEMM, no second weight copy) + glm53_gemv_install.py (router / indexer gates / drafter conv,
  fp32-exact; removes the MoE runner's duplicate router GEMM): decode -1.0 to -2.0 ms/step. Adversarial reviews: BF16 ship-candidate
  (both ranks must be enabled identically: verified by identical per-rank wiring summaries and plan hash); FP8 needs-fixes -> deployed
  with GLM53_FP8_GEMV_MAX_M=16 as the review required (M 17..64 entries not robust in a mixed step).
- Quality (prefill top-20 KL): R11 noise 0.0066, R12 noise 0.0067, R11-R12 0.0079 / 0.0059 (= noise). R9-R11 0.0088 / 0.0072 (noise 0.0071).
  Note: the quality probe texts are short, so E3 (fat experts, >256 rows) is barely exercised by it; E3 reconstructs the same weights
  and differs only in fp32 summation order (launcher default since 2026-09-07).
- The BF16 router kernel removes production cuBLAS's bf16 split-K rounding: 2.3-2.5 % of tokens at M <= 16 get the EXACT top-8
  instead of production's rounded one (measured on real router weights).
- Memory at 2M after R12: nodeA 3.3 GiB / nodeB 3.7 GiB available, no ongoing swap.

Prefill anatomy (16k cold, rank 0; from the design notes): two engine steps (13824 + 2202 tokens, Mamba-block aligned);
FP8 Marlin 18.8 % (KDA in_proj at only 28.5 TFLOPS because N > 4096 splits M into 1024-row launches that re-read a 51.5 MB weight),
EXL3 fat GEMM 23.6 %, MLA 8.9 %, KDA 7.7 %, mHC 7.2 %, NCCL 6.9 %. Next: Marlin column split for in_proj (~ -1.3 s per 16k),
a new E4 fat kernel (~ -0.7-1.3 s), NCCL Simple/LL128 for the 113 MB prefill all-reduces (~ -0.4 s).
