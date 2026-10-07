# Timeline

All dates are 2026 and all times are JST (container logs were UTC, see `methodology.md`). Status values:

- **adopted**: in production, or a standing rule.
- **rejected**: measured and not adopted, or a conclusion that was later retracted.
- **reverted**: tried in production and rolled back.
- **pending**: not decided yet.

Benchmark conditions differ by period. They are stated with each entry and defined in `methodology.md`.
From 09-27 on, decode numbers come from `bench_decode.py`: n=10, temperature 0, idle server.

---

## Phase 0: choosing a stack (08-31 to 09-02)

### 08-31: feasibility survey (adopted)
- GLM-5.3-Flash was released 2026-08-26 (320B total / 18B active, MIT licence). FP8 is about 320 GB, which
  does not fit in 2 x 128 GB, so we needed NVFP4 or EXL3.
- Community numbers collected (conditions vary):
  - 2x Spark with MTP-5: 24.7-30.3 tok/s.
  - vLLM + DFlash2 (K=7, block 8) on an NVFP4 checkpoint: 46.9 tok/s at 74.1% acceptance.
  - EXL3 4 bpw: 164 GiB, KL 0.0246 (about FP8-level), structured 65.1 / prose 27.1 tok/s.
- Silent GB10 pitfalls reported by others:
  - The FLASHINFER_CUTLASS auto-selection for FP4 MoE on sm_121 produces degenerate loops without any
    error. Use the marlin MoE backend.
  - FlashInfer autotune looks like a 60+ minute hang but is normal.

### 09-01: NVFP4 checkpoint choice (adopted, later superseded)
- We compared three community NVFP4 variants and picked a compressed-tensors W4A16 checkpoint that keeps
  the vision tower. ModelOpt-derived NVFP4 checkpoints had reports of intermittent U+FFFD token corruption.
- Our first conclusion about which variant was "safe" was wrong. It was based on a tag rather than on how
  the checkpoint was produced.

### 09-02: first deployment, NVFP4 + DFlash2 k=7 (reverted)
- Decode was about 30 tok/s (code about 53, Japanese about 22). Weights 192 GB, host MemAvailable about
  2 GB, KV pool 1.257M tokens at a 262,144 window, 100K-token prefill 1,649 tok/s.
- Two same-day hangs forced physical reboots:
  - Bulk downloads stacked on a live 93 GB-resident server.
  - A restart with a larger batch while only about 2 GB was available.
- The kernel OOM killer never fired. The system froze first.

### 09-02: EXL3 4 bpw (Mia-AI-Lab recipe, same hardware) (adopted)
- NVFP4 -> EXL3:
  - decode about 30 -> 47-52 tok/s
  - 8K prefill 1,006 tok/s (the recipe reports 938-997)
  - weights 192 -> 165 GB
  - available host memory 2 -> 4-6 GB
  - KV pool 731k tokens at a 524,288 window
  - acceptance 63.6% (90% at position 0)
- A 1M window did not fit at first: 14.52 GiB of KV needed against 12.81 available.

### 09-02: MNBT 7168 -> 16384 by guess (reverted)
- This ignored the recipe comment. 100K prefill went 1,184 -> 889 tok/s, and short-prompt TTFT went
  0.49 -> 13.1 s.
- On 09-07 the same knob was re-measured under different settings and helped (see below).

### 09-02: reporting rules (adopted)
- A prefix-cache "17.7x" turned out to be a 2.5x cold-TTFT regression (cold 23.8 / 31.2 / 59.5 s, warm
  3.62 / 3.29 / 3.37 s).
- Rule: report seconds and tok/s, never bare ratios. Report TTFT, prefill and decode separately.

## Phase 1: making EXL3 usable for interactive tool-calling clients (09-03 to 09-17)

### 09-03: prefix cache hit 0% -> 38.4% (adopted)
- DFlash2 triggers EAGLE handling of all KV groups and drops long MLA hits.
- We applied a third-party patch from the same commit and same hardware to the hybrid KV-cache
  coordinator on both ranks. The cache block is 4,608 tokens, so hits come in multiples of 4,608.

### 09-03: `clear_thinking=true` default in the proxy (adopted)
- In a 4-turn conversation, prompt tokens went 760 -> 178. Past `<think>` blocks were being re-sent.

### 09-04: SM90 sparse-MLA KV layout on compute capability 12.1 (adopted)
- KV bytes/token: 7,227 (SM90 layout) against 15,591 (SM120 packed layout), a 2.16x difference.
- On 09-06 a rebuild without the patch silently fell back to SM120 at 23,201 B/token, and the pool
  shrank to 59%.

### 09-06: launcher config of record (adopted)
- KV 14 GiB = 1,688,048 tokens, 512,000 window, 8 sequences, MNBT 7168.
- Rules:
  - Never use `restart`: NCCL init races the old container's memory release. Stop, wait 20 s, then
    start.
  - Always skip the image rebuild, or the patches are lost.

### 09-06: mixed-prefill chunk sweep, "skip is optimal" (rejected on 09-07)
- Per-stream decode with 8 parallel streams: skip 18.9 tok/s, 1024 15.5, 7168 11.4, 0 10.7, 2048 3.9,
  3584 6.2.
- This measured only the running streams. See 09-07.

### 09-06: reasoning tokens broke every small `max_tokens` caller (adopted fix)
- Reasoning counts toward `completion_tokens`. With 8 sample jobs, the median was 2,427 tokens and the
  maximum was 8,000 (cut off).
- A cap of 2,000 would have failed 5 of 8 jobs. Caps were removed and timeouts raised.

### 09-07: MNBT 16384 (adopted)
- With 8 sequences and a 512k window, prefill at about 100k / 256k / 386k tokens:
  - 2048: 882 / 883 / 876 tok/s
  - 7168: 958 / 939 / 927
  - 8192: 972 / 959 / 936
  - 16384: 1,203 / 1,184 / 1,168 (4-parallel wall time 172.1 s against 217.8 s at 7168)
  - 32768: 666 tok/s, then a CUDA illegal memory access at about 40k
- The recipe's "never 8192" rule had no traceable source.

### 09-07: `GLM53_MIXED_PREFILL_CHUNK=0` (adopted; see 09-24 for its cost)
- Three heavy jobs plus one interactive request (MNBT 16384):

  | chunk | 3-job total | interactive median |
  |---|---|---|
  | skip | 377 s | 83.4 s |
  | 0 | 244 s | 0.76 s |

### 09-09: TP=2 restart procedure (adopted)
- Restarting only the head left 260 CLOSE-WAIT sockets on the worker and a broken NCCL, while
  `docker ps` reported healthy. Restart both ranks together.

### 09-10 and 09-12: "effective concurrency 1" retracted (rejected)
- Measured on an idle engine, aggregate tok/s scaled with concurrency:
  - prose: N = 1/2/4 gave 17.8 / 32.0 / 41.2
  - JSON (thinking off): N = 1/2/4/8 gave 41.0 / 52.4 / 87.7 / 131.4
- The earlier figure came from measuring while other requests ran, with `max_tokens=400` consumed by
  reasoning.

### 09-11: image limit 4 -> 64 (adopted)
- The value 4 was a leftover design value from another model.
- Image cost: 448x448 = 258 tokens; FHD = 2,693 tokens.

### 09-12: 1M window (adopted)
- `MAX_MODEL_LEN=1000000`, KV 15 GiB = 2,031,358 tokens (+20.3%), 7,931 B/token.
- No RoPE scaling was needed: `max_position_embeddings` is 1,048,576.
- A 3M pool would need NVFP4 MLA KV. That path worsened KLD 0.024555 -> 0.060535 (2.47x), so it was
  rejected.

### 09-12: shrinking MLA KV (rejected)
- 656 B/token/layer is a literal in the sparse-MLA kernel (`GLM_NSA=656`).
- MLA accounts for 88% of the per-token budget (7,872 of 8,905 B).

### 09-12: all-reduce backends (rejected as a lever)
- With world size 2, every faster backend is excluded: SYMM_MEM needs a world size of at least 4,
  others are ROCm-only or need NVLink, and GB10 has no GPUDirect RDMA.
- Communication is 18.5 MB/step, 0.74 ms on the wire, 0.8% link utilization.

### 09-13: first kernel-level decode profile (adopted method)
- Step 124 ms. `exl3_moe` 46.1% (57.2 ms), bf16 GEMMs 37.7%, NCCL 5.5%. A spec-off pass takes 72.7 ms.
- Also fixed a launcher bug: unset variables were forwarded to the worker as empty strings, so the worker
  exited and the head waited forever.

### 09-13: speculative length sweep (rejected; solved by adaptive K on 09-22)

| k | ms/step | prose | JSON | code (tok/s) |
|---|---|---|---|---|
| 1 | 83.4 | 18.7 | 23.6 | 22.5 |
| 3 | 99.0 | 20.3 | 37.2 | 34.4 |
| 5 | 111.4 | 17.5 | 45.2 | 34.1 |
| 7 | 124.0 | 15.5 | 52.5 | 34.7 |

### 09-13: eight decode experiments, all rolled back (reverted)
- FP8 draft model, fp8 drafter KV, MoE SM split, torch.compile, MTP k=3, drafter TP=1, the unfused MoE
  path, and an NVFP4 swap on the live cluster (OOM on both nodes after a healthy-looking start).
- Details are in `what-did-not-work.md`.

### 09-17: client prompt head and effort-line placement (adopted)
- The stable system+tools head shrank from 46,317 to about 15.5-16.7k tokens using lazy tool proxies.
- The reasoning-effort line moved from the start of the prompt to just before the last user message,
  via a chat-template kwarg from an upstream PR.
- Effects:
  - The first move of a new session went from 43 s to 2.6-6 s.
  - Switching effort no longer re-reads the whole head: 13,824 tokens are cached at every effort level.
  - Cost: "low" effort's output suppression roughly halved (completion tokens 208 -> 506).

### 09-18: production engine killed by parallel debugging (adopted safeguard)
- Nineteen parallel sessions ran `python -c "import torch"` inside the live serving container. The
  global OOM killer then chose the engine core, which had the largest RSS.
- The OOM was first misread because container logs are UTC and host logs are JST.

## Phase 2: following upstream, adaptive K (09-19 to 09-24)

### 09-22: upstream merge (402 commits) + adaptive K {4,5,7} + dense FP8 for dense,kda (adopted)

| config | prose | structured | coding | sum (tok/s) |
|---|---|---|---|---|
| adaptive K {4,5,7} (chosen) | 31.16 | 70.46 | 42.90 | 144.52 |
| adaptive K {4,7} | 27.36 | 71.66 | 38.77 | 137.79 |
| fixed K=7 | 23.91 | 69.84 | 43.28 | 137.03 |
| fixed K=4 | 31.10 | 54.06 | 38.90 | 124.06 |
| spec off | 17.87 | 18.04 | 18.24 | 54.15 |

- Turning dense FP8 off lowered the sum from 144.52 to 130.27.
- Rejected in the same A/B: thin-decode MoE, cooperative MoE overlay (-8.4%), a retrained third-party
  drafter (-6%), K set {4,5,6,7}, EMA tuning.

### 09-23: NVFP4 decode feasibility (rejected)
- In the serving image, `cutlass_scaled_fp4_mm_sm120a` runs only for m = 128, 256, 384, 512 and 640.
  Every m from 1 to 96 fails.

### 09-24: decode collapse under `MIXED_PREFILL_CHUNK=0` + APC retention + effort placement (adopted bundle)
- Diagnosis:
  - With chunk 0, 16k-token prefill chunks join every decode step. At 300k+ context one chunk takes about
    20 s.
  - Over 3 h, 38% of 10-s windows produced less than 1 tok/s.
  - Prefix-cache retention was only 331,776 tokens because each cached interval stores 1 attention,
    4 mamba and 3 SWA blocks.
- One restart applied:
  - 48 upstream commits
  - retention interval 32256 (LCM of 3584 and 4608) with SWA 0
  - the effort marker moved before the first user message
- 24 h before/after (`data/client_latency_0924.csv`):
  - p90 call wait for conversations over 300k tokens 532.6 -> 84.0 s
  - retries 3.8% -> 1.1%
  - decode per token 110 -> 77 ms
- Not adopted: a long-prefill threshold of 2048. It would cut single-stream prefill from 1,203 to
  882 tok/s.

## Phase 3: our own kernels in production, R0 to R15 (09-27 to 09-28)

Each "R" stage is cumulative. Results below are tok/s with ms/step in brackets (`data/decode_bench.csv`).

| stage | change | prose | structured | coding |
|---|---|---|---|---|
| B2 | before rollout (14 d uptime, KV 1M) | 31.53 (96.65) | 73.51 (108.29) | 39.45 (104.82) |
| R0 | both nodes rebooted | 31.88 (95.12) | 75.01 (106.13) | 40.76 (102.66) |
| R1 | grouped EXL3 MoE decode kernel fork + independent resample noise | 35.57 (84.76) | 84.86 (93.77) | 46.25 (91.08) |
| R2 | + dense FP8 on MLA and shared experts | 38.35 (80.87) | 87.85 (90.61) | 48.01 (87.36) |
| R3 | + drafter linears FP8, drafter lm_head FP8 copy | 40.04 (77.20) | 91.37 (87.12) | 51.14 (86.14) |
| R6 | + KV back to 2M, memprep, memory hygiene | 39.24 (76.99) | 92.01 (86.51) | 50.18 (84.09) |
| R7 | + `NCCL_NCHANNELS=8` (pinned host memory 3.6 -> 0.53 GiB) | 39.72 (77.59) | 92.07 (86.46) | 48.02 (82.75) |
| R9b | + target lm_head FP8 | 40.05 (75.98) | 93.38 (85.25) | 58.72 (85.58) |
| R11 | E3 grouped fat-expert prefill kernels | 40.64 (75.88) | 93.30 (85.32) | 50.27 (82.97) |
| R12 | + FP8 GEMV (M<=16) + BF16 router GEMV | 40.59 (74.22) | 94.35 (84.36) | 52.88 (80.93) |
| R14 | + two RoCE rails | 41.53 (73.78) | 95.47 (83.38) | 60.94 (83.92) |
| R15 | restart, reference state for later KL | 41.25 (73.56) | 95.27 (83.55) | 54.81 (81.80) |

Notes on these stages:
- **R1**:
  - Grouped kernel: on node C with synthetic data, 1.47-1.50x at T=1 and 1.16x at T=8, with
    rel_l2 <= 1.14e-3 against production.
  - The resample fix closes a quality bug: at temperature > 0, the draft and the post-rejection resample
    used the same Gumbel noise (chi-square 12,728; operator notes).
- **R2**: +0.013 nats/token prefill KL over noise, top-1 99.58%. This is the largest accepted quality
  cost in this phase.
- **R3**: KL 0.0066, which is noise.
- **R0 -> R3**: prose +25.6%, structured +21.8%, coding +25.5%.
- **R10** (`NCCL_NCHANNELS=4`): decode -0.4%, 24k prefill -3%. Reverted.
- **R11**: real-text prefill 8.5k / 15.3k 968 / 1,206 tok/s, +17% (random text +26%; quote real text).
- **R14**: link 112 -> 224 Gb/s. Real-text 15.3k prefill R13 -> R14: 1,324 -> 1,412 tok/s (+7%).
  Cumulative from R10 that morning (E3 kernels, prefill fused cap and the second rail): 15.3k
  1,030 -> 1,412 tok/s (+37%), 8.5k 910 -> 1,370 (+51%).
- **Bandwidth decay** (09-28):
  - GB10 read bandwidth fell with uptime (node B 238 -> 264 GB/s after a reboot; operator notes). The cause is the
    driver taking 64 KiB chunks smallest-free-block first.
  - Fix: a no-root `MADV_HUGEPAGE` balloon plus compaction for 6-8 s before engine start. It raised free
    blocks of 8 MiB or more from 85 to 94-99 GiB.
  - An hourly probe of free-block sizes and idle decode ms/step has run since then. The configuration
    changed many times over that period, so it cannot isolate an uptime effect.
- **R15 profile** (`decode-anatomy.md`) found two hidden bugs:
  - Prefix-cache re-use dropped to 0 when the fresh suffix was under 2,048 tokens.
  - The kpool indexer's 4-slot tail ring was overwritten by speculative verification.

## Phase 4: R16 kit, correctness first (09-29 to 09-30)

### 09-29: R16b bundle (reverted)
- About 14 minutes of downtime at first boot: side-stream fork/join crossed vLLM's breakable
  piecewise-graph boundaries.
- After boot, decode-vs-prefill KL was 0.294 (R15: 0.00625) while long prefill KL looked normal
  (0.0048).
- Bisect: 0.294 -> 0.17 -> 0.00647. Root cause: the new MLA prefill kernel did not write the shared
  `kv_indices` buffer.
- R16i, with the fix: dvp KL 0.0081, real-text prefill 1,748 / 1,821, random 24k 1,859 tok/s
  (1.8-1.9x R10).

### 09-29: L2 pre-read during decode (rejected)
- Saved 5.8 ms per FP8 GEMV step in isolation, nothing in production: all-reduce p50 31 -> 50 us,
  mHC 13 -> 22 us. CTA 8 -> 2 made prose worse (81.9 ms).

### 09-29: block verification for sampled decoding (adopted)
- At temperature 1.0 with the same 16 seeds: prose accepted/step 3.06 -> 3.18 (+4.3% +-4.8),
  tok/s 40.9 -> 42.2. It preserves the output distribution, so it was kept although not significant.

### 09-29: DSpark preview drafter (rejected)
- At temperature 1.0: 23-28% slower than DFlash2 + block (prose 30.5 vs 42.2 tok/s).
- Tuned at T=0: -15 to -20%. Two failed starts caused 15 minutes of downtime.

### 09-29: migrating the server to TensorFold (rejected)
- Its official 2x GB10 figures on the same checkpoint are 29.7-43.8 tok/s, below production.
- 4-stream structured: vLLM 146.5 against about 78.
- The API lacks logprobs and n>1, and about 30 production overlays would have to be rebuilt.

### 09-30: production R16J-opt, then +WAKE, then r16k (adopted)
- Keeping the FP8 drafter: removing it kept acceptance (prose 3.17 -> 3.11) but added 3.5 ms/step.
- `HOSTLOOP_WAKE=auto`: structured 83.11 -> 82.20 ms/step.
- r16k: KDA strided QKV. Everything measured inside noise.

### 09-30: KV 8.25 GiB + MNBT 32256 (rejected)
- No prefill gain (32k 1,849-1,851 against 1,846-1,853) and half the KV pool. Minimum free memory
  1.73 GiB against 4.0.

### 09-30: prefill target of 3,000 tok/s set
- The comparison point is a public GX10 report: 2,929 tok/s on NVFP4, TP=2, 32k cold, at most a 160k
  context.
- Production with their script: 32k 1,864 / 128k 1,842 tok/s.
- A step-0 trace of a 13,824-token chunk took 7.31 s: routed MoE 2,511 ms, dense+shared 1,411, KDA 1,033,
  mHC 882, MLA+indexer 589, NCCL 525.
- Switching to NVFP4 was rejected: the KV pool would shrink 2.0M -> about 1.1M and the 1M context would
  be lost.

## Phase 5: prefill 1,860 -> 3,046 tok/s (10-01 to 10-03)

32k gate in tok/s, idle and cold (`data/production_ab_runs.csv`). From 10-03 each value is the median of
about 6 samples. The 10-01 and 10-02 values are single samples or ranges over the logged samples:

| date | change | gate32k | quality |
|---|---|---|---|
| 10-01 | kpool "drop lowest-scored pool" | 1,860 -> 1,830 | dvp KL +45% -> **reverted** |
| 10-01 20:00 | FlashKDA chunked prefill (fp32 state) | 2,001-2,018 | short-text KL 0.0084 -> 0.0174, others unchanged; adopted |
| 10-02 06:04 | sequence-parallel mHC prefill | 2,086-2,121 | unchanged; adopted |
| 10-02 11:42 | e4m3 routed-MoE prefill | 2,223-2,455 (real 14.8k +16%) | long KL 0.0049 -> 0.0132; adopted by operator decision |
| 10-02 17:04 | W8A8 on kda.in_proj + mla.o_proj | 2,453 -> 2,620 | long 0.0140, dvp 0.0181; adopted ("all projections" 2,752 failed dvp 0.024) |
| 10-03 | kit r16z5rev (indexer-gate fix, MLA fused index, token gather) | 2,676 against 2,637 for the old behaviour | adopted |
| 10-03 18:01 | MoE bf16 accumulate + fold shared expert + leaner mainloop | 2,690 -> 2,956.5 | long 0.0145; adopted |
| 10-03 18:59 | fused mHC post+prenorm | 2,921.5 -> 2,988.5 | dvp 0.0135; adopted |
| 10-03 20:14 | fp8 all-gather of the KDA in_proj input | 2,982.5 -> **3,046** | long 0.0142; adopted |
| 10-03 | wider W8A8 / hi+lo / SP2 / round-A arms | up to 3,186 | long KL 0.016-0.025 -> rejected |

## Phase 6: quality accounting, weights, decode polish (10-04 to 10-05)

### 10-04: teacher-forced KL harness (adopted method)
- Seven base runs of identical production had a mutual long KL of 0.0139, about the same as their KL
  against the reference. Most of the "long KL" was production nondeterminism.
- The new harness scores assistant spans only. A/A: +0.00007 [-0.00005, +0.00020].
- Findings:
  - e4m3 is the main KL source. Turning it off gives NLL -0.0020 and long KL 0.0065, but costs 19.5% of
    prefill.
  - Exact layer bands buy back at most about 0.001 NLL for a 3-4.4% speed cost (rejected).
  - Restart-to-restart differences reach about 0.001 NLL, so smaller effects cannot be judged from one
    boot.

### 10-04 13:52: TensorFold calibration checkpoint (adopted)
- NLL -0.0015 on two boots (TR3 A/A -0.00001). Speed and decode unchanged (32k gate 3,036-3,092).
- After this swap, long and short KL against the old reference jump (0.014 -> about 0.055). The
  reference was produced by the old weights, so this jump is not a degradation.

### 10-04: KDA lazy state write-back (v1 rejected, v2 adopted 23:07)
- v1: decode-vs-prefill KL 0.0115 -> 0.206, coding acceptance 4.13 -> 2.73, Japanese output garbled.
- v2 keys layers correctly. ms/step went down on every workload: structured 82.2 -> 80.3, prose
  73.3 -> 72.2, coding 84.3 -> 81.9, Japanese 69.0 -> 67.8. Acceptance and KL unchanged.

### 10-05: kpool tail positions = 2 (adopted, correctness)
- On node C, two concurrent decodes corrupted 6 of 23 generated pools: every running request shared tail
  block 0.
- The production probe showed no difference, but production runs 4 concurrent streams, so the fix was
  adopted by rule.

### 10-05: pinned MLA plan buffers + one-hop all-reduce + drafter lm_head coarse-to-exact (adopted 15:17)
- A 65,540-byte pageable copy (4 bytes over the 64 KiB async limit) blocked the host until the drafter
  graph ended.
- All three are bit-identical. Mean ms/step: structured -1.26, prose -1.2, coding -0.6, Japanese -1.6.

### 10-05: o_proj weight transplant ("ABLIT", uncensoring) enabled in production
- What: from 09:49 (the boot started at 09:44) production replaces the attention output projection (`self_attn.o_proj.weight`) of
  layers 15-44 with the bf16 o_proj of the public uncensored checkpoint
  `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` (MIT). The purpose is to stop refusals of borderline
  requests. It is the published "dealign-oproj-transplant" recipe (METHOD.md of the Hugging Face repo
  `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45`, formerly `...-ablit-l15-45-anchorstock`).
- How: the public launcher's `ablit/fetch_transplant.py` range-fetches only the 31 o_proj tensors of
  layers 15-45 (2.5 GiB) from the donor shards. At load, `overlay/ablit_runtime.py` swaps them in on
  both ranks; nothing on disk is rewritten. Layer 45 is the MTP block, which the DFlash2 path does not
  use. Settings: `ABLIT=1 ABLIT_METHOD=auto ABLIT_LAYERS=15-45 ABLIT_INCLUDE_MTP=1`. A byte copy is
  enough because the EXL3 checkpoints keep o_proj in bf16, byte-identical between the TR3 and TensorFold
  calibrations (operator notes). The donor revision was not recorded; `data/ablit_transplant_sha256.csv`
  lists the sha256 of every tensor production loads.
- Speed: no change beyond noise. ms/step before (08:52 base arm) and after (10:01 bench, 10:17 base
  arm): structured 80.12 -> 81.30 / 80.50, prose 71.60 -> 72.89 / 72.24, coding 81.71 -> 81.15 / 82.26,
  Japanese 67.46 -> 68.21 / 67.72. A/A arms differ by about 1 ms.
- Quality (`data/production_ab_runs.csv`): long KL against the reference 0.054-0.055 -> 0.057-0.059;
  teacher-forced NLL 0.4998-0.5006 -> 0.5030-0.5040 (+0.0035 nats/token, larger than the 10-04
  calibration gain of 0.0015); decode-vs-prefill KL unchanged (0.010-0.012). Per-layer relative L2
  change of the swapped weights 0.026-0.036 on the first load (operator notes; the donor card quotes a
  "mean Δrel 0.126", apparently a different metric, not reconciled).
- Refusal probes (operator notes): with ABLIT off, 2 of 10 China-related political prompts were
  deflected and 0 of the Western ones; with ABLIT on, 0 of 13 borderline requests were refused.
- All later arms were measured with it on. Turning it off is `ABLIT=0` in `.env` and a restart.

### 10-05 17:30: verify-length trimming, tau 0.3 (adopted)
- Uses drafter confidence to skip the routed experts of rows that will be rejected.
- ms/step: Japanese 66.8 -> 62.3, prose 70.7 -> 67.5, coding 80.7 -> 77.7, structured unchanged.
  tok/s: Japanese +5-6%, prose +1-9%, coding about unchanged (fewer verified rows, lower acceptance).
  Output and KL checks passed.
- The strict coding rule (min of the on arm >= 0.99 x max of the off arm) missed by 0.4%. The first
  three runs of the on arm overlapped user traffic, so it was adopted by hand.

### Current production (10-05 17:30, verified 10-06)
- structured 99.44, prose 46.58, coding 62.61, Japanese 45.47 tok/s.
- 32k gate 3,016.5 / 3,043 tok/s in the two VTRIM arms (3,016.5-3,065.5 across all production-config
  arms of 10-04/10-05). KV pool 2,003,436 tokens.

### Pending
- **Decode mHC boundary megakernel**: about 0.8 ms/step on node C. It is held because the same L2
  pre-pull idea failed twice in production.
