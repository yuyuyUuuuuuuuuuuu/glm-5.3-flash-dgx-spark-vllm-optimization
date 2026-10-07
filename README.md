# GLM-5.3-Flash on 2x NVIDIA DGX Spark (GB10) with vLLM: an optimization log

**Single-stream decode 99.4 tok/s structured, 62.6 coding, 46.6 prose, 45.5 Japanese prose (+35-59% over our pre-kernel baseline); cold 32k-token prefill about 3,000 tok/s; 2,003,436-token fp8 KV cache.** Two GB10 machines, tensor parallelism 2, EXL3 4bpw weights, DFlash2 speculative decoding. Numbers, failed experiments, wrong measurements and the code to reproduce them are all here.

| fact | value |
|---|---|
| model | GLM-5.3-Flash (also written GLM5.3 Flash, glm-5.3-flash): 320B total / 18B active MoE, 1M context |
| hardware | 2 x NVIDIA DGX Spark class machines (GB10), tensor parallelism 2, ConnectX-7 RoCE |
| weights / serving | EXL3 4 bpw (not NVFP4), vLLM fork with overlays, DFlash2 speculative decoding |
| decode, single stream | structured 99.44, coding 62.61, prose 46.58, Japanese prose 45.47 tok/s (production, 2026-10-05); n=10, temperature 0, thinking off, TTFT excluded |
| prefill | 32k-token cold prompt 3,046 tok/s (2026-10-03), same yardstick as a public GX10 report |
| KV cache | 2,003,436 tokens in 15 GiB of fp8 KV, `MAX_MODEL_LEN=1000000` |
| period | 2026-08-31 to 2026-10-06 |
| retracted or wrong measurements | `docs/what-did-not-work.md` |
| raw data | `data/*.csv`, `raw/` |
| licence | AGPL-3.0; the DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial) and is not included |

> Licence: AGPL-3.0 (`LICENSE`); upstream notices in `THIRD_PARTY_NOTICES.md`. To rebuild the setup, follow
> `REPRODUCE.md`. Note that the DFlash2 drafter used here is CC BY-NC-ND 4.0 (non-commercial).

This repository records five weeks (2026-08-31 to 2026-10-06) of work on serving **GLM-5.3-Flash**
(320B total / 18B active MoE, hybrid linear + sparse attention, 1M context) on **two GB10 machines**
(DGX Spark class) with tensor parallelism 2. It covers what worked, what did not, how much each change
moved the numbers, and how we measured them. It also lists the measurements that turned out to be wrong,
and why.

It is written for people running large MoE models on small-bandwidth unified-memory boxes. The key
constraint on this hardware is about 250-265 GB/s of DRAM read bandwidth per node, shared by the GPU,
the CPU and the NIC.

## Setup

| item | value |
|---|---|
| nodes | 2 x GB10 (128 GB unified LPDDR5X each, 273 GB/s spec, measured read ceiling 250-265 GB/s). Node A = head (rank 0), node B = worker (rank 1). A third single GB10 (node C) was used only for kernel tests. |
| interconnect | ConnectX-7 RoCE, NCCL. One rail first (112 Gb/s), two rails over two PCIe domains from 2026-09-28 (224 Gb/s) |
| parallelism | TP=2 (`world_size=2`) |
| serving stack | vLLM fork from the public Mia-AI-Lab "GLM-5.3-Flash-EXL3-2x-DGX-Sparks" launcher (vLLM `0.1.dev20051+g487ecf187` plus overlays), followed through upstream merges on 09-22 and 09-24 |
| weights | EXL3 4 bpw: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` until 2026-10-04, then `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold @6c5b2826` (same format, newer calibration). About 82.4 GiB of weights per node |
| speculative decoding | DFlash2 drafter (incoai), block size 8 (so K is at most 7), adaptive K from the set {4,5,7}, block verification for sampled decoding |
| KV | fp8 MLA KV on the SM90 sparse-MLA layout; `KV_CACHE_BYTES=16106127360` (15 GiB) holds 2,003,436 tokens; `MAX_MODEL_LEN=1000000` |
| local additions | a decode MoE kernel fork (TensorFold-style grouped EXL3 kernel), about 30 runtime overlays behind environment switches, all added through A/B tests. Each one can be switched off by itself (`code/`) |
| weight edit | from 2026-10-05 09:49, o_proj of layers 15-44 (plus the MTP layer 45, loaded but not used by the DFlash2 path; `ABLIT_LAYERS=15-45`) is replaced at load with the bf16 o_proj of the public uncensored checkpoint `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` ("ABLIT", to stop refusals of borderline requests). No speed change; long KL about +0.003, NLL +0.0035 (`docs/timeline.md`, 10-05) |

The model has 45 layers: 34 KDA (linear attention) and 11 MLA/DSA (sparse attention). 42 of them are MoE
layers with 288 routed experts each, top-8 routing, and mHC residual mixing.

Terms used below: **bpw** bits per weight. **KDA** Kimi Delta Attention (linear attention), **MLA/DSA**
multi-head latent attention with DeepSeek-style sparse key selection, **mHC** manifold-constrained hyper-connections
(several residual streams mixed per layer), **MTP** the multi-token-prediction layer shipped with the model.
**K** the number of draft tokens verified per step; **ms/step** the time of one draft + verify step;
**acc/step** accepted draft tokens per step. **MNBT** vLLM's `--max-num-batched-tokens` (the prefill chunk size).
**E3** the fused gather / gate-up / down EXL3 MoE prefill kernels. **R10, R14, r16k, ...** our successive deploy
kits; an **arm** is one configuration in an A/B run (e.g. `r16k-A`). **KL** KL divergence of the next-token
distribution against a reference configuration, **NLL** teacher-forced negative log-likelihood per token.
**TTFT** time to first token.

## Headline numbers

### Decode, single stream

Measured with the launcher's `tests/bench_decode.py`: n=10 per workload, temperature 0, thinking off,
TTFT excluded, run only when the server was idle (`num_requests_running=0`). The table shows tok/s, with
the median serving-cycle ms/step in brackets.

| workload | spec off (09-22) | tuned upstream recipe (09-22) | before our kernel work, "B2" (09-27) | current production (10-05 17:30) | B2 -> now |
|---|---|---|---|---|---|
| structured (count 1..200) | 18.04 | 70.46 | 73.51 (108.29 ms) | **99.44 (80.05 ms)** | +35% tok/s, -26% ms/step |
| prose | 17.87 | 31.16 | 31.53 (96.65 ms) | **46.58 (67.82 ms)** | +48%, -30% |
| coding | 18.24 | 42.90 | 39.45 (104.82 ms) | **62.61 (77.91 ms)** | +59%, -26% |
| Japanese prose | - | - | not measured | **45.47 (62.54 ms)** | (40.26 / 69.25 ms on 10-04) |

How to read this table:
- ms/step is the cleaner metric. tok/s = accepted tokens per step / step time. Acceptance on the coding
  workload moves from boot to boot (3.0-4.35 accepted/step), while structured is always 7.0. B2 coding
  was at 3.15 accepted/step and the current run is at 3.87, so part of the coding tok/s gain comes from
  acceptance.
- The first NVFP4 deployment (2026-09-02, a different and older benchmark) ran at about 30 tok/s
  (mixed workload). Moving to EXL3 brought that to 47-52 tok/s, before any of the work above.
- With 8 concurrent streams (10-04), total throughput reached 131.3 tok/s (prose) and 192.3 tok/s
  (coding). Single-stream numbers on the same day were 43.8 and 58.2 (`data/concurrency.csv`).

### Prefill (cold, idle server)

| measure | before | after | source |
|---|---|---|---|
| 32k-token cold prompt, kindling "gate" script (same yardstick as a public GX10 report of 2,929 tok/s on NVFP4) | 1,864 tok/s (09-30) | **3,046 tok/s** (10-03 20:14), 3,016.5-3,065.5 across the production-config arms of 10-04/10-05 | `data/production_ab_runs.csv` |
| real-text 15.3k prompt (R series) | 1,030 tok/s (09-28 R10) | 1,412 (09-28 R14: E3 prefill kernels, prefill fused cap, two RoCE rails) | `docs/timeline.md` |
| real-text 14.8k prompt (env-ab series) | 1,832 tok/s (09-30 production, r16k-A) | **2,959** (10-05) | `data/production_ab_runs.csv` |
| 97,359-token prompt, cold TTFT | about 60 s (09-28), 54.2 s (09-30 production) | **31.9 s** (10-05) | same |
| same prompt repeated (prefix cache), TTFT | 60.1 s (09-28: a bug dropped the cache, so the repeat was a full recompute) | **0.73 s** (10-05; 0.72-0.93 s in every A/B arm since 09-29) | same |
| ~100k-token prompt (09-07 MNBT tuning) | 958 tok/s (MNBT 7168) | 1,203 (MNBT 16384) | `data/prefill_scheduling.csv` |

Prefill went from about 1,000 tok/s (early September) to about 3,000 tok/s at 32k. Almost all of that
gain dates from 09-28 to 10-03. One of those steps changes numerics: e4m3 tensor-core routed-MoE prefill.
It was accepted on purpose (see "Quality" below).

### Capacity

The context window grew from 262,144 (NVFP4, 09-02) to 524,288 (EXL3), then to 512,000 and finally to
1,000,000 tokens. The KV pool grew from 731k to 2.0M tokens. Bytes per token fell from 15,591 (default
SM120 sparse-MLA layout) to 7,227 (SM90 layout). Memory per node stayed the same
(`data/kv_capacity.csv`).

### Quality

Every adopted change passed a set of probes. The probes are described in `docs/methodology.md`:
decode-vs-prefill KL, long-context prefill KL against stored reference outputs, NLL from a
teacher-forced harness, and output checks on all four workloads. Most changes are bit-identical or sit
inside production's own run-to-run noise. The exceptions were decided by the operator and are listed in
`docs/timeline.md`:

- FP8 for the dense MLA and shared-expert weights (09-28): +0.013 nats/token prefill KL over noise.
- e4m3 routed-MoE prefill (10-02): long KL 0.0049 -> 0.0132, top-1 98.65%. Accepted for +16%
  real-text prefill (turning it off on 10-04 cost 19.5% at the 32k gate). A later layer-band study
  did not buy it back at an acceptable speed cost.
- After 2026-10-05 the operator ruled out any further loss of precision for speed. Only
  output-identical changes, or changes within production nondeterminism, are allowed.

The weight swap to the TensorFold calibration (10-04) improved NLL by 0.0015 at zero speed cost. The o_proj
transplant of 10-05 (ABLIT, see Setup) costs NLL +0.0035 and stops refusals of borderline requests (0 of 13 refused,
operator notes); it was a deliberate choice, not a speed measure.

## Lessons

1. **Look for a same-hardware recipe before tuning.** "EXL3 cannot be optimized" was our configuration
   gap. The public same-hardware recipe beat NVFP4 on every axis: decode 30 -> 47-52 tok/s, weights
   192 -> 165 GB, KV pool doubled. On GB10, the NVFP4 GEMM in that image runs only when m is a multiple
   of 128. Decode shapes (m = 1-8) fail with "Error Internal", so NVFP4 brings no decode benefit there.
2. **Pick the metric that matches the user.** Batch throughput chose `MIXED_PREFILL_CHUNK=skip`. When we
   also asked how long a short question takes while heavy jobs run, the answer was 83.4 s with skip and
   0.76 s with 0. Two weeks later the same setting turned out to stall existing decoders behind 16k
   prefill chunks (38% of 10-s windows below 1 tok/s). Keep both views: the newcomer's wait and the
   incumbents' stall.
3. **n=3 lies.** A "+10.7% prose" gain at n=3 shrank to +0.1% (p=0.998) at n=10. We use n=10,
   interleave the arms, use a permutation test, and include A/A arms in every run.
4. **Prefill-only quality probes miss decode-state bugs.** In one rollout, long-context prefill KL looked
   normal (0.0048) while decode-vs-prefill KL was 0.294. A new MLA prefill kernel left a shared index
   buffer stale, and decode read another request's KV slots. Every prefill-path change now runs a
   decode-vs-prefill probe.
5. **One workload is not enough.** A KDA state-caching optimization passed the structured benchmark but
   corrupted Japanese output and cut coding acceptance from 4.13 to 2.73. Decode changes are now judged
   on structured, prose, coding and Japanese, plus a teacher-forced KL check.
6. **Single-node tests cannot show cross-rank effects.** Pre-reading weights into L2 "in idle DRAM
   windows" saved 5.8 ms in isolation and nothing in production. On GB10 the NIC shares DRAM with the GPU,
   so the all-reduce slowed down (p50 31 -> 50 us). Kernels that pass whole-graph capture on one node can
   also fail under vLLM's piecewise graphs in production.
7. **On GB10, distrust the usual instruments.** `nvidia-smi` reports memory `[N/A]` and utilization and
   power near 0. MemAvailable near 0 is normal for a pre-allocated vLLM, so watch swap activity instead.
   DRAM bandwidth decays with uptime (238 -> 264 GB/s after a reboot, operator notes) because the driver hands out 64 KiB
   chunks smallest-free-block first. Compacting memory right before engine start fixes it.
8. **Read the kernel before promising a saving.** MLA KV is 656 B/token/layer, fixed by a literal in the
   sparse-MLA kernel. "Not used arithmetically" is not "not read". Three predicted savings were wrong for
   this kind of reason.
9. **Check the hardware topology.** The CX7 on GB10 has two PCIe x4 domains. Two cables on two domains
   gave 224 Gb/s against 112 Gb/s for one. That step alone raised real-text 15.3k prefill from 1,324 to
   1,412 tok/s (+7%); the same morning's other prefill changes brought the total to +37-51%.
10. **Communication was not the decode bottleneck.** We ruled it out three times. A decode step moves
    18.5 MB (link utilization 0.8%). The real costs are routed-MoE weight reads (bandwidth-bound),
    FP8 GEMVs, and about 1,500 sub-15-us kernels per step (`docs/decode-anatomy.md`).

## Repository map

| path | contents |
|---|---|
| `REPRODUCE.md` | how to rebuild this setup on two GB10 machines: prerequisites, pinned downloads, build, install, every `.env` switch, boot checks, benches, expected numbers, known gaps |
| `docs/timeline.md` | dated, version-by-version history with numbers and adopted / rejected / reverted status |
| `docs/what-did-not-work.md` | rejected and reverted ideas with the measured reason, plus retracted conclusions |
| `docs/methodology.md` | how speed and quality were measured, and the measurement pitfalls we hit |
| `docs/decode-anatomy.md` | decode step-time breakdown against the bandwidth floor, at four points in time |
| `docs/test-status.md` | the fork's test suite step by step: our results, what to expect without the unpublished inputs, the cause of every non-PASS result |
| `data/*.csv` | per-version tables; `data/README.md` explains the columns |
| `raw/` | the raw bench JSONs, probe logs and A/B logs behind the headline numbers (`raw/README.md`) |
| `code/fork/` | the kernel fork: CUDA kernels, the vLLM integration modules, overlays, build scripts, tests, per-feature notes |
| `code/kit/` | the deploy kit production runs: launcher `start.sh` and overlays, bundle sources, `env.r16`, install / switch / boot-check tools, production `.env` without secrets |
| `code/operator/` | the bench and probe scripts behind every published number, the idle-gated restart, the second-rail helper |
| `code/build/`, `code/vllm-patches/`, `code/launcher-patches/` | extension build scripts, three vLLM file patches, one launcher patch |
| `LICENSE`, `THIRD_PARTY_NOTICES.md` | AGPL-3.0, and the notices of every upstream component |

`code/README.md` says what was left out of the code trees and why.
