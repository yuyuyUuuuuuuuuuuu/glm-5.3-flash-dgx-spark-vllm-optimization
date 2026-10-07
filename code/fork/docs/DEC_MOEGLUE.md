# DEC_MOEGLUE — the decode MoE layer's non-GEMV time (branch `dec-moeglue`)

Two independent switches, both default OFF (= production unchanged), both installed from `integrate.plugin_register`
(`glm53_moeglue.plugin_install`):

| switch | what | per MoE layer (rig, T = 5 corr40) | projected per step |
|---|---|---|---|
| `GLM53_DEC_MOEGLUE=1` (glue) | production's decode `apply_exl3_experts` in 5 launches instead of 11 | −3.0 to −4.0 µs (−0.3 to −0.5 %) | ≈ −0.15 ms |
| `GLM53_DEC_MOEGLUE_WARM=1` (warm) | L2 warm-up of the MoE sublayer's small weights during the o_proj all-reduce + mHC | −26 to −30 µs (−2.7 to −3.6 %) | ≈ −1.2 ms |
| both | | **−32 to −34 µs (−3.3 to −4.1 %)** | **≈ −1.35 ms (69.1 → ≈ 67.8 ms/step, ≈ 2 %)** |

Numerics: warm is bit-identical by construction (it only reads) and tested bitwise; glue is bit-identical with one
route per token and differs only in the fp32 summation order of a token's 8 routed contributions otherwise (rel_l2
≤ 7.4e-5 vs production, whose own atomics order is not deterministic). Extra memory: ≈ 4 KiB + 16 B per rank.
Everything below was measured on nodeC (one GB10) in the production image, against the live launcher overlay
`exl3.py df864b5` (`GPU_RUN_BIND`), paired and interleaved inside CUDA graphs; logs in `docs/logs/moeglue/`.

## 1. What production actually does per decode MoE layer

`tests/prod_moe_layer_stats.py` over the R15 rank-0 trace (40 steps × 42 MoE layers = 1680 layer calls,
`docs/logs/moeglue/prod_moe_layer_stats.log`), medians:

| piece | µs |
|---|---|
| o_proj end → router start (all-reduce 22.9 (mean 32.1) + mhc_fused_post_pre + gaps) = the **window** | 44.9 (p10 41.7, p90 62.7, mean 54.3) |
| router GEMV start → gate/up grouped GEMV start (`pre`: router, topk, ids `.to(long)`, zeros, route_ids, rot_in) | 56.4 |
| grouped gate/up / down | 452.1 / 219.9 |
| gate/up end → down start (`mid`: gateup_epilogue) | 8.4 |
| down end → last main-stream kernel (`post`: down_epilogue, cast, shared add) | 10.1 |
| **shared expert end − down GEMV end** | **−229.4** (p90 −177.6) |

- **The brief's lever (1) is already in production**: vLLM's `SharedExperts` runs the shared expert on its aux stream
  (`maybe_sync_shared_experts_stream`, forked after the router), and it finishes ~230 µs before the routed down GEMV
  in every one of the 1680 calls. The trace's shared-expert kernels are on stream 13919, the MoE's on the per-segment
  graph streams. There is no serial shared expert to hide.
- rot_in takes 23.4 µs in production (4-6 µs alone) because it runs next to the shared gate/up GEMV (8 MiB, grid 32).
- The layer is bandwidth-bound: at T = 5 corr40 (27.1 distinct experts) it reads router 2.25 MiB + shared 12 MiB +
  experts 162.6 MiB ≈ 185 MB; at the measured read ceiling (239-243 GB/s, `docs/logs/bw_ceiling2.log`) that is
  ≈ 770 µs; the rig layer (router → end) is ≈ 800-815 µs, i.e. **≈ 95 % of the achievable DRAM rate**. Removing small
  kernels only helps where the DRAM is idle; inside the layer that is ~10-15 µs (pre / mid / post), so glue is small.
- The one DRAM-idle stretch next to the MoE is the **window** before it: the o_proj all-reduce (network-bound) and the
  mHC kernels (≈ 1.6 MB of reads). ~45 µs × 240 GB/s ≈ 10 MB could be read there. That is what warm uses.

## 2. The rig (tests/moeglue_rig.py, tests/bench_moeglue_layer.py)

A production-shaped decode MoE layer, 42 layers (3 trellis sets, distinct per-layer suh/svh, router, shared expert
and mHC fn weights, so nothing is L2-resident across layer calls), one call per layer per graph replay:
`[window: spin (all-reduce stand-in, 22 µs by default) + production's mhc_fused_post_pre (tilelang, hc 4)] → router
(glm53_bf16_gemv plan) → fork aux → grouped_topk → apply_exl3_experts (production K2, or the hooked glue) ∥ aux: shared
FP8 gate_up → silu_and_mul_with_clamp → FP8 down (fp8_gemv kernels, production TABLE) → join → add`. Routing is
pre-drawn (corr40 / rand) so the bytes match production's distinct-expert counts. Each mode is its own CUDA graph;
rounds interleave the graphs (order rotated / reversed every round), 101-151 rounds; `base2` is a null repeat of
`base`. `--dump` prints one layer's kernel timeline (µs from the router start, stream).

## 3. Results

### 3.1 Final A/B (per layer call incl. the window; `rig_final3.log` corr40, `rig_final2.log` rand T=1 + corr40)

| T | base (production) | null | glue | warm | glue + warm |
|---|---|---|---|---|---|
| 1 corr40 | 319.5 µs | ×0.9998 | 316.1 (×0.9887) | 290.2 (×0.9076) | **288.8 (×0.9040, −30.7 µs)** |
| 5 corr40 | 823.6 | ×0.9994 | 819.6 (×0.9956) | 794.2 (×0.9645) | **789.8 (×0.9594, −33.8 µs)** |
| 8 corr40 | 987.7 | ×1.0003 | 984.7 (×0.9966) | 961.3 (×0.9732) | **955.6 (×0.9664, −32.1 µs)** |
| 1 rand (run 2) | 319.3 | ×1.0004 | ×0.9968 | ×0.9162 | ×0.9150 |
| 5 corr40 (run 2) | 826.5 | ×1.0001 | ×0.9965 | ×0.9652 | ×0.9620 |
| 8 corr40 (run 2) | 988.0 | ×0.9989 | ×0.9968 | ×0.9723 | ×0.9676 |

Ratios are medians of the per-round paired ratios; p25-p75 of glue+warm at T = 5: 0.9562-0.9614 (null 0.9960-1.0021).
Window length sensitivity at T = 5 (`rig_final_w10.log`, `rig_final_w50.log`; spin 10 / 50 µs + mHC):
glue+warm ×0.9744 (−21 µs) / ×0.9439 (−49 µs). Production's window (median 44.9, mean 54.3 µs) is at or above the
rig's default (22 + mHC ≈ 38 µs), so the table is the conservative case.

Kernel timeline, one layer at T = 5 (`rig_final2.log` dumps, µs from the router start):

| | router | topk | prep (ids→long, zeros, route_ids, rot_in / glue_prep) | grouped g/u starts at | shared gate/up | mhc_fused in the window |
|---|---|---|---|---|---|---|
| base | 13.4 | 6.6 | 6.4 + 3.3 + 3.9 + 14.0 | 47.9 | 38.8 | 10.2 |
| glue | 13.2 | 6.8 | 24.2 (glue_prep next to the shared GEMV) | 44.2 | 44.3 | 10.2 |
| glue + warm | 9.0-9.3 | 4.3-5.1 | 10.9-14.7 | 24.8-27.8 | 16.5-18.3 | 10.2-11.2 |

### 3.2 glue (GLM53_DEC_MOEGLUE)

`glue_prep | grouped g/u | gateup_epilogue | grouped d | glue_finish` replaces `ids.to(long) | zeros(out) | route_ids |
rot_in | grouped g/u | gateup_epilogue | grouped d | down_epilogue (fp32 atomics) | out.to(bf16)`. glue_prep = K2's
map_topk_to_local + a stable sorted position per pair + the segment table + rot_in (+ an optional
`prefetch.global.L2` of the epilogue vectors, `GLM53_DEC_MOEGLUE_PREFETCH`, default 1 — measured neutral:
glue+warm vs glue_nopf+warm 792.2 / 792.0 µs at T = 5); glue_finish = down_epilogue per pair + the per-token fp32 sum
in slot order + the cast. In the rig the `pre` span drops 51 → 42-47 µs and `post` 7.7-8.9 → 5.4-6.2 µs, but the
freed DRAM time goes to the shared gate/up GEMV that runs beside it, so the layer gains only 3-4 µs (−0.3 to −1.1 %).

### 3.3 warm (GLM53_DEC_MOEGLUE_WARM)

Mechanism: right after the o_proj GEMV of every decoder layer whose MLP is a MoE, a side stream forks from the main
stream and runs `l2_warm` (tf_exl3_moe_ext) over that layer's `hc_ffn_fn` (1.5 MiB, read by the mHC post/pre of the
window), router gate (2.25 MiB), shared gate_up (8 MiB FP8) and shared down (4 MiB): 15.75 MiB, 16 blocks × 256
threads, 8 × 16 B loads in flight per thread, regions in that order. The all-reduce and mHC run meanwhile on the main
stream; then the router, the shared GEMVs and glue_prep hit the L2 and the grouped GEMV starts ~20 µs earlier, with
the ~16 MiB read in the window instead of beside the layer's own GEMVs. The join (main waits for the side stream) is at the end of that layer's
apply_exl3_experts, when the warm has long finished.

Why these choices (all measured):
- **Real loads, not prefetch.** `tests/probe_l2_retain.py` (`probe_l2retain1.log`): after a 96 MiB flush + 40 µs
  idle, a 16 MiB read takes 58.3 µs cold, 14.3 µs if the bytes were read before (L2, 1170 GB/s), 51.2 µs after
  `prefetch.global.L2`, 58.4 µs after `prefetch.global.L2::evict_last`, 41.1 µs after `cp.async.bulk.prefetch.L2`.
  The earlier probe `probe_l2_nextpf.py` (prefetch.global.L2 of the next in_proj weight during a 40 µs idle window)
  gained nothing for the same reason (`probe_l2pf1.log`).
- **Plain `ld.global.nc`, not `.L1::no_allocate`.** `tests/probe_l2_warm_kernel.py` (`probe_l2_warm_kernel1.log`):
  after an `L1::no_allocate` warm, the full read still takes 72.7-74.8 µs (cold); after a plain-load warm 14.3 µs.
  The first production kernel used no_allocate and made the layer 3-10 % *slower* (`rig_final1.log`); fixed.
- **Few blocks, fn first.** With 96 blocks (test touch kernel, `rig_touch2/3.log`) the warm doubles mhc_fused
  (10 → 20-22 µs); with 12-24 blocks and 8 loads in flight it is 15-18 µs (`rig_touch4.log`); the production kernel
  (regions in order, hc_ffn_fn first, 16 blocks) leaves it at 10.2-11.2 µs (`rig_final2.log`) and reads at 218 GB/s
  (`probe_l2_warm_kernel1.log`). Sets / blocks (`rig_warmset.log`, vs frgd b16):
  frg ×1.0025-1.0051, rg (no mHC fn) ×1.012-1.021, frgd b24 ×0.9992-1.0007, frgd b8 ×1.009-1.034 → default frgd, 16.
- **Stream priorities do not help** (`rig_prio1.log`; CUDA graphs do honour them): shared expert at high priority
  ×1.0050 / ×1.0030 (T = 5 / 8), MoE at high priority ×1.0006 / ×0.9991; running the shared act+down after the gate/up
  GEMV (`sdlate`) ×0.9976-1.0030; forking the shared expert before the router (`base_pf`) ×0.991-1.018 (noise).

Wiring (glm53_moeglue.py): at load (vLLM's `base_loader.process_weights_after_loading`, wrapped), for each
`Glm5NextDecoderLayer` with `_mlp_is_moe`, mHC and not MTP whose routed experts use `Exl3MoEMethod` (so the join point
exists), the o_proj's quant method gets a subclass (as glm53_gemv_install does) whose `apply` calls production's apply
(fp8_gemv-hooked or not) and then forks; under torch.compile it emits the opaque custom op
`glm53_moeglue::linear_warm` instead (same output tensor, so it stays between the GEMV and the all-reduce that
consumes it). The fork runs only for ≤ `GLM53_DEC_MOEGLUE_WARM_MAX_M` tokens (64). Fork and join are in the same
CUDA-graph segment (no attention/eager-break op between an o_proj and its MoE). The wiring refuses (WARNING,
production unchanged) unless `RowParallelLinear.forward` (0d071001b54bbb4b), `Glm5NextDecoderLayer.forward`
(9c0fe21938cdc177) and `Glm5NextMoE.forward` (5492b934207c3148) are the verified versions (production's launcher
patches do not touch them). A second fork before a join (never in the model's order) joins first and is counted.
`additional_config["glm53_moeglue"] = "warm:v1:frgd:16:16:8:64"` changes the compile-cache hash (first start with
warm recompiles).

## 4. Numerics

- warm: reads only. `tests/test_moeglue_warm.py` W6-W8: o_proj output bitwise equal to production's apply at M = 1..200
  (FP8 Marlin via fp8_gemv and unquantized), 30 CUDA-graph replays of 3 × (o_proj → stand-in → hooked MoE on a real
  EXL3 layer) bitwise equal to eager, torch.compile(fullgraph) output bitwise equal.
- glue: `tests/test_moeglue.py` (`test_moeglue2.log`): (a) glue_prep == route_ids + rot_in 240/240 bitwise; (b) glue ==
  bf16 of the slot-ordered fp32 sum of K2's per-pair contributions 131/131 bitwise (one route per token → identical to
  production); (c) vs production's apply (K2, fp32 atomics in arrival order, itself not run-to-run deterministic):
  134/134, worst rel_l2 7.4e-5, worst |diff| 0.77 bf16 ulp; vs the fp64 sum of the per-pair values: glue 1.70e-3,
  production 1.70e-3 (the same error). Glue is deterministic. A per-layer self-test at load (real weights, 8 cases)
  keeps a layer on production's path if it fails. The long-context KL probe (noise floor KL 0.0051) is the
  orchestrator's check for glue; warm needs none.

## 5. Memory

glue: `inv` int32 [P_cap = 1024] in TF's shared scratch (4 KiB per rank); it allocates the bf16 output directly
(production allocates an fp32 buffer, then the bf16 copy), so peak memory per call is lower. warm: a 16-byte sink per
device, one extra CUDA stream, Python handles; no weight copies. Total < 64 KiB per rank.

## 6. Enable (both ranks) / revert

The AOT extension must be rebuilt from this branch (`python3 setup.py build_ext --inplace` / image `pip install .`):
glue needs `moe_forward_glue`, warm needs `l2_warm`; an older .so makes each switch refuse with a WARNING.

- Head: add to the `docker run` `-e` list in start.sh (next to `-e GLM53_BF16_GEMV_DEDUP_ROUTER=...`):
  `-e GLM53_DEC_MOEGLUE="${GLM53_DEC_MOEGLUE:-}" -e GLM53_DEC_MOEGLUE_WARM="${GLM53_DEC_MOEGLUE_WARM:-}"`
- Worker: add `GLM53_DEC_MOEGLUE GLM53_DEC_MOEGLUE_WARM` to the `serve_env_names` loop (`for v in ... GLM53_KPOOL_SEED_STRIDE`).
- Then `GLM53_DEC_MOEGLUE=1 GLM53_DEC_MOEGLUE_WARM=1` in the env file. For a trial without editing start.sh:
  `GLM53_EXTRA_ENV="GLM53_DEC_MOEGLUE=1 GLM53_DEC_MOEGLUE_WARM=1"` (forwarded to both ranks).
- Optional knobs (defaults are the measured best): `GLM53_DEC_MOEGLUE_PREFETCH` (1), `GLM53_DEC_MOEGLUE_WARM_SET`
  (frgd), `_WARM_MIB` (16), `_WARM_BLOCKS` (16), `_WARM_MAX_M` (64).
- Expected log lines: `glm53_moeglue installed: ...`, `glm53_moeglue: layer self-test passed (...)`,
  `glm53_moeglue warm armed: ...`, `glm53_moeglue warm wired 42 MoE sublayers of ... (15.75 MiB each ...)`,
  `compile-cache tag additional_config[glm53_moeglue]=warm:v1:...`.
- Revert: unset both on both ranks and restart.

## 7. Tests

- `tests/test_moeglue_warm.py` 97/97 (`test_moeglue_warm3.log`): W1 inert when unset; W2 env parsing; W3 l2_warm
  writes nothing, bad inputs rejected; W4 a warmed 12 MiB read runs at 1233 GB/s vs 216 GB/s cold; W5 wiring on the
  real vLLM classes (exactly the MoE decoder layers; dense-MLP, MTP and non-Exl3MoE layers not wired; region order and
  sizes; the MiB budget; an unaligned view is reduced to its 16 B-aligned interior; compile-cache tag changes the
  config hash); W6 bitwise o_proj, forks only at M ≤ 64; W7 CUDA graph capture + 30 replays bitwise, 3 warm kernels
  per replay, double fork joined first; W8 torch.compile fullgraph; W9 fingerprint mismatch wires nothing and
  disarms, glue + warm install together.
- `tests/r16/test_r16_breakable.py` (§11): the warm under vLLM's breakable PIECEWISE capture (vLLM's own Glm5Next
  forwards, the real eager-break decorator), before / after the segment guard.
- `tests/test_moeglue.py` 21/21 (`test_moeglue2.log`, see §4), `tests/test_integrate.py` 75/75
  (`test_integrate_moeglue.log`); unchanged production paths: `test_apply_fused.py` 20/20, `test_graph.py` 39/39,
  `test_fp8_integrate.py` 383/383, `test_gemv_install.py` 106/106 (`moeglue_reg_*.log`).
- Run: `GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" tests/gpu_run.sh python3 tests/test_moeglue_warm.py`
  (same for test_moeglue.py; bench: `tests/bench_moeglue_layer.py --modes base,base2,glue,base+W,glue+W --window 22 --mhc --kind corr40 --T 1,5,8`).

## 8. Risks and what production must measure

- **The all-reduce under the warm's DRAM traffic is not measured** (one GPU here; the rig's all-reduce is a spin).
  The warm reads ~15.75 MiB at ≈ 200 GB/s while NCCL's LL all-reduce polls flags / the NIC DMAs 40 KB. If the AR
  median (22.9 µs) grows by x µs, the gain per layer shrinks by x. Measure: decode tok/s and the AR / mhc_fused
  kernel durations in a torch.profiler trace with warm on vs off (same prompt set, both ranks).
- The window in production (median 44.9 µs) differs from the rig's (≈ 38 µs); the w10 / w50 runs bound the gain at
  −21 / −49 µs per layer.
- L2 pollution: 15.75 MiB per MoE sublayer displaces whatever was in the L2 (at decode: the previous layer's streamed
  weights; nothing is reused across sublayers except small tensors).
- glue is not bit-identical with 8 routes per token (summation order); KL probe by the orchestrator.
- First start with warm recompiles (compile-cache tag).

## 9. Next lever (not in this branch)

The same window exists before every attention sublayer (the MoE's all-reduce + the next mHC pre, ≈ 40 µs, 42 per
step; also after each attention for the dense layers). Warming the next layer's `hc_attn_fn` + the head of its first
projection (KDA in_proj 50 MiB / MLA qkv_a) there is the same mechanism: fork at the end of the MoE apply, join in the
next in_proj's apply (the attention core is an eager break / splitting op after it, so both ends stay in one graph
segment). Expected of the same order (≈ 20-30 µs per sublayer) if the KDA in_proj GEMV starts on L2 hits.

## 10. Adversarial review (2026-09-28, reviewer; logs `docs/logs/moeglue/review_*.log`)

Everything re-run on nodeC from this branch (AOT .so built after the last kernel change), under the shared GPU lock,
against the live overlay exl3.py.

- **Tests reproduced**: test_moeglue 21/21, test_moeglue_warm 97/97, test_integrate 75/75, test_apply_fused 20/20,
  test_graph 39/39. The patched production vLLM (all 21 launcher overlays applied in a throw-away container of the
  image) has exactly the fingerprints in `VERIFIED` / `WARM_VERIFIED`, so both switches arm in production.
- **Rig A/B reproduced** (151 paired rounds each, glue+W vs base, per layer call incl. the window):

  | window | T | glue | warm | glue + warm |
  |---|---|---|---|---|
  | 22 µs + mHC | 1 | ×0.9889 | ×0.9063 | ×0.9013 (−31.4 µs) |
  | 22 µs + mHC | 5 | ×0.9949 | ×0.9641 | ×0.9590 (−33.4 µs) |
  | 22 µs + mHC | 6 | ×0.9984 | ×0.9662 | ×0.9635 (−31.7 µs) |
  | 22 µs + mHC | 8 | ×0.9951 | ×0.9702 | ×0.9638 (−36.3 µs) |
  | 22 µs + mHC | 16 | ×0.9970 | ×0.9856 | ×0.9816 (−34.9 µs) |
  | 22 µs + mHC | 5 rand | ×0.9966 | ×0.9717 | ×0.9695 (−34.7 µs) |
  | 45 µs + mHC | 5 / 8 | ×0.9958 / ×0.9953 | ×0.9538 / ×0.9645 | ×0.9438 / ×0.9560 (−47 / −43 µs) |
  | mHC only (no spin) | 5 / 16 | ×0.9960 / ×0.9984 | ×1.0002 / ×0.9999 | ×0.9963 / ×0.9995 |

  Null repeat (base2) ×0.9989-1.0008. The warm gain is set by the DRAM-idle window: none without one, ≈ −33 µs at
  production's median all-reduce (22.9 µs), ≈ −45 µs at 45 µs. It never regressed in the rig. T > 16 cannot be run in
  the rig (fp8_gemv has no config for the shared expert at M > 16: production uses Marlin there).
- **Numerics, independent** (`tests/rv_moeglue_adversarial.py`, 83/83): n = 288, T 1..64, int32 ids, fp32 / bf16
  weights, sentinel routes, a token with no route, 3-D / row-strided / misaligned x, non-contiguous ids / weights
  (the last three are delegated to production, as they must be). Against a float64 reference of the EXL3 layer
  (dequantized trellises): glue 1.896e-3, production K2 1.896e-3, production loop 1.807e-3 (worst rel_l2). Glue vs
  production: rel_l2 ≤ 5.7e-5, max |diff| 7.8e-3 = 0.64 bf16 ulp at the row max. Production K2 differed run to run in
  45/71 cases, glue in 0/44 served cases. Glue is NOT bit-identical to production with 8 routes per token (order of
  the fp32 sum only); warm is bit-identical.
- **Fixed on this branch (f631a73)**: an eager o_proj fork whose MoE never ran, followed by a CUDA-graph capture,
  made the capture's first fork wait on an event recorded outside the capture ->
  `cudaErrorStreamCaptureInvalidated` (reproduced, `review_rv_adv.log` (4)). The pending flag now records the capture
  state of the fork; a fork from the other state is dropped without a join (counter `warm_stale_dropped`, one
  WARNING). Not expected in production's startup order (every eager forward runs o_proj -> MoE), but it would have
  been a startup crash.
- **Not measurable here, must be measured in production**: the all-reduce under the warm's DRAM traffic.
  `tests/probe_rv_loaded_latency.py`: a single-thread DRAM pointer chase takes 420 ns/hop alone, 759 ns/hop next to
  l2_warm (16 blocks), 1622 ns/hop next to a 96-block full-bandwidth read. NCCL's LL all-reduce over RoCE is made of
  a few serialized memory round trips (GPU flag writes / polls, NIC DMA, the CPU proxy's polling of the same
  LPDDR5X); at +0.34 µs per round trip a slowdown of a few µs per all-reduce is plausible, against ≈ 33 µs of gain
  per layer. Check the AR kernel median / p90 before the MoE layers with warm on vs off.
- Projected: 42 MoE calls per step × 32-36 µs ≈ 1.35-1.5 ms/step (69.1 -> ≈ 67.7 ms, ≈ 2 %), minus whatever the AR
  loses; glue alone ≈ 0.1-0.3 ms/step.

## 11. Breakable CUDA graphs (branch `sidestream`, 2026-09-29; logs `docs/logs/sidestream/`)

The first deploy-r16 boot failed its PIECEWISE CUDA-graph capture (`capturing stream has unjoined work`, details in
DEC_FP8ROOF.md section 12): vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for `Glm5Next*`, and a breakable
PIECEWISE capture is a chain of separate CUDA graphs cut at every `@eager_break_during_capture` op (KDA core, sparse-MLA
indexer, MLA attention); a side stream must rejoin inside the graph it was forked in. Both R16 flags that fork a side
stream were taken out of production's `.env`; the failing fork was fp8roof's t1, not the warm:
`tests/r16/test_r16_breakable.py` with the kit's 0501c3e modules (`breakable_before.log`) captures the warm alone
through vLLM's `BreakableCUDAGraphWrapper` (11 graphs, 10 eager breaks, 5 warms) and replays it bitwise, with quickwins
on and off (vLLM's stock forwards, `warm:noqw`). The window
o_proj → all-reduce → mHC → router → MoE apply has no eager break (the "same CUDA-graph segment" of section 3.3 holds
for breakable graphs too).

Changed anyway (`glm53_moeglue.py`), because the join was only correct by the model's structure:
`BreakableCUDAGraphCapture.add_eager` / `_end_segment` / `_begin_segment` are wrapped (idempotent, in the same chain as
fp8roof's; inert while no warm is pending): a warm still pending when a segment ends is joined into it on the segment's
capture stream. At a break (a structure change: a break between an o_proj and its MoE) WARNING once and the warm no
longer forks inside breakable segments (`warm_joined_at_break`, `warm_bk_skipped`); at the end of a capture, normal or
raising, silently (`warm_joined_at_capture_end`). Without it a Python error raised between a warm's fork and its join
during a capture surfaced as `capturing stream has unjoined work` and left the warm pending (`breakable_before.log`
inject-warm); with it the real error surfaces and nothing stays pending (`breakable_after.log` A.8). A guard that cannot
be installed disarms the warm (`but NOT armed: breakable CUDA graphs cannot be guarded`). The armed line ends with
`breakable CUDA graphs: segment guard hooked`. Eager and FULL-graph behaviour are unchanged (same forks, bitwise; the
warm forks in PIECEWISE graphs exactly as in FULL ones).

Measured (`breakable_timing.log`, DEC_FP8ROOF.md section 12's rig and method: 8 production-shaped layers with 5 MoE
sublayers, real EXL3 experts topk 8, spins for the latency-bound windows, 41 paired rounds; null repeat −10 … +53 µs):
the warm alone saves 132 / 94 / 68 µs per forward in FULL graphs and 62 / 65 / 46 µs in breakable PIECEWISE graphs at
M = 5 / 8 / 16 (≈ 26 / 19 / 14 µs per MoE sublayer in FULL; the PIECEWISE numbers are within about one noise width of
FULL); with fp8roof: 618 / 591 / 409 µs FULL, 469 / 390 / 316 µs PIECEWISE. Production decode steps replay FULL graphs
(DEC_FP8ROOF.md section 12), where the warm runs exactly as before this change. The warm forks in PIECEWISE graphs as in
FULL ones (A.3 / A.9: 5 per forward), nothing is joined at a break or at the capture end.
