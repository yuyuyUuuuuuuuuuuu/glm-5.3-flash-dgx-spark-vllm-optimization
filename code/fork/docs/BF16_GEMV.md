# BF16_GEMV — the small dense GEMMs of decode (`GLM53_BF16_GEMV`)

Status 2026-09-28, branch `bf16gemv`. Built and measured on nodeC (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`). Production (nodeA/nodeB) was not accessed; nothing
is installed there. Every number below is in a committed log under `docs/logs/bf16_gemv/`.

## 1. What the profile's small GEMMs are

R7 profile, rank 0, prose, batch 1, 40 steps (`docs/logs/prof-R7/rank0-kernel-table.txt`, trace segmented with
`tools/prodcheck/trace_seg.py`); the M of that run is 5 (K = 4). cuBLAS picks its kernel per shape, and the grid of a
`cutlass_80_wmma … 16x16` kernel with GemmIdentityThreadblockSwizzle<8> is (8·m-tiles, ceil(n-tiles / 8), split-K),
which gives N. Module code: `vllm/models/glm5next/nvidia/{model,attention}.py`, `vllm/model_executor/models/qwen3_dflash{,2}.py`
in the image (the live launcher overlay chain does not change any function used here: `tests/gemv_chain_check.sh`).

| kernel (grid) | module | weight, per rank (N × K) | calls/step | in-situ ms/step | served here |
|---|---|---|---|---|---|
| wmma 16x16_128x2 [8,3,8] + splitKreduce [9] + cast [3] | MoE router `mlp.gate` (GateLinear) — **computed twice per MoE layer**: `Glm5NextMoE.forward` and again by `MoERunner._forward_impl`, which holds the same gate | bf16 288 × 4096, replicated, fp32 logits | 84 (+84 +65) | 0.99 + 1.22 + 0.21 | yes + dedup |
| gemmSN_NN<float> grid [1,1,1] / [1,2,1] + copy [40] | MLA indexer head gate `torch.mm(x.float(), _wp_fp32)` (fp32 on purpose) | fp32 32 × 4096 (bf16-valued), replicated | 11 | 0.86 + 0.03 | yes |
| wmma 16x16_128x2 [8,2,8] + splitKreduce [5] | indexer `wk_weights_proj` | bf16 160 × 4096, replicated | 11 | 0.15 | yes |
| wmma 16x16_128x2 [8,1,8] + splitKreduce [4] | indexer kpool gate `F.linear(x, index_kpool_compress_gate)` | bf16 128 × 4096, replicated | 11 | 0.10 | yes |
| wmma 16x16_128x2 [8,8,1] | DFlash2 `attention_conv` / `mlp_conv` `kernel_projection` (5 layers × 2) | bf16 1024 × 4096, replicated | 10 | 0.49 | yes |
| wmma 16x16_128x1 [8,32,1] | indexer `wq_b` (11) and DFlash `fc` 20480→4096 (1, eager, 0.8 ms) | bf16 4096 × 1536 / 4096 × 20480 | 12 | 1.48 | no: cuBLAS already 206–228 GB/s |
| wmma 16x16_128x1 [8,40,1] | DFlash fused context K/V (5 layers) | bf16 5120 × 4096 | 1 | 0.20 | no: 223 GB/s |
| wmma 32x32_64x2_nn [8,2,32], 32x32_128x2_tn [8,1,32] | MLA absorbed W_UK / W_UV bmm (32 heads per rank) | bf16 32 × 256 × 512 each | 11 + 11 | 0.54 + 0.44 | no (next candidate, ≤ ~0.2 ms) |
| wmma 16x16_128x2 [8,605,1] | target lm_head (vocab half) | bf16 77440 × 4096 | 1 | 2.63 | no: 241 GB/s |

The router is `GateLinear` tier 6 on SM 12.x (none of its specialized tiers applies): `F.linear` in bf16, then
`.to(float32)`. All of these run inside the decode CUDA graphs except the DFlash fc.

## 2. Kernels (`kernels/gemv_bf16.cu`, extension `glm53_gemv_ext`)

- `gemm_bf16`: y = x·Wᵀ, M = 1..64, `mma.sync m16n8k16` bf16 → fp32. Tokens are the MMA's 16-row side (zero rows past
  M), 8 weight rows its n side; the k order inside a 32-wide chunk is permuted so every lane does 16-byte loads of
  its weight row and activation rows (A and B use the same map: same products, another summation order). Weights go
  global → registers one chunk ahead (L2 evict_first); activations are staged in shared memory with cp.async. Split-K
  over CTAs writes fp32 partials; the last CTA of a column group (atomic ticket) sums them in split order and resets
  the ticket → deterministic, graph-replay safe, no second kernel. Output bf16, fp32, or fp32 holding the
  bf16-rounded value (what the router's `F.linear(...).to(float32)` returns).
- `gemm_f32`: the head gate in IEEE fp32 FMA on CUDA cores from the bf16 weight rows (production's fp32 copy holds
  exactly those values), same ticket scheme.
- Launch plans (`glm53_bf16_gemv.PLANS`, `F32_MAX_M`): per shape and M bucket, the best of a sweep
  (`docs/logs/bf16_gemv/sweep.log`), used only where it beat production's op; everywhere else production's exact op
  runs.

## 3. Speed (cold weights, CUDA-graph replay, µs per call; `docs/logs/bf16_gemv/bench_served.log`)

"new" = exactly what `GLM53_BF16_GEMV=1` runs (plan or production op). `router_pair` = one MoE layer's two gate calls
(the second on warm L2) against one call (dedup).

| M | router old → new µs (GB/s) | router pair old → new | head gate old → new | wk_weights_proj | kpool gate | DFlash2 conv proj |
|---|---|---|---|---|---|---|
| 5 | 17.5 → 13.7 (135 → 173) | 27.4 → 13.6 | 75.4 → 6.6 | 10.8 → 8.9 | 9.9 → 7.8 | 40.6 → 38.9 |
| 8 | 17.5 → 13.7 | 27.6 → 13.7 | 73.6 → 6.8 | 10.8 → 9.0 | 9.9 → 7.8 | 45.1 → 39.1 (186 → 215) |
| 16 | 17.5 → 14.5 | 28.3 → 14.4 | 72.2 → 8.4 | 10.8 → 9.3 | 9.9 → 8.1 | 45.2 → 39.6 |
| 24 | 18.1 → 14.6 | 32.7 → 14.6 | 13.4 → 10.0 | = | = | = |
| 32 | 18.2 → 14.9 | 32.6 → 15.0 | 13.6 → 11.6 | = | = | = |
| 48 | 18.0 → 15.4 | 31.2 → 15.3 | = | = | = | = |
| 64 | = | 31.2 → 18.0 (dedup only) | = | = | = | = |

"=" : production's op (the kernel was not faster there). Weights: router 2.25 MiB, head gate 0.25, wk 1.25, kpool 1.0,
conv 8.0 MiB; the kernel reaches 172 GB/s on 2.25 MiB (floor ≈ 13.5 µs: launch + DRAM latency + one fixup).

Predicted saving per decode step = Σ calls/step × (old − new), isolated timings:

| configuration | router + dedup | head gate | wk | kpool | conv | total | of the replaced calls |
|---|---|---|---|---|---|---|---|
| profiled: B = 1, K = 4 (target M = 5, drafter M = 8) | 0.58 ms | 0.76 | 0.02 | 0.02 | 0.06 | **1.44 ms / 78 ms (1.8 %)** | 2.66 ms → 1.22 ms (−54 %) |
| B = 2, K = 7 (M = 16, drafter 16) | 0.58 | 0.70 | 0.02 | 0.02 | 0.06 | 1.38 ms | |
| B = 4, K = 7 (M = 32, drafter 32) | 0.74 | 0.02 | 0 | 0 | 0 | 0.76 ms | |
| B = 8, K = 7 (M = 64) | 0.55 (dedup) | 0 | 0 | 0 | 0 | 0.55 ms | |

In the trace the same calls cost 4.0 ms/step of kernel time (the second router GEMM's splitKreduce waits 27 µs behind
the shared expert's Marlin on the aux stream: Marlin's 99 KiB of shared memory + 1 KiB reserved leaves no room on any
of the 48 SMs). How much of that is on the critical path is not measurable on one GB10; the isolated estimate above
is the prediction, a production profile A/B is the confirmation.

## 4. Numerics

- `tests/test_bf16_gemv.py` (`docs/logs/bf16_gemv/test_bf16_gemv.log`): every M 1..64 on 7 shapes, every plan knob,
  strided x, 3 output modes, determinism, graph replay: max |y − y₆₄| / Σ|x·w| ≤ 1.4e-6 (fp32 out), bf16 outputs within
  1 ulp of the exactly rounded value; head gate 2.8e-8 (cuBLAS fp32: 3.0e-8).
- **Production's small-M cuBLAS path is not fp32-accurate**: at M ≤ 16 the split-K GEMMs (router, wk, kpool; wq_b at
  M = 32) differ from round_bf16(exact) in 40–52 % of outputs, up to hundreds of ulps on small outputs (consistent
  with bf16 split-K partials: `splitKreduce_kernel<…, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16>`). The
  kernel: ≤ 0.1 % of outputs, ≤ 1 ulp. For router / wk / kpool at M ≥ 32 cuBLAS (another kernel) is exact too.
- Router top-8 on the real router weights and e_score_correction_bias of layers 10–13 (local GLM-5.3-Flash checkpoint),
  x = rmsnorm(randn) × post_attention_layernorm.weight, production's `grouped_topk` kernel, 1.31 M tokens
  (`docs/logs/bf16_gemv/router_topk_agreement.log`):

  | | M = 5, 8, 16 | M = 32, 48 |
  |---|---|---|
  | top-8 id set: new vs production cuBLAS | 2.33–2.45 % of tokens differ | 0–0.005 % |
  | production cuBLAS vs exact (f64 → bf16) | 2.33–2.45 % | 0–0.005 % |
  | new vs exact | 0–0.005 % | 0–0.005 % |
  | production at M = 5 vs production at M = 32/48 (same tokens) | 2.33–2.45 % | |

  So the routing differences the kernel introduces are production's own split-K error being removed: with the kernel
  the routing no longer depends on the batch size beyond ≤ 0.005 % of tokens (today: 2.3–2.5 % between M ≤ 16 and
  M ≥ 32). Max |Δ weight| where the sets agree: 8.5e-4 (weights sum to 2.5).

## 5. Integration (`glm53_gemv_install.py`, loaded by `integrate.plugin_register`)

- Enable `GLM53_BF16_GEMV=1` (read at plugin load). Unset: no op registered, nothing patched (tested). Optional:
  `GLM53_BF16_GEMV_KINDS=router,idx_wk,idx_head,idx_kpool,draft_conv` (subset), `GLM53_BF16_GEMV_DEDUP_ROUTER=0`.
- Wiring after the weights are loaded (vLLM `base_loader.process_weights_after_loading`, wrapped): every served module
  is self-tested on its real weight at every M bucket it will serve (exact-value bound, determinism, tickets reset);
  a module that fails keeps production's path (tested with a mutated kernel). Router: `GateLinear.forward`
  (per-instance opt-in). Indexer: `Indexer.forward` replaced by a copy that differs only in the two GEMM statements,
  installed only if the production function has the fingerprint the copy was made from (the AST difference is tested).
  wk / conv: their `UnquantizedLinearMethod` instance becomes a subclass whose `apply` calls the op. Dedup:
  `runner.gate = None` for each `Glm5NextMoE` whose runner holds its gate (same input, same deterministic kernel →
  the runner uses the logits the model already computed), only with the fingerprinted `MoERunner._forward_impl`.
- Custom ops `glm53_gemv::linear` / `::head_gate` with fake impls: opaque to dynamo, so the kernel-or-production choice
  depends on the real M at run / capture time (one compiled graph serves every M: tested at M = 5, 8, 40, 64, 100,
  and under CUDA-graph capture/replay).
- Compile cache: `additional_config["glm53_bf16_gemv"] = "v1:<kinds>:dedup=<0|1>:plans=<digest>"` changes
  `VllmConfig.compute_hash()`, so vLLM's AOT / inductor caches never reuse a graph traced in the other mode. Any stale
  graph that still calls the op for an unregistered module gets production's op (tested: unknown handle → bitwise
  production).
- Logs: `glm53_bf16_gemv plugin loaded … -> installing|off`, `installed on <Model>: {kinds}`, per module `not served:
  <reason>`, and once per (kind, M) the path the captured graphs contain (`router M=5 -> gemv plan (4, 2, 256)` /
  `-> production op`).
- Tests: `tests/test_gemv_install.py` (real classes; real router / indexer / DFlash2 weights when mounted; 106 checks),
  `tests/test_gemv_drafter_load.py` (real plugin entry + real `DefaultModelLoader` building the DFlash2 drafter: the
  hook wires all 10 conv projections, 94 checks), both in `docs/logs/bf16_gemv/`. `tests/run_all.sh` builds the
  extension and runs `test_bf16_gemv` / `test_gemv_install`.

## 6. Risks and what was not verified

- Never run in production: no TP=2 run, no real target model (no weights on nodeC), no end-to-end step time. The
  saving is a sum of isolated call timings; overlap with the shared-expert stream changes the real number.
- The dedup changes the MoE layer's stream schedule (no router GEMM left beside the shared expert). Routing is
  bitwise the same as without the dedup.
- Routing differs from today's production on ~2.4 % of tokens at M ≤ 16 (production's error removed, see §4);
  acceptance rate / quality were not measured.
- `torch.library` custom-op dispatch costs ~10–30 µs of CPU per call in eager mode; production decode is graph-captured
  (ENFORCE_EAGER=0); with ENFORCE_EAGER=1 leave the switch off.
- Memory: split-K scratch S·64·N fp32 per module (router 0.28 MiB × 42, wk 0.31 × 11, kpool 0.25 × 11, head gate
  0.25 × 11; conv none): ≈ 21 MiB per rank, allocated at load. `_wp_fp32` is production's own (built at load instead
  of at the first forward).
- A production file other than the fingerprinted versions disables the affected part with a WARNING (router /
  indexer / dedup independently); the plan table was measured on nodeC's GB10.
