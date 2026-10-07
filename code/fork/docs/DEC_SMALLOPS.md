# DEC_SMALLOPS — small decode kernels outside the MoE (`GLM53_DEC_SMALLOPS`)

Status 2026-09-28, branch `dec-smallops` (from `deploy-r15` 45047ef). Built and measured on nodeC (one GB10, production
image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, other jobs sharing the GPU: every timing is a
paired, interleaved A/B inside CUDA graphs under `/tmp/tf-gpu-bench.lock`). Production (nodeA/nodeB) was not accessed.
Every number below is in a committed log under `docs/logs/smallops/`.

## 1. What ships (both bit-identical to production)

| kind | production op (per step) | replacement | nodeC A/B, µs per call (median of 15 interleaved rounds × 20 graph replays) | per step |
|---|---|---|---|---|
| `dconv` | DFlash2 `_grouped_conv`: 10 eager bf16 elementwise kernels per call, **20 calls** (5 drafter layers × attention/mlp conv × prepare/finish) | 1 kernel (`glm53_smallops.dconv`) through custom op `glm53_so::dconv` | T=8: 14.93 → **1.90**; T=16: 15.26 → 2.15; T=32: 16.91 → 2.52 | **−0.26 ms** (T = 8 per request) |
| `mhc` | `vllm::mhc_fused_post_pre_tilelang` small-M branch = `mhc_fused_tilelang` + `mhc_pre_big_fuse_with_norm_tilelang`, **89 calls** (45 layers × 2 − layer 0's standalone pre) | smallops `mhc_fused` kernel on an exact bf16 copy of `hc_{attn,ffn}_fn` + production's own pre kernel, **served for M ≤ 8 only** (review: M 9..14 were slower, §9) | M=5: 15.87 → **13.66**; M=6: 16.20 → **14.04**; M=8: 17.29 → 16.73 (whole op, 89 distinct cold weights); M > 8: production | **−0.20 ms (M=5) / −0.19 ms (M=6)**, −0.05 ms (M=8) |

Projected saving (rank 0 = rank 1, both on the critical path): prose (M = 5/6) **≈ 0.45 ms/step** of the 69.1 ms
profiled step (≈ 0.65 %, ≈ +0.26 tok/s at 40.6 tok/s), structured/coding (K = 7, M = 8) ≈ 0.31 ms/step. Spreads are
2–8 % per variant on the shared GPU (the dconv T=16/32 B rows 15–25 %: one contended round), the deltas are far above
them. The production trace shows the mHC kernels ~20–30 % slower than nodeC (13.4 vs 11.0 µs for mhc_fused_tilelang at
M = 5), so the in-production saving may differ; the orchestrator's A/B decides. Logs: `bench_ab_v2.log` (as wired;
v1 = same with 11 rounds), `bench_v1.log` / `bench_mhc_v2.log` / `mb_mhc_nb.log` / `mb_mhc_direct.log` /
`mb_mhc_m8.log` (kernel variants).

Why these are faster:
- `dconv`: production's `_grouped_conv` runs eagerly in the drafter (the trace shows 10 `at::native` elementwise
  kernels per call, ~15 µs of launches for 64 KiB of data). The kernel does the same bf16 op chain per element:
  `c_k = bf16(base + delta)`, `o = bf16(c_0·x)`, `o = bf16(o + bf16(bf16(c_k·shift_k(x))·mask))` — each step rounded to
  bf16 exactly where PyTorch rounds, so the result is bitwise production's (256 cases incl. real DFlash2 base kernels
  and kernel projections, T 1..64, block sizes 5..8, taps 1..3, ±0 / huge / subnormal inputs).
- `mhc`: `mhc_fused_tilelang` (decode M ≤ 16) launches one CTA per (token, 2–3 outputs, K split): every fp32 weight
  element is fetched by M CTAs and 1.5 MiB of fp32 weights stream per call. The smallops kernel (grid 6 output groups ×
  8 splits for M < 8; 8 × 4 with all 8 tokens in one stage for M = 8; 12 × 4 with double-buffered 4-token stages
  for M = 9..16) keeps each CTA's weights in registers and serves all M tokens with them, with
  the same thread ↔ h mapping, FMA order, xor-butterfly and 8-warp summation as the TileLang kernel (checked against its
  PTX, `docs/logs/smallops/mhc_src/`), so the split partials, sum of squares and post-mapped residual are bitwise
  production's. It reads a **bf16 copy** of the weight: GLM's `hc_*_fn` are bf16 values stored as fp32 (verified per
  weight at load; a weight that is not exactly representable is not served), so the copy is exact and halves the
  bytes. With fp32 weights the kernel is not faster cold (12.1 vs 11.0 µs, `mb_mhc_nb.log`: DRAM-bound on 1.5 MiB),
  with L2-warm weights it is 5.45 vs 8.58 µs (`bench_mhc_v2.log`).

## 2. Wiring (`glm53_smallops_install.py`, called from `integrate.plugin_register`)

- Env read once at plugin load; unset → nothing registered, nothing touched (logged "off").
- `dconv`: after the drafter's weights are loaded (`base_loader.process_weights_after_loading`, wrapped like
  `glm53_gemv_install`), for a `DFlashGroupedConv` module: the module global `_grouped_conv` of its module
  (`vllm.model_executor.models.qwen3_dflash2`) must have the verified source fingerprint (`238266efc04dc1cd`, image =
  launcher overlay, md5 196c5504…), then a self-test on that module's real base kernels (T = 8..32, both sides) must be
  bitwise; only then `_grouped_conv` becomes a wrapper calling custom op `glm53_so::dconv` (opaque to torch.compile,
  fake impl). The op itself falls back to the original function for anything the kernel does not take (non-bf16,
  layout, alignment, `T < taps` where production's `F.pad` path is kept verbatim).
- `mhc`: after the target's weights are loaded: every `hc_{attn,ffn}_fn` (fp32 [24, 16384], contiguous, exactly
  bf16-representable; layer 0's `hc_attn_fn` is skipped — it only feeds the standalone pre) gets a bf16 copy; a
  self-test runs the override path against production's function on up to 3 real weights, M = 1..8, all four outputs
  bitwise; then the CUDA kernel of `vllm::mhc_fused_post_pre_tilelang` is overridden with `torch.library`
  (`Library("vllm", "IMPL")`, kept alive; graphs traced by torch.compile are unchanged — they call the same op). The
  override serves a call only if `fn` is a registered weight (keyed by `data_ptr`, same shape/stride/dtype, `_version`
  unchanged since the copy — an in-place write sends the weight back to production, logged once), `norm_weight` is
  given, 1 ≤ M ≤ `MHC_SERVE_M_MAX` = 8 and shapes/dtypes/contiguity/alignment are the expected ones; everything else (M > 8,
  unknown weights) runs production's Python function with exactly the arguments received. The small-M branch is
  production's code (same allocations, `n_splits` = 8 for M < 8 else 4, same pre kernel call) with the kernel swapped;
  installed only when production's function has the verified fingerprint (`4924eeb1bfbe2394`).
- Compile cache: `additional_config["glm53_dec_smallops"] = "v1:<kinds>"` (same mechanism as `glm53_bf16_gemv`), so no
  artifact traced in one mode is reused in the other (the dconv wrapper is traced if the drafter is compiled).
- TP ranks: every decision depends only on shapes, dtypes, the (replicated) weights and deterministic self-tests, and
  the result is bitwise production's either way; nothing touches a collective.
- CUDA graphs: no host sync, no host-dependent branch inside captured code; allocations through the caching
  allocator; `cudaFuncSetAttribute` happens once per kernel at first use (legal during capture).
- AOT: `setup.py` builds `glm53_smallops_ext` (`kernels/smallops.cu`); `tests/_aot_smallops.py` builds it with the
  same flags and imports it (OK). JIT only with `GLM53_SMALLOPS_JIT` / `TF_EXL3_JIT` (nodeC tests).

## 3. Numerics

Bit-identical to the current production path for every served call, by construction and by test:
- `tests/test_smallops_kernels.py` → `docs/logs/smallops/test_kernels_v1.log` (re-run after the M = 8 launch change:
  `test_v2.log`, both test files ALL PASSED): dconv 256 cases; mhc_fused 624 calls
  (M 1..16 × 3 trials × real hc_attn/ffn_fn of layers 0/1/10 fp32 and bf16 copy + random fp32 weights): yp, rp and
  residual_cur bitwise.
- `tests/test_smallops_install.py` → `test_install_v1.log`: the installed path end to end — `DFlashGroupedConv._convolve`
  and `torch.ops.vllm.mhc_fused_post_pre_tilelang` bitwise equal to production eager, inside a CUDA graph and through
  `torch.compile(fullgraph=True)`; M = 1..8 served (1..16 before the review), 9..16/20/24/32 production; fallbacks (T < taps, fp32 input, unregistered
  fn, norm_weight None, in-place-modified weight); uninstall restores production.
No KL probe needed (no value changes); the orchestrator's probe should show the noise floor.

## 4. Memory (per rank)

- `mhc`: 89 bf16 copies × 0.75 MiB = **66.75 MiB** (the fp32 parameters stay: prefill M > 16 and the deep_gemm path use
  them). This is 2.75 MiB over the 64 MiB guideline; justification: −0.17…−0.24 ms/step at M = 5/6. To stay under,
  run `GLM53_DEC_SMALLOPS_KINDS=dconv`.
- `dconv`: 0 (outputs come from the graph pool like production's temporaries, fewer of them).

## 5. Enable / revert (both ranks)

```
.env:  GLM53_DEC_SMALLOPS=1              # optional: GLM53_DEC_SMALLOPS_KINDS=dconv,mhc (default = both)
start.sh head docker run (after the GLM53_BF16_GEMV_DEDUP_ROUTER line):
        -e GLM53_DEC_SMALLOPS="${GLM53_DEC_SMALLOPS:-}" \
        -e GLM53_DEC_SMALLOPS_KINDS="${GLM53_DEC_SMALLOPS_KINDS:-}" \
start.sh worker serve_env_names list (the `for v in ...` loop): add GLM53_DEC_SMALLOPS GLM53_DEC_SMALLOPS_KINDS
```
The fork bundle (`/opt/glm53/tf/site`, copied into site-packages by `patch_tf_bundle.py`) must contain the updated
`integrate.py` plus `glm53_smallops.py`, `glm53_smallops_install.py` and `glm53_smallops_ext.cpython-312-aarch64-linux-gnu.so`
(all produced by `pip install --no-build-isolation .` of this branch: `setup.py` lists them). Check in both ranks' logs: `glm53_dec_smallops plugin loaded ... installing`,
`glm53_dec_smallops on Glm5NextForConditionalGeneration: mhc (89 weights, bf16 copies 66.75 MiB)` (or similar model
class) and `... on DFlash2Qwen3ForCausalLM: dconv (10 conv modules, [...qwen3_dflash2])`, no `NOT serving` warning.
First start after enabling recompiles (new compile-cache tag). Revert: unset `GLM53_DEC_SMALLOPS`, restart.

## 6. Levers investigated and not shipped (with the measurement that decided)

- **MLA decode BF16 GEMMs** (indexer `wq_b` [4096, 1536] = cutlass 16x16_128x1 [8,32,1]; absorbed `W_UK_T` bmm
  (32, 256 → 512) = 32x32_64x2_nn [8,2,32]; `W_UV` bmm (32, 512 → 256) = 32x32_128x2_tn [8,1,32]; ~1.8 ms/step in the
  profile). Built `tc_gemm` (`kernels/smallops_tc.cuh`): tensor-core small-M GEMM that issues the same HMMA.16816
  chain per 16×16 tile in the same k order as cuBLAS's wmma kernels — **bitwise equal to cuBLAS** for wq_b (M 2..16),
  W_UK and W_UV (M 1..16), incl. the kernel switches cuBLAS makes at M = 1 and M ≥ 12 (`tests/test_smallops_tc.py`).
  But on nodeC cuBLAS already runs at the **pure-read floor** (`probe_read_floor.py`: 12.6 MB 59 µs, 8.4 MB 36.5 µs,
  42 MB 179 µs): cuBLAS wq_b 59.2 µs, W_UK 40.8, W_UV 41.9, drafter ctx-K/V [5120,4096] 192; tc_gemm's best configs
  72.5 / 41.1 / 45.7 / 243 µs (`bench_tc_v1.log`). Not wired. The production trace's larger times (wq_b 77 µs, W_UK
  58 µs) are not reproduced on nodeC with fresh allocations; a kernel cannot be shown to win there from nodeC data.
- **KDA decode chain**: `fused_recurrent_gated_delta_rule` (grid [1,16,32], 1 warp, 29–31.5 µs) reads the fp32 state
  (32 heads × 128 × 128 × 4 B = 2 MiB) and, for spec decode, writes the state after **every** verified token (5 × 2 MiB
  at K = 4) so any accepted prefix can be resumed. With the following o_proj (FP8 16.8 MB) the pair moves ~29 MB in
  114 µs ≈ 254 GB/s: already at the DRAM floor (the state writes drain from L2 during o_proj, which is why o_proj shows
  only ~155 GB/s). A faster recurrent kernel only moves the write-back. The structural lever is the state traffic:
  the kernel's update is S_t = S_{t−1}·diag(e^{g_t}) + u_t k_tᵀ with u_t = β_t (v_t − S_{t−1} diag(e^{g_t}) k_t), so
  a step can store one base state plus, per token, the fp32 vectors it already computes (e^{g_t}, normalized k_t, u_t:
  1.5 KiB per head instead of a 64 KiB state) and the next step's kernel replays the accepted records onto the base
  with the same fp32 operations in the same order (bitwise the same state) → per layer per step read 2 + write 2 MiB
  instead of read 2 + write T × 2 MiB: ≈ 8 MiB less at K = 4 (14 MiB at K = 7) ≈ 272–480 MB ≈ **1.1–1.9 ms/step**. It
  changes what vLLM's spec-decode state slots hold (every reader of a slot — prefill hand-over, prefix-cache state
  copies, the mamba-align patches — would have to understand base + records), so it is a larger project than this
  stream; not attempted.
  `causal_conv1d_update`: its per-channel arithmetic does not depend on the launch's BLOCK_N (256 in production, 48
  CTAs), so a smaller block is bitwise the same (36/36 variants checked, `probe_conv_blockn.py`), but in a 34-layer
  graph it costs only 2.5–4.2 µs per call on nodeC and BLOCK_N 64 saves ≤ 0.4 µs (`probe_conv_blockn.log`): not
  wired. Fusing it with the recurrent kernel / `layer_norm_gated` (2.9 µs) would save launches only: ≤ ~0.15 ms/step,
  not done.
- **L2 prefetch of the next mHC weight during the all-reduce** (DRAM idle): the fork/join onto a side stream costs more
  than it saves in a graph (decode-like sequence GEMV → AR stand-in → mHC: +8 µs per layer, `bench_mhc_v2.log`).
- **mHC with fp32 weights, other splits / outputs per CTA** (NB 1..6, S 8/16): never beats production cold
  (`mb_mhc_nb.log`); the bf16 copy is what makes it faster. Reading the tokens straight from L2 instead of staging
  them in shared memory (`DIRECT`) is slower for every M (`mb_mhc_direct.log`).
- **mhc_pre_big_fuse_with_norm** (4.1 µs, 90/step; in-graph kernel floor on this GPU is 0.53 µs,
  `probe_graph_floor.py`): one CTA per token, warp 0 runs the 20-iteration Sinkhorn (2 named-barrier all-reduces per
  iteration) while warps 1–2 stream the residual for the weighted sum and the 4096-wide RMSNorm; a faster bitwise
  version must keep those reduction trees; ≤ ~1.5 µs per call (≤ 0.14 ms/step), not done. PDL between the two mHC
  kernels is not available: vLLM disables it on SM 12.x (`is_arch_support_pdl`: "PDL lowering races KDA state
  kernels on GB10").
- **Drafter FP8 candidates** (change drafter numerics only → acceptance, target distribution stays exact; would be a
  separate sub-flag): the fused context-K/V weight (bf16 5120 × 4096, 42 MB, 192 µs/step; `patch_drafter_fp8` keeps it
  BF16 on purpose) → FP8 ≈ −95 µs/step and −21 MB; `kernel_projection` ×10 (bf16 1024 × 4096) → FP8 ≈ −0.2 ms/step.
  Not implemented (needs an acceptance-rate A/B in production).

## 7. FR-Spec (frequency-pruned drafter lm_head) break-even — analysis only

Drafter lm_head = target's FP8 lm_head, 77440 × 4096 per rank: 1356 µs/step + ~80 µs logits all-gather. Pruning to the
V′ most frequent tokens (fraction f = V′/154880) saves ≈ 1.436 ms × (1 − f) per step and needs an extra FP8 copy of
f × 317 MB per rank. Its cost: a draft position whose target token lies outside the subset is rejected, so the
per-position acceptance a becomes ≈ a·c (c = coverage: the share of target tokens inside the subset).
- Prose: ~3 tokens/step at K = 4 ⇒ a ≈ 0.745; tokens/step τ(a) = (1 − a⁵)/(1 − a), elasticity (a/τ)·dτ/da = 1.44.
  Break-even: 1.44 (1 − c) = (1.436/69.1)(1 − f) ⇒ **1 − c ≤ 0.0144 (1 − f)**: V′ = 32k (f = 0.21) needs c ≥ 98.9 %,
  V′ = 16k needs ≥ 98.7 %, V′ = 64k ≥ 99.2 %. Upper bound of the gain (c = 1): +2.1 % × (1 − f), i.e. ≤ +1.7 % at 32k.
- Structured (K = 7, ~7.9 tokens/step at ~85 ms/step ⇒ a ≈ 0.997, elasticity ≈ 3.5): needs c ≥ 1 − 0.0048 (1 − f),
  i.e. ≥ 99.6 % coverage at V′ = 32k.
Verdict: at most ~1.7 % for prose and a loss as soon as more than ~1 % of generated tokens (identifiers, numbers,
Japanese/Chinese text in a 155k vocabulary) fall outside the subset; not worth an extra 60+ MB per rank.

## 8. Files

`kernels/smallops.cu` (bindings), `kernels/smallops_kernels.cuh` (dconv, mhc_fused, l2_prefetch),
`kernels/smallops_tc.cuh` (tc_gemm, not wired), `glm53_smallops.py` (wrappers), `glm53_smallops_install.py` (wiring),
tests: `test_smallops_kernels.py`, `test_smallops_install.py`, `test_smallops_tc.py`, benches: `bench_smallops.py`
(`dconv`, `mhc`, `mhcsitu`, `tc`), `bench_smallops_ab.py` (as wired), `mb_smallops.py` (mHC variants),
`probe_read_floor.py`, `probe_l2_prefetch.py`, `probe_graph_floor.py`, `probe_conv_blockn.py`,
`smallops_dump_mhc_src.py` (TileLang sources), `_aot_smallops.py` (AOT build check).

## 9. Adversarial review (2026-09-28, separate reviewer)

Reproduced on nodeC under `/tmp/tf-gpu-bench.lock`. Logs: `review_tests.log`, `review_mhc_m_sweep.log`, `review_bench.log`.

- **Correctness reproduced and extended, all bitwise** (`tests/review_smallops_adversarial.py` + the implementer's two
  test files, JIT and the AOT `setup.py` build): mhc chained through 13 real `hc_*_fn` of checkpoint layers 0/1/10..14
  like the model (outputs of one op feed the next), M 1..16, residual scales 1×, 300×, 1e-3×: 624 calls; a CUDA graph
  captured once and replayed on 6 fresh data sets per M; outer shape (2, 3), 2-D `post`, fp32 `norm_weight` (served,
  bitwise); non-contiguous x, misaligned residual, M = 0, other-stride fn (production path, same result/exception).
  fp64 reference at M = 5: split-summed partials rel_l2 1.36e-7 (max-abs 5.1e-6), sum of squares 2.4e-8, residual_cur
  (bf16 rounding) 1.66e-3 — identical for production and override. dconv: all 10 real DFlash2 base kernels with the
  real `kernel_projection` coefficient layout, T 8..40 incl. T not a multiple of the block, strided x rows: 420 calls;
  graph replay on fresh data; inf/NaN inputs give the same NaN positions (NaN payload bits can differ: PyTorch writes
  0x7FC0, the kernel 0x7FFF; irrelevant). Source fingerprints re-derived from the image sources / overlay (match).
- **Fix: the mHC override lost for M = 9..14** (`review_mhc_m_sweep.log`, as-wired whole op, us/call production −
  override): M 9..14 −1.47 −0.70 −0.33 −0.06 −0.84 −0.24 (slower), M 15/16 +0.52/+0.80, M 1..8 +1.8..+3.3 (M 8 +0.9).
  Production decodes M = 10 / 12 with two requests at K = 4 / 5 (capture sizes include 10, 12, 15, 16), so the served
  range is now M ≤ 8 (`MHC_SERVE_M_MAX`); after the change M 9..16 measure +0.00..+0.03 (production path). M 15/16
  could be added back for +0.5..0.8 us/call; not done (inside the noise of the decode-context bench).
- As-wired A/B after the fix (`review_bench.log`): mHC M 5 / 6 / 8 +2.05 / +2.73 / +0.91 us/call, dconv T 8 / 16 / 32
  +13.2 / +13.2 / +14.4 us/call — the implementer's numbers reproduce (earlier runs: M 5 +2.33, +2.53, +2.32).
- Decode-like context (`tests/review_bench_mhc_ctx.py`: a 32 MiB default-policy read before every mHC op, inputs
  chained): the per-call gain is noisy on the shared GPU (single runs from −6 to +8 us) but its median over all runs is
  larger than back-to-back (M = 5 ≈ +3.9, M = 8 ≈ +3.8 us/call); the production trace's 13.4 us mhc_fused (vs 11.0 on
  nodeC back-to-back) is consistent with the op being slower when its inputs are cold. The 0.45 ms/step projection
  (89 × ~2.2 + 20 × ~13) is therefore not optimistic for prose at one request.
- Production per-step counts re-derived from `p6h.json.gz`: 89 `mhc_fused_tilelang` [5, 12, 8] per step; 20 grouped
  conv chains per step (10 kernels each, 16.9 us wall median per chain, T = 8, taps = 2, H = 4096).
- Memory measured under `expandable_segments:True` (production's allocator config): exactly 0.75 MiB per registered
  weight (66.75 MiB for 89), over the 64 MiB guideline; `KINDS=dconv` = 0 MiB.
- Residual risk (not a bug today): the stale-weight guard (`_version`) runs when a graph is captured, not on replay, and
  `param.data.copy_()` does not bump `_version`; any future code that rewrites `hc_*_fn` after load through `.data`
  would be served from the stale bf16 copy. Nothing in production (image, overlays, ABLIT = o_proj only) does.
