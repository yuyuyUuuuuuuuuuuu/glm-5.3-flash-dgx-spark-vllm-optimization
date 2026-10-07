# PREFILL_QUICKWINS — exact prefill quick wins (`GLM53_PREFILL_QUICKWINS`)

Status 2026-09-28, branch `quickwins` (= `deploy-r13` + this change). Built and measured on nodeC (one GB10, production
image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`) with production's module set: the launcher
overlay `exl3.py` and the SM90_KV=1 mounts (flashinfer 0.6.18, patched `vllm/platforms/cuda.py` and
`flashinfer_mla_sparse_sm90.py`, `FLASHINFER_DISABLE_VERSION_CHECK=1`; `tests/qw/prod_env.sh`). Production (nodeA/nodeB)
was not accessed; nothing is installed there. Every number below is in a committed log under
`docs/logs/prefill_quickwins/`.

**Result: −263 ms of rank-0 kernel time per 13,824-token chunk and −24 ms per 1,791-token chunk, every item
bit-identical to production (torch.equal on production shapes).** For the R13 15.3k real-text prefill (1324 tok/s,
~11.6 s) that projects to ~−0.28 s, **~+2.5 % prefill throughput** (projection: the saving is GPU kernel time and step 0
runs the GPU ~99 % busy, so it should turn into wall time about 1:1; not measured in production). This is what the
"quick wins" lever is worth; it is not a path to 2x on its own.

## 1. Items and measured savings

`tests/qw/bench_quickwins.py` (`bench_quickwins.log`): median of 45 CUDA-event timings per variant (3 rounds x 15,
interleaved), production's statement vs the fast path on the same inputs, production shapes (TP=2 rank: 32 MLA heads,
KDA 32 x 128, hidden 4096, mHC 4 streams). Calls per step from the R13 prefill trace (prof5): 11 MLA layers, 34 KDA
layers, 5 aux layers + the last layer, 11 indexer head gates.

| item | what changes | 13824 tok: prod → fast per call | per 13.8k chunk | 1791 tok: per call | per 1.8k chunk | design estimate |
|---|---|---|---|---|---|---|
| `mla_bmm` (a) | W_UK / W_UV absorption bmm: cuBLAS `cutlass_80_wmma 32x32` → Triton strided batched GEMM | 10.95 → 6.71 ms (both GEMMs) | **−46.6 ms** | 1.32 → 0.94 | −4.2 ms | −90 ms |
| `mla_index` (b) | SM90 index fill + convert + clamp + copy (4 kernels, 2 temporaries) → 1 kernel writing `kv_indices` | 3.69 → 1.02 ms | **−29.4 ms** | 0.35 → 0.12 | −2.4 ms | −26 ms |
| `kda_conv` (c) | merged q\|k\|v conv (channel-first output) + 3 `.contiguous()` copies → 3 conv launches writing q, k, v contiguous | 6.83 → 3.48 ms | **−113.8 ms** | 0.85 → 0.46 | −13.3 ms | −190 ms (incl. l2norm fusion, §5) |
| `mhc_aux` (d) | aux layers: the next layer no longer recomputes the aux layer's hc_post | 17.17 → 12.46 ms | **−23.5 ms** | 2.09 → 1.57 | −2.6 ms | −23 ms |
| `mhc_mean` (d) | 4-stream mean computed inside the post kernel (5 aux + last layer) | 43.8 → 29.1 ms (6 calls) | **−14.7 ms** | 5.64 → 3.80 | −1.8 ms | −11 ms |
| `idx_gate` (extra) | indexer fp32 head gate: `x.float()` + cuBLAS simt sgemm → Triton IEEE dot on bf16 x (M ≥ 10240 only) | 3.79 → 0.65 ms | **−34.6 ms** | not applied (M < 10240) | 0 | not identified in the design (§3) |
| **total** | | | **−262.8 ms** | | **−24.4 ms** | |
| (e) MoE teardown | not done (§5) | | | | | −75…−120 ms |

The per-item numbers reproduce the trace's kernel times for production's path (prod column vs prof5: MLA bmm 11 x
10.95 = 120 ms vs 122.5 ms in the trace; index 40.6 vs 39.5; KDA conv + copies 232 vs 222; head gate 41.7 vs 40.7).
An earlier run (`bench_quickwins_run0.log`, before `idx_gate` existed, also T = 4608) agrees within 3 %.

Why (a) saves half of what the design estimated: the two absorption GEMMs are memory-bound, not compute-bound. Each moves
~680 MB at T = 13824 (W_UK reads q 226 MB and writes 453 MB; W_UV the reverse), so ~3.1 ms each is the floor at ~220
GB/s; the Triton kernel runs at 3.07 / 3.23 ms (`bench_mla_triton_sweep.log`), i.e. at that floor. The design's
"~70 TFLOPS → 36 ms per chunk" did not count the bytes.

cuBLAS alternatives were measured first (`bench_mla_cublas_variants.log`): contiguous weights keep the wmma kernel
(no gain, or slower for W_UV), per-head `torch.mm` loops (cuBLAS then picks `cutlass_80_tensorop` / `nvjet` kernels)
reach 3.8–4.1 ms per GEMM at T = 13824 (28–31 TFLOPS) but not at 1791, and a contiguous copy of q costs more than it
saves. None reaches the Triton kernel's 3.1–3.2 ms, which is also a single launch per GEMM.

## 2. How each item works (and why it is bit-identical)

Every fast path runs only for a step with ≥ `GLM53_PREFILL_QUICKWINS_MIN_T` tokens (default 256, above every CUDA-graph
capture size, which are ≤ 64 in production) and never while a CUDA graph is being captured. Anything else, or an
unexpected dtype / layout, runs production's own statement unchanged (the helpers carry production's statement verbatim
as their fallback).

- **mla_bmm** — `torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)` (`MLAAttention.forward_impl`) and
  `torch.bmm(x, self.W_UV, out=out.transpose(0, 1))` (`_v_up_proj`) become one Triton kernel on the same strided tensors
  (production's `W_UK_T` / `W_UV` are strided views of `kv_b_proj.weight`; nothing is copied or re-laid-out). bf16 x bf16
  → fp32 `tl.dot` with K in 64-wide steps (= the k16 mma sequence the wmma kernel uses), bf16 store. Measured
  bit-identical to production's cuBLAS result for T in {256, 1791, 4608, 13824}, 3 seeds, including a q that is a
  padded row view (W.2). The config (tile / order) does not change the bits (BK is 64 in all configs).
- **mla_index** — `FlashInferMLASparseSM90Impl.forward_mqa` (the mounted, read-only patched file) ran
  `full_like(-1)` → convert/compact (`SINGLE_TILE`, `COMPACT_TO_FRONT`) → `clamp_(min=0)` → int32 `copy_` into the
  wrapper's `kv_indices`. One kernel does the same per row (same block-table arithmetic, same in-row cumsum compaction,
  valid prefix then a 0 tail) and writes `kv_indices` directly. Used only when the top-k width is a power of two (2048
  in production: production's convert is then single-tile and deterministic). W.3 runs the patched and the original
  `forward_mqa` (stub wrapper) on 13824 / 1791-token batches with 1–3 requests, prior context, out-of-range block ids
  and interior -1 gaps: `kv_indices` identical.
- **kda_conv** — in `Glm5NextLinearAttention._forward` (prefill branch) the merged conv's output is allocated by
  `empty_like` of a non-dense transposed slice, i.e. channel-first, so q, k, v reach FLA as column slices and
  `l2norm_fwd(q.contiguous())`, `l2norm_fwd(k.contiguous())`, `v.contiguous()` copy them (3 x 1.05 ms per layer). The
  conv is per channel, so it is run three times (q, k, v channel ranges of the same merged weight and conv state, same
  metadata) with an output tensor laid out so that q, k, v come out contiguous; the three `.contiguous()` become no-ops.
  The conv function used is production's `causal_conv1d_fn` recompiled with one extra `out=None` parameter (the default
  keeps `empty_like`). W.4: q, k, v and the updated conv state identical for 13824 / 1791 / multi-request batches, with and
  without initial state, conv state in both the DS and the transposed SD layout.
- **mhc_aux** — at the 5 EAGLE3/DFlash2 aux layers, `Glm5NextModel.forward` materializes
  `hc_post(hidden, residual, post, comb)` for the aux value; the next layer's fused post→pre recomputed the same post.
  The materialized streams are passed to the next layer as its input with `post=None`, which takes the standalone pre
  branch. For T > 16 the fused op is post + `tf32_hc_prenorm_gemm` + `pre_big_fuse`, and the standalone pre is the same
  GEMM + `pre_big_fuse` with the same `n_splits` formula (`compute_num_split`), so the next layer computes the same bits
  (W.5: residual, post_mix, comb_mix, layer_input identical at T = 17, 64, 257, 1791, 13824). Not applied to the layer 0
  path, sequence-parallel runs, MTP layers or T ≤ 16.
- **mhc_mean** — `hc_contract` (aten `mean(dim=1)`) re-read the 453 MB post output. A Triton post kernel reproduces the
  tilelang `mhc_post_tilelang_kernel` bit for bit: its generated CUDA is `x = c*d; x = x + a_i*b_i` (i = 0..3), which
  nvcc contracts to `fma(c, d, a0*b0)` then `fma(a_i, b_i, x)` (found by comparing variants: mul+add and
  `fma(a0, b0, c*d)` differ in 734 / 1455 of 226M elements, this one in 0; `probe_post_mean.log`). The mean is aten's
  sequential fp32 sum `((o0+o1)+o2)+o3` of the bf16-rounded streams times 0.25 (the pairwise orders differ in 1 element).
  Aux layers: post + mean in one pass; last layer: mean only (the 453 MB post output is never written). W.5 checks
  post output, mean, and the last-layer helper at all five T with streams scaled 2^-12…2^11.
- **idx_gate** (not in the design's lever-5 list; see §3) — the indexer head gate at prefill runs through this fork's
  `glm53_gemv::head_gate` op, whose large-M branch is production's `torch.mm(x.float(), w32)`: a 226 MB bf16→fp32 copy
  (1.4 ms) plus cuBLAS `cutlass_80_simt_sgemm_64x64_8x5` (2.3 ms, fp32 on CUDA cores, N = 32). A Triton kernel reads x
  as bf16 (exact conversion) and does an IEEE fp32 `tl.dot` (k sequential), 0.65 ms. cuBLAS only runs that sgemm
  unsplit from M ≈ 10k on: the sweep (`probe_head_gate_sweep.log`) shows split-K (memset + atomics, or splitKreduce) and
  different bits for M ≤ 9216, one sgemm and identical bits for M ≥ 10240. So the fast path is limited to
  M ≥ `GATE_MIN_M` = 10240 (the 13824-token chunks), and the first call at each M also runs production's op and
  compares: a mismatch (e.g. another cuBLAS) turns the item off with a WARNING and returns production's result (W.6
  forces one).

## 3. What the trace attribution found beyond the design

- The "fp32 GEMM, source not identified" (`cutlass_80_simt_sgemm_64x64`, 25 ms per step) in the design's last section is
  the indexer head gate inside this fork's own `glm53_gemv::head_gate` (prefill falls back to production's
  `torch.mm(x.float(), w32)`); with its `.float()` copy it is 40.7 ms per step. Now `idx_gate`.
- The "Marlin unpad" copy (34 x 3.06 ms = 104 ms per step, `aten::contiguous` on a [13824, 12576] slice) is still in the
  R13 trace. It belongs to the Marlin workstream (`fp8largem`), not done here.
- 38 x `Memcpy DtoD` at the top level (36 ms per step) remain unattributed (no enclosing op in the trace).

## 4. Enable, wiring, memory

- Enable: `GLM53_PREFILL_QUICKWINS=all` or a comma list of `mla_bmm, mla_index, kda_conv, mhc_aux, mhc_mean, idx_gate`
  (read once when the vLLM plugin loads, `integrate.plugin_register`, independent of `TF_EXL3_MOE`). Unset / empty /
  `0` / `off`: nothing is installed. `GLM53_PREFILL_QUICKWINS_MIN_T` (default 256, must be ≥ 65).
- Mechanism (plugin hook, no file on disk is changed; the SM90 file is a read-only mount in production anyway): each
  item's production function is recompiled from its own source with the listed statements replaced by a helper call
  (`transplant`), in its module's globals, and set on its class — only if the function's AST fingerprint is one verified
  here (sha256 of `ast.dump`, first 16 hex): `MLAAttention.forward_impl` 7b7cbfcfa94c8b92, `._v_up_proj`
  558a22f54b818511, `FlashInferMLASparseSM90Impl.forward_mqa` e43adcd5a6e2acde (same in the image file and the mounted
  `.patched` file), `Glm5NextLinearAttention._forward` 1e4f45149fceddf5 (keeps its `@eager_break_during_capture`),
  `causal_conv1d_fn` 718ef047cdc3f7ac, `Glm5NextModel.forward` 224750fda049837c, `Glm5NextDecoderLayer.forward`
  9c0fe21938cdc177, this fork's `glm53_gemv_install._head_gate_impl` f2406f3c25bb4297 (wrapped, not recompiled). The
  launcher's runtime edits (`patch_dense_fp8.py` on kda.py / model.py constructors, ABLIT hook in `load_weights`) do not
  touch these functions (fingerprints computed before / after applying them: identical). An unverified function or a
  missing edit anchor → that item logs a WARNING and stays off (all-or-nothing per item). Modules not yet imported
  when the plugin loads are patched right after their first import (a `sys.meta_path` hook); W.0 checks this in a fresh
  process through `integrate.plugin_register()` (and that nothing is patched when the variable is unset).
- Production's model forward runs eagerly (no `support_torch_compile` on GLM5Next; ENFORCE_EAGER=0 only enables CUDA
  graphs, captured at decode sizes ≤ 64), so a recompiled Python function is what runs; no Dynamo / Inductor cache holds
  the old code.
- Logs: `glm53 prefill quickwins: <item> installed in <module> ...` (INFO) per item and process; `... active: <item>
  (first call ...)` at the first prefill that uses it; WARNING with the reason for a refused item. `STATS` counts fast
  and production calls per item.
- Memory: no persistent allocation; the Triton kernels compile on first use (a few seconds once; Triton's on-disk cache
  afterwards). Transient memory goes down: at 13824 tokens `mla_index` −113 MB (no -1-filled temporary), `kda_conv`
  −340 MB (one [3, T, 4096] buffer instead of the conv output plus three contiguous copies), `mhc_mean` −453 MB at the
  last layer (post output not materialized), `idx_gate` −226 MB (no fp32 copy of x); `mla_bmm` and `mhc_aux` 0.
- TP=2: set it on both ranks (and add it to the head `docker run -e` list: the R1 incident). A rank without it only runs
  slower; the results are the same bits either way.

## 5. Not done, and why

- **(e) MoE teardown** (design: −75…−120 ms). `out.to(bf16)` (61.9 ms per step) happens inside the
  `vllm::moe_forward_shared` custom op (overlay `apply_exl3_experts`), the shared + routed add (62.6 ms) outside it, after
  the op returns; fusing them exactly means the op returning the fp32 routed output and every transform between the op
  and the add seeing fp32 instead of bf16 — a change to the op's contract across the MoE runner that the E3 / E4 /
  prefill-cap workstreams also sit on. The fp16 cast of x (41.8 ms) is read by both E3's `fm_gather` and the thin
  kernel; folding it into `fm_gather` needs a rebuild of the image's `exl3_fat_moe` extension and a thin-kernel change.
  The fp32 zero fill (48.6 ms) is required by both kernels' atomic accumulation. All of these belong to the MoE
  workstream.
- **l2norm fused into the KDA conv** (the other part of the design's −0.19 s for (c), ~2 ms per layer = −68 ms per
  step). It needs a modified conv kernel, and the fused sum of squares would not reproduce `l2norm_fwd`'s reduction order
  (fp32-order class, "≤ 1 bf16 ulp in rare elements" per the design), so it is not exact. Left for a separate change
  with its own KL check.

## 6. Tests and files

- `tests/qw/test_quickwins.py` (223 checks, `test_quickwins.log`): W.0 plugin path in a fresh process, W.1 wiring
  (fingerprints, anchors, recompiled AST = production's with exactly the edits, every name an edit uses resolves,
  globals, decorator kept, uninstall restores the original objects, refusal is all-or-nothing), W.2–W.6 bitwise parity per item as described in §2, plus
  pass-through below MIN_T and during CUDA-graph capture.
- `tests/qw/bench_quickwins.py`, `tests/qw/bench_mla_gemm.py` (cuBLAS variants), `tests/qw/bench_mla_triton.py` (config
  sweep), `tests/qw/probe_post_mean.py`, `tests/qw/probe_head_gate.py`, `tests/qw/probe_fp.py`, `tests/qw/prod_env.sh`
  (production mounts for `tests/gpu_run.sh`, which gained `GPU_RUN_BIND_DIR` and `GPU_RUN_ENV`).
- `tests/test_integrate.py` still passes with the new `plugin_register` step (75/75, `test_integrate.log`).
- Files: `glm53_prefill_quickwins.py`, `integrate.py` (plugin_register), `setup.py` (py_modules), `tests/run_all.sh`
  (new step; run it with `tests/qw/prod_env.sh` sourced so the SM90 file is the production one), logs
  `docs/logs/prefill_quickwins/`.

## 7. Rollout (not done here)

1. Build the bundle from this branch (the wheel's `py_modules` now include `glm53_prefill_quickwins`).
2. `.env`: `GLM53_PREFILL_QUICKWINS=all`; add it to the head `docker run` env passthrough in start.sh.
3. Check on BOTH ranks: six `glm53 prefill quickwins: ... installed` lines (idx_gate only where GLM53_BF16_GEMV is on) and,
   after the first long prompt, the `... active:` lines; no `NOT installed` / `turned off` WARNING.
4. Measure long-prompt prefill tok/s (8.5k / 15.3k real text) and decode (unchanged by construction). The expected
   quality result is "no change" (bit-identical); a long-prompt KL check against the previous stage should read at the
   noise floor.
5. Rollback: empty the variable and restart.
