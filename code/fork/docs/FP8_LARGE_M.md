# FP8_LARGE_M — exact fast path for large-M (prefill) FP8 linears (`GLM53_FP8_LARGE_M`)

Status 2026-09-28, branch `fp8largem`. Built and measured on nodeC (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, the launcher overlay `exl3.py` bound over the image's
with `GPU_RUN_BIND`). Production (nodeA/nodeB) was not accessed; nothing is installed there. Every number below is in a
committed log under `docs/logs/fp8_large_m/`.

## 1. Why

Production R12 profile of a 15.6k-token real-text prefill, rank 0 (two engine steps: 13824 + 1791 tokens; trace
`prof4/rankx-3139-20260928-035431.json.gz`, Marlin kernels grouped per `_C::marlin_gemm` call): FP8 Marlin = 2.98 s, of which

| layer (per rank, N × K) | Marlin launches per call | ms per call at M = 13824 / 1791 | calls | total |
|---|---|---|---|---|
| KDA `in_proj_qkvbfg_a` 12576 × 4096 (Marlin-padded to 12608) | 14 (1024-row slices, 64-wide n tile) | 50.7 / 6.6 | 34 + 34 | **1.95 s** |
| dense `gate_up` 12288 × 4096 | 14 / 3 | 46.1 / 6.0 | 3 + 3 | 0.16 s |
| drafter `fc` 4096 × 20480 (`GLM53_DRAFT_FP8=layers,fc`) | 2 | 52.6 / 4.5 | 1 + 1 | 0.06 s |
| everything else (N ≤ 4096 except fc, and MLA `q_b` 8192 × 1536) | | 70+ TFLOPS already | | 0.81 s |

For N > 4096, Marlin splits M into 1024-row launches (its `max_par` limit): 25–28 TFLOPS, against 85–90 TFLOPS for a
plain cuBLAS BF16 GEMM of the same shape (`probe1_marlin_vs_cublas.log`). nodeC reproduces production's launches
(`probe_marlin_launches.log` vs the trace): in_proj 1024-row launches 3.5–4.2 ms on nodeC / 3.46–4.20 ms in production,
the last 512 rows 1.79 / 1.87 ms, and at M = 1791 production's 1024 + 704 + 63-row launches (4.02 + 2.68 + 0.24 ms) match
nodeC's per-row rate; dense gate_up 3.5–4.2 ms on nodeC / 3.35–3.60 ms in production per 1024 rows (~5 % faster there).

## 2. What it does

`fp8_gemv.py` (the existing `Glm53DenseFp8Method` wrapper) gets a second, independent path. For a call with
M ≥ `LARGE_TABLE[(Npad, K)]`, outside CUDA-graph capture, on a layer that passed its large-M self-test:

1. for each chunk of output columns (`large_plan`, bounded by `GLM53_FP8_LARGE_M_TEMP_MIB`, default 128 MiB):
   `dequant` (`kernels/fp8_large_m.cu`) writes the e4m3 weight rows **exactly** as BF16 [nc, K], read in place from the
   Marlin repack (the layout derived in `kernels/fp8_gemv.cu`; the per-channel scale is **not** folded in);
2. the GEMM, with `out = bf16(fp32 sum * scale[n])` — one rounding, the value class of Marlin's `bf16(sum * s)` —
   written straight into the [M, N] output (no Marlin padding, no unpad copy). Two backends
   (`GLM53_FP8_LARGE_M_GEMM`), **bitwise identical in every test** (0 of 1.9·10⁹ compared elements differ):
   - `tilelang` (default, `auto`): one TileLang kernel per K (`fp8_large_m_tl.py`; symbolic M, N and output row
     stride; block 128 × 256 × 32, 4 stages, 256 threads, swizzle 10) with the scale multiply in its epilogue — no fp32
     temp. JIT-compiled at the first large-M self-test (~2 s per K, 2 kernels per rank). TileLang is in the image and
     production already runs TileLang kernels (the `mhc_*_tilelang` ones in the R12 trace);
   - `cublas` (fallback, or forced): per chunk of rows `y32 = torch.mm(x, w.t(), out_dtype=torch.float32)` (fp32
     accumulation, fp32 output, so every split-K partial is fp32 too), then `scale_cast`: `bf16(y32 * scale[n])`;
3. a bias (Marlin-permuted, un-permuted on the fly) is added afterwards and rounded again, exactly Marlin's epilogue order.

| (Npad, K) | layers | served from M | why that M (crossover, 2 runs per backend: tilelang / cublas) |
|---|---|---|---|
| (12608, 4096) | KDA in_proj (34) | 640 | 1.42×, 1.34× / 1.26×, 1.25× (512: 1.16×, 1.14× / 1.08×, 1.08×) |
| (12288, 4096) | dense gate_up (3), drafter gate_up (5, decode-sized in practice) | 640 | 1.38×, 1.33× / 1.18×, 1.20× (512: 1.11×, 1.08× / 1.05×, 1.12×) |
| (4096, 20480) | drafter fc (1) | 8192 | see below |

(`crossover_tilelang_run{1,2}.log`, `crossover_run{1,2}.log`; the TileLang path has a ~1.5 ms floor at small M:
dequant 0.8 ms + the kernel's per-call overhead.)

Everything else stays Marlin: other shapes, M below the threshold (all of decode: CUDA graphs are ≤ 64 tokens and the path
also declines under capture), a failed self-test, any non-CUDA error.

**fc threshold.** The fc's Marlin baseline (256-thread N ≤ 4096 kernel, one launch up to 8192 rows) is noisy on nodeC
(round-to-round spreads up to 22 %): between M = 3072 and 6144 the measured gain moved between 1.00× and 1.65× from run to
run (`crossover_run{1,2}.log`, `bench_fp8_large_m.run{1,2}.log`), and production's fc Marlin time at M = 1791 (4.5 ms)
is at the fast end of nodeC's (4.3–5.7 ms). From 8192 rows the gain is 1.57–1.72× in every run, and production's own
8192-row launch is slow (trace: 36.1 ms for the first 8192 rows at M = 13824, 38 TFLOPS; nodeC 25.7–31.5 ms). So the fc
is served from M = 8192 only; it runs once per step, on the step's tokens, so this costs little.
MLA `o_proj` 4096 × 8192 looked like a target on nodeC (Marlin 45–53 TFLOPS at M = 13824) but production's trace has it
at 72 TFLOPS (12.9 ms), a difference nodeC does not reproduce, so it is not in the table.

### Alternatives measured and rejected

- **Scale in the cuBLAS epilogue** (per-row alpha vector, `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO`): not
  offered on this GPU — all 20 bf16 algorithms of cuBLASLt 13.1 report pointer-mode mask 3 (host / device scalar) and the
  heuristic returns `CUBLAS_STATUS_NOT_SUPPORTED` for every vector mode (`probe_cublaslt_alpha_vector.log`).
- **BF16-output GEMM, then scale**: rounds twice (`bf16(bf16(sum) * s)`), not the Marlin value class.
- **Folding the scale into the BF16 weight** (what production's `GLM53_KDA_BF16_LARGE_M` does): `e4m3 × scale` needs up
  to 12 significant bits, so the BF16 weight is rounded — a weight-quantization difference, not a summation-order one —
  and production keeps it as a persistent copy (98 MiB per layer, 3.26 GiB per rank; ~3.3 GiB is free).
- **Column-splitting into ≤ 4096-wide Marlin calls**: `marlin_gemm` takes the B tiles as one contiguous
  [K/16, 4·Npad] array (row stride = 4·Npad int32); a column block is a strided view it cannot address. It needs either a
  re-tiled second copy (51.5 MB per in_proj layer, 1.7 GiB per rank) or re-tiling in place, which changes the layout the
  decode kernels (`fp8_gemv` and Marlin) read. Also 12608 = 64 × 197 cannot be covered by 256-wide tiles. Not done.
- **Overlapping `scale_cast` with the next row chunk's GEMM on a second stream**: no gain, the GEMM fills the GPU
  (`probe5_components_pipeline.log`, "pipe" vs "seq"). The fused TileLang epilogue replaced it.
- **TileLang tile configurations** (`probe_tilelang_gemm.log`: symbolic M, static N; `probe_tilelang_strided.log`:
  symbolic M and N, strided output as shipped): without the block swizzle 21–42 TFLOPS, with it 73–94 TFLOPS;
  128 × 256 × 32 / 4 stages / swizzle 10 reaches 91–93 TFLOPS at M = 13824 with a static N and 83–87 TFLOPS as shipped;
  5 stages of 256 × 128 × 32 exceed GB10's 99 KiB of shared memory. Every configuration was bitwise equal to
  `bf16(torch.mm(fp32) * scale)`. The kernel runs on torch's current stream (`probe_tilelang_stream.log`: input produced
  on a busy side stream, 5/5 correct).
- **Chunk plans**: speed is flat within ~5 % between 26 MiB and 300 MiB of transient (`probe7_chunk_plans.log`) as long
  as neither chunk dimension gets small; `large_plan` takes the fewest column chunks that leave ≥ 2048 rows per fp32 chunk
  (plans at 128 MiB: in_proj 2 × 6336 columns × 3200 rows, gate_up 2 × 6144 × 3392, fc 2 × 2048 × 6144).

## 3. Numerics

- `dequant` is bitwise `fp8.to(bfloat16)` for every production large-M shape, unaligned row ranges, and all 254 non-NaN
  e4m3 byte patterns (subnormals, ±0, ±448); the fp32 scale is bitwise the stored per-channel scale
  (`aot/test_fp8_large_m.log` D, A). `scale_cast` is bitwise torch's `(y32 * alpha).to(bf16)` (+ bias in Marlin's order) (S).
- The only freedom is the fp32 sum: both backends give `bf16(torch.mm(x, W_e4m3^T, fp32) * scale)` bitwise, and
  `|y32 − float64 dot| ≤ 0.014–0.038 × (K/16 + 2)·2⁻²³·Σ|x·w|` (F). The TileLang kernel's result for a row does not
  depend on M or on the column chunking (M = 1 … 777, partial tiles, column slices; T), and the two backends agreed on
  every one of the 1.9·10⁹ elements compared in N.
- Against float64 and production's Marlin, synthetic weights and **real GLM-5.3-Flash weights** (KDA in_proj rank-0
  shards of layers 1/20/42, dense gate_up of layer 1, the DFlash2 drafter fc), activations with ×30 outlier channels,
  M = 512 … 13824 (N section; every element within 0.5 ulp + the fp32 bound, for both):

| weights | correctly rounded: new / Marlin | bitwise == Marlin | \|new − Marlin\| ≤ 1 ulp / > 1 ulp | rel_l2 vs Marlin |
|---|---|---|---|---|
| KDA in_proj (real, 3 layers; synthetic) | 99.89–99.92 % / 99.96–99.98 % | 99.91–99.93 % | 0.063–0.088 % / 0.0034–0.0061 % | 0.8–1.1e-4 |
| dense gate_up (real; synthetic) | 99.89–99.92 % / 99.94–99.96 % | 99.94–99.95 % | 0.047–0.057 % / 0.0026–0.0035 % | 0.6–0.9e-4 |
| drafter fc, K = 20480 (real; synthetic) | 99.35–99.44 % / 99.66–99.70 % | 99.59–99.68 % | 0.30–0.38 % / 0.018–0.028 % | 1.9–2.5e-4 |

  cuBLAS accumulates each output over longer fp32 chains than Marlin (which splits K across warps and blocks), so it is
  correctly rounded slightly less often (−0.05 to −0.3 points); same class, both within the fp32 bound.
- Deterministic (repeated calls bitwise identical) and chunk-plan invariant: three other (nc, mc) plans gave 100.00 %
  bitwise the default's output for every shape at M = 1791 and 13824 (C) — the result of a row does not depend on how the
  prompt is chunked.
- Model-level quality (prefill top-20 KL) was not measured: it needs production. The existing quality probe's texts are
  short; prompts must have ≥ 640 tokens (≥ 8192 for the fc) to exercise this path.

## 4. Speed (nodeC, through production's `Glm53DenseFp8Method.apply`, interleaved rounds, median of 5)

`bench_fp8_large_m.run{1,2}.log` (Marlin / TileLang backend / cuBLAS backend in rotated order each round; spreads
≤ 9.6 % for in_proj and gate_up except one 19 %; the fc's up to 22 %):

| layer | M | Marlin ms (TFLOPS) | tilelang ms (TFLOPS) | speedup run1 / run2 | cublas backend speedup |
|---|---|---|---|---|---|
| KDA in_proj | 1791 | 7.47 (24.7) | 3.23 (57.1) | 2.31 / 2.25 | 1.95 / 1.92 |
| | 2202 | 9.30 (24.4) | 3.74 (60.7) | 2.49 / 2.40 | 2.10 / 2.10 |
| | 4608 | 19.20 (24.7) | 6.61 (71.8) | 2.90 / 2.81 | 2.12 / 2.09 |
| | 8192 | 34.02 (24.8) | 11.02 (76.6) | 3.09 / 3.12 | 2.27 / 2.25 |
| | 13824 | 57.49 (24.8) | 17.77 (80.2) | **3.24 / 3.24** | 2.30 / 2.29 |
| dense gate_up | 640 | 2.30 (28.1) | 1.87 (34.4) | 1.23 / 1.30 | 1.17 / 1.21 |
| | 1791 | 6.60 (27.3) | 3.23 (55.9) | 2.04 / 2.10 | 1.73 / 1.76 |
| | 13824 | 51.00 (27.3) | 17.41 (79.9) | **2.93 / 2.95** | 2.13 / 2.13 |
| drafter fc | 8192 | 33.15 (41.5) | 19.23 (71.5) | 1.72 / 1.67 | 1.73 / 1.68 |
| | 13824 | 50.27 (46.1) | 30.70 (75.5) | **1.64 / 1.56** | 1.60 / 1.75 |

(in_proj at its threshold M = 640: 1.42× / 1.34× in the crossover runs.) At M = 13824 the TileLang backend's in_proj
call is the kernel (16.9–17.0 ms, 83–84 TFLOPS: within ~5 % of cuBLAS's own BF16-output GEMM of the shape, 16.0–16.4 ms)
plus `dequant` (0.78–0.79 ms). The cuBLAS backend spends 18–20 ms in its fp32-output GEMMs plus 4.4 ms in `scale_cast`.

**Per prompt, per rank** (both TP ranks do the same work in parallel; 34 in_proj + 3 gate_up + 1 fc per chunk):

| | Marlin | new (tilelang) | saved | (cublas backend) |
|---|---|---|---|---|
| nodeC, 15,615 tokens (13824 + 1791) | 2436 / 2433 ms | 811 / 815 ms | **1.63 / 1.62 s** | 1100 / 1101 ms |
| nodeC, 16,026 tokens (13824 + 2202) | 2505 / 2498 ms | 831 / 835 ms | 1.67 / 1.66 s | 1124 / 1120 ms |
| nodeC, 35,840 tokens (2 × 13824 + 8192) | 5595 / 5588 ms | 1800 / 1797 ms | 3.80 / 3.79 s | 2480 / 2482 ms |
| the profiled production prefill, 15,615 tokens: Marlin kernel times from the R12 trace (in_proj 1723.6 + 223.7, gate_up 138.4 + 18.1, fc 52.6 ms; unpad copies excluded), new-path times from nodeC (in_proj 17.7 / 3.28, gate_up 17.4 / 3.2, fc 30.7–35.9 ms) | 2156 ms | ~805–810 ms | **~1.35 s** | ~1089 ms (−1.07 s) |

That prefill ran at R12's ~1214 tok/s (~12.9 s), so ~1.35 s is ≈ 10 % (→ ~1350 tok/s). It assumes the new path runs as
fast on nodeA/nodeB as on nodeC: same GPU model, and nodeC already reproduces production's Marlin launches (§1).

Measurement hygiene: nodeC's GPU is shared with other services; one bench run was contended (Marlin in_proj 122 ms,
spreads up to 119 %) and is kept as `bench_fp8_large_m.contended.log` only as an example of a run to discard.

## 5. Memory

- Persistent: the un-permuted fp32 scale, 4·N bytes per served layer (`aot/test_fp8_large_m_integrate.log` L.4: 115,840 B for one layer of each shape).
  In production per rank: 34 in_proj + 3 dense gate_up + 5 drafter gate_up + 1 fc ≈ 2.0 MB, plus one int64 permutation
  index per Npad (≈ 0.2 MB).
- Transient, one call (`aot/test_fp8_large_m_integrate.log` L.9, peak allocated above the inputs, incl. the output):

| layer, M | Marlin | tilelang | cublas |
|---|---|---|---|
| in_proj 2202 / 13824 | 106 / 664 MiB (padded output + unpad copy) | 151 / 430 MiB | 156 / 459 MiB |
| gate_up 2202 / 13824 | 55 / 327 MiB | 148 / 420 MiB | 151 / 452 MiB |
| fc 8192 / 13824 | 67 / 111 MiB | 144 / 188 MiB | 192 / 236 MiB |

  Bounded by output + `GLM53_FP8_LARGE_M_TEMP_MIB` (128; the TileLang backend needs only the BF16 weight chunk:
  98 MiB for in_proj, 96 MiB gate_up, 80 MiB per fc chunk). The in_proj layers — 34 of the 38 served calls per step —
  peak ~234 MiB lower than Marlin at 13824; gate_up and fc ~77–93 MiB higher. These are caching-allocator blocks that
  other prefill temporaries reuse.
- Load: one self-test per layer (M = 320, forced multi-chunk plan), milliseconds each, plus the TileLang JIT of 2
  kernels (~2 s each) at the first self-test of each K.

## 6. How it is enabled, fallbacks, evidence

- `GLM53_FP8_LARGE_M` ∈ {1, on, true, yes}: read once when vLLM loads general plugins (`integrate.plugin_register` →
  `fp8_gemv.plugin_register`). Unset / empty / 0 / off → nothing is patched by it, its extension is not imported (test L.1;
  with only `GLM53_FP8_GEMV=1`, every large-M call is bitwise production's apply). Independent of `GLM53_FP8_GEMV`:
  either variable installs the same two wrappers, each path runs only under its own variable (L.3, L.7).
- `GLM53_FP8_LARGE_M_TEMP_MIB` (16..4096, default 128): transient budget; an invalid value refuses the path with a WARNING.
- `GLM53_FP8_LARGE_M_GEMM` (auto = default | tilelang | cublas): the GEMM backend. With auto/tilelang, a TileLang import,
  compile or launch failure at the first self-test switches the whole path to cublas with a WARNING (test L.12); an
  invalid value refuses the path.
- AOT extension `tf_fp8_large_m_ext` (`kernels/fp8_large_m.cu`, in `setup.py`, so the wheel/bundle carries it); needs
  `torch.mm(out_dtype=float32)` (torch ≥ 2.9; the image has 2.13), checked at install.
- Fallbacks, each bitwise production's apply (L.6): failed self-test (WARNING at load), pre-launch error (WARNING once,
  counted), CUDA-graph capture (Marlin is captured), production's KDA BF16 copy present, method not ready, fp16.
  A strided or misaligned x is copied to a dense one for the TileLang kernel (test T). CUDA errors are re-raised,
  never masked.
- Engagement evidence in the server log, per rank: `tf_fp8_gemv large-M path installed (GLM53_FP8_LARGE_M) ...
  TileLang (fused scale) GEMM`, one `large-M self-test passed ... (tilelang) ... bitwise == cuBLAS backend ...` INFO
  line per shape at load (3 with `GLM53_DRAFT_FP8=layers,fc`), and one `first large-M call served for <N>x<K>
  (M=…, tilelang GEMM ...)` INFO line per shape at the first long prefill. A `TileLang GEMM of the large-M path
  failed` WARNING means the cuBLAS backend is serving (≈ 1.07 s instead of ≈ 1.35 s saved per 15.6k prompt).

## 7. Rollout notes (not done: production was not touched)

1. Rebuild the bundle wheel (it now contains `tf_fp8_large_m_ext`); `.env`: `GLM53_FP8_LARGE_M=1`, **and** add it to the
   head's explicit `docker run -e` list (the R1 incident: the head passes listed variables only).
2. Per rank: the INFO lines above (and no TileLang WARNING); `MemAvailable` after start and during a ≥ 24k-token
   prefill (the transient is ≤ 93 MiB above Marlin for gate_up/fc and ~234 MiB below it for in_proj). The TileLang
   JIT writes its cache under the container's `$HOME/.tilelang` (as production's own TileLang kernels do).
3. Quality: prefill KL with prompts ≥ 640 tokens (the short probe texts would not exercise the path); expected at the
   noise level (only fp32 summation order changes).
4. Speed: `tools/prodcheck/prefill_probe.py` real-text 8.5k / 15.3k and random 24k, before/after; decode bench unchanged
   by construction (M ≤ 64 never reaches the path).
5. Rollback: empty `GLM53_FP8_LARGE_M` and restart.

## 8. Next

- Fold the dequant into the TileLang kernel (load the Marlin fp8 tiles, convert in shared memory): removes the 0.8 ms
  per in_proj call (~5 %) and the BF16 weight temp.
- A static-N variant for the two full-weight shapes (91–93 TFLOPS static vs 83–87 symbolic at M = 13824, ~1 ms per
  in_proj call).

## Files

- `kernels/fp8_large_m.cu` — `dequant`, `scale_cast` (extension `tf_fp8_large_m_ext`, VERSION 1).
- `fp8_large_m_tl.py` — the TileLang kernel (JIT).
- `fp8_gemv.py` — `LARGE_TABLE`, `large_plan`, `large_forward`, `selftest_large`, `try_large`, install/uninstall.
- Tests: `tests/test_fp8_large_m.py` (kernels, numerics, real weights; `GPU_RUN_RO` mounts
  `${HOME}/models/GLM-5.3-Flash-EXL3-TR3-4bpw-partial` and `${HOME}/models/GLM-5.3-Flash-DFlash2-dc77ff1c`),
  `tests/test_fp8_large_m_integrate.py` (wrapper against the real production module), `tests/bench_fp8_large_m.py`,
  `tests/bench_fp8_large_m_crossover.py`; probes `tests/probe_large_m{,5,6,7}.py`, `tests/probe_cublaslt_alpha.{py,cu}`,
  `tests/probe_marlin_launches.py`, `tests/probe_tilelang_{gemm,strided,stream}.py`.
  `tests/run_all.sh` builds and runs the two tests and the bench.
- Logs: `docs/logs/fp8_large_m/` (`aot/` = the AOT extension against the overlay module, `image/` = against the image's
  `exl3.py`; the others ran with the overlay module).
