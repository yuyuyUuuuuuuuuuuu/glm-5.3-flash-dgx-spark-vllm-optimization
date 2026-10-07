# PREFILL_CAP — a smaller thin/fat split for the E3 prefill branch (`GLM53_PREFILL_FUSED_CAP`)

Status 2026-09-28, branch `prefillcap`. Built and measured on nodeC (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`) against both production module versions (the
launcher overlay `docs/ref/prod_live/overlay_exl3.py` and the image's `quantization/exl3.py`). Production (nodeA/nodeB)
was not accessed; nothing is installed there. Every number below is in a committed log under `docs/logs/prefill_cap/`.

## 1. What the profile says

R12 profile of a 15.6k real-text prefill, rank 0 (`prof4/rankx-3139-20260928-035431`): two engine steps, 13824 tokens
and a 1791-token remainder, 42 MoE layers each. Per MoE layer call (median over the 42 layers):

| chunk | thin `exl3_moe_kernel<4,256>` | E3 `fm_gather + fm_gateup + fm_down` | sum over 42 layers |
|---|---|---|---|
| 13824 tokens | 30.3 ms | 47.8 ms | thin 1.24 s, E3 2.02 s |
| 1791 tokens | 26.7 ms | 0.7 ms | thin 1.13 s, E3 0.03 s |

Production's E3 branch (`apply_exl3_fused_moe`, overlay :1509-1526) sends every expert with <= cap = 256 rows
(`layer._exl3_fused_temps[0].shape[1]`, `EXL3_TEMP_ROWS_FUSED=256`) to the fused thin kernel and the rest to E3. The thin
kernel re-decodes the whole expert for every 16-row block and runs `exl3_moe_max_concurrency()` = 6 experts at a time
(8 SMs each), so the remainder chunk, where almost every expert has <= 256 rows, costs nearly as much as the 13824 chunk.

## 2. What it does

`glm53_prefill_cap.py` wraps production's `apply_exl3_fused_moe` (called by module-global name from
`apply_exl3_experts`). With `GLM53_PREFILL_FUSED_CAP=n` (1 <= n < 256), a call is run with
`layer._exl3_fused_temps` swapped for temps with n rows (same dtypes, same concurrency, allocated once per
(device, hidden, intermediate, concurrency, n) and shared by all layers, like production's own) and restored in `finally`,
only when all of these hold:

- tokens > the layer's fused cap (256): prefill only; decode and every CUDA-graph capture size (<= 64) never qualify;
- the call takes production's E3 branch: tier `grouped`, `EXL3_FAT_GROUPED=1`, `EXL3_MOE_ROW_TILE` off (the other
  fat tiers would get slower with a small cap);
- no CUDA graph is being captured (no allocation during capture).

Production's own code then sends experts with <= n rows to the thin kernel (it skips `token_count >
max_tokens_per_expert`, `docs/ref/xl_exl3_moe_kernel.cuh:56`) and experts with > n rows to E3 (it reads the same cap).
No production code is copied or changed.

- Enable: `GLM53_PREFILL_FUSED_CAP=<n>` (read once when the vLLM plugin loads, `integrate.plugin_register`, independent
  of `TF_EXL3_MOE`). **Recommended n = 1** (section 3). Unset, empty, `0`, `off`: nothing is installed (inert).
- Refusal (WARNING, nothing installed): an invalid value; a production module whose `apply_exl3_fused_moe`,
  `apply_exl3_grouped_fat`, `build_grouped_fat_tables` or `_grouped_scratch` is not the fingerprinted version
  (sha256 of the AST; identical in both known module versions: `fa19d593…`, `b84e24e4…`, `5ebe863a…`, `a9c935fd…`).
- Per call pass-through with a WARNING once: a prefill call on a layer that is not on the E3 branch; n >= cap.
- Logs: `glm53 prefill fused cap installed: GLM53_PREFILL_FUSED_CAP=n ...` at load (INFO) and
  `glm53 prefill fused cap active: first prefill MoE call with T tokens ran with thin cap n ...` at the first prefill.
- Composes with the TF fork's K2 apply hook in either order (both keep the next function in `_tf_exl3_orig`; K2's
  fingerprint check follows it). Production order (plugin_register): TF first, so the chain is cap -> K2 -> production.
- Revert: unset and restart vLLM.
- TP=2: set it on both ranks (the head `docker run` lists env variables explicitly — the R1 incident); a rank without it
  is only slower, not different in results beyond the numerics class below.

## 3. Speed (nodeC, per MoE layer call of production's apply through K2 + the wrapper)

`tests/bench_prefill_cap.py`: 288 experts, hidden 4096, intermediate 1024 per rank, top-8, production prefill env
(`EXL3_FUSED_MOE=1 EXL3_FAT_KERNEL=1 EXL3_FAT_GROUPED=1 EXL3_TEMP_ROWS_FUSED=256 MAX_NUM_BATCHED_TOKENS=16384`), 2 layers
(different weights and routing), 5 rounds with the n order rotated, median. Routing: Gumbel top-8 over per-call expert
popularity logits — `real` lognormal sigma 1.1, `zipf` -1.3·log(rank+8), `collapsed` sigma 3.0 (random-token prompts).
Histograms are printed in the logs (e.g. real T=13824: <=16:2 17-32:16 33-64:32 65-128:54 129-256:62 257-512:68
513-1024:37 >1024:17).

**Calibration**: the `real` routing reproduces production's per-layer kernel times (profiled call, n = 0):
T=13824 thin 34.1-34.6 / E3 47.7-48.5 ms (production 30.3 / 47.8), T=1791 thin 27.8-28.0 / E3 0.8-0.9 ms
(production 26.7 / 0.7).

ms per layer call, run 1 / run 2 (`bench_run1.log`, `bench_run2.log`), speedup vs n = 0 (production today):

| routing | T | n=0 (prod) | n=1 | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|---|---|---|
| real | 1791 | 28.5 / 29.0 | 13.1 / 14.1 (2.17 / 2.05x) | 13.5 / 14.1 | 13.8 / 14.2 (2.07 / 2.05x) | 15.5 / 16.6 | 18.5 / 19.3 | 23.6 / 25.6 |
| real | 4608 | 46.8 / 46.7 | 24.5 / 24.4 (1.91 / 1.91x) | 24.2 / 24.6 | 24.4 / 24.8 | 25.1 / 25.7 | 29.3 / 29.8 | 37.8 / 38.5 |
| real | 13824 | 86.5 / 87.8 | 65.4 / 66.5 (1.32 / 1.32x) | 65.7 / 66.7 | 65.4 / 66.6 | 66.0 / 67.0 | 67.9 / 69.2 | 73.9 / 74.7 |
| zipf | 1791 | 24.4 / 25.0 | 13.6 / 13.6 (1.79 / 1.84x) | 13.8 / 14.0 | 14.4 / 14.4 | 16.0 / 16.3 | 18.5 / 18.7 | 21.9 / 22.4 |
| zipf | 4608 | 46.8 / 47.5 | 24.1 / 24.4 (1.94 / 1.94x) | 24.3 / 24.6 | 24.4 / 24.8 | 27.0 / 27.2 | 30.7 / 31.4 | 36.1 / 36.5 |
| zipf | 13824 | 94.8 / 95.4 | 65.6 / 66.8 (1.44 / 1.43x) | 65.7 / 66.4 | 65.6 / 66.4 | 65.6 / 66.7 | 67.6 / 68.8 | 77.3 / 77.9 |
| collapsed | 1791 | 17.2 / 17.5 | 12.3 / 12.2 (1.40 / 1.43x) | 12.5 / 12.9 | 12.7 / 12.9 | 13.1 / 13.3 | 14.2 / 14.5 | 15.1 / 15.4 |
| collapsed | 4608 | 31.7 / 32.0 | 24.1 / 24.3 (1.32 / 1.32x) | 24.2 / 24.5 | 24.6 / 25.0 | 24.8 / 25.5 | 26.2 / 26.9 | 28.8 / 29.3 |
| collapsed | 13824 | 73.8 / 74.5 | 65.4 / 66.3 (1.13 / 1.12x) | 65.8 / 66.6 | 65.6 / 66.2 | 66.5 / 67.3 | 67.5 / 68.6 | 70.7 / 71.8 |

(n = 48 and 96 are in the logs; they sit between their neighbours.) Short prefill calls (`bench_smallT.log`, one run,
real / collapsed): T=257 1.20x / 1.98x, T=384 1.36x / 1.73x, T=512 1.39x / 1.50x, T=768 1.70x / 1.72x, T=1024
1.88x / 1.52x at n = 1. No n is slower than n = 0 in any of the 28 measured (run, routing, T) cells; n = 1 is the fastest
or within 1.0 % of the fastest n in all 28 (the per-cell spread is 0.3-12 %); n = 16 is 0-6 % behind n = 1 at
T >= 1791 and 3-13 % behind at T <= 1024.

Kernel breakdown (real, one profiled call, run 2): T=13824 n=0 thin 34.6 + E3 48.5 + other 2.8 ms -> n=16 thin 0.2 +
E3 64.2 + other 2.5 ms; T=1791 n=0 thin 28.0 + E3 0.8 -> n=16 thin 3.4 + E3 9.8 ms. The rows that move (experts with
17..256 rows: 12.0k rows at T=1791, 18.8k at T=13824) cost the thin kernel 24.6 / 34.4 ms and E3 9.0 / 15.7 ms:
2.7x / 2.2x cheaper per moved row.

**Production projection (not measured in production)**: per 15.6k prompt per rank, (86.5 - 65.4) + (28.5 - 13.1) =
36.5 ms per layer pair x 42 layers = **~1.5 s** of MoE time (~1.3 s if production's own per-layer times, 30.3 + 47.8 and
26.7 + 0.7 ms, are taken as the baseline). The R12 15.3k real-text prefill runs at 1214 tok/s (12.6 s): expected
~11.1-11.3 s, **+11-14 % prefill throughput**. Decode is untouched (section 5).

After this change the MoE prefill time is E3 (~61-64 ms per 13824-token layer call); a faster fat kernel (bigger tiles:
E3 decodes each expert's weights once per 64-row tile) is the next MoE lever.

## 4. Numerics (`tests/test_prefill_cap.py`, `test_live.log` overlay module, `test_image.log` image module)

Experts with n < rows <= 256 move from the thin kernel to E3, the class production already uses for experts with > 256
rows; the fp32 output accumulation order changes. The two differ in (read from the sources): the thin kernel stores gate
/ up in fp16 temps and computes the activation in fp16 (`docs/ref/xl_hadamard_inner.cuh:360-387`) as
min(silu(g), L) * clamp(u, -L, L); E3 computes it in fp32 (`docs/ref/mia_exl3-fat-kernel/exl3_fat_moe.cu:349-369`) as
silu(min(g, L)) * clamp(u, -L, L). The clamp position differs only where g > L (L = 10: 10 vs silu(10) = 9.99955,
4.5e-5 relative, ten times below one fp16 rounding).

- Parity vs production's cap-256 path, production shapes, T in {1791, 4608, 13824} x routing {real, collapsed}:
  rel-L2 <= 7.4e-4 (n = 1), 7.1e-4 (n = 16), 4.2e-4 (n = 128); max |diff| <= 8.3e-4 of max |out|; per-row rel <= 9.4e-4.
  Noise floor of the cap-256 path itself (run twice, E3's fp32 atomics): rel-L2 <= 8e-9.
- float64 reference (the `kernels/exl3_format_ref.py` definition, weights dequantized with `exllamav3_ext.reconstruct`,
  checked bit-equal to the reference unpack), 96 sampled tokens, SwiGLU limit 10:
  - pure thin vs pure E3 (T = 2048, 64 experts x exactly 256 rows: all thin at cap 256, all E3 at n = 1 / 16):
    thin rel-L2 8.49e-4 (max abs 7.2e-3 of |ref| max 7.3), **E3 5.57e-4 (max abs 3.9e-3)**, ratio 0.66;
  - real routing T = 1791: cap 256 8.17e-4, n = 1 5.57e-4 (ratio 0.68), n = 16 5.84e-4 (0.71).
  (The reference uses the thin kernel's clamp position.) The difference class is "E3 instead of thin", and E3 is the
  more accurate of the two here. The difference vs today (<= 7.4e-4) is the size of either path's own error vs float64
  (5.6-8.5e-4).
- Not measured: KL on real prompts (production quality probe). The R11/R12 note applies: the quality probe texts are
  short, so E3 is barely exercised by them; prefer a long-prompt KL check when rolling out.

## 5. Decode, memory, capacity, wiring (all in `tests/test_prefill_cap.py`, 157 checks, both module versions)

- Decode unchanged: T = 1, 8, 64, 256 through the full stack with the wrapper installed pass through untouched (same temps
  object, swap counter unchanged); a CUDA graph captured at T = 64 through wrapper -> K2 is served by TF, replay = eager
  (6.7e-8). A call made during capture with T > cap passes through (no allocation).
- Hook guards (fake production apply): swap only for tokens > cap on a grouped layer, restored after the call and after
  an exception, pass-through for tier `kernel`, `EXL3_FAT_GROUPED=0`, `EXL3_MOE_ROW_TILE=1`, n >= cap.
- Memory: the small temps are n x 122,880 B (6 x n x (2 x 4096 + 2 x 1024) x 2 B): 120 KiB at n = 1, 1.9 MiB at
  n = 16 (production's 256-row temps: 31.5 MiB, unchanged and still used by decode).
- E3 scratch: sized from `token_sorted.numel()` = tokens x 8, independent of the cap; with `MAX_NUM_BATCHED_TOKENS=16384`
  it is 131,072 rows (1,342,177,280 B) >= 110,592 needed at T = 13824, identical before / after every n (checked per
  (routing, T)), never grown. The segment-table bound `rows_cap / 64 + n_experts` holds for any cap. A production
  `EXL3_FAT_GROUPED_TOPK` < 8 would make E3 grow its scratch once on the first long call — production's behaviour today,
  not affected by the cap.
- Wiring: env parsing (unset / "" / 0 / off inert, invalid -> WARNING), refusal of a module with a changed E3 function,
  both install orders with TF (K2 still installs), uninstall restores production's function object.
  `tests/test_integrate.py` (75 checks) and `tests/test_apply_fused.py` (20) pass with the new plugin_register step.

## 6. Rollout (not done here)

1. Build the bundle from this branch (the wheel's `py_modules` now includes `glm53_prefill_cap`; `integrate.py` calls it).
2. `.env`: `GLM53_PREFILL_FUSED_CAP=1`; add it to the head `docker run` env passthrough in start.sh.
3. Check on BOTH ranks: `glm53 prefill fused cap installed: GLM53_PREFILL_FUSED_CAP=1` and, after the first long prompt,
   `glm53 prefill fused cap active: ...`; `exl3 e2 diag ... grouped_calls` keeps growing, `grouped_scratch_bytes` unchanged.
4. Measure: long-prompt prefill tok/s (8.5k / 15.3k real text), decode bench unchanged, long-prompt KL vs the previous stage.
5. Rollback: empty the variable and restart.

Files: `glm53_prefill_cap.py`, `integrate.py` (plugin_register), `setup.py` (py_modules), `tests/prefill_cap_common.py`,
`tests/test_prefill_cap.py`, `tests/bench_prefill_cap.py`, `tests/run_all.sh` (new step), logs `docs/logs/prefill_cap/`.
