# MLA_PREFILL: exact sparse-MLA prefill attention kernel (`GLM53_MLA_PREFILL`)

Status 2026-09-28, branch `mlaprefill` (from `deploy-r13`). Built and measured on nodeC (one GB10, sm_121a) in the
production image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, with production's SM90_KV mounts
(`tests/mla_env.sh`: FlashInfer 0.6.18 over site-packages, the patched `vllm/platforms/cuda.py` and
`flashinfer_mla_sparse_sm90.py`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, and the launcher overlay `exl3.py`).
Production (nodeA/nodeB) was not touched. Every number below is in a committed log under `docs/logs/mla_prefill/`.

## 1. Result

Per MLA layer and rank (32 heads), synthetic inputs at production shapes (section 8). "forward_mqa" is the whole
backend call: production = index conversion + clamp + copy into the wrapper + FA2 run; new = index conversion + kernel.

| case | production FA2 kernel | new kernel (v4) | production forward_mqa | new forward_mqa | error vs fp32 (rel L2 max / mean) |
|---|---|---|---|---|---|
| 13824 rows, ctx 1..13824 (step 0 of a 15.6k prompt) | 99.8 ms | **28.6 ms** | 103.9 ms | **30.4 ms** | 2.8e-3 / 1.76e-3 (FA2 exact lengths: 2.9e-3 / 1.74e-3) |
| same, independent top-k sets | 99.6 | 29.1 | 103.6 | 30.7 | 2.9e-3 / 1.76e-3 |
| 1791 rows, ctx 13825..15615 (step 1) | 14.3 | **4.2** | 14.5 | **4.4** | 2.9e-3 / 1.75e-3 |
| 13824 rows, ctx 86001..99824, neighbour-sharing top-k | 109.3 | **32.5** | 114.7 | **33.6** | 2.7e-3 / 1.76e-3 |
| same, independent top-k (KV 51 MB, larger than L2) | 119.6 | 42.7 | 123.6 | 44.9 | 2.8e-3 / 1.75e-3 |
| 256 rows at ctx 15k (smallest call it serves by default) | 2.24 | 0.67 | | | 2.7e-3 |

Logs: `final_T13824_s0_sticky.log`, `final_T13824_s0_indep.log`, `final_T1791_small.log`, `final_T13824_s86000.log`.

Projected saving (11 MLA layers): **-0.81 s per 13.8k chunk** (1.14 s -> 0.33 s of forward_mqa), -0.11 s for the 1.8k
remainder, about -0.9 s per 13.8k chunk at 100k context. For the 15.6k real-text prefill of the R13 profile
(1324 tok/s, step 0 = 10.6 s) that is roughly 11.8 s -> 10.9 s, **about +8 % prefill throughput** from this kernel alone.
Memory cost: none beyond production (same output tensor, same conversion buffers; the kernel uses 88.5 KB of shared
memory per CTA and no global workspace; production's clamp/copy into the FA2 wrapper is skipped).

## 2. What runs in production today, and why it is slow

`FLASHINFER_MLA_SPARSE_SM90` (SM90_KV=1) runs every prefill row as its own varlen batch row of FlashInfer's FA2 kernel
`BatchMLAPagedAttentionKernel<CAUSAL=0, STAGES=1, QK_SHARD=0, CKV=512, KPE=0, TILE_Q=64, TILE_KV=16>`: on sm_121's
99 KB shared memory it falls into the 1-stage / 16-key bucket, half the 64-row Q tile is padding (32 heads per row), QK
is computed by both warpgroups and gathers are not overlapped with math. Measured here: 99.8 ms per layer at 13824 rows,
17 useful TFLOPS (the R13 trace shows 100.5-101.3 ms).

Tensor-core ceiling on this GB10 (`tests/probe_mma_peak.py`): bf16 `mma.sync` m16n8k16 with fp32 accumulate
**107 TFLOPS**, e4m3 m16n8k32 214 TFLOPS. The useful work of one 13.8k layer is 1.716 TFLOP, so the bf16 MMA floor is 16 ms.

## 3. L1 (ceiling probe): FlashInfer's SM120 multi-group sparse prefill kernel

`kernels/mla_prefill/l1_mg.cu` compiles FlashInfer 0.6.18's `sparse_mla_prefill_mg_kernel<GLM_NSA, FP8, 32, 2048, 64>`
(32 heads per CTA sharing one gather, 64-key tiles, FP8 QK with per-128 Q scales, 2-pass FP8 P.V) against a temporary
656-byte packed copy of the 512-byte cache (`pack656`, 0.05 ms) and Q padded to 576 (4.4 ms copy):

- 56.2 ms kernel, 60.7 ms with packing, 30.5 TFLOPS: only 1.8x over FA2, far from the 20-35 ms estimate.
- rel L2 error 4.4e-2 mean / 0.20 max, 25x FA2's, from the FP8 Q (the image default and SGLang on GB10 use this).

So the in-image kernel is neither fast enough nor accurate enough; L2 was built instead.

## 4. The exact kernel (`kernels/mla_prefill/mla_prefill_kernel.cuh`)

Numerics follow production FA2 operand for operand: K = V = bf16(fp8 e4m3) (times bf16(ckv_scale), rounded, when the
scale is not 1: FA2's own `__hmul2`), Q bf16, QK `mma.sync` bf16 -> fp32, p = `ex2.approx(s*scale_log2 - m*scale_log2)`,
P rounded to bf16 for the PV `mma.sync`, row sum from the fp32 p, out = o * (1/l) rounded to bf16. The only differences
are fp32 summation order (4 dim-quarter partial sums added in a fixed order) and the running-max schedule (per 32 keys
instead of per 16), which changes which bf16 value p rounds to but not the error bound. Error vs an fp32 reference equals
FA2's (table above); the fp8 -> bf16 conversion is bit-exact for all 256 codes and for pow2 and non-pow2 scales
(`tests/test_mla_prefill.py`).

Structure of variant 4 (default; variants 2 and 3 kept, selectable with `GLM53_MLA_PREFILL_VARIANT`):

- Persistent CTAs (one per SM, 12 warps: 8 math + 4 IO with `setmaxnreg` 232/40) walk the tokens round-robin; the KV ring
  (4 stages x 32 keys, rows padded to 528 B, `cp.async.bulk` per 512-byte row, mbarrier full/empty) and the barrier
  phases continue across tokens, so the next token's first gathers overlap the current one's tail, and the next Q is
  loaded during the last sub-tile.
- Per 32-key sub-tile: QK — warp (dim quarter wq, key half kh) computes S[32 heads x 16 keys] over its 128 dims with Q in
  registers (64 regs), both 16-head groups sharing one converted K fragment; the fp8 bytes are placed with `prmt` and
  turned into bf16 with a shift/mask/`mul.rn.bf16x2` by 2^120 (exact for normals, subnormals and zero).
  Softmax — warp w owns heads 4w..4w+3: sums the 4 partials (fixed order), masks positions >= the row's valid count,
  keeps the running max / row sum and writes P (bf16) and the rescale factor to shared memory (double-buffered).
  PV — warp w accumulates O[32 heads x its 64 dims]; V fragments come from `ldmatrix.trans` on the fp8 bytes
  (even/odd dims split into two n8 tiles), again shared by both head groups.
  QK of sub-tile n runs before PV of sub-tile n-1 (software pipeline); one CTA-wide barrier per sub-tile.
- Row lengths come from the device-side valid counts of `triton_convert_req_index_to_global_index`, not from a host plan.

Where the time goes (ablations, `tests/probe_mla_ablation.py`, variant 3 at 13.8k): without any MMA the kernel still
takes 18 ms; the MMA pipe is about 55 % busy. The rest is fp8 conversion (about 2 ms), the per-sub-tile softmax phase
(1.7 ms), gathers (2 ms), shared-memory exchanges and the Q/O DRAM streams (2.3 ms). What did not help: L2 cache
hints (KV evict_last, Q/O evict_first: +1.6 ms at 13.8k, equal at 100k; `GLM53_MLA_L2HINT`), replacing the bf16
multiply (-0.3 ms), pipelining alone. What helped: sharing each converted K/V fragment across both head groups
(v2 32.6 -> v3 29.0 ms at 13.8k), a 4 x 32-key KV ring (100k context, independent top-k: 47.6 -> 41.5 ms) and
persistent CTAs on top of it (13.8k: 32.7 -> 28.7 ms; 100k with neighbour-sharing top-k: 34.5 -> 31.9 ms).

## 5. Integration: `glm53_mla_prefill.py`

Loaded from `integrate.plugin_register` in every vLLM process; **inert unless `GLM53_MLA_PREFILL=1`**. It wraps
`FlashInferMLASparseSM90Impl.forward_mqa` (a runtime wrap: production mounts the backend file read-only). A call goes to
the new kernel only if all of these hold, otherwise production's forward_mqa runs unchanged (bitwise, tested):

- not inside CUDA-graph capture (decode steps are replayed graphs and far below 256 rows, so **decode is untouched**);
- at least `GLM53_MLA_PREFILL_MIN_TOKENS` rows (default 256);
- 32 heads, kv_lora_rank 512, no rope, fp8 KV cache, contiguous 512-byte cache rows, bf16 q with 16-byte aligned rows;
- `GLM53_MLA_PREFILL_MIXED=0` additionally keeps steps that contain decode rows on FA2 (default 1: a mixed step with
  >= 256 rows runs all its rows, decode rows included, on the new kernel, with the same numerics class).

It returns output in the same head-major layout FA2 returns (`empty_like(q_nope)`), which `_v_up_proj` transposes back.
The AOT extension `glm53_mla_prefill_ext` is built by `setup.py`. Logs: "glm53_mla_prefill installed: ... variant 4 ...",
then "N calls on the exact kernel" at 1, 10, 1000, 100000 calls with the fallback counts; a call refused for a reason
other than small / capturing / mixed logs a WARNING once.

Env: `GLM53_MLA_PREFILL` (1/exact), `GLM53_MLA_PREFILL_MIN_TOKENS` (256), `GLM53_MLA_PREFILL_MIXED` (1),
`GLM53_MLA_PREFILL_VARIANT` (4). Rollback: unset `GLM53_MLA_PREFILL` and restart. The launcher must pass the variables to
both ranks (the head `docker run` lists env vars explicitly: same incident class as R1).

## 6. Finding: production's FA2 plan attends 4 keys that the indexer did not select

`flashinfer_mla_sparse_sm90.py.patched` plans FA2 on the host with `lens = ctx if ctx <= 2048 else 2048 + ctx % 4`, but
the GLM5Next kpool indexer keeps `pool_ids[:, :511]` (`sparse_attn_indexer_kpool.py:580, 874` in the image), so the
real valid count is `ctx` for ctx < 2048 and **2044 + ctx % 4** otherwise. `tests/test_mla_integrate.py` reproduces
this with the image's own `expand_pools_and_append_tail` + `triton_convert_req_index_to_global_index`
(`test_mla_integrate.log`: ctx 2048 -> valid 2044, plan 2048; ctx 100000 -> 2044 vs 2048; 17/24 probed rows).
For every row with ctx >= 2048, in prefill and in decode, FA2 therefore also attends:

- 4 - ctx % 4 copies of global slot 0 (the -1 tail is clamped to 0 "so over-scheduled reads land on a valid page"), and
- when ctx % 4 != 0, the first ctx % 4 entries of the next row's kv_indices (the next query's top-ranked keys, or for
  the last row of a step stale entries from an earlier step).

On the synthetic data this raises the worst-row error vs fp32 from 2.9e-3 to 0.9-1.4 (mean 1.7e-3 -> 4.3e-3); the size
in the real model depends on what slot 0 holds and how often a duplicated key dominates, and was not measured.
The new kernel uses the device-side valid counts, so with `GLM53_MLA_PREFILL=1` prefill no longer has this; decode
still does. A host-side fix (`lens - 4` when `lens >= 2048`) would be wrong for the short-prefill path (all prompt
rows <= 2048, where ctx = 2048 really has 2048 valid keys), so it is not included; the robust fix for decode is to
feed FA2 exact device-side lengths or a masked tail. The long-context KL probe will see this change together with the
kernel change.

**Handoff defect (found and fixed after deploy-r16, docs/HANDOFF.md).** The wrapper first skipped the copy of the
converted slots into the wrapper's process-wide `kv_indices` ("not needed" by the kernel). Decode needs it. Because of
the plan above, the last row of every decode step with ctx % 4 != 0 reads 1-3 entries past the rows that step wrote.
With production FA2 those entries are the request's own prefill selections. Without the copy they were an older
call's slot ids, i.e. another request's keys. That drove the production decode-vs-prefill KL from 0.006 to 0.17-0.29.
The wrapper now writes exactly production's bytes after the kernel launch. `tests/test_mla_integrate.py` checks the
whole buffer, and `tests/handoff/check.sh` checks it in the engine.

## 7. What others have (checked 2026-09-28)

- FlashInfer PR #5488 (open, not merged): `compute_precision="bf16_qk"` for GLM53 sparse MLA on SM120 — BF16 QK but
  still FP8 P.V, single-group kernel (16 heads per CTA), compact 528-byte rows; rel RMS 2.2e-3 vs fp32 on RTX PRO 6000;
  no GB10 timings, SM121 unvalidated.
- The SM120 sparse-MLA refactors (#4802, #5197) and the SGLang GLM-5.3 arm64 image use the FP8 MG path measured as L1.
- GLM-5.3 GB10 serving kits found by search (MiaAI-Lab, LimeChain, sparkglm, tonyd2wild, Reederey87, barrydeen): their
  descriptions name FlashInfer's FA2 SM90 path or the FP8 GLM_NSA path; none of the pages checked describes an exact
  bf16 sparse prefill kernel (only sparkglm's README was read in full). sparkglm calls the 2048-vs-2051 candidate
  mismatch a "semantic approximation" of the SM121 path.

## 8. Tests and reproduction

```
source tests/mla_env.sh
tests/gpu_run.sh python3 -u tests/test_mla_prefill.py     # conversion, parity vs fp32 and FA2, edges, kv_scale, determinism (v4/v3/v2)
tests/gpu_run.sh python3 -u tests/test_mla_integrate.py   # hook against production's backend file, fallbacks, capture, plan finding
tests/gpu_run.sh python3 -u tests/bench_mla_prefill.py --T 13824 --variants v0,v0x,l1,l2,hook --l2-variants 2,4
tests/gpu_run.sh python3 -u tests/probe_mla_ablation.py 0 1 2 4 24   # timing ablations (wrong results by design)
```
`tests/run_all.sh` runs the two tests and the benchmark with the SM90_KV mounts. Top-k inputs reproduce the production
layout (511 pools x 4 + tail, compacted by vLLM's converter) in three regimes: `sticky` (neighbour rows share ~90 % of
pools), `indep` (independent pools per row), `local`; the KV cache is fp8 in a shuffled block table.

Not verified here: the end-to-end KL / quality probe in production (both the kernel and section 6 change the output),
the TP=2 serving path itself, and real top-k sets (synthetic ones bracket the L2 behaviour: sticky vs independent).
