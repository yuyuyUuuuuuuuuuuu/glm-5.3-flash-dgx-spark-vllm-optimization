# OPT_DENSE — dense projections / MLA prefill / indexer (branch opt-dense, from r16z2rev 60b53bf)

2026-10-02, nodeC only (`tests/gpu_run.sh` / `tests/handoff/run.sh` under `/tmp/tf-gpu-bench.lock`). nodeA/nodeB
read only. Every number below says how it was measured; logs in `docs/logs/opt_dense/`.
All new switches default OFF (unset = production byte for byte).

## Results

| item | switch | numerics | measured | expected per 32k request |
|---|---|---|---|---|
| idx_gate per-M NaN-aware verdict (d73bea9) | quickwins (on) | unchanged (bitwise item) | production log shows idx_gate switched off at the M=16384 profile run; Triton head gate 0.648 vs 3.564 ms @13824 | ~-32 ms/13.8k chunk => ~-70 ms |
| MLA prefill fused index pass (9a3a61a) | GLM53_MLA_PREFILL (on), `_FUSED_INDEX=0` reverts | bitwise (whole kv_indices buffer) | forward_mqa 33.00 -> 30.28 ms @13824, 10.93 -> 10.12 @4289 | ~-70 ms |
| FP8 SP all-gather into served KDA in_proj (6c54e15) | GLM53_DENSE_W8A8_FP8AG=1 | bitwise (in_proj output) | quant 0.770 -> 0.389 ms/KDA layer; AG bytes halved (loopback NCCL 25.9 -> 13.5 ms incl. quant) | ~-130 ms (model: 2.7 ms AG/56.6 MB on RoCE) |
| W8A8 drafter fc (1647923) | GLM53_DENSE_W8A8_ONLY=...,draft.fc | drafter only (rel 2.7e-2) | 28.5 -> 19.7 ms @13824, 14.6 -> 6.8 @4289 | ~-25 ms; acceptance unmeasured |
| hi+lo W8A8 (64cbb0d) | GLM53_DENSE_W8A8_HILO=..., _SEL=first | quality UP at ~+23 ms/chunk | MoE-mini KL: allx+HA 0.00340 < today's sub 0.00392 | enables allx: ~-150 ms/chunk net => ~-0.35 s (KL frozen-first 0.00384 vs sub 0.00392) |

Gate estimate if all land (32k = 2 x 13,824 + 4,289, today ~12.5 s at 2,620 tok/s): bitwise items ~-0.27 s
(+~2 %), allx+HA ~-0.35 s more (+~3 %) -> roughly 2,620 -> ~2,750-2,800, subject to the production A/B.

### hi+lo W8A8 (the quality lever)
The W8A8 error is the e4m3 mantissa of the activations (scale choice irrelevant). Measured on real activations
(`tests/optdense/kl_driver_hilo.py`), the error energy of o_proj / down_proj / q_b inputs sits in a few, STABLE input
channels (selection from held-out rows == oracle). Their e4m3 residual is appended as C extra K columns of the same
GEMM with the same per-token scale. Error^2 left: kda.o C256 0.46, shared.down C128 0.38, dense.down C512 0.49,
q_b C256 0.59, mla.o C512 0.64 (in_proj C512 0.82, gate_up 0.6-0.8: spread out, not worth it).

KL on the handoff MoE mini (996 positions vs W8A8 off; A/A = 0):

| arm | KL mean | p99 |
|---|---|---|
| sub (production: in_proj + mla.o) | 0.00392 | 0.0235 |
| sub + mla.o C512 (per call / frozen first) | 0.00293 / 0.00303 | 0.0152 / 0.0226 |
| allx (sub + kda.o, q_b, shared.gate_up, shared.down, dense.gate_up) | 0.00459 | 0.0252 |
| **allx + HA** (per call / frozen first) | **0.00340 / 0.00384** | 0.0184 / 0.0243 |
| allx + HA, fast path in the real engine (REPACK_LD ext built on nodeC) | 0.0038354 (== frozen-first prototype to 7 digits) | 0.0243 |
| all | 0.00559 | 0.0379 |
| all + HA (per call / frozen first) | 0.00430 / 0.00446 | |
| all + HB (HA + in_proj, fused_qkv_a, gate_ups C512) | 0.00391 | 0.0243 |

HA = `kda.o_proj:256,shared.down_proj:128,dense.down_proj:512,mla.q_b_proj:256,mla.o_proj:512`.
Fast path: strided repack (`fp8_marlin_to_std` row-strided out, ext attr `REPACK_LD`), one Triton pass for
`[q(x) | q(r_S)]` (q(x) == the image's per-token quant bit for bit), custom GEMM with K+C (swept tile configs);
== the torch prototype bitwise (`tests/optdense/test_hilo.py`). Cost per call @13824: kda.o +0.16, mla.o +0.43,
q_b +0.45, dense.down +0.27, shared.down +0.18 ms. Extra UMA: one [N, K+C] fp8 scratch per K+C (~100 MB for HA).
The channel set freezes at each layer's first real served call (the profile run is skipped): numerics then depend
on the first prompt after boot (deterministic afterwards). With FP8AG, in_proj never takes hi+lo (pre-quantized).

**Production A/B proposal**: `GLM53_DENSE_W8A8_ONLY=<allx>` + `GLM53_DENSE_W8A8_HILO=<HA>` +
`GLM53_DENSE_W8A8_HILO_SEL=first` (+ `GLM53_DENSE_W8A8_FP8AG=1`), gates as DEPLOY_R16Z (long KL <= ~0.0165,
dvp <= ~0.023). Needs: rebuilt AOT `tf_fp8_w8a8_ext` from this branch's `kernels/fp8_w8a8.cu` (REPACK_LD; without it
the slow torch prototype serves) and make_start_sh forwarding of `GLM53_DENSE_W8A8_FP8AG`, `_HILO`, `_HILO_SEL` to
both ranks; the generated start.sh ONLY-validation case list must gain `draft.fc` if that is used.

## Negative results (do not retry)
* e4m3 P.V inside the MLA prefill kernel: <= -2.4 ms/layer upper bound (cost-model ablation), not built (b5bd997).
* Prefetching the next layer's repack on a side stream under the current GEMM: 0 hidden (the persistent GEMM holds
  every SM); quant || repack also 0 (both DRAM-bound) - `bench_overlap.py`.
* Indexer BF16 GEMMs: wq_b 2.0 ms (87 TF), wk_weights_proj 0.55 + kpool gate 0.58 (DRAM-bound reads of the same x);
  fusing the two saves <= ~5 ms/chunk - `bench_indexer_gemm.py`.
* hi+lo on in_proj / fused_qkv_a / gate_up: error^2 only 0.6-0.82 at C512 (energy spread over channels).
* Resident std-layout weight copy: rejected earlier (3.5 GiB/rank); the per-call repack of the shipped subset is
  ~20 ms/chunk and cannot be hidden (above).

## Open ideas (not done)
* Lower the drafter-fc large-M threshold (8192) for M=4289 (Marlin 14.6-16.1 ms there) - exact path, ~-5 ms/request.
* hi+lo without the per-layer [N, K+C] scratch (reuse the base scratch with a wider stride) to save ~100 MB UMA.
* fp8 all-gather for the MLP input when gate_up projections are W8A8-served (dense layers only; the MoE router
  needs bf16).
