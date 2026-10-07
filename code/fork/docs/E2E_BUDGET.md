# End-to-end byte budget of one decode step (GLM-5.3-Flash EXL3 4bpw, TP=2) — 2026-09-27

Sources: HF shard headers of Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (tests/model_byte_budget.py; per-tensor table derived
from the same headers), GB10 read ceiling 250 GB/s (tests/bw_ceiling.py), production step composition from the 2026-09-22
measurements (memory: verifier forward without speculation 55.4 ms = 18 tok/s; speculative step ~105 ms; step grows ~8 ms per
extra K). Production nodeA/nodeB were NOT accessed for this document.

## Checkpoint composition
| part | GiB (whole model) |
|---|---|
| routed experts (EXL3 4bpw, 43 layers incl. MTP) | 145.55 |
| attention (BF16; 34 KDA linear-attention + 11 DSA/MLA layers) | 11.74 |
| shared experts (BF16) | 2.02 |
| lm_head / embed_tokens (BF16) | 1.18 / 1.18 |
| dense MLP layers 0-2 (BF16) | 0.94 |

## Per rank, per verifier forward
- routed experts: 6 MiB per expert per rank; T=1 reads 8 x 42 = 1.97 GiB. Grows with the number of DISTINCT experts per call.
- current production exl3_moe: 42 x 353 us = 14.8 ms at T=1 (27% of the 55.4 ms forward). TF fork: 42 x 226 us = 9.5 ms -> -5.3 ms.
- GLM53_DENSE_FP8 (Mia's switch; production uses dense,kda). Not yet enabled groups, computed from the tensor table:
  - mla  (fused_qkv_a replicated, q_b column-, o_proj row-parallel): 1144 MiB BF16 -> FP8 saves 572 MiB = 2.40 ms @250 GB/s
  - shared (shared experts, column/row-parallel):                     1008 MiB BF16 -> FP8 saves 504 MiB = 2.11 ms @250 GB/s
  Mia's KL panel (0.005-0.017 nats/token, argmax 95-98%) covers dense,kda only; mla/shared quality is UNMEASURED.

## Implication
Kernel work on the routed experts alone is bounded: the TF path already reads at 222-235 GB/s against a 250 GB/s ceiling.
The next largest levers per forward are (1) the fork (-5.3 ms at T=1, more at larger T), (2) GLM53_DENSE_FP8 += mla,shared
(-4.5 ms, config-only, needs a quality check), (3) the drafter share of the speculative step (~half of 105 ms), outside EXL3.

## Measured on nodeC (2026-09-27): GLM53_DENSE_FP8 += mla,shared  (tests/fp8_groups_check.py, docs/logs/fp8_groups_check.log)
Real BF16 weights (HF), production quantization (Glm53DenseFp8Method: per-output-channel e4m3, Marlin), rank-0 shards of TP=2.
- Output relative error per matrix (median/max): mla 2.54e-2/2.62e-2, shared 2.58e-2/2.71e-2 vs production-accepted
  kda 2.79e-2/2.84e-2 and dense 2.28e-2/2.30e-2. Unchanged with 1/256 outlier channels x30. Marlin runs on every mla/shared shape.
- Time saved per forward per rank (BF16 F.linear vs FP8 Marlin, cold weights, summed over 11 MLA + 42 shared layers):
  T=1 7.59 ms, T=8 5.02 ms, T=64 4.58 ms. (BF16 cuBLAS at T=1 runs these shapes at ~146 GB/s.)
- Per-matrix error parity is necessary, not sufficient: MLA projections feed attention scores. Model-level KL (Mia's
  scripts/quality/kl_panel.py on production) must be run before adopting.
