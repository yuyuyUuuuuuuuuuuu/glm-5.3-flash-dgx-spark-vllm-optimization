# OPT_DECODEKIT — review of the decode path + kit tooling (branch `opt-decodekit`, from `r16z2rev` 60b53bf)

Status 2026-10-03. Everything ran on nodeC (GPU jobs through `tests/handoff/run.sh` / `tests/gpu_run.sh` under the
shared flock); production (nodeA/nodeB) was only read (files, `docker logs`, masked). Scratch + raw logs:
`${HOME}/tf-exl3-assets/opt-decodekit/`; committed logs: `docs/logs/opt_decodekit/`.

## 1. Items

| # | item | kind | status | effect |
|---|---|---|---|---|
| 1 | `GLM53_MLA_EXACT_LENS` — FA2 sparse-MLA decode plan 4 keys/row too long (slot-0 copies + the next row's first keys; last row: a stale call's) | correctness | implemented, default off, wired through kit + boot_checks (docs/MLA_EXACT_LENS.md) | per-row error of the over-read up to rel_l2 0.78 in a synthetic probe (§2); handoff-mini dvp KL unchanged (0.00566 vs 0.00597, noise); host +10.8 µs/step off the critical path |
| 2 | kpool "drops an arbitrary pool" (memory note 2026-09-30) | correctness (measured) | probes committed | prefill: always drops the lowest. decode (`persistent_topk`): usually bottom 3 %, at 64k sometimes rank 275-380 of 512 |
| 3 | drop-lowest dvp regression (production 0.0075 -> 0.0109) | correctness | root cause found + fixed (§3) | the score-SORTED column order causes it: stock 0.0057/0.0071/0.0084, sorted 0.0103/0.0109, order-preserving 0.0081 / 0.0075 |
| 3b | `GLM53_KPOOL_DROP_LOWEST=1` costs ~0.65 ms per decode step (torch helper, ~58 µs × 11 indexers) | decode speed | fixed: one Triton kernel, order-preserving by default (docs/KPOOL_DROP_LOWEST.md, last section) | 670-680 → 23-27 µs/step (= the stock slice); `fusedsort` (bitwise == old helper) 42-44 µs |
| 6 | handoff harness: a 0-byte worktree .so (left by a single-file GPU_RUN_BIND) silently replaced the kit's MLA prefill ext | test-rig defect | fixed in tests/handoff/run.sh (-s + ABORT) | 4 runs of this branch voided and redone |
| 4 | `env_r16.sh off mhcsp` left `GLM53_MHC_SP2=1` → start.sh refuses after restart2.sh stopped both ranks | kit defect | fixed: cascade + generic launcher pre-validation of every `.env` edit | tests/kit/test_env_r16_guard.sh ALL OK |
| 5 | `apply_r16.sh` validated only `#switch` enums: a `#knob` W8A8_ONLY name the kit does not serve / SP2 without SP / a missing overlay were installed and refused only after both ranks stopped | kit defect | fixed: the kit's own `validate_numeric_config` on the resulting tree + .env before anything is written | tests/kit/test_apply_r16_guard.sh ALL OK (A4 control: the production kit's apply accepts both) |

## 2. FA2 over-read, per row (`tests/mla_exactlens/probe_order.py`, `docs/logs/opt_decodekit/probe_order2.log`)

Real kernels (production FA2 0.6.18 page-size-1 + the exact kernel), GLM shapes, 64 decode-like rows at ctx 9,000 /
30,001, fp32 reference over the exact selected set:

| case | FA2 production plan rel_l2 mean / max | FA2 exact plan | exact kernel |
|---|---|---|---|
| ctx 9000 sticky, q_sigma 2 | 7.2e-03 / **7.8e-01** | 1.74e-03 / 2.6e-03 | 1.75e-03 / 2.7e-03 |
| ctx 9000 sticky, q_sigma 8 | 2.5e-03 / **8.1e-01** | 1.56e-03 / 2.4e-03 | 1.57e-03 / 2.7e-03 |
| ctx 30001 local, q_sigma 8 | 1.7e-03 / 1.8e-01 | 1.54e-03 / 2.7e-03 | 1.54e-03 / 2.7e-03 |

The max is not the last row (the "rows but the last" column is the same): the rows whose extra keys (slot 0 / the
next row's first pool) carry attention mass. Key ORDER, with exact lengths, moves the output only at rounding level
(rel_l2 ~1e-3, the same as FA2-vs-fp32); with the production plan order matters up to 0.4-1.5 (it decides WHICH
next-row keys are over-read).

## 3. drop-lowest: the regression is the column ORDER

Handoff mini (`tests/handoff/run.sh`, kit r16z2rev, GLM53_MLA_PREFILL=1 + production decode features, prompts 6000 +
3001, 96 greedy tokens, KL(decode || fresh prefill) at 96 positions; `exl_ab/<arm>/`; every arm's log shows the MLA
prefill installed):

| arm | config | dvp KL mean | p95 | top-1 |
|---|---|---|---|---|
| off1 / off3 / off4 | stock (three runs) | 0.00566 / 0.00711 / 0.00841 | 0.070-0.076 | 94 / 93 / 96 |
| on1 | + GLM53_MLA_EXACT_LENS=1 | 0.00597 | 0.064 | 93 |
| dloff | GLM53_KPOOL_DROP_LOWEST=1, r16l torch helper (score-sorted) | 0.01027 | 0.0802 | 94 |
| dlfs2 | same, `fusedsort` kernel (bitwise == helper) | 0.01088 | 0.0802 | 95 |
| dlon | score-sorted + exact lengths | 0.01051 | 0.0903 | 91 |
| **dlfused / dlfused2** | **drop the lowest, op's column order (`fused`; dlfused2 = the default, ORDER unset)** | **0.00814 / 0.00749** | 0.075 / 0.084 | 95 / 94 |

Stock spreads 0.0057-0.0084 run to run (the top-k op's column order is nondeterministic). Score-sorted drop-lowest
sits at 0.0103-0.0109 in all three runs (with or without exact lengths) = the production regression; dropping the same
pool with the stock order is inside the stock spread. So the regression is the ORDER of the kept pools (sparse-MLA
accumulation order, decode FA2 vs the prefill kernel), not the dropped pool. Void runs: off2 / dloff2 / dlstock /
dlfs (MLA prefill silently off, item 6).

## 4. Recommendations

* Kit tools (items 4, 5, revert tracer lines): ship with the next kit (host-only scripts; `test_kit_scripts.sh` on
  r16z2rev + these tools ALL OK 111, plus tests/kit/test_{env,apply}_r16_guard.sh ALL OK).
* `GLM53_KPOOL_DROP_LOWEST=1` with this branch's overlay: the correct pool set (fixes the 2026-09-30 defect: decode
  dropped a rank-275..380 pool at 64k) at stock decode cost and stock dvp (two runs: 0.00814 / 0.00749). Ship with
  the next kit; production A/B = the usual dvp + long KL probes. Never the r16l overlay (+0.65 ms/step, +0.004 dvp).
* `GLM53_MLA_EXACT_LENS`: correctness cleanup, no measured quality gain; ship with something else that needs a
  restart.
