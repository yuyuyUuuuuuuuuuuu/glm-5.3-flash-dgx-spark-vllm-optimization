# data/

CSV tables behind the docs. All numbers come from production (2 x GB10, TP=2) unless a row says
"node C" (a single GB10 used for kernel tests). Times are JST. `source` names the original log, JSON or
report file on the measuring machine. The raw files behind the headline numbers are in `raw/` (see
`raw/README.md`); the others are not published. The `operator notes` source means the number was written down at
the time and no raw log survives.

| file | contents |
|---|---|
| `decode_bench.csv` | every `bench_decode.py` round from 09-27 to 10-05, one row per workload |
| `production_ab_runs.csv` | every production A/B arm log from 09-29 to 10-05 (decode + prefill + APC + quality probes), 72 rows |
| `speculation.csv` | K sweeps, adaptive K, drafter alternatives, block verification, verify trimming |
| `prefill_scheduling.csv` | MNBT and mixed-prefill sweeps (09-02 to 09-30) |
| `prefill_anatomy.csv` | where a prefill chunk's time goes (09-28, 09-30) |
| `prefill_kernel_microbench.csv` | per-kernel savings measured on node C, with the production A/B outcome |
| `decode_anatomy.csv` | decode step breakdowns (09-13, 09-28 R7, 09-28 R15, 10-05) |
| `decode_step_vs_rows.csv` | 10-05 2-rank profile by number of verify rows M |
| `concurrency.csv` | per-stream and aggregate tok/s at 1/2/4/8 streams |
| `ablit_transplant_sha256.csv` | the 31 o_proj tensors of the ABLIT transplant production loads: donor shard, shape, size and sha256 |
| `kv_capacity.csv`, `kv_budget_per_token.csv` | KV pool size, bytes per token, and per-component budget |
| `client_latency_0924.csv` | 24 h before/after the 09-24 scheduling bundle, measured on interactive client traffic |

## decode_bench.csv
- `run`, `workload`: run label and workload (structured / prose / coding / ja). `conc` runs use 1-8
  streams.
- `concurrency`, `temperature`, `thinking`: request settings (all temperature 0, thinking off).
- `tokps_median/min/max`: per-request decode tok/s over n=10, TTFT excluded. A very low `min` means a
  request overlapped other traffic.
- `aggregate_tokps`: total tok/s across streams (concurrency > 1 only).
- `accepted_per_step`, `drafted_per_step`: speculative tokens accepted / proposed per step (medians).
- `ms_per_step_median`: serving-cycle ms per draft step (`decode_ms_per_draft_step_median`). It is not
  a kernel time. For concurrency > 1 the tool divides the step across streams.
- `ttft_s`, `completion_tokens`: as logged.
- `stage`, `flag`: what the run was and any caveat (contaminated, rejected, reverted, current).

## production_ab_runs.csv
- `*_tokps`, `*_ms_step`, `*_acc`: decode per workload, as above. The `ja` columns start on 10-04.
- `pf_real8.5k_tokps`, `pf_real14.8k_tokps`: real-text prefill, median of 3. `pf_rand24k_tokps` uses
  random tokens and overstates MoE speed.
- `apc97k_cold_ttft_s`, `apc97k_warm_cached`, `apc97k_warm_ttft_s`: a 97,359-token prompt sent cold and
  then repeated. The warm run should have 96,768 tokens cached.
- `stream_gap_ms_*`: inter-token gap p50/p90/p99/max, plus the count of gaps over 200 ms, at 2k and 40k
  context.
- `dvp_kl`, `dvp_top1_pct`: decode-vs-prefill KL over 4,608 generated positions.
- `long_kl`, `long_kl_p95`, `long_top1_pct`: prefill KL at positions >= 4,608 of 3 long texts against
  stored reference outputs.
- `short_kl`: prefill KL on 6 short texts.
- `klh_nll`, `klh_dvptf`: teacher-forced harness NLL and decode-vs-prefill KL (from 10-04).
- `gate32k_tokps`: cold 32k prefill on an idle node. In the env-ab rows it is the median of about 6
  samples. In the 09-30 to 10-02 rows it is the first logged sample. Empty means the gate was not run or
  not logged.
- `kl_reference`: which reference the KL columns are against. Levels are comparable only within one
  reference.
- `arm`, `change`: the configuration under test.
- `status`:
  - adopted / rejected / reverted
  - `control (production config at the time)` for base and A/A arms, which were not changes
- Excluded as invalid: one contaminated base arm (structured 52.58 tok/s), one arm stopped because the
  dvp probe was busy, one empty log, and one partial log.

## Long-format tables
`prefill_scheduling.csv`, `speculation.csv` and the anatomy files have one row per measurement, with
the conditions in their own column. Values shown as `a / b / c` follow the order given in the `metric`
or `workload` column.
