# raw/

The raw files behind the headline numbers, so they can be checked without trusting the tables. Everything else in
`data/` cites its source file by name only.

| path | what | backs |
|---|---|---|
| `bench/vtrim-prod_{structured,prose,coding,ja}.json` | the `bench_decode.py` output of the current production configuration (10-05 17:30): 10 runs each, every run's timings, acceptance and generated text | `README.md` decode table, "current production" column; `data/decode_bench.csv` rows `vtrim-prod` |
| `bench/B2-comp0_{structured,prose,coding}.json` | the same for the pre-kernel-work reference (09-27 22:32) | the "B2" column |
| `measure/envab-*-20261005-152205.log` | the full probe set of each arm of the last production A/B (base, VTRIM on twice, an explicit-off arm) | `data/production_ab_runs.csv` rows of 10-05 15:22 |
| `measure/envab-*-20261003-194105.log`, `measure/envab-*-20261003-201531.log` | the probe sets of the 10-03 evening A/B series | rows of 10-03 19:41 and 20:15 (their 32k gate values are in `env-ab/`) |
| `env-ab/{run4b,run5,z8pvt}/` | the A/B driver's log and per-arm results (`res-<arm>.txt`: 32k gate median tok/s, long KL, long top-1 %, dvp KL, structured decode tok/s) | the 32k gate column; `run4b` holds the 10-03 20:14 arm `d` at 3,046 tok/s |

Produced by `code/operator/bench_round.sh`, `code/operator/r16_measure.sh` and `code/kit/tools/arm_env_ab.sh`. Host
names, home paths and addresses were replaced as in `code/`, and the A/B driver's lines about pausing local clients were dropped from `run.log`; nothing else was edited. The prompts are the synthetic
benchmark prompts in those scripts. `DEPLOY_DIR=raw python3 code/operator/bench_check.py vtrim-prod` re-runs the
output checks on these files.
