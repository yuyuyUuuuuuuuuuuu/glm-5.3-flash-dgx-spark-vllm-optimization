# tf-exl3-fork — optimization record (branch `opt`, 2026-09-27)

Goal: take the verified TF path (master) closer to the measured read ceiling of GB10 (250 GB/s), weighting the
production decode regime (DFlash2, adaptive k in {4, 5, 7}, up to 8 sequences: T = 5..64 per MoE call, correlated
routing), without giving up any part of the correctness contract (docs/DESIGN.md, tests/run_all.sh).

Everything below was measured on nodeC (GB10, 48 SMs, 24 MiB L2) inside the production image through
`tests/gpu_run.sh`, synthetic weights at the real dimensions (K = 4096, N = 1024, n = 288, topk = 8; 3 layers x
4 routings cycled, cold weights). nodeC is shared with other GPU services. Two kinds of measurement are used:

- **Paired A/B** (`tests/bench_variant_ab.py`): two TF configurations in one process, each captured into CUDA graphs of
  12 calls, >= 9 alternating rounds (order flipped every round); statistic = median over rounds of the per-round
  ratio B/A (< 1 = B faster). Every accept / reject decision (repeated as a separate run at the time), the re-audit
  (`--repeat 2` in one run) and the master -> final comparison use this.
- **E.5** (`tests/bench_ab_decode.py`, the last step of run_all): TF vs production exl3_moe, alternated within the
  run (9 rounds, medians of per-call us). The absolute us and the TF-vs-production speedups come from this.

Noise: a null run (`bench_variant_ab.py 0 0`: identical configurations, 26 configurations x 2 repeats,
`docs/logs/ab/null_0_vs_0.log`) gives |B/A - 1| median 0.20%, 90th percentile 0.76%, 95th 1.09%, max 2.67% (a
contended round at corr40 T = 8: 0.973 with a 30% round spread); no configuration had both repeats beyond +-0.8% in
the same direction. So a single ratio within about +-1% (occasionally +-2.7%) is noise. The re-audit below calls a
change significant at a configuration only when **both repeats** are below 0.992 (the original decisions asked for
the same sign in two runs and at neighbouring T; the example they cited, 0.984 for identical configurations, is 1.6%).

Logs (all committed; `tests/logs/` itself is gitignored scratch):
- `docs/logs/run_all/` — the final `tests/run_all.sh`: console output (`run_all.txt`) and every step's log;
- `docs/logs/ab/` — the paired runs behind the tables below (null, master -> final, one run per shipped item vs its
  predecessor), full per-configuration output: A / B us, GB/s, spreads, bitwise / E.1 verdicts;
- `docs/logs/history/` — the files of the original optimization runs, exactly as they were written (several
  confirmation runs were saved as summaries only; they are kept as they are and the re-audit in `ab/` supersedes
  them as evidence).
The speed tables below are generated from these files by `python3 tests/doc_tables.py` (standard library only).

Routing kinds: `rand` = independent random top-8 per row; `corr40` = DFlash-like correlated routing
(`tests/harness.py:correlated_ids`: rows grouped into sequences of 5/6/8 rows, each sequence drawing its rows'
experts from a 40-expert pool of its own; distinct experts per row 4.1 at T = 8, 3.0 at T = 64).

## Result

Commits on `opt` (in order): `29dd39c` K1, `92723fa` K2, `308eb89` K3, `871ff49` K6, `b47abe8` K3b, `7e6793d` C2,
`27a5188` K2 log/switch test, `cd4fd7c` docs, then the review fixes below. The shipped configuration is kernel
variant 0 = grid order ORD 1 (K1) + L2 evict_first weight loads (K3) + split-count table 6 (K6 + C2) + L2 discard of
dead intermediates (K3b), plus the apply-level routing kernel (K2, `TF_EXL3_APPLY`, default on). Variant 1 is the
master kernel set, kept for A/B.

### Master -> final, paired (`docs/logs/ab/full_master_vs_final.log`)

`tests/bench_variant_ab.py 1:0 0:1 --mode full --repeat 2`: A = the master TF (master kernels = variant 1, K2 off:
production's routing prelude + the TF exl3_moe path), B = the final TF (variant 0, K2 on), both as CUDA graphs of 12
production `apply_exl3_fused_moe` calls, in one process, 9 alternating rounds, two repeats. Before timing, a
one-route-per-token call of each is compared (E.1: the split-count tables differ); after timing, every output of both
timed graphs is compared with production exl3_moe (E.1).

| routing | T | distinct | A = master TF (µs) | B = final TF (µs) | B/A run 1 | B/A run 2 | spread A / B (run 1) | check |
|---|---|---|---|---|---|---|---|
| rand | 1 | 8.0 | 263.9 | 218.5 | 0.829 | 0.830 | 4.3% / 5.3% | E.1(split table differs) True, E.1 worst 8.0e-04 |
| rand | 2 | 15.6 | 474.8 | 415.2 | 0.878 | 0.880 | 4.3% / 7.8% | E.1(split table differs) True, E.1 worst 7.6e-04 |
| rand | 4 | 30.2 | 859.6 | 772.2 | 0.904 | 0.911 | 4.9% / 4.1% | E.1(split table differs) True, E.1 worst 7.6e-04 |
| rand | 5 | 38.2 | 1062.0 | 971.6 | 0.919 | 0.921 | 3.9% / 7.2% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| rand | 8 | 58.5 | 1630.9 | 1507.1 | 0.922 | 0.924 | 2.8% / 3.0% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| rand | 12 | 81.8 | 2273.5 | 2116.2 | 0.933 | 0.931 | 2.2% / 3.4% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| rand | 16 | 105.2 | 2882.9 | 2662.8 | 0.924 | 0.928 | 3.7% / 2.1% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| rand | 24 | 143.8 | 3946.4 | 3632.9 | 0.923 | 0.931 | 0.8% / 1.7% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| rand | 32 | 169.0 | 4633.7 | 4345.9 | 0.936 | 0.941 | 1.0% / 1.0% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| rand | 48 | 213.7 | 5892.6 | 5547.0 | 0.941 | 0.941 | 0.8% / 0.9% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| rand | 64 | 241.2 | 6719.8 | 6325.3 | 0.940 | 0.944 | 1.3% / 1.9% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| rand | 96 | 266.7 | 7570.9 | 7026.4 | 0.930 | 0.929 | 0.5% / 1.2% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| rand | 128 | 280.0 | 8131.5 | 7499.2 | 0.922 | 0.920 | 2.5% / 1.6% | E.1(split table differs) True, E.1 worst 7.3e-04 |
| corr40 | 1 | 8.0 | 281.8 | 235.9 | 0.837 | 0.834 | 11.0% / 19.5% | E.1(split table differs) True, E.1 worst 7.7e-04 |
| corr40 | 2 | 14.5 | 435.0 | 380.5 | 0.871 | 0.879 | 6.6% / 6.2% | E.1(split table differs) True, E.1 worst 7.8e-04 |
| corr40 | 4 | 24.2 | 690.0 | 620.6 | 0.900 | 0.900 | 4.7% / 3.0% | E.1(split table differs) True, E.1 worst 8.0e-04 |
| corr40 | 5 | 26.8 | 765.3 | 688.8 | 0.901 | 0.906 | 4.7% / 0.8% | E.1(split table differs) True, E.1 worst 7.7e-04 |
| corr40 | 8 | 33.5 | 956.9 | 868.2 | 0.908 | 0.909 | 2.6% / 3.7% | E.1(split table differs) True, E.1 worst 7.6e-04 |
| corr40 | 12 | 57.3 | 1591.0 | 1461.2 | 0.919 | 0.915 | 2.0% / 4.4% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| corr40 | 16 | 68.1 | 1891.4 | 1742.2 | 0.922 | 0.925 | 2.9% / 3.8% | E.1(split table differs) True, E.1 worst 7.5e-04 |
| corr40 | 24 | 101.6 | 2834.3 | 2594.9 | 0.918 | 0.919 | 1.7% / 3.2% | E.1(split table differs) True, E.1 worst 7.3e-04 |
| corr40 | 32 | 126.4 | 3568.2 | 3333.8 | 0.936 | 0.936 | 4.6% / 7.9% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| corr40 | 48 | 165.0 | 4664.0 | 4393.5 | 0.939 | 0.933 | 2.8% / 2.2% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| corr40 | 64 | 194.6 | 5527.5 | 5161.5 | 0.941 | 0.933 | 2.3% / 2.7% | E.1(split table differs) True, E.1 worst 7.4e-04 |
| corr40 | 96 | 230.4 | 6619.4 | 6119.2 | 0.925 | 0.926 | 0.7% / 1.1% | E.1(split table differs) True, E.1 worst 7.3e-04 |
| corr40 | 128 | 257.2 | 7552.2 | 6917.8 | 0.918 | 0.917 | 0.8% / 1.2% | E.1(split table differs) True, E.1 worst 7.3e-04 |
|B/A - 1|: median 7.82%, 90th percentile 12.21%, 95th 16.31%, max 17.07%; configurations whose two runs are both below 1 - 0.8%: 26 of 26, both above 1 + 0.8%: 0

```
all B/A: min 0.829 median 0.922 max 0.944 over 52
corr40 T=5..64: min 0.901 median 0.921 max 0.941 over 16
rand T=5..64: min 0.919 median 0.931 max 0.944 over 16
rand+corr40 T=1..4: min 0.829 median 0.879 max 0.911 over 12
rand+corr40 T=96..128: min 0.917 median 0.924 max 0.930 over 8
```
In the production range (corr40, T = 5..64) the final TF takes **0.90-0.94x** the master TF's time (rand:
0.92-0.94x); at T = 1..4 0.83-0.91x (K2's saving of the routing prelude is nearly fixed per call, so it weighs most
there); at T = 96..128 0.92-0.93x. The two repeats agree within 0.009 at every configuration.

### E.5 vs production exl3_moe (`docs/logs/run_all/bench_ab_decode.log`, final run_all)

`tests/bench_ab_decode.py`, 9 alternating rounds, median us per call; floor = distinct experts x 6 MiB / 250 GB/s;
"apply graph" = production `apply_exl3_fused_moe` captured and replayed (what production runs).

| routing | T | distinct | floor (µs) | apply graph: XL / TF (µs) | × | bare graph: XL / TF (µs) | × | TF GB/s (% of 250) |
|---|---|---|---|---|---|---|---|---|
| rand | 1 | 8.0 | 201.3 | 392.1 / 225.3 | 1.74 | 357.4 / 223.2 | 1.60 | 225.5 (90.2%) |
| rand | 2 | 15.6 | 392.2 | 586.4 / 415.0 | 1.41 | 572.7 / 420.5 | 1.36 | 233.2 (93.3%) |
| rand | 4 | 30.2 | 761.3 | 1052.2 / 819.3 | 1.28 | 1014.7 / 814.9 | 1.25 | 233.6 (93.4%) |
| rand | 5 | 38.2 | 962.6 | 1308.4 / 1004.8 | 1.30 | 1269.0 / 1014.2 | 1.25 | 237.3 (94.9%) |
| rand | 8 | 58.5 | 1472.2 | 1912.5 / 1587.2 | 1.20 | 1861.8 / 1549.8 | 1.20 | 237.5 (95.0%) |
| rand | 12 | 81.8 | 2057.3 | 2621.4 / 2193.1 | 1.20 | 2565.0 / 2173.5 | 1.18 | 236.6 (94.7%) |
| rand | 16 | 105.2 | 2648.7 | 3289.4 / 2716.8 | 1.21 | 3253.1 / 2711.8 | 1.20 | 244.2 (97.7%) |
| rand | 24 | 143.8 | 3617.6 | 4503.4 / 3708.7 | 1.21 | 4416.4 / 3713.0 | 1.19 | 243.6 (97.4%) |
| rand | 32 | 169.0 | 4253.0 | 5302.1 / 4547.0 | 1.17 | 5189.5 / 4445.2 | 1.17 | 239.2 (95.7%) |
| rand | 48 | 213.7 | 5377.1 | 6612.6 / 5595.8 | 1.18 | 6576.8 / 5570.1 | 1.18 | 241.3 (96.5%) |
| rand | 64 | 241.2 | 6069.2 | 7474.3 / 6349.2 | 1.18 | 7373.5 / 6340.9 | 1.16 | 239.3 (95.7%) |
| rand | 96 | 266.7 | 6710.9 | 8239.1 / 7119.0 | 1.16 | 8162.1 / 7104.0 | 1.15 | 236.2 (94.5%) |
| rand | 128 | 280.0 | 7046.4 | 8726.2 / 7592.5 | 1.15 | 8649.6 / 7604.4 | 1.14 | 231.7 (92.7%) |
| corr40 | 5 | 26.8 | 673.2 | 940.9 / 706.0 | 1.33 | 901.0 / 704.2 | 1.28 | 239.0 (95.6%) |
| corr40 | 8 | 33.5 | 843.1 | 1157.9 / 883.5 | 1.31 | 1124.7 / 897.9 | 1.25 | 234.7 (93.9%) |
| corr40 | 12 | 57.3 | 1442.8 | 1877.4 / 1499.2 | 1.25 | 1846.1 / 1481.5 | 1.25 | 243.5 (97.4%) |
| corr40 | 16 | 68.1 | 1713.4 | 2192.2 / 1773.3 | 1.24 | 2157.5 / 1762.3 | 1.22 | 243.1 (97.2%) |
| corr40 | 24 | 101.6 | 2556.4 | 3196.6 / 2630.9 | 1.22 | 3129.6 / 2628.7 | 1.19 | 243.1 (97.3%) |
| corr40 | 32 | 126.4 | 3181.4 | 3985.1 / 3344.2 | 1.19 | 3929.6 / 3332.2 | 1.18 | 238.7 (95.5%) |
| corr40 | 48 | 165.0 | 4152.4 | 5156.9 / 4351.0 | 1.19 | 5112.5 / 4367.2 | 1.17 | 237.7 (95.1%) |
| corr40 | 64 | 194.6 | 4896.8 | 6054.5 / 5157.9 | 1.17 | 5977.4 / 5169.3 | 1.16 | 236.8 (94.7%) |

corr40 T = 5..64: apply graph 1.17-1.33x faster than production exl3_moe, bare TF call at 93.9-97.4% of the
250 GB/s ceiling. Absolute times move by several percent between runs on the shared nodeC (e.g. the final TF's
apply graph at rand T = 8: 1587.2 us here, 1507.1 us in `ab/full_master_vs_final.log`; round spreads 2-17% are
printed per configuration in the log); the paired ratios above are the stable quantity.

Stop condition (headroom to the 250 GB/s floor < 3% at T = 1, 8, 64, random routing, bare graph), this run:
T = 1 +10.9%, T = 8 +5.3%, T = 64 +4.5% (the final TF's apply graph in `ab/full_master_vs_final.log`: 218.5 /
1507.1 / 6325.3 us = +8.5% / +2.4% / +4.2%): not met. The plan is exhausted (every
item built and measured, or gated out by its own probe), which is the other stop condition. See "Open problems".

Per-stage breakdown (graph replay of each stage alone over the 12 sets, us; GB/s over the distinct experts'
matrices, % of 250), same run:
```
  rand   T=  1 total    223.2 | route_prep 2.4 rot_in 2.1 grouped_gu 150.7 gateup_epi 4.0 grouped_down 79.5 down_epi 2.9 | sum 241.5 | gu 222.6 GB/s (89.0%) down 211.1 GB/s (84.4%)
  rand   T=  8 total   1549.8 | route_prep 2.5 rot_in 5.7 grouped_gu 1027.4 gateup_epi 4.3 grouped_down 502.1 down_epi 5.4 | sum 1547.2 | gu 238.8 GB/s (95.5%) down 244.3 GB/s (97.7%)
  rand   T= 64 total   6340.9 | route_prep 2.6 rot_in 33.7 grouped_gu 4224.1 gateup_epi 29.9 grouped_down 2070.2 down_epi 24.6 | sum 6385.1 | gu 239.5 GB/s (95.8%) down 244.3 GB/s (97.7%)
  rand   T=128 total   7604.4 | route_prep 3.3 rot_in 101.9 grouped_gu 5005.9 gateup_epi 20.4 grouped_down 2484.5 down_epi 53.5 | sum 7669.4 | gu 234.6 GB/s (93.8%) down 236.3 GB/s (94.5%)
  corr40 T=  5 total    704.2 | route_prep 2.5 rot_in 4.4 grouped_gu 473.2 gateup_epi 3.8 grouped_down 228.9 down_epi 4.4 | sum 717.3 | gu 237.1 GB/s (94.8%) down 245.1 GB/s (98.0%)
  corr40 T=  8 total    897.9 | route_prep 2.6 rot_in 5.6 grouped_gu 586.7 gateup_epi 4.3 grouped_down 285.1 down_epi 5.3 | sum 889.5 | gu 239.5 GB/s (95.8%) down 246.4 GB/s (98.6%)
  corr40 T= 64 total   5169.3 | route_prep 2.6 rot_in 32.8 grouped_gu 3443.1 gateup_epi 29.6 grouped_down 1681.3 down_epi 24.3 | sum 5213.6 | gu 237.0 GB/s (94.8%) down 242.7 GB/s (97.1%)
```
Stage-alone figures above 100% come from the stage graphs re-reading experts shared by consecutive routings of
the same layer (partial L2 reuse).

### Re-audit of every shipped item (`docs/logs/ab/`, paired, 9 rounds x 2 repeats, full T set)

Each item against its predecessor with the final code (B/A < 1 = the item is faster; "significant" = both repeats
below 0.992, which no configuration of the null run reached):

| item | log | A -> B | significant gains (of 26 configurations; C2: of 28) | production range corr40 T = 5..64, run 1 / run 2 | significant losses |
|---|---|---|---|---|---|
| K1 grid order | `k1_ord_1_vs_2.log` | variant 1 -> 2 | 14 | T5 1.003/1.005, T8 1.002/0.999, T16 0.990/0.997, T32 0.990/0.986, T64 0.985/0.983 | 0 |
| K3 evict_first | `k3_evictfirst_2_vs_6.log` | 2 -> 6 | 25 | T5 0.959/0.962, T8 0.953/0.960, T16 0.958/0.975, T32 0.971/0.970, T64 0.975/0.976 | 0 |
| K6 split table 1 | `k6_table1_6_vs_9.log` | 6 -> 9 | 7 | T5 1.024/0.988, T8 0.963/0.983, T12 0.973/0.987, T16 0.983/0.995, T24 1.002/0.998 (T5..24 differ only in down SK 2 vs 1; T >= 32 same table) | 0 |
| C2 split table 6 | `c2_table6_9_vs_12.log` | 9 -> 12 | 4 (rand and corr40 T = 96, 128) | identical table up to T = 64 (0.981-1.012, none significant: noise); T80 0.993/0.989, T96 0.985/0.981, T128 0.980/0.978 | 0 |
| K3b L2 discard | `k3b_discard_12_vs_0.log` | 12 -> 0 | 9 (T >= 24) | T5 1.000/1.001, T8 1.001/1.005, T32 0.989/0.990, T48 0.987/0.979, T64 0.984/0.990 | 0 |
| K2 apply path | `k2_apply_0_vs_1.log` | K2 off -> on | 14 | T5 1.006/0.939, T8 0.982/0.972, T12 0.979/0.986, T24 0.971/0.977, T64 0.990/0.981 | 0 |
| null | `null_0_vs_0.log` | 0 -> 0 | 0 | T5 1.000/0.999, T8 0.973/1.011, T64 0.997/0.999 | 0 |

No item is significantly slower anywhere. K3 carries the largest share; K1 re-measured smaller than in its original
runs (0.95-1.00 then, 0.978-1.011 now; with correlated routing it helps only from T = 16), K6 helps at T <= 12
but with noisy repeats, C2 only above T = 64 (as intended), K3b from T = 24. The product of the six items' mean
ratios is 0.918-0.949 over corr40 T = 5..64 against 0.903-0.937 measured directly: the items account for most of
the gain, with up to 4 points (corr40 T = 5) not explained by the product (interaction or noise); the direct
master -> final measurement is the one quoted.

## Measurement infrastructure (Step 0, E0)

- `tests/harness.py`: `seq_lengths`, `correlated_ids`, `routing_ids("rand" | "corr<pool>")`.
- `tests/bench_ab_decode.py` (E.5): T in {1, 2, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128} random and
  {5, 8, 12, 16, 24, 32, 48, 64} correlated; distinct experts and empty segment blocks (S_cap - nseg) per
  configuration; GB/s, % of the 250 GB/s ceiling and the floor per call; gate/up and down GB/s and % per stage;
  per-round-median speedups; 9 alternating rounds (order flipped every round); `TF_EXL3_BENCH_VARIANT` to time
  another kernel variant.
- `ext.set_variant(id)` (kernels/exl3.cu `kVariants`): a process-wide kernel-variant id read on the host at launch
  (a captured graph keeps its variant); 0 = shipped, 1 = master. `tests/bench_variant_ab.py A B [--mode apply]`:
  the paired A/B, with a bitwise check of both variants' outputs on one-route-per-token routing (E.1 when the
  split-count tables differ) and an E.1 read-back of every timed graph against production exl3_moe.
- `tests/test_variants.py` (in run_all): 1000 fresh-routing CUDA-graph replays of the hooked production apply for
  every variant, bitwise equal within a split-count table, E.1 across tables.

E0 acceptance: with the master kernels the extended bench reproduces the 12:52 random-routing numbers within the
round spread at every T except T = 1, where XL was slow in the same measurement (GPU contention at the start of the
run): TF bare graph T = 2 431.0 (12:52: 429.2), T = 8 1582.7 (1566.6), T = 64 6714.3 (6731.1), T = 128 8089.6
(8111.7); T = 1 246.3 (226.3) with XL 410.4 (352.8).

## Probes

### P1 — access pattern vs bytes in flight (`tests/bw_pattern.py`, logs `docs/logs/history/p1_*.log`)

A load_inline replica of the grouped GEMV with the exact grid, pointer tables and route tables; variants are
template instances timed against the shipped `ext.grouped` in the same rounds (7 rounds, median; "ratio" = variant /
shipped kernel).

Finding 1 (calibration holds): load-only D = 1 vs the shipped kernel: gu T=1 0.988, T=8 1.001, T=64 0.979,
corr T=8 1.007, corr T=32 0.982.

Finding 2 (load-only patterns can reach 240-246 GB/s): e.g.
```
--- gu rand T=64 (load-only)   real ext.grouped  4407.9 us 229.7 GB/s | v6 L PF2 4211.7 us 240.4 GB/s ratio 0.959 | v12 L D4 POL 4185.3 us 241.9 GB/s ratio 0.947
--- down rand T=64 (load-only) real ext.grouped  2190.8 us 231.1 GB/s | v6 L PF2 2056.4 us 246.2 GB/s ratio 0.952
```
Finding 3 (with compute, nothing beat the shipped kernel; L2 prefetch hurts): gu rand T=64: v20 (compute
calibration, full unroll, asm-volatile loads) 1.033, v21 D2 1.009, v23 D4 1.009, v24 PF4 1.037, v25 PF99 1.068,
v33 LEAN D4 1.013, v36 persistent 1.097, v37 LEAN D2 evict_first 1.015. ptxas re-schedules the register ring (the
D = 8 load-only variant keeps about 3 k tiles in flight, and REG drops from 72 at D = 1 to 63-64 at D = 2..8), so
"deep pipelining" in the source does not become deep pipelining in SASS.

Finding 4 (grid order is the lever): the same kernels with blocks in n-block-fastest order (order 1):
```
--- gu rand T=8:   real 1041.5 us | v1 L order0 0.990  order1 0.967  order2 0.972 | v20 C order0 1.050  order1 0.976  order2 0.986
--- gu rand T=64:  real 4395.2 us | v1 L order0 0.988  order1 0.950  order2 0.965 | v20 C order0 1.044  order1 0.976  order2 0.978
--- down rand T=64: real 2155.6 us | v1 L order0 0.993  order1 0.962  order2 0.967 | v20 C order0 1.041  order1 1.002  order2 1.005
```
Finding 5 (occupancy / carveout): pinning 4..8 blocks/SM and a 50% carveout moved the compute variant v31 by
0.977..1.133 (no gain); load-only v4 at 4 blocks/SM reached 246 GB/s for down at T = 8 (0.954-0.966), but that
does not carry over to the compute kernels.

Decision (plan: "Pick ... >= 242 GB/s with compute" and "Grid order enters K1 if >= 1%"): no pipelining variant
qualified; grid order did (3-7%). K1 = grid order.

### P2 — Z / intermediate round trip (`tests/probe_roundtrip.py`)

Before K1 (`docs/logs/history/p2_roundtrip.log`): gue + de excess (in-pipeline cost minus stage-alone cost) = +0.68% of
the call at rand T=8, +0.97% T=16, +0.81% T=32, +0.95% T=64 (61.3 us), +0.37% T=128; corr +1.20% T=16, +1.00%
T=32, +0.96% T=64 -> a marginal Go for the L2 and fusion tracks. C2 pair level [gu(SK), gue(SK)]: SK 2 / 1 vs 4 =
0.9996 / 1.0007 at rand T=64, 0.9845 / 0.9835 at corr T=64, 0.9789 / 0.9749 at T=128.

After K1 + K3 + K6 (`docs/logs/history/p2_roundtrip_after_k6.log`): gue + de excess -0.66% (T=8), -0.39% (T=16), +0.84%
(T=32), -0.04% (T=64), -0.60% (T=128), corr +0.02% / +0.07% / +0.18% -> the Z round trip no longer costs anything
measurable; the fixup-fusion track (F1/F2/F3/C) is below its 1% gate and was not built. What remained: rot_in ->
gate/up excess +29.6 us (T=32), +33.3 us (T=64), corr +42.0 us (T=64) = the writeback of the 8 MiB of dirty xg/xu
-> K3b.

### P3 — production's routing prelude (`tests/probe_prelude.py`, `docs/logs/history/p3_prelude.log`)

Production apply_exl3_fused_moe with a no-op exl3_moe, captured and replayed:
```
rand   T=  1: prelude-only graph   23.6 us/call | full apply (TF)   253.8 | bare TF   231.0 | apply - bare   25.4
corr40 T=  5: prelude-only graph   23.7 us/call | full apply (TF)   755.6 | bare TF   725.0 | apply - bare   32.4
corr40 T=  8: prelude-only graph   23.6 us/call | full apply (TF)   898.9 | bare TF   867.8 | apply - bare   31.4
corr40 T= 16: prelude-only graph   24.7 us/call | full apply (TF)  1938.9 | bare TF  1907.0 | apply - bare   30.4
corr40 T= 32: prelude-only graph   45.7 us/call | full apply (TF)  3324.5 | bare TF  3243.4 | apply - bare   60.6
rand   T= 64: prelude-only graph   51.0 us/call | full apply (TF)  6434.3 | bare TF  6363.9 | apply - bare   65.4
```
Go for K2 (>= 10 us).

## Attempts (in execution order)

### K1 — grouped GEMV grid order: n block fastest (KEPT, `29dd39c`)
What: `grouped_kernel` template knob ORD; ORD 1 = grid (N/64, S_cap, mats*SK): the 16 (gate/up) / 64 (down) blocks
that read the same k rows of one matrix are adjacent in launch order, so co-resident blocks stream whole 8-32 KiB
trellis rows instead of 512-byte pieces of many experts' matrices. Same arithmetic in the same order (Z
bit-identical, E-U2c). Falls back to ORD 0 when S_cap > 65535 (gridDim.y); until the review fix below that fallback
raised for every evict_first / LEAN variant, incl. the shipped one (now: ORD 0 + evict_first instance, pre-launch check,
E-U2c / E-U7).
Why: P1 finding 4.
Before/after (paired, master = 1 vs ORD 1, bare graph call, 9 rounds):
```
run 1  rand   T1:0.988  T2:0.992  T4:0.980  T5:0.985  T8:0.970  T12:0.972  T16:0.964  T24:0.953  T32:0.956  T48:0.957  T64:0.956  T96:0.963  T128:0.963
       corr40 T1:0.992  T2:0.997  T4:0.981  T5:0.984  T8:0.979  T12:0.970  T16:0.969  T24:0.965  T32:0.959  T48:0.968  T64:0.960  T96:0.962  T128:0.971
run 2  rand   T1:0.992  T2:0.994  T4:0.965  T5:0.975  T8:0.976  T12:0.967  T16:0.966  T24:0.962  T32:0.955  T48:0.961  T64:0.962  T96:0.957  T128:0.957
       corr40 T1:0.989  T2:0.989  T4:0.985  T5:0.980  T8:0.975  T12:0.981  T16:0.973  T24:0.961  T32:0.961  T48:0.961  T64:0.962  T96:0.963  T128:0.960
```
Grouped stages (run 1): gate/up 0.967 (T=8), 0.949 (T=64); down 0.981 / 0.973 -> grouped total -3.1% (T=8), -4.5%
(T=64). Accept: the plan's K1 thresholds were written for pipelining (grouped >= 3% at T 8/64: met; bare call
>= 2.5% at corr T 5..64: met at 16/32/64, missed at T=5 (1.6%) and T=8 (2.1%)); it passes the plan's own
grid-order criterion (Step 5.1: within +-1% everywhere, >= 0.3% faster with correlated routing) by a wide margin,
does not change a single arithmetic operation, and was kept on that basis. memcheck (test_units, test_stage_vs_xl)
0 errors, racecheck (test_units) 0 hazards, REG 77, LOCAL 0.

Sub-variants measured against ORD 1 (`docs/logs/history/k1_variants.log`, 7 rounds, B/A vs ORD 1):
- ORD 2 (expert-major, empty segments last): rand T1:1.043 T8:1.001 T64:0.998, corr 0.997/0.997/1.000 -> no gain, not shipped.
- PIPE (double-buffered k loop): rand 1.003/1.008/1.008, corr 1.001/1.005/1.008 -> REJECTED.
- LEAN (one 4 KiB reduction buffer, 6-8 blocks/SM): rand 1.014/1.037/1.028, corr 1.048/1.034/1.034 -> REJECTED
  (ptxas capped it at 64 registers with an 8-byte stack; the serial reduction adds 3 barriers).
- PIPE + LEAN: rand 1.023/1.020/1.026, corr 1.021/1.012/1.015 -> REJECTED.
- evict_first weights: rand 0.946/0.977/0.984, corr 0.987/0.959/0.993 -> became K3.

Re-audit (`docs/logs/ab/k1_ord_1_vs_2.log`, variant 1 -> 2, final code): significant at 14 of 26 configurations,
smaller than in the runs above: rand T >= 8 0.978-0.994, corr40 T >= 16 0.979-0.997, corr40 T <= 8 0.995-1.011 (no
gain there); no significant loss.

### K2 — production's decode apply from the router ids (KEPT, `92723fa`, `27a5188`)
What: `route_ids_kernel` (one 1024-thread block) does production's prelude (map_topk_to_local with its exact rules
incl. EP map and an empty map, argsort by local expert, token_sorted / fp16 weight_sorted gathers, expert_count)
and route_prep in one launch; rot_in reads the bf16 hidden state (fp16 RN, as `.half()`); `moe_forward_ids`.
`integrate.install` patches `apply_exl3_fused_moe` (active only while exl3_moe is our dispatcher; everything
else -> production's apply with the args received). `tf_exl3_moe.apply_fused` serves exactly the calls plan()
would serve, sets production's decode side effects, allocates `out` with torch.zeros like production (1 allocation
per call vs production's 17), exception policy as launch(). `TF_EXL3_APPLY=0` turns K2 off. A load-time K2
self-test per layer (bitwise vs the exl3_moe path on one route per token, E.1 on topk) disables only K2 on a
mismatch; its result is in the per-layer INFO line.
Why: P3 (prelude 23.6-51 us per call).
Before/after (paired, production apply graphs, K2 off = 0 vs on = 1, 9 rounds):
```
run 1  rand   T1:0.874  T2:0.941  T4:0.962  T5:0.961  T8:0.980  T12:0.993  T16:0.988  T24:0.988  T32:0.991  T48:0.990  T64:1.000  T96:0.988  T128:0.995
       corr40 T1:0.891  T2:0.918  T4:0.952  T5:0.968  T8:0.967  T12:0.983  T16:0.988  T24:0.980  T32:0.988  T48:0.988  T64:0.985  T96:0.996  T128:0.993
run 2  rand   T1:0.889  T2:0.928  T4:0.968  T5:0.978  T8:0.983  T12:0.989  T16:0.996  T24:0.987  T32:0.992  T48:0.991  T64:0.987  T96:0.997  T128:0.992
       corr40 T1:0.868  T2:0.938  T4:0.947  T5:0.955  T8:0.971  T12:0.983  T16:0.994  T24:0.987  T32:0.984  T48:0.990  T64:0.989  T96:0.989  T128:0.995
```
(run 1 in us: T=1 261.0 -> 227.8, corr T=5 760.3 -> 731.8, corr T=8 949.2 -> 913.9, corr T=64 5359.3 -> 5302.1.)
Accept (plan): >= 15 us at T {1, 5, 8} (33 / 28.5 / 35 us), >= 1.5% at corr T=8 (3.3%), T > 16 <= 1.01 (max 1.000):
met. Tests (`tests/test_apply_fused.py`, in run_all): route_ids == production prelude + route_prep 400/400 (the
first run caught an empty-expert-map bug: 390/400); bf16 rot_in bit-identical to rot_in(x.half()); E.1 vs
production apply 143/143 (sentinel, EP map, duplicates, hot expert = R, over cap, all-sentinel rows, corr); bitwise
vs the exl3_moe path 6/6; 9 delegation cases incl. production's own RuntimeError for a CPU expert map; CUDA graph
100/100; no host sync; uninstall restores all three patches and keeps a later third-party patch; TF_EXL3_APPLY.
memcheck 0 errors, racecheck / synccheck on route_ids 0 hazards.

Re-audit (`docs/logs/ab/k2_apply_0_vs_1.log`, final code): significant (both repeats < 0.992) at 14 of 26 configurations,
T = 1 0.889-0.908, corr40 T = 8 0.982/0.972, T = 24 0.971/0.977; no loss.

### K3 — L2 evict_first on the weight loads (KEPT, `308eb89`)
What: `ld.global.nc.L1::no_allocate.L2::cache_hint` with `createpolicy.fractional.L2::evict_first` for the trellis
words (streamed once per call), so the per-call intermediates stay in L2. Same arithmetic.
Before/after (paired, bare call, 9 rounds; run 1 = ORD1+evict_first vs ORD1 shipped, run 2 = previous shipped vs new):
```
run 1  rand   T1:0.969  T2:0.967  T4:0.969  T5:0.978  T8:0.965  T12:0.975  T16:0.973  T24:0.984  T32:0.989  T48:0.989  T64:0.987  T96:0.989  T128:0.987
       corr40 T1:0.971  T2:0.972  T4:0.969  T5:0.963  T8:0.967  T12:0.963  T16:0.963  T24:0.973  T32:0.981  T48:0.980  T64:0.985  T96:0.977  T128:0.984
run 2  rand   T1:0.967  T2:1.000  T4:0.972  T5:0.966  T8:0.975  T12:0.976  T16:0.980  T24:0.980  T32:0.985  T48:0.991  T64:0.999  T96:0.983  T128:0.988
       corr40 T1:0.970  T2:0.967  T4:0.969  T5:0.964  T8:0.956  T12:0.973  T16:0.971  T24:0.974  T32:0.992  T48:0.988  T64:0.982  T96:0.974  T128:0.980
```
Grouped stages alone 0.98-1.00 (the gain is in the intermediates). Accept (>= 1% at T 32/64 with random or
correlated routing, no regression): met. 1000-replay bit-identity added (`tests/test_variants.py`).

Re-audit (`docs/logs/ab/k3_evictfirst_2_vs_6.log`, variant 2 -> 6): significant at 25 of 26 configurations, 0.953-0.996.

### K6 — split-count tables at small T (KEPT, `871ff49`; the plan's side task, extended)
What: the K splits of the two GEMVs become a static table in P (CUDA-graph safe), selected by the variant; the
scratch bound is the maximum over every table; a wanted split that does not divide another shape falls back.
Why: re-sweep (`tests/sweep_grouped_cfg.py`, 9 rounds, GEMV alone, us) with the new kernel:
```
T= 1 gate/up 4,4,4: 146.3 | 4,4,8: 139.7 | 4,4,16: 141.4        down 4,4,2: 68.3 | 4,4,1: 74.9 | 4,4,4: 67.8 | 4,4,8: 67.6
T= 2 gate/up 4,4,4: 271.0 | 4,4,8: 265.5 | 4,4,16: 270.1        down 4,4,2: 133.6 | 4,4,1: 138.9 | 4,4,4: 133.6 | 4,4,8: 139.1
T= 4 gate/up 4,4,4: 524.2 | 4,4,8: 517.4 | 4,4,16: 533.7        down 4,4,2: 258.2 | 4,4,1: 273.2 | 4,4,4: 250.5 | 4,4,8: 257.9
T= 5 gate/up 4,4,4: 653.1 | 4,4,8: 652.9                          down 4,4,1: 324.5 | 4,4,2: 314.0 | 4,4,4: 321.6
T= 8 gate/up 4,4,4: 975.6 | 4,4,8: 993.3                          down 4,4,1: 504.1 | 4,4,2: 488.0 | 4,4,4: 490.6
T=12 gate/up 4,4,4: 1396.8 | 4,4,8: 1439.8                        down 4,4,1: 713.0 | 4,4,2: 704.6 | 4,4,4: 712.7
T=16 gate/up 4,4,4: 1787.6 | 4,4,8: 1811.0                        down 4,4,1: 874.4 | 4,4,2: 854.2 | 4,4,4: 890.2
T=24 gate/up 4,4,4: 2369.7 | 4,4,8: 2448.7                        down 4,4,1: 1191.6 | 4,4,2: 1158.1 | 4,4,4: 1222.3
T=32 gate/up 4,4,4: 2858.9 | 4,4,8: 2935.2                        down 4,4,1: 1417.5 | 4,4,2: 1416.8 | 4,4,4: 1472.7
```
Tables (paired A/B vs table 0 = master's, 9 rounds):
```
table 1 (gu 8 at P<=32; down 2 at P<=192)            rand T1:0.986 T2:0.976 T4:0.980 T5:0.981 T8:0.989 T12:0.990 T16:0.990 T24:0.996 T32:0.999
                                                     corr T1:0.985 T2:0.981 T4:0.979 T5:0.987 T8:0.981 T12:0.987 T16:0.990 T24:0.997 T32:1.001
table 2 (as 1, down 4 at P<=32)                      rand T1:0.984 T2:0.975 T4:0.979 T5:0.984 T8:0.987 T12:0.988 T16:0.981 T24:0.995 T32:1.002
                                                     corr T1:0.981 T2:0.984 T4:0.985 T5:0.986 T8:0.986 T12:0.987 T16:0.984 T24:0.991 T32:1.000
table 3 (as 1, gu 8 only at P<=8)                    rand T1:0.986 T2:0.977 T4:0.979 T5:0.984 T8:0.987 T12:0.985 T16:0.992 T24:0.994 T32:0.997
                                                     corr T1:0.987 T2:0.985 T4:0.978 T5:0.989 T8:0.993 T12:0.987 T16:0.981 T24:0.997 T32:0.997
table 3 / table 1, 21 rounds x 2 runs                rand T1:1.000 T2:1.005 T4:1.004 T5:0.998 | T1:1.001 T2:1.003 T4:1.003 T5:1.002
                                                     corr T1:1.001 T2:1.005 T4:1.003 T5:1.003 | T1:1.000 T2:1.004 T4:1.000 T5:1.001
```
Table 1 kept. Confirmation (previous shipped variant 6 vs new 0, full T set):
```
rand   T1:0.989  T2:0.970  T4:0.976  T5:0.978  T8:0.985  T12:0.991  T16:0.985  T24:0.991  T32:1.001  T48:1.005  T64:0.997  T96:1.004  T128:0.997
corr40 T1:0.986  T2:0.967  T4:0.973  T5:0.969  T8:0.980  T12:0.983  T16:0.985  T24:0.993  T32:0.999  T48:1.003  T64:1.002  T96:0.994  T128:0.999
```
(T >= 32 is the same table there: noise.) A different table changes the fp32 summation tree of the split partials
(not bit-identical to table 0, E.1 unchanged at worst 1.14e-03); E-U4 (numpy emulation) and E-U6 (bit-identity to
the exl3_moe binary on its own intermediates) were extended to the gate/up epilogue at SK = 8 and pass.

Re-audit (`docs/logs/ab/k6_table1_6_vs_9.log`, variant 6 -> 9): significant at 7 of 26 (rand T = 1, 4, 5, 8; corr40
T = 2, 8, 12), repeats noisy at T <= 5 (e.g. corr40 T = 5 1.024 / 0.988), no significant loss; T >= 32 same table.

### K5 — programmatic dependent launch (REJECTED; code reverted, `docs/patches/k5_pdl_rejected.diff`)
What: rot_in / grouped / epilogues launched with cudaLaunchKernelEx + programmatic stream serialization; every
kernel `griddepcontrol.wait` before any read of predecessor data, any global write and any exit; the route kernels
trigger at their top; the GEMVs read the route tables with ld.global.cg and issue their first k tile's weight loads
(+ L2 prefetch of the next 3) before the wait. Outputs bitwise equal over 1000 graph replays.
Numbers (paired vs shipped, 9 rounds; variants: 12 prologue + prefetch in every block, 13 wait first, 14 prologue
without prefetch, 15 prefetch only in the first 256 blocks, 16 prefetch in every block while P <= 64):
```
12 run 1 rand T1:0.989 T2:0.999 T4:0.997 T5:0.993 T8:0.999 T16:0.996 T32:0.990 T64:0.998 T128:0.997
         corr T1:0.984 T2:0.990 T4:0.992 T5:0.998 T8:0.993 T16:1.009 T32:1.004 T64:1.005 T128:1.002
12 run 2 rand T1:0.984 T2:0.986 T4:1.004 T5:0.987 T8:0.999 T16:0.999 T32:1.004 T64:1.008 T128:1.003
         corr T1:0.980 T2:0.995 T4:0.978 T5:0.992 T8:0.999 T16:1.001 T32:1.004 T64:1.004 T128:1.007
13       rand T1:1.005 T2:1.013 T4:1.006 T5:0.987 T8:1.000 T16:1.003 T32:1.003 T64:0.997 T128:0.995
         corr T1:1.004 T2:1.006 T4:1.007 T5:1.003 T8:1.011 T16:1.000 T32:1.009 T64:0.997 T128:1.002
14       rand T1:1.006 T2:1.003 T4:1.007 T5:1.007 T8:1.000 T16:0.998 T32:0.996 T64:1.006 T128:0.995
         corr T1:0.992 T2:1.004 T4:0.997 T5:1.002 T8:1.005 T16:1.004 T32:1.002 T64:1.001 T128:1.004
15       rand T1:0.994 T2:0.997 T4:0.997 T5:0.995 T8:0.999 T16:1.000 T32:0.997 T64:1.006 T128:0.998
         corr T1:0.993 T2:0.985 T4:0.992 T5:0.997 T8:0.999 T16:1.001 T32:1.005 T64:1.001 T128:0.998
16 run 1 rand T1:0.983 T2:0.991 T4:0.991 T5:0.996 T8:1.003 T12:1.002 T16:1.008 T24:1.002 T32:0.999 T48:0.997 T64:0.996 T96:1.002 T128:0.997
         corr T1:0.983 T2:0.966 T4:0.987 T5:0.996 T8:0.999 T12:1.001 T16:0.999 T24:1.004 T32:1.002 T48:1.004 T64:0.997 T96:0.995 T128:0.997
16 run 2 rand T1:0.983 T2:0.990 T4:0.991 T5:1.002 T8:0.997 T12:1.003 T16:0.996 T24:1.000 T32:1.000 T48:1.004 T64:0.991 T96:1.004 T128:1.000
         corr T1:0.981 T2:0.993 T4:0.992 T5:0.995 T8:1.009 T12:1.003 T16:0.998 T24:1.008 T32:0.994 T48:1.002 T64:1.000 T96:0.995 T128:0.987
```
Consistent only at T <= 4 (T = 1: -1.7..-2.0%, ~4 us); the plan's accept (>= 3 us at T {1, 5, 8}) fails at T = 5
and 8, and DFlash decode calls are T >= 5. Rejected; not worth programmatic edges and memory-ordering rules in
production graphs for a non-production T range.

### OPT-2 — 2/4/8 warps per block in rot_in / gateup_epilogue / down_epilogue (REJECTED; reverted)
Same per-warp code (bit-identical). vs shipped, 9 rounds:
```
4 warps rand T1:1.012 T5:0.997 T8:1.000 T16:1.004 T32:0.999 T64:0.993 T128:1.001 | corr T1:1.002 T5:0.998 T8:0.993 T16:0.988 T32:0.989 T64:1.005 T128:0.993
2 warps rand T1:1.001 T5:1.005 T8:0.998 T16:1.009 T32:0.989 T64:1.004 T128:1.004 | corr T1:1.002 T5:0.973 T8:0.996 T16:1.000 T32:1.003 T64:1.000 T128:0.998
8 warps rand T1:1.008 T5:0.996 T8:1.000 T16:0.990 T32:1.003 T64:1.001 T128:0.993 | corr T1:1.015 T5:0.995 T8:1.000 T16:0.997 T32:1.000 T64:1.005 T128:0.994
```
No signal: the small kernels are not bound by block scheduling (at T = 64 rot_in writes 8 MiB).

### K4 — split-K fixup fusion (F3, F2) (NOT BUILT: gate not met)
Gate: P2 gue + de excess >= 1% of the call at T = 32 or 64. After K3 it is -0.66..+0.84% (P2 above); the whole
epilogue time left to fuse is gue 4 us + de 5 us at T = 8 and 27 + 24 us at T = 64. F5 (fuse rot_in into gate/up)
needs F2 and was not built either.

### K3b — discard.global.L2 of dead intermediates (KEPT, `b47abe8`)
What: after its reads, gateup_epilogue drops its Z lines (row j, its 128 columns, every split plane of both
matrices) and its share of row j of xg / xu (read only by the finished gate/up GEMV); down_epilogue drops its Z
lines and its share of row j of xd. Each dropped line has exactly one reader in the call; the next writer is
stream-ordered after the dropping kernel; only the forward paths discard (never the stage entry points).
Why: P2 after K6 (the writeback of the dirty intermediates).
Before/after (paired, shipped without discard vs with, 9 rounds x 2, then 21 rounds x 2 at T = 32..64):
```
run 1  rand   T1:1.001  T2:1.002  T4:0.994  T5:0.997  T8:0.999  T12:0.998  T16:1.006  T24:0.992  T32:0.995  T48:0.984  T64:0.992  T96:0.994  T128:0.987
       corr40 T1:0.999  T2:1.001  T4:1.001  T5:1.001  T8:0.985  T12:0.999  T16:0.997  T24:0.996  T32:0.990  T48:0.985  T64:0.991  T96:0.995  T128:0.994
run 2  rand   T1:0.982  T2:1.002  T4:0.999  T5:1.004  T8:0.999  T12:0.997  T16:0.988  T24:0.992  T32:0.992  T48:0.986  T64:0.993  T96:0.990  T128:0.990
       corr40 T1:0.999  T2:0.999  T4:1.002  T5:1.001  T8:1.003  T12:0.992  T16:0.980  T24:0.995  T32:0.984  T48:0.983  T64:0.991  T96:0.986  T128:0.980
21 x 2 rand   T32:0.987  T48:0.988  T64:0.993  T32:0.989  T48:0.984  T64:0.994
       corr40 T32:0.988  T48:0.984  T64:0.988  T32:0.990  T48:0.983  T64:0.988
```
Accept (>= 1% at T 32/64 with random or correlated routing, no regression): met with correlated routing (T=32
1.0-1.6%, T=64 1.2%). E.1 incl. the NaN-poisoned scratch cases unchanged, 1000-replay bit-identity, memcheck clean.

Re-audit (`docs/logs/ab/k3b_discard_12_vs_0.log`, variant 12 -> 0): significant at 9 of 26, all at T >= 24 (0.979-0.991).

### C2 — fewer gate/up splits at large P (KEPT as table 6, `7e6793d`)
Allowed by the plan once F2 was not built and P2's pair-level gain at T >= 32 was >= 1% (corr T=64 1.55%, T=128
2.5%). vs shipped table 1, 9 rounds:
```
table 4 (gu 2 at P > 192)  rand T5:0.996 T8:1.002 T16:0.999 T32:0.999 T48:0.989 T64:1.002 T96:0.995 T128:0.990
                           corr T5:0.999 T8:1.003 T16:0.997 T32:1.002 T48:0.990 T64:0.989 T96:0.986 T128:0.984
table 5 (gu 2 at P > 32)   rand T5:1.010 T8:1.009 T16:1.007 T32:1.001 T48:1.003 T64:0.998 T96:1.000 T128:0.983   -> REJECTED
                           corr T5:1.004 T8:1.018 T16:1.009 T32:0.997 T48:0.996 T64:0.995 T96:0.998 T128:0.987
table 4, 15 rounds x 2     rand T32:1.005 T48:1.001 T64:0.994 T96:0.991 T128:0.989 | T32:1.003 T48:0.997 T64:1.005 T96:0.991 T128:0.985
                           corr T32:1.001 T48:1.002 T64:1.000 T96:0.985 T128:0.984 | T32:1.001 T48:0.999 T64:1.000 T96:0.984 T128:0.984
```
No gain at T <= 64 -> table 6 = table 1 + gate/up 2 splits only at P > 512 (identical to table 1 up to T = 64,
bitwise by construction). Table 6 vs 1 (15 rounds): rand T64:0.998 T80:1.000 T96:0.995 T128:0.989, corr
T64:0.997 T80:0.994 T96:0.996 T128:0.984 (T8, same table: 0.984 / 1.001 = noise). The gate/up epilogue at rand T = 128 drops from 153.8 us
(`docs/logs/history/bench_after_k3b.log`, table 1) to 24.9 us (`docs/logs/history/run_all_pre_docs.txt`) and 20.4 us
(`docs/logs/run_all/bench_ab_decode.log`): the 16 MiB of partials now fit the L2.

Re-audit (`docs/logs/ab/c2_table6_9_vs_12.log`, variant 9 -> 12, T incl. 80): significant at rand / corr40 T = 96 and
128 (0.978-0.990), T = 80 0.989-0.995, identical table up to T = 64 (0.981-1.012, none significant). The review found the SK = 2 branch
had no bitwise test; it now has (E-U2c / E-U4 / E-U6 at SK 2, and the forward test above P = 512 in test_variants).

### Tile width / warps at large T (REJECTED, sweep only)
`tests/sweep_grouped_cfg.py`, 7 rounds, GEMV alone: T=32 gate/up 4,4,4 2926.5 us | 8,4,4 3108.7 | 4,8,4 3061.2;
down 4,4,1 1467.6 | 8,4,1 1550.4 | 4,8,1 1505.9; T=64 gate/up 4167.5 | 4508.4 | 4439.8; down 2098.3 | 2232.8 |
2172.0 (the wider tiles also run the old grid order, so this is not a clean comparison).

### Not tried
- cp.async ring (P1 variant b): not built; neither register rings nor L2 prefetch helped once compute was in the
  loop (P1 finding 3), and the LEAN buffer that would have held it was itself slower.
- Persistent / dataflow kernels: P1 variant h (persistent) load-only 0.963-0.988 vs the shipped kernel, no better
  than the non-persistent D = 4 variant (0.958-0.983); with compute (v36, LEAN D = 2) 1.056-1.112. Not built.
- OPT-3 (one rot_in when gate/up share suh): depends on the real checkpoint's `_exl3_shared_w13_suh`; rot_in is
  0.4-0.7% of the call at T = 8..64.
- Wider trellis loads, TMA, cooperative grids, clusters, fp32 atomics into Z: rejected by the plan.

## Open problems

1. T = 1 (not a DFlash point) stays at 90-93% of the ceiling in the bench: the four small kernels take ~10 us of
   ~220 us, and the down GEMV at T = 1 (1024 blocks of 16 KiB) measured 227-246 GB/s depending on the run. PDL
   removes ~4 us there (rejected for T >= 5, see K5); `docs/patches/k5_pdl_rejected.diff` is the starting point if
   T <= 4 ever matters.
2. T = 48..64 stay at 95-97% (headroom at rand T = 64: 4.5% in the final run_all, 4.2% in the paired run): the GEMVs
   run at 237-245 GB/s. Load-only replicas reach
   241-246 GB/s with deeper prefetch (P1), but every compute-carrying pipelining variant was slower; a kernel that
   keeps >= 20 KiB/SM in flight without extra L2 requests or registers (e.g. cp.async into an L1-carveout-neutral
   ring) is the remaining idea.
3. All numbers are synthetic weights on one GB10. The real checkpoint, TP = 2 on nodeA/nodeB, the real top-k
   distribution and the real concurrency are untested (docs/STATUS.md). The per-layer INFO line now also reports
   the K2 self-test, so a production log shows whether K2 stayed on.
4. Noise of a single paired ratio on the shared nodeC (null run, `docs/logs/ab/null_0_vs_0.log`): |B/A - 1| 90th
   percentile 0.76%, max 2.67%; gains below about 1% need both repeats and neighbouring T to agree (K5 at T >= 5 and
   OPT-2 never did).

## Review fixes (2026-09-27, second review of branch `opt`)

Each finding was re-checked against the code before changing anything. Every code/test fix comes with a test that
fails on the unfixed code: for the kernel bug the unfixed kernel itself, for the coverage gaps a mutant that the
old test files (HEAD `cd4fd7c`) pass and the new ones fail, run on nodeC and then removed (`git diff` clean). The
mutants' exact code and every log: `docs/logs/review_mutants/` (index in `MUTANTS.txt`).

| Finding | Re-check | Fix | Test, and what it catches |
|---|---|---|---|
| ord-fallback-uninstantiated / F3 (kernels/exl3.cu, minor) | real. On the unfixed kernel, a grouped launch at S_cap = 65543 raised `variant N not instantiated` for variants 0, 5, 6, 7 and 9-15 (every SK), and `moe_forward` at P = 62000 (S_cap 65875) raised it from `launch_grouped` after route_prep and rot_in were enqueued. Not reachable from production (P <= P_cap = 1024 there) | `grouped_code_for()` resolves a variant for one launch (PIPE fallback; the ORD 0 fallback also drops LEAN and keeps POL) and the ORD 0 + evict_first instance (code 1) is compiled; `tf_set_variant` refuses a variant that could resolve to a missing instance; `moe_forward` / `moe_forward_ids` check both grouped launches before their first kernel | E-U2c: every variant, gate/up SK 2 / 4 / 8 and down SK 1 / 2, at S_cap 5 and 65543, Z bit-identical to the master kernel (unfixed kernel: 11 variants fail). E-U7: `moe_forward` with P = 62000 (small dims, n = 62000 experts over 4 real ones) runs and equals the same tokens served by two ORD 1 calls bit for bit (unfixed kernel: raises). compute-sanitizer memcheck of the fallback instances: 0 errors |
| F1 (tests/test_graph.py, minor) | real: with K2 on by default every capture went through K2, and the counter could not tell the paths apart | the capture / 100-replay / two-layer (EP map, shared scratch) suite runs on all three serving paths: K2, the dispatcher with `TF_EXL3_APPLY=0`, the dispatcher after `disable_apply`; the serving path of each capture is identified by `tf_apply_calls` | mutant "`plan()` returns None while capturing" (TF silently absent from every graph on the dispatcher path): old test_graph 14/14 pass, new 28/39 (11 checks fail) |
| F2 / F5 (gate/up SK 2 above P = 512 untested bitwise, minor) | real: no bitwise test ran gate/up SK 2 or the K3b discards at SK 2 (the experimental variant 14, SK 2 above P = 32, was compared only within E.1), and the shipped variant never ran SK 2 in any bitwise comparison | E-U2c gate/up SK 2 and 8; E-U4 (numpy emulation) SK 2; E-U6 (the exl3_moe binary's own intermediates) SK 2; test_variants: bare forward at P = 513 / 520 / 640 / 768 / 1024 (one route per token, B = P), scratch poisoned with NaN and flushed out of L2, shipped variant 0 == variant 12 (table 6 without discards) bit for bit and == a repeat; tables 1 / 0 within E.1 | mutant (a) "no fp16 rounding of the split sum at SK 2" (a parity-precision break): all old tests pass, incl. E.1 at P = 520..1024 (379/379); new E-U4 fails (13002 ulp, 64% of values) and E-U6 fails (72.6% of xd differ). Mutant (b3) "a gate/up epilogue block drops one live Z line of another block at SK 2": old test_variants / test_graph / test_e2e_vs_xl / test_apply_fused pass; the new forward test fails in 3 of 3 runs. (A gross mutant, every block dropping its neighbour row at SK 2, was already caught by the old E.1 tests; the new test also catches it. A first single-line mutant aimed at row P - 1, a sentinel pair in these tests, was inert and failed nothing.) STATUS now states exactly which SKs the variant bit-identity covers |
| F1 (docs/OPTIMIZATION.md, major) | real: `tests/logs/` is gitignored; the doc's table rows are not in the log it names (overwritten by a later run); several confirmation logs were saved as summaries | logs are committed under `docs/logs/`: `history/` = the original run files exactly as they were (some summary-only; kept, not re-typed), `run_all/` = the final run_all (every step's log and its console output), `ab/` = a re-audit of every shipped item and of master -> final as paired A/B runs with full per-configuration output (A / B us, spreads, bitwise / E.1 verdicts), two repeats each. `review_mutants/` = the mutation checks of this review. The speed tables in this file and in STATUS.md are generated by `tests/doc_tables.py` from those files | `python3 tests/doc_tables.py` reproduces every table below from the committed logs |
| F2 (docs/STATUS.md, minor) | real: 0.87 came from K2's T = 1 ratio; the stale "T = 96..128 is 1.05-1.08" line contradicted the table | headline recomputed from the paired master -> final run (corr40 and rand T = 5..64); stale line replaced | numbers generated by tests/doc_tables.py |
| F3 (tests/bench_ab_decode.py, minor) | real: the stage loop rebound `kind`, so progress / correctness lines read `down_epi` | loop variable renamed `stage`; the correctness line also carries the routing kind; a check asserts the label equals the configuration's kind | mutant (the old loop variable) on a shortened bench: 2 checks fail (`routing-kind label 'down_epi' != 'rand'`); fixed: 133/133 |
| F4 (docs/OPTIMIZATION.md, minor) | real: the master -> final table set two separate runs side by side (the master T = 1 row from a contended run); the noise figure (+-1%) was smaller than the example it cited (0.984) | master -> final is now a paired A/B in one process (`tests/bench_variant_ab.py 1:0 0:1 --mode full`: master kernels with production's prelude + the exl3_moe path vs the shipped variant with K2, on production apply graphs); the noise is measured by a null run (`0 0`, identical configurations) and stated from it; a change is called a gain only when both repeats are below the null's 90th percentile | ab/null_0_vs_0.log, ab/full_master_vs_final.log |
