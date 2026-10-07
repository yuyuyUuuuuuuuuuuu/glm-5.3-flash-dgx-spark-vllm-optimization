# Methodology

## Speed

### Decode
- **Tool**: the launcher's `tests/bench_decode.py`, driven by a small wrapper that runs one round per
  workload.
- **Settings**: n=10 requests per workload, temperature 0, thinking off, concurrency 1, 200 completion
  tokens.
- **Gating**: a run starts only when `vllm:num_requests_running == 0` and nothing is waiting. Any run
  that overlapped user traffic is flagged in `data/decode_bench.csv` (look for min tok/s outliers).
- **Workloads**:
  - *structured*: count from 1 to 200. Acceptance is always 7.0/step, so this workload isolates step
    time.
  - *prose*: acceptance about 2.0-2.4/step.
  - *coding*: acceptance 3.0-4.35/step. It moves from boot to boot, so compare acceptance before you
    compare coding tok/s.
  - *ja*: Japanese prose, acceptance about 1.8/step. Added on 2026-10-04 after a bug corrupted Japanese
    output while structured looked fine.
- **Metrics**:
  - `tok/s`: median over requests, TTFT excluded.
  - `ms/step`: `decode_ms_per_draft_step_median`, a serving-cycle ratio. It is not a kernel timing:
    a profiled step and a benchmark ms/step differ by a few ms.
  - `acc/step`: accepted tokens per step, median.
- **Sampled decoding**: production defaults are temperature 1.0, top_p 0.95, repetition penalty 1.05.
  `bench_decode.py` cannot measure the effect of sampling-only changes (block verification, drafter
  precision). For those we used the same 16 seeds at temperature 1.0 on both arms.
- **Concurrency**: the same tool with 2, 4 and 8 parallel streams (`data/concurrency.csv`).

### Prefill
- **Real text**: 8.5k and 14.8k-token real documents, median of 3. The R series (09-27 to 09-28) used
  8.5k and 15.3k texts, so the two series are not directly comparable.
- **Random tokens** (24k): reported, but routed-MoE routing collapses on random text, which inflates it.
  Quote real text.
- **32k gate**: the public "gate/prefill.py" yardstick from the GX10 report we compared against. It
  sends cold 32k prompts (128k rows also exist) to an idle node, about 6 samples, median. Rows taken
  while users were active are discarded, and arms without enough idle samples are marked invalid.
  Between restarts it varies by +-1%.
- **APC**: a 97,359-token prompt sent cold, then repeated. The warm run should have 96,768 tokens
  cached, which is the last multiple of 4,608. A 40k repeat should have 36,864 cached.
- **Stream gaps**: inter-token gap percentiles at 2k and 40k context, and a count of gaps over 200 ms.

### A/B procedure in production
- **Arm order**: `base, X, ..., b2, X2`. Each arm gets a full restart. The idle wait before a restart
  can last up to 900 s.
- **Preflight**: per-rank boot checks. Both ranks must log the switch as engaged. A first rollout once
  passed new variables only to the worker, and the per-rank check caught it.
- **Decision**:
  - The gain must exceed a margin of 1-2% on the gate, or the ms/step change must be beyond A/A noise.
  - Every quality probe must stay inside its limit.
  - Ties are kept at base.
- **Hand adoption**: some arms were adopted by hand although a strict rule failed, because user traffic
  contaminated one arm. Each case is written down in `timeline.md`.
- **Speed noise floors** (from A/A arms):
  - gate32k +-1%
  - decode ms/step +-0.5-1 ms
  - single-boot NLL resolution about 0.001
- **Timing**: user traffic makes A/B gates unusable after about 07:00 local time. Long arms run at night.

## Quality

All probes compare against stored reference outputs or against the same server's own prefill.

| probe | what it measures | size | reference values |
|---|---|---|---|
| decode-vs-prefill KL ("dvp") | generate 1,536 tokens from each of 3 prompts, then re-score them with a prefill pass. KL between decode-time and prefill-time logits | 4,608 positions | R15 0.00625 (top-1 98.48%). Production control arms after e4m3 vary 0.0118-0.0224 on identical configs |
| teacher-forced dvp ("dvptf", klh harness) | same, but teacher-forced on fixed texts, assistant spans only | 12 texts x 2 | about 0.010-0.013 |
| klh NLL | mean NLL of fixed assistant spans, with request-level t-intervals | 12 texts x 2 | about 0.500-0.504 |
| long KL | prefill top-k KL at positions >= 4,608 of 3 long texts against stored outputs | 28,275 positions | 0.0047 (R16 era), 0.014 (e4m3 era) |
| short KL | prefill KL on 6 short texts | 962 positions | 0.006-0.025 |
| output checks | temperature-0 output digests and acceptance per workload | 4 workloads | |
| concurrent-quality probe | KL of the same prompts decoded solo against 4-way concurrent | 4 prompts x 12 streams | ratio about 1.0 |

### Caveats
- **Reference changes**:
  - Until 2026-10-04 13:52, long, short and dvp references were the R15 outputs on the TR3 weights.
  - After the switch to the TensorFold calibration, long KL jumps 0.014 -> about 0.055 and short KL
    0.019 -> about 0.063 because the reference came from the old weights. NLL actually improved by
    0.0015.
  - A new klh reference ("v2") applies from 14:39 that day.
  - From 10-05 09:49 production runs the o_proj transplant (ABLIT=1, `docs/timeline.md`). It shifts long
    KL to about 0.058 and the teacher-forced NLL by +0.0035, so arms before and after it are not comparable
    on those two columns.
  - Compare levels only within the same reference. `data/production_ab_runs.csv` has a `kl_reference`
    column.
- **Production nondeterminism**: 7 identical base runs had a mutual long KL of 0.0139, about equal to
  their KL against the reference. The same request on the same boot gives about 0.005. Long KL therefore
  cannot resolve small effects. The teacher-forced harness and A/A arms were built to deal with this.
- **Prefill probes do not exercise decode-state handoff.** A rollout with long KL at 0.0048 (normal) had
  dvp at 0.294 (broken). Every prefill-path change gets a dvp probe.

## Measurement pitfalls we hit

1. **Wrong workload.**
   - A thinking-on, `max_tokens=400` test spent every token on reasoning, so "17-18 tok/s" was the speed
     of thinking.
   - A structured-only check passed a change that broke Japanese and coding.
   - A random-text prefill gain (+26%) was larger than the real-text one (+17%).
2. **Contaminated by live traffic.**
   - Early runs had min tok/s outliers of 2.0-2.2.
   - A concurrency conclusion was drawn while other requests were running.
   - A "+10% from a sysctl" was later traced to a concurrent request.
   - Check `num_requests_running` before every round, and repeat any arm whose min tok/s collapses.
3. **Broken instruments on GB10.**
   - `nvidia-smi` reports memory.used as `[N/A]`, and utilization and power near 0 even at 220 GB/s.
   - MemAvailable near 0 is normal for a pre-allocated vLLM.
   - Use swap in/out (`vmstat si/so`), `dmesg` OOM lines, vLLM's own "GPU KV cache size" and
     `request_success_total{error,abort}` instead.
4. **UTC against JST.** Container logs are UTC and the host is JST. An OOM kill was first denied because
   the search used the wrong clock window. `sar` kept the only clean evidence.
5. **Ratios.** "17.7x faster prefix cache" was a 2.5x cold regression in seconds. Report seconds and
   tok/s.
6. **Sample size.** n=3 produced a +10.7% that was +0.1% at n=10. The decode noise is a stdev of
   4-6 tok/s with low-side outliers.
7. **Test below the block size.** Prefix caching works in 4,608-token blocks. The last chunk is excluded
   and DFlash drops one more block, so prompts under about 16k cache only on the 2nd or 3rd repeat.
8. **Single-node rigs.**
   - They cannot show RoCE all-reduce contention.
   - A single-process engine on node C read a different block size (576), which faked a "prefix hit
     breaks" bug.
   - They captured whole CUDA graphs while production uses piecewise graphs.
9. **Other people's numbers.** Check TP, K, temperature and hardware first. Examples: a "+68%" drafter
   (TP=4, K=2), "145-151 tok/s" (2x RTX PRO 6000), and a 2,929 tok/s prefill (NVFP4, capped at a 160k
   context).
10. **Health is not capacity.** The models endpoint returned 200, and both nodes then ran out of memory
    under load.
11. **Grep is not a judge.** One A/B script aborted because it matched "self-test FAILED 0" as a string.
    Another passed an empty value through a string comparison. Use anchored patterns, numeric
    comparisons, `set -o pipefail`, and check exit codes.

## Operational rules that came out of incidents
- Restart both ranks: stop, wait 20-25 s, auto-detect the RoCE GID (its index moves across reboots),
  then start with the image rebuild skipped. Wait for `running=0` and `waiting=0` first.
- Never run `import torch` or similar inside the serving container. Nineteen parallel debugging shells
  did that and the global OOM killer chose the engine core.
- Never stack downloads or other large allocations on a live node. Run `free -g` first.
- On unified memory, enforce test-job GPU memory caps with a mechanism, not a convention. We use a
  per-process memory fraction from `sitecustomize` plus a container memory limit. A node froze when four
  test jobs ran at once.
- Keep nothing important under `/tmp`; it disappeared with that reboot.
- Anchor configuration edits (`^KEY=`). A loose `sed` once rewrote a comment, and on another occasion
  the live `.env`.
