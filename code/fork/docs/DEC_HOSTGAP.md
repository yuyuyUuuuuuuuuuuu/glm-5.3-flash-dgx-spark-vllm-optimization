# DEC_HOSTGAP — where the GPU idles inside a decode step, what production still pays, and the per-step tracer

Status 2026-09-30, branch `hostgap` (from r16j 905c07c): implementer commit 810543c + adversarial review (this file's
§0 and every "review" note). Everything was run on **nodeC**; production (nodeA/nodeB) was not accessed. Sources:
- the production decode profile `${HOME}/tf-exl3-assets/decode-prof/p6h.json.gz` (torch profiler, CPU + GPU,
  rank 0, 39 decode steps, median step 69.10 ms). **It is an R15 profile: it predates `GLM53_DEC_HOSTLOOP`.**
- the 01:00 two-rank profiler pair, analysed in `docs/logs/hostloop/skew_0100_pair.log` (R15 as well);
- the deployed kit `${HOME}/tf-exl3-deploy16.r16j` (env.r16, start.sh, boot_checks.sh);
- the logs under `docs/logs/hostgap/` (implementer) and `docs/logs/hostgap/review/` (review).

## 0. Review summary (read this first)

**What production runs today.** Per the owner's deployment record (2026-09-30 03:30, "R16J-opt"): DFlash2 + block
verification (`GLM53_REJECTION_METHOD=block`) + kpool ring + **`GLM53_DEC_HOSTLOOP=1`** + FP8 drafter, on top of R16i
(APC short-suffix fix, smallops dconv, prefill quick wins, the fixed MLA prefill). fp8roof / moeglue warm were measured
ineffective in production on 09-29 (their .env state was not re-verified from here). Of this document's knobs:

| knob | in production | reaches both ranks with the r16j start.sh |
|---|---|---|
| `GLM53_DEC_HOSTLOOP=1` (host tail off the critical path) | **ON** | yes (env.r16 line) |
| `GLM53_DEC_HOSTLOOP_WAKE=auto` (cold NCCL host node) | **off** (never an env.r16 line; DEPLOY_R16 §0 "not shipped") | **yes**: start.sh forwards `_WAKE`, `_WAKE_TICK_US`, `_WAKE_PIN` |
| `GLM53_DEC_HOSTLOOP_METER=<N>` (per-rank gap + eager-collective meter) | off | **yes** |
| `GLM53_DEC_TRACE*` (this branch's tracer) | not installed (the r16j site/ has no glm53_dectrace.py) | **no**: start.sh does not forward the names and `GLM53_EXTRA_ENV` rejects every `GLM53_*` name |

**Nothing on this branch removes anything that production does not already have.** Of the implementer's "removable
2.3–2.6 ms/step", the 1.756 ms host tail is `GLM53_DEC_HOSTLOOP`, which is already ON in production (so the p6h
numbers describe a state production left on 09-30), and the cold-host-node part is `GLM53_DEC_HOSTLOOP_WAKE`, which
already ships in the r16j kit and is forwarded to both ranks, only not enabled. Both came from the `dec-hostloop`
branch (09-28), not from this one. What this branch adds is analysis, the tracer, and (review) kit fixes.

**Review defects found and fixed here** (details in the sections):
1. The committed aligner (`tests/align_dectrace.py` @810543c) used the eager collectives' **host return times** as the
   shared clock. They are not one: vLLM's `pynccl.all_reduce` only enqueues. On p6h the eager all-reduce's host call
   returns a median 127 µs (max 6.6 ms) before its kernel ends, and the eager all-gather's a median **58 ms** before
   its kernel even starts (`review/review_eager_coll_host_vs_gpu.log`). On a physically modelled pair the committed
   aligner reports the host skew with the **wrong sign** (−200 µs for a true +250 µs), a clock offset off by 450 µs,
   and **zero** all-reduce wait where rank 0 waits 300 µs (`review/old_aligner_on_physical_pair.log`). Its A1 test
   passed only because the synthetic data encoded the same false assumption. Replaced: the tracer times each eager
   collective on the GPU (CUDA events), and the aligner derives the rank wait from the two ranks' GPU durations (§3.1).
2. The committed tracer stamped `sched` with the **previous** step's index (it recorded before incrementing). T2's
   check was a tautology (`steps == steps`), and the implementer's own `node1_proof.log` shows step ids `[0, 1, 2]`.
   Fixed; T2 now checks every event's step id.
3. The committed tracer resolved the rank with `parallel_state.tensor_model_parallel_is_initialized`, which does not
   exist in this vLLM (g487ecf187). It fell back to `LOCAL_RANK`: unset in the harness (the implementer's engine
   file is `dectrace-rank-1-...`), and 0 or unset on each of the two single-GPU production nodes, so neither production
   file would name its TP rank. Fixed (`model_parallel_is_initialized`); the review's engine file says `#! rank 0`.
4. In production decode the target runs as a FULL CUDA-graph replay (`ModelCudaGraphManager.run_fullgraph`) and never
   calls the model's forward, so the committed `tgt_b/tgt_e` never fire on a decode step. Added `fg_b/fg_e` around the
   replay.
5. The report claims "D1/D2 PASS in the production image", but the committed `docs/logs/hostgap/node1_proof.log` says
   `D2 ... FAIL` / `RESULT: FAIL`: the `rs` hook was never installed on the real tree. D2 no longer requires it (DFlash2's
   MRv2 path bypasses `RejectionSampler`).
6. The tracer must install **after** glm53_hostloop, whose fingerprint check reads the model-runner methods. It does
   (finder order in `integrate.plugin_register`), but nothing tested it. Added D3 (hostloop + trace through the real
   plugin loader: the hostloop fast path IS installed) and D3n, a negative control: with the tracer installed first,
   hostloop refuses.
7. §1's attribution table credited 0.525 ms/step of idle to "the planner's blocking D2H (`_kv_lens_host`)", i.e. to
   hostloop. 0.521 ms of it ends at an **NCCL all-reduce kernel**: the GPU idles before a graph's first all-reduce
   waiting for NCCL's cold host node while the host happens to sit in the copy. That is WAKE's target, not hostloop's
   (`review/review_gap_memcpy_next.log`).
8. Kit: `env_r16.sh on trace` wrote two `.env` lines that the r16j start.sh never forwards, printed only a warning,
   and ended with "(restart both ranks to apply)": a restart of both production ranks for no trace. It also wrote the
   default directory `/tmp/glm53-dectrace` inside the container (not mounted, gone with the container). Now `on trace`
   **refuses** with `.env` unchanged unless the installed start.sh forwards both names to both ranks, and the default
   directory is `/root/.cache/vllm/glm53-dectrace` (bind-mounted from each node's host). S.12 of `test_kit_scripts.sh`
   also checked the wrong tree whenever `PREV_KIT` was set (S.10/S.11 export `LAUNCHER_DIR=$L2`); fixed.
9. Kit: `boot_checks.sh` required the literal hostloop line `meter off, wake off`, so **any production A/B of
   `GLM53_DEC_HOSTLOOP_WAKE` (the removal for the cold positions) or `GLM53_DEC_HOSTLOOP_METER` (the measurement) is
   reported as a failure** by the deployed boot_checks (B.12 against the r16j boot_checks: 4 FAIL, rc 1,
   `review/test_boot_checks_vs_r16j.log`). It now expects the
   line with the head container's values, requires the worker to carry the same values, and flags wake's refusal and
   self-disable lines.
10. The report says `test_boot_checks.sh` B.10/B.11 "fail identically against the unmodified deployed kit". Against
    `${HOME}/tf-exl3-deploy16.r16j` the unmodified suite is **ALL OK** (`review/test_boot_checks_base_vs_r16j.log`).

## 1. Attributing every GPU-idle interval to the host code that was running (p6h, R15, before hostloop)

Tool: `tests/gap_host_attr.py <trace>` (log `docs/logs/hostgap/gap_host_attr_p6h.log`, reproduced by the review). It
splits the trace into steps on the once-per-step `marlin::Marlin` kernel and pairs every idle gap >10 µs with the
innermost host `cpu_op` / `cuda_runtime` event covering the gap midpoint. **Caveat (review):** the host op covering a
gap is not always its cause. When the host is blocked in a sync, the GPU is idle for a reason inside its own queue.
`tests/review_gap_memcpy_next.py` prints the GPU op that ends each such gap.

| host code running while the GPU is idle | gaps/step | ms/step | what actually removes it |
|---|---|---|---|
| `<HOST IDLE>`: pure Python between ops (no host op recorded) | 14.3 | 0.613 | hostloop (the host tail overlaps the drafter) |
| `aten::_to_copy`/`slice`/`sub`/`sum`/`empty_strided`/`index`/`copy_` (the post-copy metadata builds) | ~25 | 0.62 | hostloop |
| `cudaMemcpyAsync` (host blocked in the planner's `seq_lens.cpu()`) | 1.6 | 0.525 | **not hostloop**: 0.521 ms/step of these gaps end at an `ncclDevKernel_AllReduce`, i.e. the first all-reduce of a CUDA graph waiting for NCCL's cold host node → WAKE |
| `cudaGraphLaunch`, `TorchDynamo`, `vllm::all_reduce`, `cudaLaunchKernel`, small aten | ~8 | 0.19 | mostly hostloop |
| **total idle >10 µs** | ~49 | **1.955** | |

Cross-checks (`tests/hostloop_trace_ana.py`, log `trace_ana_p6h.log`, reproduced): the blocking D2H returns, then
**1756 µs** of host work run with the GPU drained until the target's first kernel. Collectives: 104/step, 4.41 ms
inside collectives, **2.27 ms beyond the 20.6 µs transfer floor**, 1.02 ms of GPU idle right before a collective:

| position in the step | beyond the floor (rank 0 waits) | GPU idle right before | cause | removal |
|---|---|---|---|---|
| 0: target embedding AR (eager) | 324 µs | 41 µs | rank 1's host tail is longer (01:00 pair: r1 2041 vs r0 1749 µs; arrival skew 957 µs mean, 306 median) | hostloop (both tails off the critical path): **ON in production**; residual unmeasured |
| 1: first AR inside the target graph | 240 µs | 53 µs | NCCL's cold host node at graph start | **WAKE** (off in production) |
| 92: first AR inside the drafter graph | 293 µs | **507 µs** | NCCL's cold host node at graph start | **WAKE** (off in production) |
| the other 101 | ~1–10 µs each (1.29 ms total) | ~4 µs | GPU speed difference between the two nodes | structural |

## 2. What production still pays, and the concrete removal for the cold positions

With hostloop ON, the p6h host tail (1.756 ms) and the pos-0 wait that it caused should be off the critical path
(nodeC: gap 1.327 → 0.044 ms/step, `node1_proof.log`; the deploy doc's production-shape estimate is −1.9 ms). **This
has not been measured in production.** The 09-30 production A/Bs measured other features at temperature 1.0.

Still paid, projected from the R15 profiles: the two cold NCCL host-node positions. On p6h rank 0 that is ≈0.56 ms
idle + 0.53 ms wait per step. On the 01:00 pair: pos 92 idle 344/445 µs (r0/r1) + wait 145/89 µs; pos 1 idle 77/188 µs
+ wait 127/14 µs. The pos-0 wait (324 µs p6h, 957 µs mean on the 01:00 pair) belongs to hostloop and should now be
small.

**Concrete removal, no new kit needed:** `GLM53_DEC_HOSTLOOP_WAKE=auto` (docs/DEC_HOSTLOOP.md §4). It is in the r16j
site/ (`glm53_hostloop.py`), the r16j start.sh forwards it to both ranks, and it needs only one `.env` line plus a
restart of both ranks. nodeC evidence: cold graph host-node latency median 441 → 89 µs, p90 827 → 149 µs
(`docs/logs/hostloop/bench_hostloop_wake_auto.log`); GPU bandwidth unaffected (−0.04 % vs A/A +0.03 %,
`docs/logs/hostloop/review/bench_round2_wake_bw.log`); cost ≈15 % of one core in SCHED_IDLE tickers. Projected
−0.4 … −0.9 ms/step. **Not measured in production**; a single-node probe has no RoCE traffic and none of production's
CPU load, and single-node "gains" have not held on the two nodes before (the fp8roof lesson). A system-level
alternative with the same mechanism: disable cpuidle LPI-2/3 on nodeA/nodeB (root).

**How to measure what is left, and the WAKE A/B, without a new kit:** `GLM53_DEC_HOSTLOOP_METER=200` on both ranks
(forwarded by the r16j start.sh) logs every 200 decode steps per rank: `gap_ms` (drafter end → next target start on
the GPU = what hostloop removed), `step_ms`, and `c0/c1`, the GPU durations of the two eager collectives. The rank with
the longer c0 arrived first, and the c0 difference is the pos-0 skew. The cold positions are inside the graphs and
invisible to both the meter and the tracer; they show up in `step_ms` (A/B) or in a two-rank profiler pair
(`GLM53_DEC_PROF_DIAG=1` + SIGUSR2, `tests/skew_from_traces.py`). **Use this branch's `boot_checks.sh`**: the r16j one
fails the hostloop check as soon as METER or WAKE is set (defect 9). A production A/B restarts both ranks per arm, so
it needs the owner's go and an idle window (`tools/wait_idle.sh`).

## 3. `GLM53_DEC_TRACE`: the per-step tracer (measurement only, no numerics change)

`glm53_dectrace.py` (in `setup.py py_modules`, installed from `integrate.plugin_register` **after** glm53_hostloop;
inert unless `GLM53_DEC_TRACE` is set). It wraps existing functions only and never touches a value. On a rank, per
execute_model call n, host events (`n t ev`, `t = time.perf_counter_ns`):

```
sched  execute_model entered            prep   prepare_inputs returned
fg_b/fg_e    target FULL-graph replay launched/returned (production decode)
pw_b/pw_e    target PIECEWISE run (ModelCudaGraphManager.run_pw_graph; breakable-cudagraph replay, no forward call)
tgt_b/tgt_e  target model forward (eager only: prefill beyond the capture sizes, warm-up, capture)
sam_b/sam_e  sample_tokens (the drafter's propose runs inside it)       dr_b/dr_e  speculator.propose
rs_b/rs_e    RejectionSampler.forward (not on DFlash2's MRv2 path)
ar_b/ar_e, ag_b/ag_e  eager all-reduce / all-gather HOST call (the enqueue, NOT the wait)
exec_e execute_model returned
```

plus GPU records `n t gar|gag dur_ns` (`GLM53_DEC_TRACE_GPU`, default on): CUDA events recorded on vLLM's current
stream (the stream pynccl enqueues on) just before and after each eager collective. At most 8 per step, none during
CUDA-graph capture; they are resolved lazily with `query()` (never a sync), drained from `step_done` once 64 are
pending, with the pending list bounded at 512 (drops are counted in the stop line). A failing event record switches
GPU timing off and leaves the host events and the collective untouched. Production decode has 2 eager collectives per
step (target embedding AR, target logits AG); the other 102 run inside CUDA graphs and are invisible to Python.

### 3.1 What is a shared clock (review)

No host event is one. A collective's GPU duration on a rank = its wait for the other rank + the transfer, and both
ranks finish it together (01:00 pair end-time residual 7.7 µs median). So per step and position:
**arrival skew (r1 later) = dur_r0 − dur_r1**, **wait_r = dur_r − min(dur_r0, dur_r1)**, with no clock at all.
"Arrival" is when that GPU reached the collective's slot in its stream; on a host-bound step the duration also
contains the few µs between the begin event and the NCCL launch. Cross-rank host skew (sched, prep, fg_b, …) needs an
offset from outside: `--offset-us X`, or `--wallclock` (the files' `#! clock <perf_counter_ns> <time_ns>` pairs, as good
as the nodes' NTP sync). The aligner never derives an offset from collective returns.

Knobs (read once at install; both ranks need the same values):
- `GLM53_DEC_TRACE=<dir>`: one `<dir>/dectrace-rank<R>-pid<PID>.ndjson` per process, renamed once the TP rank is
  known (`#! rank R` line); truncated at boot. `0`/`off` = off.
- `GLM53_DEC_TRACE_RING=<n>` (4096): records kept in memory before a flush. `GLM53_DEC_TRACE_FLUSH=<n>` (64): also
  flush every n steps (one `write()` per flush, plus a `#! clock` line).
- `GLM53_DEC_TRACE_MAX_STEPS=<n>` (20000, 0 = unlimited): stop and close after n steps. This bounds the file
  (~20 records × ~25 B per step ⇒ ≈10 MB per 20 k steps ≈ 23 min of decode at 70 ms/step).
- `GLM53_DEC_TRACE_EVENTS=<csv>` (`-name` drops one); `GLM53_DEC_TRACE_GPU=0`: host events only.

Overhead (measured): 203 ns per record and 253 ns per wrapped call on the host (T6). In the production image, a
GPU-timed eager collective costs 4.8 µs more host time than the bare call: 0.7 µs for the host events and 4.1 µs for
the two CUDA events (G4, median of 320 paired calls). Two per decode step plus ~11 other records gives ≈ 12–13 µs per
~70 ms step (≈0.02 %). Not measured on the real engine's step time (§5).

Offline: `tests/align_dectrace.py <rank0> <rank1> [--csv F] [--steps N] [--offset-us X | --wallclock]`. It matches
steps by index and prints (1) the per-position arrival skew and waits from the GPU records, (2) each rank's host
windows on its own clock (`prep→fg_b` = the host work before the target launch, which hostloop moved off the critical
path), and (3) cross-rank host skew only with an outside offset. `SUMMARY {...}` is machine-readable.

**Relation to `GLM53_DEC_HOSTLOOP_METER`** (already in production's kit): the meter gives per-rank medians of the gap
and of c0/c1 in the log. The tracer adds per-step pairing of the two ranks and the host windows. The meter answers
"what is left" today; the tracer needs a new kit.

### 3.2 Installing it (not done; needs a new kit)

The r16j start.sh forwards no `GLM53_DEC_TRACE*` name, and `GLM53_EXTRA_ENV` refuses `GLM53_*` names. To install:
add the six names to `tools/deploy16/make_start_sh.py` `KNOBS_DEFAULT` (head `-e` line + worker `serve_env_names`,
like every R16 knob), rebuild the kit (the start.sh hash chain changes), and ship `glm53_dectrace.py` + this
`integrate.py` in site/. Then `tools/env_r16.sh on trace` writes `GLM53_DEC_TRACE=/root/.cache/vllm/glm53-dectrace` +
`GLM53_DEC_TRACE_MAX_STEPS=20000`; the files land on each node's host under the vLLM cache directory (head
`$CACHE_ROOT`, worker `$WORKER_VLLM_CACHE`). With the r16j start.sh, `on trace` refuses and leaves `.env` unchanged
(S.12).

## 4. Evidence (nodeC)

| what | command | result |
|---|---|---|
| attribution numbers reproduce | `tests/gap_host_attr.py`, `tests/hostloop_trace_ana.py` on p6h | identical to the committed logs |
| host return ≠ completion | `tests/review_eager_coll_host_vs_gpu.py p6h.json.gz` | AR: kernel end − host return med 127 µs (max 6574); AG: kernel start − host return med 58 ms |
| the memcpy-credited idle is the cold host node | `tests/review_gap_memcpy_next.py p6h.json.gz` | 0.521 of 0.525 ms/step ends at `ncclDevKernel_AllReduce` |
| committed aligner is wrong on physical data | 810543c `align_dectrace.py` on the A1 pair | host skew −200 (true +250) µs, offset 3.65 (true 3.20) ms, AR wait 0 (true 300) µs |
| tracer + aligner, host | `python3 tests/test_dectrace.py` | T1–T7, A1 PASS (`review/test_dectrace_host.log`) |
| tracer in the production image | `tests/hostloop_gpu.sh python3 tests/test_dectrace.py --plugin --gpu` | PASS (`review/test_dectrace_image.log`): D1 unset → no hooks, no file; D2 hooks incl. `run_fullgraph`, `plugin_register` creates no CUDA context; **D3** hostloop fast path installed next to the tracer via the real plugin loader; **D3n** tracer-first → hostloop refuses (`execute_model`/`prepare_inputs` fingerprints); **G1** gar 1827.0 µs vs calibrated 1825.6 µs, host call 13.3 µs (returns long before the GPU), drain 5 µs while the GPU is busy, no sync; **G2** no event during `torch.cuda.graph` capture, capture + 2 replays correct; G3 GPU off → host only; **G4** +4.8 µs per GPU-timed collective |
| real engine: tracer off / on / off | `tests/handoff/run.sh <out> {hl_notrace,hl_trace,hl_notrace2} GLM53_DEC_HOSTLOOP=1 [GLM53_DEC_TRACE=/out/hl_trace/dectrace GLM53_DEC_TRACE_FLUSH=1] -- --shadow 0 --prompts 3001 --gen 24` (DFlash2 mini model, production-composed container) | generated tokens (1 request × 24) **identical** in all three; top-5 logprob drift traced-vs-untraced 6.84e-2 = untraced-vs-untraced 6.49e-2 (cross-process baseline); trace: `#! rank 0` (resolved), 33 steps, **`fg_b/fg_e` on all 25 FULL-graph decode steps**, `prep→fg_b` median 842 µs (no hostloop), clean `#! stopped exit` (`review/engine_trace.log`, the file `review/engine_hl_trace.dectrace-rank0.ndjson`). Note: the harness wraps `execute_model` itself (run_engine.py:887), so hostloop refuses in **every** harness arm (`NOT installed: … execute_model=029b0b50…`, harness-only); hostloop + tracer coexistence is D3 |
| aligner on the real engine file | `tests/align_dectrace.py f f` | runs, warns "same rank twice", no GPU records at TP=1 (`review/align_engine_selfcheck.log`) |
| boot_checks markers printable by the shipped code | `tools/deploy16/check_boot_strings.py tools/deploy16/boot_checks.sh <scratch kit with this branch's site/*.py> prod-launcher` | 99 markers, 0 not printable (`review/check_boot_strings.log`); r16j baseline 96/0 |
| kit scripts | `PREV_KIT=…r16i tools/deploy16/test_kit_scripts.sh <scratch kit>` | ALL OK, 71 checks incl. S.10/S.11 and the new S.12 (`review/test_kit_scripts.log`) |
| boot checks | `tools/deploy16/test_boot_checks.sh <scratch kit>` | ALL OK incl. B.12 (`review/test_boot_checks.log`); B.12 against the r16j boot_checks: 4 FAIL = the A/B hazard |

The scratch kit is the r16j kit with this branch's `tools/env_r16.sh`, `tools/boot_checks.sh`, `env.r16` and their
MANIFEST lines regenerated.

## 5. NOT RUN, and risks

- NOT RUN: **two real ranks** (nodeC has one GPU; NCCL refuses two ranks on one device). The GPU-duration method rests
  on NCCL's shared end time, shown on the real 01:00 pair (7.7 µs residual), and on G1's CUDA-event mechanics, but the
  tracer has never run on the two production ranks.
- NOT RUN: anything in production (no WAKE / METER A/B, no install of the tracer). What production pays today after
  hostloop is **projected**, not measured.
- NOT RUN: the tracer's cost on the real engine's step time. The engine runs check output identity and event content,
  not ms/step (the mini model's steps are too short and noisy for a µs-level A/B).
- Risk (WAKE): production cores are busier than nodeC's (engine-core spin-wait, NCCL proxies, API server), so the gain
  may be smaller; tickers cost ≈15 % of one core per rank.
- Risk (tracer): one Python closure per wrapped call on the hot path while tracing (≈µs/step); with
  `GLM53_DEC_TRACE_GPU` on, two event records per eager collective on the decode stream.
- The attribution is from one batch-1 R15 profile; ratios shift with batch size and with hostloop now on.

## 6. Independent verification (2026-09-30, after the review; nodeC only)

Re-run from the committed code, logs in `docs/logs/hostgap/verify/`: the p6h numbers (127.0 µs / 58.0 ms host return
vs kernel, 0.521 of 0.525 ms/step ending at `ncclDevKernel_AllReduce`, 1.955 ms/step idle, host tail 1756 µs,
2.27 ms/step beyond the floor), the 810543c aligner's wrong answer on the A1 pair (offset 3.65 ms, sched −200 µs,
AR wait 0) and the new aligner's exact one, host tests, the production-image tests (G1 1826.9 vs 1826.3 µs calibrated,
G4 +4.9 µs), `test_kit_scripts` 71 ok, `test_boot_checks` ALL OK (4 B.12 FAILs with the r16j boot_checks),
`check_boot_strings` 99/0 (`reruns.log`). All reproduced.

**Defect found and fixed here: PIECEWISE steps had no target event.** Production auto-enables the breakable cudagraph
(`VLLM_USE_BREAKABLE_CUDAGRAPH`, "Auto-enabling ..." in the harness log too). A PIECEWISE step runs the target through
`ModelCudaGraphManager.run_pw_graph` → `BreakableCUDAGraphWrapper._replay`, which replays the segments and never calls
the model's forward. So `tgt_b/tgt_e` did not fire there, `fg_b/fg_e` neither, and the docs claimed tgt_* covered
PIECEWISE. The review's own engine trace shows it: steps 2, 4, 5, 7 are `sched prep exec_e sam_b dr_b dr_e sam_e`,
a target forward with no target event. Fix: `pw_b/pw_e` around `run_pw_graph`, set on `ModelCudaGraphManager` only
(the method is inherited from `CudaGraphManager`; the drafters' managers are siblings and stay unwrapped, checked by
T2 on stubs and D2 on the real tree). The aligner gains `prep->PIECEWISE run` and `PIECEWISE run (target)` windows.
Evidence: T2 (and its negative control: without the wrap, T2 fails on hooks 9, the missing pw pair and the class
check); image D1–D3/G1–G4 PASS with the new hook (D3: hostloop's fast path still installs next to the tracer); engine
run with the review's arguments: steps 2, 4, 5, 7 now carry `pw_b/pw_e`, every step with `prep` has a target event
(25 FULL, 4 PIECEWISE, 1 eager, plus the capture step), and the 24 generated tokens equal both of the review's untraced
arms (`engine_pw_trace.log`).

Observations, not changed:
- The aligner pools each eager-collective position (AR0, AG0, …) over all matched steps. FULL, PIECEWISE and eager
  steps can differ in arrival skew; use `--csv` and split by the step's target event when prefill/PIECEWISE steps are
  frequent in the window.
- The eager all-gather goes through `torch.distributed.all_gather_into_tensor` (ProcessGroupNCCL, its own stream), not
  pynccl. The CUDA events on the current stream still bracket it: it is a synchronous op, so the current stream waits
  for the NCCL end event. The duration adds the cross-stream event waits on both ranks, and those cancel in the skew
  (read from the vLLM/PyTorch source, not measured: nodeC cannot run two NCCL ranks).
- boot_checks builds the expected hostloop line from the raw container value (`meter every $METER steps`). hostloop
  prints `int(METER)`, so a non-canonical value such as `0200` would be reported MISS.
- `env_r16.sh on trace` (with a forwarding start.sh) removes hand-set `GLM53_DEC_TRACE_RING/FLUSH/EVENTS/GPU` lines
  before writing its two lines. S.12 checks for this, so it is intended.
