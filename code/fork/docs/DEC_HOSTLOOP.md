# DEC_HOSTLOOP — the decode host loop and the rank skew (`GLM53_DEC_HOSTLOOP*`, `GLM53_DEC_PROF_DIAG`)

Status 2026-09-28, branch `dec-hostloop` (from `deploy-r15` 45047ef). Built, tested and measured on nodeC (one GB10,
production image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` with the production mounts
flashinfer 0.6.18 + `cuda.py` + `flashinfer_mla_sparse_sm90.py.patched`, `tests/hostloop_gpu.sh`). Production
(nodeA/nodeB) was not accessed. Every number below is in a committed log under `docs/logs/hostloop/`.

Four independent knobs, all **default off = exactly the current production behaviour**, installed from
`integrate.plugin_register` (module `glm53_hostloop.py`; the profiler part is in `glm53_runtime.py`):

| knob | what | measured on nodeC | projected in production |
|---|---|---|---|
| `GLM53_DEC_HOSTLOOP=1` | plan the sparse-MLA per-row KV lengths from a pinned post-sampling snapshot instead of the full-stream `seq_lens.cpu()` sync: the host prepares step n+1 while the GPU still runs the drafter | GPU gap drafter→target 1.67 → 0.044 ms, step −1.61 ms (1 req); 2.88 → 0.044 ms, −2.82 ms (4 req) | **−1.9 ms/step** |
| `GLM53_DEC_HOSTLOOP_WAKE=auto` | keep the GPU-interrupt CPUs out of GB10's deep idle state (433 µs exit) and pin the CUDA host-callback thread, which runs the host node NCCL puts in front of every graph's first network collective | cold graph host node mean 478 → 111 µs, p90 827 → 149 µs | **−0.4 … −0.9 ms/step** (≈ −0.7) |
| `GLM53_DEC_HOSTLOOP_METER=N` | per-rank CUDA-event meter, one INFO line every N decode steps: host-loop gap, step time, host wait, eager collectives (skew) — no profiler needed | matches the bench's own events within 15 µs | measurement only |
| `GLM53_DEC_PROF_DIAG=1` | SIGUSR2 profiler: boot preflight per rank, CUDA-activity check per session with kineto's stderr, one automatic retry, rank-numbered file names | D1–D3 PASS | measurement only |

Together: 69.1 → ≈ 66.5 ms/step on the R15 prose profile (−2.6 ms, −3.8 %), no numerics change, no GPU memory.

## 1. Where the idle time is (R15 trace, rank 0; the stream's item 1)

Tools: `tests/hostloop_trace_ana.py <trace>` (one rank) and `tests/skew_from_traces.py <rank0> <rank1>` (two ranks,
clocks aligned on collective end times: residual 7.7 µs median). Logs: `trace_ana_R15_rank0.log`,
`skew_0100_pair.log`.

**The host timeline of one decode step (vLLM model runner V2, async scheduling, DFlash2).** The worker's main
thread runs one step ahead of the GPU: while the GPU executes step n, the host has already launched step n's target
graph, its sampling and its drafter graph, and is in `execute_model(n+1)`. Exactly one call per step blocks the
host on the GPU: `FlashInferMLASparseSM90Builder._kv_lens_host` (the production-mounted
`flashinfer_mla_sparse_sm90.py`) needs the exact per-row KV lengths on the host for flashinfer's `plan()`, and under
async scheduling it takes them with `cam.seq_lens[:num_reqs].cpu()`, a pageable D2H `cudaMemcpyAsync` that waits for
everything queued on the stream — **the end of step n's drafter** (in the trace it blocks 62.5 ms). Everything the
host still has to do for step n+1 after that runs with the GPU idle:

| R15 rank 0, 39 steps, medians | value |
|---|---|
| host tail: return of the blocking copy → launch of the eager target embedding all-reduce | **1756 µs** |
| GPU work issued in that tail (36 memcpy + 47 kernel launches: flashinfer `plan()` copies, the 4 GDN/KDA metadata builds, MRv2 input gathers/scatters, embedding) | 83 ops, 125 µs of GPU time |
| top-level aten time in the tail (the rest is Python) | 1073 µs |

This is the whole of `dec_gaps.py`'s "Memcpy→Memcpy 15.5×/step 525 µs", "scatter/index/memcpy chains ~600 µs" and
"before the target's first kernels ~140 µs": each small op runs immediately on the drained GPU, which then idles
until Python issues the next one. CUDA runtime calls longer than 100 µs in the whole trace: per step exactly this
copy (main thread), one `cudaEventSynchronize` on MRv2's async-output thread (off the main thread) and the two
`cudaGraphLaunch` calls (target 2.17 ms, drafter 0.54 ms of host time; the GPU starts the graph before the call
returns, so they overlap execution); no other blocking sync (`.item()`, `.tolist()`, `synchronize`) on the main
thread.

**The collectives.** 104 per step: 1 eager AR (target embedding), 101 in-graph ARs, 1 eager AllGather (target
logits), 1 in-graph AllGather (drafter logits). Mean per step 4.41 ms inside collectives, 2.27 ms of it beyond the
20.6 µs transfer floor, 1.02 ms of GPU idle right before collective kernels. Almost all of it is at three positions:

| position in the step | mean beyond floor (waiting for rank 1) | mean GPU idle right before | cause |
|---|---|---|---|
| 0: target embedding AR (eager) | 324 µs | 41 µs | rank 1 arrives later: its host tail is longer (below) |
| 1: first AR inside the target graph | 240 µs | 53 µs | NCCL host node at graph start (§4) |
| 92: first AR inside the drafter graph (drafter embedding) | 293 µs | **507 µs** | NCCL host node at graph start (§4) |
| all other 101 | ~1–10 µs each (1.29 ms total) | ~4 µs each | GPU speed difference between the two nodes |

`dec_gaps.py`'s "elementwise→AllReduce 2×/step 551 µs" = positions 0 and 92 (41 + 507 µs).

**Two ranks** (the only two-rank pair with CUDA activity on both: 01:00 today, an earlier build; the R15 rank-1
trace is CPU-only, §6). Both ranks leave the blocking copy together (rank 1 − rank 0: **+4 µs**), rank 1 launches
the eager embedding all-reduce **+298 µs** later (host tail 1749 µs on nodeA, 2041 µs on nodeB): the skew at the
target start is nodeB's slower host tail, and rank 0 waits for it at position 0 (959 µs mean in that build; arrival
difference median 306 µs). At positions 1 and 92 **both** ranks idle before the kernel (77/188 µs and 344/446 µs).

## 2. How rank 1 gets its per-step input (item 2)

`v1/executor/multiproc_executor.py`: the head's EngineCore enqueues every `execute_model` / `sample_tokens` call into
one `MessageQueue` (`distributed/device_communicators/shm_broadcast.py`) with `n_reader=2, n_local_reader=1`. Rank 0
reads a shared-memory ring (`SpinCondition`, production `GLM53_SPINWAIT_MS=16`), rank 1 (nodeB, `--headless`) a zmq
XPUB/SUB TCP socket (`MessageQueue.recv`: `poll` then `recv_multipart`, no spinning). `worker_busy_loop` dequeues
and runs the calls in order. With async scheduling and the batch queue (`EngineCore.step_with_batch_queue`),
`execute_model(n+1)` is enqueued right after step n−1's output is processed, i.e. ~1 step (~60 ms) before the
worker asks for it, and the worker is never waiting in `dequeue` in steady decode.

Bound (`tests/bench_mq_broadcast.py`, the production MessageQueue with the spinwait patch, both readers on nodeC,
rank 1 over TCP loopback, 948-byte pickled decode `SchedulerOutput`): a reader idle for one decode step needs
~1.0 ms to wake on **either** path (cold CPU), the ranks differ by < 0.1 ms (`bench_mq_broadcast.log`). The traces
confirm it is off the critical path: +4 µs between the ranks leaving their per-step sync. Nothing to fix here; the
rank-1 lateness is its host tail, hidden by §3.

## 3. `GLM53_DEC_HOSTLOOP=1`: no full-stream sync in the planner

The values the `.cpu()` returns do not depend on the drafter. `seq_lens(n+1)[i] = num_computed_tokens.gpu[idx_i] +
query_len_i` (`_prepare_pos_seq_lens_kernel`), and `num_computed_tokens.gpu` is written only by `post_update`
(sampling of step n, **before** the drafter) and by `add_requests`' staged writes (new/resumed requests, whose value
the host knows exactly). So:

- `GPUModelRunner.postprocess_sampled` (after `post_update`) queues a 4-byte-per-request D2H copy of
  `num_computed_tokens.gpu` into a pinned buffer + an event — both on the stream before the drafter;
- `add_requests` records the slots it (re)filled since that snapshot (their device value = the host value);
- `prepare_inputs` stashes the step's batch layout (cleared when `execute_model` returns);
- `_kv_lens_host`, when the metadata is this step's (same `seq_lens` buffer, same `query_start_loc`), waits for the
  snapshot **event** (ready when step n's sampling finished, ~6 ms before the drafter ends), rebuilds `seq_lens`
  on the host and runs production's own `_kv_lens_host` on it (`.cpu()` of a CPU tensor is a no-op): the per-row
  lengths given to `plan()` are the same integers.

The host then does `plan()`, the GDN builds, the embedding all-reduce launch and the target graph launch while the
drafter runs; the target starts right after the drafter (+ the 125 µs of small ops, now back-to-back).

**Safety.** Anything unexpected — other metadata, no snapshot, a stash that does not match, a value outside
[0, host upper bound], adaptive verification or PCP active, or a vLLM whose relevant functions differ from the
fingerprinted ones (7 functions, `VERIFIED_FINGERPRINTS`, taken from the production image with the production
mounts; the overlays do not touch these files: checked against the overlay-applied tree) — takes production's own
path for that call, counted in `STATS`, never an error. The first `GLM53_DEC_HOSTLOOP_VERIFY` (64) fast calls and
then 1 in `GLM53_DEC_HOSTLOOP_VERIFY_EVERY` (1024) also run production's copy and compare the whole vector; any
difference → WARNING and the fast path is switched off for the process. The plan inputs are identical integers on
both paths, so the TP ranks' decisions cannot diverge (only when each host waits changes; no collective changes).
Nothing is captured into CUDA graphs (the snapshot is skipped during capture).

**Correctness** (`tests/test_hostloop.py`, `test_hostloop_s0.log`): the REAL MRv2 methods (finish/add/update
requests, gather_batch_req_state, prepare_inputs, postprocess_sampled, RequestState, InputBuffers, the triton
kernels) on a runner with stub collaborators, randomized async-spec-decode schedule (adaptive K ∈ {4,5,7}, random
acceptance, chunked prefills mixed with decodes, finish/preempt/re-add with slot reuse, FULL-graph request padding,
contexts across index_topk 2048, kpool 4, optimistic scheduler num_computed exactly like async scheduling), 3 seeds ×
400 steps: host seq_lens == device seq_lens and production `_kv_lens_host` on the fast path `torch.equal` to the
original on **every** step (1200/1200 fast); host time of the fast call 0.79 ms median with a 30 ms "drafter"
queued behind the snapshot vs 29.5 ms for the original. Negative controls: dropping the fresh-slot bookkeeping is
caught by verification and switches the path off (N1); foreign metadata / missing snapshot → production path
(N2/N3). **Numerics: bit-identical** (same integers into `plan()`, nothing else changes). Plugin wiring
(`tests/test_hostloop_plugin.py`, P1–P6): env unset → nothing patched; on → patched at the worker's import of the
model runner, no CUDA context from the plugin; fingerprint mismatch → WARNING, not installed.

**Speed** (`tests/bench_hostloop.py`, `bench_hostloop_prodshape.log`): the real MRv2 prologue + the real
production `_kv_lens_host` + the real flashinfer `plan()` through production's `_SM90State` (32 heads/rank, fp8 KV,
top-k 2048), the other builders as host busy time, target/drafter as CUDA graphs; ABBA-interleaved rounds, medians:

| config | production gap / step | GLM53_DEC_HOSTLOOP gap / step | saving |
|---|---|---|---|
| target 62, drafter 6.0, builders 1.5 ms, 1 req | 1.668 / 65.495 ms (p90 gap 1.688) | 0.044 / 63.884 ms (p90 0.047) | **1.61 ms/step** |
| same, 4 req | 2.876 / 66.495 ms (p90 3.573) | 0.044 / 63.674 ms (p90 0.047) | **2.82 ms/step** |
| drafter only 2.0 ms, 1 req | 1.745 / 61.932 ms | 0.142 / 60.352 ms | 1.58 ms/step |

**Projection (R15 prose):** the rank-0 idle 1756 − 125 = 1.63 ms + the position-0 wait for rank 1's longer tail
0.32 ms leave the critical path → **≈ −1.9 ms/step** (the 125 µs of small ops stay, now back-to-back). Condition:
the host tail (≈ 2 ms on nodeB) stays shorter than the drafter (≈ 6 ms), true at batch 1–4 (above).

Memory: one pinned int32 host vector of `max_num_reqs`; no GPU memory.

## 4. `GLM53_DEC_HOSTLOOP_WAKE=auto`: the cold NCCL host node at every graph start

NCCL 2.30.7 (`src/enqueue.cc`, the production `LD_PRELOAD` version): a collective captured into a CUDA graph with
network proxy ops gets a **host node** (`cudaLaunchHostFunc(hostStream, hostStreamPlanCallback)`) that posts its
proxy ops, and the collective kernel depends on it ("Make to-be-launched kernels dependent on just-launched host
stream tasks"). The host nodes of a graph form a chain that starts at the graph start (`misc/strongstream.cc`: the
capture stream joins "without any dependencies"), so the first collective of each graph waits for the CUDA driver's
host-callback thread to wake up. On GB10 that is slow:

| probe (nodeC, `tests/probe_hostnode*.py`) | cold host-node latency |
|---|---|
| host node after 2 ms of GPU work, any host-thread state (event / pageable copy / stream sync / Python busy / sleep) | 340–545 µs median (two runs), p90 0.6–1.0 ms |
| context schedule flag auto / spin / blocking-sync | no change (195–570 µs at ≥ 1 ms spacing) |
| host nodes 20–100 µs apart ("hot" thread) | 36–40 µs |
| a SCHED_IDLE busy process on **every** CPU (no core can idle) | **78 µs** |

GB10's cpuidle state LPI-3 has a **433 µs exit latency** (LPI-2 231 µs; `/sys/devices/system/cpu/cpu0/cpuidle`)
and is used heavily. The host-node wake-up = the GPU interrupt's CPU (nvidia MSI-X vectors: CPU 0 and 12 on nodeC,
`/proc/irq/*/nvidia`) + the callback thread's CPU coming out of deep idle. In the R15 decode that is the 507 µs idle
+ 293 µs wait at position 92 and the 240 µs at position 1 (§1), paid by both ranks every step.

`GLM53_DEC_HOSTLOOP_WAKE=<cpus>|auto` starts, at the first real decode step, one helper process with one SCHED_IDLE
thread per listed CPU that sleeps `GLM53_DEC_HOSTLOOP_WAKE_TICK_US` (100 µs) in a loop — the core never predicts a
long idle, stays in WFI — and pins the host-callback thread (found with one no-op `cudaLaunchHostFunc` on a private
stream; the test proves it is the same thread that runs graph host nodes) to the first CPU
(`GLM53_DEC_HOSTLOOP_WAKE_PIN=0` to not pin). `auto` = the effective CPUs of all nvidia vectors (8 on nodeC:
0,11–17; `/proc/interrupts` is masked in containers, `/proc/irq` is readable). Cost: 14.6 % of one core in total,
10 MB RSS; the helper dies with the worker (`PR_SET_PDEATHSIG`); SCHED_IDLE threads yield to every normal thread.
No numerics, no GPU memory, no collective change; if the helper dies, wake switches itself off (logged).

Paired ABBA (`tests/bench_hostloop_wake.py`, graph = sleep 2 ms → host node → sleep 0.1 ms, 192 samples per arm):

| arm | median | p90 | p99 | mean |
|---|---|---|---|---|
| production | 441 µs | 827 | 1221 | 478 |
| WAKE=auto (tickers + pin) | **89 µs** | **149** | 228 | **111** |
| WAKE=auto, WAKE_PIN=0 (other run; its production arm: mean 424) | 187 µs | 247 | 341 | 190 |
| WAKE=12,0 tickers / busy spin | 137 / 137 µs | 154 / 176 | 299 / 420 | 120 / 135 |

Tests (`tests/test_hostloop_wake.py`, W1–W4 PASS): same thread for the probe host function and graph host nodes,
pinned; one helper with SCHED_IDLE threads on the listed CPUs; the helper dies with its parent; a dead helper
switches wake off. Plugin P5/P6: wake alone patches only `prepare_inputs` and starts nothing at import; a bad value
→ WARNING, nothing patched.

**Projection:** position 92 (800 µs of idle + wait) → max over the two ranks of ~110–150 µs; position 1 (~290 µs) →
mostly hidden behind the ~0.4 ms of compute before it: **≈ −0.7 ms/step (range 0.4–0.9)**. The production cores are
busier than nodeC's (engine-core spinwait, NCCL proxies, API server), so the real baseline may be better or worse:
measure (§8). A system-level alternative with the same effect (owner/root, not done): disable cpuidle LPI-2/3 on
nodeA/nodeB (`/sys/devices/system/cpu/cpu*/cpuidle/state{2,3}/disable`).

## 5. `GLM53_DEC_HOSTLOOP_METER=N`: per-rank host-loop and skew meter

Independent of the other knobs. CUDA events on the model runner's stream at `StepTimingCollector.forward_start`,
after `sample_tokens` and around every **eager** TP collective (`GroupCoordinator._all_reduce_out_place` /
`_all_gather_out_place`; production decode has two: c0 = target embedding AR, c1 = target logits AllGather; the
other 102 are inside CUDA graphs). Every N real decode steps each rank logs
`[glm53-hostloop] meter rank R (N steps, hostloop=on|off|switched-off): gap_ms …; step_ms …; hwait_ms …; c0_ms …;
c1_ms …; stats=…` — gap = GPU time from the end of step n's drafter to step n+1's forward start, step = forward to
forward, hwait = host time inside `_kv_lens_host`, cJ = collective duration (the rank with the longer c0 arrived
first; the difference of the two ranks' c0 medians = the skew at the target start). Validated against the bench's
own events (`bench_hostloop_meter.log`: gap 1.411 vs 1.426 ms, 0.044 vs 0.044 ms; a 50 µs eager op reads 0.055 ms).
Skipped during graph capture; events are pooled (a handful of event records per step).

## 6. `GLM53_DEC_PROF_DIAG=1`: why the rank-1 trace was CPU-only (item 3)

The nodeB worker's R15 trace (`p6w`, 06:30) has 67760 `cpu_op` events and **no** `cuda_runtime` or kernel events
(the head's trace of the same moment and the same worker type's 01:00 trace are complete), with a 6.37 s stall
inside its first traced step. So CUPTI recorded nothing in that session; kineto/CUPTI print the reason only to the
worker's stderr. The profiler records kernels in every in-process situation tried on nodeC
(`tests/probe_profiler_cupti.py`: a first and a second session, a side thread blocked in event syncs, CUDA-graph
replays, glm53_runtime's SIGUSR2 path; the probe's second SIGUSR2 trace was lost to the same-second file-name
collision fixed below; with the fix, preflight + SIGUSR2 = two sessions both with kernels, `test_prof_diag` D1).
What differs on nodeB cannot be read from here (no ssh by rule), so the fix makes the failure explain itself and
retries:

- boot preflight per rank (after warm-up): one tiny session, logs `[glm53-prof] rank R preflight: CUDA activity
  recorded (…)` or a WARNING `NO CUDA activity recorded` with kineto/CUPTI's stderr lines (captured at fd level) and
  the context (free device/host memory — CUPTI allocates device buffers and nodeB's free memory is tight — the mapped
  CUPTI libraries, CUPTI env knobs, `CUDA_INJECTION64_PATH`, driver). It also initializes CUPTI at boot instead of
  inside a traced step;
- every SIGUSR2 session: kineto stderr captured around start/stop, device events counted; zero → WARNING with the
  context and **one automatic re-armed session**;
- file names `rank<torch.distributed rank>-<pid>-<time>-s<k>` (both ranks were `rankx`; two sessions in the same
  second overwrote each other).

`tests/test_prof_diag.py` D1–D3 PASS (normal; simulated CPU-only session → 2 warnings, exactly one retry; diag off →
old names, no preflight). With both traces complete, `tests/skew_from_traces.py` gives the per-collective skew.

## 7. Considered and not done

- Pinned/merged H2D copies, moving builder work into the graph: with §3 all of it runs while the drafter runs; the
  remaining GPU cost is the 125 µs of small ops, not worth an invasive change.
- Async scheduling: already on (MRv2 + batch queue 2); the executor's broadcast is off the critical path (§2).
- Removing NCCL's graph host nodes (proxy ops need them on the IB transport; would need an NCCL change) or starting
  them earlier (needs the drafter's eager prefix inside its graph). A pre-warm host node on a side branch was probed
  (`probe_hostnode3.py`): not robust (worse at 1–2 ms lead). §4 attacks the latency itself.

## 8. Enable on BOTH ranks, what to measure, revert

`start.sh` (the head passes only its explicit `-e` list; the worker gets `serve_env` built from `serve_env_names`):

1. add to the `for v in … ; do serve_env+=…` list (≈ line 2181, next to `GLM53_MEM_HYGIENE GLM53_TF_PROFILE …`):
   `GLM53_DEC_HOSTLOOP GLM53_DEC_HOSTLOOP_VERIFY GLM53_DEC_HOSTLOOP_VERIFY_EVERY GLM53_DEC_HOSTLOOP_METER
   GLM53_DEC_HOSTLOOP_WAKE GLM53_DEC_HOSTLOOP_WAKE_TICK_US GLM53_DEC_HOSTLOOP_WAKE_PIN GLM53_DEC_PROF_DIAG`
2. add to the head `docker run` (≈ line 2380, next to `-e GLM53_TF_PROFILE_STEPS=…`):
   `-e GLM53_DEC_HOSTLOOP="${GLM53_DEC_HOSTLOOP:-}" -e GLM53_DEC_HOSTLOOP_VERIFY="${GLM53_DEC_HOSTLOOP_VERIFY:-}"
   -e GLM53_DEC_HOSTLOOP_VERIFY_EVERY="${GLM53_DEC_HOSTLOOP_VERIFY_EVERY:-}"
   -e GLM53_DEC_HOSTLOOP_METER="${GLM53_DEC_HOSTLOOP_METER:-}" -e GLM53_DEC_HOSTLOOP_WAKE="${GLM53_DEC_HOSTLOOP_WAKE:-}"
   -e GLM53_DEC_HOSTLOOP_WAKE_TICK_US="${GLM53_DEC_HOSTLOOP_WAKE_TICK_US:-}"
   -e GLM53_DEC_HOSTLOOP_WAKE_PIN="${GLM53_DEC_HOSTLOOP_WAKE_PIN:-}" -e GLM53_DEC_PROF_DIAG="${GLM53_DEC_PROF_DIAG:-}"`
3. the env file: `GLM53_DEC_HOSTLOOP=1`, `GLM53_DEC_HOSTLOOP_WAKE=auto`, and for the measurement runs
   `GLM53_DEC_HOSTLOOP_METER=200`, `GLM53_DEC_PROF_DIAG=1`. Install `glm53_hostloop.py` with the other fork modules
   (it is in `setup.py`'s `py_modules`).

Production measurement (the numbers above are nodeC emulations; these decide):

1. A = R15 + `METER=200` (+ `PROF_DIAG=1`), B = A + `GLM53_DEC_HOSTLOOP=1`, C = B + `WAKE=auto`; same prompts,
   bench_decode prose / structured / coding tok/s and ms/step for each.
2. From both ranks' logs: meter `gap_ms` (expect ~1.8 → ~0.15 ms with B), `step_ms`, `c0_ms` on each rank (rank 0's
   c0 should fall from ~0.3 ms), `stats` (`fast` rising, `mismatch` 0, `verified` ≥ 64, fallbacks ≈ 0) and no
   `fast path switched OFF` / `NOT installed` warnings; for C: `wake: … tickers on cpu […]`,
   `host-callback thread … pinned`, no `wake switched off`.
3. One SIGUSR2 on both workers per arm → `tests/skew_from_traces.py rank0 rank1`: positions 0/1/92 of the table in
   §1 (C should cut the idle-before at 92 from ~0.35–0.5 ms to ~0.1 ms on both ranks). Rank 1's preflight line tells
   whether its trace will have CUDA activity; if not, its WARNING says why.
4. Numerics: B is bit-identical by construction (same integers into `plan()`); the long-context KL probe should equal
   A's exactly. C and the meter do not touch numerics. Check CPU/SoC power and GPU clocks under C (14.6 % of a core).

Revert: unset the variables (or `=0`) and restart; nothing persists. Runtime switches: B switches itself off on the
first verification mismatch; C on a helper failure.

## 9. Files

`glm53_hostloop.py` (fast path, meter, wake), `glm53_runtime.py` (`GLM53_DEC_PROF_DIAG`), `integrate.py`
(install), `setup.py`; tests: `test_hostloop.py`, `test_hostloop_plugin.py`, `bench_hostloop.py`,
`test_hostloop_wake.py`, `bench_hostloop_wake.py`, `probe_hostnode{,2,3,4}.py` + `hostnode/` (C host function built
on demand), `probe_hostloop_copy.py`, `probe_profiler_cupti.py`, `test_prof_diag.py`, `bench_mq_broadcast.py`,
`hostloop_trace_ana.py`, `skew_from_traces.py`, `hostloop_gpu.sh` (production mounts + the shared GPU lock),
`gpu_run.sh` (`GPU_RUN_BIND_DIR`, `GPU_RUN_ENV`). Logs: `docs/logs/hostloop/`.

## 10. Adversarial review (2026-09-28, nodeC, production image + production mounts)

Logs: `docs/logs/hostloop/review/`. Everything below was re-run by the reviewer, not copied.

**Correctness of `GLM53_DEC_HOSTLOOP` (bit-identical, confirmed).**
- Writers of `num_computed_tokens.gpu` in the production vLLM tree (image + every overlay in `GLM53_OVERLAY_ORDER` and
  the tf bundle; none of them edits `model_runner.py`, `input_batch.py`, `states.py`): `post_update` (sampling, before
  the drafter), `post_update_num_computed_tokens` (non-last PP rank / pooling only), `add_request`'s staged write.
  The DFlash2 speculator, `MambaHybridModelState.preprocess_state/postprocess_state` and `update_requests` (host mirror
  only) do not write it; resumed requests come back as new requests under MRv2 (fresh slots). GLM-5.3 is `IsHybrid`
  -> `MambaHybridModelState.prepare_attn` builds the metadata **without** `positions`, so production's
  `_kv_lens_host` takes the `seq_lens.cpu()` branch that the fast path replaces (fast path is reachable in production).
- `tests/test_hostloop.py` re-run for seeds 1-9 (3 x 1200 steps): all equal, N1-N3 caught (`test_hostloop_seeds1-9.log`).
- New `tests/review_hostloop_async.py`: a truly asynchronous host loop (no per-step sync at all, sampling inputs via
  pinned non-blocking copies, device truth captured by a stream-ordered D2D copy of `seq_lens`): 4 seeds x 600 steps,
  2400/2400 fast, 0 mismatches against production's `_kv_lens_host` on the device values (`async_wake_plugin_diag.log`).
- R15 trace (`tests/review_trace_tail_syncs.py`): between the blocking copy and the target `cudaGraphLaunch` the main
  thread issues no other D2H copy and no other sync (the only `cudaStreamSynchronize` is the `.cpu()` itself), so no
  second blocking point appears once the first is removed.

**Speed (reproduced, paired ABBA, 270 steps per arm).** `bench_hostloop.py` target 62 / drafter 6 / builders 1.5 ms:
1 req step 65.415 -> 63.794 ms (-1.62), 4 req 66.691 -> 63.885 (-2.80), longer tail 2.5 ms + drafter 4 ms: -2.62.
From the R15 rank-0 trace (`tests/review_trace_gap.py`): drafter end -> first target-graph kernel median 1.94 ms,
mean 2.15 ms per step; with the fast path ~0.15-0.2 ms remain (small ops + eager AR floor) -> **-1.75 (median) to
-1.95 (mean) ms/step**, consistent with the -1.9 projection.

**`GLM53_DEC_HOSTLOOP_WAKE`.** Latency reproduced (`bench_hostloop_wake auto`: mean 444 -> 130 us and 422 -> 123 us).
GPU cost of the tickers (new `tests/bench_wake_gpu_bw.py`, 9.7 GB bandwidth-bound graph, 480 replays per arm): WAKE
-0.16 % / -0.04 %, A/A control +0.06 % / +0.03 % -> no measurable bandwidth cost on nodeC.
**Risk found: pinning.** With one normal-priority busy thread on CPU 0 (the pin target of `auto`; in production the
worker's spinning main thread, NCCL proxy threads or the engine core can land there), the pinned callback thread
shows **p99 3.0-3.1 ms** (two runs; one scheduler slice) and mean 125 / 271 us, while `WAKE_PIN=0` gives p99 291 /
332 us, mean 107 / 175 us. Production measurement should include an arm C' = C + `GLM53_DEC_HOSTLOOP_WAKE_PIN=0`
and compare step_ms p99, not only medians.

**Fixes committed by the review.**
- `_wake_step`: when wake switches itself off (helper died), the host-callback thread was left pinned to one CPU with
  no tickers; it now gets its previous affinity back (`W4` checks it).
- `glm53_runtime._diag_on`: `GLM53_DEC_PROF_DIAG=0` (or off/false/no) enabled the diagnostics (any non-empty value
  was on), contrary to "revert = unset or =0"; now off like the other knobs.
