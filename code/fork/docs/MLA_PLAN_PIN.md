# MLA_PLAN_PIN - page-locked staging for production's sparse-MLA plan (GLM53_MLA_PLAN_PIN)

Site module `glm53_mla_planpin.py` (r16z8p). Knob `GLM53_MLA_PLAN_PIN` (env.r16 `#knob`, `unset|0|1`, default unset =
production byte for byte). Forwarded to BOTH ranks by the kit's start.sh; boot_checks compares head and worker.

## 1. The defect (2026-10-05 production profile; evidence ${HOME}/tf-exl3-assets/prof-1005/ana/)

Production's FLASHINFER_MLA_SPARSE_SM90 backend is the launcher-mounted `vllm-patches/flashinfer_mla_sparse_sm90.py.patched`
(sha256 526399d8; nodeA's head mount, nodeB's worker copy `${WORKER_HOME}/flashinfer_mla_sparse_sm90.py.patched` and the
nodeC copy are identical, read-only check 2026-10-05). `_SM90State.plan` replans flashinfer's
`BatchMLAPagedAttentionWrapper` (flashinfer 0.6.18, fi618 `mla/_core.py` sha256 58d06e61, identical on all three) once
per metadata build from three cached CPU staging tensors allocated **pageable** (lines ~253-257):

    _qo_cpu, _kv_cpu   max_tokens + 1 int32   (qo / kv indptr)
    _lens_cpu          max_tokens int32       (per-row kv lengths)

With `use_cuda_graph=True` the wrapper copies them into its fixed device buffers with `copy_(non_blocking=True)`.
Production runs max-num-batched-tokens 16384, so the indptr copies are 16385 x 4 = **65540 B, 4 B over CUDA's 64 KiB
limit for asynchronous pageable H2D copies**: such a copy synchronizes the host with the stream first. Each plan()
therefore blocks until the GPU has drained everything queued before it - the previous step's DFlash2 drafter graph
(~5.7 ms every step on nodeA) - and afterwards the GPU idles while the host finishes the rest of the step's host work.

Replication (nodeC, `prof-1005/ana/verify/nodeC/h2d_graph.log`): host call time of `copy_(non_blocking=True)` behind a
7.6 ms graph: pageable 16384 int32 (65536 B) 4.9 us, **pageable 16385 int32 (65540 B) 7584 us**, pinned 16385 int32 3.1 us.

## 2. The change (GLM53_MLA_PLAN_PIN=1)

`_SM90State.plan` is replaced - only after the sources it replaces and relies on match production's fingerprints
(`_SM90State.plan` 5c9836599374a0ec, `BatchMLAPagedAttentionWrapper.plan` bcadfb24d6959d4a; sha256 of `ast.dump`, as
glm53_mla_exactlens; anything else -> `NOT installed`, production path) - by the same statements with page-locked staging
in a **2-slot ring**. Each slot owns:

* pinned `qo` / `kv` (max_tokens + 1) and `lens` (max_tokens) staging tensors;
* its own **page-locked int workspace** for flashinfer's planner (slot 0 = the wrapper's own
  `_pin_memory_int_workspace_buffer`, slot 1 a second 8 MiB pinned buffer);
* one CUDA event, recorded on the current stream right after the wrapper's plan (= after its last async H2D copy).

Before a slot is rewritten its event must have completed (`event.synchronize()` only if `query()` says it has not; counted
as a "ring wait"). Values are computed exactly as production computes them (clamp / mul / fill / slice-assign), then
`wrapper.plan(...)` gets the slot's tensors and the slot's int workspace. `kv_indices` (the device top-k buffer) is
untouched, as in production. Production's attribute names `_arange_cpu/_qo_cpu/_kv_cpu/_lens_cpu` keep pointing at the
latest plan's staging. Inside a CUDA-graph capture the patched plan raises production's RuntimeError.

Fallback (a failed first-plan self-test or ring allocation): the process switches back to production's plan **as
production runs it** - every slot's pending copy is waited for, each wrapper gets its own page-locked int workspace back
(slot 0's) and the production attribute names are cleared (`_arange_cpu = None`), so production's plan re-allocates its
PAGEABLE staging on the next call. (Review fix 2026-10-05: before it, the fallback left production's plan staging in the
last slot's pinned tensors with the shared int workspace and no ring, i.e. the unprotected naive pinning of 2.1 -
`tests/r16z8p/test_planpin_adv.py` V1: 23 / 24 wrong snapshots under the back-to-back stress through that fallback at
max_tokens 16384, 0 / 24 with the fix.)

### 2.1 Why the int workspace is in the ring (the hazard a plain pin_memory=True would create)

flashinfer's `MLAPlan` (fi618 `include/flashinfer/attention/scheduler.cuh` ~1581-1837) writes the work schedule into the
wrapper's page-locked int workspace ON THE HOST and then `cudaMemcpyAsync`s it to the device int workspace, **without a
sync**. Today the pageable 65540 B indptr copy of the NEXT plan synchronizes the host with the stream before that next
plan rewrites the page-locked schedule - by accident, that implicit sync is what keeps the previous schedule copy safe.
With pinned staging the sync is gone: a second plan before the GPU reached the first plan's copy would overwrite the
schedule (and the staging) that copy still has to read, so the forward of the first plan would run with the second plan's
schedule / indptr (wrong attention or an illegal address when the shapes differ). Measured (P5 below): pinning only the
three tensors ("naive") gives **23 / 24 wrong stream-ordered snapshots** in a back-to-back stress at max_tokens 16384;
the ring gives 0 / 24. Production itself has the same latent schedule hazard whenever its copies are <= 64 KiB (no
implicit sync): at max_tokens 7168 production's own plan gives 23 / 24 wrong snapshots in the same stress, all in the int
workspace - the ring removes that too. In steady decode only one plan runs per step and the previous plan's copies have
long completed (bench: ring waits 0), so the ring costs nothing there and protects every other schedule
(several builds per step, prefill chunks, a host running ahead).

The int workspace bytes beyond each work array's written length (each array is allocated for 16384 works, only the
first `total_num_works` entries are written) keep stale bytes of an earlier plan - in production of the previous plan,
with the ring of the plan two calls earlier. No kernel reads them (the work ranges come from `work_indptr`); the exactness
checks below compare every byte the kernels read.

### 2.2 Cost

+6 pinned staging tensors (2 slots x (2 x 65540 + 65536) B at max_tokens 16384) and +8 MiB page-locked host memory (the
second slot's int workspace) per process that owns an `_SM90State` (each TP rank's worker). No device memory.

## 3. Logs (every line printed by glm53_mla_planpin.py)

* every vLLM process: `glm53_mla_planpin plugin loaded (pid P): GLM53_MLA_PLAN_PIN=<repr> -> off, production's pageable
  plan staging unchanged` | `... -> installing (mode on)`; a refused value: `... -> off (GLM53_MLA_PLAN_PIN must be ...)`
* at the backend import: `glm53_mla_planpin: patched _SM90State.plan (pid P): page-locked staging, 2-slot ring with a
  per-slot CUDA event and int workspace (sources verified: ...) PROOF mode=on` | WARNING `glm53_mla_planpin: NOT installed: ...`
* first plan of the process: `glm53_mla_planpin: rank R self-test: 3/3 device plan buffers == pinned staging (max_tokens
  16384: indptr 65540 B, lens 65536 B per slot page-locked, 2 slots, +8.4 MiB pinned) PROOF mode=on` | WARNING
  `... self-test FAILED (...): production's pageable plan staging from now on`; a failed ring allocation: WARNING
  `... ring allocation failed (...): production's pageable plan staging from now on`
* 64th plan: `[glm53-mla-planpin] rank R serving confirmed (mode on): N plans through the pinned ring, ring waits W`
  (the A/B PROOF; the boot's capture runs usually reach it), then a stats line every 100000 plans.

boot_checks (r16z8p rows): head == worker for the value; off -> the plugin's off line on both ranks, no patch line;
on -> plugin + patch + self-test on both ranks, never NOT installed / not loaded / ring allocation failed / self-test
FAILED; `--after-traffic` -> the serving line on both ranks.

## 4. Verification (nodeC, production image + the production mounts via tests/hostloop_gpu.sh)

`tests/r16z8p/test_planpin.py` (logs docs/logs/r16z8p/test_planpin.log, test_planpin_run1.log): ALL OK (both runs)

| row | result |
|---|---|
| P0 fingerprints | `_SM90State.plan` 5c9836599374a0ec, `BatchMLAPagedAttentionWrapper.plan` bcadfb24d6959d4a == VERIFIED |
| P1/P2 env + off | unset/''/0 off, 1 on, 6/6 other values refused; off: class untouched, off line logged |
| P3 install | patched through the already-imported module, production's plan kept as `_glm53_orig`, idempotent |
| P4 sequential exactness | max_tokens 16384 and 7168, 40 plans each (decode K+1 rows, prefill chunks, full 16384, edge sizes): device qo/kv indptr, kv_len_arr, `_plan_info` and every kernel-read int-workspace byte identical to production's plan after every plan (40/40 each) |
| P5 async race stress | GPU kept behind the host (spin after every plan, no host sync), stream-ordered snapshots == the synchronous reference: ring 0/24 (16384) and 0/24 (7168), ring waits 22 each; production pageable 16384 0/24; naive pinning 23/24 wrong (the stress detects the hazard); production pageable 7168 23/24 wrong (latent, int workspace only) |
| P6 stream ordering | the plan's int-workspace and indptr copies queue behind a running kernel on torch's current stream (default and a side stream): the per-slot event covers them |
| P7 capture / fallback | inside a capture: production's RuntimeError; module disabled: production's plan |
| P8 logs | self-test line at the first plan, serving-confirmed at 64 plans |

`tests/r16z8p/test_planpin_adv.py` (review, logs docs/logs/r16z8p/review_test_planpin_adv.log): ALL OK

| row | result |
|---|---|
| V1 fallback after the ring served | injected self-test failure, then the back-to-back stress at 16384 through the fallback: 0/24 wrong, staging pageable again, the wrapper's own int workspace back (be04148's fallback: 23/24 wrong, staging still pinned) |
| V2 CUDA-graph replay reader | a captured graph reads the fixed device plan buffers after each of 32 back-to-back plans (2 per step, mixed decode / prefill / full sizes, GPU behind the host): ring 0 wrong at 16384 (waits 15) and 7168 (waits 25); production 16384 0 wrong; production 7168 31 wrong (its latent race) |
| V3 leak | 60000 plans without host sync: RSS +40 kB, CUDA allocated +0 B, pinned host allocator unchanged |

`tests/r16z8p/bench_planpin.py` (logs docs/logs/r16z8p/bench_planpin_run1.log, bench_planpin.log; REAL MRv2 prologue + REAL `_kv_lens_host` with
GLM53_DEC_HOSTLOOP's fast path + REAL flashinfer plan; target / drafter = 20.0 / 5.1 ms CUDA-graph spins; ABC-interleaved,
8 rounds x 30 steps per arm):

| arm | gap drafter -> forward p50 | step p50 | plan() host p50 |
|---|---|---|---|
| A max_tokens 16384, production (pageable) | 1.332 ms | 24.899 ms | 4.725 ms |
| B max_tokens 16384, GLM53_MLA_PLAN_PIN=1 | 0.049 ms | 23.620 ms | 0.147 ms |
| C max_tokens 7168, production (no-stall reference) | 0.045 ms | 23.611 ms | 0.089 ms |

B - A: **-1.28 ms/step** (gap -1.28 ms), B == C within 0.01 ms; ring waits 0 over 240 steps; kernel-visible plan state A vs B
8/8 identical; the indptr `copy_` host call itself with a drafter graph queued: pageable 4.79 ms vs pinned 0.003 ms. Second
run on the shipped module bytes: A gap 1.394 / step 24.666 ms, B 0.049 / 23.317 ms, C 0.044 / 23.301 ms: **-1.35 ms/step**,
8/8 identical. (The profile analysis' estimate on production: -1.1 ms/step, range -0.9 .. -1.6.)

Engine smoke (the r16z8p kit composed byte for byte, real vLLM engine, DFlash2 drafter, mini target, max_num_batched_tokens
16384): pin and all arms rc 0, 128/128 tokens, outputs inside the rig's A/A band, every planpin boot / serving line present,
no failure line (DEPLOY_R16Z8P.md section 5; logs docs/logs/r16z8p/engine_*).

## 5. Not covered / risks

* Speed on production is measured only by the A/B (nodeC has one GB10; the step gain depends on production's drafter
  length and host work after plan()).
* A future flashinfer or SM90 patch with different sources is refused by the fingerprint check (production path, WARNING
  `NOT installed`, boot_checks MISS when the knob is on).
