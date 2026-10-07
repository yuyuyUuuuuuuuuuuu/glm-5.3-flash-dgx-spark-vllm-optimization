# MOE_E4M3 — the routed-MoE prefill on e4m3 tensor cores (`GLM53_MOE_E4M3`)

Status 2026-10-01, branch `moee4m3` (on poolfix / r16l). Built and measured on nodeC only (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, every GPU run through `tests/gpu_run.sh` under
`flock /tmp/tf-gpu-bench.lock`), against both production module versions (the image's `quantization/exl3.py` and the
launcher overlay `docs/ref/prod_live/overlay_exl3.py` that production runs). nodeA/nodeB were not touched; nothing is
installed anywhere. Weights: the real GLM-5.3-Flash EXL3-TR3-4bpw **layer-10 routed experts, TP=2 rank-0 shard**
(`tests/moee4m3/extract_layer.py`, production's own `shard_exl3_col/row` slicing, from the read-only shards in
`${HOME}/tf-exl3-assets/moee4m3-shards/`). Activations and routing are synthetic (routing = the calibrated
`tests/prefill_cap_common.routing` kinds). Every number below is in a committed log under `docs/logs/moee4m3/`.

## 0. Verdict

| | production (E3, prefill cap 1) | GLM53_MOE_E4M3 | delta |
|---|---|---|---|
| T=13824, real routing, `apply_exl3_experts` per layer call | 64.60 ms | **42.68 ms** | **-21.9 ms (x1.51)** |
| T=13824, zipf / collapsed | 64.47 / 64.04 | 42.35 / 41.42 | -22.1 / -22.6 |
| T=4289 (the 32k request's last chunk), real / zipf / collapsed | 21.82 / 22.11 / 21.63 | 17.70 / 17.61 / 17.32 | -4.1 / -4.5 / -4.3 |
| T=1791, real / zipf / collapsed | 13.08 / 13.40 / 12.21 | 12.01 / 12.15 / 10.12 | -1.1 / -1.2 / -2.1 |
| T=512, real / zipf / collapsed | 9.56 / 9.23 / 6.53 | 8.93 / 8.93 / 5.45 | -0.6 / -0.3 / -1.1 |

(`bench_final_image.log`; the launcher-overlay module gives the same: T=13824 real 64.25 -> 43.13 ms, T=4289 21.93 ->
17.86 ms, `bench_final_live.log`. Faster than production in every measured cell, 12 routing x T cells per module.)

**Projection per 13,824-token chunk (42 MoE layers): -0.92 s** at the bench's baseline (21.9 ms x 42), **-0.87 s** if
scaled to production's own traced per-layer time (routed MoE 2,510 ms / 42 = 59.8 ms + thin 0.7 ms vs the bench's 63.0:
x0.96) — i.e. **~12 % of the 7.31 s chunk**. For the profiled 31,937-token request (2 x 13,824 + 4,289): ~-1.9 s of
17.2 s TTFT (~1,856 -> ~2,100 tok/s). [E] (projection, not measured in production.)

Numerics (the gate for shipping is the KL measured with the moefq emulation, see section 6): the kernel IS that
spec — stage by stage >99.999 % of e4m3 bytes identical, end to end within the spec's own fp32-vs-fp64 noise, and
closer to the exact (fp64) spec than the fp32 emulation itself.

## 1. What it computes (the spec)

The numerics of the moefq e4m3 quality emulation (`GLM53_MOE_E4M3_EMU`, branch moefq,
`overlay/glm53_moe_e4m3_emu.py`, vendored verbatim as `tests/moee4m3/emu_spec_copy.py`, sha256 18990ef2...), per
routed (token, expert) row:

```
W8     = e4m3_rn( W_q )                    W_q = trellis decode (kernels/exl3_format_ref.py), NO scale
gate/up input:   v = H(float(x16) * float(suh)) ; s = amax(|v|)/448 (1 if 0) ; q = e4m3_rn_sat(v / s)
                 acc = q . W8  (fp32 accumulate) ; g = H(acc_g * s) * svh_g ; u = H(acc_u * s) * svh_u
SwiGLU:          a16 = fp16( silu(min(g, L)) * clamp(u, -L, L) )                    (torch's statement)
down input:      v = H(float(a16) * float(suh_d)) ; s_d = amax/448 ; q_d = e4m3_rn_sat(v / s_d)
down:            out[token] += ((H(acc_d * s_d)) * svh_d) * w        fp32 atomics, w = the fp32 router weight
return out.to(x.dtype)
```

H = 128-point Walsh-Hadamard / sqrt(128). Differences from the emulation's torch code are fp32 rounding order only:
the scale is applied after the exact e4m3 GEMM instead of before it, H is a butterfly instead of an fp32 matmul, and
the fp32 sums run in a different order. The per-row scale is over the rank-local row (4096 for gate/up, 1024 for down
at TP=2), exactly as the emulation computes it on each rank. Requirements checked per layer: K4 trellis, hidden 4096,
local intermediate 1024, gate and up sharing one input rotation (`_exl3_shared_w13_suh`, true for the checkpoint:
one gathered row feeds both).

## 2. Kernels (`kernels/moe_e4m3.cu`, extension `glm53_moe_e4m3_ext`)

Built without `--use_fast_math` / `-ftz` (IEEE division, expf and subnormal e4m3 like torch's own kernels).

1. **Routing tables** (`glm53_moe_e4m3.plan`, torch + one tiny kernel, no host sync): pairs sorted by local expert
   (stable), pair -> row map, row token / fp32 weight / expert, 128-row segment tables over every expert with rows
   (`seg_tables`), row count. Non-local / invalid experts (expert_map) sort last and are never computed.
2. **Gather** (`gather2`): one block per (token, 2 of its experts), 4 warps per row: x16 * suh (exact in fp32), the
   128-point Hadamard in registers, the row amax combined through shared memory, per-row scale, e4m3 rn/sat, one
   coalesced 32-bit store per lane. Single pass; the block also zeroes its slice of the fp32 accumulator `out`.
   Rows are stored **k-permuted inside every 32-byte group** (the mma k32 fragment is built from two decoded k16
   trellis tiles without moving data between lanes, so mma position p holds weight k = perm(p); the A rows carry
   the same permutation, the dot products are unchanged — header of the .cu).
3. **Fused persistent kernel** (`fused`, the default path): one CTA per SM (16 warps, 128 registers, 81 KB smem).
   - Items: gate/up = (128-row segment, 128 intermediate columns) — warps 0-7 gate, 8-15 up, each warp one 16-column
     block for all 128 rows (each trellis tile decoded once per CTA, feeding 8 row blocks); down = (segment, 256
     hidden columns). A global ticket counter hands out items in the order GU(seg 0..L-1), [8 GU of seg L+r,
     16 DN of seg r] for r = 0.., the last L down segments (L = 12).
   - Mainloop: A (e4m3 rows) through a 4-stage cp.async ring of 128-K stages (one barrier per 4 k32 steps); B
     (trellis words) bypasses shared memory — each lane loads its own words of the next stage into registers;
     decode (TensorFold mcg) -> fp16 -> `cvt e4m3x2` -> `mma.sync.m16n8k32.e4m3.e4m3.f32`. Both pipelines continue
     across items.
   - Gate/up epilogue: scale, Hadamard, svh, clamp, SiLU (IEEE expf / division), fp16 store of a16.
   - Dataflow: the CTA that finishes the 8th gate/up item of a segment quantizes that segment's down-input rows
     (Hadamard of a16 * suh_d, per-row scale, e4m3; a16 read through L2), then releases `ready[seg]`
     (`st.release.gpu`); a down item waits on it (`ld.acquire.gpu`, bounded spin that traps after ~30-60 s rather than
     hanging). Only smaller tickets are ever waited on and all CTAs are resident: deadlock-free.
   - Down epilogue: scale, Hadamard, svh, router weight, `red.global.add.v4.f32` into out.
   The point of fusing: the down's fp32 scatter-add (1.8 GB of contributions per 13.8k chunk-layer) is L2/DRAM bound
   and serialized with compute when the kernels run back to back; interleaved on every SM it overlaps the
   compute-bound gate/up.
4. Separate kernels (`gateup`, `actq`, `down`, `gather`): the same arithmetic, sequential (and a two-stream chunked
   schedule); kept for A/B and diagnosis (`glm53_moe_e4m3.SCHED`).

### How it got here (T=13824 real routing, ms per layer call; logs in `docs/logs/moee4m3/`)

| step | e4m3 path | what changed / what was measured |
|---|---|---|
| v1 (`bench_dev1.log`) | 54.05 | first correct version: gather 6.0, gate/up 21.4 (87 TFLOPS), actq 1.6, down 21.8 |
| anatomy v1 (`anatomy_v1.log`) | | gate/up: epilogue 1.6, copies 4.5, decode 4.8 ms; down: **scatter 12.5 ms** of 22.1 (no-atomics 10.3; balanced routing 6.5) |
| v2 mainloop (`anatomy_v3.log`) | | B through registers + 128-K stages: gate/up 21.4 -> 18.4, down mainloop 10.1 -> 8.3; stage count 3/4/5 irrelevant |
| gather (block per token, coalesced stores; then single pass + zeroing) | | 6.9 -> 4.4 -> 3.9 incl. the 226 MB zeroing of `out` |
| sequential kernels (`bench_dev2.log`) | 47.4 | two-stream chunking with a fixed SM split: worse except 28/20 (44.3) — down needs many SMs for its compute |
| **fused** (`bench_dev3.log` ff.) | **41.0-41.6** | lag 4..32 equal within noise; fused without atomics 35.2, with plain stores 37.9, L2-resident atomics 39.6: the atomics' L2 throughput is what is left |
| tried, no gain | | explicit RED vs ATOM, 2 row blocks per epilogue round, predicate-free full-tile mainloop, 3/5 stages, 64-K stages (worse), column-slab ordering of the down (worse: the e4m3 rows re-read 16x) |

Where the 41 ms go (T=13824 real, measured with removal variants): gather 3.9 (bandwidth: x read once, 453 MB e4m3
+ 226 MB zeros written), routing tables 0.25, fused kernel ~37 = gate/up mainloop ~16.4 (of which mma + ldmatrix
~12.0 = 180 TFLOPS on balanced routing, trellis decode ~3, copies ~1.2, partial tail tiles ~1.7) + gate/up epilogue
~2 + down mainloop ~8 + in-kernel actq ~1.5 + the non-overlapped part of the scatter ~5-7. The mainloop's mma part
runs at the kill test's ceiling (166-175 TFLOPS, `docs/PF3000_KILLTESTS_FP8.md`); the decode issue (77 % of the
mainloop's instructions) and the fp32 scatter are structural. The DRAM traffic of the whole layer (weights 1.8 GB,
activations ~0.8 GB, scatter RMW ~3.6 GB) bounds any variant of this design at ~25 ms.

## 3. Correctness (`tests/moee4m3/test_moe_e4m3.py`, 56 checks, `test_image.log` / `test_live.log`: PASS on both modules)

Stage by stage on the SAME inputs (real layer-10 experts, real and collapsed routing, T = 2048 / 1536 / 1024):

| stage | result |
|---|---|
| A1 gather: e4m3 bytes vs torch's quantization of H(x * suh) | 99.99897-99.99921 % identical; the rest (344-680 of 33-67 M) are adjacent e4m3 values (fp32 Hadamard order -> a tie flips); scales max rel 7.3e-7 |
| A2 gate/up: fp16 SwiGLU output vs the spec computed from the kernel's own e4m3 rows | rel-L2 1.6-1.9e-5, 99.35-99.39 % of fp16 values bit-identical |
| A3 actq: down e4m3 bytes / scales | 99.9990-99.9991 % identical, scales max rel 7.9e-7 |
| A4 down + scatter vs **fp64** of the same e4m3 operands (the mma's fp32 accumulation) | rel-L2 1.12e-7, max 1.8-2.9e-7 |

End to end, output vs the spec (`B` lines):

| case | kernel vs fp32 emulation (rel-L2 / max) | kernel vs fp64 spec | fp32 emulation vs fp64 spec | e4m3 vs production |
|---|---|---|---|---|
| real T=2048 | 1.20e-3 / 7.5e-3 | 5.4e-4 / 5.1e-3 | 1.16e-3 | 0.0651 |
| collapsed T=1536 | 1.10e-3 / 1.0e-2 | 7.4e-4 / 7.3e-3 | 1.10e-3 | 0.0651 |
| real T=1024, x8 activations | 1.41e-3 / 1.1e-2 | 5.8e-4 / 4.4e-3 | 1.38e-3 | 0.0715 |
| **real T=13824** | 1.27e-3 / 1.0e-2 | 7.3e-4 / 9.3e-3 | 1.24e-3 | 0.0651 |

"max" = max |diff| / max |ref|. The spec's own fp32 implementation is ~1.2e-3 from its fp64 one (an fp32 rounding
difference flips an e4m3 or fp16 rounding now and then, and the next quantization amplifies it), so ~1e-3 is the
noise floor of "identical to the emulation"; the kernel is **closer to the exact spec than the emulation is** (5-7e-4
vs 1.1-1.4e-3) and differs from production by the same 6.51 % as the emulation (the e4m3 class, as the offline
analysis predicted). Edge cases: expert_map with half the global ids non-local 8.5e-4, T=1 3.6e-7, T=17 6.4e-4, one
expert holding every token 1.0e-3, x300 activations with an all-zero row 1.0e-3 (finite).

## 4. Wiring (`overlay/glm53_moe_e4m3.py`, `overlay/patch_moe_e4m3.py`, kit r16e4)

- **Where the files live.** `glm53_moe_e4m3.py` and `glm53_moe_e4m3_ext*.so` (built by `tools/moee4m3/build.py` into
  `overlay/`) ship in the bundle's OVERLAY dir, not in `site/`; `setup.py` and `integrate.py` are r16l's bytes. With
  the switch unset the bundle never runs the overlay patch, so nothing of the feature reaches site-packages: an OFF
  kit composes r16l's tree byte for byte (`off_equals_prev`, below).
- **Knob** `GLM53_MOE_E4M3` (env.r16 `#switch`, forwarded to BOTH ranks by the r16e4 start.sh and validated there):
  unset / empty / `0` = stock; `1` = `patch_tf_bundle.py` runs `patch_moe_e4m3.py`, which copies the module and the
  extension into site-packages and appends a marked block to site-packages `integrate.py` that wraps
  `plugin_register` so `glm53_moe_e4m3.plugin_install()` runs after every other plugin step. Anything else: the
  start.sh refuses it before stopping the ranks; the patch would SystemExit; the module's `enabled()` raises and
  `install()` refuses with a WARNING. The module accepts exactly unset / "" / 0 / 1.
- **What is served.** The wrapper of production's module-global `apply_exl3_experts` is the outermost one (moeglue
  and the TF K2 hook fingerprint and wrap production's function first, unchanged). A call is served when
  **tokens > the layer's fused temp rows** (`EXL3_TEMP_ROWS_FUSED`, 256 in production), no stream capture is in
  progress, `fused` is not False, hidden == 4096 and the layer passed its checks. This gate is a token count, not
  "is a prefill": production runs `GLM53_MIXED_PREFILL_CHUNK=0`, so **decode / verify tokens that the scheduler
  batches into the same step as a prefill chunk go through the e4m3 path too** (behaviour kept as is: that is exactly
  what the moefq emulation's KL measures). A step without a prefill is served only if it exceeds the cap; production's
  largest decode/verify step is MAX_NUM_SEQS x (K+1) = 4 x 8 = 32 <= 256. The module computes that bound from the
  container env (MAX_NUM_SEQS, SPEC_METHOD, DFLASH_TOKENS / MTP_TOKENS, EXL3_TEMP_ROWS_FUSED) at install and with the
  real cap in the summary, and logs `glm53_moe_e4m3 DECODE BOUND EXCEEDED: ...` (WARNING; a boot_checks failure) if
  pure decode/verify steps could exceed the cap. Decode-only steps and every CUDA-graph capture size otherwise go to
  the wrapped function unchanged.
- **Per-layer self-test, at model load.** `Exl3MoEMethod.process_weights_after_loading` is wrapped: right after
  production built a layer's pointer tables the layer is checked (pointer tables, K4, local intermediate 1024, shared
  gate/up rotation, <= 1024 local experts) and self-tested against the spec (192 tokens routed to 8 of the layer's real
  experts, rel-L2 <= 5e-3; measured 9.2e-4; ~73 ms per layer, ~3 s per rank). So every test has run before vLLM's
  profile run and long before the engine reports ready, whatever `GLM53_BOOT_SHAPE_WARMUP` does (that warmup runs
  AFTER ready, from start.sh's post_ready_warmup, and is not relied on). If the class method were missing, the test
  runs lazily at the layer's first eligible call; vLLM's `profile_run` (`v1/worker/gpu/model_runner.py`:
  `_dummy_run(self.max_num_tokens, skip_attn=True, is_profile=True)` -> `execute_model`) executes every MoE layer with
  max_num_batched_tokens tokens, i.e. above the cap — read from the image's source, not observed in an engine run.
- **What a failure does.** A Python-level failure (shape check, self-test mismatch, a Python exception) keeps that
  layer on production's path for the life of the process, with a WARNING, and is counted. A **device-side fault**
  (an illegal address in a kernel, the fused kernel's watchdog `__trap` after ~30-60 s of waiting on a segment that
  never becomes ready) is a **sticky CUDA error**: no fallback can catch it, every later CUDA call of the process
  fails and the rank dies (at TP=2 the other rank then fails its collectives). Because the self-tests run at model
  load, such a fault shows up as a failed boot, not mid-serving; the recovery is `tools/env_r16.sh off moee4m3` +
  restart.
- **Summary (countable by boot_checks).** At the first apply call (vLLM's profile run) each process logs
  `glm53_moe_e4m3 summary: N/M layers served, F fell back (<reasons>); self-test FAILED a, raised b; decode bound
  MAX_NUM_SEQS x (K+1) = X (K=k) vs fused cap C` (WARNING if F > 0; logged again if a lazily checked layer changes the
  counts). boot_checks requires `42/42 layers served, 0 fell back (none); self-test FAILED 0, raised 0` on both ranks.
- **Memory**: none at steady state. The e4m3 rows, scales and the fp16 SwiGLU rows are views into production's grouped
  fat scratch (`_grouped_scratch`, the h13 / h2 buffers production allocates for these same prefill calls; with the
  load-time self-test they are allocated during model load instead of during the profile run), plus a 4096-int sync
  buffer per device. Per call: routing tables O(T x top-k) and the fp32 output (production allocates the same).
- **Logs**: `[glm53-moe-e4m3] glm53_moe_e4m3.py: installed`, `... integrate.py: armed`, `glm53_moe_e4m3 installed:
  GLM53_MOE_E4M3=1, ... per-layer self-test at load time`, `glm53_moe_e4m3 layer self-test passed (rel-L2 ...)` once,
  the summary, `glm53_moe_e4m3 active: first prefill routed-MoE call with N tokens ...`; WARNING on any refusal.

Tests: `tests/moee4m3/test_patch_moe_e4m3.py` (host, 17 checks: unset / "" / 0 install nothing; on / true / 2
refused; =1 copies both files byte for byte, integrate.py = original + the marker block only; idempotent; a drifted
block / a missing file refused; the armed plugin_register runs every original step first and plugin_install last; an
exception there does not escape). `tests/moee4m3/test_wiring.py` (37 checks, both production module versions): W1 OFF
— after the stock `integrate.plugin_register()` production's function object is unchanged and outputs equal the
pre-registration ones within production's own fp32-atomics spread; W2 ON through an `integrate.py` armed by the patch
— wrapper directly over production's apply, no double wrap, decode-sized call passed through, a CUDA graph captured at
T=64 through the hook replays equal to eager production, prefill served after one self-test, served == run(), the
e4m3-class difference vs production; W4 — the self-test runs inside `process_weights_after_loading`, the summary line,
the first prefill call needs no second self-test, the decode bound (4 x 8 = 32 <= 256; 64 x 8 = 512 > 256 warns),
uninstall restores the class method, values on / true / 2 refused; W3 — over the TF fork (K2) + prefill cap 1 stack.

Build: `python3 tools/moee4m3/build.py [--nvcc=-DME_DEBUG_VARIANTS]` inside the image -> `overlay/glm53_moe_e4m3_ext*.so`
(`make_kit.sh` refuses a .so older than the last commit of `kernels/moe_e4m3.cu` / the build script).

## 5. How to run

```
python3 tests/moee4m3/extract_layer.py 10 0 2          # host, numpy: the real TP=2 rank-0 layer-10 shard (1.7 GB, outside the repo)
RO=${HOME}/tf-exl3-assets/moee4m3
flock /tmp/tf-gpu-bench.lock env GPU_RUN_RO=$RO tests/gpu_run.sh bash -c 'python3 tools/moee4m3/build.py && python3 tests/moee4m3/test_moe_e4m3.py'
flock /tmp/tf-gpu-bench.lock env GPU_RUN_RO=$RO tests/gpu_run.sh python3 tests/moee4m3/test_wiring.py
flock /tmp/tf-gpu-bench.lock env GPU_RUN_RO=$RO GPU_RUN_ENV="BENCH_T=13824,4289,1791,512" tests/gpu_run.sh python3 tests/moee4m3/bench_moe_e4m3.py
# against the launcher overlay production runs: add GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py"
```

`tools/moee4m3/anatomy.py` (debug build) times the kernels with parts removed; `tools/moee4m3/selftest_time.py` the
self-test.

## 6. The emulation's weight cache (found while checking against the spec) — affects the KL measurement

`glm53_moe_e4m3_emu._WeightCache` keys its decoded-weight cache on `trellis.untyped_storage().data_ptr()`. In
production every expert's `LinearEXL3.trellis` is a view (`make_linear_exl3(layer.w13_trellis[e, 0], ...)`,
`.contiguous()` of a contiguous slice = no copy) into the layer's one stacked Parameter, so **all 288 experts of a
layer share one storage pointer**: with the default `GLM53_MOE_E4M3_EMU_CACHE_MB=2048`, every expert of a layer gets
the weights of the first expert decoded in that layer. Verified on production's own `process_weights_after_loading`
layer (`E` lines of `test_image.log`: 1 distinct key for experts 0..3, `get(expert 0) is get(expert 1)` True). The
emulation's own unit test (`emu_vs_offline.py`) uses separate tensors per expert and cannot see it. A KL measured
with the cache on would measure "every expert = expert X", not e4m3. Fix: key on `trellis.data_ptr()` (the view's
pointer, distinct per expert) or run with `GLM53_MOE_E4M3_EMU_CACHE_MB=0`. This branch did not modify moefq.

Also for that measurement: the emulation's GEMMs and Hadamards are torch fp32 matmuls; if the serving process allows
TF32 (`torch.backends.cuda.matmul.allow_tf32` / `set_float32_matmul_precision('high')`), they run in TF32 and the
emulation is no longer this spec. Not checked in vLLM here (NOT RUN).

## 7. Kit (r16e4) and remaining work

Kit integration is done (docs/DEPLOY_R16E4.md = the kit's ROLLOUT.md): env.r16 `#switch GLM53_MOE_E4M3 0|1`,
`tools/env_r16.sh on|off moee4m3` (refuses with `.env` unchanged unless the installed start.sh forwards the knob to both
ranks and the overlay files + bundle script are r16e4's), `make_start_sh.py --stage r16e4` (default; = r16l +
`launcher/start.sh.moee4m3.patch`), `boot_checks.sh` (container env head
== worker; the install / self-test / 42/42 summary / active lines when ON, their absence when OFF, never a FAILED /
raised / not-served / DECODE BOUND line), `off_equals_prev.sh`, `test_kit_scripts.sh` S.16, `test_boot_checks.sh`
B.15, `make_kit.sh` shipping the overlay files and this doc. Kit run results: `docs/logs/moee4m3/kit_r16e4/`.

Remaining:
1. The KL gate — first the emulation's KL with its cache bug fixed (section 6); then, if it ships, a two-rank boot
   with the switch on (boot_checks), a prefill timing (expected -0.87..-0.92 s per 13.8k chunk) and the usual quality
   probe.
2. Possible further speed (estimates, not built): the fp32 scatter's L2 atomics (~5-7 ms of the 41 non-overlapped);
   partial tail tiles (~1.7 ms); a cheaper self-test (4 experts) if boot time matters.

## 8. NOT RUN

- Anything on nodeA/nodeB or in production containers; no two-rank (TP=2) run — one rank's shard on one GB10.
- KL / quality probe of the e4m3 path in the serving engine (that is the moefq emulation's job; see section 6).
- Real captured activations / real routing (synthetic Gaussian activations, calibrated synthetic routing); layers
  other than layer 10.
- Nsight Compute: `ncu` exists on the host but the driver has `RmProfilingAdminOnly: 1` and the GPU containers run
  unprivileged as uid 1000 — kernels were profiled by removal variants (`anatomy*.log`) instead.
- A vLLM engine boot with the switch on (load-time self-tests in the real loader, the profile run reaching the hook,
  CUDA graph capture of all decode sizes through the hook, the summary line in a real log); covered only by the
  harness-level tests above and by reading vLLM's profile_run source. The kit scripts ran (section 7), not a boot.
- A device-side fault injection (the sticky-error path of section 4 is documented from CUDA semantics, not provoked).
