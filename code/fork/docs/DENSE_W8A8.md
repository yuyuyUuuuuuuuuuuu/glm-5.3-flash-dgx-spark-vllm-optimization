# DENSE_W8A8 — W8A8 (fp8 activations x fp8 weights) for the dense + shared-expert FP8 linears, PREFILL only (`GLM53_DENSE_W8A8`)

Status 2026-10-01, branch `w8a8` (= `poolfix` r16l + the pffp8 kill-tests). Node1 only (one GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, every GPU run `tests/gpu_run.sh` under
`flock /tmp/tf-gpu-bench.lock`, 40 GiB torch cap). Node2/nodeB were not accessed; nothing was installed anywhere.
Inputs: pf3000 plan step 3 (`${HOME}/tf-exl3-assets/prefill3000/PLAN.md`), the kill test and its adversarial
review (`docs/PF3000_KILLTESTS_FP8.md` 2 + 6).

**Verdict: SHIP as an env-gated feature (default OFF = production byte-identical).** Net saving measured through the
real hooked `Glm53DenseFp8Method.apply`: **+519.0 ms per 13,824-token chunk per rank** (production 1342.3 ms ->
W8A8 823.2 ms, production's own launcher overlay `exl3.py`; +510.4 with the image's module), **+63.2 ms** at the
tail-chunk size M=1791, at a persistent cost of **322 MiB/rank** (the KV pool does NOT shrink: production pins it
with `--kv-cache-memory`, KV_CACHE_BYTES=16106127360 - see the section 2 correction of docs/DENSE_W8A8_2.md; this
doc's earlier "2.00M -> 1.96M" line assumed utilization sizing). Quality: the only new
rounding against production is the per-token e4m3 activation quantization (weights and per-channel scales are
production's exact bytes); the handoff gates of section 6 passed.

## 1. What production runs today, and what changes

Production quantizes every dense projection of the allow-listed groups (`GLM53_DENSE_FP8=dense,kda,mla,shared`) to
e4m3 per output channel and stores it **only** as the Marlin repack (int32 [K/16, 4·Npad] + bf16 permuted scale
x 2^120; the bf16 weight is deleted at load). At prefill those weights are served by the exact-BF16-dequant +
TileLang W8A16 path (`GLM53_FP8_LARGE_M=1`, Npad x K in the LARGE_TABLE) or by `marlin_gemm` — measured 1342 ms
per 13,824-token chunk per rank over the 192 dense calls. The pf3000 kill test (Test C) measured `cutlass_scaled_mm`
W8A8 at our shapes 1.14-2.41x faster per GEMM, but left the weight layout open: a resident standard-layout fp8 copy
costs 3.50 GiB/rank (KV -> ~1.58M, rejected), and the only re-layout available then (production's bf16 dequant
kernel + fp8 cast, 6·N·K bytes of traffic) cost 106.9 ms/chunk (its own DRAM floor).

## 2. The layout decision: a byte-exact per-call repack kernel

`kernels/fp8_w8a8.cu` (`tf_fp8_w8a8_ext`) un-permutes the Marlin payload **fp8 -> fp8** (2·N·K bytes of traffic):
one block covers four k-tiles of one 64-wide n-tile, each thread assembles one row's 16 contiguous fp8 bytes from
four int32 words (`__byte_perm`) and stores one uint4. The mapping is the one `kernels/fp8_gemv.cu:5-13` documents
(and `tests/test_fp8_gemv.py::layout` verified bit-exactly); `tests/fp8_w8a8_unit.py` L proves the repack equals
(a) the independent torch formula, (b) production's own e4m3 quantization bytes, and (c) the exact bf16 dequant cast
back to fp8, on every production shape plus a padded/unaligned one — 100.00% byte equality, as in the kill test.

So the served GEMM is: repack into a scratch buffer -> per-token e4m3 quantization of each piece (the image's own
`dynamic_per_token_scaled_fp8_quant`) -> `torch.ops._C.cutlass_scaled_mm` with per-token scales [M,1] x the SAME
per-channel scale Marlin multiplies by (`fp8_gemv.large_alpha`, the un-permuted stored bf16 scale as fp32; pieces of
2048 rows when the fp8 weight exceeds `GLM53_DENSE_W8A8_PIECE_MIB` (default 16) MiB, else one call — reproducing the
kill test's per-shape optimum). Output bf16, fp32 accumulation, one rounding — the same epilogue class as
production's `bf16(sum * s)`.

**Cost measured (M=13,824, chunk totals): repack 37.1 ms/chunk** (the kill-test floor for this route is 32.7 ms at
230 GB/s; the dequant route was 106.9) **+ per-token quant 90.7 ms + piece GEMMs = 823.2 ms/chunk total** vs
production's 1342.3.

**Memory per rank: 322.2 MiB persistent** — the repack scratch, one fp8 [max N, K] buffer per distinct K (the KDA
in_proj 51.5 MiB, the drafter-shaped K=20480 buffer never allocated in production), + 0.2 MiB of cached per-channel
scales. Transient per call <= scratch + one piece + activations. CORRECTION (DENSE_W8A8_2.md A3): production PINS the KV
pool (`--kv-cache-memory`, KV_CACHE_BYTES=16106127360, env_nonsecret.txt:13, start.sh:337-343), so the pool stays
2,003,436 tokens - the "2,003,436 -> 1,961,403" figure below assumed utilization sizing and does not apply; the
322 MiB/rank comes out of the headroom instead. (The 3.50 GiB/rank resident alternative would have taken a
UTILIZATION-sized pool to ~1.58M.)


## 3. What is served, and what never is

`fp8_w8a8.py` wraps `Glm53DenseFp8Method.apply` (installed AFTER `fp8_gemv` in `integrate.plugin_register`, so the
W8A8 wrapper is the outer one and every decline falls through fp8_gemv to production's own apply). A call is served
iff: eager (CUDA-graph capture is declined — decode graphs are <= 64 tokens), not under torch.compile (the DFlash
drafter), `x.shape[-1] == k` and M >= `GLM53_DENSE_W8A8_MIN_M` (default 512), the layer's group is one production
routed (`self.group in the module's own _glm53_dense_fp8_groups()`; the drafter's "draft" group is never served),
the weights are the Marlin layout this module reads, n % 16 == 0 and k % 16 == 0 (cutlass operand alignment; every
production shape has it — a layer that does not declines and production serves it), and the layer passed its
self-test. Any non-CUDA error before a launch declines (counted, not repeated); CUDA errors propagate like
production's.

Load-time self-tests (per layer, fail-closed -> the layer stays on production's path): the layout check once per
(Npad, K) (repack bytes == the bf16-dequant bytes, section 2) and a GEMM smoke per layer (per-token quantized random
rows through cutlass vs production's Marlin: rel_l2 <= 6e-2, all finite). 18/18 layers of the bench rig passed at
2.62e-2..2.73e-2.

## 4. Numerics (what the owner is accepting)

Weights contribute NOTHING new (the repack is byte-exact, the scale is Marlin's own). The one new rounding is the
per-token e4m3 activation quantization: per-GEMM output rel_l2 vs production's path **2.65e-2 .. 2.66e-2** on every
production shape at every M in {64, 512, 2048, 13824} (Gaussian activations; the kill test measured 2.42e-2..2.65e-2
with Gaussian AND x30-outlier channels). This is a numerics class like R2 (`GLM53_DENSE_FP8`), reached through the
KDA in_proj/o_proj too — the KDA recurrent gates and the MLA projections see it. The pf3000 plan/critic predicted
~+0.04 nats/token of prefill KL over noise for the full FP8 class (step 3 alone is the smaller half of that).
Decode is untouched (Marlin), so the decode-vs-prefill consistency probe reads W8A8 prefill against Marlin decode —
section 6 for the measured band. The gates before enabling in production:

1. `env_r16.sh on w8a8` + restart both ranks + `boot_checks.sh` (=1 row) + `--after-traffic` (the first-served line).
2. The handoff gates of section 6 (already run on nodeC; re-run `tests/handoff/run.sh` after any change).
3. Idle A/B on production (owner window): the yardstick x4 seeds at 32k/128k, plus `tools/prodcheck/quality_long.py`
   3 x 14k and a >= 100k needle, W8A8 on vs off, against the R11/R12 noise floor 0.0066-0.0078 nats/token.

## 5. Measured on nodeC (logs in `docs/logs/w8a8/`)

* Unit (`tests/fp8_w8a8_unit.py`, JIT and AOT builds): **127/127 checks PASS** — layout (13 shapes x 3
  references), scratch reuse/growth, per-token quant 2.646e-2, per-shape GEMM class (table below), bias path
  1.56e-3 vs Marlin(bias), dispatch declines (M < 512, fp16 x, capture, unknown group, broken layout, n % 16),
  install idempotent, uninstall restores the originals exactly.
* Speed (`tests/bench_fp8_w8a8.py` through the real hooked apply, production env values, interleaved median of 5,
  real weights where the partial checkpoint has them; `bench_fp8_w8a8_overlay_exl3.log` = production's launcher
  overlay module, `bench_fp8_w8a8_image_exl3.log` = the image's):

| shape (N x K) | calls/chunk | production current ms | W8A8 ms (repack + quant+GEMMs) | speedup | tail M=1791 |
|---|---|---|---|---|---|
| kda.in_proj 12576x4096 | 34 | 16.9 | 10.1 (0.6 + 9.6) | 1.67x | 3.06 -> 1.76 (1.74x) |
| kda.o_proj 4096x4096 | 34 | 5.7 | 3.7 | 1.53x | 1.15x |
| mla.qkv_a 2048x4096 | 11 | 2.9 | 2.2 | 1.34x | 1.02x |
| mla.q_b 8192x1536 | 11 | 4.9 | 2.7 | 1.81x | 1.27x |
| mla.o_proj 4096x8192 | 11 | 17.2 | 7.4 | 2.33x | 1.30x |
| shared.gate_up 2048x4096 | 42 | 2.9 | 2.3 | 1.24x | 1.03x |
| shared.down 4096x1024 | 42 | 1.6 | 1.2 | 1.39x | 1.18x |
| dense.gate_up 12288x4096 | 3 | 16.5 | 9.6 | 1.70x | 1.75x |
| dense.down 4096x6144 | 3 | 8.8 | 5.4 | 1.64x | 1.21x |
| **chunk totals** | 192 | **1342.3** | **823.2 (repack 37.1, quant 90.7)** | **+519.0 ms** | **+63.2 ms** |

  (drafter fc 4096x20480: NOT served — group "draft", same path in both arms, cancels in the difference.)
* M sweep: in_proj 1.00x at M=256 (declines at M < 512), 2.0-2.1x at M=512, 1.62x at M=13824; shared.gate_up
  0.76-0.88x at M=512-1024 (-0.06..-0.04 ms/call, <= 2.5 ms/chunk), >= 1.0x from M=1791. The default min M 512
  costs at most a few ms per chunk on the small shapes and is where prefill begins.
* Memory: persistent 322.2 MiB scratch + 0.2 MiB scales; peak CUDA allocation of a full W8A8 chunk 2.188 GiB above
  the pre-chunk state (weights already resident; scratch 322 MiB of that).

## 6. Real-engine gates on the handoff mini model (tests/handoff/run.sh, production-composed container)

The stock handoff mini (`GLM-5.3-Flash-handoff-mini`) is a BF16 model with `quantization_config` REMOVED at build
time (tests/handoff/build_mini.py), so its target linears never take `Glm53DenseFp8Method` — a first A/B on it
(/tmp/w8a8-ab) exercised only the wiring: the chain installs the module, the plugin installs, the graphs capture,
every greedy token is identical on/off (nothing was served; the shadow feature pass reads IDENTICAL trivially).
For a real gate, a **config-only variant** was built (`GLM-5.3-Flash-handoff-mini-w8a8`: the same real
KDA/DSA/mHC/dense-MLP weights + `quantization_config {quant_method: exl3, bits: 4}`; the mini has no routed
experts, so no EXL3 shard is needed and the target's KDA/MLA/dense projections take the production quant path at
load exactly as in production). A/B on it, five fresh engines, FULL decode graphs + PIECEWISE prefill, the real
DFlash2 drafter, production env (`GLM53_FP8_GEMV=1`, `GLM53_FP8_LARGE_M=1`, `GLM53_LMHEAD_FP8=1`), the r16l kit's
composed launcher; `--consistency N` = KL(decode token j || fresh prefill of prompt + the first j tokens), the R16
state-handoff probe (table: `docs/logs/w8a8/handoff/ab_summary.txt`):

| run | switch | prompts | gen A0/A1 | decode ms/step p50 (46 verify steps) | KL mean / max | shadow ctrl/feat | capture |
|---|---|---|---|---|---|---|---|
| off | unset | 6000,3001 | 24 / 24 | 84.7 | +0.0001 / 0.0004 | IDENTICAL / IDENTICAL | FULL |
| on | 1 | 6000,3001 | 24 / 24 | 86.2 | **+0.0061 / 0.0492** | IDENTICAL / IDENTICAL | FULL |
| on2 | 1 | 6000,3001 | 24 / 24 | 84.9 | **+0.0063 / 0.0495** | IDENTICAL / IDENTICAL | FULL |
| offlong | unset | 20005,3001 (MNBT 13824) | 24 / 24 | 85.3 | +0.0002 / 0.0006 | IDENTICAL / IDENTICAL | FULL |
| onlong | 1 | 20005,3001 (MNBT 13824) | 24 / 24 | 85.5 | -0.0004 / 0.0000 | IDENTICAL / IDENTICAL | FULL |

* **The path engages end to end in the real engine**: the on arms log `tf_fp8_w8a8: first call served
  (M=16384, min M 512, pieces of 2048 rows)` (vLLM's profile run) and every self-test at load passed; FULL decode
  graphs + PIECEWISE prefill capture succeed with the switch on in every run.
* **Determinism**: the two on engines generated IDENTICAL greedy tokens for both requests, and their consistency
  KL agrees to 3 decimals (0.0061/0.0492 vs 0.0063/0.0495) — the W8A8 path is run-to-run stable (the cutlass
  accumulation is deterministic at a fixed shape/piece layout).
* **The numerics are visible where they should be**: off vs on greedy tokens are identical on the 6000-token
  prompt but diverge from token 4 on the 3001-token prompt and from token 11 on the 20005-token prompt (a ~2.6e-2
  per-GEMM perturbation moving tiny argmax margins — on a 10-layer mini model, not a quality signal by itself);
  decode ms/step p50 84.7-86.2 across all five arms (on-off ≈ +0.7%, within noise; decode never touches W8A8).
* **Consistency probe**: with W8A8 prefill against the untouched Marlin decode, KL(decode||fresh prefill) is
  +0.006 mean / 0.049 max, stable across engines — inside the band the r16l A/B saw (1e-3..7e-2) and below the
  R16i production baseline (0.0081); the off arms read ~0 (identical numerics). In the multi-chunk long runs the
  probe's positions are all prefill-sampled (both arms' logprobs come from prefill forwards), so it reads ~0 and
  cannot rank the arms there.
* The shadow's feature pass (P vs F) is trivially IDENTICAL for a knob the harness cannot force off (it knows the
  R16 features' env, not GLM53_DENSE_W8A8); the control (P vs P) IDENTICAL is the meaningful part.

## 7. Deploy-kit wiring (r16n)

Exactly the kpooldown pattern: env.r16 `#switch GLM53_DENSE_W8A8 0|1` (never added by apply_r16.sh; unset = stock),
`env_r16.sh on|off w8a8` (on refuses with .env unchanged unless the installed start.sh forwards the switch to BOTH
ranks and the bundle carries patch_dense_w8a8.py + fp8_w8a8.py + the .so), start.sh stage r16n
(`launcher/start.sh.w8a8.patch` on top of the pinned r16l stage 971c984c: head -e after GLM53_KPOOL_DROP_LOWEST,
worker serve_env_names, validate_numeric_config empty/0/1; 1 needs the overlay files and a bundle that runs them),
`patch_tf_bundle.py` runs `patch_dense_w8a8.py` (installs fp8_w8a8.py + tf_fp8_w8a8_ext*.so into site-packages ONLY
when =1, removes them on =0, fail-closed on any other state; unset = skip line), `integrate.plugin_register` imports
the module only when the env is set. The module is deliberately NOT in the bundle `site/` (the composed tree with
every knob unset stays byte-identical to the previous kit's, the r16k glm53_dectrace precedent). `boot_checks.sh`
B.15: on = applied + installed + `tf_fp8_w8a8 installed:` per rank (+ `first call served` with --after-traffic);
off = the bundle skip line or the =0 stock line; never = the plugin's refusal line, an ineligible call, a failed
self-test. Kit-level suites: `test_kit_scripts.sh` S.16, `test_boot_checks.sh` B.15, `kit_chain.sh` K.2 (the chain
gains `launcher/start.sh.w8a8.patch`), `off_equals_prev.sh` (=0 == the previous kit; =1 differs ONLY by the two
installed bundle entries, vllm/ untouched), `check_boot_strings.py`.

**r16y integration (the one edit to this wiring):** the w8a8 branch carried the conditional `fp8_w8a8` import in the
bundle's site `integrate.py` itself. The combined kit r16y ships `site/` byte-identical to the previous kit's (r16n)
with every switch unset (the requirement `off_equals_prev` proves on the real fs), so the arming moved to compose
time: `patch_dense_w8a8.py` inserts the marked import block into the INSTALLED `integrate.py` at =1 (right after the
`fp8_roof` step, the same place and plugin order this branch shipped: after `fp8_gemv`, so the W8A8 wrapper is the
outer one) and removes it again at =0 (byte-exact restore, drift-fail closed, `compile()`-checked), exactly the way
`patch_moe_e4m3.py` arms `integrate.py` for `GLM53_MOE_E4M3` (whose block appends at the END; the two compose, and
the 16-combination test `tests/mhc_sp/review2_combo16_r16y.py` proves it). Everything else in this section — the
switch, the start.sh stage, the overlay files, the module/kernel bytes — is this branch's, unchanged. The branch's
repo-root `integrate.py` keeps the conditional block for the repo-level rigs; the kit's site `integrate.py` is
r16n's until =1 arms it.

Interaction notes: (a) `GLM53_DEC_FP8ROOF` (decode L2 prefetch) does not see W8A8-served calls (its accounting hooks
inside fp8_gemv's wrapper) — it is a decode feature, W8A8 serves prefill only. (b) The prefill decode-consistency
probe and the handoff harness's bitwise shadow (P/C/F) are expected to read DIFFERENT with W8A8 on: it is a numerics
change by design (the shadow is run OFF; section 6). (c) `GLM53_KDA_BF16_LARGE_M=1` (not set in production) retains
a bf16 in_proj copy and its layers are never served (declined like production's own path).

## 8. What was NOT run

1. Anything on nodeA/nodeB (no production boot, no idle A/B, no yardstick with the switch on) — section 4's step 3.
2. A quality ranking beyond the handoff gates: the mini model's 10 layers cannot rank a ~2.6e-2 per-GEMM class
   against the noise floor at depth; the production quality probe (quality_long.py, needle >= 100k) is the owner's
   gate before enabling.
3. `GLM53_DENSE_W8A8_MIN_M` tuning below 512 (production prefill chunks are 13,824 or tails >= 1,791 in practice).
4. Concurrent decode + prefill contention (mixed prefill is off in production, GLM53_MIXED_PREFILL_CHUNK=0).
5. TP=3 (production is TP=2; the repack kernel's layout holds for any Npad but nothing was measured at TP=3).
