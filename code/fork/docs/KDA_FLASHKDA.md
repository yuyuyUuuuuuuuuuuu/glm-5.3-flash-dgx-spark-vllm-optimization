# KDA_FLASHKDA — FlashKDA 17a037d (fp32 recurrent state) for the KDA chunked prefill (`GLM53_KDA_FLASHKDA`)

Status 2026-10-01, branch `fkda` (from `poolfix` = r16l `f77b1b1`, the kit production runs). The KDA **chunked
prefill** of GLM-5.3-Flash (`chunk_kda_with_fused_gate`, the Triton FLA chain production runs today) is routed through
FlashKDA @ `17a037d98da546deb4591e967cf961a43c034d8b` ("Keep the recurrent state in fp32 between tiles", vLLM #58846)
when `GLM53_KDA_FLASHKDA=1`. **Opt-in and default OFF: unset = the stock Triton chain, the composed container is
byte-identical, and the extension is not even installed** (the `off_equals_prev` rows below). The recurrent decode path
(`fused_recurrent_kda`, including r16k's `GLM53_KDA_STRIDED_QKV`) is untouched. Plan step 2 of
`${HOME}/tf-exl3-assets/prefill3000/PLAN.md`; the pfkda kill-test that gated this (`docs/PF3000_KILLTESTS_KDA.md`
on branch `pfkda`) measured **3.38-3.82x** per layer and a final-state error vs fp64 of **1.31x** Triton's in the
long-memory gate regime (its original 1.65-1.71x KILL was an artefact of a forget-in-10-tokens input regime, per the
review) — verdict UNDECIDED there, decided here by the real-weight end-to-end tests of section 5.

## 1. Why the image's own FlashKDA is not used, and what ships instead

The image ships `vllm/_flashkda_C.abi3.so`: a **pre-fix, bf16-recurrent-state** build on the 14-argument-era ABI,
imported only by `vllm/models/kimi_k3/nvidia/kda.py` (never on the GLM-5.3-Flash path). It must never be used — 17a037d
exists precisely to keep the recurrent state fp32. So this feature ships its own build as **`_flashkda_fp32_C`**:
kernel sources untouched (FlashKDA 17a037d + CUTLASS 5c149f5, the pfkda staging), only the registration shim
(`flashkda_registration.cpp` @ vLLM nightly ddd6fbca, the 16-argument `fwd` schema with the trailing
`checkpoint_state`/`checkpoint_offsets`) has its namespace renamed, which makes the fp32 build coexist with the image's
pre-fix one instead of colliding on `torch.library(_flashkda_C)`. Built inside the production image for **`sm_121a`**
(GB10), CPU-only compile, `tests/fkda/build_flashkda_fp32.sh` → `overlay/_flashkda_fp32_C.abi3.so`.

Bit-identity of the rename: `tests/fkda/check_rename.sh` ran the pfkda `_flashkda_C` and the `_flashkda_fp32_C` builds
on identical inputs (single-sequence T=13,824 and a 3-sequence varlen call with row-strided beta and per-sequence
initial states; long-memory gate regime) and compared every output byte — **ALL OK, bit-equal, max|diff| = 0**
(`docs/logs/fkda/`, this also covers the varlen case the pfkda kill-test never measured).

**Kit r16y ships the fkda2 PRECISION build instead of the stock 17a037d arithmetic** (`docs/KDA_FLASHKDA2.md`, the
r16o change carried into the combined kit): same kernel/tiling/MMA structure, but the decayed q/k, `u` and `out` are
computed in fp32 and rounded to bf16 once, so the two builds are no longer bit-equal — `check_rename.sh` keys its
expectation on the build it dumped (the shipped `dd1788c2…` → bounded deviation, worst rel-L2 7.5e-3 / rel-Linf
8.6e-3 on those inputs; any stock build → bit-equal, the r16n expectation above).

## 2. What the overlay does (`overlay/patch_flashkda.py`, run by `patch_tf_bundle.py` when `GLM53_KDA_FLASHKDA=1`)

Installs `glm53_flashkda.py` + `_flashkda_fp32_C.abi3.so` into dist-packages and patches
`vllm/models/glm5next/nvidia/kda.py` at three anchors (preflighted in full, fail-closed, atomic, idempotent):
the import, a `configure()` call at the end of `Glm5NextLinearAttention.__init__`, and the chunked-prefill call site —
the Triton `chunk_kda_with_fused_gate(...)` call is kept **verbatim** as the else branch.

The wrapper (`glm53_flashkda.chunk_prefill`) keeps production's inputs/outputs contract:

| | Triton chain (off) | FlashKDA (on) |
|---|---|---|
| q/k/v | `_rearr(q_ns/k_ns/v_ns)`, `[1,T,H,D]` column slices of the merged short-conv output (token stride `3·proj`) → chain calls `.contiguous()` internally | same tensors; the wrapper `.contiguous()`s them (the op reads dense strides; ~340 MB/rank/layer copied, the same class of copy the Triton chain pays) |
| gate | `raw_g=g1` + `fused_kda_gate_chunk_cumsum` (bounded `lower_bound·sigmoid(exp(A_log)·(g+dt_bias))`, safe_gate) | **raw** `g1`, the bounded gate computed in-kernel from `A_log`/`dt_bias`/`lower_bound` |
| beta | **pre-sigmoided fp32** (the Triton kernels don't sigmoid) | **raw bf16 logits** (`beta_ns`), sigmoid in-kernel — same as the decode path's `fused_recurrent_kda(..., sigmoid_beta=True)` |
| l2norm | `l2norm_fwd(q/k)` before the chain | in-kernel |
| output | `[1,T,H,D]` bf16, **written in place into `v`** (pfkda finding; nothing downstream reads `v` after the call) | `[1,T,H,D]` bf16 in its own workspace out buffer (does NOT alias `v`) |
| final state | fp32 `[N,H,D,D]` | fp32 `[N,H,D,D]` (fp32 between tiles, 17a037d's point), handed to `scatter_states` → decode exactly as before |
| varlen | `cu_seqlens = non_spec_query_start_loc` | same `cu_seqlens`, int32, `initial_state.shape[0] == N` validated |

Buffers come from the rank's `current_workspace_manager()` (graph-safe, like every production kernel workspace), sized
once from `max_num_batched_tokens`/`max_num_seqs`: final state `[max_seqs,H,D,D]` fp32 + FlashKDA scratch
(`get_workspace_size`) + output `[1,max_tokens,H,D]`. They are **RESERVED at boot** (review D1): `configure()` calls
`get_simultaneous(*specs)` during the first KDA layer's `__init__` — `GPUWorker.init_device` initializes the workspace
manager before the model is constructed and `lock_workspace()` only runs after profiling/capture, so the reservation
cannot hit the locked-growth assert, a shortfall fails at boot with a `[glm53-kda-flashkda]` line instead of at the
first prefill, and the memory is on the books before vLLM measures the free memory for the KV pool. Measured on the
reserved line: **0.56 GiB/rank** at MNBT 16,384 / max_num_seqs 4 (the mini engine and production's MNBT) and 0.48 GiB
at 13,824 / 8 — the PLAN's ~0.6 GiB/rank estimate; an owner enabling it at a tight KV pool sees it deducted honestly at
boot.

With the quickwins allowlist extension (review B1) the patch also edits the installed
`glm53_prefill_quickwins.py`: the `kda_conv` entry of `VERIFIED` gains the AST fingerprint of the FlashKDA-patched
`Glm5NextLinearAttention._forward` (`9715f9b548cfa694`, alongside the stock `1e4f45149fceddf5`), because this overlay
edits the same function on disk and the plugin would otherwise refuse kda_conv (-114 ms/chunk, boot_checks MISS).
Before installing it the patcher proves the composition: the KDA_CONV edit still anchors exactly once on the
FlashKDA-patched source and the composed source compiles; `tests/fkda/check_quickwins_compat.py` (exit 0) and
`tests/fkda/check_kda_conv_install.sh` (the real runtime `install_item("kda_conv")` transplant accepting the patched
`_forward`) verify both directions, and with `GLM53_KDA_FLASHKDA=0`/unset nothing is touched.

Guards (all raise, i.e. fail closed): schema/arity (the 16-arg fp32-state fwd), `safe_gate` must be True, recurrent
state dtype must be fp32, token/sequence capacity, `initial_state` fp32, `cu_seqlens` int32. A one-time boot line
`[glm53-kda-flashkda] kda.py: chunked prefill -> FlashKDA 17a037d (_flashkda_fp32_C, fp32 recurrent state), GLM53_KDA_FLASHKDA=1; buffers RESERVED at boot: 0.56 GiB for ...`
is what `boot_checks.sh` greps for (with the bundle's install line and the quickwins-extension line).

## 3. Operator wiring (kit r16n = r16l + this switch)

| piece | content |
|---|---|
| `launcher/start.sh` | the r16n stage (`make_start_sh.py --stage r16n`, default; byte-differs from the r16m stage only in the F4 note's corrected numbers): F1 validate_numeric_config (empty/0/1; `1` also needs `overlay/tf/overlay/patch_flashkda.py` and a `patch_tf_bundle.py` that runs it), F2 head `-e` after `GLM53_KPOOL_DROP_LOWEST`, F3 worker `serve_env_names`, F4 note; sha256 `811728b2a4700726…` (= the committed `launcher/start.sh`; diff vs r16l `971c984c…` = `launcher/start.sh.flashkda.patch`) |
| `overlay/patch_flashkda.py`, `overlay/glm53_flashkda.py`, `overlay/_flashkda_fp32_C.abi3.so` | the patch, its wrapper, the fp32-state extension (r16y: the fkda2 precision build, 4,657,048 B, sha256 `dd1788c2…`; r16n shipped the stock 17a037d arithmetic as `c286213f…`, 4,657,088 B — `docs/KDA_FLASHKDA2.md`; the `.so` rides in the kit's `overlay/`, NOT `site/`, so `install_site` never puts it into a switch-unset tree) |
| `overlay/patch_tf_bundle.py` (+ its `launcher/overlay/` copy) | runs `patch_flashkda.py` behind `GLM53_KDA_FLASHKDA` (after kpooldown) |
| `env.r16` | `#switch GLM53_KDA_FLASHKDA 0|1` (operator switch: `apply_r16.sh` never writes it) |
| `tools/env_r16.sh` | `on|off flashkda` (`on` refuses, .env unchanged, unless the installed start.sh forwards the knob to BOTH ranks and the overlay+bundle are the r16n ones) |
| `tools/boot_checks.sh` | flashkda section: head == worker env, per-rank markers = install line + the quickwins `kda_conv VERIFIED +` line + the RESERVED-at-boot line |
| `tools/deploy16/off_equals_prev.sh` | `=0`/unset == PREV byte for byte; `=1` differs exactly in `vllm/models/glm5next/nvidia/kda.py` (+ its stale pyc), the extended `glm53_prefill_quickwins.py` (the kda_conv VERIFIED entry, a site/ entry of every r16 kit) and the two installed top-level files (also added to every compose's tree listing) |
| `tools/deploy16/kit_chain.sh` | the flashkda stage joins the patch chain (F1-F4 == `launcher/start.sh.flashkda.patch`) |

## 4. Speed (nodeC GB10, production image, one GPU job under `/tmp/tf-gpu-bench.lock`)

`tests/fkda/bench_prefill_on_off.py`: per-layer KDA chunked prefill, H = 32 per rank (TP=2 of 64 heads), D = 128, bf16,
safe_gate/`lower_bound=-5.0`, single sequence (production chunks; `GLM53_MIXED_PREFILL_CHUNK=0`), zero and nonzero
initial states, production-shaped strided views (merged-projection column slices, row-strided beta). Median of 20,
CUDA events. "wrapper" = exactly what the patched `kda.py` runs (including the four `.contiguous()` copies and the
workspace-manager fetch); "op alone" = the bare `torch.ops._flashkda_fp32_C.fwd` (pfkda's measurement, for the
decomposition). Per-chunk = per-layer × 34 KDA layers, the pfkda projection.

| T | init | Triton chain | wrapper (what the patched kda.py runs) | speed-up | op alone (pfkda's measurement) | saving × 34 layers |
|---|---|---|---|---|---|---|
| 13,824 | zeros | 23.02 ms | 9.65 ms | **2.39x** | 6.47 ms (3.56x) | **455 ms** |
| 13,824 | state | 22.90 ms | 9.62 ms | **2.38x** | 6.50 ms (3.52x) | **452 ms** |
| 4,608 | zeros | 7.69 ms | 3.19 ms | 2.41x | 2.19 ms (3.52x) | 153 ms |
| 4,608 | state | 7.68 ms | 3.18 ms | 2.42x | 2.18 ms (3.53x) | 153 ms |
| 1,791 | zeros | 2.98 ms | 1.31 ms | 2.27x | 0.88 ms (3.40x) | 57 ms |
| 1,791 | state | 3.03 ms | 1.27 ms | 2.40x | 0.88 ms (3.45x) | 60 ms |

| T | init | Triton chain | wrapper, strided q/k/v | wrapper + kda_conv (contiguous q/k/v) | op alone | saving/chunk (strided / kda_conv) |
|---|---|---|---|---|---|---|
| 13,824 | zeros | 23.01 ms | 9.71 ms (2.37x) | **6.46 ms (3.56x)** | 6.52 ms (3.53x) | **452 / 563 ms** |
| 13,824 | state | 22.99 ms | 9.73 ms (2.36x) | **6.56 ms (3.50x)** | 6.57 ms (3.50x) | **451 / 559 ms** |
| 4,608 | zeros | 7.70 ms | 3.20 ms (2.40x) | 2.19 ms (3.51x) | 2.17 ms (3.55x) | 153 / 187 ms |
| 4,608 | state | 7.70 ms | 3.25 ms (2.37x) | 2.22 ms (3.46x) | 2.20 ms (3.50x) | 152 / 186 ms |
| 1,791 | zeros | 3.07 ms | 1.29 ms (2.38x) | 0.89 ms (3.47x) | 0.87 ms (3.52x) | 60 / 74 ms |
| 1,791 | state | 2.98 ms | 1.29 ms (2.31x) | 0.88 ms (3.37x) | 0.87 ms (3.41x) | 57 / 71 ms |

(json: `docs/logs/fkda/bench_on_off.json`.) The Triton column reproduces the R13 trace's 22.4 ms/layer and pfkda's
23.4 ms; the op-alone column reproduces pfkda's 6.45-6.53 ms — the renamed build measures the same kernel.

Reading the columns: **production always runs `GLM53_PREFILL_QUICKWINS=all`, whose kda_conv item already hands the KDA
core contiguous q/k/v** (it re-launches the conv per channel into a `[3, T, P]` buffer — bit-identical to production's
merged conv, and this patch keeps it installable, §2). The "wrapper + kda_conv" column is therefore the number an
enabling owner sees: **~6.5 ms/layer at the full chunk = 3.5x, ~0.56 s per 13,824-token chunk** — right at the plan's
step-2 mid estimate (0.46 s) and the pfkda op-level 0.58 s. The "wrapper, strided" column (2.4x, ~0.45 s) is the same
call on production's merged-conv views without kda_conv, i.e. what the wrapper costs when the quick win is absent; its
~3.2 ms/layer overhead over the bare op is the dense copies of q/k/v (~340 MB at T=13,824) plus the workspace fetch.
The 32k gate projection: 2 chunks × 0.56 s + the 1,791-token tail ≈ **-1.2 s of 16.9 s (≈ +7%)**.

A possible follow-up (not done here) is fusing the remaining g1 copy into the conv epilogue.

## 5. Quality on the real-weight mini model (the deciding gates)

`tests/handoff/run.sh` (production-composed container, `HANDOFF_KIT=${HOME}/tf-exl3-deploy16.r16l`, FULL decode
graphs + PIECEWISE prefill, DFlash2 spec, fp8 KV, greedy, `--shadow 1`), model `GLM-5.3-Flash-handoff-mini`
(real GLM-5.3-Flash KDA/DSA/indexer/mHC/dense weights at TP=2 per-rank shapes) + `-dflash2`. The shadow runs every
prefill forward three times from identical starting bytes of every KV allocation: P (production, FlashKDA off),
C (P again, control), F (the configuration). Off runs have `feature == control == IDENTICAL`.

| run | config | prefill forwards (shadow) | P vs C | F vs P: first diverging module | owned KDA recurrent-state max abs diff | F vs P layer-output max abs | decode vs fresh prefill (KL, B_j of request A0) | generated tokens on/off |
|---|---|---|---|---|---|---|---|---|
| off_base | knob unset, 6,000 + 3,001 | 4,608 / 1,392 / 3,001 | IDENTICAL | - (F == P) | 0 | - | (baseline) | - |
| on_flashkda | =1, same prompts, `--consistency 8` | same | IDENTICAL | `model.layers.0.self_attn` (before any fast path) | 1.6e-3 .. 3.4e-3 (fp32 state) | 0.51 .. 1.91 (bf16 layer out) | -6.2e-5 .. -4.5e-3 (8 B_j; production's own off/on runs measured -5e-4 .. +7e-2) | **48/48 identical** (both requests, all 24 tokens) |
| on_long | =1, 30,000-token prompt (3 chunks, two with initial state) | 13,824 / 13,824 / 2,352 | IDENTICAL | same | 2.4e-3 / 2.7e-3 / 4.5e-3 — **flat, no growth across chunks** | 2.2 .. 3.1 | - | - |
| on_multi | =1, two requests in ONE 6,001-token step (`--batch-a 2`, varlen N=2, per-request state scatter) | 6,001 | IDENTICAL | same | 1.9e-3 | 0.51 | 4 B_j | **48/48 identical** (2 requests × 24 tokens) vs off_batched |
| off_batched | knob unset, same `--batch-a 2` configuration | 6,001 | IDENTICAL | - (F == P) | 0 | - | - | reference arm |
| off_multi | knob unset, 6,000 + 3,001 again | 4,608 / 1,392 / 3,001 | IDENTICAL | - (F == P) | 0 | - | - | repeat: deterministic |

| qw_off | QUICKWINS=all + MLA_PREFILL=1 (the r16n env.r16 ship set), knob unset | same as off_base | IDENTICAL | `model.layers.3.self_attn` (the first MLA layer: glm53_mla_prefill is exact-class but not bitwise vs FA2) | ≤ 4.1e-3 (layers ≥ 3 only, propagated) | 0.62 .. 1.12 | -6.2e-5 .. -4.5e-3 band | reference arm |
| qw_on | QUICKWINS=all + MLA_PREFILL=1 + `GLM53_KDA_FLASHKDA=1` | same | IDENTICAL | `model.layers.0.self_attn` (the first KDA layer) | 1.8e-3 .. 2.3e-3 | 0.53 .. 1.58 | -1.6e-5 .. 7.8e-4 (8 B_j) | **48/48 identical** vs qw_off; `kda_conv (first call …)` in BOTH arms' logs |

Logs: `/tmp/fkda/handoff/*/` (`harness.log`, `shadow.json`, `result.json`, `records.pt` per run; committed copies and
the raw evidence under `${HOME}/tf-exl3-assets/fkda-evidence/`); wrapper unit rig
`docs/logs/fkda/wrapper_unit.json` (production-shaped strided views, single + 3-sequence varlen, out rel-RMS
7.9e-3 / final-state rel-RMS 6.5e-3 vs the Triton chain — bf16-level, matching pfkda's 5.7e-3/4.8e-3 — 2.40x median,
fail-closed guards all refusing).

Kernel-level state precision (the review's follow-up rigs, `docs/logs/fkda/review/`, real A_log/dt_bias, long-memory
gate regime): per-row varlen mixes (short rows + a 4,608-token row in one call) FlashKDA-vs-Triton final-state rel-RMS
2.3e-3 .. 2.6e-3 against a Triton-vs-bf16-rounded-initial-state reference of 1.7e-3 (i.e. below 1.5x the round-trip
noise floor, at every row length 1..4608, all finite); chained 8 × 13,824-token chunks vs fp64: rel-L2 5.6e-3 vs
Triton's 4.1e-3, **ratio 1.35-1.38, flat from chunk 1 to chunk 8** (110,592 tokens carried).

Read of the P-vs-F diffs: every difference traces to the first KDA layer's chunked-prefill output (the shadow's first
diverging module is `model.layers.0.self_attn` in all runs, before any fast path, with quickwins/MLA prefill idle) —
the KDA recurrent-state rows the request owns differ by bf16-tile arithmetic (max|Δ| ~2.4e-4..1.6e-3 on fp32 states,
i.e. the ~0.05-0.16% pfkda measured against the Triton chain), **flat across chunks** (13,824-token chunk 1 vs the
initial-state chunks of the 30k prompt: no growth), and everything else (later layers' conv states, MLA KV pages,
indexer caches, kpool tails, kv_indices) differs only by propagation. The conv-state diffs of KDA layers ≥ 1 are the
same propagated bf16-level perturbation (layer 0's request-owned conv state is bit-identical, as it must be: conv reads
only the embeddings); the non-request-owned "stale" rows the harness already documents production kernels writing past
their step also change (their content is not a function the two passes share). Decode-side: the state the prefill hands
decode is compared bitwise as part of the same raw buffers, and decode itself is production's path unchanged.

**The deciding gates of the pfkda review are met**: greedy token identity on/off (including the multi-request chunk and
across standalone/joint prefill of the same prompt), and a decode-vs-fresh-prefill consistency band that overlaps
production's own runs. That is the end-to-end real-weight test the kill-test said would have to justify the extra 0.2 pp
prefill-state delta — it passes on the mini model at ≤ 30k tokens. The production-context consequence (0.8M-token
sessions accumulating 0.05-0.16%-of-state differences through spec-decode) remains the one thing nodeC cannot measure;
the 30k run's flatness across chunks is the best available proxy.

## 6. Not run / open risks

- **Production-scale memory accounting**: the ~0.5 GiB/rank workspace is measured at the mini engine's MNBT 16,384 and
  the bench's 13,824; production's MNBT (16,384) is the first, `max_num_seqs` scales only the tiny state buffer. Not
  measured inside a KV-pool-starved boot.
- **800k-context sessions** (the pfkda review's deciding-test caveat above): the flat-across-chunks result is at 30k.
  The upgrade path stays reversible in one knob (`env_r16.sh off flashkda` + idle restart).
- **Decode-path drift of a FlashKDA-produced state over spec steps** (DEC_KDA_FI-style per-step drift): not measured
  separately; the B_j consistency and 46-57 greedy verify steps are the end-to-end cover.
- FlashKDA's `checkpoint_state`/`checkpoint_offsets` (the 16-arg tail) are passed as `None` — the state-handoff
  rollback of spec decode relies on production's own `scatter_states`, exactly as the Triton path does.
- TP=2 rank symmetry: single-rank measurements; production runs the same call on both ranks with rank-local heads
  (H=32), which is what the bench and the TP=1 mini engine exercise.
- The Triton chain mutates `v` in place and FlashKDA does not; no current reader depends on it (checked the layer's
  tail: `v_ns` is dead after the call), but a future edit that reads `v_ns` expecting the output would silently regress.
- The image's `vllm/_flashkda_C` (bf16-state) stays in place untouched; nothing on the GLM path imports it, and the
  fp32 build's separate namespace keeps them apart even if something ever did.

## 7. Reproduce (nodeC)

```
tests/fkda/build_flashkda_fp32.sh                              # CPU-only, no lock (staging under /tmp/fkda, /tmp/pf3000)
flock /tmp/tf-gpu-bench.lock tests/fkda/check_rename.sh        # renamed build == pfkda build (bit-equal)
python3 tests/fkda/check_quickwins_compat.py <image kda.py>    # B1: kda_conv installs on the patched _forward (exit 0)
FKDA_CPU_USER=root tests/fkda/check_kda_conv_install.sh        # B1 e2e: OFF untouched; ON + the real quickwins transplant
flock /tmp/tf-gpu-bench.lock FKDA_USER=root tests/fkda/gpu_run.sh -c \
  'python3 /w/tests/fkda/wrapper_unit.py --out /fkda/out/wrapper_unit.json'
flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh -c \
  'python3 /w/tests/fkda/bench_prefill_on_off.py --out /fkda/out/bench_on_off.json'
HANDOFF_KIT=${HOME}/tf-exl3-deploy16.r16l tests/handoff/run.sh /tmp/fkda/handoff on_flashkda \
  GLM53_KDA_FLASHKDA=1 -- --consistency 8
HANDOFF_KIT=${HOME}/tf-exl3-deploy16.r16l tests/handoff/run.sh /tmp/fkda/handoff on_multi \
  GLM53_KDA_FLASHKDA=1 -- --prompts 3000,3001 --batch-a 2 --consistency 4
HANDOFF_KIT=${HOME}/tf-exl3-deploy16.r16l tests/handoff/run.sh /tmp/fkda/handoff off_batched -- --prompts 3000,3001 --batch-a 2
```
(Kit r16n: `tools/deploy16/make_kit.sh <dir>` with `PREV_KIT=<the r16l kit production runs> PREV_ENV_ADD=none`
`tools/deploy16/off_equals_prev.sh <kit> <prev> <log dir>`, `tools/deploy16/test_kit_scripts.sh <kit> <log dir>`,
`tools/deploy16/test_boot_checks.sh`, `tools/deploy16/check_boot_strings.py`.)
