# KDA_STRIDED_QKV — strided q/k/v/beta in the KDA recurrent decode (`GLM53_KDA_STRIDED_QKV`)

Status 2026-09-30, branch `kdatrim` (from `r16j` 905c07c = what production runs). Backport of the KDA half of
upstream vLLM PR #55736 (merged 2026-09-10; full diff `${HOME}/tf-exl3-assets/upstream/pr55736.diff`) onto
vLLM 0.1.dev20051+g487ecf187. Built and measured on nodeC (GB10, production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, shared with other jobs: every run under
`/tmp/tf-gpu-bench.lock` via `tests/gpu_run.sh`). Production (nodeA/nodeB) was not touched; **the feature is
opt-in and default OFF: unset `GLM53_KDA_STRIDED_QKV` = the image's stock bytes and production behaviour,
byte-identical.** Reviewed 2026-09-30 (§7): the first version could not be switched on in production at all (the
r16j start.sh never forwards the variable to either rank) and the kit could not be built; both fixed on this branch
(r16k start.sh stage, `#switch` in env.r16), and the per-rank saving re-measured at production's TP=2 shapes.

## 1. What production wastes

GLM-5.3-Flash's decode KDA core calls `fused_recurrent_kda`
(`vllm/third_party/flash_linear_attention/ops/kda.py`, called once per KDA layer per step from
`vllm/models/glm5next/nvidia/kda.py:546` and `:613`; 34 of 45 layers are KDA). The layer hands it:

* `q/k/v` — column slices of the merged short-conv output (`qkv_spec.split(...)` then `.reshape(1,-1,H,D)`):
  token stride `3·proj`, not `proj`;
* `beta` — a column slice of the fused `qkvbfg_a` projection (token stride `3·proj + H + 2·D`);
* `g` — already contiguous (`[1, T, H, D]`).

The 487ecf187 wrapper (`ops/kda.py:167-172`) copies each of q/k/v/beta with `.contiguous()` before launching
the Triton kernel, and the kernel (`ops/fused_recurrent.py`) addresses tokens as `bos*H*K` / `bos*HV*V`
anchored to those dense copies. That is four small D2D copies per KDA layer per step whose only job is to
undo a stride the kernel could read in place. At the production verify widths (batch 1, 5 / 6 / 8 tokens:
DFlash2 with GLM53_ADAPTIVE_K_SET=4,5,7, + 1) each copy moves ≤ 64 KiB per rank (≤ 128 KiB at the unsharded H=64) — pure launch/latency floor, paid inside every
captured graph, every step, forever.

## 2. What the backport changes (opt-in overlay `overlay/patch_kda_strided_qkv.py`)

Exactly upstream #55736's `fused_recurrent` + `kernels` halves, mapped onto the 487ecf187 files (upstream
patched the same code vendored under `glm5next/nvidia/ops/third_party/kda/`; in this image it lives in
`vllm/third_party/flash_linear_attention/ops/{fused_recurrent.py,kda.py}`):

1. `ops/fused_recurrent.py` — `token_stride()` (upstream's helper, verbatim semantics: per-token `[H, D]` /
   `[H]` block contiguous, non-overlapping tokens, dense batch when `B > 1`; anything else raises with the
   shape in the assert); four runtime kernel args `stride_{q,k,v,beta}_t`; the kernel's pointer bases and
   per-token advance use them instead of the hardcoded `H*K` / `HV*V` / `HV·(V|1)`; the non-KDA wrapper
   (`fused_recurrent_gated_delta_rule_fwd`) passes its (contiguous) inputs' strides, which reproduces the old
   addressing exactly.
2. `ops/kda.py` — `fused_recurrent_kda_fwd` passes `token_stride(...)` for q/k/v/beta; `fused_recurrent_kda`
   stops copying q/k/v/beta (g keeps its copy — the kernel reads the gate through a dense layout the caller
   does not guarantee); `o = torch.empty_like(k)` becomes `torch.empty(k.shape, ...)` because `k` may now be
   a strided view and the kernel indexes `o` densely (reachable only when the caller passes no `out=`).

The overlay is fail-closed and idempotent (anchor preflight of BOTH files before either is written → patched /
already-present / SystemExit, atomic replace, pyc cleared), registered in `overlay/patch_tf_bundle.py` behind
`GLM53_KDA_STRIDED_QKV` (unset/empty: the bundle skips it; `0`: the overlay prints and touches nothing; `1`: patch;
anything else: SystemExit). It is an **operator switch** (env.r16 `#switch GLM53_KDA_STRIDED_QKV 0|1`, like r16j's
`GLM53_REJECTION_METHOD`): `apply_r16.sh` never writes it, `tools/env_r16.sh on|off kdaqkv` sets/removes it, and it
reaches the containers only through a start.sh that forwards it to both ranks (the r16k stage, §4).

Not ported, deliberately (the PR's other two thirds):

* **MLA absorbed query (part 2 of #55736)** — both halves target code production does not run:
  * the *concat skip*: production's sparse MLA backend is `flashinfer_mla_sparse_sm90`, whose
    `forward_mqa` **requires** the `(q_nope, q_pe)` tuple (`NotImplementedError` otherwise) and passes the two
    halves straight to the flashinfer kernel — the `torch.cat(q, dim=-1)` the PR skips exists only in
    `flashinfer_mla_sparse.py`, a different backend file, and the one cat that could happen in
    `mla_attention.py` only fires with `dcp_world_size > 1` (production: DCP off, TP=2 on nodeA + nodeB). The NoPE half is moot
    either way: the model *is* NoPE (`qk_rope_head_dim=0`, `mla_use_nope=true`), but there is no concat on
    this path to skip.
  * the *token-major bmm write* (`torch.bmm(..., out=mqa_ql_nope.transpose(0,1))` into a token-major buffer):
    this branch (`q_pad_num_heads is None`) *is* the branch production takes, so it is applicable — but it
    cannot change the number of kernels (one bmm either way) and its only possible gain is whatever the
    flashinfer MLA kernel (fa2 on GB10; production's start.sh-mounted `flashinfer_mla_sparse_sm90.py` picks
    `fa2` for sm_12x, fa3 is Hopper-only) does with the transposed vs contiguous `q_nope` handed to
    `wrapper.run(...)`. Measured in §5.2.
* **duplicate router GEMM (part 3 of #55736)** — production already removed it at runtime:
  `glm53_gemv_install.py`'s `GLM53_BF16_GEMV_DEDUP_ROUTER` (default 1) clears the MoE runner's `gate` so
  `Glm5NextMoE.forward`'s pre-computed `router_logits` are the only router GEMM. Nothing to port.

## 3. Numerics contract

Pure layout change: the kernel loads the same values (same `tl.load`, masks, fp32 math, same token order)
through strided pointers instead of dense copies, so **outputs and final recurrent states must be bitwise
equal** to production's — asserted at `rtol=0 / atol=0`, not a tolerance. Verified in
`tests/test_kda_strided_qkv.py` (log: `docs/logs/kdaqkv/test_kda_strided_qkv.log`):

| check | result |
|---|---|
| patched (strided in place) vs stock (.contiguous() copies), outputs + final states, shapes 1×1 / 7×1 / 1×8 / 3×3, H=64 K=128 | bitwise equal |
| patched strided vs patched contiguous (upstream's own invariant) | bitwise equal |
| both vs a pure-fp32 recurrence (upstream's reference) | max abs diff ≤ 5e-4 (out) / ≤ 8e-7 (state), tolerance 1e-2 / 1e-4 |
| unaddressable layouts (batch slice, overlapping tokens, head-strided) | raise in `token_stride` (loud) |
| FULL CUDA-graph capture + replay (production decode replay mode), stock and patched | capture ok, two replays bitwise-equal to eager (state reset between, as production reloads it); `out=` still written in place |
| copies per KDA layer call (dispatch record, batch 1, 8 tokens) | stock `clone` ×5 → patched ×1 = the four q/k/v/beta copies gone (§5.1); patched replay = exactly 1 CUDA kernel |

## 4. Wiring (deploy kit, `env_r16.sh` style)

| file | change |
|---|---|
| `overlay/patch_kda_strided_qkv.py` | the overlay (this feature) |
| `overlay/patch_tf_bundle.py` + `launcher/overlay/patch_tf_bundle.py` | PATCHES entry + docstring row; the two copies are one file (make_kit.sh refuses otherwise) |
| `tools/deploy16/make_start_sh.py` | stage **r16k** (new default) = r16j's start.sh (cea89226, pinned) + K1 `validate_numeric_config` (empty/0/1; `1` also needs the overlay file and a bundle script that runs it) + K2 head `-e GLM53_KDA_STRIDED_QKV` after `GLM53_KPOOL_RING` + K3 the worker `serve_env_names` entry + K4 a note; `launcher/start.sh` = its output (7736abf8), `launcher/start.sh.kdaqkv.patch` = the r16j → r16k diff |
| `tools/deploy16/env.r16` | `#switch GLM53_KDA_STRIDED_QKV 0\|1` (apply_r16 never adds it and refuses another value; revert_r16 removes it; kit_chain K.2 checks it reaches both ranks) |
| `tools/deploy16/env_r16.sh` | feature `kdaqkv`: `on` refuses (with `.env` unchanged) unless the installed start.sh forwards the knob to both ranks, the overlay file is installed and the bundle runs it; then writes `GLM53_KDA_STRIDED_QKV=1`; `off` removes it (= stock) |
| `tools/deploy16/boot_checks.sh` | switch section: head == worker value; `1` → both ranks' `patched` + `applied` lines; unset/empty → skip line, `0` → the overlay's stock line; no `patched` line when off; containers without the variable (an r16j start.sh) → stock |
| `tools/deploy16/{test_boot_checks,kit_chain,off_equals_prev,make_kit}.sh` | B.12 cases; K.2 patch chain + negative controls; composed-tree off/0/1 states; the doc in the kit |
| `docs/KDA_STRIDED_QKV.md` | this document; logs under `docs/logs/kdaqkv/` (review: `docs/logs/kdaqkv/review/`) |

Operator flow (nodeA, only with a kit built from this branch = the r16k start.sh installed by `apply_r16.sh`):
`tools/env_r16.sh on kdaqkv` → restart both ranks idle-gated (`tools/wait_idle.sh` + `restart2.sh`) →
`tools/boot_checks.sh` (both ranks must print `[glm53-kda-strided-qkv] ops/fused_recurrent.py: patched; ops/kda.py:
patched` and `[glm53-tf-bundle] patch_kda_strided_qkv.py: applied (GLM53_KDA_STRIDED_QKV=1)`); revert with
`tools/env_r16.sh off kdaqkv` + restart. **With the r16j kit production runs today the switch cannot be turned
on**: its start.sh forwards neither to the head nor to the worker, so a `GLM53_KDA_STRIDED_QKV=1` line in `.env` is
silently ignored (`docs/logs/kdaqkv/review/forwarding.txt`, case `r16j-on`); `env_r16.sh on kdaqkv` now refuses
there instead of writing a dead line. No r16k ROLLOUT document is written (make_kit.sh still ships DEPLOY_R16J.md).

## 6. Risks / blast radius

* The two patched files also serve non-GLM importers (`models/bailing_moe_v3.py`,
  `model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` call the same `fused_recurrent_kda`). For them
  the math is unchanged (same kernel, same strides for their contiguous inputs — the non-KDA wrapper keeps
  reproducing the old addressing), but a caller that relied on the wrapper silently fixing a layout
  `token_stride` cannot address (e.g. a `[B, T]` scalar beta, overlapping tokens) now gets a loud
  `AssertionError` instead of a copy — the same contract upstream #55736 ships and its tests assert. Neither
  model is in production.
* First call per process JIT-compiles one extra Triton binary (the strided kernel); production compiles at
  warm-up before graph capture, and the capture itself is unaffected (runtime stride args are baked into the
  captured launch, and the static decode buffers keep their strides across replays — proven by the capture +
  double-replay check in §3).
* The kernel-count claim is per KDA layer call; production is **TP=2** (nodeA + nodeB): 32 KDA heads per rank
  (`in_proj_qkvbfg_a` 12576 wide per rank; the production trace's recurrent grid `[1,16,32]`), so the copies are
  half the bytes of the implementer's H=64 rig with the same launch count. Re-measured at the per-rank shape in
  §7 (−0.30 … −0.33 ms/step bare, −0.28 … −0.44 with in_proj/o_proj-sized DRAM traffic around each core).
* Not verified on nodeA/nodeB (never touched): an operator A/B there should watch the first boot's
  `[glm53-kda-strided-qkv]` bundle line and run the same boot checks as for any R16 feature.

## 5. Measurement (nodeC, batch 1)

Rig: `tests/bench_kda_strided_qkv.py`, production shapes (H=64 KDA heads, head_dim 128), the SAME static
strided input buffers for both sides, FULL CUDA graphs (what production replays), 15 interleaved rounds ×
20 replays, medians; log `docs/logs/kdaqkv/bench_kda_strided_qkv.log`.

### 5.1 Kernel count and GPU time

`tests/bench_kda_strided_qkv.py` (log `docs/logs/kdaqkv/bench_kda_strided_qkv.log`, run 2026-09-30 batch16):
FULL CUDA-graph replays, 15 interleaved rounds × 20 replays, medians, SAME static strided buffers for both
sides, batch 1, full-model geometry (H=64 KDA heads, head_dim 128):

| shape (batch 1) | stock (4 copies + kernel) | patched (kernel only) | delta |
|---|---|---|---|
| T=1 (plain decode) | 0.327 ms/step | 0.324 ms/step | **−0.002 ms/step** (nil, within noise) |
| T=5 verify rows | 3.998 ms/step | 3.643 ms/step | **−0.355 ms/step** |
| T=8 verify rows (the widest production verify: adaptive K 7 + 1) | 6.885 ms/step | 6.452 ms/step | **−0.434 ms/step** |

Per-layer (T=8): 202.5 → 189.5 µs (−13.0 µs); the 34-layer step KDA-core time drops by ≈ 0.38–0.43 ms
across the three runs of this rig (batch11/15/16 logs agree: −0.384 / −0.377 / −0.434; T=5 spread was
larger: −0.18 … −0.36). That is the KDA-core share of one decode step at the UNSHARDED head count (H=64), not a
production shape: production is TP=2 (32 KDA heads per rank; the implementer's "TP3 / ~⅓ per rank" was wrong). The
per-rank figure is §7's (−0.25 … −0.33 ms/step bare) and §8's re-run; the in-production A/B decides.

Kernel count: torch.profiler on the eager call, CPU dispatch record (this torch build sends
`Tensor.contiguous()` on a strided tensor to `aten::clone`, no `aten::contiguous` record):
stock **clone ×5** (q/k/v/beta + the profile window's own state clone) vs patched **clone ×1** — exactly
the four copies removed per KDA layer per step, and the captured graph replays exactly **1 CUDA kernel**
per layer call on the patched side (tests/test_kda_strided_qkv.py, 41/41 checks:
`docs/logs/kdaqkv/test_kda_strided_qkv.log`).

### 5.2 The PR's MLA token-major bmm write (applicability probe)

`tests/bench_mla_bmm_layout.py`, production MLA geometry (N=64, qk_nope_head_dim=256, kv_lora_rank=512,
NoPE), batch 1 × 8 tokens, standalone flashinfer 0.6.18 `BatchMLAPagedAttentionWrapper` with the patched
sm90 impl's backend selection (**fa2 on sm_12x**; fa3 is Hopper-only — the probe first ran with the wrong
`fa3` and JIT-failed with `no kernel image is available`, see §6). Synthetic top-k 64, not a production
MLA bench (log: `docs/logs/kdaqkv/bench_mla_bmm_layout.log`):

| probe | result |
|---|---|
| bmm old (out into `(N,B,L)`, handed over transposed) vs new (straight into the transposed view of a `(B,N,L)` buffer) | outputs **bitwise equal**; graph launch time 26.67 vs 26.68 µs (a wash) |
| the flashinfer kernel with old transposed-view `q_nope` vs new contiguous `q_nope` | outputs **bitwise equal**; interleaved A/B 16.63 vs 16.68 µs per run (median of 15×20) — **no measurable difference**; 1 kernel launch each, no hidden `.contiguous()` copy |

So the token-major write buys nothing on this backend (the fa2 kernel accepts the strided query at no
cost, and no copy kernel lurks behind it), and it is **not ported**: `GLM53_KDA_STRIDED_QKV` covers only
the KDA half. (A profiler-only first impression said 27.5 vs 14.3 µs; the interleaved measurement shows
that was CUPTI warm-up on the first profiled pass, which is why the probe reports the interleaved medians.)

### 5.3 Per-step verdict

At the production verify shapes (batch 1, T = 5 / 6 / 8 rows: GLM53_ADAPTIVE_K_SET=4,5,7), the KDA half of #55736
removes **four small D2D copies per KDA layer per step** (34 KDA layers, on each rank). The implementer's H=64 rig
(the unsharded model, not a production shape) gave −0.355 ms (T=5) / −0.434 ms (T=8); **at production's per-rank
shape (TP=2, H=32) the saving is ≈ 0.30–0.33 ms per step** (§7; ≈ 9 µs per layer, launch/latency-floor bound), both
ranks in parallel. Against the profiled production step (69.1 ms in docs/DEC_SMALLOPS.md §1) that is ≈ 0.45 % of
the step: worth having, not transformative, and at the edge of what an end-to-end tok/s A/B can resolve. nodeC
numbers only; nil at T=1 (a single-token slice is already contiguous, the stock path copies nothing).

## 7. Review (2026-09-30; logs `docs/logs/kdaqkv/review/`)

Re-ran everything the implementer reported and measured what it did not.

**Defects found and fixed on this branch**

| # | severity | defect | evidence | fix |
|---|---|---|---|---|
| 1 | high | the switch could never take effect: the r16j start.sh forwards `GLM53_*` knobs to the containers by an explicit list (head `-e` lines, worker `serve_env_names`), `GLM53_KDA_STRIDED_QKV` was on neither, and `GLM53_EXTRA_ENV` refuses `GLM53_*` names; `env_r16.sh on kdaqkv` wrote a line both ranks ignore | `forwarding.txt` case `r16j-on`: head/worker `-e` absent, both bundles leave the files at e33dcadb (stock) | make_start_sh.py stage r16k (K1–K4), `launcher/start.sh` regenerated; cases `r16k-on` / `r16k-on+block`: `1` on both ranks, both patched (db4ee7d6) |
| 2 | high | the kit could not be built: `launcher/overlay/patch_tf_bundle.py` (the copy make_kit.sh requires to equal `overlay/patch_tf_bundle.py`) was not updated | `cmp` differed at line 21; make_kit.sh exits 3 on it | copies synced |
| 3 | medium | no boot evidence / no guard: boot_checks.sh did not know the feature, env_r16.sh `on` checked nothing, the value `0` (this repo's usual "off") would have stopped both containers inside the bundle after both ranks were down | — | `#switch` in env.r16, env_r16.sh precheck, start.sh validation (empty/0/1 + files), overlay treats `0` as stock, boot_checks.sh section + test_boot_checks B.12 (10 cases) |
| 4 | medium | the per-step saving was reported at H=64 (unsharded) with "production TP3" — production is TP=2 (32 KDA heads per rank) | — | re-measured below; doc corrected |
| 5 | low | the drift test could not see a partial write (fused_recurrent.py was already patched when kda.py drifted) | — | drift now starts from pristine files and asserts fused_recurrent.py is NOT written; `0` case added (16/16) |

**Re-runs** (nodeC, production image, under `/tmp/tf-gpu-bench.lock`): `tests/test_kda_strided_qkv.py` 41/41
(`review/test_kda_strided_qkv.log`); `tests/test_kdaqkv_bundle.py` 16/16.

**Production per-rank shapes** (`tests/review_kdaqkv_tp2.py`, `review/review_tp2.log`, 102/102, peak 8.8 GiB):
H=32, `projected` 12576 wide with q/k/v AND beta as column slices of it (causal_conv1d_update writes in place:
`out = x`), and the mixed-step layout (index_select → contiguous conv buffer):

* bitwise vs stock over the WHOLE state tensor + outputs + the `out=` slice path: 1×5, 1×6, 1×8, 2×8, 8×8, 8×5
  in place; 3×6, 1×8, 4×1 mixed; 1×1, 4×1 in place (1×1: no copy on either side);
* FULL CUDA graphs of a 34-layer step with per-layer state/input/output buffers (1×8, 4×6): stock graph ==
  patched graph bitwise, and the patched replay == an eager patched run after re-filling every static buffer with
  new data (the graph reads live buffers, nothing baked), two trials each;
* kernels per layer in a FULL replay: stock 5 (4 `elementwise_kernel` copies + the recurrent kernel), patched 1,
  at T=5 and T=8;
* timing, 34-layer FULL graphs, paired + interleaved + alternating order, 3 blocks × 12 rounds × 10 replays
  (per-block median of the paired differences):

| per-rank step (H=32, batch 1) | stock | patched | paired delta per block (ms/step) |
|---|---|---|---|
| bare, T=5 | 2.546 ms | 2.241 ms | −0.304, −0.311, −0.293 |
| bare, T=6 | 2.918 ms | 2.613 ms | −0.302, −0.310, −0.309 |
| bare, T=8 | 3.680 ms | 3.361 ms | −0.309, −0.327, −0.309 |
| + in_proj (51.5 MB) / o_proj (16.8 MB) reads around each core, T=5 | 20.401 ms | 20.095 ms | −0.281, −0.374, −0.438 |
| + same, T=8 | 21.135 ms | 20.742 ms | −0.405, −0.356, −0.366 |

**Deploy wiring** (`review/forwarding.txt`, `review/env_r16_kdaqkv.txt`): the r16k start.sh's own
`validate_numeric_config` + `launch_cluster` on a scratch tree (docker/ssh shimmed) forwards the value identically
to both ranks for unset / empty / 0 / 1 / 1+block, and each rank's bundle, run in the production image with that
rank's value, leaves the two files at stock (e33dcadb) for unset/empty/0 and patches both (db4ee7d6) for 1;
`true` and `1` without the overlay file are refused by validate_numeric_config. env_r16.sh: refused (`.env`
byte-identical) on the r16j tree and on an r16j tree with only the new overlay + bundle; on the r16k tree `on` adds
exactly the one line (idempotent), `off` restores `.env` byte for byte.

**Kit** (`tools/deploy16/make_kit.sh` from this branch with `PREV_KIT` = the r16j kit, `PREV_ENV_ADD=none`; the AOT
`.so` files are the r16j kit's, byte-identical, since no kernel changed): 97 files; vs the r16j kit only `env.r16`,
`launcher/start.sh`, `launcher/overlay/patch_tf_bundle.py`, `tools/boot_checks.sh`, `tools/env_r16.sh` changed and
`overlay/patch_kda_strided_qkv.py` was added (site/ identical).

* `tools/deploy16/off_equals_prev.sh <kit> <r16j kit>` (`review/off_equals_prev.txt`, ALL OK): the composed serving
  tree (every file under site-packages/vllm + the bundle's site entries, 5454 files, per the launcher's own overlay
  order in the production image) with the switch unset is **byte-identical** to the r16j kit's in S1 (production's
  env) and S2 (every env.r16 line), also with `=0`; with `=1` exactly `ops/fused_recurrent.py` and `ops/kda.py` differ.
  (Run with a private lock file: its containers are CPU-only; the shared GPU lock starved it for 30 min.)
* `tools/deploy16/test_kit_scripts.sh` with PREV_KIT = r16j (`review/test_kit_scripts.txt`): 60 ok, ALL OK. S.11 is
  the r16i → r16j update test and is now skipped when the previous kit already has that switch (it produced 7 false
  FAILs with PREV = r16j); new S.12 (9 checks) covers the kdaqkv switch from production's state at r16j: `on` refused
  on the r16j tree, the update keeps `.env` byte-identical and reports the switch unset, `on` adds exactly one line
  that the installed start.sh accepts (and refuses with r16j's bundle script), `off` restores `.env`, a bad value is
  refused unprinted before anything is written, revert with the switch on restores files and `.env` byte for byte.
* `tools/deploy16/test_boot_checks.sh` (`review/test_boot_checks.txt`): 41 ok incl. B.12; `check_boot_strings.py`:
  106 markers, 0 not printable by the shipped code.
* `tools/deploy16/kit_chain.sh` K.1-K.3 passed on this kit; its K.2 previous-stage check assumed the previous kit is
  the r16 stage (r16i) and now picks r16 or r16j by the previous kit's sha (then applies blockverify.patch and/or
  kdaqkv.patch). K.4 (the plugin census in GPU containers; its `docker run --gpus` bypasses tests/gpu_run.sh's
  memory cap) was NOT run by either review (`review/kit_chain_std.txt` does not exist).


## 8. Second review (2026-09-30 afternoon; logs `docs/logs/kdaqkv/review2/`)

Independent re-run after nodeC's 15:13 crash (nothing taken from §7's logs on trust). GPU work only through
`tests/gpu_run.sh` (40 GiB PyTorch cap, `--memory 64g`) under `/tmp/tf-gpu-bench.lock`, one job at a time.

**Code**: the overlay's hunks are upstream #55736's KDA half verbatim (token_stride asserts, the four runtime stride
args, strided pointer bases/advance, dense `o`); `bos` is `int64` in the kernel, so `bos * stride` (12576-wide rows)
cannot overflow; `inplace_final_state=False` allocates `final_state` with `q.new_empty` (dense), unaffected. No other
production overlay or site/ module edits or fingerprints `ops/fused_recurrent.py` / `ops/kda.py` (quickwins
`kda_conv` rewrites only the prefill branch of `Glm5NextLinearAttention._forward`, which feeds `chunk_kda`).
Production's FP8 in_proj output is Marlin-padded (12608 wide); the kernel takes any row stride ≥ H·D, so that is
the same case as 12576 for the addressing.

**Re-runs / new checks**

| check | result |
|---|---|
| `tests/review_kdaqkv_tp2.py` (§7's rig, re-run) | 102/102; bare −0.305/−0.309/−0.311 (T=5), −0.313/−0.311/−0.310 (T=6), −0.250/−0.290/−0.285 (T=8) ms/step per block; with in_proj/o_proj-sized traffic −0.33 … −0.44; stock 5 → patched 1 kernel per layer (`tp2_rerun.log`) |
| `tests/review2_kdaqkv_paths.py` (new) | 25/25 (`review2.log`): the overlay-patched files bound over the image's own paths import as `vllm.third_party.flash_linear_attention.ops.*` and `vllm.models.glm5next.nvidia.kda` binds the patched `fused_recurrent_kda`; **breakable-graph eager replay**: `eager_break_during_capture` replays `_forward` on `weak_ref_tensor` views — strides/data_ptr preserved, patched on the weak refs == stock on the originals (1×8, 3×5, 4×1); the glm5next spec path end to end (split → in-place `causal_conv1d_update` → `qkv.split` → reshape → `out=` slice) bitwise (1×8, 2×6, 1×5); the other launcher the overlay edits (non-KDA `fused_recurrent_gated_delta_rule`, varlen decode with GVA and spec verify) bitwise |
| MLA token-major probe at the per-rank head count (`MLA_PROBE_N=32`, production flashinfer 0.6.18 + sm90 mounts) | bmm bitwise, 16.41 vs 16.41 µs; fa2 kernel with transposed vs contiguous `q_nope` bitwise, 37.84 vs 37.62 µs/run interleaved (≈ −0.2 µs × 11 MLA layers ≈ 2 µs/step: noise) — part 2 stays not ported (`mla_n32.log`) |
| kit from this branch (`make_kit.sh`, PREV = r16j) | 97 files; vs r16j only BUILD/SOURCE, env.r16, launcher/start.sh, launcher/overlay/patch_tf_bundle.py, tools/boot_checks.sh, tools/env_r16.sh differ; + overlay/patch_kda_strided_qkv.py, docs/KDA_STRIDED_QKV.md |
| `make_start_sh.py` | `--stage r16j` reproduces production's start.sh cea89226 byte for byte (== the r16j kit's); default (r16k) = 7736abf8 = committed launcher/start.sh; r16j → r16k diff = K1–K4 only (22 lines) = `launcher/start.sh.kdaqkv.patch` |
| `test_boot_checks.sh`, `test_kit_scripts.sh` (PREV = r16j) on that kit | 41 ok / 60 ok, ALL OK (B.12, S.12 as in §7) |
| `off_equals_prev.sh` on that kit vs r16j (private lock, CPU-only containers) | ALL OK: composed serving tree (5454 files) byte-identical to r16j's with the switch unset (S1 production env, S2 every env.r16 line) and with `=0`; `=1` changes exactly `ops/fused_recurrent.py` + `ops/kda.py` (`off_equals_prev.txt`) |
| `tests/test_kdaqkv_bundle.py` | 16/16 |

A note for anyone extending the non-KDA check: with a slot index 0 (the null block) the stock kernel leaves that
sequence's `o` rows unwritten, so stock vs stock differs once the allocator hands out dirty memory — the test uses
slots ≥ 1 (not a patch effect: stock-vs-stock reproduced it).

**Decision left to the orchestrator/owner**: the kit's start.sh is no longer production's cea89226 (the r16k stage adds
K1–K4). There is no way to reach both ranks without it (the r16j start.sh forwards `GLM53_*` only by explicit list and
`GLM53_EXTRA_ENV` refuses `GLM53_*`). With the switch unset the only runtime difference is an extra empty
`-e GLM53_KDA_STRIDED_QKV=` on both containers, and the composed serving tree is byte-identical to r16j's.

**Not verified** (no nodeA/nodeB access): the real boot on the two production nodes, the end-to-end tok/s A/B, and
kit_chain K.4 (GPU census outside the capped gpu_run.sh). The expected production effect is ≈ 0.3 ms of a ≈ 69 ms
step (≈ 0.4 %), below what a single end-to-end tok/s A/B resolves; judge the A/B by the boot lines + no regression.
