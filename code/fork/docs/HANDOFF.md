# Prefill -> decode state handoff (deploy-r16 decode-consistency regression)

## 1. The production symptom

`tools/prodcheck/kpool_decode_consistency.py` (3 real-text prompts of ~6k tokens; A = temperature-0 decode of 1536
tokens with logprobs 5; B = a fresh prefill, own cache_salt, of prompt + A's tokens; KL(A || B) per generated position):

| server | KL mean | top-1 | note |
|---|---|---|---|
| R15 (deploy-r15) | 0.00625 | 98.48 % | flat over positions |
| R16b: quickwins=all, MLA prefill, hostloop, smallops dconv, kpool ring, APC | 0.294 | 95.31 % | worst in positions [0, 128): 0.49 |
| R16c: R16b minus kpool ring and hostloop | 0.172 | 96.46 % | |
| R16d: R16c minus quickwins and MLA prefill | 0.00647 | 98.83 % | = baseline |

`quality_long.py` (prefill prompt_logprobs only) had R16b at the noise floor against R15 (KL 0.0048). So the prefill
outputs were fine and something the prefill features leave for decode was wrong.

## 2. The harness (tests/handoff/)

`tests/handoff/run.sh` starts the production image on nodeC with `--network none`, composed the way both ranks compose
their containers after the R16 rollout:
- the kit's `GLM53_OVERLAY_ORDER` chain, with every patch taken from the read-only production launcher copy;
- the kit's APC overlays and `patch_tf_bundle.py`, with this worktree's modules as the bundle;
- the SM90_KV mounts;
- production's non-secret env, with every R16 knob empty except the ones under test.

It runs `run_engine.py`, which drives the real vLLM engine: V2 runner, breakable piecewise CUDA graphs, fp8 KV,
FLASHINFER_MLA_SPARSE_SM90, prefix caching in mamba align mode, DFlash2 with adaptive K {4,5,7}. The engine runs on the
10-layer handoff mini model (`build_mini.py`). It holds real GLM-5.3-Flash KDA / DSA+indexer / mHC / dense-MLP weights,
cut to the TP=2 per-rank shapes, plus the real DFlash2 drafter. Each prompt is its own request:
- 200 tokens: below 256, so it takes production FA2 and leaves its slots in the process-wide `kv_indices`, as other
  traffic does in production.
- 6000 tokens: runs as a 4608-token chunk without initial state, then a 1392-token chunk with initial state. The
  mamba-align chunker floors chunk ends to 4608, exactly as production does for the probe's 6k prompts.
- 3001 tokens: one chunk.
- 14000 or 14400 tokens: a 13824-token chunk, so idx_gate's M >= 10240 path runs.

Each prompt is followed by 16-24 greedy tokens of spec-verify decode (M = 5, 6 or 8).

**Shadow.** Every eager prefill forward (>= 256 tokens) runs three times in the same process, each time from the same
snapshot of every KV allocation (KDA conv / recurrent slots, MLA pages, indexer K cache + scales, kpool tail pages,
drafter KV) and of the side buffers (SM90 `kv_indices` / `kv_len_arr`, top-k buffer):
- **P**: production. Every fast path is forced onto production's own statement (quickwins MIN_T = inf, idx_gate off,
  MLA prefill MIN_TOKENS = inf).
- **C**: P again, as a control.
- **F**: the configured features. The engine continues from F.

The runs are compared bit for bit on four things: the outputs (hidden states and DFlash2 aux states), every decoder
layer's, self_attn's and mlp's output (to find the first module that leaves P), every state byte (per layer view and
per row, marked when the row is the request's own), and the side buffers.

Two more probes:
- **Sparse-MLA probe (feature pass).** Computes production FA2, the installed path and fp32 references over the same q
  and KV. The references cover the valid keys, the planned `kv_len`, and FA2's own addressing.
- **Decode-step probe.** Records FA2's plan against the `kv_indices` row stride, and which slot ids the last row of
  each step reads past the step.

**Making it exact.** The first control run was not bit-identical. Production's prefill top-k
(`torch.ops._C.top_k_per_row_prefill`) returns the same set but in a different order on repeated calls: order drift on
73,484 of 135,015 rows, set drift on 0. FA2's plan reads past each row (section 4), so the order changes which keys
it attends. The harness replaces that order with a canonical one: score descending, then index ascending, keeping the
op's selection. It applies this identically in P, C and F. After that, C == P on every forward of every run.

`tests/handoff/check.sh` (a `run_all.sh` step) runs three configurations and asserts them with `check.py`:
- `qw_all`: prompts 200 / 14400 / 3001;
- `r16_prefill`: quickwins=all + MLA prefill;
- `mla_defect`: the old behaviour, to prove that the check detects the defect.

## 3. Bisect (logs: docs/logs/handoff/)

| configuration | prefill forwards (chunk paths) | P vs F | fast paths in decode |
|---|---|---|---|
| `kda_conv` | 4608 no-init, 1392 with init, 3001 | **bit-identical** (outputs, every KV byte, side buffers) | none |
| `mla_bmm` | same | **bit-identical** | none |
| `mla_index` | same | **bit-identical** | none |
| `mhc_aux` | same | **bit-identical** | none |
| `mhc_mean` | same | **bit-identical** | none |
| `idx_gate` | 13824 no-init (runtime check passed, fast path used), 3001 | **bit-identical** | none |
| `GLM53_MLA_PREFILL=1` | all | differs from the first MLA layer (layer 3 self_attn), by design (different kernel) | none |
| quickwins=all + MLA prefill | all | exactly the MLA-prefill-alone difference (same element counts) | none |

The prime suspect kda_conv (three `causal_conv1d_fn` launches on channel slices of the transposed SD-layout conv
state) is refuted. Its q/k/v, the 10-column conv state and the recurrent state are bit-identical on both
`has_initial_state` paths.

## 4. Culprit: GLM53_MLA_PREFILL, through the FA2 wrapper's process-wide kv_indices

Production plans FA2 (the builder, outside the CUDA graph) with `kv_len = ctx if ctx <= 2048 else 2048 + ctx % 4`, on
a `kv_indices` buffer with row stride W = 2048 (`kv_indptr = row * 2048`). The kpool indexer keeps
`pool_ids[:, :511]`, so the valid count is 2044 + ctx % 4 (docs/MLA_PREFILL.md section 6). Every FA2 row with
ctx >= 2048, in prefill and in decode, therefore attends 4 extra entries:
- 4 - ctx % 4 copies of slot 0;
- ctx % 4 entries **past its row**.

For rows inside a step, those extra entries are the next row's top-ranked slots. For the **last row of a step** they
are whatever an earlier `forward_mqa` with more rows left in the process-wide buffer.

Measured on the mini model (harness MLA probe, 15 prefill calls):
- FA2 against an fp32 reference over the valid keys: max row error 8-37 % (mean 0.5-0.7 %) on ctx >= 2048 rows, and
  0.25 % below 2048.
- FA2 against a reference that reads exactly what its plan addresses: **0.187 %**. The overrun explains FA2's whole
  excess error.
- The MLA prefill kernel against the valid-key reference: at most **0.19 %** on every row.

Production decode verify steps have 5, 6 or 8 rows. Their last row reads 1-3 entries at `kv_indices[M*W ...]`. No
decode step ever writes row 8, so production FA2 relies on the request's own prefill having written those rows.
`glm53_mla_prefill` (deploy-r16) serves every prefill >= 256 tokens and never wrote `kv_indices`. Its original
comment: "the -1 tail clamp and the copy into the FA2 wrapper's kv_indices are not needed". So after an MLA-prefill
prefill, decode's last row read slot ids left by an older FA2 call: another request's keys, which decode attended.

Decode-step probe (200 / 6000 / 3001 prompts, 69 decode steps, 34 of which have the last row reading past the step):

| configuration | steps whose past-the-step slots are NOT the request's |
|---|---|
| production (off) | 0 |
| MLA prefill, deploy-r16 behaviour | **9**: steps 27-31, the first decode steps after the 6000-token prefill (K = 7, 8 rows), reading slots 4608-4610 of the finished 200-token request |
| MLA prefill, fixed | 0 |
| quickwins=all + MLA prefill, fixed | 0 |

This matches the production pattern:
- the foreign reads cluster at the start of decode ("worst in [0, 128)"): while adaptive K stays at 7, the steps
  have 8 rows, and row 8 is never refreshed;
- B is a clean prefill on the exact kernel, while A attends foreign keys;
- quality_long, which runs no decode, could not see it.

A side observation, present in R15 as well: the same plan overrun makes the last row of a production FA2 **prefill**
step read past the step. In the idx_gate run, the 3001-token prefill's last row read slot 5856, which is not the
request's. That is the root defect in the vLLM patch, not an R16 regression (section 6).

## 5. Fix (glm53_mla_prefill.py)

After the kernel launch, the wrapper now writes the same bytes production's `forward_mqa` writes
(`kv_indices[:T*W].copy_(topk_slots.reshape(-1).clamp_(min=0).to(int32))`; stream-ordered after the kernel, which has
already read the prefix). What prefill leaves for decode in that buffer is then production's:
- `test_mla_integrate.py`: whole-buffer equality with production from a sentinel, and a regression guard with the
  write off;
- harness: on every prefill call, bit-equal to production's for the same call, and 0 foreign decode reads.

The cost is one int32 copy of T x 2048 per MLA layer, which production's FA2 path also pays. `STATE.write_kv_indices`
exists only so the harness can reproduce the defect; production never sets it.

## 6. Not fixed here: the plan itself

Production's FA2 still attends slot 0 and neighbouring rows' keys on every ctx >= 2048 row, in R15 as well. A
host-side `lens - 4` is wrong for the short-prefill path, where ctx = 2048 really has 2048 keys. The robust fix feeds
FA2 exact lengths, or a masked tail, for every row, prefill and decode. It changes R15's decode numerics and the
vLLM patch file that production mounts, so it needs its own validation: KL probe before and after on R15.

## 7. Proven, and how to verify in production

- **Bit-identical prefill handoff:** kda_conv, mla_bmm, mla_index, mhc_aux, mhc_mean. Each leaves outputs and every
  KV / state byte equal to production on both chunk paths and on single-chunk prompts.
- **idx_gate:** bit-identical at M = 13824. It keeps its runtime self-check, and turns itself off on a mismatch.
- **MLA prefill:** safe with this fix only. The state it leaves for decode (`kv_indices`) is production's. Its
  attention output is not bit-identical to FA2, but it is more accurate than FA2 against fp32.
- **kpool ring and smallops dconv:** a run with quickwins=all + MLA prefill (fixed) + `GLM53_KPOOL_RING=1` +
  smallops dconv (`r16b_fixed`) has the handoff intact (0 foreign decode reads, kv_indices equal on every call). Their
  own decode numerics are covered by their own tests, not by this harness.
- **Not proven by this work:** hostloop. `GLM53_DEC_HOSTLOOP` refuses to install inside the harness, because its
  fingerprint check sees the harness's `execute_model` wrapper. So the extra step from R16c to R16b (kpool ring +
  hostloop, 0.172 -> 0.294) is not explained here. In production, re-enable hostloop and the kpool ring one at a time
  after the prefill set, each with its own probe run.

Production check after the rollout, on an idle server: run `kpool_decode_consistency.py`. The KL mean must return to
the R15/R16d level of about 0.006 (top-1 about 98.5 %), and must be flat over positions, including [0, 128).
`quality_long.py` must stay at the noise floor against R15.

Run the check: `tests/handoff/check.sh [out]`. Build the mini model once: `python3 tests/handoff/build_mini.py`
(5.1 GiB, deterministic, MANIFEST.sha256).
