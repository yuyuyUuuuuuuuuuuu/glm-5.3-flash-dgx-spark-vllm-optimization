# KPOOL_DROP_LOWEST — the kpool indexer drops the LOWEST-scored pool, deterministically (`GLM53_KPOOL_DROP_LOWEST`)

Status 2026-09-30, branch `poolfix` (from `r16k` 2864d05 = what production runs). Production (nodeA/nodeB) was not
touched; **the fix is opt-in and default OFF: unset `GLM53_KPOOL_DROP_LOWEST` = the image's stock bytes and production
behaviour, byte-identical.**

## 1. The defect (confirmed by review)

GLM-5.3-Flash's sparse indexer pools every `index_kpool` (4) consecutive keys into one fp8 entry, so the
sparse-attention top-k selects `select_k = topk_tokens // index_kpool` = **512 pools** (topk_tokens 2048) and
`expand_pools_and_append_tail` expands them to tokens, appending the request's trailing incomplete pool as the last
one: the history region needs `select_k - 1` pools + the tail = `topk_tokens` tokens. The pool ids come from

* decode: `torch.ops._C.persistent_topk` (`vllm/model_executor/layers/sparse_attn_indexer_kpool.py` ~:815)
* prefill: `torch.ops._C.top_k_per_row_prefill` (~:559)

Both return the top-k **SET** deterministically but the **ORDER of the columns is not** (which pool lands in which
column varies per call; `tests/probe_persistent_topk_det2.py` on branch `fa2plan`, and the `info` line of
`tests/kpool_drop_lowest_det.py`: 32 distinct orders in 32 calls for both ops). The callers convert without sorting
(~:849 / ~:571) and keep `pool_ids[:, : select_k - 1]` (~:875 decode, ~:581 prefill) — they drop whichever pool sits in
the LAST column, an arbitrary pool per run, instead of the lowest-scored one. A wrongly dropped pool is one of the 512
best candidates of the sparse MLA, so the attended token set differs run to run: quality noise and
decode-vs-prefill consistency drift. (The op's SET is stable — the det test also proves that — so the defect is purely
the unsorted truncation.)

## 2. The fix (`GLM53_KPOOL_DROP_LOWEST=1`)

`overlay/patch_kpool_drop_lowest.py` inserts one module-level helper and replaces the two truncations with a call to
it (`pool_keep = _kpool_keep_highest_pools(logits, pool_topk, select_k - 1)`):

* `scores = logits.gather(1, pool_ids.clamp_min(0))` — the SAME fp32 logits the top-k op ranked (DeepGEMM returns
  fp32); invalid (-1) fills score `-inf`, so **a valid pool is never dropped while a -1 is kept** (a row with fewer
  valid pools than `select_k - 1` keeps all of them plus its -1s; the expand kernel maps `pid < 0` to -1 tokens as
  before, so this is semantics-preserving).
* `key = (order_preserving_uint32(score) - 2**31) * 2**32 + (2**32 - 1 - pool_id)` — the float bit trick: a
  non-negative float keeps its fp32 bits with the sign bit set, a negative one is bit-inverted, so the unsigned
  32-bit results order like the float (incl. ±0/+inf); subtracting 2³¹ makes them signed, which the int64 key needs
  (an unsigned high half shifted up wraps into the sign bit, and torch's SIGNED descending sort would put every
  positive score below every negative one; reading the uints as int32 instead breaks the order at 2³¹). The low half
  tie-breaks equal scores by the LOWER pool id, making every key distinct.
* `torch.sort(key, dim=-1, descending=True)` → first `select_k - 1` columns → gather the pool ids. The kept ids AND
  their order are a function of the SET {pool ids, scores} alone, not of the op's column order.

No host sync (gather/where/sort only; no `.item()`, `nonzero` or boolean-mask indexing) and CUDA-graph safe: shapes
are `[rows, select_k]` / `[rows, keep]`, static per captured batch shape; the temporaries join the graph's private
pool exactly like `pool_topk` and the persistent-topk workspace already do, and nothing is allocated at replay time
that capture did not allocate. The third kpool path (prefill with `positions is None`, `expand_pools_to_tokens` with
the full `select_k` columns behind a validity mask) drops nothing and is untouched, as is the non-kpool eager indexer.

## 3. Wiring (exactly like kdaqkv on r16k)

* `overlay/patch_tf_bundle.py` runs it (`GLM53_KPOOL_DROP_LOWEST`, after `patch_kpool_tail_ring.py`, which owns a
  different span of the same file); unset/empty → skip line, `0` → "stock, files untouched", anything else → the
  container stops (fail closed).
* `tools/deploy16/env.r16`: `#switch GLM53_KPOOL_DROP_LOWEST 0|1` (never appended by `apply_r16.sh`; unset = stock).
* `tools/deploy16/make_start_sh.py --stage r16l` (the default): the r16k stage (pinned `7736abf8…`) + the knob
  forwarded to BOTH ranks (head `-e` after `GLM53_KDA_STRIDED_QKV`, worker `serve_env_names`) + validation
  (empty/0/1; 1 needs `overlay/tf/overlay/patch_kpool_drop_lowest.py` and a `patch_tf_bundle.py` that runs it).
  `launcher/start.sh` = that r16l stage (`971c984c…`), `launcher/start.sh.pooldown.patch` = r16k → r16l.
* `tools/deploy16/env_r16.sh on|off kpooldown` (on refuses, .env unchanged, without the r16l kit), boot_checks switch
  section, `kit_chain.sh` K.2 chain, `off_equals_prev.sh` (=0 == prev kit; =1 → only `sparse_attn_indexer_kpool.py`
  differs), `test_boot_checks.sh` B.14, `test_kit_scripts.sh` S.15.

## 4. Operator steps

With the r16l kit installed (DEPLOY_R16L.md):

1. `tools/env_r16.sh on kpooldown` → appends exactly `GLM53_KPOOL_DROP_LOWEST=1` (refuses with .env unchanged if the
   installed start.sh/bundle/overlay are not r16l's — e.g. still r16k: that start.sh does not forward the knob and the
   line would be silently ignored).
2. `tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh r16l-pooldown-on` (restart both ranks).
3. `tools/boot_checks.sh` → `ok [head=worker] container env GLM53_KPOOL_DROP_LOWEST 1`, `info kpooldown ...: on`, and
   on each rank `[glm53-kpool-drop-lowest] sparse_attn_indexer_kpool.py: patched (GLM53_KPOOL_DROP_LOWEST=1: …)` +
   `[glm53-tf-bundle] patch_kpool_drop_lowest.py: applied (GLM53_KPOOL_DROP_LOWEST=1)`.
4. Back off: `tools/env_r16.sh off kpooldown` (+ idle-gated restart) → both ranks'
   `patch_kpool_drop_lowest.py: GLM53_KPOOL_DROP_LOWEST unset -> skipped (stock)`; the composed tree is r16k's byte
   for byte again (off_equals_prev). A full revert to r16k is `tools/revert_r16.sh` (removes the switch line with the
   env.r16 names).

## 5. Verified on nodeC (logs in `docs/logs/poolfix/`)

* `tests/kpool_drop_lowest_unit.py` (tests/gpu_run.sh): ALL OK — the shipped helper's kept SET == the top
  `select_k-1` by score and the order == score desc / pool id asc, against an independent numpy reference, for random
  scores, quantized ties, an all-equal row, -1 fills (valid < keep, exactly keep valid), ±0/±inf/extremes and a NaN
  row (graceful); the dropped pool is the lowest-scored selected one (ties: higher id).
* `tests/kpool_drop_lowest_det.py` (tests/gpu_run.sh): ALL OK — with the REAL ops
  (`torch.ops._C.persistent_topk` at production's select_k 512, `torch.ops._C.top_k_per_row_prefill`): 32 distinct
  column orders in 32 calls, stable SET, and the patched selection is BITWISE identical across 32 runs; kept set ==
  the top select_k-1 (op selection and the row's valid-range global top); -1 fills never displace a valid pool; no
  CUDA→host sync (`sync_debug_mode=error`); CUDA-graph capture + replay bitwise equal to eager, `[rows, select_k-1]`
  int64 shape kept.
* Real engine (tests/handoff/run.sh, mini model, FULL graphs + PIECEWISE prefill, DFlash2, six fresh engines =
  2 arms x {harness-canonical, production} prefill topk x 2 repeats, DEPLOY_R16L.md §3 for the full table):
  tokens identical in every run (24/24, same ids); decode ms/step p50 90.7-93.0 (on-off ≈ +0.4 %, within noise);
  consistency KL in the same 1e-3..1e-2 band in all runs, and IDENTICAL to 4 decimals across the three on runs while
  the off runs' varies; with production's raw prefill op the shadow P/C/F passes are IDENTICAL (flag on) vs
  DIFFERENT (flag off) — production's unpatched prefill is not bitwise reproducible, the patched one is; FULL
  capture 12/12 on every boot.

## 6. Risks / open items

* The helper adds one gather + one int64 sort of `[rows, select_k]` per indexer call (measured paired decode
  ms/step p50 90.7-93.0 on the mini model, on-off ≈ +0.4 % = within noise; production's wider rows are the same
  shape class, and the sort is over 512 int64 keys per row).
* Quality: dropping the LOWEST-scored pool is the intended semantics, but that it beats an arbitrary drop is NOT
  proven by the mini model (identical greedy outputs); production should A/B with tools/prodcheck/quality_probe.py
  after enabling.
* `torch.sort` on int64 inside FULL graphs relies on the caching allocator's graph pool (as every allocating op in
  the captured decode region already does); proven by the capture/replay test and the in-engine FULL capture run.
* The op's raw order nondeterminism itself is NOT fixed (only its consequence); a future upstream fix that sorts
  in-place would make this overlay's anchors drift → the bundle fails closed (container refuses to start), revert by
  `env_r16.sh off kpooldown`.

## Review fix (review, 2026-10-01): prefill ids are cu_seqlen_ks-relative

`top_k_per_row_prefill` returns pool ids RELATIVE to each row's `cu_seqlen_ks` (request-local pools), while the
prefill logits columns are chunk-global (the op ranks `logits[r, ks[r] + id]`). The first version gathered
`logits[r, id]` - the WRONG scores for every row of the 2nd+ request of a multi-request prefill chunk (random pool
dropped, deterministically). The helper now takes `col_off` (prefill passes `chunk.cu_seqlen_ks`; decode ids are
columns). Probe: `tests/kpool_drop_lowest_rowstart.py` (real op, two requests in one chunk: old gather wrong on
4/4 rows of request 2, fixed gather correct on 8/8). The engine A/B had missed it: one request per prefill chunk.
Cost (same probe, 50 calls per CUDA graph, GPU time): ~58-59 us per call at select_k=512, rows 5/8/16, per
indexer layer per decode step (int64 torch.sort alone ~26 us).

## opt-decodekit (2026-10-03): one Triton kernel, order-preserving by default (decode cost ~0.65 ms/step -> ~0, dvp regression gone)

Cost: the torch helper above costs ~58 us per call (about 15 small kernels incl. a 512-wide int64 `torch.sort`) =
**~0.65 ms per decode step** over the 11 MLA indexers (one CUDA graph of 11 calls, `tests/kpool_drop_lowest_bench.py`:
670-680 us per step at rows 8/32, pools 16k/262k). `_glm53_kpool_drop_lowest_kernel` (one program per row, the same
int64 keys): **order-preserving `fused` (the default) 23-27 us per step = the stock `.to(int64)[:, :511]` it replaces**,
bitwise == the torch order-preserving variant; `fusedsort` 42-44 us, BITWISE == the r16l torch helper (random scores,
exact ties, +-0.0, -1 fills, prefill col_off). Unexpected layouts fall back to the torch code.
`GLM53_KPOOL_DROP_LOWEST_ORDER` (nodeC only, not forwarded by start.sh): unset/fused | fusedsort | sorted | stock.

Quality (handoff mini, GLM53_MLA_PREFILL=1 + production decode features, KL(decode || fresh prefill), 96 positions):
stock 0.00566 / 0.00711 / 0.00841 (three runs), score-sorted drop-lowest 0.01027 / 0.01088 (torch helper / fusedsort
kernel: identical p95 0.0802 and max 0.1042 = same selection), **order-preserving drop-lowest 0.00814 / 0.00749** (ORDER=fused / unset). The
2026-10-01 production regression (0.0075 -> 0.0109) is reproduced by the score-SORTED order, not by dropping the
lowest pool: the order-preserving kernel keeps the correct set at stock dvp and stock cost.
(Runs off2/dloff2/dlstock/dlfs of this branch are void: a 0-byte worktree .so had silently disabled the MLA prefill;
tests/handoff/run.sh now refuses that.) Logs: docs/logs/opt_decodekit/kpool_drop_lowest_*.log, exl_ab/<arm>/.
