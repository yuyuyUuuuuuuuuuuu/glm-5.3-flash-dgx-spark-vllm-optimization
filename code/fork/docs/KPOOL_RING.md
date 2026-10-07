# KPOOL_RING — kpool indexer tail ring sized for speculative decoding (`GLM53_KPOOL_RING`)

Status 2026-09-28, branch `kpoolring` (from `deploy-r15`). Backport of upstream vLLM
[#58454](https://github.com/vllm-project/vllm/pull/58454) (merged 2026-09-25, `2617fe93`) as a fail-closed overlay.
Built and tested on nodeC against the production image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (vLLM `0.1.dev20051+g487ecf187`) and the
read-only copy of the production launcher (`${HOME}/tf-exl3-assets/prod-launcher/`). **Production (nodeA/nodeB)
was not touched. Nothing is installed there.** Every number below comes from a committed log under
`docs/logs/kpoolring/`.

## 1. Summary

- **Bug:** production's DFlash2 speculative decoding writes wrong pooled indexer keys for about half of the pools of
  the text the model generates itself. These keys are what the sparse indexer scores to pick top-k context once a
  context passes `index_topk` = 2048 tokens. So on long contexts the model looks at the wrong parts of its own
  earlier output. The keys stay in the indexer cache and are reused through the prefix cache on later turns of the
  same conversation. There is no crash and no error log. Quality degrades silently.
- **Fix:** the per-request tail ring holds `kpool * next_pow2(cdiv(kpool + num_spec, kpool))` = **16** slots instead of
  `index_kpool` = 4. This is upstream's formula and kernels. The overlay edits 3 files of the image.
- **Cost:** none measurable. The KV pool is unchanged (583 block ids, 2,003,436 tokens at MNBT 16384) because the ring
  fits inside the indexer page it already occupies. The V2 runner's tail block table shrinks by 4x.
- **Default off:** with `GLM53_KPOOL_RING` unset, the image comes out byte-identical to what `deploy-r15` produces
  today (checked over the full production overlay chain).
- **Rollout:** 2 files, a 2-line `start.sh` edit (both ranks) and one `.env` line, then a restart. See section 7. To
  revert, empty the variable and restart (section 8).

## 2. The bug

The GLM-5.3-Flash sparse indexer (11 MLA layers) pools every `index_kpool` = 4 consecutive keys into one fp8 entry.
The raw K and gate score of the pool in progress live in a per-request **tail block**. That block is a ring addressed
by `pos % ring`, and on the image `ring == index_kpool == 4` (`Glm5NextTailCache.get_kv_cache_spec`,
`KpoolTailSpec(block_size=index_kpool)`).

A verify step runs `1 + k` tokens per request (k ≤ 7: `num_speculative_tokens` 7, adaptive K ∈ {4, 5, 7}) through
`_kpool_decode_update_batched_kernel` before acceptance is known. For each token in position order, the kernel
(a) compresses the pool from the ring if the token completes one (`pos % 4 == 3`), then (b) stashes the token at
`pos % 4`. The stashes of rejected drafts are never undone.

Take the example from the upstream PR at production shape (`test_rejected_pool_completing_draft_directed`):

```
positions 0..5 committed; verify step = [6 | drafts 7..13]; only 6 is accepted (draft 7 rejected)
ring (4 slots) after the step:  slot0 <- 12  slot1 <- 13  slot2 <- 10  slot3 <- 11
next step re-verifies 7 (pool 1 = positions 4..7): reads slots 0,1,2 expecting K(4),K(5),K(6), gets drafts 12,13,10
```

Pool 1 is then written with the wrong keys (124 of its 128 fp8 key bytes differ from the reference). It is never
recomputed, because positions 4 to 6 are committed and never re-run.

**When it fires:** a verify step corrupts exactly one pool when both of these hold: the next pool to complete already
holds a committed token (the new length P' is not a multiple of 4), and the same step stashed a draft at or past that
pool's last position + 1 (the draft lands on the slot of the pool's first member). With k drafts and a accepted, this
means `a ≤ k − 5 + (P' mod 4)`. For k = 7 that is any step that accepts at most 3, 4 or 5 drafts (depending on
alignment) and does not end on a pool boundary. In other words, most steps. The simulation below corrupts
117 to 122 of about 240 pools.

**Not affected:**
- Prefill-built pools, i.e. prompts. `_kpool_compress_insert` pools from the batch and does not read the ring.
- Plain decode (k = 0). `test_plain_decode_only`: 0 wrong pools for every build.

**Effect:** the fp8 pooled key is only the indexer's scoring key, which picks the top 512 pools (2048 tokens) out of
the context. The MLA attention itself reads true KV. So the damage is wrong context selection over model-generated
text, on every context longer than 2048 tokens.

## 3. Evidence (nodeC, production image)

GPU: `tests/kpoolring/test_kpool_ring_gpu.py` → `docs/logs/kpoolring/test_kpool_ring_gpu.log`, **16/16**, peak 9.2 MiB.

The test uses no hand-copied kernel. Each build is the image's own files after the **shipped** overlay scripts have
run on copies:

- `prod` = image + seed-stride. This is what production runs today.
- `ring` = `prod` + this overlay.
- `upstream` = vLLM's `kpool_compress.py` at `2617fe93`, copied verbatim, as a cross-check.

Each build's ring size comes from its own patched `attention.py`, and its seed argument is the expression its own
indexer file passes. The tail is an `as_strided` view with production's padded block stride (idx page 152,064 B),
filled with garbage beforehand. Tail slots come from the image's `compute_kpool_tail_slot_mapping` and decode
batches from the image's scatter helpers. The reference is the prefill writer over the accepted token stream, and a
sequential accepted-only decode was also checked equal to it.

| test | prod (ring 4) | ring build (16) | upstream (16) |
|---|---|---|---|
| directed PR case, k = 7 | pool 1 WRONG (124/128 key bytes) | exact | exact |
| simulation, 4 concurrent requests (chunked prefill, prefix resume into a garbage tail block, 2-token prompt joining late), adaptive k ∈ {4,5,7}, 3 seeds, ~320 rejecting verifies each | **118/244, 123/251, 123/242 pools wrong** (117/121/122 of them decode-built) | 0 | 0; KV cache bitwise equal to the ring build |
| mixed batch (one request plain decode, others verify) | 84/204 | 0/204 | 0/204 |
| plain decode only | 0/167 | 0/167 (KV bitwise == prod) | 0/167 |
| prefix resume with a 1-token fresh suffix, then spec decode | 32/100 | 0/100 | 0/100 |

The failure is caused by the ring size and nothing else:

- The ring build's own kernel, run at ring 4, reproduces prod bit for bit (`ring-kernel@ring4`: same pools wrong,
  KV `torch.equal`).
- At ring == pool size, the patched kernel is bitwise identical to prod over upstream's 5 cases and a 20-seed fuzz.
- **Review addition:** `test_ring_size_bound` shows that the bound that matters is `ring ≥ kpool + num_spec` (11):
  - ring 8 corrupts 27/244 pools;
  - ring 12 (≥ 11, not a power of two) and ring 16 are exact and bitwise equal to upstream.

  Upstream rounds up to a power of two only so that the ring divides the attention block.

The other GPU tests are ports of upstream's tests:

- decode writer == prefill writer for pool 4/16 and ring of 1 or 2 pools;
- upstream's rejected-draft test (ring 4 corrupts, ring 8 exact);
- the 5 reference cases plus fuzz against an independent torch reference at ring 16 and 32;
- leading invalid tail slot;
- the `TailRingMirror` arithmetic;
- the image's `KpoolTailMetadataBuilder.build` puts every token at `own_block * 16 + pos % 16`.

The pre-#58454 decode wrapper asserts `ring == pool_size`, so a spec-only fix would have been rejected
(`test_prod_kernel_rejects_a_ring`).

CPU: `tests/kpoolring/test_patch_kpool_tail_ring.py` → `test_patch_kpool_tail_ring.log`, 28 checks. It covers install
states, fail-closed paths, both re-run orders with seed-stride, the bundle registration, and that the patched decode
kernel and wrapper are AST-equal to upstream `2617fe93`.

Chain: `tests/kpoolring/chain_overlays.sh on|off` → `chain_on.log` / `chain_off.log` (commit `a0c0846`). It replays the
launcher's `GLM53_OVERLAY_ORDER` (21 patches, the exact in-container loop both ranks run) on a fresh `--rm` container
of the production image, with the production env and the branch's bundle overlay dir:

- every patch exits 0;
- a second run leaves all 2,423 `vllm/*.py` byte-identical;
- `tests/kpoolring/chain_check.py` imports every reachable module (including the PD connectors), builds the tail spec
  at production settings, and runs the image's own KV accounting (section 5).

## 4. The fix and what the overlay edits

`overlay/patch_kpool_tail_ring.py` edits 3 files and preflights all of them before writing any:

1. **`models/glm5next/nvidia/attention.py`:** `Glm5NextTailCache.get_kv_cache_spec` returns
   `block_size = sliding_window = ring` using upstream's formula and upstream's
   `cache_config.block_size % ring == 0` assert. This image's `[1 head × 2·head_dim]` packing is kept. The spec also
   logs `[glm53-kpool-ring] indexer tail ring: 16 slots per request (...)` once per process.
2. **`models/glm5next/nvidia/ops/kpool_compress.py`:** decode kernel and wrapper. They get a `RING` constexpr exactly
   as upstream has it:
   - stash at `pos % RING`;
   - block `tail_slot // RING`;
   - pool reads `(pool_start + s) % RING`;
   - the wrapper takes `ring = tail.shape[2]` and asserts `ring ≥ pool_size`, `ring % pool_size == 0`.

   The region between the file's two decode banners is pinned by sha256, both pristine and patched.
3. **`model_executor/layers/sparse_attn_indexer_kpool.py`:** the prefill seed call passes `tail_kv_cache.shape[2]`
   (the ring) as the seed kernel's `kpool` argument.

**Why the seed is fixed at the call site and not inside `_kpool_tail_seed_kernel` as upstream does:** that kernel's
text is owned by `patch_kpool_tail_seed_stride.py`, a verbatim upstream MiaAI-Lab #264 / vLLM #57477 backport that
re-verifies it byte for byte on every run. Leaving it untouched keeps both overlays idempotent in either order.
The review checked that this is exactly equivalent for everything the decode path reads:

- **Addressing.** The stride-fixed seed kernel uses its `KPOOL` argument for:
  - the block `t // KPOOL`;
  - the row `t % KPOOL`;
  - the look-ahead `tslot[i + KPOOL]` with `ahead // KPOOL == blk`.

  The score half comes from `KPOOL_HEAD = tail.stride(1)`, which is `ring·head_dim` for the runner's padded view.
  Tail slots come from `compute_kpool_tail_slot_mapping`, which uses the spec's `block_size`, so they are
  `block·16 + pos % 16`. With `KPOOL = 16` the block and row are therefore exactly upstream's `RING` addressing.
- **Look-ahead.** Upstream's look-ahead stays at `kpool` (4), so it seeds each request's last 4 prefill tokens. The
  ring build seeds the last 16. Every seeded row holds the true K and score of its own position, and the 16 positions
  map to 16 distinct rows. The rows of the last 4 positions are identical to upstream's
  (`test_prefill_seed_ring_padded_stride_multi_request`: 58 rows written versus upstream's 22, in-progress rows
  equal, nothing outside the requests' own rows touched).
- **Reads.** A decode completion at c reads positions c−3 … c−1. Decode starts at the prefill length L, and the first
  completion is at c ≥ L with c−3 ≥ ⌊L/4⌋·4 ≥ L−3. So the only prefill rows ever read are among the last 4. The
  extra 12 rows are never read.
- **End to end.** The simulations run the ring build's seed and upstream's seed through the same decode schedule. Their
  KV caches come out bitwise equal (`torch.equal` in every simulation test).

**Other sites audited** in the image and in the launcher overlays, for anything that addresses the ring with
`index_kpool`:

- `compute_kpool_tail_slot_mapping` / `KpoolTailMetadataBuilder` use `kv_cache_spec.block_size`.
- The `[glm53-kpool-tail-slotmap]` clamp in `v1/worker/block_table.py` and the V2 generic slot kernel use the
  group's block size. Their output for the tail is overwritten by the builder.
- The kernel block size comes from `KpoolTailBackend` `MultipleOf(1)`, which equals the manager block of 16, so no
  split.
- The padded strided view (`v1/worker/gpu/attn_utils.py`) derives only the block stride from the padded page.
- The mooncake connector sizes the tail halves from `unpadded_page_size_bytes // 2`, and the offloading connector
  from `unpadded_page_size_bytes`. Both are spec-derived. PD is not used in production.
- `flashinfer_mla_sparse*.py` use `index_kpool` only for the raw tail tokens in the main MLA cache, not the ring.
- `KpoolTailSpec` opts out of prefix caching, so the tail's `sliding_window` only feeds the kv-event metadata.
  `patch_hybrid_prefix_hit.py`'s DFlash replay and `patch_apc_per_group_retention.py` select the drafter by exact
  type `SlidingWindowSpec`, and `patch_kv_capacity_log.py` counts the tail as 0 ids.
- Prefix-hit replay: hits land on 4608-token (or hash-block) boundaries, which are pool aligned. So the in-progress
  pool at the prompt end lies in the freshly recomputed suffix, which is at least 1 token, and gets seeded. Rows of
  positions before the hit hold a previous owner's garbage and are never read
  (`test_prefix_resume_one_token_suffix`: 0/100).
- No other file in `vllm/` touches the tail ring. The AMD `kpool_compress.py` is not in this image.

## 5. Memory and KV accounting

The tail co-owns the indexer tensor: each tail block is a block id of the shared pool, padded to the indexer page.
The ring changes how many bytes of that page are used, not how many pages there are.
`chain_check.py` runs the image's own `kv_cache_utils` at production geometry (11 MLA, 11 indexer, 11 tail,
34 KDA in align mode with 7 speculative blocks, 5 DFlash2 SWA layers, `KV_CACHE_BYTES` 16106127360,
`max_model_len` 1,000,000) for the stock tail spec and the ring spec:

| | stock (ring 4) | ring 16 |
|---|---|---|
| block ids (num_blocks) | 583 | 583 |
| GPU KV capacity at MNBT 16384 (the production boot line) | 2,003,436 tokens | 2,003,436 tokens |
| scheduler / hash block | 4608 / 1152 | 4608 / 1152 |
| block ids per 1M-token request | 291 | 291 |
| tail padded page (bytes of the indexer page it rides in) | 152,064 B | 152,064 B |
| tail bytes actually used per block (unpadded page) | 2,048 B | 8,192 B |
| tail group block size | 4 | 16 |

MNBT 7168 and 18432 are equal too, and so are all KV tensor sizes. Tail memory is still 1 block id per request.

**Non-KV:** the V2 runner sizes each group's block table as `cdiv(max_model_len, block_size)` columns, the one-block
tail included. The tail width goes from 250,016 to 62,504 columns: `BlockTables` 8,988,672 B → 2,989,056 B at
max_num_reqs 4 (5.72 MiB less per rank, linear in max_num_seqs).

**Kernel cost:** the decode kernel loads and stores the same number of rows as before, and gets one extra Triton
specialization (the `RING` constexpr), compiled at boot warmup. The seed writes up to 16 rows per request per
prefill per layer instead of 4.

## 6. Install contract

- The bundle runs the overlay only when `GLM53_KPOOL_RING` is non-empty. The overlay itself accepts only `1` and
  refuses anything else.
- `patch_tf_bundle.py` registers it after `patch_kpool_tail_seed_stride.py`. The overlay **requires** the #57477 seed
  (`GLM53_KPOOL_SEED_STRIDE=1`, already on in production). Without it the overlay exits with `SystemExit` and the
  container does not start: no ring size is correct with the dense seed addressing.
- Each file is in one of three states:
  - pristine → patched;
  - patched (marker plus the exact patched text or region sha) → `already present`;
  - anything else (partial marker, drifted anchor, a new image) → `SystemExit` and **no file written**.

  Writes are atomic replaces and the pyc files are cleared. Re-runs in either order with seed-stride are no-ops
  (CPU test and chain run 2).
- A new image that already contains #58454 fails the anchors and stops the container. Unset the variable for such an
  image.
- **Default-off is byte-identical to production today.** `tests/kpoolring/off_equals_base.sh` →
  `off_equals_deploy-r15.log` runs the full launcher chain with `GLM53_KPOOL_RING` unset for this branch and for
  `deploy-r15`. The 24 `vllm/` files the chain changes have identical sha256 in both, and every other file is the
  image's own. The bundle only prints `patch_kpool_tail_ring.py: GLM53_KPOOL_RING unset -> skipped (stock)`.

## 7. Rollout (operator, nodeA, not done here)

Both ranks must get the variable. Each rank's worker computes its own KV-cache spec, and a rank without it would size
a 4-slot tail group while the other sizes a 16-slot one. The launcher passes env to the **head** only from its
explicit `-e` list (the R1 incident). The worker gets env from the `serve_env` name list.

0. **Wait for idle.** `num_requests_running` and `num_requests_waiting` must both be 0. Check that
   `GLM53_KPOOL_SEED_STRIDE=1` is in `.env`.
1. **Baseline**, before any change, on nodeA:
   `python3 tools/prodcheck/kpool_decode_consistency.py ~/kpool-before.json`.
   This probe measures the bug directly. For 3 fixed real-text prompts (~6k tokens) it generates 1,536 tokens with
   spec decode (temperature 0, logprobs 5). It then prefills prompt + generated ids under a separate `cache_salt`, so
   every pool is prefill-built, and compares the two distributions position by position (KL over the top-5 union,
   top-1 agreement, per position bucket). It refuses to run while the server is busy and never prints the key. It
   was validated offline against a mock of the image's completion protocol
   (`tests/kpoolring/test_consistency_probe_mock.py`, including a mutation check of the position alignment). It has
   **not** been run against a live server. The prefill-path probes `quality_probe.py` / `quality_long.py` cannot see
   this bug; run them only as a no-regression check.
2. **Files.** Back up the old files, then copy them into `~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/` and check sha256
   against the branch:

   | branch file | destination | sha256 prefix |
   |---|---|---|
   | `overlay/patch_kpool_tail_ring.py` | `overlay/tf/overlay/patch_kpool_tail_ring.py` | `c3b1c86ab4f1bd37` |
   | `overlay/patch_tf_bundle.py` | `overlay/patch_tf_bundle.py` (the launcher's `TF_BUNDLE_PATCH_HOST`) | `723123edd94122fc` |

   `start.sh` copies `overlay/tf/` and `patch_tf_bundle.py` to the worker on every start, so both ranks run the same
   files. `validate_overlay_artifacts` does not look inside `overlay/tf/overlay/`. If the ring file is missing while
   the variable is set, the containers stop at boot after the old pair is already down. So check that the file is
   present before restarting.
3. **start.sh:** `python3 tools/prodcheck/edit_start_env_kpoolring.py` prints the dry-run diff. Then run it again with
   `--apply`. It makes 2 anchored edits:
   - adds `GLM53_KPOOL_RING` to the worker `serve_env` list;
   - adds `-e GLM53_KPOOL_RING="${GLM53_KPOOL_RING:-}"` after the head's `GLM53_KPOOL_SEED_STRIDE` line.

   It keeps a `start.sh.bak-kpoolring-<ts>` backup, runs `bash -n`, and a second run is a no-op. It was verified on
   the read-only copy of the production `start.sh` (`edit_start_env_dryrun.log`) and applied twice to a scratch copy
   (`edit_start_env_apply_scratch.log`).
4. **.env:** back it up, then add `GLM53_KPOOL_RING=1`.
5. **Restart** both nodes when idle: `~/tf-exl3-deploy/restart2.sh kpoolring` (stop, 25 s, memprep, GID fix, start).
   Containers are recreated from the image on every start.
6. **Check each rank** (head `docker logs`, and the worker's on nodeB):
   - `[glm53-kpool-ring] kpool_compress.py: patched; sparse_attn_indexer_kpool.py: patched; attention.py: patched`
   - `[glm53-tf-bundle] patch_kpool_tail_ring.py: applied (GLM53_KPOOL_RING=1)`
   - `[glm53-kpool-ring] indexer tail ring: 16 slots per request (index_kpool=4, num_speculative_tokens=7, block_size=…)`
   - `GPU KV cache size: 2,003,436 tokens`, unchanged.

   A rank that prints `GLM53_KPOOL_RING unset -> skipped (stock)` did not get the variable: stop and fix step 3 or 4.
7. **After:**
   - run `python3 tools/prodcheck/kpool_decode_consistency.py ~/kpool-after.json`;
   - run `... --compare ~/kpool-before.json ~/kpool-after.json`: KL should drop and stop growing with generated
     position. The size of the drop on the real model is not known in advance, because decode and prefill also
     differ by numerics;
   - `quality_long.py` before/after should stay at noise, since prefill pools are unchanged;
   - `bench_decode.py` should be unchanged, since there is no extra work per step.

The restart empties the prefix cache, which drops every pool built with the 4-slot ring. From then on,
decode-generated pools equal prefill-built ones.

## 8. Revert

Set `GLM53_KPOOL_RING=` (empty) or remove the line from `.env`, then restart. Both ranks print
`patch_kpool_tail_ring.py: GLM53_KPOOL_RING unset -> skipped (stock)` and the image files stay stock (fresh
containers). The `start.sh` edit and the new files are inert when the variable is empty. For a full revert, restore
`start.sh.bak-kpoolring-*`, the old `overlay/patch_tf_bundle.py` and `.env.bak-*`, and delete
`overlay/tf/overlay/patch_kpool_tail_ring.py`. If seed-stride ever has to be turned off, turn this one off first: it
refuses to install without seed-stride.

## Files

- `overlay/patch_kpool_tail_ring.py` — the overlay (3 files, preflight-all-then-write, sha-pinned decode region).
- `overlay/patch_tf_bundle.py` — `PATCHES += ("patch_kpool_tail_ring.py", "GLM53_KPOOL_RING")` after seed-stride.
- `tools/prodcheck/edit_start_env_kpoolring.py` — the `start.sh` edit (both ranks).
- `tools/prodcheck/kpool_decode_consistency.py` — the before/after decode-vs-prefill probe.
- Tests:
  - `tests/kpoolring/test_kpool_ring_gpu.py` (GPU, `tests/gpu_run.sh`, run under `flock /tmp/tf-gpu-bench.lock`);
  - `test_patch_kpool_tail_ring.py` (CPU);
  - `chain_overlays.sh` + `chain_check.py` (production chain replay);
  - `off_equals_base.sh` (default-off identity);
  - `test_consistency_probe_mock.py` (probe, offline);
  - `ref/kpool_compress_vllm58454.py` and `ref/pr58454.diff` (upstream, verbatim).
- Logs: `docs/logs/kpoolring/`.
