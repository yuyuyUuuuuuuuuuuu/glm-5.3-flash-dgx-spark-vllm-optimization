#!/usr/bin/env python3
"""Post-chain checks for the kpool tail-ring overlay, run INSIDE the production image after the launcher's overlay
chain (tests/kpoolring/chain_overlays.sh). CPU only.

1. every module the ring change can reach imports (the three edited files, the tail backend / slot mapping, the V1 and
   V2 block tables, kv_cache_utils / coordinator / managers, the PD connectors that size tail transfers);
2. Glm5NextTailCache.get_kv_cache_spec at production settings (block 4608, index_kpool 4, k = 7) and the ring table
   of upstream's test (k = 0, 1, 4, 7, 13), plus the block-size assert;
3. KV accounting with the image's own kv_cache_utils at production geometry (per rank, TP=2): 11 MLA layers (fp8,
   512 B/token), 11 kpool indexers (132 B/pool), 11 tails, 34 KDA layers (align mode, 7 speculative blocks), 5 DFlash2
   drafter SWA layers (4 KV heads/rank, bf16, window 2048), KV_CACHE_BYTES=16106127360, max_model_len 1,000,000.
   The stock tail spec (block 4) and the ring spec (block 16) must give the same groups' block ids, tensor sizes,
   num_blocks, scheduler/hash block sizes and per-request block demand.
Prints `mode=<on|off>` facts; exits 1 on any failed expectation for that mode.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from types import SimpleNamespace

import torch

MODE = os.environ.get("KPOOLRING_MODE", "on")
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


# ---------------------------------------------------------------------------------------------------- 1. imports
MODULES = [
    "vllm.models.glm5next.nvidia.ops.kpool_compress",
    "vllm.model_executor.layers.sparse_attn_indexer_kpool",
    "vllm.models.glm5next.nvidia.attention",
    "vllm.v1.attention.backends.mla.indexer",
    "vllm.v1.worker.block_table",
    "vllm.v1.worker.gpu.block_table",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.kv_cache_coordinator",
    "vllm.v1.core.single_type_kv_cache_manager",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker",
]
mods = {}
for name in MODULES:
    try:
        mods[name] = importlib.import_module(name)
        check(True, f"import {name}")
    except Exception as exc:  # noqa: BLE001
        check(False, f"import {name}: {type(exc).__name__}: {exc}")

kc = mods.get("vllm.models.glm5next.nvidia.ops.kpool_compress")
idx = mods.get("vllm.model_executor.layers.sparse_attn_indexer_kpool")
attn = mods.get("vllm.models.glm5next.nvidia.attention")
if kc is not None and idx is not None and attn is not None:
    # without a GPU vllm.triton_utils hands back the plain function (no JITFunction.fn)
    def params(k):
        return list(inspect.signature(getattr(k, "fn", k)).parameters)

    dec_params = params(kc._kpool_decode_update_batched_kernel)
    seed_params = params(kc._kpool_tail_seed_kernel)
    idx_src = inspect.getsource(idx)
    spec_src = inspect.getsource(attn.Glm5NextTailCache.get_kv_cache_spec)
    print(f"mode={MODE} decode kernel constexprs: {[p for p in dec_params if p.isupper()]}")
    print(f"mode={MODE} seed kernel constexprs: {[p for p in seed_params if p.isupper()]}")
    ring_on = MODE == "on"
    check(("RING" in dec_params) == ring_on, f"decode kernel RING constexpr present == {ring_on}")
    check("TAIL_BLOCK_ELEMS" in seed_params, "seed kernel addresses the padded stride (#57477 applied)")
    check(("tail_kv_cache.shape[2],\n" in idx_src) == ring_on, f"seed call passes the ring == {ring_on}")
    check(("next_power_of_2" in spec_src) == ring_on, f"tail spec sizes the ring == {ring_on}")

# ---------------------------------------------------------------------------------------------------- 2. the spec
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

KPOOL, HEAD_DIM, BLOCK = 4, 128, 4608


def tail_spec(block: int, num_spec: int, prefix: str) -> KpoolTailSpec:
    """The installed Glm5NextTailCache.get_kv_cache_spec. Without a GPU VllmConfig() cannot infer a device, so the
    layer is built without its __init__ (which only registers it in the compilation config) and given the attributes
    that __init__ sets; the GPU test (test_kpool_ring_gpu.py) builds it through the real __init__."""
    cache = attn.Glm5NextTailCache.__new__(attn.Glm5NextTailCache)
    torch.nn.Module.__init__(cache)
    cache.head_dim, cache.dtype, cache.prefix = HEAD_DIM, torch.bfloat16, prefix
    cache.cache_config = SimpleNamespace(block_size=block)
    cache._index_kpool = KPOOL
    return cache.get_kv_cache_spec(SimpleNamespace(num_speculative_tokens=num_spec))


if attn is not None:
    table = []
    for k in (0, 1, 4, 5, 7, 13):
        s = tail_spec(BLOCK, k, f"tail.k{k}")
        table.append((k, s.block_size, s.sliding_window, s.page_size_bytes))
    print(f"mode={MODE} (num_spec, block_size, sliding_window, unpadded page B) at block {BLOCK}: {table}")
    # upstream test table (0, 4), (1, 8), (4, 8), (7, 16), (13, 32) plus k=5: cdiv(9, 4) = 3 -> 4 pools -> 16
    want = {0: 4, 1: 8, 4: 8, 5: 16, 7: 16, 13: 32} if MODE == "on" else {k: 4 for k in (0, 1, 4, 5, 7, 13)}
    check(
        all(bs == sw == want[k] for k, bs, sw, _ in table),
        f"ring sizes {[(k, bs) for k, bs, _, _ in table]} == {sorted(want.items())}",
    )
    s7 = tail_spec(BLOCK, 7, "tail.prod")
    check(s7.page_size_bytes == 2 * s7.block_size * HEAD_DIM * 2, f"tail unpadded page {s7.page_size_bytes} B")
    if MODE == "on":
        for blk in (640, 3584, 4608):
            check(blk % tail_spec(blk, 7, f"tail.b{blk}").block_size == 0, f"ring divides block {blk}")
        try:
            tail_spec(100, 7, "tail.bad")
            check(False, "block 100 with ring 16 must assert")
        except AssertionError as exc:
            check("must be a multiple of the tail ring (16)" in str(exc), f"block 100 asserts: {exc}")

# ---------------------------------------------------------------------------------------------------- 3. KV accounting
from vllm.v1.core import kv_cache_utils as U  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

KV_BYTES = 16106127360
MAX_LEN = 1_000_000
NUM_SPEC = 7
MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
KDA_LAYERS = [i for i in range(45) if i not in MLA_LAYERS]


def vcfg(mnbt: int) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, mamba_cache_mode="align", enable_prefix_caching=True,
            prefix_match_unit=None, block_size=BLOCK,
        ),
        model_config=SimpleNamespace(max_model_len=MAX_LEN),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt, disable_hybrid_kv_cache_manager=False),
        max_concurrent_batches=2,  # V2 model runner + async scheduling, PP=1: pp_size + 1
        max_in_flight_tokens=2 * mnbt,
        num_speculative_tokens=NUM_SPEC,
        kv_transfer_config=None,
        # production: DFlash2 drafter with 5 decoder layers (read by the launcher's GLM53_DRAFT_KV_COMPACT=1 preflight,
        # `_glm53_draft_kv_compact`, which deploy-r16's launcher env turns on)
        speculative_config=SimpleNamespace(num_speculative_tokens=NUM_SPEC, method="dflash", use_dflash=lambda: True,
                                           draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                                               num_hidden_layers=5))),
    )


def spec_dict(tail: KpoolTailSpec) -> dict:
    mla_page = BLOCK * 512
    shapes = MambaStateShapeCalculator.kda_state_shape(2, 64, 128, conv_kernel_size=4, num_spec=NUM_SPEC)
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    d = {}
    for i in MLA_LAYERS:
        d[f"model.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=512, dtype=torch.uint8, cache_dtype_str="fp8"
        )
        d[f"model.layers.{i}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=132, dtype=torch.uint8, compress_ratio=KPOOL
        )
        d[f"model.layers.{i}.self_attn.indexer.tail_cache"] = tail
    for i in KDA_LAYERS:
        d[f"model.layers.{i}.self_attn"] = MambaSpec(
            shapes=tuple(shapes), dtypes=dtypes, block_size=BLOCK, page_size_padded=mla_page,
            mamba_cache_mode="align", num_speculative_blocks=NUM_SPEC,
        )
    for j in range(5):
        d[f"drafter.layers.{j}.self_attn.attn"] = SlidingWindowSpec(
            block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, sliding_window=2048
        )
    return d


def account(tail: KpoolTailSpec, mnbt: int) -> dict:
    cfg = vcfg(mnbt)
    groups = U.get_kv_cache_groups(cfg, spec_dict(tail))
    kvc = U.get_kv_cache_config_from_groups(cfg, groups, KV_BYTES)
    per_group = []
    for g in kvc.kv_cache_groups:
        s = g.kv_cache_spec
        inner = next(iter(s.kv_cache_specs.values())) if hasattr(s, "kv_cache_specs") else s
        per_group.append(
            (type(inner).__name__, len(g.layer_names), inner.block_size, s.page_size_bytes,
             -(-s.max_memory_usage_bytes(cfg) // s.page_size_bytes))
        )
    sched, hashb = U.resolve_kv_cache_block_sizes(kvc, cfg)
    conc = U.get_max_concurrency_for_kv_cache_config(cfg, kvc)
    tail_inner = [
        next(iter(g.kv_cache_spec.kv_cache_specs.values()))
        for g in kvc.kv_cache_groups
        if hasattr(g.kv_cache_spec, "kv_cache_specs")
        and isinstance(next(iter(g.kv_cache_spec.kv_cache_specs.values())), KpoolTailSpec)
    ][0]
    return {
        "num_blocks": kvc.num_blocks,
        "tensor_bytes": sorted((t.size, tuple(t.shared_by)) for t in kvc.kv_cache_tensors),
        "groups": per_group,
        "scheduler_block": sched,
        "hash_block": hashb,
        "blocks_per_1M_request": sum(p[4] for p in per_group),
        "capacity_tokens": int(conc * MAX_LEN),
        "tail_unpadded_page": tail_inner.unpadded_page_size_bytes,
        "tail_padded_page": tail_inner.page_size_bytes,
    }


if attn is not None:
    stock = KpoolTailSpec(  # exactly the image's stock Glm5NextTailCache.get_kv_cache_spec
        block_size=KPOOL, num_kv_heads=1, head_size=2 * HEAD_DIM, head_size_v=0, dtype=torch.bfloat16,
        sliding_window=KPOOL,
    )
    live = tail_spec(BLOCK, NUM_SPEC, "tail.acct")
    for mnbt in (7168, 16384, 18432):
        a, b = account(stock, mnbt), account(live, mnbt)
        for key in ("num_blocks", "scheduler_block", "hash_block", "blocks_per_1M_request", "capacity_tokens",
                    "tail_unpadded_page", "tail_padded_page"):
            print(f"mode={MODE} mnbt={mnbt} {key}: stock-tail {a[key]}  this-mode-tail {b[key]}")
        print(f"mode={MODE} mnbt={mnbt} groups (type, layers, block, page B, blocks/1M req):")
        for ga, gb in zip(a["groups"], b["groups"]):
            print(f"    stock {ga}  |  now {gb}")
        same = {k: a[k] == b[k] for k in ("num_blocks", "tensor_bytes", "scheduler_block", "hash_block",
                                           "blocks_per_1M_request", "capacity_tokens", "tail_padded_page")}
        check(all(same.values()), f"mnbt={mnbt}: ring spec leaves the KV pool unchanged {same}")
    check(account(stock, 7168)["num_blocks"] == 583, "per-block bytes reproduce production's 583 block ids")
    # After the launcher chain (patch_mamba_align_state_free.py) an align-mode mamba group holds
    # 1 + max_concurrent_batches + num_speculative_blocks blocks; with MNBT 16384 the model then reproduces the
    # production boot line "GPU KV cache size: 2,003,436 tokens" (KV_CACHE_BYTES=16106127360, max_model_len 1M).
    if "max_concurrent_batches" in inspect.getsource(MambaSpec.max_memory_usage_bytes):
        cap = account(stock, 16384)["capacity_tokens"], account(live, 16384)["capacity_tokens"]
        check(cap == (2003436, 2003436), f"MNBT 16384 capacity (stock tail, this mode) {cap} == production log 2,003,436")

print(f"mode={MODE} GLM53_DRAFT_KV_COMPACT={os.environ.get('GLM53_DRAFT_KV_COMPACT', '<unset>')!r} "
      f"GLM53_APC_DRAFTER_LOW_PRIORITY={os.environ.get('GLM53_APC_DRAFTER_LOW_PRIORITY', '<unset>')!r} (from the launcher env)")
print(f"mode={MODE} chain_check: {'ALL OK' if not fails else 'FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
