"""Shared inputs / reference / production-FA2 baseline for the sparse-MLA prefill work (docs/MLA_PREFILL.md).

Shapes are GLM-5.3-Flash on one TP=2 rank: 32 heads, kv_lora_rank 512, qk_rope_head_dim 0, index_topk 2048,
index_kpool 4, fp8 e4m3 KV cache [num_blocks, 64, 512] (uint8 storage, per-tensor k_scale), sm_scale 1/sqrt(256).

Top-k indices reproduce the production layout (vllm/models/glm5next/nvidia/ops/kpool_compress.py
expand_pools_and_append_tail, called with pool_ids[:, :511]): columns [0, 2044) are 511 selected pools x 4 tokens
(-1 where fewer pools exist), columns [2044, 2047) the trailing incomplete pool, column 2047 = -1. The attention
backend then converts them with vLLM's triton_convert_req_index_to_global_index(return_valid_counts=True).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

HEADS, D, TOPK, KPOOL, PBS = 32, 512, 2048, 4, 64
SM_SCALE = 1.0 / math.sqrt(256.0)
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FI_INC = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/include"


def _shim() -> list[str]:
    import sys
    sys.path.insert(0, str(REPO))
    from tf_exl3_moe import _cuda_include_shim
    return _cuda_include_shim()


def build_ext(name: str, source: str, extra: list[str] | None = None):
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    inc = _shim()
    flags = ["-O3", "-std=c++17", "-use_fast_math", "-DNDEBUG", "--expt-relaxed-constexpr", *inc, f"-I{FI_INC}",
             *(extra or [])]
    import hashlib
    src = REPO / source
    h = hashlib.sha256(b"".join(f.read_bytes() for f in sorted(src.parent.glob("*.cu*"))) + " ".join(flags).encode())
    return load(name=f"{name}_{h.hexdigest()[:10]}", sources=[str(src)], extra_cuda_cflags=flags,
                extra_cflags=["-O3", *inc], verbose=False)


# ------------------------------------------------------------------------------------------------ top-k patterns
def topk_logical(start: int, T: int, regime: str, seed: int = 0) -> np.ndarray:
    """[T, 2048] int32 logical token indices in the production layout for query positions start..start+T-1.
    regime: 'indep' (each row picks 511 pools uniformly), 'sticky' (neighbour rows share ~90% of pools; the newest
    complete pool is always taken), 'local' (half the pools from the most recent 1024 pools, half uniform)."""
    rng = np.random.default_rng(seed)
    out = np.full((T, TOPK), -1, dtype=np.int32)
    npool_sel = TOPK // KPOOL - 1                                            # 511
    cur: np.ndarray | None = None
    for r in range(T):
        pos = start + r
        ctx = pos + 1
        pool_len, tail = ctx // KPOOL, ctx % KPOOL
        if pool_len <= npool_sel:
            pools = rng.permutation(pool_len).astype(np.int64) if regime != "sorted" else np.arange(pool_len)
            cur = None
        elif regime == "indep":
            pools = rng.choice(pool_len, npool_sel, replace=False)
        elif regime == "local":
            recent = min(pool_len, 1024)
            a = rng.choice(np.arange(pool_len - recent, pool_len), npool_sel // 2, replace=False)
            rest = np.setdiff1d(np.arange(pool_len), a, assume_unique=True)
            b = rng.choice(rest, npool_sel - a.size, replace=False)
            pools = rng.permutation(np.concatenate([a, b]))
        elif regime == "sticky":
            if cur is None:
                cur = rng.choice(pool_len, npool_sel, replace=False)
            else:
                keep = cur.copy()
                nrep = npool_sel // 10
                slots = rng.choice(npool_sel, nrep, replace=False)
                cand = rng.integers(0, pool_len, size=nrep * 3)
                cand = np.setdiff1d(cand, keep)[:nrep]
                keep[slots[: cand.size]] = cand
                newest = pool_len - 1
                if newest not in keep:
                    keep[rng.integers(0, npool_sel)] = newest
                cur = keep
            pools = cur
        else:
            raise ValueError(regime)
        n = pools.size
        toks = (pools[:, None] * KPOOL + np.arange(KPOOL)[None, :]).reshape(-1)
        out[r, : toks.size] = toks
        out[r, npool_sel * KPOOL: npool_sel * KPOOL + tail] = pool_len * KPOOL + np.arange(tail)
    return out


class Case:
    """One prefill chunk of one request: query rows at positions start..start+T-1 against its paged fp8 cache."""

    def __init__(self, T: int, start: int = 0, regime: str = "sticky", seed: int = 0, k_scale: float = 1.0,
                 q_sigma: float = 2.0, extra_blocks: int = 64, device: str = "cuda"):
        self.T, self.start, self.regime, self.k_scale = T, start, regime, k_scale
        g = torch.Generator(device="cpu").manual_seed(seed)
        ctx_total = start + T
        nb_req = (ctx_total + PBS - 1) // PBS
        self.num_blocks = nb_req + extra_blocks
        self.block_table = torch.randperm(self.num_blocks, generator=g)[:nb_req].to(torch.int32).to(device)[None]
        # latent c_kv after the kv_a norm: unit scale per channel with a few larger channels, stored fp8 / k_scale
        chan = torch.ones(D)
        chan[torch.randperm(D, generator=g)[:16]] = 4.0
        kv = torch.randn(self.num_blocks, PBS, D, generator=g) * chan / k_scale
        self.cache = kv.to(torch.float8_e4m3fn).view(torch.uint8).to(device)          # [NB, 64, 512] uint8
        self.q = (torch.randn(T, HEADS, D, generator=g) * (q_sigma / math.sqrt(1.0 + 15 * 16 / D))) \
            .to(torch.bfloat16).to(device)
        self.topk = torch.from_numpy(topk_logical(start, T, regime, seed)).to(device)  # logical, prod layout
        pos = torch.arange(start, start + T, device=device)
        ctx = pos + 1
        self.ctx = ctx
        # production host plan: SM90 builder _kv_lens_host
        self.lens_prod = torch.where(ctx <= TOPK, ctx, TOPK + ctx % KPOOL).to(torch.int32)
        self.req_id = torch.zeros(T, dtype=torch.int32, device=device)
        self.slots, self.valid = convert(self.req_id, self.block_table, self.topk)

    def pairs(self) -> int:
        return int(self.valid.sum())


def convert(req_id, block_table, topk):
    from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
    slots, valid = triton_convert_req_index_to_global_index(
        req_id, block_table, topk, BLOCK_SIZE=PBS, NUM_TOPK_TOKENS=topk.shape[1], return_valid_counts=True)
    return slots, valid.to(torch.int32)


# ------------------------------------------------------------------------------------------------ FA2 (production)
class ProdFA2:
    """Production's FlashInferMLASparseSM90 path (flashinfer_mla_sparse_sm90.py.patched): one varlen row per query,
    page_size 1, kv_indices = compacted top-k slots with the -1 tail clamped to 0, host-planned lengths."""

    def __init__(self, max_tokens: int, device="cuda"):
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
        self.ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.max_tokens = max_tokens
        self.kv_indices = torch.zeros(max_tokens * TOPK, dtype=torch.int32, device=device)
        self.kv_len_arr = torch.full((max_tokens,), TOPK, dtype=torch.int32, device=device)
        self.w = BatchMLAPagedAttentionWrapper(
            self.ws, qo_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device), kv_indices=self.kv_indices,
            kv_len_arr=self.kv_len_arr, use_cuda_graph=True, backend="fa2")

    def plan(self, T: int, lens: torch.Tensor):
        ar = torch.arange(self.max_tokens + 1, dtype=torch.int32)
        qo = torch.clamp(ar, max=T)
        kv = qo * TOPK
        l = torch.full((self.max_tokens,), TOPK, dtype=torch.int32)
        l[:T] = lens.cpu().to(torch.int32)
        self.w.plan(qo, kv, self.kv_indices, l, HEADS, D, 0, 1, False, SM_SCALE,
                    q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)

    def fill(self, slots: torch.Tensor):
        T, width = slots.shape
        self.kv_indices[: T * width].copy_(slots.reshape(-1).clamp(min=0).to(torch.int32))

    def run(self, q: torch.Tensor, cache_u8: torch.Tensor, k_scale: float):
        T = q.shape[0]
        flat = cache_u8.view(torch.float8_e4m3fn).reshape(-1, 1, D)
        q_pe = q.new_empty(T, HEADS, 0)
        return self.w.run(q, q_pe, flat, flat[..., D:], ckv_scale=float(k_scale), kpe_scale=1.0)


# ------------------------------------------------------------------------------------------------ fp32 reference
def reference(case: Case, rows: torch.Tensor, lens: torch.Tensor | None = None, extra_slots=None) -> torch.Tensor:
    """fp32 attention for the given rows over their first `lens` compacted slots (default: the valid count)."""
    kv = case.cache.view(torch.float8_e4m3fn).reshape(-1, D).float() * case.k_scale
    outs = []
    for r in rows.tolist():
        n = int(case.valid[r] if lens is None else lens[r])
        s = case.slots[r, :n].long()
        K = kv[s]                                              # [n, 512]
        q = case.q[r].float()                                  # [32, 512]
        logits = (q @ K.T) * SM_SCALE
        p = torch.softmax(logits, dim=-1)
        outs.append(p @ K)
    return torch.stack(outs)


def err_stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o, r = out.float(), ref.float()
    num = (o - r).pow(2).sum(-1).sqrt()
    den = r.pow(2).sum(-1).sqrt().clamp_min(1e-30)
    rel = num / den
    return {"rel_l2_max": float(rel.max()), "rel_l2_mean": float(rel.mean()),
            "max_abs": float((o - r).abs().max())}


def cuda_time(fn, warmup: int = 3, iters: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2], ts[0]
