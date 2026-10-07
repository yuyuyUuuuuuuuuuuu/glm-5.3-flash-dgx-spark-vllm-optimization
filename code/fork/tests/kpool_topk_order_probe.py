#!/usr/bin/env python3
"""[opt-decodekit-rev] Column ORDER of the real top-k ops (select_k 512): is it score-sorted, position-sorted, or
neither? Spearman rank correlation of column index vs score / vs pool id, averaged over rows.
Run: tests/gpu_run.sh python3 tests/kpool_topk_order_probe.py"""
import torch
import vllm._custom_ops  # noqa: F401
dev = "cuda"; K = 512
g = torch.Generator(device=dev).manual_seed(5)


def spearman(a, b):
    ra = a.argsort(dim=1).argsort(dim=1).float(); rb = b.argsort(dim=1).argsort(dim=1).float()
    ra -= ra.mean(1, keepdim=True); rb -= rb.mean(1, keepdim=True)
    return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).mean().item()


for NP in (2048, 16384, 65536):
    sc = torch.randn(16, NP, device=dev, generator=g)
    lens = torch.full((16,), NP, dtype=torch.int32, device=dev)
    dst = torch.empty(16, K, dtype=torch.int32, device=dev)
    torch.ops._C.persistent_topk(sc, lens, dst, torch.empty(1 << 20, dtype=torch.uint8, device=dev), K, NP)
    col = torch.arange(K, device=dev).expand(16, K).float()
    ids = dst.long(); s = sc.gather(1, ids)
    print(f"persistent_topk       pools {NP:6d}: spearman(col, score) {spearman(col, s):+.3f}  spearman(col, pool id) {spearman(col, ids.float()):+.3f}")
    ks = torch.zeros(16, dtype=torch.int32, device=dev); ke = lens.clone()
    torch.ops._C.top_k_per_row_prefill(sc, ks, ke, dst, 16, sc.stride(0), sc.stride(1), K)
    ids = dst.long(); s = sc.gather(1, ids)
    print(f"top_k_per_row_prefill pools {NP:6d}: spearman(col, score) {spearman(col, s):+.3f}  spearman(col, pool id) {spearman(col, ids.float()):+.3f}")
