#!/usr/bin/env python3
"""[opt-decodekit] Which pool does production's `pool_ids[:, :select_k-1]` actually drop?

The 0930 review called the drop "arbitrary" because the top-k ops return their columns in a nondeterministic ORDER.
This probe measures, with the image's REAL ops at production's select_k (512), the score RANK (0 = best,
511 = worst among the 512 selected) of the pool in the LAST column, which is the one the stock truncation drops:

  * torch.ops._C.persistent_topk          (decode path, sparse_attn_indexer_kpool.py ~:815)
  * torch.ops._C.top_k_per_row_prefill    (prefill path, ~:559)

over several score distributions (unique gaussian, heavily tied/quantised, recency-skewed) and row lengths
(production contexts: 2k .. 256k tokens = 512 .. 65,536 pools). If the last column is (almost) always rank 511,
the stock truncation already drops the lowest-scored pool and the "arbitrary drop" defect does not exist in
practice; the DROP_LOWEST patch can then only change behaviour on exact ties.

Run: tests/gpu_run.sh python3 tests/kpool_lastcol_probe.py
"""
from __future__ import annotations

import collections
import sys

import numpy as np
import torch

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
SELECT_K = 512


def ranks_of_last(scores: torch.Tensor, dst: torch.Tensor, ks: torch.Tensor | None, lens) -> list[tuple[int, int, int]]:
    """Per row with >= SELECT_K valid pools: (rank of last col among the selected, n_ties_at_last, n_valid_sel)."""
    sc = scores.float().cpu().numpy()
    d = dst.cpu().numpy().astype(np.int64)
    out = []
    for r in range(d.shape[0]):
        row = d[r]
        if (row < 0).any():
            continue
        off = int(ks[r].item()) if ks is not None else 0
        s = sc[r, row + off]
        last = s[-1]
        rank = int((s > last).sum())                 # 0 = best
        ties = int((s == last).sum()) - 1
        is_min = bool(last == s.min())
        out.append((rank, ties, is_min))
    return out


def main() -> int:
    import vllm  # noqa: F401  registers torch.ops._C.*
    dev = "cuda"
    ws = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
    g = np.random.default_rng(11)
    worst = 0
    for n_pools in (600, 2048, 16384, 65536):
        for dist in ("gauss", "tied16", "recency"):
            rows = 8
            if dist == "gauss":
                sc = g.standard_normal((rows, n_pools)).astype(np.float32) * 3
            elif dist == "tied16":
                sc = np.round(g.standard_normal((rows, n_pools)) * 16).astype(np.float32) / 16
            else:
                base = g.standard_normal((rows, n_pools)).astype(np.float32)
                sc = base + np.linspace(-2, 3, n_pools, dtype=np.float32)[None, :]
            t = torch.from_numpy(sc).to(dev).contiguous()
            lens = torch.full((rows,), n_pools, dtype=torch.int32, device=dev)
            # decode op
            dhist = collections.Counter()
            for _ in range(16):
                dst = torch.full((rows, SELECT_K), -1, dtype=torch.int32, device=dev)
                torch.ops._C.persistent_topk(t, lens, dst, ws, SELECT_K, n_pools)
                for rank, ties, is_min in ranks_of_last(t, dst, None, lens):
                    dhist[(rank, ties > 0, is_min)] += 1
            # prefill op (one chunk, every row spans [0, n_pools))
            ks = torch.zeros(rows, dtype=torch.int32, device=dev)
            ke = torch.full((rows,), n_pools, dtype=torch.int32, device=dev)
            phist = collections.Counter()
            for _ in range(16):
                dst = torch.full((rows, SELECT_K), -1, dtype=torch.int32, device=dev)
                torch.ops._C.top_k_per_row_prefill(t, ks, ke, dst, rows, t.stride(0), t.stride(1), SELECT_K)
                for rank, ties, is_min in ranks_of_last(t, dst, ks, lens):
                    phist[(rank, ties > 0, is_min)] += 1

            def summ(h):
                tot = sum(h.values())
                lowest = sum(v for (rk, ti, mn), v in h.items() if mn)  # last col holds the minimum selected score
                exact511 = sum(v for (rk, ti, mn), v in h.items() if rk == SELECT_K - 1 and not ti)
                tied = sum(v for (rk, ti, mn), v in h.items() if ti)
                worst_rank = min(rk for (rk, _, _) in h) if h else -1
                return tot, exact511, tied, worst_rank, lowest
            for name, h in (("persistent_topk", dhist), ("top_k_per_row_prefill", phist)):
                tot, ex, tied, best_rank, lowest = summ(h)
                print(f"{name:22s} pools={n_pools:6d} dist={dist:8s} rows={tot:4d}  last col = rank 511 (unique): {ex:4d}"
                      f"  tied: {tied:4d}  last col == min score: {lowest:4d}  smallest rank seen in last col: {best_rank}")
                if lowest != tot:
                    worst += 1
    print(f"summary: {worst} (op, length, dist) cells where the last column was NOT always the lowest-scored "
          f"(or a tie of the lowest score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
