#!/usr/bin/env python3
"""[opt-decodekit] Does torch.ops._C.persistent_topk / top_k_per_row_prefill modify its logits input in place?
And with PRISTINE scores (host copy taken before the op), which rank does the last column hold?
Run: tests/gpu_run.sh python3 tests/kpool_topk_inplace_probe.py"""
import sys, collections
import numpy as np, torch

def main():
    import vllm  # noqa
    dev = "cuda"; K = 512
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)
    g = np.random.default_rng(3)
    for n_pools in (600, 1500, 2048, 2500, 4096, 8192, 16384, 32768, 65536):
        rows = 8
        sc = (g.standard_normal((rows, n_pools)) * 3).astype(np.float32)
        t = torch.from_numpy(sc).to(dev)
        before = t.clone()
        lens = torch.full((rows,), n_pools, dtype=torch.int32, device=dev)
        hist = collections.Counter(); modified = 0
        for _ in range(8):
            dst = torch.full((rows, K), -1, dtype=torch.int32, device=dev)
            torch.ops._C.persistent_topk(t, lens, dst, ws, K, n_pools)
            torch.cuda.synchronize()
            if not torch.equal(t, before):
                modified += 1
                t.copy_(before)
            d = dst.cpu().numpy().astype(np.int64)
            for r in range(rows):
                s = sc[r, d[r]]
                top = np.sort(sc[r])[::-1][:K]
                assert np.array_equal(np.sort(s)[::-1], top), "selected set is not the true top-k"
                hist[int((s > s[-1]).sum())] += 1
        lowest = hist.get(K - 1, 0); tot = sum(hist.values())
        print(f"persistent_topk pools={n_pools:6d}: input modified in place in {modified}/8 calls; "
              f"last col is the lowest of the selected in {lowest}/{tot} rows; min rank in last col {min(hist)}")
    return 0

sys.exit(main())
