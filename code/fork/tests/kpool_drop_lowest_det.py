#!/usr/bin/env python3
"""[glm53-kpool-drop-lowest] Determinism / no-sync / CUDA-graph test on nodeC with the REAL top-k ops.

Exercises the helper AS INSTALLED by overlay/patch_kpool_drop_lowest.py (applied to a pristine copy of the image's
sparse_attn_indexer_kpool.py, imported as a module) against the image's own
``torch.ops._C.persistent_topk`` (decode, select_k 512 = production) and ``torch.ops._C.top_k_per_row_prefill``
(prefill, one chunk):
  D1 bitwise determinism of the patched path: the top-k ops return a deterministic SET in a nondeterministic ORDER
     (logged per call), so the raw dst columns are shuffled between runs; the helper's output must be BITWISE
     identical across 32 runs despite that (this is the defect: the old ``[:, :select_k - 1]`` truncation changed).
  D2 the helper's kept set == the top select_k-1 pools by score (per row, against an independent numpy reference
     over the op's selection; scores are made unique per row so the global top select_k-1 is the same set);
  D3 -1 fills of the real op (rows with fewer valid pools than select_k): no valid pool dropped while a -1 is kept;
  D4 no CUDA -> host sync: torch.cuda.set_sync_debug_mode("error") around a helper call;
  D5 FULL-graph safety: the helper captured inside a torch.cuda.CUDAGraph, replayed after mutating the static
     scores in place, is bitwise equal to the eager result on the same scores (no allocation at replay, no shape
     change - decode runs inside FULL graphs).
Run: tests/gpu_run.sh python3 tests/kpool_drop_lowest_det.py
"""
from __future__ import annotations

import sys

import kpool_drop_lowest_unit as U  # loads the patched helper via the overlay (PRISTINE copy + overlay)

import numpy as np  # noqa: E402
import torch  # noqa: E402

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
SELECT_K = 512          # production decode: topk_tokens 2048 // index_kpool 4, the persistent_topk branch
NUM_POOLS = 4096        # a 16k-token context at kpool 4
ROWS = 8


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    fn, tmp = U.load_patched_helper(dev)
    try:
        import vllm  # noqa: F401  registers torch.ops._C.*
        g = np.random.default_rng(7)
        sc_np = (g.standard_normal((ROWS, NUM_POOLS)) * 4).astype(np.float32)
        sc_np += np.arange(NUM_POOLS, dtype=np.float32) * 1e-3   # unique scores per row: the global top is unique
        sc_t = torch.from_numpy(sc_np).to(dev)
        lens = torch.tensor([NUM_POOLS, 511, 2048, 7, NUM_POOLS, 600, 3, 999], dtype=torch.int32, device=dev)

        # ---- decode op (persistent_topk): a deterministic SET, a nondeterministic ORDER
        ws = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
        outs, sel_sets, orders, dsts = [], [], [], []
        for _ in range(32):
            dst = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
            torch.ops._C.persistent_topk(sc_t, lens, dst, ws, SELECT_K, NUM_POOLS)
            outs.append(fn(sc_t, dst, SELECT_K - 1))
            sel_sets.append(tuple(tuple(sorted(row)) for row in dst.cpu().tolist()))   # the SET of every row
            orders.append(tuple(map(tuple, dst.cpu().tolist())))                # the raw column order
            dsts.append(dst.clone())
        U.ck(len(set(sel_sets)) == 1, "persistent_topk returns the same SET (per row) on every one of 32 runs")
        n_orders = len({o for o in orders})
        print(f"info  persistent_topk raw column ORDER varied across runs: {n_orders} distinct orders in 32")
        U.ck(len({tuple(o.cpu().numpy().ravel().tobytes()) for o in outs}) == 1,
             "D1 the patched decode selection is BITWISE identical across 32 runs (raw order varies)")
        sel0 = dsts[0].cpu().numpy()
        ref = U.reference(sc_np, sel0, SELECT_K - 1)
        got = outs[0].cpu().numpy()
        U.ck(all(sorted(got[r].tolist()) == sorted(ref[r].tolist()) for r in range(ROWS)),
             "D2 kept SET == the top select_k-1 pools by score (persistent_topk rows)")
        # global top select_k-1 of the row's VALID range [0, seq_len) (scores unique per row, so the op's top
        # select_k of that range is the global top select_k of it); only rows with >= select_k valid pools
        masked = np.where(np.arange(NUM_POOLS)[None, :] < lens.cpu().numpy()[:, None], sc_np.astype(np.float64), -np.inf)
        full = np.argsort(-masked, kind="stable", axis=1)[:, : SELECT_K - 1]
        long_rows = [r for r in range(ROWS) if lens[r].item() >= SELECT_K]
        U.ck(all(sorted(got[r].tolist()) == sorted(full[r].tolist()) for r in long_rows),
             "D2 kept SET == the GLOBAL top select_k-1 pools of the row's valid range (rows with >= select_k)")
        # D3 rows with fewer valid pools than select_k (seq_lens 7, 3, 511, 600, 999): their -1s are only kept
        # when no valid pool was dropped, i.e. every valid pool of the op's selection survives the cut
        n_valid = torch.minimum(lens, torch.full_like(lens, SELECT_K)).cpu().numpy()
        for r in range(ROWS):
            if n_valid[r] >= SELECT_K:
                continue
            kept_valid = int((got[r] >= 0).sum())
            U.ck(kept_valid == min(SELECT_K - 1, n_valid[r]),
                 f"D3 row {r} (seq {lens[r].item()} pools): {kept_valid} valid pools kept, no valid pool lost to a -1")
        # the all-valid rows: exactly one pool dropped, the lowest-scored selected one
        for r in range(ROWS):
            if n_valid[r] < SELECT_K:
                continue
            sel_sc = np.take_along_axis(sc_np, sel0, axis=1)
            order = np.lexsort((sel0, -sel_sc))
            want = np.take_along_axis(sel0, order[:, SELECT_K - 1:], axis=1)
            U.ck(sorted(set(got[r].tolist()) ^ set(sel0[r].tolist())) == sorted(want[r].tolist()),
                 f"D3 row {r}: the single dropped pool is the lowest-scored selected one")

        # ---- prefill op (top_k_per_row_prefill): rows are (ks, ke) spans, -1 fill outside
        ks = torch.tensor([0, 100, 0, 50, 10, 0, 20, 1], dtype=torch.int32, device=dev)
        ke = torch.tensor([NUM_POOLS, 400, 600, 60, NUM_POOLS, 5, 21, 2], dtype=torch.int32, device=dev)
        pouts, psets, porders = [], [], []
        for _ in range(32):
            pdst = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
            torch.ops._C.top_k_per_row_prefill(
                sc_t, ks, ke, pdst, ROWS, sc_t.stride(0), sc_t.stride(1), SELECT_K
            )
            pouts.append(fn(sc_t, pdst, SELECT_K - 1))
            psets.append(tuple(tuple(sorted(row)) for row in pdst.cpu().tolist()))
            porders.append(tuple(map(tuple, pdst.cpu().tolist())))
        U.ck(len(set(psets)) == 1, "top_k_per_row_prefill returns the same SET on every one of 32 runs")
        print(f"info  top_k_per_row_prefill raw column ORDER varied across runs: "
              f"{len({tuple(map(tuple, o)) for o in porders})} distinct orders in 32")
        U.ck(len({tuple(o.cpu().numpy().ravel().tobytes()) for o in pouts}) == 1,
             "D1 the patched prefill selection is BITWISE identical across 32 runs")
        psets_np = np.array(list(psets[0]), dtype=np.int32)
        pref = U.reference(sc_np, psets_np, SELECT_K - 1)
        pgot = pouts[0].cpu().numpy()
        U.ck(all(sorted(pgot[r].tolist()) == sorted(pref[r].tolist()) for r in range(ROWS)),
             "D2 kept SET == the top select_k-1 pools by score (top_k_per_row_prefill rows)")
        for r, (a, b) in enumerate(zip(ks.cpu().numpy(), ke.cpu().numpy())):
            nv = int(b) - int(a)
            if nv >= SELECT_K:
                continue
            U.ck(int((pgot[r] >= 0).sum()) == max(0, min(SELECT_K - 1, nv)),
                 f"D3 prefill row {r} (span {nv} pools): no valid pool lost to a -1 fill")

        # ---- D4 no host sync
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fn(sc_t, dsts[0], SELECT_K - 1)
            synced = False
        except Exception as exc:  # a syncing op raises here
            synced = True
            print(f"FAIL sync: {exc}")
        finally:
            torch.cuda.set_sync_debug_mode("default")
        U.ck(not synced, "D4 no CUDA -> host sync inside the patched selection (sync_debug_mode=error)")

        # ---- D5 FULL graph capture + replay
        s_static = sc_t.clone()
        i_static = dsts[0].clone()
        for _ in range(3):
            fn(s_static, i_static, SELECT_K - 1)      # warmup (autotune / workspace)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            g_out = fn(s_static, i_static, SELECT_K - 1)   # recorded, not executed: read it after a replay
        gr.replay()
        torch.cuda.synchronize()
        eager_ref = fn(s_static, i_static, SELECT_K - 1)
        U.ck(torch.equal(g_out, eager_ref), "D5 graph replay == eager on the capture inputs")
        mut = (g.standard_normal((ROWS, NUM_POOLS)) * 4).astype(np.float32)
        mut += np.arange(NUM_POOLS, dtype=np.float32) * 1e-3
        i2 = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
        torch.ops._C.persistent_topk(torch.from_numpy(mut).to(dev), lens, i2, ws, SELECT_K, NUM_POOLS)
        s_static.copy_(torch.from_numpy(mut).to(dev))
        i_static.copy_(i2)
        gr.replay()
        torch.cuda.synchronize()
        U.ck(torch.equal(g_out, fn(s_static, i_static, SELECT_K - 1)),
             "D5 graph replay after mutating the static scores == eager on the same scores")
        U.ck(tuple(g_out.shape) == (ROWS, SELECT_K - 1) and g_out.dtype == torch.int64,
             "D5 the captured statement keeps the [rows, select_k-1] int64 shape (no shape change)")
        print(f"det: {'ALL OK' if not U.fail else 'FAILURES'}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return U.fail


if __name__ == "__main__":
    sys.exit(main())
