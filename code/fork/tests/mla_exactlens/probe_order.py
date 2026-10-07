#!/usr/bin/env python3
"""[opt-decodekit] Is the sparse-MLA output sensitive to the ORDER of the selected keys (beyond rounding), and how
large is production's 4-key over-read? Synthetic GLM shapes (tests/mla_prefill_common.py), the real kernels:
production FA2 (flashinfer 0.6.18 BatchMLAPagedAttention, page_size 1, as flashinfer_mla_sparse_sm90.py.patched
drives it) with production's plan and with the exact plan, and the GLM53_MLA_PREFILL exact kernel (AOT ext).
Orders per row: 'orig' (random pool order), 'perm' (another random pool order), 'desc' (pools by their fp32 mean
attention logit, descending = what GLM53_KPOOL_DROP_LOWEST emits). Error = rel L2 vs the fp32 reference over the
valid keys (rows are the same SET in every order).
Run: GPU_RUN_ENV_EXTRA= tests/hostloop_gpu.sh python3 tests/mla_exactlens/probe_order.py  (with tests/mla_env.sh)"""
import sys, os
sys.path.insert(0, "/w/tests"); sys.path.insert(0, "/w")
import numpy as np, torch
import mla_prefill_common as C

def reorder(case, mode, rng):
    """New logical top-k with the same pool SET per row in another order; tail columns kept."""
    tk = case.topk.cpu().numpy().copy()
    kv = case.cache.view(torch.float8_e4m3fn).reshape(-1, C.D).float() * case.k_scale
    bt = case.block_table[0].cpu().numpy()
    npool = C.TOPK // C.KPOOL - 1
    for r in range(tk.shape[0]):
        hist = tk[r, : npool * C.KPOOL].reshape(npool, C.KPOOL)
        valid = hist[:, 0] >= 0
        pools = hist[valid]
        if mode == "perm":
            pools = pools[rng.permutation(len(pools))]
        elif mode == "desc":
            toks = torch.from_numpy(pools[:, 0].astype(np.int64))
            slots = torch.from_numpy(bt[(toks // C.PBS).numpy()].astype(np.int64)) * C.PBS + toks % C.PBS
            K = kv[slots.to(kv.device)]
            sc = (case.q[r].float() @ K.T).mean(0).cpu().numpy()
            pools = pools[np.argsort(-sc, kind="stable")]
        hist[: len(pools)] = pools
        tk[r, : npool * C.KPOOL] = hist.reshape(-1)
    return torch.from_numpy(tk).to(case.topk.device)

def main():
    import glm53_mla_prefill as M
    os.environ["TF_EXL3_JIT"] = os.environ.get("TF_EXL3_JIT", "0")
    ext = M.load_ext()
    rng = np.random.default_rng(5)
    for T, start, regime, qs in ((64, 9000, "sticky", 2.0), (64, 9000, "sticky", 8.0), (64, 30001, "local", 8.0), (8, 50002, "sticky", 8.0)):
        case = C.Case(T, start, regime, seed=31, q_sigma=qs)
        fa = C.ProdFA2(T + 1)  # one spare row: production's buffer is max_num_batched_tokens rows, so the last row's
        #                         over-read lands on a stale (here: zero = slot 0) row, not past the allocation
        ref = C.reference(case, torch.arange(T))
        print(f"== T={T} start={start} regime={regime} q_sigma={qs}: valid {int(case.valid.min())}..{int(case.valid.max())}, "
              f"production plan {int(case.lens_prod.min())}..{int(case.lens_prod.max())}")
        outs = {}
        for mode in ("orig", "perm", "desc"):
            tk = case.topk if mode == "orig" else reorder(case, mode, rng)
            slots, valid = C.convert(case.req_id, case.block_table, tk)
            assert torch.equal(valid, case.valid)
            res = {}
            for plan_name, lens in (("prod-plan", case.lens_prod), ("exact-plan", valid)):
                fa.kv_indices.fill_(0)
                fa.plan(T, lens)
                fa.fill(slots)
                res["FA2 " + plan_name] = fa.run(case.q, case.cache, case.k_scale).clone()
            out = torch.empty_like(case.q)
            ext.run(case.q, case.cache.reshape(-1, 512), slots, valid, out, float(C.SM_SCALE), float(case.k_scale), M.STATE.variant)
            res["exact kernel"] = out
            torch.cuda.synchronize()
            for k, o in res.items():
                st = C.err_stats(o, ref)
                stl = C.err_stats(o[:-1], ref[:-1])
                outs[(mode, k)] = o
                print(f"   {mode:4s} {k:15s} vs fp32: rel_l2 mean {st['rel_l2_mean']:.2e} max {st['rel_l2_max']:.2e}"
                      f" | rows but the last: mean {stl['rel_l2_mean']:.2e} max {stl['rel_l2_max']:.2e}")
        for k in ("FA2 prod-plan", "FA2 exact-plan", "exact kernel"):
            for m2 in ("perm", "desc"):
                st = C.err_stats(outs[(m2, k)], outs[("orig", k)].float())
                print(f"   order sensitivity {k:15s} orig vs {m2}: rel_l2 mean {st['rel_l2_mean']:.2e} max {st['rel_l2_max']:.2e}")
    return 0

sys.exit(main())
