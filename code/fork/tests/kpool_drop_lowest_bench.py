#!/usr/bin/env python3
"""[opt-decodekit] Decode cost + equivalence of the drop-lowest helper: production stock slice vs the shipped
score-sorted helper (GLM53_KPOOL_DROP_LOWEST=1) vs the order-preserving experiment (GLM53_KPOOL_DROP_LOWEST_ORDER=stock).
The helper text is exec'd from overlay/patch_kpool_drop_lowest.py's HELPER (the bytes the overlay inserts).
Timed as production runs it: inside one CUDA graph, 11 calls (the 11 MLA layers' indexers) per replay.
Also checks: the stock-order variant keeps exactly the SAME SET as the sorted helper, and the stock column order.
Run: tests/gpu_run.sh python3 tests/kpool_drop_lowest_bench.py"""
import importlib.util, os, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pdl", os.path.join(REPO, "overlay/patch_kpool_drop_lowest.py"))
pdl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdl)


import tempfile
_TD = tempfile.mkdtemp()  # inside the container (triton.jit needs a real source file)


def helper(order: str):
    os.environ["GLM53_KPOOL_DROP_LOWEST_ORDER"] = order
    name = "kpool_helper_" + (order or "sorted")
    path = os.path.join(_TD, name + ".py")
    with open(path, "w") as fh:
        fh.write("import torch\n" + pdl.HELPER)
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    return mod._kpool_keep_highest_pools


sorted_h, stock_h, fused_h, fsort_h, default_h = (helper("sorted"), helper("stock"), helper("fused"), helper("fusedsort"),
                                                  helper(""))
dev = "cuda"
SELECT_K, KEEP, LAYERS = 512, 511, 11
fail = 0


def ck(c, msg):
    global fail
    print(("ok   " if c else "FAIL ") + msg)
    fail |= (not c)


for rows in (8, 32):
    for pools in (16384, 262144):
        g = torch.Generator(device=dev).manual_seed(rows * 7 + pools)
        logits = torch.randn(rows, pools, device=dev, generator=g)
        pool_topk = torch.topk(logits, SELECT_K, dim=1).indices.to(torch.int32)
        perm = torch.argsort(torch.rand(rows, SELECT_K, device=dev, generator=g), dim=1)
        pool_topk = pool_topk.gather(1, perm)  # nondeterministic op order stand-in
        pool_topk[0, 5] = -1  # one -1 fill
        pool_topk[1, -3:] = -1  # a trailing -1 run (a row with fewer valid pools than select_k)
        a, b = sorted_h(logits, pool_topk, KEEP), stock_h(logits, pool_topk, KEEP)
        same_set = torch.equal(a.sort(dim=1).values, b.sort(dim=1).values)
        # stock order: b == pool_topk with one column removed, order kept
        ids = pool_topk.to(torch.int64)
        ids_l, b_l = ids.tolist(), b.tolist()
        order_ok = all(any(ids_l[r][:d] + ids_l[r][d + 1:] == b_l[r] for d in range(SELECT_K)) for r in range(rows))
        ck(same_set and order_ok, f"rows {rows} pools {pools}: stock-order variant == sorted helper's SET, op order kept")
        f = fused_h(logits, pool_topk, KEEP)
        ck(torch.equal(f, b) and f.dtype == torch.int64 and f.shape == (rows, KEEP),
           f"rows {rows} pools {pools}: fused Triton kernel == torch stock-order variant bitwise (int64, [rows, keep])")
        fs = fsort_h(logits, pool_topk, KEEP)
        ck(torch.equal(fs, a) and fs.dtype == torch.int64, f"rows {rows} pools {pools}: fusedsort Triton kernel == the r16l sorted torch helper BITWISE")
        ck(torch.equal(default_h(logits, pool_topk, KEEP), f), f"rows {rows} pools {pools}: the default (ORDER unset) is the order-preserving fused kernel")
        lq = torch.round(logits * 2) / 2  # many exact fp32 ties (+ some -0.0)
        lq[:, ::97] = -0.0; lq[:, 1::97] = 0.0
        ck(torch.equal(fsort_h(lq, pool_topk, KEEP), sorted_h(lq, pool_topk, KEEP))
           and torch.equal(fused_h(lq, pool_topk, KEEP), stock_h(lq, pool_topk, KEEP)),
           f"rows {rows} pools {pools}: exact score ties and +-0.0: fusedsort == sorted, fused == stock (bitwise)")
        off = torch.zeros(rows, dtype=torch.int32, device=dev)
        lg2 = torch.cat([torch.randn(rows, 7, device=dev, generator=g), logits], 1); off += 7
        ck(torch.equal(fused_h(lg2, pool_topk, KEEP, off), b) and torch.equal(stock_h(lg2, pool_topk, KEEP, off), b)
           and torch.equal(fsort_h(lg2, pool_topk, KEEP, off), a),
           f"rows {rows} pools {pools}: col_off (prefill's cu_seqlen_ks) honoured by both")
        ck(bool((b[0] == -1).sum() == 0) and bool((a[0] == -1).sum() == 0) and int((b[1] == -1).sum()) == 2 == int((a[1] == -1).sum()),
           f"rows {rows} pools {pools}: a -1 fill is dropped first in both (row 0: the only -1; row 1: one of 3)")
        res = {}
        for name, fn in (("stock slice", lambda: pool_topk.to(torch.int64)[:, :KEEP]),
                         ("sorted helper", lambda: sorted_h(logits, pool_topk, KEEP)),
                         ("stock-order helper", lambda: stock_h(logits, pool_topk, KEEP)),
                         ("fused kernel", lambda: fused_h(logits, pool_topk, KEEP)),
                         ("fusedsort kernel", lambda: fsort_h(logits, pool_topk, KEEP))):
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    outs = [fn() for _ in range(LAYERS)]
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                outs = [fn() for _ in range(LAYERS)]
            for _ in range(20):
                gr.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            N = 200
            e0.record()
            for _ in range(N):
                gr.replay()
            e1.record(); torch.cuda.synchronize()
            res[name] = e0.elapsed_time(e1) / N * 1000
        print(f"rows {rows:3d} pools {pools:6d}: per decode step ({LAYERS} indexers, one graph): "
              + ", ".join(f"{k} {v:.1f} us" for k, v in res.items()))
print("RESULT:", "ALL OK" if not fail else "FAILURES")
sys.exit(fail)
