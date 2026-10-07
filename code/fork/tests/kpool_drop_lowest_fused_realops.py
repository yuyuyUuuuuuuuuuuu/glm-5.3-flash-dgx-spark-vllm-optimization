#!/usr/bin/env python3
"""[opt-decodekit-rev] The DEFAULT drop-lowest path (order-preserving Triton kernel, GLM53_KPOOL_DROP_LOWEST_ORDER
unset) against the image's REAL top-k ops. The opt-decodekit branch ran tests/kpool_drop_lowest_{unit,det}.py only on
the torch "sorted" helper (the namespace pins ORDER=sorted) and tests/kpool_drop_lowest_rowstart.py also pins
sorted, so the shipped default was only checked on synthetic permutations (tests/kpool_drop_lowest_bench.py).

  F1 decode persistent_topk (select_k 512, lens incl. < 512 -> -1 fills, garbage logits past each row's length as with
     clean_logits=False): over 32 runs the kept SET per row is constant and == the reference top-511 (by score, then
     the lower id) of the op's selection; output == the op's row with exactly one column removed (order kept)
  F2 a valid pool is never dropped while a -1 is kept
  F3 no host sync (torch.cuda.set_sync_debug_mode("error"))
  F4 FULL-graph: persistent_topk + the kernel captured together, scores mutated in place, replay == eager (as a SET
     and as "op row minus one column")
  F5 prefill top_k_per_row_prefill, multi-request chunk with per-row varying cu_seqlen_ks (ks-relative ids): the
     kernel with col_off=ks keeps the reference top-(K-1) per row; WITHOUT col_off it would not (control)
  F6 the default kernel == the torch "stock" order-preserving variant bitwise on the real op outputs
Run: GPU_RUN_RO=... tests/gpu_run.sh python3 tests/kpool_drop_lowest_fused_realops.py
"""
import importlib.util, os, sys, tempfile
import torch
import vllm._custom_ops  # noqa: F401  registers torch.ops._C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pdl", os.path.join(REPO, "overlay/patch_kpool_drop_lowest.py"))
pdl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdl)
_TD = tempfile.mkdtemp()


def helper(order):
    if order is None:
        os.environ.pop("GLM53_KPOOL_DROP_LOWEST_ORDER", None)
    else:
        os.environ["GLM53_KPOOL_DROP_LOWEST_ORDER"] = order
    name = "kh_" + (order or "default")
    path = os.path.join(_TD, name + ".py")
    open(path, "w").write("import torch\n" + pdl.HELPER)
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


mdef, mstock = helper(None), helper("stock")
assert mdef._GLM53_KPOOL_DL_ORDER == "fused" and mdef._GLM53_KPOOL_DL_FUSED
fdef, fstock = mdef._kpool_keep_highest_pools, mstock._kpool_keep_highest_pools
dev = "cuda"
fail = 0


def ck(c, msg):
    global fail
    print(("ok   " if c else "FAIL ") + msg)
    fail |= (not c)


def ref_keep(scores_row, ids_row, keep):
    # top keep by (-score, id) over valid ids; -1 last
    v = [(-float(scores_row[i]), i) for i in ids_row if i >= 0]
    v.sort()
    out = [i for _, i in v[:keep]]
    return sorted(out + [-1] * (keep - len(out)))


def is_row_minus_one(op_row, out_row):
    return any(op_row[:d] + op_row[d + 1:] == out_row for d in range(len(op_row)))


SELECT_K, KEEP, NP, ROWS = 512, 511, 16384, 8
g = torch.Generator(device=dev).manual_seed(3)
sc = torch.randn(ROWS, NP, device=dev, generator=g) * 3
lens = torch.tensor([NP, 511, 2048, 7, NP, 600, 3, 9999], dtype=torch.int32, device=dev)
for r in range(ROWS):  # garbage past each row's length (clean_logits=False): huge values the op must not select
    sc[r, int(lens[r]):] = 1e30
ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)
sets, orders, ok_order, ok_ref, ok_neg, ok_stock = set(), set(), True, True, True, True
scl = sc.cpu()
for it in range(32):
    dst = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
    torch.ops._C.persistent_topk(sc, lens, dst, ws, SELECT_K, NP)
    out = fdef(sc, dst, KEEP)
    ok_stock &= torch.equal(out, fstock(sc, dst, KEEP))
    dl, ol = dst.tolist(), out.tolist()
    orders.add(tuple(map(tuple, dl)))
    sets.add(tuple(tuple(sorted(x)) for x in ol))
    for r in range(ROWS):
        ok_order &= is_row_minus_one(dl[r], ol[r])
        if it < 4:
            ok_ref &= sorted(ol[r]) == ref_keep(scl[r], dl[r], KEEP)
        nvalid_in = sum(1 for x in dl[r] if x >= 0)
        nvalid_out = sum(1 for x in ol[r] if x >= 0)
        ok_neg &= nvalid_out == min(nvalid_in, KEEP)
print(f"info  persistent_topk distinct raw orders over 32 runs: {len(orders)}")
ck(len(sets) == 1, "F1 decode: the kept SET per row is identical over 32 runs")
ck(ok_ref, "F1 decode: kept set == reference top-511 by (score desc, id asc) of the op's selection")
ck(ok_order, "F1 decode: output == the op's row with exactly one column removed (order preserved)")
ck(ok_neg, "F2 a valid pool is never dropped while a -1 is kept")
ck(ok_stock, "F6 decode: default kernel == torch stock-order variant bitwise (real op outputs)")

dst = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
torch.ops._C.persistent_topk(sc, lens, dst, ws, SELECT_K, NP)
torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("error")
try:
    o = fdef(sc, dst, KEEP); okns = True
except Exception as e:  # noqa
    okns = False; print("  ", e)
torch.cuda.set_sync_debug_mode(0)
ck(okns, "F3 no host sync in the default kernel call")

# F4: graph
st_sc = sc.clone(); st_dst = torch.full((ROWS, SELECT_K), -1, dtype=torch.int32, device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        torch.ops._C.persistent_topk(st_sc, lens, st_dst, ws, SELECT_K, NP); go = fdef(st_sc, st_dst, KEEP)
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    torch.ops._C.persistent_topk(st_sc, lens, st_dst, ws, SELECT_K, NP)
    go = fdef(st_sc, st_dst, KEEP)
ok4 = True
for t in range(5):
    st_sc.copy_(torch.randn(ROWS, NP, device=dev, generator=g) * 3)
    for r in range(ROWS):
        st_sc[r, int(lens[r]):] = 1e30
    gr.replay(); torch.cuda.synchronize()
    gl, dl = go.tolist(), st_dst.tolist()
    e = fdef(st_sc, st_dst, KEEP)
    ok4 &= torch.equal(e, go)
    scl2 = st_sc.cpu()
    for r in range(ROWS):
        ok4 &= is_row_minus_one(dl[r], gl[r]) and sorted(gl[r]) == ref_keep(scl2[r], dl[r], KEEP)
ck(ok4, "F4 FULL graph (topk + kernel captured): replay after in-place score mutation == eager, set == reference")

# F5: prefill, real op, varying ks per row
K = 512
ks_l = [0, 0, 300, 300, 300, 2000, 2000, 5000]
ke_l = [300, 290, 2000, 1500, 900, 5000, 4000, 9000]
ks = torch.tensor(ks_l, dtype=torch.int32, device=dev); ke = torch.tensor(ke_l, dtype=torch.int32, device=dev)
lg = torch.randn(8, 9000, device=dev, generator=g)
pt = torch.empty(8, K, dtype=torch.int32, device=dev)
torch.ops._C.top_k_per_row_prefill(lg, ks, ke, pt, 8, lg.stride(0), lg.stride(1), K)
lgc, ptl = lg.cpu(), pt.tolist()
o = fdef(lg, pt, K - 1, ks).tolist()
o_nooff = fdef(lg, pt, K - 1).tolist()
ok5, ctrl = True, False
for r in range(8):
    row = lgc[r, ks_l[r]:ke_l[r]]
    ok5 &= sorted(o[r]) == ref_keep(row, ptl[r], K - 1) and is_row_minus_one(ptl[r], o[r])
    ctrl |= sorted(o_nooff[r]) != ref_keep(row, ptl[r], K - 1)
ck(ok5, "F5 prefill (real top_k_per_row_prefill, per-row ks): col_off=ks keeps the reference top-(K-1), order kept")
ck(ctrl, "F5 control: without col_off some ks>0 row keeps a different set (the offset matters)")
ck(torch.equal(fdef(lg, pt, K - 1, ks), fstock(lg, pt, K - 1, ks)), "F6 prefill: default kernel == torch stock-order bitwise")
print("RESULT:", "ALL OK" if not fail else "FAILURES")
sys.exit(fail)
