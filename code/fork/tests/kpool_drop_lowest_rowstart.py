"""Review probe: does top_k_per_row_prefill return ks-RELATIVE pool ids (then logits.gather(1, ids) reads the
wrong columns for rows with cu_seqlen_ks > 0)?  Plus a decode-shape microbench of the helper."""
import importlib.util, os, sys, time, torch
import vllm._custom_ops  # noqa  registers torch.ops._C
spec = importlib.util.spec_from_file_location("p", "/w/overlay/patch_kpool_drop_lowest.py"); p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
os.environ.setdefault("GLM53_KPOOL_DROP_LOWEST_ORDER", "sorted")  # exec'd source: no triton.jit (opt-decodekit)
ns = {"torch": torch}; exec(p.HELPER, ns); helper = ns["_kpool_keep_highest_pools"]
torch.manual_seed(0); dev = "cuda"
K = 64
# two "requests" in one chunk: rows 0..3 attend cols [0,300), rows 4..7 attend cols [300,700)
ks = torch.tensor([0]*4 + [300]*4, dtype=torch.int32, device=dev)
ke = torch.tensor([300]*4 + [700]*4, dtype=torch.int32, device=dev)
logits = torch.randn(8, 700, device=dev)
out = torch.empty(8, K, dtype=torch.int32, device=dev)
torch.ops._C.top_k_per_row_prefill(logits, ks, ke, out, 8, logits.stride(0), logits.stride(1), K)
print("row4 ids min/max", int(out[4].min()), int(out[4].max()))
ref = [set((torch.topk(logits[r, ks[r]:ke[r]], K).indices).tolist()) for r in range(8)]
rel = all(set(out[r].tolist()) == ref[r] for r in range(8))
print("ids are ks-relative:", rel)
keep = helper(logits, out, K - 1)
good = [set(torch.topk(logits[r, ks[r]:ke[r]], K - 1).indices.tolist()) for r in range(8)]
print("helper(logits) correct per row:", [set(keep[r].tolist()) == good[r] for r in range(8)])
keep2 = helper(logits, out, K - 1, ks)
print("helper(logits, col_off=ks) correct per row:", all(set(keep2[r].tolist()) == good[r] for r in range(8)))
# microbench helper at decode shapes
for rows in (5, 8, 16):
    lg = torch.randn(rows, 65536, device=dev)
    ids = torch.stack([torch.randperm(65536, device=dev)[:512] for _ in range(rows)]).int()
    for _ in range(20): helper(lg, ids, 511)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(50): o = helper(lg, ids, 511)
    torch.cuda.synchronize(); n = 40; t = time.perf_counter()
    for _ in range(n): g.replay()
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / n / 50 * 1e6
    print(f"rows={rows} select_k=512 graph-replay helper {us:.1f} us/call")
