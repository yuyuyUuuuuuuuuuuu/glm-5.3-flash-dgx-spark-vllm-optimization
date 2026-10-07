"""本番 decode 経路 exllamav3_ext.exl3_moe の速度基準線(合成重み)。
複数「層」の expert 集合を巡回させ、毎回コールドな重みを読ませる(キャッシュ再利用で帯域を過大評価しないため)。
bytes = 触れた distinct expert 数 × (gate+up+down) の trellis バイト。suh/svh は無視できる大きさなので除外。"""
import time, torch
from prod_baseline import load_xl, make_experts, ptr_tables, run_prod

xl = load_xl(); dev = "cuda"
D, NI, E, TOPK, L = 4096, 768, 64, 8, 8
layers = [make_experts(E, D, NI, dev, seed=s) for s in range(L)]
ptrs = [ptr_tables(ex, dev) for ex in layers]
tile_bytes = lambda k, n: (k // 16) * (n // 16) * 64 * 2
per_expert = 2 * tile_bytes(D, NI) + tile_bytes(NI, D)
print(f"D={D} NI={NI} E={E}/layer L={L} topk={TOPK}  per-expert trellis {per_expert/2**20:.2f} MiB")
print(f"{'tok':>4} {'distinct':>8} {'us/call':>9} {'GB/s':>7}")
for T in (1, 2, 4, 8, 16, 32, 64):
    g = torch.Generator(device="cpu").manual_seed(T)
    xs = [(torch.randn(T, D, generator=g) * 0.5).half().to(dev) for _ in range(L)]
    ids = [torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(dev) for _ in range(L)]
    ws = [torch.softmax(torch.randn(T, TOPK, generator=g), -1).to(dev) for _ in range(L)]
    distinct = sum(int(torch.unique(i).numel()) for i in ids) / L
    for i in range(L):
        run_prod(xl, xs[i], ids[i], ws[i], layers[i], ptrs[i], D, NI)
    torch.cuda.synchronize()
    it = 40
    t0 = time.perf_counter()
    for r in range(it):
        i = r % L
        run_prod(xl, xs[i], ids[i], ws[i], layers[i], ptrs[i], D, NI)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / it
    print(f"{T:>4} {distinct:>8.1f} {dt*1e6:>9.1f} {distinct*per_expert/dt/1e9:>7.1f}")
print("(注) us/call は routing 構築と temps/out 確保を含む本番 apply 相当。GB10 理論 273 GB/s")
