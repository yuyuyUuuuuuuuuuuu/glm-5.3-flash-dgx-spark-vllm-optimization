import sys, torch
run, L, a, b = sys.argv[1:5]
A = torch.load(f"{run}/phtrace_L{L}_{a}.pt", weights_only=False); B = torch.load(f"{run}/phtrace_L{L}_{b}.pt", weights_only=False)
print(f"L={L} {a} vs {b}: n {A['n']} computed {A['computed']}")
def d(x, y):
    if x.shape != y.shape: return f"shape {tuple(x.shape)} vs {tuple(y.shape)}"
    nd = int((x != y).sum()); 
    return f"{nd}/{x.numel()} differ, max|d| {float((x-y).abs().max()):.3g} (|x|max {float(x.abs().max()):.3g})" if nd else "IDENTICAL"
order = sorted(A["kda"], key=lambda n: int(n.split(".")[2]))
for name in order:
    ka, kb = A["kda"][name], B["kda"][name]
    print(f"  {name:28s} in {d(ka['qkv_in'], kb['qkv_in'])} | rec0 {d(ka['rec'], kb['rec'])} | conv0 rows0-2 {d(ka['conv'][:3], kb['conv'][:3])} | out {d(ka['out'], kb['out'])}")
for name in A["mla"]:
    ma, mb = A["mla"][name], B["mla"][name]
    print(f"  {name:28s} q {d(ma['q'], mb['q'])} | topk {d(ma['topk'].float(), mb['topk'].float())} | out {d(ma['out'], mb['out'])}")
for k in ("kda_conv", "kda_init", "kda_fkda", "idx_topk"):
    for i, (x, y) in enumerate(zip(A.get(k, []), B.get(k, []))):
        print(f"  {k}[{i}] {d(x, y)}")
