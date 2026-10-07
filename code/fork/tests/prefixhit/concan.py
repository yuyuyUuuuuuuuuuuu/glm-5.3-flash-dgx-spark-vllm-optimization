import sys, torch
run = sys.argv[1]
for i in (0, 1):
    try:
        d = torch.load(f"{run}/conc_req{i}.pt", weights_only=False)
    except FileNotFoundError:
        continue
    P = d["prompt_len"] // 4
    print(f"{run} req {i}: prompt pools {P}, total pools {d['n'] // 4}")
    for name in sorted(d["ref"]):
        def deq(x):
            v = x[:, :128].contiguous().view(torch.float8_e4m3fn).float()
            s = x[:, 128:132].contiguous().view(torch.float32)
            return v * s
        a, b = deq(d["got"][name]), deq(d["ref"][name])
        rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
        def stats(r):
            if r.numel() == 0: return "-"
            return f"p50 {r.median():.3g} p99 {r.quantile(0.99):.3g} max {r.max():.3g} n>0.25: {int((r > 0.25).sum())}/{r.numel()}"
        print(f"   {name.split('.')[2]:>3}: prompt {stats(rel[:P])} | gen {stats(rel[P:])} | gen>0.25 idx {(rel[P:] > 0.25).nonzero().flatten().tolist()[:10]}")
