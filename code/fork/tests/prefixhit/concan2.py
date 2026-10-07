import sys, torch
run, i = sys.argv[1], int(sys.argv[2])
d = torch.load(f"{run}/conc_req{i}.pt", weights_only=False)
P = d["prompt_len"] // 4
for name in sorted(d["ref"])[:2]:
    def deq(x):
        return x[:, :128].contiguous().view(torch.float8_e4m3fn).float() * x[:, 128:132].contiguous().view(torch.float32)
    a, b = deq(d["got"][name]), deq(d["ref"][name])
    rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
    bad = ((rel > 0.25) | rel.isnan()).nonzero().flatten().tolist()
    print(name, "bad pools:", bad[:40], "-> token positions", [p * 4 for p in bad[:3]], "... block-relative entries", [p % 1152 for p in bad[:20]])
