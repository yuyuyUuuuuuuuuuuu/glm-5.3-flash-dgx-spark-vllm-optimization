import sys, torch
run, L, a, b = sys.argv[1:5]
A = torch.load(f"{run}/phdump_L{L}_{a}.pt", weights_only=False); B = torch.load(f"{run}/phdump_L{L}_{b}.pt", weights_only=False)
for name in sorted(A["views"]):
    if not name.endswith("[0]"):
        continue
    va, vb = A["views"][name], B["views"][name]
    ta, tb = va["data"][0].float(), vb["data"][0].float()     # checkpoint column (state at `computed`)
    rows = (ta != tb).any(-1).nonzero().flatten().tolist()
    zr_a = [(ta[i] == 0).all().item() for i in range(ta.shape[0])]
    print(f"{name} shape {tuple(ta.shape)} differing state rows {rows}  fresh zero rows {[i for i,z in enumerate(zr_a) if z]}")
    break
