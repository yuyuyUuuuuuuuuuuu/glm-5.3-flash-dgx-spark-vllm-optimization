import sys, torch
run, L, a, b = sys.argv[1:5]
A = torch.load(f"{run}/phdump_L{L}_{a}.pt", weights_only=False); B = torch.load(f"{run}/phdump_L{L}_{b}.pt", weights_only=False)
bs = 4608; kpool = 4
for name in sorted(A["views"]):
    if not name.endswith("indexer.k_cache"):
        continue
    va, vb = A["views"][name], B["views"][name]
    ta = va["data"].reshape(len(va["ids"]), -1, 132); tb = vb["data"].reshape(len(vb["ids"]), -1, 132)
    for ci, col in enumerate(va["cols"]):
        d = (ta[ci] != tb[ci]).any(-1).nonzero().flatten().tolist()
        if d:
            # entries are pool slots inside the logical block (1152 pools per 4608-token block)
            toks = [(col * bs + e * kpool) for e in d]
            nbytes = [(ta[ci][e] != tb[ci][e]).sum().item() for e in d[:4]]
            print(f"{name} col {col}: {len(d)} pool entries differ: entries {d[:3]}..{d[-3:]} = token pos {toks[0]}..{toks[-1]+3}; bytes/entry {nbytes}")
            # zero entries?
            print("    fresh entry zero:", bool((ta[ci][d[0]] == 0).all()), " hit entry zero:", bool((tb[ci][d[0]] == 0).all()),
                  " fresh scale bytes", ta[ci][d[0]][128:].tolist(), " hit", tb[ci][d[0]][128:].tolist())
