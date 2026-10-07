import sys, torch
run, L = sys.argv[1], sys.argv[2]
arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fresh", "hit"]
D = {a: torch.load(f"{run}/phdump_L{L}_{a}.pt", weights_only=False) for a in arms}
a, b = arms[0], arms[1]
A, B = D[a], D[b]
print(f"L={L} {a} rid {A['rid']} computed {A['computed']} n {A['n']} | {b} rid {B['rid']} computed {B['computed']}")
for name in sorted(A["views"]):
    va, vb = A["views"][name], B["views"].get(name)
    if vb is None:
        print("  missing", name); continue
    ta, tb = va["data"], vb["data"]
    m = va.get("m", 1)
    ta = ta.reshape(len(va["ids"]), -1); tb = tb.reshape(len(vb["ids"]), -1)
    line = []
    for i, col in enumerate(va["cols"]):
        x, y = ta[i], tb[i]
        if x.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            xb, yb = x.view(torch.uint8), y.view(torch.uint8)
            nd = int((xb != yb).sum())
            line.append(f"c{col}:{nd}/{xb.numel()}B")
        else:
            xf, yf = x.float(), y.float()
            nd = int((xf != yf).sum())
            mx = float((xf - yf).abs().max()) if nd else 0.0
            sc = float(xf.abs().max())
            line.append(f"c{col}:{nd}/{xf.numel()} max{mx:.3g}/{sc:.3g}")
    print(f"  {name:48s} {str(tuple(ta.shape)):28s} {ta.dtype} ids {va['ids'][:8]}/{vb['ids'][:8]}: " + " ".join(line))
