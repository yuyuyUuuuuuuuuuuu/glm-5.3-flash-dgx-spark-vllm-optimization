"""History of the row hash of A0's cached blocks across steps: does a cached (shared) block change after A0?"""
import json, sys, torch
run = sys.argv[1]
r = json.load(open(run + "/result.json"))
rec = torch.load(run + "/records.pt", weights_only=False)["records"]
groups = r["meta"]["groups"]
a0 = [v for k, v in r["bh"].items() if k.startswith("0-") or k == "0"][0]
a0_steps = r["requests"][0]["steps"]
last = a0_steps[1] - 1
print("A0 block ids per group:", [b[:9] for b in a0])
base = {x["step"]: x for x in rec}[last]["view_hash"]
for g in groups:
    gid, bs = g["gid"], g["block_size"]
    ncols = min(len(a0[gid]), 32400 // bs) if bs > 4 else 0
    for vname in base:
        if vname.split("[")[0] not in g["layers"]:
            continue
        for c in range(ncols):
            b = a0[gid][c]
            if b == 0:
                continue
            ch = [x["step"] for x in rec if x["step"] > last and int(x["view_hash"][vname][b]) != int(base[vname][b])]
            if ch:
                print(f"g{gid} {vname} col {c} block {b}: changed at steps {ch[:6]}{'...' if len(ch) > 6 else ''}")
