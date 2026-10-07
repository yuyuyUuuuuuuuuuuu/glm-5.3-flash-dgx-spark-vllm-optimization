"""Compare, per KV view, the rows a fresh request's final short prefill step reads (its own blocks) against the rows
the paired prefix-hit request reads (the shared cached blocks), from the per-step row hashes in records.pt."""
import json, sys, torch, collections
run = sys.argv[1]
r = json.load(open(run + "/result.json"))
rec = torch.load(run + "/records.pt", weights_only=False)["records"]
bystep = {x["step"]: x for x in rec}
groups = r["meta"]["groups"]
layer2g = {}
for g in groups:
    for ln in g["layers"]:
        layer2g[ln] = g
def vgroup(vname):
    base = vname.split("[")[0]
    return layer2g.get(base)
for ph in r["PH"]:
    L = ph["len"]
    f, h = ph["fresh"], ph["hit"]
    hc = h["cached"]
    if hc == 0:
        continue
    # pre-step snapshot = snapshot after the previous step
    def last_prefill_step(arm):
        st = [s for s in range(*arm["steps"]) if bystep[s]["prefill"]]
        return st[-1]
    sf, sh = last_prefill_step(f), last_prefill_step(h)
    vf, vh = bystep[sf - 1]["view_hash"], bystep[sh - 1]["view_hash"]
    diffs = collections.Counter(); tot = collections.Counter(); ex = {}
    for vname in vf:
        g = vgroup(vname)
        if g is None:
            continue
        gid = g["gid"]; bs = g["block_size"]
        bf, bh_ = f["blocks"][gid], h["blocks"][gid]
        if g["spec"] == "MambaSpec":
            cols = [hc // bs - 1]
        elif bs == 4:          # kpool tail ring (1 block per request)
            cols = [0]
        else:
            cols = list(range(hc // bs))
        for c in cols:
            if c >= len(bf) or c >= len(bh_):
                continue
            a, b = int(vf[vname][bf[c]]), int(vh[vname][bh_[c]])
            key = (gid, g["spec"], vname.split(".")[-1] if "[" not in vname else vname.split(".")[-1])
            tot[key] += 1
            if a != b:
                diffs[key] += 1; ex.setdefault(key, (vname, c, bf[c], bh_[c]))
    print(f"L={L} hit={hc} fresh step {sf} hit step {sh}")
    for k in sorted(tot):
        print(f"   g{k[0]} {k[1]:24s} {k[2]:40s} differ {diffs[k]}/{tot[k]}  {ex.get(k, '')}")
