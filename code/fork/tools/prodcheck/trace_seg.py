import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1])); ev = d["traceEvents"]
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"], key=lambda e: e["ts"])
def short(n):
    n = re.sub(r"<.*", "", n).replace("void ", "").replace("(anonymous namespace)::", "")
    n = re.sub(r"cutlass::Kernel2", "cutlass", n)
    return n[:34]
def full(e):
    n = e["name"]
    m = re.search(r"cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_(\w+?)_(tn|nn|nt)", n)
    if m: return "wmma_bf16_" + m.group(1) + "_" + m.group(2)
    if "Marlin" in n: return "marlin"
    return short(n)
idx = [i for i, e in enumerate(k) if "AllGather" in e["name"]]
a, b = idx[20], idx[22]; seq = k[a:b]
segs = []; cur = []
for e in seq:
    cur.append(e)
    if "AllReduce" in e["name"] or "AllGather" in e["name"]:
        segs.append(cur); cur = []
segs.append(cur)
print("segments", len(segs))
for i, s in enumerate(segs):
    tot = sum(e["dur"] for e in s)
    kind = "MOE" if any("grouped_kernel" in e["name"] for e in s) else ("KDA" if any("delta_rule" in e["name"] for e in s) else ("MLA" if any("mla::" in e["name"] for e in s) else ""))
    items = collections.OrderedDict()
    for e in s:
        n = full(e); c, t = items.get(n, (0, 0.0)); items[n] = (c + 1, t + e["dur"])
    desc = " | ".join(f"{n}x{c}:{t:.0f}" for n, (c, t) in items.items())
    print(f"{i:3d} {kind:3s} {tot:7.0f}us  {desc[:400]}")
