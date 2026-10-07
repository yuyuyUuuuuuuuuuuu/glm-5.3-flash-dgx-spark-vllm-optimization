import gzip, json, os, sys, re, collections, statistics as st
d = json.load(gzip.open(os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "decode-prof/p6h.json.gz")))
ev = d["traceEvents"]
k = sorted([e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and e.get("ph")=="X"], key=lambda e: e["ts"])
mar = [i for i,e in enumerate(k) if "marlin::Marlin" in e["name"]]
steps = list(zip(mar[:-1], mar[1:]))
byM = collections.defaultdict(list)
for a,b in steps:
    seg = k[a:b]
    Ms = [e["args"]["grid"][0] for e in seg if "mhc_fused_tilelang_kernel" in e["name"]]
    M = collections.Counter(Ms).most_common(1)[0][0] if Ms else None
    g = [e for e in seg if "grouped_kernel" in e["name"]]
    gu = [e["dur"] for e in g if e["args"]["grid"][2] in (8,) or e["args"]["grid"][1]==43 and e["args"]["grid"][2]==8]
    span = (k[b]["ts"]-k[a]["ts"])/1e3
    tot = sum(e["dur"] for e in g)/1e3
    cats = collections.Counter()
    for e in seg:
        n = e["name"]
        if "grouped_kernel" in n: c="moe_grouped"
        elif "fp8_gemv" in n: c="fp8_gemv"
        elif "AllReduce" in n or "AllGather" in n: c="nccl"
        elif "Marlin" in n: c="marlin"
        elif "cutlass" in n: c="cutlass_bf16"
        elif "gemm_bf16" in n: c="bf16_gemv"
        elif "recurrent" in n: c="kda_recur"
        else: c="other"
        cats[c]+=e["dur"]/1e3
    byM[M].append((span, tot, len(g), cats))
for M in sorted(byM, key=lambda x: (x is None, x)):
    v = byM[M]
    print("M=%s steps=%d step_ms med %.2f  moe_ms med %.2f  ngrouped %s" % (M, len(v), st.median(x[0] for x in v), st.median(x[1] for x in v), collections.Counter(x[2] for x in v)))
    keys = sorted(set().union(*[x[3].keys() for x in v]))
    print("   " + "  ".join("%s=%.2f" % (c, st.median(x[3].get(c,0) for x in v)) for c in keys))
