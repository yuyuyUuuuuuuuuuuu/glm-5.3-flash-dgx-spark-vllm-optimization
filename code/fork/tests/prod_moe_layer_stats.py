"""Per-MoE-layer timeline of the production decode trace (R15, rank 0; $TF_EXL3_ASSETS/decode-prof/p6h.json.gz):
router -> first grouped GEMV (pre), grouped g/u, mid, grouped d, post, gap to the all-reduce, where the shared expert
(aux stream) ends relative to the down GEMV. Host-side analysis only (docs/DEC_MOEGLUE.md section 1)."""
import gzip, json, os, sys, re, collections, statistics as st
d = json.load(gzip.open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "decode-prof/p6h.json.gz")))
ev = d["traceEvents"]
k = sorted([e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and e.get("ph")=="X"], key=lambda e: e["ts"])
main_s = None
S = collections.defaultdict(list)
n=0
for i,e in enumerate(k):
    if "rot_in" not in e["name"]: continue
    s = e["args"]["stream"]
    # walk back on same stream to router gemm_bf16_kernel [18,4,1]
    j=i
    while j>0 and not ("gemm_bf16_kernel" in k[j]["name"] and k[j]["args"].get("stream")==s): j-=1
    router=k[j]
    # previous AR on same stream (before router)
    jj=j
    while jj>0 and not ("AllReduce" in k[jj]["name"]): jj-=1
    ar_prev=k[jj]
    # forward: grouped kernels on same stream
    g=[m for m in range(i+1,min(i+40,len(k))) if "grouped_kernel" in k[m]["name"] and k[m]["args"].get("stream")==s][:2]
    if len(g)<2: continue
    g1,g2=k[g[0]],k[g[1]]
    # next AR after g2
    m=g[1]
    while m<len(k) and "AllReduce" not in k[m]["name"]: m+=1
    if m>=len(k): continue
    ar=k[m]
    # last kernel on main stream s before ar
    mains=[x for x in k[g[1]:m] if x["args"].get("stream")==s]
    lastmain=mains[-1]
    # aux kernels between router and ar on other streams (FP8 / act)
    aux=[x for x in k[j:m] if x["args"].get("stream")!=s and ("fp8" in x["name"] or "act_and_mul" in x["name"])]
    auxend=max(x["ts"]+x["dur"] for x in aux) if aux else float('nan')
    end_main = lastmain["ts"]+lastmain["dur"]
    S["ar_prev_to_router"].append(router["ts"]-(ar_prev["ts"]+ar_prev["dur"]))
    S["AR_before_moe"].append(ar_prev["dur"])
    jo = jj - 1                                   # o_proj = the last FP8 GEMV that ends before this all-reduce starts
    while jo > 0 and not ("fp8_gemv" in k[jo]["name"] and k[jo]["ts"] + k[jo]["dur"] <= ar_prev["ts"] + 1): jo -= 1
    S["oproj_end_to_router(window)"].append(router["ts"] - (k[jo]["ts"] + k[jo]["dur"]))
    S["oproj_us"].append(k[jo]["dur"])
    S["pre(router->g1)"].append(g1["ts"]-router["ts"])
    S["g1"].append(g1["dur"]); S["g2"].append(g2["dur"])
    S["mid"].append(g2["ts"]-(g1["ts"]+g1["dur"]))
    S["post(g2end->mainend)"].append(end_main-(g2["ts"]+g2["dur"]))
    S["gap(mainend->AR)"].append(ar["ts"]-end_main)
    S["auxend-g2end"].append(auxend-(g2["ts"]+g2["dur"]))
    S["AR"].append(ar["dur"])
    S["router->AR"].append(ar["ts"]-router["ts"])
    S["rot_in"].append(e["dur"])
    S["naux"].append(len(aux))
    n+=1
print("layers", n)
for kk,v in S.items():
    v=sorted(v); q=lambda p: v[int(p*(len(v)-1))]
    print(f"{kk:24s} med {st.median(v):8.1f}  p10 {q(.1):8.1f}  p90 {q(.9):8.1f}  mean {st.mean(v):8.1f}")
