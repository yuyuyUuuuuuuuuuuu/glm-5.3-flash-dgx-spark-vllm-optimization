"""[dec-hostloop review] In a production decode trace, list every CUDA runtime call on the worker main thread between the
return of the blocking per-step seq_lens D2H copy and the target cudaGraphLaunch: any other D2H copy or sync there would
become the new blocking point once GLM53_DEC_HOSTLOOP removes the first one. Usage: python3 <this> trace.json.gz
"""
import gzip, json, collections, statistics as S, sys
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
rts = sorted((e for e in ev if e.get("cat") in ("cuda_runtime","cuda_driver") and e.get("ph")=="X"), key=lambda e:e["ts"])
gpu = [e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and e.get("ph")=="X"]
gcorr = collections.defaultdict(list)
for g in gpu:
    c = g.get("args",{}).get("correlation")
    if c is not None: gcorr[c].append(g)
blocking = [e for e in rts if e["name"]=="cudaMemcpyAsync" and e["dur"]>1000]
print("blocking", len(blocking), "tids", collections.Counter(b["tid"] for b in blocking))
names_all = collections.Counter()
dtoh = collections.Counter()
sync = collections.Counter()
per_step_d2h = []
for b in blocking[:]:
    t0 = b["ts"]+b["dur"]
    # next graph launch on same thread
    nxt = [x for x in rts if x["tid"]==b["tid"] and x["ts"]>t0 and x["name"]=="cudaGraphLaunch"]
    if not nxt: continue
    t1 = nxt[0]["ts"]
    win = [x for x in rts if x["tid"]==b["tid"] and t0<=x["ts"]<t1]
    nd = 0
    for x in win:
        names_all[x["name"]]+=1
        if "ynchronize" in x["name"] or "Query" in x["name"]:
            sync[(x["name"])]+=1
        if "Memcpy" in x["name"]:
            for g in gcorr.get(x["args"].get("correlation"),[]):
                nm = g["name"]
                if "DtoH" in nm or "Device -> Pageable" in nm or "Device -> Pinned" in nm:
                    dtoh[(nm, round(x["dur"]))]+=1; nd+=1
    per_step_d2h.append(nd)
print("runtime calls in window (per all steps):")
for k,v in names_all.most_common(): print("  ",k,v)
print("sync-like:", sync)
print("DtoH memcpys in window:", dtoh.most_common(20))
print("per-step D2H count:", collections.Counter(per_step_d2h))
# memcpy kinds in window
kinds = collections.Counter()
for b in blocking:
    t0=b["ts"]+b["dur"]
    nxt=[x for x in rts if x["tid"]==b["tid"] and x["ts"]>t0 and x["name"]=="cudaGraphLaunch"]
    if not nxt: continue
    t1=nxt[0]["ts"]
    for x in rts:
        if x["tid"]==b["tid"] and t0<=x["ts"]<t1 and "Memcpy" in x["name"]:
            for g in gcorr.get(x["args"].get("correlation"),[]): kinds[g["name"]]+=1
print("memcpy kinds:", kinds.most_common())
# what's the blocking copy's gpu name
bk = collections.Counter()
for b in blocking:
    for g in gcorr.get(b["args"].get("correlation"),[]): bk[g["name"]]+=1
print("blocking kinds", bk)
