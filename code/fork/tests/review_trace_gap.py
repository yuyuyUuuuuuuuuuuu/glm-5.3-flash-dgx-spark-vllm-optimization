"""[dec-hostloop review] Per step in a production decode trace: GPU time from the drafter end (start of the blocking
seq_lens D2H) to the first kernel of the next target CUDA graph (to_graph), i.e. what GLM53_DEC_HOSTLOOP can remove
(minus ~0.13 ms of small ops + the eager AR floor that remain). Usage: python3 <this> trace.json.gz
"""
import gzip, json, statistics as S, sys
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
rts = [e for e in ev if e.get("cat") in ("cuda_runtime","cuda_driver") and e.get("ph")=="X"]
byc = {e["args"].get("correlation"): e for e in rts if e.get("args",{}).get("correlation") is not None}
gpu = sorted([e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and e.get("ph")=="X"], key=lambda e:e["ts"])
d2h = [g for g in gpu if g["name"]=="Memcpy DtoH (Device -> Pageable)"]
rows=[]
for d in d2h:
    t0=d["ts"]
    ar = next((g for g in gpu if g["ts"]>t0 and "AllReduce" in g["name"] and byc.get(g["args"].get("correlation"),{}).get("name")!="cudaGraphLaunch"), None)
    if ar is None or ar["ts"]-t0>20000: continue
    g1 = next((g for g in gpu if g["ts"]>=ar["ts"]+ar["dur"] and byc.get(g["args"].get("correlation"),{}).get("name")=="cudaGraphLaunch"), None)
    # launch call of that graph
    gl = byc.get(g1["args"].get("correlation"))
    # idle between AR end and graph first kernel
    ops_between = [g for g in gpu if ar["ts"]+ar["dur"]<=g["ts"]<g1["ts"]]
    rows.append(dict(to_graph=g1["ts"]-t0, ar_end_to_graph=g1["ts"]-(ar["ts"]+ar["dur"]), nb=len(ops_between),
                     busy_between=sum(g["dur"] for g in ops_between), launch_call_to_first=g1["ts"]-gl["ts"], launch_dur=gl["dur"]))
for k in rows[0]:
    v=sorted(r[k] for r in rows); print("%-22s med %8.1f  mean %8.1f  p10 %8.1f p90 %8.1f" % (k, S.median(v), S.mean(v), v[len(v)//10], v[9*len(v)//10]))
