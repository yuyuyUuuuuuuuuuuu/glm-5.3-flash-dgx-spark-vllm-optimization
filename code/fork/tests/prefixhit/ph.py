import json, math, sys
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k]/za*math.log((a[k]/za)/(b[k]/zb)) for k in keys)
def top(p): return max(p, key=p.get)
def margin(p):
    v = sorted(p.values(), reverse=True); return v[0]-v[1]
for path in sys.argv[1:]:
    r = json.load(open(path))
    print("==", path)
    print(f"{'L':>6} {'hitc':>6} {'sfx':>5} | {'KL f/f2':>8} {'KL f/h':>8} {'KL h/h2':>8} {'KL f/h2':>8} | top1 f=f2 f=h h=h2 | margin_f  p_f(top)")
    for rec in r["PH"]:
        L = rec["len"]; f, f2, h, h2 = (rec[a]["lp"][0] for a in ("fresh", "fresh2", "hit", "hit2"))
        hc = rec["hit"]["cached"]; hc2 = rec["hit2"]["cached"]
        print(f"{L:6d} {hc:6d}{'' if hc==hc2 else '/'+str(hc2)} {L-hc:5d} | {kl(f,f2):8.4f} {kl(f,h):8.4f} {kl(h,h2):8.4f} {kl(f,h2):8.4f} |"
              f"  {int(top(f)==top(f2))}   {int(top(f)==top(h))}   {int(top(h)==top(h2))}  | {margin(f):7.3f} {math.exp(max(f.values())):.3f}")
