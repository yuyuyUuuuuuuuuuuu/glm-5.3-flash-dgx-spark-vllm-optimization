import json, math, sys
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k]/za*math.log((a[k]/za)/(b[k]/zb)) for k in keys)
top = lambda p: max(p, key=p.get)
for path in sys.argv[1:]:
    r = json.load(open(path))
    print("==", path, r["model"])
    print(f"{'ckpt':>6} {'sfx':>4} {'hit cached':>11} | KL f/f2  KL f/h  KL h/h2  KL f/h2 | top1 f=f2 f=h h=h2 | p(top) fresh  maxabs dlp(top5) f/h")
    for g in r["runs"]:
        if "hit" not in g: continue
        f, f2, h, h2 = (g[k]["lp"] for k in ("fresh", "fresh2", "hit", "hit2"))
        t5 = sorted(f, key=f.get, reverse=True)[:5]
        mx = max(abs(f[t] - h.get(t, -99)) for t in t5)
        print(f"{g['ckpt']:6d} {g['suffix']:4d} {str(g['hit']['cached'])+'/'+str(g['hit2']['cached']):>11} | {kl(f,f2):7.4f} {kl(f,h):7.4f} {kl(h,h2):7.4f} {kl(f,h2):7.4f} |"
              f"   {int(top(f)==top(f2))}    {int(top(f)==top(h))}    {int(top(h)==top(h2))}  | {math.exp(max(f.values())):.3f}  {mx:.4f}")

def dl(pa, pb):
    ks = [k for k in pa if k in pb]
    d = [abs(pa[k] - pb[k]) for k in ks]
    return (sum(d) / len(d) if d else float("nan")), (max(d) if d else float("nan")), len(ks)
for path in sys.argv[1:]:
    r = json.load(open(path))
    print("-- |dlogprob| over shared top-20 tokens (mean / max / n):  f-f2 | f-h | h-h2 | f-h2")
    for g in r["runs"]:
        if "hit" not in g: continue
        f, f2, h, h2 = (g[k]["lp"] for k in ("fresh", "fresh2", "hit", "hit2"))
        fmt = lambda t: f"{t[0]:.4f}/{t[1]:.3f}/{t[2]}"
        print(f"   ckpt {g['ckpt']} sfx {g['suffix']:4d}: {fmt(dl(f,f2))} | {fmt(dl(f,h))} | {fmt(dl(h,h2))} | {fmt(dl(f,h2))}")
