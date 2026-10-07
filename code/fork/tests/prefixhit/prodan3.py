import json, math, sys
def norm(p):
    z = math.log(sum(math.exp(v) for v in p.values())); return {k: v - z for k, v in p.items()}
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k]/za*math.log((a[k]/za)/(b[k]/zb)) for k in keys)
def ent(p):
    q = norm(p); return -sum(math.exp(v) * v for v in q.values())
top = lambda p: max(p, key=p.get)
r = json.load(open(sys.argv[1]))
print("model", r["model"], "| question tokens", r["q_tokens"])
for g in r["runs"]["groups"]:
    toks = {k: g[k]["tokens"] for k in ("hit", "fresh", "fresh2", "hit2")}
    print(f"suffix {g['suffix']}: cached hit {g['hit']['cached']} hit2 {g['hit2']['cached']} fresh {g['fresh']['cached']} fresh2 {g['fresh2']['cached']}")
    for k, v in toks.items():
        print(f"   {k:6s} {''.join(v)!r}")
    n = min(len(g[k]["lp"]) for k in toks)
    for j in range(n):
        same = len({tuple(toks[k][:j]) for k in toks}) == 1
        if not same:
            break
        f, f2, h, h2 = (g[k]["lp"][j] for k in ("fresh", "fresh2", "hit", "hit2"))
        print(f"   pos {j}: H(fresh) {ent(f):.3f} p_top {math.exp(max(norm(f).values())):.3f} | KL f/f2 {kl(f,f2):.5f}  f/h {kl(f,h):.5f}  h/h2 {kl(h,h2):.5f}  f/h2 {kl(f,h2):.5f} | top1 f=f2 {int(top(f)==top(f2))} f=h {int(top(f)==top(h))}")
