import json, math, sys
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k]/za*math.log((a[k]/za)/(b[k]/zb)) for k in keys)
def ent(p):
    z = math.log(sum(math.exp(v) for v in p.values())); return -sum(math.exp(v-z)*(v-z) for v in p.values())
top = lambda p: max(p, key=p.get)
r = json.load(open(sys.argv[1]))
print("warm", r["runs"]["warm"]["len"], "cached", r["runs"]["warm"]["cached"], "B", r["B"])
for g in r["groups"]:
    A = {k: g[k] for k in ("hit", "fresh", "fresh2", "hit2")}
    print(f"sfx {g['suffix']:4d}: cached hit {A['hit']['cached']} hit2 {A['hit2']['cached']} fresh {A['fresh']['cached']} fresh2 {A['fresh2']['cached']} | decode_next {g['decode_next']} first ids " + " ".join(f"{k}={A[k]['ids'][0]}" for k in A))
    # first divergence vs fresh
    for k in ("fresh2", "hit", "hit2"):
        a, b = A["fresh"]["ids"], A[k]["ids"]
        d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        if d is None:
            print(f"   fresh vs {k:6s}: identical {len(a)} tokens")
        else:
            pf = A["fresh"]["lp"][d]; gap = pf[str(a[d])] - pf.get(str(b[d]), min(pf.values()) - 1)
            print(f"   fresh vs {k:6s}: diverge at {d} (fresh margin there {gap:.3f})")
    f, f2, h, h2 = (A[k]["lp"][0] for k in ("fresh", "fresh2", "hit", "hit2"))
    print(f"   pos0 H {ent(f):.2f} | KL f/f2 {kl(f,f2):.4f} f/h {kl(f,h):.4f} h/h2 {kl(h,h2):.4f} f2/h {kl(f2,h):.4f} | top1 f=h {int(top(f)==top(h))} f=f2 {int(top(f)==top(f2))}")
