import json, math, sys
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k]/za*math.log((a[k]/za)/(b[k]/zb)) for k in keys)
R = sys.argv.pop(1)
for n in sys.argv[1:]:
    r = json.load(open(R + n + "/result.json"))
    F, Hh, hc = r["B"], r["B_hit"], r["B_hit_cached"]
    groups = {}
    for j, (f, h, c) in enumerate(zip(F, Hh, hc)):
        groups.setdefault(c, []).append((j, kl(f, h), max(f, key=f.get) == max(h, key=h.get)))
    for c, v in sorted(groups.items()):
        ks = [x[1] for x in v]
        print(f"{n}: hit {c:6d} tokens (j {v[0][0]}..{v[-1][0]}): fresh-vs-hit KL mean {sum(ks)/len(ks):.5f} max {max(ks):.4f} "
              f"p50 {sorted(ks)[len(ks)//2]:.5f} top1 {sum(x[2] for x in v)}/{len(v)}; fresh cached {sorted(set(r['B_cached']))}")
if len(sys.argv) > 2:
    a, b = (json.load(open(R + n + "/result.json")) for n in sys.argv[1:3])
    g0, g1 = a["requests"][0]["gen"], b["requests"][0]["gen"]
    same = next((j for j, (x, y) in enumerate(zip(g0, g1)) if x != y), len(g0))
    for key in ("B", "B_hit"):
        ks = [kl(a[key][j], b[key][j]) for j in range(min(same + 1, len(a[key])))]
        print(f"{sys.argv[1]} vs {sys.argv[2]} {key} (j <= {same}, identical tokens): KL max {max(ks):.5f} mean {sum(ks)/len(ks):.5f} "
              f"nonzero {sum(k > 1e-9 for k in ks)}/{len(ks)}")
