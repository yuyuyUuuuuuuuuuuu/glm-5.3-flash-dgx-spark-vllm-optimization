"""decode-vs-fresh-prefill consistency of a handoff run (result.json: A = request 0 decode top-5 logprobs, B_j = fresh
prefill of prompt + gen[:j], its next-token top-5). KL over the top-5 union with a floor, top-1 agreement.
Usage: cons.py <run dir> [...]"""
import json, math, sys
def kl(pa, pb):
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    a = {k: math.exp(pa.get(k, floor)) for k in keys}; b = {k: math.exp(pb.get(k, floor)) for k in keys}
    za, zb = sum(a.values()), sum(b.values())
    return sum(a[k] / za * math.log((a[k] / za) / (b[k] / zb)) for k in keys)
for d in sys.argv[1:]:
    r = json.load(open(d + "/result.json"))
    A, B = r["logprobs"], r.get("B", [])
    n = min(len(A), len(B))
    ks = [kl(A[j], B[j]) for j in range(n)]
    t1 = sum(max(A[j], key=A[j].get) == max(B[j], key=B[j].get) for j in range(n))
    mean = sum(ks) / n if n else float("nan")
    print(f"{d}: n={n} KL(A||B) mean {mean:.5f} max {max(ks):.4f} p50 {sorted(ks)[n//2]:.5f} | top-1 agree {t1}/{n}"
          f" | first8 {[round(x,4) for x in ks[:8]]}")
