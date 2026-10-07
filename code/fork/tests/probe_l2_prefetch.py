"""Does l2_prefetch make the next kernel's weight L2-resident? mhc_fused (smallops, M=5) over 89 distinct weights:
  cold: [mhc(w_i)] x 89;  pf-serial: [prefetch(w_i); mhc(w_i)] x 89;  twice: [mhc(w_i); mhc(w_i)] x 89 (2nd warm)
  pf-only: [prefetch(w_i)] x 89"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glm53_smallops as SO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_smallops import graph_of, ab
dev = torch.device("cuda")
M, hc, H, n3, L = 5, 4, 4096, 24, 89
S = SO.mhc_splits(M)
ws = [(torch.randn(n3, hc * H, device=dev) * 0.02).bfloat16().float() for _ in range(L)]
x = torch.randn(M, H, device=dev).bfloat16(); res = torch.randn(M, hc, H, device=dev).bfloat16()
post = torch.rand(M, hc, device=dev); comb = torch.rand(M, 16, device=dev)
yp = torch.empty(S, M, n3, device=dev); rp = torch.empty(S, M, device=dev); ro = torch.empty_like(res)
def cold():
    for i in range(L): SO.mhc_fused(comb, post, res, x, ws[i], S, yp, rp, ro)
def twice():
    for i in range(L):
        SO.mhc_fused(comb, post, res, x, ws[i], S, yp, rp, ro); SO.mhc_fused(comb, post, res, x, ws[i], S, yp, rp, ro)
def pf_serial(ctas):
    def f():
        for i in range(L):
            SO.l2_prefetch(ws[i], ctas); SO.mhc_fused(comb, post, res, x, ws[i], S, yp, rp, ro)
    return f
def pf_only(ctas):
    def f():
        for i in range(L): SO.l2_prefetch(ws[i], ctas)
    return f
g = {"cold": graph_of(cold), "twice (2 calls)": graph_of(twice), "pf16 serial": graph_of(pf_serial(16)),
     "pf48 serial": graph_of(pf_serial(48)), "pf96 serial": graph_of(pf_serial(96)),
     "pf16 only": graph_of(pf_only(16)), "pf48 only": graph_of(pf_only(48)), "pf96 only": graph_of(pf_only(96))}
r = ab(g, L)
for n, (m, sp) in r.items():
    print(f"{n:<20s} {m:7.2f} us per layer  spread {sp*100:.1f}%")
