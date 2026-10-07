"""Pure-read floor on this GPU: time to stream N distinct buffers of B bytes each (graph of L reads), several CTA
counts; the practical lower bound for a weight-streaming kernel of that size (launch + ramp + drain included)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_smallops as SO
from bench_smallops import graph_of, ab
dev = torch.device("cuda")
for mb, L in ((12.58, 11), (8.39, 11), (41.9, 3), (1.57, 89)):
    n = int(mb * 1e6) // 16 * 16
    bufs = [torch.empty(n // 2, dtype=torch.bfloat16, device=dev).fill_(1.0) for _ in range(L)]
    g = {}
    for ctas in (48, 96, 192, 384, 768):
        def f(ctas=ctas):
            for b in bufs:
                SO.l2_prefetch(b, ctas)
        g[f"read {mb:.2f} MB, {ctas} CTAs"] = graph_of(f)
    r = ab(g, L)
    for k, (m, sp) in r.items():
        print(f"{k:<28s} {m:8.2f} us  ({mb * 1e3 / m:6.1f} GB/s)  spread {sp * 100:4.1f}%", flush=True)
