"""bench_prod_decode の 16ms が何に使われているか分解する。"""
import time, torch
from prod_baseline import load_xl, make_experts, ptr_tables, routing, MOE_ACT_SILU
xl = load_xl(); dev = "cuda"
D, NI, E, T, TOPK = 4096, 768, 64, 1, 8
ex = make_experts(E, D, NI, dev); P = ptr_tables(ex, dev)
x = (torch.randn(T, D, device=dev) * 0.5).half()
ids = torch.randperm(E, device=dev)[:TOPK].view(1, TOPK)
w = torch.softmax(torch.randn(T, TOPK, device=dev), -1)
conc = int(xl.exl3_moe_max_concurrency(0)); print("concurrency:", conc)
temps = tuple(torch.empty(s, dtype=torch.float16, device=dev) for s in
              [(conc,128,D),(conc,128,D),(conc,128,NI),(conc,128,NI)])
ec, ts, wsr = routing(ids, w, E)
out = torch.zeros(T, D, dtype=torch.float32, device=dev)
base = (x, out, ec, ts, wsr, *temps, MOE_ACT_SILU, 4, 4, 4,
        P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"],
        P["down_trellis"], P["down_suh"], P["down_svh"], True, False, True, False, True, False, 7.0)
def ev(fn, it=30):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000
for label, fn in [
    ("exl3_moe(num_active=-1)", lambda: xl.exl3_moe(*base, -1)),
    ("exl3_moe(num_active=TOPK)", lambda: xl.exl3_moe(*base, TOPK)),
    ("exl3_moe(no num_active)", lambda: xl.exl3_moe(*base)),
]:
    try: print(f"{label:28s} {ev(fn):9.1f} us")
    except Exception as e: print(f"{label:28s} ERR {type(e).__name__}: {str(e)[:90]}")
print(f"{'routing()':28s} {ev(lambda: routing(ids, w, E)):9.1f} us")
print(f"{'max_concurrency()':28s} {ev(lambda: xl.exl3_moe_max_concurrency(0)):9.1f} us")
