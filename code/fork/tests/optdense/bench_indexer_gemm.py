"""Indexer BF16 GEMMs at production prefill shapes (per MLA layer): wq_b [T,1536]x[1536,4096], wk_weights_proj
[T,4096]x[4096,160], kpool gate F.linear [T,4096]x[4096,128]; what cuBLAS picks via F.linear vs alternatives."""
import statistics, sys, torch
import torch.nn.functional as F
dev = "cuda"


def tmed(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


from torch.profiler import profile, ProfilerActivity
for T in [int(v) for v in sys.argv[1:]] or [13824, 4289]:
    for name, K, N in (("wq_b", 1536, 4096), ("wk_wp", 4096, 160), ("gate", 4096, 128)):
        x = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        ref = F.linear(x, w)
        t_lin = tmed(lambda: F.linear(x, w))
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            F.linear(x, w); torch.cuda.synchronize()
        kn = [e.name[:60] for e in p.events() if e.device_type.name == "CUDA"]
        wt = w.t().contiguous()
        t_mm = tmed(lambda: torch.mm(x, wt))
        eq = torch.equal(torch.mm(x, wt), ref)
        fl = 2 * T * K * N
        print(f"T={T} {name:6s} [{K}x{N}] F.linear {t_lin:.3f} ms ({fl / t_lin / 1e9:.0f} TF) kernels {kn} | "
              f"mm(x, w.t().contiguous()) {t_mm:.3f} ms bitwise {eq}", flush=True)
