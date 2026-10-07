"""GEMM cfg sweep for the hi+lo K + C shapes at M 13824 / 4289 (custom CUTLASS GEMM, random fp8 operands)."""
import os, statistics, sys
sys.path.insert(0, "/w")
import torch
os.environ.setdefault("TF_EXL3_JIT", "1")
import fp8_w8a8 as W
dev = "cuda"


def tmed(fn, n=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


E = W.ext()
ws = torch.empty(W.WS_BYTES, dtype=torch.uint8, device=dev)
for n, k in ((4096, 4096 + 256), (4096, 8192 + 512), (8192, 1536 + 256), (4096, 6144 + 512), (4096, 1024 + 128)):
    w = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
    alpha = torch.rand(n + 256, device=dev) + 0.5
    for M in (13824, 4289):
        a = torch.randn(M, k, device=dev).to(torch.float8_e4m3fn)
        sa = torch.rand(M, device=dev) + 0.5
        out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
        res = []
        for cfg in range(4):
            for sw in (1, 2, 4, 8):
                for ro in (0, 1, 2):
                    try:
                        res.append((tmed(lambda: E.fp8_w8a8_gemm(out, a, w, sa, alpha, cfg, sw, ro, ws)), cfg, sw, ro))
                    except Exception:
                        pass
        res.sort()
        base = W.gemm_choice(n, k - {4352: 256, 8704: 512, 1792: 256, 6656: 512, 1152: 128}[k], M)
        tb = [t for t, c, s, r in res if (c, s, r) == base]
        print(f"({n}, {k}) M={M}: best " + ", ".join(f"{t:.3f} cfg({c},{s},{r})" for t, c, s, r in res[:3]) +
              f" | base cfg {base}: {tb[0] if tb else float('nan'):.3f}", flush=True)
