"""Full-output check that every new-vs-Marlin difference is explained by fp32 summation order: for all elements,
|y - ym| <= ulp(max(|y|,|ym|)) + 2 * (K/16+2) * 2^-23 * sum_k |x_k * w_k| * s  (both within the fp32 bound).
Also runs the kernels once under exact-size allocations (for compute-sanitizer when RV_SMALL=1)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import fp8_gemv as F
from fp8_bench_common import quantize_like_prod, marlin

dev = "cuda"
small = os.environ.get("RV_SMALL") == "1"
g = torch.Generator(device=dev).manual_seed(11)
F.STATE.large = True
F.STATE.large_gemm = "tilelang"
Lx = F.ext_large()


def ulp(a):
    e = torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -126)))
    return torch.pow(2.0, e - 7)


for n, k in ((12576, 4096), (12288, 4096), (4096, 20480)):
    w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
         torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
    layer, fp8, sc = quantize_like_prod(w)
    del w
    alpha = F.large_alpha(layer.weight_scale, n)
    wd = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
    Lx.dequant(wd, layer.weight, 0, n, k)
    for kind in (("normal",) if small else ("normal", "outlier", "tiny", "rows")):
        M = 130 if small else 4096
        x = torch.randn(M, k, device=dev, generator=g)
        if kind == "outlier":
            x[:, torch.randperm(k, device=dev, generator=g)[:8]] *= 300.0
        elif kind == "tiny":
            x = x * 1e-4
        elif kind == "rows":
            x = x * torch.pow(10.0, torch.empty(M, 1, device=dev).uniform_(-4, 2, generator=g))
        x = x.to(torch.bfloat16)
        y = F.large_forward(x, layer.weight, alpha, None, n, k)
        yc = F.large_forward(x, layer.weight, alpha, None, n, k, backend="cublas")
        ym = marlin(layer, x, n, k)
        S = torch.mm(x.float().abs(), wd.float().abs().t()) * alpha[:n]
        bound = ulp(torch.maximum(y.float().abs(), ym.float().abs())) + 2 * (k / 16 + 2) * 2.0 ** -23 * S
        r = ((y.float() - ym.float()).abs() / (bound + 1e-38))
        print(f"{n}x{k} M={M} {kind:7s}: max |new-Marlin| / (1 ulp + 2 fp32 bounds) = {r.max().item():.3f}, "
              f"elements > 1: {(r > 1).sum().item()}, tilelang==cublas {torch.equal(y, yc)}, "
              f"exact zeros new/Marlin {(y == 0).sum().item()}/{(ym == 0).sum().item()}", flush=True)
        del x, y, yc, ym, S, bound, r
    del layer, fp8, sc, wd
    torch.cuda.empty_cache()
print("DONE")
