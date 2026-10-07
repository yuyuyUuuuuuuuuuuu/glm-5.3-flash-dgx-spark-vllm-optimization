"""Probe 6: Marlin M scan (N <= 4096 large-K shapes), data dependence of speed (power), small-M crossover."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import gpu_guard  # noqa: E402
import torch  # noqa: E402
from fp8_bench_common import quantize_like_prod, marlin  # noqa: E402
from probe_large_m import t_us  # noqa: E402
import fp8_gemv as G  # noqa: E402

dev = "cuda"


def newpath(E, layer, alpha, n, k, x, mc=2048):
    M = x.shape[0]
    wd = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
    out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
    E.dequant(wd, layer.weight, 0, n, k)
    for m0 in range(0, M, mc):
        mm = min(mc, M - m0)
        t = torch.mm(x[m0:m0 + mm], wd.t(), out_dtype=torch.float32)
        E.scale_cast(out[m0:m0 + mm], t, alpha, None)
    return out


def main():
    gpu_guard(8.0)
    G.STATE.large = True
    E = G.ext_large()
    g = torch.Generator(device=dev).manual_seed(0)
    for n, k in [(4096, 8192), (4096, 20480)]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        line = f"marlin M scan N={n} K={k}:"
        for M in (4096, 6144, 8192, 8256, 9216, 10240, 12288, 13824):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            t = t_us(lambda: marlin(layer, x, n, k))
            line += f" M{M}:{t/1e3:.2f}ms/{2*M*n*k/t/1e6:.0f}TF"
            if M == 13824:
                t2 = t_us(lambda: [marlin(layer, x[a:a + 4608], n, k) for a in range(0, M, 4608)])
                line += f" (3x4608 split: {t2/1e3:.2f}ms)"
            del x
        print(line, flush=True)
        del layer, fp8
        torch.cuda.empty_cache()
    # data dependence (power): in_proj at M = 13824
    n, k = 12576, 4096
    w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    layer, fp8, sc = quantize_like_prod(w)
    del w
    npad = layer.weight.shape[1] // 4
    alpha = G.large_alpha(layer.weight_scale, n)
    M = 13824
    base = torch.randn(M, k, device=dev, generator=g)
    for name, x in [("randn", base.to(torch.bfloat16)), ("zeros", torch.zeros(M, k, device=dev, dtype=torch.bfloat16)),
                    ("sparse50", (base * (torch.rand(M, k, device=dev, generator=g) < 0.5)).to(torch.bfloat16)),
                    ("rms+outliers", (base * torch.where(torch.rand(1, k, device=dev, generator=g) < 0.01, 20.0, 1.0)
                                      ).to(torch.bfloat16))]:
        tm = t_us(lambda: marlin(layer, x, n, k))
        tn = t_us(lambda: newpath(E, layer, alpha, n, k, x))
        print(f"data {name:13s}: marlin {tm/1e3:.2f} ms, new {tn/1e3:.2f} ms, x{tm/tn:.2f}", flush=True)
    # small-M crossover
    for n, k in [(12576, 4096), (12288, 4096)]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        npad = layer.weight.shape[1] // 4
        alpha = G.large_alpha(layer.weight_scale, n)
        line = f"crossover N={n}:"
        for M in (128, 192, 256, 320, 384, 448, 512, 768, 1024, 1536):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            tm = t_us(lambda: marlin(layer, x, n, k), iters=9)
            tn = t_us(lambda: newpath(E, layer, alpha, n, k, x), iters=9)
            line += f" M{M}:{tm/1e3:.2f}/{tn/1e3:.2f}={tm/tn:.2f}x"
        print(line, flush=True)
        del layer, fp8
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
