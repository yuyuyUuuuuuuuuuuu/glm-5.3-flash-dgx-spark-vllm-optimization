"""Probe 7: chunking (N chunk for the BF16 weight temp, M chunk for the fp32 temp) vs speed; fc crossover."""
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


def make(E, layer, alpha, n, k, M, nc, mc):
    nchunks = -(-n // nc)
    c64 = -(-(-(-n // nchunks)) // 64) * 64
    tmp = torch.empty(c64, k, dtype=torch.bfloat16, device=dev)

    def f(x):
        out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
        for n0 in range(0, n, c64):
            c = min(c64, n - n0)
            E.dequant(tmp[:c], layer.weight, n0, c, k)
            for m0 in range(0, M, mc):
                mm = min(mc, M - m0)
                t = torch.mm(x[m0:m0 + mm], tmp[:c].t(), out_dtype=torch.float32)
                E.scale_cast(out[m0:m0 + mm, n0:n0 + c], t, alpha[n0:n0 + c], None)
        return out
    return f, (c64 * k * 2 + min(mc, M) * c64 * 4) / 2**20


def main():
    gpu_guard(8.0)
    G.STATE.large = True
    E = G.ext_large()
    g = torch.Generator(device=dev).manual_seed(0)
    for n, k, Ms in [(12576, 4096, (2202, 4608, 13824)), (12288, 4096, (2202, 13824)),
                     (4096, 20480, (2202, 3072, 4096, 6144, 13824))]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        npad = layer.weight.shape[1] // 4
        alpha = G.large_alpha(layer.weight_scale, n)
        for M in Ms:
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            tm = t_us(lambda: marlin(layer, x, n, k), iters=7)
            line = f"N={n} K={k} M={M}: marlin {tm/1e3:.2f} ms |"
            for nc, mc in [(n, 2048), (n, 4096), (8192, 4096), (6400, 4096), (4608, 4096), (4608, 8192), (2048, 8192),
                           (1664, 16384)]:
                f, mib = make(E, layer, alpha, n, k, M, nc, mc)
                t = t_us(lambda: f(x), iters=7)
                line += f" nc{nc}/mc{mc}[{mib:.0f}MiB]:{t/1e3:.2f}({tm/t:.2f}x)"
            print(line, flush=True)
            del x
            torch.cuda.empty_cache()
        del layer, fp8
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
