"""Probe 5: component times (dequant / fp32-out GEMM / scale_cast) and a 2-stream pipeline over M chunks."""
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


def main():
    gpu_guard(8.0)
    G.STATE.large = True
    E = G.ext_large()
    g = torch.Generator(device=dev).manual_seed(0)
    side = torch.cuda.Stream()
    for n, k in [(12576, 4096), (12288, 4096), (4096, 20480), (4096, 8192)]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        npad = layer.weight.shape[1] // 4
        alpha = G.large_alpha(layer.weight_scale, n)
        wd = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
        for M in (2202, 13824):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
            y32 = torch.empty(M, n, dtype=torch.float32, device=dev)
            td = t_us(lambda: E.dequant(wd, layer.weight, 0, n, k))
            tg = t_us(lambda: torch.mm(x, wd.t(), out_dtype=torch.float32, out=y32))
            tb = t_us(lambda: torch.mm(x, wd.t()))
            ts = t_us(lambda: E.scale_cast(out, y32, alpha, None))
            tm = t_us(lambda: marlin(layer, x, n, k))

            def seq(mc):
                def f():
                    E.dequant(wd, layer.weight, 0, n, k)
                    for m0 in range(0, M, mc):
                        mm = min(mc, M - m0)
                        t = torch.mm(x[m0:m0 + mm], wd.t(), out_dtype=torch.float32)
                        E.scale_cast(out[m0:m0 + mm], t, alpha, None)
                return f

            def pipe(mc):
                def f():
                    E.dequant(wd, layer.weight, 0, n, k)
                    main = torch.cuda.current_stream()
                    side.wait_stream(main)
                    for m0 in range(0, M, mc):
                        mm = min(mc, M - m0)
                        t = torch.mm(x[m0:m0 + mm], wd.t(), out_dtype=torch.float32)
                        ev = torch.cuda.Event()
                        ev.record(main)
                        with torch.cuda.stream(side):
                            side.wait_event(ev)
                            E.scale_cast(out[m0:m0 + mm], t, alpha, None)
                            t.record_stream(side)
                    main.wait_stream(side)
                return f
            fl = 2.0 * M * n * k
            line = (f"N={n} K={k} M={M}: marlin {tm/1e3:.2f} ms ({fl/tm/1e6:.0f} TF) | dequant {td/1e3:.3f} | gemm32 {tg/1e3:.2f} "
                    f"({fl/tg/1e6:.0f} TF) | gemm16 {tb/1e3:.2f} ({fl/tb/1e6:.0f} TF) | scale_cast {ts/1e3:.3f} "
                    f"({M*n*6/ts/1e3:.0f} GB/s) |")
            for mc in (1024, 2048, 3456, 4608, M):
                if mc > M:
                    continue
                a = t_us(seq(mc), iters=5, warm=1)
                b = t_us(pipe(mc), iters=5, warm=1)
                line += f" mc{mc}: seq {a/1e3:.2f} ({tm/a:.2f}x) pipe {b/1e3:.2f} ({tm/b:.2f}x)"
            print(line, flush=True)
            del x, out, y32
            torch.cuda.empty_cache()
        del layer, fp8, wd
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
