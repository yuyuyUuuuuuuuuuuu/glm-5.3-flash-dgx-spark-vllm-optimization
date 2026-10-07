"""Probe 1: Marlin FP8 vs plain cuBLAS BF16 GEMM (bf16 out / fp32 out) at prefill M on production shapes (nodeC). The
BF16 weight here is fp8.to(bf16) held in memory (the ceiling the large-M path chases; it keeps no such copy)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import gpu_guard  # noqa: E402
import torch  # noqa: E402
from fp8_bench_common import quantize_like_prod, marlin  # noqa: E402

dev = "cuda"


def t_us(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        res.append(a.elapsed_time(b) * 1000)
    res.sort()
    return res[len(res) // 2]


def main():
    gpu_guard(8.0)
    g = torch.Generator(device=dev).manual_seed(0)
    print("torch", torch.__version__, torch.cuda.get_device_name())
    shapes = [(12576, 4096), (12288, 4096), (8192, 1536), (4096, 4096), (4096, 8192), (2048, 4096)]
    for n, k in shapes:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        wb = fp8.to(torch.bfloat16)  # exact e4m3 -> bf16
        del w
        for M in (1791, 2202, 4608, 13824):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            fl = 2.0 * M * n * k
            tm = t_us(lambda: marlin(layer, x, n, k))
            tb = t_us(lambda: torch.mm(x, wb.t()))
            try:
                tf = t_us(lambda: torch.mm(x, wb.t(), out_dtype=torch.float32))
            except Exception as e:  # noqa: BLE001
                tf = float('nan'); print("  out_dtype fp32 failed:", repr(e)[:200])
            print(f"N={n:6d} K={k:5d} M={M:6d}: marlin {tm/1e3:8.3f} ms {fl/tm/1e6:6.1f} TF | cublas bf16 {tb/1e3:8.3f} ms "
                  f"{fl/tb/1e6:6.1f} TF | cublas fp32out {tf/1e3:8.3f} ms {fl/tf/1e6:6.1f} TF", flush=True)
            del x
        del layer, fp8, wb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
