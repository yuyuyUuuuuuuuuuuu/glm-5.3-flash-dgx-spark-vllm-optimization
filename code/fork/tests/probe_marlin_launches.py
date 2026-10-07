"""Probe: per-launch Marlin kernel times on nodeC for the in_proj / gate_up / fc shapes at production chunk sizes, to
compare with production's R12 trace (docs/FP8_LARGE_M.md §2): M = 640, 1024, 1536, 2202, 13824."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import gpu_guard  # noqa: E402
import torch  # noqa: E402
from fp8_bench_common import quantize_like_prod, marlin  # noqa: E402

dev = "cuda"


def main():
    gpu_guard(8.0)
    g = torch.Generator(device=dev).manual_seed(0)
    for n, k in [(12576, 4096), (12288, 4096), (4096, 20480)]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, _, _ = quantize_like_prod(w)
        del w
        for M in (640, 1024, 1536, 2202, 13824):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            for _ in range(3):
                marlin(layer, x, n, k)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    marlin(layer, x, n, k)
                torch.cuda.synchronize()
            ks = [(e.name, e.device_time_total / 1e3) for e in prof.events()
                  if e.device_type == torch.autograd.DeviceType.CUDA]
            mk = [t for nm, t in ks if "Marlin" in nm]
            other = [(nm[:40], round(t, 3)) for nm, t in ks if "Marlin" not in nm][:2]
            per = len(mk) // 3
            print(f"N={n} K={k} M={M}: {per} Marlin launches per call, ms: " +
                  " | ".join(" ".join(f"{t:.2f}" for t in mk[i * per:(i + 1) * per]) for i in range(3)) +
                  f"; other kernels per call: {other}", flush=True)
            del x
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
