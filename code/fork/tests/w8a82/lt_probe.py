import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard  # noqa
import torch
from torch.utils.cpp_extension import load
dev = "cuda"
SH = {"kda.in_proj": (12576, 4096), "kda.o_proj": (4096, 4096), "mla.qkv_a": (2048, 4096), "mla.q_b": (8192, 1536),
      "mla.o_proj": (4096, 8192), "shared.gate_up": (2048, 4096), "shared.down": (4096, 1024),
      "dense.gate_up": (12288, 4096), "dense.down": (4096, 6144)}


def timed(fn, reps=5, rounds=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(reps):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / reps)
    return statistics.median(ts)


def main():
    gpu_guard(8.0)
    import os
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    sys.path.insert(0, "/w")
    import fp8_w8a8 as W
    inc = W._include_shim()
    E = load(name="lt_probe", sources=[str(Path(__file__).with_name("lt_probe.cu"))], extra_cuda_cflags=["-O3", *inc],
             extra_cflags=["-O3", *inc],
             extra_ldflags=["-L/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib", "-l:libcublasLt.so.13"],
             verbose=False)
    import vllm._custom_ops as ops
    g = torch.Generator(device=dev).manual_seed(0)
    ws = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    for name, (n, k) in SH.items():
        w = torch.randn(n, k, device=dev, generator=g).to(torch.float8_e4m3fn)
        sb = (torch.rand(n, device=dev, generator=g) * 1e-3 + 1e-3).float()
        for M in (13824, 4289, 1791, 512):
            x = (torch.randn(M, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            q, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            sa = sa.float().reshape(-1).contiguous()
            ref = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
            torch.ops._C.cutlass_scaled_mm(ref, q, w.t(), sa.view(-1, 1), sb.view(-1, 1), None)
            out = torch.empty_like(ref)
            try:
                nres = E.lt_mm(out, q, w, sa, sb, ws, -1, 1)
            except Exception as e:  # noqa
                print(name, M, "OUTER_VEC failed:", repr(e)[:200], flush=True)
                return
            d = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            best = (1e9, -1)
            for i in range(nres):
                try:
                    t = timed(lambda: E.lt_mm(out, q, w, sa, sb, ws, i, 1), reps=3, rounds=3)
                except Exception:
                    continue
                best = min(best, (t, i))
            t0 = timed(lambda: E.lt_mm(out, q, w, sa, sb, ws, -1, 1))
            pr = 2048 if n * k > 16 * 2 ** 20 else M
            def cut():
                for r0 in range(0, M, pr):
                    r = min(pr, M - r0)
                    torch.ops._C.cutlass_scaled_mm(ref[r0:r0 + r], q[r0:r0 + r], w.t(), sa[r0:r0 + r].view(-1, 1),
                                                   sb.view(-1, 1), None)
            tc = timed(cut)
            print(f"{name:15s} M{M:6d}: cutlass(shipped pieces) {tc:.3f} | lt heur0 {t0:.3f} | lt best {best[0]:.3f}"
                  f" (algo {best[1]}/{nres}) | rel diff vs cutlass {d:.1e}", flush=True)


if __name__ == "__main__":
    run_main(main)
