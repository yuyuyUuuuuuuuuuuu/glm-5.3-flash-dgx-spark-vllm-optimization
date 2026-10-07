"""w8a82 probe: where do the W8A8 GEMM ms go, and do other fp8 GEMM backends / piece sizes beat the shipped choice?
Synthetic fp8 operands (timing only). Per shape x M: cutlass_scaled_mm single call, pieces of 1024/2048/4096 rows,
torch._scaled_mm rowwise (cuBLASLt) if supported, per-token quant alone, repack-equivalent copy alone."""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard  # noqa
import torch

dev = "cuda"
SHAPES = {  # name: (N, K, calls/chunk)
    "kda.in_proj": (12576, 4096, 34), "kda.o_proj": (4096, 4096, 34), "mla.qkv_a": (2048, 4096, 11),
    "mla.q_b": (8192, 1536, 11), "mla.o_proj": (4096, 8192, 11), "shared.gate_up": (2048, 4096, 42),
    "shared.down": (4096, 1024, 42), "dense.gate_up": (12288, 4096, 3), "dense.down": (4096, 6144, 3)}


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
    import vllm._custom_ops as ops
    Ms = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["13824", "4289", "1791", "512"])]
    g = torch.Generator(device=dev).manual_seed(0)
    rowwise_ok = True
    tot = {}
    for name, (n, k, calls) in SHAPES.items():
        w = (torch.randn(n, k, device=dev, generator=g)).to(torch.float8_e4m3fn)
        wt = w.t()
        sb = torch.rand(n, 1, device=dev, generator=g) * 1e-3 + 1e-3
        for M in Ms:
            x = (torch.randn(M, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            q, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            sa = sa.float().reshape(-1, 1).contiguous()
            out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
            res = {}
            res["single"] = timed(lambda: torch.ops._C.cutlass_scaled_mm(out, q, wt, sa, sb, None))
            for pr in (1024, 2048, 4096):
                if pr >= M:
                    continue
                def f(pr=pr):
                    for r0 in range(0, M, pr):
                        r = min(pr, M - r0)
                        torch.ops._C.cutlass_scaled_mm(out[r0:r0 + r], q[r0:r0 + r], wt, sa[r0:r0 + r], sb, None)
                res[f"p{pr}"] = timed(f)
            if rowwise_ok:
                try:
                    sbt = sb.reshape(1, -1).contiguous()
                    o2 = torch._scaled_mm(q, wt, scale_a=sa, scale_b=sbt, out_dtype=torch.bfloat16)
                    ref = out.float()
                    torch.ops._C.cutlass_scaled_mm(out, q, wt, sa, sb, None)
                    d = ((o2.float() - out.float()).norm() / out.float().norm()).item()
                    res["torch_rowwise"] = timed(lambda: torch._scaled_mm(q, wt, scale_a=sa, scale_b=sbt,
                                                                           out_dtype=torch.bfloat16))
                    res["rw_diff"] = d
                except Exception as e:  # noqa
                    print("torch._scaled_mm rowwise unsupported:", repr(e)[:300], flush=True)
                    rowwise_ok = False
            try:
                one = torch.ones((), device=dev)
                res["torch_tensorwise"] = timed(lambda: torch._scaled_mm(q, wt, scale_a=one, scale_b=one,
                                                                          out_dtype=torch.bfloat16))
            except Exception as e:  # noqa
                res["torch_tensorwise"] = float("nan")
            res["quant"] = timed(lambda: ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True))
            tmp = torch.empty_like(w)
            res["copy_w"] = timed(lambda: tmp.copy_(w))
            best_g = min(v for kk, v in res.items() if kk in ("single", "p1024", "p2048", "p4096"))
            fl = 2.0 * M * n * k
            print(f"{name:15s} N{n} K{k} M{M:6d}: " + " ".join(
                f"{kk}={v:.3f}" if kk != "rw_diff" else f"{kk}={v:.1e}" for kk, v in res.items()) +
                f" | best cutlass {best_g:.3f} ms = {fl / best_g / 1e9:.0f} TFLOPS", flush=True)
            for kk, v in res.items():
                if kk != "rw_diff":
                    tot.setdefault((M, kk), 0.0)
                    tot[(M, kk)] += calls * v
            tot.setdefault((M, "best"), 0.0)
            tot[(M, "best")] += calls * best_g
            del x, q, sa, out
        del w, wt
        torch.cuda.empty_cache()
    print("\nchunk totals (calls/chunk weighted; shapes missing a variant count 0 for it):")
    for (M, kk), v in sorted(tot.items()):
        print(f"  M={M:6d} {kk:18s} {v:8.1f} ms")


if __name__ == "__main__":
    run_main(main)
