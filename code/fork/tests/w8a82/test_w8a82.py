"""w8a82 checks (nodeC, production image): exit 1 on any failure.
R  repack v2 (smem-staged) == v1 byte for byte on every production shape + an adversarial N; v1/v2/copy timings.
C  custom CUTLASS GEMM == torch.ops._C.cutlass_scaled_mm bitwise for every production shape, every M bucket and every
   cfg (0..3) at the table's swizzle/raster, incl. a non-multiple-of-tile M; and w8a8_forward(custom) == w8a8_forward
   (cutlass_mm pieces) bitwise (pieces are row splits with per-token scales: identical rows).
K  knob: GLM53_DENSE_W8A8_GEMM=cutlass_mm -> STATE.custom False (w8a8's path), a bad value refuses install.
Run: GPU_RUN_ENV="TF_EXL3_JIT=1" flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/w8a82/test_w8a82.py
"""
import os, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from harness import Checks, gpu_guard, run_main  # noqa
import torch  # noqa
import fp8_w8a8 as W  # noqa
from fp8_bench_common import UNIQUE_SHAPES, quantize_like_prod  # noqa
dev = "cuda"


def timed(fn, reps=10, rounds=5):
    for _ in range(3):
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
    ck = Checks()

    def chk(msg, cond):
        print(("  ok   " if cond else "  FAIL ") + msg, flush=True) if not str(msg).startswith("C ") or not cond else None
        return ck(cond, msg)
    E = W.ext()
    print(f"extension VERSION {getattr(E, 'VERSION', None)}, has gemm {hasattr(E, 'fp8_w8a8_gemm')}", flush=True)
    chk("ext VERSION 2 + gemm", getattr(E, "VERSION", None) == 2 and hasattr(E, "fp8_w8a8_gemm"))
    g = torch.Generator(device=dev).manual_seed(1)
    import vllm._custom_ops as ops
    W.STATE.custom = True
    shapes = list(UNIQUE_SHAPES) + [(4096 - 40, 1024)]
    t1 = t2 = tc = 0.0
    for n, k in shapes:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, fp8, scale = quantize_like_prod(w)
        del w
        a = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=dev)
        b = torch.empty_like(a)
        E.fp8_marlin_to_std(a, layer.weight, n, k, 1)
        E.fp8_marlin_to_std(b, layer.weight, n, k, 2)
        chk(f"R {n}x{k} v2 == v1 == production fp8", torch.equal(a.view(torch.uint8), b.view(torch.uint8)) and
                 torch.equal(b.view(torch.uint8), fp8.view(torch.uint8)))
        x1 = timed(lambda: E.fp8_marlin_to_std(a, layer.weight, n, k, 1))
        x2 = timed(lambda: E.fp8_marlin_to_std(b, layer.weight, n, k, 2))
        xc = timed(lambda: b.copy_(a))
        t1, t2, tc = t1 + x1, t2 + x2, tc + xc
        print(f"   repack {n}x{k}: v1 {x1:.3f} ms, v2 {x2:.3f} ms, plain copy {xc:.3f} ms", flush=True)
        if n % 16 or n > 20000:          # the lm_head shape is never a W8A8 layer
            del layer, fp8, scale, a, b
            torch.cuda.empty_cache()
            continue
        alpha = W.large_alpha(layer.weight_scale, n)
        ws = torch.empty(W.WS_BYTES, dtype=torch.uint8, device=dev)
        for m in (512, 1791, 3071, 3072, 4289, 13824, 13856):
            x = (torch.randn(m, k, device=dev, generator=g) * 0.3).to(torch.bfloat16)
            x[::97, ::31] *= 40.0          # some outlier channels/tokens
            q, sa = W.quant_per_token(x)
            ref = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
            torch.ops._C.cutlass_scaled_mm(ref, q, b.t(), sa, alpha[:n].view(n, 1), None)
            for cfg in range(4):
                _, sw, ro = W.gemm_choice(n, k, m)
                out = torch.empty_like(ref)
                E.fp8_w8a8_gemm(out, q, b, sa.view(-1), alpha, cfg, sw, ro, ws)
                chk(f"C {n}x{k} M{m} cfg{cfg} == cutlass_scaled_mm", torch.equal(out, ref))
            W.GEMMCHECK[(n, k)] = True
        # forward custom vs pieces (bitwise)
        key = W._key(layer.weight, layer.weight_scale, k)
        W.ALPHA[key] = alpha
        x = (torch.randn(13824, k, device=dev, generator=g) * 0.3).to(torch.bfloat16)
        W.GEMMCHECK[(n, k)] = True
        yc = W.w8a8_forward(x, layer.weight, layer.weight_scale, n, k, layer_key=key)
        W.GEMMCHECK[(n, k)] = False
        yp = W.w8a8_forward(x, layer.weight, layer.weight_scale, n, k, layer_key=key)
        chk(f"C {n}x{k} forward custom == cutlass_mm pieces (bitwise)", torch.equal(yc, yp))
        del layer, fp8, scale, a, b, x, yc, yp
        torch.cuda.empty_cache()
    print(f"repack totals over the unique shapes: v1 {t1:.3f} v2 {t2:.3f} copy {tc:.3f} ms", flush=True)
    # K: the knob
    import importlib
    os.environ["GLM53_DENSE_W8A8_GEMM"] = "cutlass_mm"
    chk("K cutlass_mm accepted", W._problem() is None and W.CFG.gemm == "cutlass_mm" and not W.STATE.custom)
    os.environ["GLM53_DENSE_W8A8_GEMM"] = "bogus"
    chk("K bogus refused", (W._problem() or "").startswith("GLM53_DENSE_W8A8_GEMM"))
    os.environ.pop("GLM53_DENSE_W8A8_GEMM")
    chk("K default custom", W._problem() is None and W.STATE.custom)
    ck.summary()


if __name__ == "__main__":
    run_main(main)
