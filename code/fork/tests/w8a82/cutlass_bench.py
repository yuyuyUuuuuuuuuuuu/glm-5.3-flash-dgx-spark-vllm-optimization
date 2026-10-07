"""Tune the custom CUTLASS SM120 W8A8 GEMM variants vs the image's cutlass_scaled_mm (single + shipped pieces).
Checks bit-equality vs cutlass_scaled_mm (same epilogue arithmetic) and times every (cfg, swizzle, raster)."""
import os, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/w")
from harness import run_main, gpu_guard  # noqa
import torch
dev = "cuda"
SH = {"kda.in_proj": (12576, 4096, 34), "kda.o_proj": (4096, 4096, 34), "mla.qkv_a": (2048, 4096, 11),
      "mla.q_b": (8192, 1536, 11), "mla.o_proj": (4096, 8192, 11), "shared.gate_up": (2048, 4096, 42),
      "shared.down": (4096, 1024, 42), "dense.gate_up": (12288, 4096, 3), "dense.down": (4096, 6144, 3)}
CFGS = {0: "C128x128x128", 1: "P128x128x128", 2: "P64x128x128", 3: "C128x256x64", 4: "C256x128x64", 5: "P64x256x128"}


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
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    import fp8_w8a8 as W
    from torch.utils.cpp_extension import load
    C = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
    inc = W._include_shim() + [f"-I{C}/include", f"-I{C}/tools/util/include"]
    E = load(name="cutlass_probe", sources=["/w/tests/w8a82/cutlass_probe.cu"],
             extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG", *inc],
             extra_cflags=["-O3", *inc], verbose=False)
    import vllm._custom_ops as ops
    Ms = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["13824", "4289", "1791", "512"])]
    cfgs = [int(a) for a in (sys.argv[2].split(",") if len(sys.argv) > 2 else list(map(str, CFGS)))]
    swz = [1, 2, 4, 8]
    g = torch.Generator(device=dev).manual_seed(0)
    ws = torch.empty(16 << 20, dtype=torch.uint8, device=dev)
    tot = {}
    for name, (n, k, calls) in SH.items():
        w = torch.randn(n, k, device=dev, generator=g).to(torch.float8_e4m3fn)
        sb = (torch.rand(n, device=dev, generator=g) * 1e-3 + 1e-3).float()
        for M in Ms:
            x = (torch.randn(M, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            q, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            sa = sa.float().reshape(-1).contiguous()
            ref = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
            torch.ops._C.cutlass_scaled_mm(ref, q, w.t(), sa.view(-1, 1), sb.view(-1, 1), None)
            pr = 2048 if n * k > 16 * 2 ** 20 else M

            def shipped():
                for r0 in range(0, M, pr):
                    r = min(pr, M - r0)
                    torch.ops._C.cutlass_scaled_mm(ref[r0:r0 + r], q[r0:r0 + r], w.t(), sa[r0:r0 + r].view(-1, 1),
                                                   sb.view(-1, 1), None)
            t_ship = timed(shipped)
            out = torch.empty_like(ref)
            best = (1e9, None)
            line = []
            for c in cfgs:
                cb = (1e9, None)
                for s in swz:
                    for ro in (0, 1, 2):
                        try:
                            E.gemm(c, out, q, w, sa, sb, s, ro, ws)
                        except Exception as e:  # noqa
                            cb = (cb[0], f"ERR {str(e)[:60]}")
                            break
                        t = timed(lambda: E.gemm(c, out, q, w, sa, sb, s, ro, ws), reps=3, rounds=3)
                        if t < cb[0]:
                            cb = (t, (s, ro))
                if cb[1] is not None and not isinstance(cb[1], str):
                    E.gemm(c, out, q, w, sa, sb, cb[1][0], cb[1][1], ws)
                    eq = torch.equal(out, ref)
                    d = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                    line.append(f"{CFGS[c]}={cb[0]:.3f}(s{cb[1][0]}r{cb[1][1]}{',EQ' if eq else f',d{d:.0e}'})")
                    if cb[0] < best[0]:
                        best = (cb[0], (c, cb[1]))
                else:
                    line.append(f"{CFGS[c]}={cb[1]}")
            fl = 2.0 * M * n * k
            print(f"{name:15s} M{M:6d}: shipped {t_ship:.3f} ({fl / t_ship / 1e9:.0f} TF) | best {best[0]:.3f} "
                  f"({fl / best[0] / 1e9:.0f} TF) {best[1]} | " + " ".join(line), flush=True)
            tot.setdefault(M, [0.0, 0.0])
            tot[M][0] += calls * t_ship
            tot[M][1] += calls * best[0]
        del w
        torch.cuda.empty_cache()
    for M, (a, b) in sorted(tot.items()):
        print(f"chunk M={M}: shipped GEMMs {a:.1f} ms -> best custom {b:.1f} ms ({b - a:+.1f})")


if __name__ == "__main__":
    run_main(main)
