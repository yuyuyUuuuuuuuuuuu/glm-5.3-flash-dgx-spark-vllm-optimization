"""opt-dense: can the per-call Marlin->std repack of the NEXT served layer hide under the current layer's W8A8 GEMM?
Per production subset shape at M (default 13824): component walls (repack / per-token quant / custom GEMM / whole
apply) and the wall of 'GEMM(A) on the main stream || repack(B) on a side stream' (launched after / before the GEMM,
with and without a low-priority side stream) vs the serial sum. Usage: source tests/w8a82/env.sh;
tests/gpu_run.sh python3 tests/optdense/bench_overlap.py [M ...]"""
import os, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard, load_prod  # noqa
import torch
dev = "cuda"


def tmed(fn, n=9):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M", "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    os.environ["GLM53_DENSE_FP8"] = "dense,kda,mla,shared"
    os.environ["GLM53_DENSE_W8A8"] = "1"
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    cls = prod.Glm53DenseFp8Method
    F.install(prod)
    W.install(prod)
    g = torch.Generator(device=dev).manual_seed(1)
    Ms = [int(v) for v in sys.argv[1:]] or [13824, 4289]
    shapes = (("kda.in_proj", 12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a"),
              ("mla.o_proj", 4096, 8192, "mla", "model.layers.3.self_attn.o_proj"),
              ("kda.o_proj", 4096, 4096, "kda", "model.layers.0.self_attn.o_proj"),
              ("dense.gate_up", 12288, 4096, "dense", "model.layers.0.mlp.gate_up_proj"))
    lo, hi = torch.cuda.Stream.priority_range()
    side_lo = torch.cuda.Stream(priority=lo)       # lo = numerically larger = lower priority
    side = torch.cuda.Stream()
    for name, n, k, grp, pre in shapes:
        lays = []
        for _ in range(2):
            w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
            lay = L(w)
            m = cls(grp, pre)
            m.process_weights_after_loading(lay)
            lays.append((m, lay))
        (mA, A), (mB, B) = lays
        for M in Ms:
            x = (torch.randn(M, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            cls.apply(mA, A, x); cls.apply(mB, B, x)            # self-tests
            keyA = W._key(A.weight, A.weight_scale, k)
            alpha = W.ALPHA[keyA]
            wA = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=dev)
            wB = torch.empty_like(wA)
            W.repack(wA, A.weight, n, k)
            q, sa = W.quant_per_token(x)
            out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
            t_apply = tmed(lambda: cls.apply(mA, A, x))
            t_rep = tmed(lambda: W.repack(wB, B.weight, n, k))
            t_q = tmed(lambda: W.quant_per_token(x))
            t_g = tmed(lambda: W.custom_gemm(out, q, wA, sa, alpha, n, k, M))

            def ovl(stream, before):
                cur = torch.cuda.current_stream()
                ev = torch.cuda.Event()
                ev.record(cur)
                if before:
                    stream.wait_event(ev)
                    with torch.cuda.stream(stream):
                        W.repack(wB, B.weight, n, k)
                    W.custom_gemm(out, q, wA, sa, alpha, n, k, M)
                else:
                    W.custom_gemm(out, q, wA, sa, alpha, n, k, M)
                    stream.wait_event(ev)
                    with torch.cuda.stream(stream):
                        W.repack(wB, B.weight, n, k)
                done = torch.cuda.Event()
                done.record(stream)
                cur.wait_event(done)

            def qovl(stream):                        # repack || the NEXT layer's quant (memory-bound vs memory-bound)
                cur = torch.cuda.current_stream()
                ev = torch.cuda.Event(); ev.record(cur)
                stream.wait_event(ev)
                with torch.cuda.stream(stream):
                    W.repack(wB, B.weight, n, k)
                W.quant_per_token(x)
                done = torch.cuda.Event(); done.record(stream); cur.wait_event(done)

            r = {f"after/{nm}": tmed(lambda: ovl(s, False)) for nm, s in (("side", side), ("side_lo", side_lo))}
            r.update({f"before/{nm}": tmed(lambda: ovl(s, True)) for nm, s in (("side", side), ("side_lo", side_lo))})
            r["quant||repack"] = tmed(lambda: qovl(side))
            print(f"{name} [{n}x{k}] M={M}: apply {t_apply:.3f} ms = repack {t_rep:.3f} + quant {t_q:.3f} + "
                  f"GEMM {t_g:.3f} (+{t_apply - t_rep - t_q - t_g:+.3f} rest); serial GEMM+repack {t_g + t_rep:.3f}; "
                  + "; ".join(f"{a} {b:.3f} (hidden {t_g + t_rep - b:+.3f})" for a, b in r.items() if a != "quant||repack")
                  + f"; quant||repack {r['quant||repack']:.3f} vs serial {t_q + t_rep:.3f}", flush=True)
            del x, q, sa, out, wA, wB
        del lays, A, B
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_main(main)
