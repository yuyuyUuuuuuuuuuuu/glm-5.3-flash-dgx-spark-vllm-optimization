"""Kernel-level breakdown of the shipped W8A8 apply per production shape (torch.profiler), real weights where the
partial checkpoint has them. Usage: ... prof_shipped.py [M]"""
import os, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pf3000"))
from harness import run_main, gpu_guard, load_prod  # noqa
import torch
from bench_test_c import SHAPES, load_real  # noqa
dev = "cuda"


def main():
    gpu_guard(8.0)
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 13824
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M", "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    os.environ["GLM53_DENSE_FP8"] = "dense,kda,mla,shared"
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    cls = prod.Glm53DenseFp8Method
    F.install(prod)
    prod_apply = cls.apply
    os.environ["GLM53_DENSE_W8A8"] = "1"
    rep = W.install(prod)
    print("w8a8 install:", rep, flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    from torch.profiler import profile, ProfilerActivity
    tot = {}
    wall_tot = 0.0
    prod_tot = [0.0]
    shapes = dict(SHAPES)
    shapes["kda.f_b_proj"] = (4096, 128, "kda", "model.layers.0.self_attn.f_b_proj", 34, None)
    shapes["kda.g_b_proj"] = (4096, 128, "kda", "model.layers.0.self_attn.g_b_proj", 34, None)
    only = os.environ.get("PROF_ONLY")
    for name, (n, k, grp, pre, calls, src) in shapes.items():
        if only and name not in only.split(","):
            continue
        if grp == "draft":
            continue
        w = load_real(src) if src else (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre)
        m.process_weights_after_loading(lay)
        x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        for _ in range(3):
            cls.apply(m, lay, x)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(5):
            e0.record()
            for _ in range(3):
                cls.apply(m, lay, x)
            e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) / 3)
        wall = statistics.median(ts)
        tp = []
        for _ in range(5):
            e0.record()
            for _ in range(3):
                prod_apply(m, lay, x)
            e1.record(); torch.cuda.synchronize(); tp.append(e0.elapsed_time(e1) / 3)
        pwall = statistics.median(tp)
        prod_tot[0] += calls * pwall
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(3):
                cls.apply(m, lay, x)
            torch.cuda.synchronize()
        agg = {}
        for ev in p.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                nm = ev.name
                key = ("repack" if "marlin_to_std" in nm else "quant" if "quant" in nm.lower() else
                       "gemm" if ("cutlass" in nm.lower() or "gemm" in nm.lower() or "sm120" in nm or "Kernel" in nm)
                       else nm[:50])
                agg[key] = agg.get(key, 0.0) + ev.device_time / 1000.0 / 3
        s = sum(agg.values())
        print(f"{name:15s} M{M} production {pwall:.3f} | w8a8 wall {wall:.3f} ms | kernels {s:.3f}: " +
              ", ".join(f"{k_}={v:.3f}" for k_, v in sorted(agg.items(), key=lambda t: -t[1])), flush=True)
        for k_, v in agg.items():
            tot[k_] = tot.get(k_, 0.0) + calls * v
        wall_tot += calls * wall
        del lay, m, x, w
        torch.cuda.empty_cache()
    print(f"\nchunk (M={M}) production {prod_tot[0]:.1f} ms; w8a8 wall {wall_tot:.1f} ms "
          f"({wall_tot - prod_tot[0]:+.1f}); kernels: " + ", ".join(f"{k_}={v:.1f}" for k_, v in
                                                                         sorted(tot.items(), key=lambda t: -t[1])))
    print("counters:", W.summary())


if __name__ == "__main__":
    run_main(main)
