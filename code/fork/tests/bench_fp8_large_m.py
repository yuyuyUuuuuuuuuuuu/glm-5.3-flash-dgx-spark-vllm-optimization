"""GLM53_FP8_LARGE_M speed on nodeC (GB10) at production shapes, through production's Glm53DenseFp8Method.apply:
the wrapped apply (large-M path; TileLang backend = default, and the cuBLAS backend) vs production's own apply
(Marlin), interleaved rounds (order rotated per round), median ms per call.

Shapes (per rank, TP=2): KDA in_proj 12576x4096 (34 layers), dense gate_up 12288x4096 (3 layers), drafter fc
4096x20480 (1, GLM53_DRAFT_FP8=layers,fc). M: the thresholds, 1791 / 2202 (remainder chunks of 15.6k / 16k prompts),
4608, 13824 (the full prefill chunk). Also: the kernels each path launches at M = 13824 (profiler), and the per-prompt
sum for a 16,026-token prompt (13824 + 2202) = 34 in_proj + 3 gate_up + 1 fc calls per chunk.
"""
import inspect
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import run_main, gpu_guard, load_prod  # noqa: E402
import torch  # noqa: E402
from test_fp8_integrate import single_rank_tp, L  # noqa: E402

dev = "cuda"
SHAPES = {"kda.in_proj": (12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a", 34),
          "dense.gate_up": (12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj", 3),
          "draft.fc": (4096, 20480, "draft", "model.fc", 1)}


def main():
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    cls = prod.Glm53DenseFp8Method
    orig_apply = cls.apply
    rep = F.install(prod)
    assert rep["installed"] and F.STATE.large, rep
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    g = torch.Generator(device=dev).manual_seed(0)
    per_prompt = {"marlin": 0.0, "large": 0.0, "cublas": 0.0}
    per_prompt24 = {"marlin": 0.0, "large": 0.0, "cublas": 0.0}
    per_prompt15 = {"marlin": 0.0, "large": 0.0, "cublas": 0.0}
    rounds = int(os.environ.get("BENCH_ROUNDS", "5"))
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, rounds {rounds} (A/B interleaved), "
          f"transient budget {F.CFG.large_temp_mib} MiB")
    for name, (n, k, grp, pre, calls) in SHAPES.items():
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        m.process_weights_after_loading(lay)
        del w
        min_m = F.LARGE_TABLE[(lay.weight.shape[1] // 4, k)]
        for M in sorted({min_m, 1791, 2202, 4608, 8192, 13824}):
            F.STATE.large_gemm = "tilelang"
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            def with_backend(be):
                def f():
                    F.STATE.large_gemm = be
                    return m.apply(lay, x)
                return f
            fns = {"marlin": lambda: orig_apply(m, lay, x), "large": with_backend("tilelang"),
                   "cublas": with_backend("cublas")}
            for f in fns.values():
                f()
            torch.cuda.synchronize()
            ts = {a: [] for a in fns}
            reps = 3 if M >= 4608 else 8
            for r in range(rounds):
                order = ("marlin", "large", "cublas")
                for a in order[r % 3:] + order[:r % 3]:
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for _ in range(reps):
                        fns[a]()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[a].append(e0.elapsed_time(e1) / reps)
            med = {a: statistics.median(v) for a, v in ts.items()}
            spread = {a: (max(v) - min(v)) / statistics.median(v) * 100 for a, v in ts.items()}
            fl = 2.0 * M * n * k
            served = M >= min_m
            print(f"{name:13s} [{n}x{k}] M={M:5d}: Marlin {med['marlin']:8.3f} ms ({fl / med['marlin'] / 1e9:5.1f} TFLOPS, "
                  f"spread {spread['marlin']:.1f}%) | wrapped {med['large']:8.3f} ms ({fl / med['large'] / 1e9:5.1f} TFLOPS, "
                  f"spread {spread['large']:.1f}%) {'large path' if served else 'Marlin (below threshold)'} | "
                  f"x{med['marlin'] / med['large']:.2f} | cublas backend {med['cublas']:8.3f} ms "
                  f"x{med['marlin'] / med['cublas']:.2f} (spread {spread['cublas']:.1f}%)", flush=True)
            if M in (13824, 2202):
                for a in per_prompt:
                    per_prompt[a] += calls * med[a]
            if M in (13824, 1791):
                for a in per_prompt15:
                    per_prompt15[a] += calls * med[a]
            if M == 8192:
                for a in per_prompt24:
                    per_prompt24[a] += calls * med[a]
            if M == 13824:
                for a in per_prompt24:
                    per_prompt24[a] += 2 * calls * med[a]
            if M == 13824:
                F.STATE.large_gemm = "tilelang"
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    m.apply(lay, x)
                    torch.cuda.synchronize()
                ks = {}
                for e in prof.events():
                    if e.device_type == torch.autograd.DeviceType.CUDA:
                        nm = e.name[:90]
                        c, t = ks.get(nm, (0, 0.0))
                        ks[nm] = (c + 1, t + e.device_time_total / 1e3)
                print("     large-path kernels at M=13824 (tilelang backend): " +
                      "; ".join(f"{c}x {nm} {t:.2f} ms" for nm, (c, t) in sorted(ks.items(), key=lambda kv: -kv[1][1])))
            del x
        del lay, m
        torch.cuda.empty_cache()
    for label, pp in (("15,615-token prompt (13824 + 1791)", per_prompt15),
                      ("16,026-token prompt (13824 + 2202)", per_prompt), ("35,840-token prompt (2 x 13824 + 8192)",
                                                                           per_prompt24)):
        print(f"per {label}, 34 in_proj + 3 gate_up + 1 fc per chunk: Marlin {pp['marlin']:.0f} ms -> large-M path "
              f"{pp['large']:.0f} ms (saves {pp['marlin'] - pp['large']:.0f} ms per rank; cuBLAS backend "
              f"{pp['cublas']:.0f} ms)")
    print("counters:", F.summary())


run_main(main)
