"""Crossover M of the large-M path vs Marlin per shape (nodeC), through production's apply, A/B interleaved.
The path is forced on for every M here (LARGE_TABLE thresholds set to 1) to find where it starts to win."""
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


def main():
    gpu_guard(8.0)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    cls = prod.Glm53DenseFp8Method
    orig_apply = cls.apply
    F.install(prod)
    for key in F.LARGE_TABLE:
        F.LARGE_TABLE[key] = 1
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    g = torch.Generator(device=dev).manual_seed(1)
    rounds = int(os.environ.get("BENCH_ROUNDS", "9"))
    for name, n, k, grp, pre, Ms in [
            ("kda.in_proj", 12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a", (256, 384, 448, 512, 640, 768, 1024)),
            ("dense.gate_up", 12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj", (256, 384, 512, 640, 768, 1024)),
            ("draft.fc", 4096, 20480, "draft", "model.fc", (2202, 3072, 4096, 4608, 6144, 8192))]:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        m.process_weights_after_loading(lay)
        del w
        line = f"{name} backend {F.STATE.large_gemm} plan(M=13824) {F.large_plan(n, k, 13824)}:"
        for M in Ms:
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            fns = {"marlin": lambda: orig_apply(m, lay, x), "large": lambda: m.apply(lay, x)}
            for f in fns.values():
                f()
            ts = {a: [] for a in fns}
            for r in range(rounds):
                for a in (("marlin", "large") if r % 2 == 0 else ("large", "marlin")):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for _ in range(4):
                        fns[a]()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[a].append(e0.elapsed_time(e1) / 4)
            ratios = sorted(a / b for a, b in zip(ts["marlin"], ts["large"]))
            med = {a: statistics.median(v) for a, v in ts.items()}
            line += (f" M{M}: {med['marlin']:.2f}/{med['large']:.2f} ms x{med['marlin'] / med['large']:.2f} "
                     f"[{ratios[0]:.2f}..{ratios[-1]:.2f}];")
            del x
        print(line, flush=True)
        del lay, m
        torch.cuda.empty_cache()


run_main(main)
