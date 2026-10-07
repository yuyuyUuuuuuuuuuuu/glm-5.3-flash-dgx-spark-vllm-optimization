"""Timing of the small-M dense GEMMs of decode: production's op (cuBLAS through torch) vs kernels/gemv_bf16.cu,
at the production shapes and M values, cold weights, inside CUDA graphs (docs/BF16_GEMV.md).

Method: nw copies of the weight (together >= 96 MB, 4x the 24 MB L2) and one hot activation tensor; one CUDA graph
holds `calls` consecutive calls cycling through the copies (as production's graph cycles through layers); the graph
is replayed `reps` times, each replay timed with CUDA events; us/call = median replay / calls. GB/s = weight bytes /
us/call. Baseline = exactly production's ops:
  router          F.linear(x, W).to(float32)     (GateLinear tier 6 on SM 12.x: no specialized router kernel)
  idx_head_gate   torch.mm(x.float(), W32)       (Indexer: W32 = wk_weights_proj.weight[128:].t().contiguous().float())
  others          F.linear(x, W)                 (UnquantizedLinearMethod / F.linear, bf16 out)
--sweep times every launch plan and prints the best per (shape, M bucket).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import glm53_bf16_gemv as G  # noqa: E402

# name: (N, K, kind, calls per decode step in the R7 profile (rank 0, prose, 40 steps), where it runs)
SHAPES = {
    "router": (288, 4096, "router", 84, "graph, 2 per MoE layer (42 layers)"),
    "idx_wk_wp": (160, 4096, "bf16", 11, "graph, MLA indexer wk_weights_proj"),
    "idx_head_gate": (32, 4096, "f32", 11, "graph, MLA indexer fp32 torch.mm"),
    "idx_kpool_gate": (128, 4096, "bf16", 11, "graph, MLA indexer kpool gate F.linear"),
    "idx_wq_b": (4096, 1536, "bf16", 11, "graph, MLA indexer wq_b"),
    "draft_conv": (1024, 4096, "bf16", 10, "graph, DFlash2 conv kernel_projection (5 layers x 2)"),
    "draft_ctx_kv": (5120, 4096, "bf16", 1, "DFlash context K/V fused projection"),
    "draft_fc": (4096, 20480, "bf16", 1, "DFlash fc, eager"),
    # router with the dedup: production = 2 identical gate calls per MoE layer (the 2nd on warm L2), new = 1 call
    "router_pair": (288, 4096, "router_pair", 42, "graph, per MoE layer: 2 gate calls -> 1"),
}
M_LIST = (1, 5, 6, 8, 16, 24, 32, 40, 48, 64)


def graph_time(fn, calls: int, reps: int) -> float:
    """us per call: median over reps of one graph replay holding `calls` calls of fn(i)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(min(calls, 3)):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / calls)
    del g
    return statistics.median(ts)


class Case:
    def __init__(self, name: str, dev) -> None:
        self.name = name
        self.N, self.K, self.kind, self.per_step, self.where = SHAPES[name]
        self.bytes = self.N * self.K * 2
        self.nw = max(2, math.ceil(96 * 2 ** 20 / self.bytes))
        gen = torch.Generator(device=dev).manual_seed(1)
        self.W = [(torch.randn(self.N, self.K, device=dev, generator=gen) * 0.02).to(torch.bfloat16)
                  for _ in range(self.nw)]
        self.W32 = [w.t().contiguous().float() for w in self.W] if self.kind == "f32" else None
        self.ctx = [G.GemmCtx(self.N, self.K, dev, f32=self.kind == "f32",
                              max_splits=None if self.kind == "f32" else 32) for _ in range(self.nw)]
        self.calls = max(self.nw, min(64, 8 * self.nw)) if self.bytes < 2 ** 24 else self.nw * 2

    def base_fn(self, x):
        if self.kind == "router_pair":
            def f(i):
                F.linear(x, self.W[i % self.nw]).to(torch.float32)
                return F.linear(x, self.W[i % self.nw]).to(torch.float32)
            return f
        if self.kind == "router":
            return lambda i: F.linear(x, self.W[i % self.nw]).to(torch.float32)
        if self.kind == "f32":
            return lambda i: torch.mm(x.float(), self.W32[i % self.nw])
        return lambda i: F.linear(x, self.W[i % self.nw])

    def new_fn(self, x, plan=None, use_pol=1, served=False):
        """served: exactly what production runs with GLM53_BF16_GEMV=1 (the plan table's plan, else the production
        op; for router_pair one call)."""
        if served:
            import glm53_gemv_install as I
            M = x.shape[0]
            if self.kind == "f32":
                if M <= G.F32_MAX_M:
                    return lambda i: G.gemm_f32(x, self.W[i % self.nw], self.ctx[i % self.nw])
                return self.base_fn(x)
            if G.serve_plan(self.N, self.K, M) is None:
                if self.kind == "router_pair":   # dedup still removes the second call
                    return lambda i: F.linear(x, self.W[i % self.nw]).to(torch.float32)
                return self.base_fn(x)
            om = 2 if self.kind in ("router", "router_pair") else 0
            return lambda i: G.gemm(x, self.W[i % self.nw], self.ctx[i % self.nw], out_mode=om,
                                    plan=G.serve_plan(self.N, self.K, M))
        if self.kind == "f32":
            return lambda i: G.gemm_f32(x, self.W[i % self.nw], self.ctx[i % self.nw])
        om = 2 if self.kind in ("router", "router_pair") else 0
        return lambda i: G.gemm(x, self.W[i % self.nw], self.ctx[i % self.nw], out_mode=om, plan=plan,
                                use_pol=use_pol)

    def plans(self, M):
        out = []
        for S, wn, kch in itertools.product((1, 2, 3, 4, 6, 8, 12, 16, 32), (2, 4, 8), (128, 256, 512)):
            if not G._valid(self.N, self.K, S, wn, kch, M):
                continue
            groups = (self.N // 8 + wn - 1) // wn
            if groups * S > 4096 or (groups * S < 24 and self.N >= 1024):
                continue
            if M > 32 and kch == 512 and wn < 4:
                continue
            out.append((S, wn, kch))
        return out


def kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(0)
        torch.cuda.synchronize()
    return [e.name[:90] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--names", action="store_true", help="print the cuBLAS kernels of each baseline at M=5")
    ap.add_argument("--served", action="store_true", help="new = what GLM53_BF16_GEMV=1 runs (plan table / fallback)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    G.load_ext()
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "?"
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head, dirty = "?", ""
    print("ext", G.EXT_SOURCE, "torch", torch.__version__, "device", torch.cuda.get_device_name(), "| git", head,
          "(dirty)" if dirty else "", "| args", " ".join(sys.argv[1:]))
    rows = []
    for name in a.shapes.split(","):
        c = Case(name, dev)
        print(f"== {name}: N={c.N} K={c.K} {c.bytes / 2**20:.2f} MiB x {c.nw} copies, {c.calls} calls/graph; "
              f"{c.per_step}/step ({c.where})", flush=True)
        if a.names:
            x = torch.randn(5, c.K, device=dev).to(torch.bfloat16)
            print("   baseline kernels (M=5):", kernel_names(c.base_fn(x)))
            print("   new kernels (M=5):", kernel_names(c.new_fn(x)))
        for M in [int(v) for v in a.ms.split(",")]:
            x = torch.randn(M, c.K, device=dev).to(torch.bfloat16)
            tb = graph_time(c.base_fn(x), c.calls, a.reps)
            best = None
            if a.sweep and c.kind != "f32":
                res = []
                for pl in c.plans(M):
                    for up in (1, 0):
                        t = graph_time(c.new_fn(x, pl, up), c.calls, a.reps)
                        res.append((t, pl, up))
                res.sort()
                best = res[0]
                top = " ".join(f"{p}/{u}:{t:.1f}" for t, p, u in res[:4])
                print(f"   M={M:2d} sweep top: {top}", flush=True)
            tn = graph_time(c.new_fn(x, served=a.served), c.calls, a.reps)
            if a.served:
                plan = (("gemm_f32" if M <= G.F32_MAX_M else "production") if c.kind == "f32"
                        else (G.serve_plan(c.N, c.K, M) or "production"))
            else:
                plan = None if c.kind == "f32" else G.plan_for(c.N, c.K, M)
            r = dict(shape=name, N=c.N, K=c.K, M=M, base_us=tb, new_us=tn, base_gbs=c.bytes / tb / 1e3,
                     new_gbs=c.bytes / tn / 1e3, speedup=tb / tn, plan=plan, per_step=c.per_step,
                     sweep_best=(best[0], best[1], best[2]) if best else None)
            rows.append(r)
            sb = f"  sweep best {best[0]:7.1f} us {best[1]} pol={best[2]}" if best else ""
            print(f"   M={M:2d} base {tb:7.1f} us {r['base_gbs']:6.1f} GB/s | new {tn:7.1f} us {r['new_gbs']:6.1f} GB/s"
                  f" | x{r['speedup']:.2f} plan {plan}{sb}", flush=True)
        del c
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
