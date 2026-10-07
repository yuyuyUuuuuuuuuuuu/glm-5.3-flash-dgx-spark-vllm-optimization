"""rejected-idea sizing: drafter MLP at W4A16 (Marlin GPTQ uint4b8, group 128) vs production's FP8 (fp8_gemv TABLE)
at the drafter's 8 rows, cold weights (5 layer copies rotated), CUDA graphs, median us per call."""
import statistics as st, sys, torch
sys.path.insert(0, "/w"); sys.path.insert(0, "/w/tests")
import fp8_gemv as G
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new, apply_gptq_marlin_linear
from vllm.scalar_type import scalar_types
from fp8_bench_common import random_marlin_layers
dev = "cuda"
E = G.ext(); G.STATE.ext = E; G.STATE.enabled = True
for (N, K) in ((12288, 4096), (4096, 6144)):
    M = 8
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    w4 = []
    for i in range(5):
        w = torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.02
        w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, 128, False)
        w4.append((q_w, s, g_idx, sort_idx))
        del w, w_ref
    ws = marlin_make_workspace_new(torch.device(dev))
    f8 = random_marlin_layers(N, K, 5, dev) if True else None
    cfg = G.select_config(N, K, M)
    y8 = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    def r4(i):
        q_w, s, g_idx, sort_idx = w4[i]
        return apply_gptq_marlin_linear(input=x, weight=q_w, weight_scale=s, weight_zp=None, g_idx=g_idx,
                                        g_idx_sort_indices=sort_idx, workspace=ws, wtype=scalar_types.uint4b8,
                                        output_size_per_partition=N, input_size_per_partition=K, is_k_full=True)
    def r8(i):
        L = f8[i]
        E.fp8_gemv_out(y8, x, L.weight, L.weight_scale.view(-1), None, N, K, *cfg, True)
    gs = {}
    for name, fn in (("w4 marlin", r4), ("fp8 prod", r8)):
        fn(0); fn(1); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(10): fn(i % 5)
        gs[name] = g
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = {k: [] for k in gs}
    for r in range(9):
        for k, g in gs.items():
            g.replay(); torch.cuda.synchronize(); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            t[k].append(e0.elapsed_time(e1) * 100)
    print(f"N={N} K={K} M={M}: " + "  ".join(f"{k} {st.median(v):.1f} us" for k, v in t.items()), flush=True)
