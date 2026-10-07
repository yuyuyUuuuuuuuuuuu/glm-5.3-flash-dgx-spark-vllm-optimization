"""Shared helpers for the FP8 Marlin-layout GEMM tests/benchmarks (production shapes, quantization, timing)."""
import statistics
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

# Every FP8-Marlin linear of one production decode step, per rank (TP=2), from the model code + config and matched
# to the R7 profile (docs/logs/prod-R7 / tools/prodcheck/fp8_marlin_calls.py): name -> (N, K, calls per step)
PROD_SHAPES = {
    "kda.in_proj":    (12576, 4096, 34),
    "kda.f_b":        (4096, 128, 34),
    "kda.g_b":        (4096, 128, 34),
    "kda.o_proj":     (4096, 4096, 34),
    "mla.fused_qkv_a": (2048, 4096, 11),
    "mla.q_b":        (8192, 1536, 11),
    "mla.o_proj":     (4096, 8192, 11),
    "dense.gate_up":  (12288, 4096, 3),
    "dense.down":     (4096, 6144, 3),
    "shared.gate_up": (2048, 4096, 42),
    "shared.down":    (4096, 1024, 42),
    "draft.qkv":      (3072, 4096, 5),
    "draft.o":        (4096, 2048, 5),
    "draft.gate_up":  (12288, 4096, 5),
    "draft.down":     (4096, 6144, 5),
    "draft.lmhead":   (77440, 4096, 1),
}
# Production Marlin time per call (us, median) in the R7 decode-step profile, rank 0 (batch 1, prose, 78 ms/step):
# tools/prodcheck/fp8_marlin_calls.py -> docs/logs/fp8_gemv/r7_rank0_marlin_calls.txt. The KDA in_proj Marlin call is
# followed by marlin_unpad_output's copy (N 12608 -> 12576): 2.4 us per call in the same trace (not in this table).
PROD_R7_US = {
    "kda.in_proj": 238.3, "kda.f_b": 5.3, "kda.g_b": 5.0, "kda.o_proj": 106.9, "mla.fused_qkv_a": 39.8,
    "mla.q_b": 58.8, "mla.o_proj": 151.2, "dense.gate_up": 227.8, "dense.down": 108.7, "shared.gate_up": 41.7,
    "shared.down": 25.0, "draft.qkv": 55.3, "draft.o": 40.4, "draft.gate_up": 222.0, "draft.down": 107.0,
    "draft.lmhead": 1385.4,
}
PROD_R7_UNPAD_COPY_US = 2.4
# the same (N, K) serve several names
UNIQUE_SHAPES = sorted({(n, k) for n, k, _ in PROD_SHAPES.values()}, key=lambda s: -s[0] * s[1])


def quantize_like_prod(w):
    """Glm53DenseFp8Method.process_weights_after_loading (launcher overlay exl3.py) for a BF16 [N, K] weight.
    Returns (layer with Marlin weight / weight_scale / workspace, fp8 [N, K], stored per-channel scale [N] bf16)."""
    n, k = w.shape
    wf = w.float()
    scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    fp8 = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    del wf
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = w.dtype
    layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.to(w.dtype), requires_grad=False)
    layer.weight_block_size = None
    scales_stored = layer.weight_scale.detach().clone()
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    layer.glm53_fp8_n, layer.glm53_fp8_k = n, k
    return layer, fp8, scales_stored


def marlin(layer, x, n, k, bias=None):
    return apply_fp8_marlin_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                                   workspace=layer.workspace, size_n=n, size_k=k, bias=bias)


def random_marlin_layers(n, k, count, dev="cuda", seed=0):
    """count independent random layers (Marlin-packed); weights ~ N(0, 0.02) with per-row spread."""
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for _ in range(count):
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        layer, _, _ = quantize_like_prod(w)
        del w
        out.append(layer)
    return out


def graph_time_us(fns, reps_per_graph, replays=5, rounds=5):
    """Capture reps_per_graph calls (cycling through fns) into one CUDA graph, replay; median us per call."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[: min(len(fns), 3)]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps_per_graph):
            fns[i % len(fns)]()
    g.replay()
    torch.cuda.synchronize()
    res = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        res.append(a.elapsed_time(b) * 1000 / (replays * reps_per_graph))
    del g
    return statistics.median(res), res
