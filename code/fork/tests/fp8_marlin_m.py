"""Marlin FP8 time vs M and input layout (why is KDA o_proj 107 us in the production trace?)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from fp8_bench_common import random_marlin_layers, marlin, graph_time_us
dev = "cuda"
for n, k in ((4096, 4096), (12576, 4096), (4096, 6144)):
    copies = max(2, -(-96 * 2**20 // (n * k)))
    layers = random_marlin_layers(n, k, copies)
    line = f"N={n} K={k}:"
    for M in (1, 2, 4, 5, 6, 7, 8, 16):
        x = torch.randn(M, k, device=dev).to(torch.bfloat16)
        t, _ = graph_time_us([lambda l=l: marlin(l, x, n, k) for l in layers], 4 * copies, rounds=3)
        line += f" M{M}:{t:.1f}"
    xb = torch.randn(8, 2 * k, device=dev).to(torch.bfloat16)[:, :k]
    t, _ = graph_time_us([lambda l=l: marlin(l, xb, n, k) for l in layers], 4 * copies, rounds=3)
    line += f" | strided M8:{t:.1f}"
    xz = torch.zeros(8, k, device=dev).to(torch.bfloat16)
    t, _ = graph_time_us([lambda l=l: marlin(l, xz, n, k) for l in layers], 4 * copies, rounds=3)
    line += f" zeros M8:{t:.1f}"
    print(line, flush=True)
    del layers
