"""DEC_DLMH stage 0b: speed of the int4 coarse lm_head GEMV (Triton, per-rank shard 77440 x 4096) vs production's
FP8 lm_head (fp8_gemv, the TABLE config) at the drafter's row counts, cold weights (2 copies rotated: 2 x >150 MB
>> 24 MiB L2), CUDA-graph replay, interleaved rounds.

  tests/gpu_run.sh python3 tests/dlmh/bench_coarse.py
"""
import statistics as st
import sys

import torch

sys.path.insert(0, "/w")
import fp8_gemv as G  # noqa: E402
import glm53_dlmh as D  # noqa: E402
D.parse_env()


dev = "cuda"
N, K = 77440, 4096
torch.manual_seed(0)
E = G.ext()


def fp8_layers(n):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    import torch.nn as nn
    out = []
    for i in range(n):
        h = nn.Module()
        w = torch.randn(N, K, device=dev) * 0.02
        sc = w.abs().amax(1).clamp(min=1e-12) / 448.0
        h.weight = nn.Parameter((w / sc[:, None]).to(torch.float8_e4m3fn), requires_grad=False)
        h.weight_scale = nn.Parameter(sc.to(torch.bfloat16), requires_grad=False)
        h.output_size_per_partition, h.input_size_per_partition, h.orig_dtype = N, K, torch.bfloat16
        h.weight_block_size = None
        prepare_fp8_layer_for_marlin(h, size_k_first=False)
        del w
        out.append(h)
    return out


L8 = fp8_layers(2)
coarse = []
for h in L8:
    wq, s = D.build_coarse_from_marlin(h.weight, h.weight_scale.view(-1), N, K, group=int(D.CFG.group))
    coarse.append((wq, s))
torch.cuda.synchronize()
print(f"coarse bytes/copy {(coarse[0][0].numel() * 4 + coarse[0][1].numel() * 2) / 1e6:.1f} MB, fp8 "
      f"{L8[0].weight.numel() * 4 / 1e6:.1f} MB", flush=True)


def capture(fn, reps=4):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(0); fn(1)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i % 2)
    return g, reps


def timeit(g, reps, n=20):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / reps)
    return st.median(ts)


for M in (7, 14):
    x = (torch.randn(M, K, device=dev) * 1.0).to(torch.bfloat16)
    cfg = G.select_config(N, K, M)
    yf = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    def prod(i):
        E.fp8_gemv_out(yf, x, L8[i].weight, L8[i].weight_scale.view(-1), None, N, K, *cfg, True)

    res = {"fp8_prod": capture(prod)}
    for (bn, bw, nw, ns) in D.TUNE_CANDIDATES:
        if int(D.CFG.group) % bw:
            continue
        yc = torch.empty(M, N, device=dev, dtype=torch.float32)

        def co(i, bn=bn, bw=bw, nw=nw, ns=ns, yc=yc):
            D.coarse_gemv(x, coarse[i][0], coarse[i][1], yc, group=int(D.CFG.group), block_n=bn, block_w=bw,
                          num_warps=nw, num_stages=ns)
        try:
            res[f"coarse bn{bn} bw{bw} w{nw} s{ns}"] = capture(co)
        except Exception as exc:  # noqa: BLE001
            print("skip", bn, bw, nw, ns, repr(exc)[:200])
    rounds = {k: [] for k in res}
    for r in range(7):
        for k, (g, reps) in (list(res.items()) if r % 2 == 0 else list(res.items())[::-1]):
            rounds[k].append(timeit(g, reps, 5))
    cb = (coarse[0][0].numel() * 4 + coarse[0][1].numel() * 2)
    for k, v in sorted(rounds.items(), key=lambda kv: st.median(kv[1])):
        us = st.median(v)
        by = L8[0].weight.numel() * 4 if k == "fp8_prod" else cb
        print(f"M={M:2d} {k:32s} {us:8.1f} us  {by / us / 1e3:6.1f} GB/s", flush=True)
