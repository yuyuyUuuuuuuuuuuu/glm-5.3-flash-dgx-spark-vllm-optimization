"""Step-level timing of GLM53_DEC_KDA_LAZY vs production's KDA spec-verify on nodeC (one GB10 = one TP rank's shapes).

One decode step = 34 KDA layers, each: in_proj stand-in GEMV (bf16 weight, 51.5 MB) -> the KDA recurrent call
(production's strided kernel, or the lazy verify kernel) -> o_proj stand-in (16.8 MB) -> MoE stand-in (177 MB),
captured as ONE CUDA graph per mode (production's FULL decode graph). Lazy mode then runs the eager commit (one
launch for all 34 layers) after the replay, as postprocess_state does. Per step: A (accepted rows + 1) per request
drawn from a production-like distribution. Paired, alternating order; median ms per step and per-step saving.
Usage: bench_kda_lazy_step.py <M> [rounds] [nseq]
"""
import random
import statistics as st
import sys

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NL, H, KD, V, TMAX = 34, 32, 128, 128, 8
dev = "cuda"
torch.manual_seed(0)
random.seed(0)


def gemv_w(nbytes, K=4096):
    return torch.randn(K, nbytes // (2 * K), device=dev, dtype=torch.bfloat16) * 0.01


W_in = [gemv_w(51_500_000) for _ in range(NL)]
W_o = [gemv_w(16_800_000) for _ in range(NL)]
W_m = [gemv_w(177_000_000) for _ in range(NL)]
NSLOT = 1 + NSEQ * TMAX
# one KV-cache-like buffer, every layer's state a view (as vLLM's hybrid allocator does)
big = torch.randn(NL, NSLOT, H, V, KD, device=dev, dtype=torch.float32) * 0.01
states = [big[l] for l in range(NL)]
a_log = [torch.randn(H, device=dev) * 0.1 for _ in range(NL)]
g_bias = [torch.randn(H * KD, device=dev) * 0.1 for _ in range(NL)]
T = M * NSEQ
qkv = (torch.randn(1, T, 3 * H * KD, device=dev) * 1.5).to(torch.bfloat16)
q = qkv[..., : H * KD].view(1, T, H, KD)
k = qkv[..., H * KD: 2 * H * KD].view(1, T, H, KD)
v = qkv[..., 2 * H * KD:].view(1, T, H, V)
proj = (torch.randn(1, T, 12576, device=dev) * 2).to(torch.bfloat16)
beta = proj[..., 12288:12288 + H]
gg = (torch.randn(1, T, H, KD, device=dev) * 2).to(torch.bfloat16)
cu = torch.arange(0, (NSEQ + 1) * M, M, device=dev, dtype=torch.int32)
idx = torch.arange(1, 1 + NSEQ * TMAX, device=dev, dtype=torch.int32).view(NSEQ, TMAX)
nacc = torch.ones(NSEQ, device=dev, dtype=torch.int32)
x = torch.randn(T, 4096, device=dev, dtype=torch.bfloat16)
out = torch.empty(q.shape, dtype=q.dtype, device=dev)

L.allocate(NL, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True
slots = [L.register(states[l], a_log[l], g_bias[l]) for l in range(NL)]
assert all(s is not None for s in slots), slots


def step(lazy):
    for l in range(NL):
        y = x @ W_in[l]
        if lazy:
            L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, states[l], cu, idx, nacc, out, a_log[l], g_bias[l], -5.0,
                          slots[l])
        else:
            prod_frk(q=q, k=k, v=v, g=gg, beta=beta, initial_state=states[l], use_qk_l2norm_in_kernel=True,
                     cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=nacc, out=out, sigmoid_beta=True,
                     a_log=a_log[l], g_bias=g_bias[l], compute_gate=True, lower_bound=-5.0)
        z = x @ W_o[l]
        w = x @ W_m[l]
    return y, z, w


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        fn()
    return gph


G = {"prod": capture(lambda: step(False)), "lazy": capture(lambda: step(True))}
num_sampled = torch.ones(NSEQ, device=dev, dtype=torch.int32)
idx_map = torch.arange(NSEQ, device=dev, dtype=torch.int32)


def draw_A():
    # production-like: per-position acceptance 0.86 / 0.74 / 0.73 / 0.78 / 0.6 ... (metrics, 2026-10-03), capped at M
    a = 1
    for p in (0.86, 0.74, 0.73, 0.78, 0.62, 0.6, 0.6):
        if a >= M or random.random() > p:
            break
        a += 1
    return a


ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
times = {"prod": [], "lazy": [], "lazy_graph": [], "commit": []}
for r in range(ROUNDS + 2):
    num_sampled.copy_(torch.tensor([draw_A() for _ in range(NSEQ)], dtype=torch.int32))
    order = ("prod", "lazy") if r % 2 == 0 else ("lazy", "prod")
    for mode in order:
        torch.cuda.synchronize()
        ev[0].record()
        G[mode].replay()
        ev[1].record()
        if mode == "lazy":
            L.commit(num_sampled, idx_map, NSEQ, num_computed=None, block_size=None, align=False)
        ev[2].record()
        torch.cuda.synchronize()
        if r >= 2:
            times[mode].append(ev[0].elapsed_time(ev[2]))
            if mode == "lazy":
                times["lazy_graph"].append(ev[0].elapsed_time(ev[1]))
                times["commit"].append(ev[1].elapsed_time(ev[2]))
med = {kk: st.median(vv) for kk, vv in times.items()}
pair = [p - q_ for p, q_ in zip(times["prod"], times["lazy"])]
print(f"M={M} nseq={NSEQ} rounds={ROUNDS}: prod {med['prod']:.3f} ms  lazy {med['lazy']:.3f} ms (graph "
      f"{med['lazy_graph']:.3f} + commit {med['commit']:.3f})  paired saving median {st.median(pair):.3f} ms "
      f"[min {min(pair):.3f}, max {max(pair):.3f}]  spread prod {100*(max(times['prod'])-min(times['prod']))/med['prod']:.1f}%",
      flush=True)
