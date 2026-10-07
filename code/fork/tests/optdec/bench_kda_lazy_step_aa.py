"""opt-decode-rev: A/A-controlled re-run of bench_kda_lazy_step.py (same step model, same production kernel).

Four CUDA graphs captured in one process: prod1, prod2 (production's strided KDA kernel, two independent captures of
the identical step) and lazy1, lazy2 (lazy verify + eager commit). Every round replays all four in a fresh random
order (same A draw for the round). Reported, per pair, the median of the per-round differences:
  A/A prod1 - prod2, A/A lazy1 - lazy2  (noise floor and capture/placement artifacts)
  A/B prod - lazy                        (mean of the two prod arms minus the mean of the two lazy arms, per round)
plus the HOST time of the commit() call (enqueue only; the GPU is still busy with the graph): in production the commit
is enqueued between sampling and the drafter, so this is host work added per verify step.
Usage: bench_kda_lazy_step_aa.py <M> [rounds] [nseq]
"""
import random
import statistics as st
import sys
import time

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
NSEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NL, H, KD, V, TMAX = 34, 32, 128, 128, 8
dev = "cuda"
torch.manual_seed(0)
random.seed(1)


def gemv_w(nbytes, K=4096):
    return torch.randn(K, nbytes // (2 * K), device=dev, dtype=torch.bfloat16) * 0.01


W_in = [gemv_w(51_500_000) for _ in range(NL)]
W_o = [gemv_w(16_800_000) for _ in range(NL)]
W_m = [gemv_w(177_000_000) for _ in range(NL)]
NSLOT = 1 + NSEQ * TMAX
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


ARMS = ("prod1", "prod2", "lazy1", "lazy2")
G = {a: capture((lambda lz: (lambda: step(lz)))(a.startswith("lazy"))) for a in ARMS}
num_sampled = torch.ones(NSEQ, device=dev, dtype=torch.int32)
idx_map = torch.arange(NSEQ, device=dev, dtype=torch.int32)
L.COUNTERS["commits"] = 10 ** 6 + 1          # past the self-check window (the A/B is about the fast commit)
import os  # noqa: E402
os.environ["GLM53_DEC_KDA_LAZY_VERIFY"] = "0"
os.environ["GLM53_DEC_KDA_LAZY_VERIFY_EVERY"] = "0"


def draw_A():
    a = 1
    for p in (0.86, 0.74, 0.73, 0.78, 0.62, 0.6, 0.6):
        if a >= M or random.random() > p:
            break
        a += 1
    return a


ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
times = {a: [] for a in ARMS}
host_commit = []
for r in range(ROUNDS + 2):
    num_sampled.copy_(torch.tensor([draw_A() for _ in range(NSEQ)], dtype=torch.int32))
    order = list(ARMS)
    random.shuffle(order)
    for arm in order:
        torch.cuda.synchronize()
        ev[0].record()
        G[arm].replay()
        if arm.startswith("lazy"):
            t0 = time.perf_counter()
            L.commit(num_sampled, idx_map, NSEQ, num_computed=None, block_size=None, align=False)
            if r >= 2:
                host_commit.append((time.perf_counter() - t0) * 1e3)
        ev[2].record()
        torch.cuda.synchronize()
        if r >= 2:
            times[arm].append(ev[0].elapsed_time(ev[2]))
med = {a: st.median(v) for a, v in times.items()}
aa_p = [a - b for a, b in zip(times["prod1"], times["prod2"])]
aa_l = [a - b for a, b in zip(times["lazy1"], times["lazy2"])]
ab = [(a + b) / 2 - (c + d) / 2 for a, b, c, d in zip(times["prod1"], times["prod2"], times["lazy1"], times["lazy2"])]


def ci(xs):
    s = sorted(xs)
    n = len(s)
    return s[max(0, int(0.1 * n))], s[min(n - 1, int(0.9 * n))]


print(f"M={M} nseq={NSEQ} rounds={ROUNDS}: medians prod1 {med['prod1']:.3f} prod2 {med['prod2']:.3f} "
      f"lazy1 {med['lazy1']:.3f} lazy2 {med['lazy2']:.3f} ms | A/A prod {st.median(aa_p):+.3f} "
      f"[p10 {ci(aa_p)[0]:+.3f}, p90 {ci(aa_p)[1]:+.3f}]  A/A lazy {st.median(aa_l):+.3f} "
      f"[{ci(aa_l)[0]:+.3f}, {ci(aa_l)[1]:+.3f}] | A/B prod-lazy {st.median(ab):+.3f} "
      f"[{ci(ab)[0]:+.3f}, {ci(ab)[1]:+.3f}] | host commit() {st.median(host_commit)*1e3:.0f} us "
      f"(p90 {ci(host_commit)[1]*1e3:.0f})", flush=True)
