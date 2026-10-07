"""opt-decode-rev: what one commit costs in each mode (34 layers, production shapes of one TP rank, nodeC).

fast   : the single commit launch (the normal path)
check  : a self-checked step (first GLM53_DEC_KDA_LAZY_VERIFY commits, then one in VERIFY_EVERY)
repair : every commit after a self-check mismatch (production's kernel per layer + table building)
Wall time per commit() call with a device sync after it (host + GPU), median of R calls, after a verify of M rows.
Usage: probe_kda_lazy_commit_modes.py [M] [nseq] [R]
"""
import os
import statistics as st
import sys
import time

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402

M = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 1
R = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NL, H, KD, V, TMAX = 34, 32, 128, 128, 8
dev = "cuda"
torch.manual_seed(0)
L._ORIG["frk"] = prod_frk
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
out = torch.empty(q.shape, dtype=q.dtype, device=dev)
L.allocate(NL, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True
slots = [L.register(states[l], a_log[l], g_bias[l]) for l in range(NL)]
num_sampled = torch.full((NSEQ,), max(1, M - 2), device=dev, dtype=torch.int32)
idx_map = torch.arange(NSEQ, device=dev, dtype=torch.int32)


def verify():
    for l in range(NL):
        L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, states[l], cu, idx, nacc, out, a_log[l], g_bias[l], -5.0,
                      slots[l])


def run(mode):
    os.environ["GLM53_DEC_KDA_LAZY_VERIFY"] = "0"
    os.environ["GLM53_DEC_KDA_LAZY_VERIFY_EVERY"] = "1" if mode == "check" else "0"
    L.ST.repair = mode == "repair"
    ts = []
    for i in range(R + 3):
        verify()
        torch.cuda.synchronize()
        L.COUNTERS["commits"] = 10 ** 6 + 2 * i + 1     # odd: VERIFY_EVERY=1 still checks (n % 1 == 0)
        t0 = time.perf_counter()
        L.commit(num_sampled, idx_map, NSEQ)
        torch.cuda.synchronize()
        if i >= 3:
            ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


res = {m: run(m) for m in ("fast", "check", "repair", "fast")}
print(f"M={M} nseq={NSEQ}: commit wall ms (host+GPU, synced) fast {res['fast']:.3f}  checked step "
      f"{res['check']:.3f}  repair mode {res['repair']:.3f}  (mismatches {L.COUNTERS['verify_mismatch']})",
      flush=True)
