"""Probe: what the KDA spec-verify per-token recurrent-state stores cost inside a bandwidth-bound decode step.

One "step" = 34 KDA layers, each: in_proj stand-in GEMV (bf16, 51.5 MB) -> fused_recurrent_kda (the image's kernel,
spec-decoding path, production shapes H=32 K=V=128 fp32 state, 1 request, T = M rows) -> o_proj stand-in (16.8 MB)
-> MoE stand-in (177 MB). Captured as one CUDA graph per mode, replayed alternately.
  stock : ssm_state_indices row = 8 real slots -> the kernel stores the state after every token (M x 2 MiB)
  one   : only the initial column is a real slot (num_accepted = 1 -> column 0), others NULL (0) -> 1 store (t = 0)
  none  : same as one but the initial slot index is also used... (not possible: idx<=0 skips the program) -> skipped
Plus the cost of a stand-alone 'commit' pass (the same kernel, T = A tokens, one real column) per layer, eager-graph.
"""
import sys, time, statistics as st
import torch
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda

torch.manual_seed(0)
dev = "cuda"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 34
H, Kd, V = 32, 128, 128
NSLOT = 8
def gemv_w(nbytes, K=4096):
    N = nbytes // (2 * K)
    return torch.randn(K, N, device=dev, dtype=torch.bfloat16) * 0.01
W_in = [gemv_w(51_500_000) for _ in range(L)]
W_o = [gemv_w(16_800_000) for _ in range(L)]
W_m = [gemv_w(177_000_000) for _ in range(L)]
state = [torch.randn(1 + NSLOT, H, V, Kd, device=dev, dtype=torch.float32) * 0.01 for _ in range(L)]
a_log = torch.randn(H, device=dev) * 0.1
g_bias = torch.randn(H * Kd, device=dev) * 0.1

def make_inputs(M):
    q = torch.randn(1, M, H, Kd, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, M, H, Kd, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, M, H, V, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, M, H, Kd, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu = torch.tensor([0, M], device=dev, dtype=torch.int32)
    return q, k, v, g, beta, out, cu

def run_step(M, idx, nacc, inp, x):
    q, k, v, g, beta, out, cu = inp
    for l in range(L):
        y = x @ W_in[l]
        fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state[l], use_qk_l2norm_in_kernel=True,
                            cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=nacc, out=out,
                            sigmoid_beta=True, a_log=a_log, g_bias=g_bias, compute_gate=True, lower_bound=-5.0)
        z = x @ W_o[l]
        w = x @ W_m[l]
    return y, z, w

def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        fn()
    return gph

res = {}
for M in [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "5,6,8").split(",")]:  # one M per process: graphs of several M in one process crashed (pool reuse)
    inp = make_inputs(M)
    x = torch.randn(M, 4096, device=dev, dtype=torch.bfloat16)
    nacc = torch.ones(1, device=dev, dtype=torch.int32)
    idx_stock = torch.arange(1, NSLOT + 1, device=dev, dtype=torch.int32).view(1, NSLOT)
    idx_one = torch.zeros(1, NSLOT, device=dev, dtype=torch.int32); idx_one[0, 0] = 1
    graphs = {"stock": capture(lambda: run_step(M, idx_stock, nacc, inp, x)),
              "one": capture(lambda: run_step(M, idx_one, nacc, inp, x))}
    # commit pass alone: T = A tokens, only column A-1 real, initial col 0 real too (A>1 also stores at t=0: worst case)
    A = 3
    inpA = make_inputs(A)
    idx_c = torch.zeros(1, NSLOT, device=dev, dtype=torch.int32); idx_c[0, 0] = 1; idx_c[0, A - 1] = 2
    def commit():
        q, k, v, g, beta, out, cu = inpA
        for l in range(L):
            fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state[l], use_qk_l2norm_in_kernel=True,
                                cu_seqlens=cu, ssm_state_indices=idx_c, num_accepted_tokens=nacc, out=out,
                                sigmoid_beta=True, a_log=a_log, g_bias=g_bias, compute_gate=True, lower_bound=-5.0)
    graphs["commit"] = capture(commit)
    times = {kk: [] for kk in graphs}
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(12):
        order = list(graphs) if r % 2 == 0 else list(graphs)[::-1]
        for kk in order:
            torch.cuda.synchronize()
            ev0.record(); graphs[kk].replay(); ev1.record(); torch.cuda.synchronize()
            times[kk].append(ev0.elapsed_time(ev1))
    med = {kk: st.median(v[2:]) for kk, v in times.items()}
    spread = {kk: (max(v[2:]) - min(v[2:])) / st.median(v[2:]) * 100 for kk, v in times.items()}
    print(f"M={M} L={L}: stock {med['stock']:.3f} ms  one-store {med['one']:.3f} ms  saving {med['stock']-med['one']:.3f} ms"
          f"  ({(med['stock']-med['one'])/L*1000:.1f} us/layer; written {M-1} x 2 MiB less per layer)"
          f"  commit(A={A}) {med['commit']:.3f} ms  spread% {spread}", flush=True)
