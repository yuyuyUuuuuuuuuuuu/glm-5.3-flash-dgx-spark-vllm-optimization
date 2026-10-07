"""KDA lazy verify kernel: launch-config sweep at production's decode shapes (H=32 local heads, K=V=128, N=1..2,
T = 5 / 6 / 8 rows), 34 distinct state tensors (cold, like 34 layers), inside a CUDA graph. Every variant's outputs
(o + the saved scratch rows) are compared BITWISE with production's launch config (BV=8, num_warps=1, num_stages=3).
Usage: tests/gpu_run.sh python3 tests/decode6/bench_kda_verify_cfg.py
"""
import statistics
import sys

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402

dev = "cuda"
H, KD, V = 32, 128, 128
NL = 34
TMAX = 8
LB = -5.0
g = torch.Generator(device=dev).manual_seed(5)
a_log = (torch.randn(H, device=dev, generator=g) * 0.5).float()
g_bias = (torch.randn(H * KD, device=dev, generator=g) * 0.5).float()
L.allocate(NL, 2, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True
K_ = L._kernels()
NSLOT = 1 + 2 * TMAX
states = [(torch.randn(NSLOT, H, V, KD, device=dev, generator=g) * 0.05).float() for _ in range(NL)]


def make(N, T):
    tot = N * T
    qkv = (torch.randn(1, tot, 3 * H * KD + 64, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    q = qkv[..., : H * KD].view(1, tot, H, KD)
    k = qkv[..., H * KD: 2 * H * KD].view(1, tot, H, KD)
    v = qkv[..., 2 * H * KD: 3 * H * KD].view(1, tot, H, V)
    bproj = (torch.randn(1, tot, H + 96, device=dev, generator=g) * 2).to(torch.bfloat16)
    beta = bproj[..., 7:7 + H]
    gg = (torch.randn(1, tot, H, KD, device=dev, generator=g) * 2).to(torch.bfloat16)
    cu = torch.arange(0, N + 1, device=dev, dtype=torch.int32) * T
    idx = (torch.arange(N * TMAX, device=dev, dtype=torch.int32).view(N, TMAX) + 1)
    nacc = torch.full((N,), 2, device=dev, dtype=torch.int32)
    return q, k, v, gg, beta, cu, idx, nacc


def launch(l, inp, out, BV, nw, ns):
    q, k, v, gg, beta, cu, idx, nacc = inp
    N = cu.numel() - 1
    T = q.shape[1]
    grid = (1, V // BV, N * H)
    K_["verify"][grid](
        q, k, v, gg.contiguous(), beta, out, states[l], cu, idx, nacc, a_log, g_bias, KD ** -0.5,
        L.ST.sk[l], L.ST.sv[l], L.ST.sg[l], L.ST.sb[l], L.ST.meta[l],
        N, T, H=H, HV=H, K=KD, V=V, BK=KD, BV=BV,
        stride_init_state_token=states[l].stride(0), stride_indices_seq=idx.stride(0),
        stride_q_t=L._token_stride(q), stride_k_t=L._token_stride(k), stride_v_t=L._token_stride(v),
        stride_beta_t=L._token_stride(beta), TMAX=TMAX, META_W=3 + TMAX, LAZY=True, LOWER_BOUND=LB,
        num_warps=nw, num_stages=ns)


def snap(l):
    return [L.ST.sk[l].clone(), L.ST.sv[l].clone(), L.ST.sg[l].clone(), L.ST.sb[l].clone(), L.ST.meta[l].clone()]


CFGS = [(8, 1, 3), (8, 1, 1), (8, 1, 2), (8, 2, 3), (8, 4, 3), (4, 1, 3), (16, 1, 3), (16, 2, 3), (32, 2, 3),
        (32, 4, 3), (2, 1, 3)]
for N, T in ((1, 5), (1, 6), (1, 8), (2, 5), (2, 8)):
    inp = make(N, T)
    outs = {}
    ref = None
    times = {c: [] for c in CFGS}
    graphs = {}
    for c in CFGS:
        o = torch.empty(inp[0].shape, dtype=torch.bfloat16, device=dev)
        try:
            launch(0, inp, o, *c)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"N={N} T={T} cfg BV={c[0]} nw={c[1]} ns={c[2]}: launch failed {type(e).__name__}")
            continue
        s = snap(0)
        if ref is None:
            ref = (o.clone(), s)
            outs[c] = True
        else:
            same = torch.equal(o.view(torch.int16), ref[0].view(torch.int16)) and all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(s, ref[1]))
            outs[c] = same
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for l in range(NL):
                launch(l, inp, o, *c)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for l in range(NL):
                launch(l, inp, o, *c)
        graphs[c] = gr
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    for r in range(15):
        for c in graphs:
            flush.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[c].replay()
            e1.record()
            e1.synchronize()
            times[c].append(e0.elapsed_time(e1) * 1000 / NL)
    base = statistics.median(times[CFGS[0]])
    for c in graphs:
        m = statistics.median(times[c])
        print(f"N={N} T={T} BV={c[0]:2d} nw={c[1]} ns={c[2]}: {m:6.2f} us/layer ({m - base:+6.2f}) "
              f"bitwise==production {outs[c]}", flush=True)
print("done")
