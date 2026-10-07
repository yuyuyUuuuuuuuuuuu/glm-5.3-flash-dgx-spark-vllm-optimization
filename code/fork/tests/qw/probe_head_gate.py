"""Probe: indexer head gate (production: torch.mm(hidden_states.float(), w32), w32 = fp32 copy of 32 bf16 weight rows,
cuBLAS cutlass_80_simt_sgemm at prefill M): can a Triton IEEE-fp32 kernel reading bf16 x give the same bits, faster?"""
import torch, triton, triton.language as tl
DEV = "cuda"


@triton.jit
def gate_kernel(X, W, O, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rn = tl.arange(0, N)
    acc = tl.zeros((BM, N), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(X + rm[:, None].to(tl.int64) * K + rk[None, :], mask=rm[:, None] < M, other=0.0).to(tl.float32)
        b = tl.load(W + rk[:, None] * N + rn[None, :])
        if MODE == 0:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:   # explicit sequential fma over k
            for kk in tl.static_range(BK):
                sel = (tl.arange(0, BK) == kk)
                ak = tl.sum(tl.where(sel[None, :], a, 0.0), axis=1)
                bk = tl.sum(tl.where(sel[:, None], b, 0.0), axis=0)
                acc = tl.fma(ak[:, None], bk[None, :], acc)
    tl.store(O + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=rm[:, None] < M)


def t(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2]


import sys
Ms = [int(v) for v in sys.argv[1:]] or [1024, 1791, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 7168, 8192, 9216, 10240,
                                        11264, 12288, 13000, 13824, 14336, 16384]
for T in Ms:
    g = torch.Generator(device=DEV).manual_seed(T)
    x = (torch.randn(T, 4096, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    wb = (torch.randn(32, 4096, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w32 = wb.t().contiguous().float()
    ref = torch.mm(x.float(), w32)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        torch.mm(x.float(), w32); torch.cuda.synchronize()
    names = sorted({e.name[:50] for e in p.events() if e.device_type.name == "CUDA" and "elementwise" not in e.name})
    o = torch.empty(T, 32, device=DEV)
    gate_kernel[(triton.cdiv(T, 32),)](x, w32, o, T, K=4096, N=32, BM=32, BK=32, MODE=0, num_warps=4)
    tp = t(lambda: torch.mm(x.float(), w32))
    tt = t(lambda: gate_kernel[(triton.cdiv(T, 32),)](x, w32, o, T, K=4096, N=32, BM=32, BK=32, MODE=0, num_warps=4))
    print(f"M={T:6d} equal {torch.equal(o, ref)} prod {tp:.3f} ms triton {tt:.3f} ms cuBLAS {names}", flush=True)
