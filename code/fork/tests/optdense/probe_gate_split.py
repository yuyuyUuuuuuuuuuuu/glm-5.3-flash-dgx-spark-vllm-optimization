"""Can a Triton IEEE-fp32 kernel reproduce cuBLAS's split-K head gate bits (M <= 9216: memset + simt sgemm)?
Variant: S equal K-splits, each summed k-sequentially (tl.dot ieee, like the unsplit sgemm the M >= 10240 path matches),
then combined in split order: ((s0 + s1) + s2) ..."""
import sys, torch, triton, triton.language as tl
DEV = "cuda"


@triton.jit
def gate_split_kernel(X, W, O, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                      S: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, N)
    KS: tl.constexpr = ((K + S - 1) // S + 7) // 8 * 8      # cutlass gemm_k_size: ceil(K/S) rounded up to kTileK=8
    tot = tl.zeros((BM, N), dtype=tl.float32)
    for s in tl.static_range(S):
        acc = tl.zeros((BM, N), dtype=tl.float32)
        kend = min((s + 1) * KS, K)
        for k0 in range(s * KS, kend, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(X + rm[:, None].to(tl.int64) * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < kend),
                        other=0.0)
            b = tl.load(W + rk[:, None] * N + rn[None, :], mask=rk[:, None] < kend, other=0.0)
            acc = tl.dot(a.to(tl.float32), b, acc, input_precision="ieee")
        tot = tot + acc
    tl.store(O + rm[:, None].to(tl.int64) * N + rn[None, :], tot, mask=rm[:, None] < M)


for T in [int(v) for v in sys.argv[1:]] or [1791, 4289, 6912, 9216]:
    g = torch.Generator(device=DEV).manual_seed(T)
    x = (torch.randn(T, 4096, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    wb = (torch.randn(32, 4096, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w32 = wb.t().contiguous().float()
    ref = torch.mm(x.float(), w32)
    res = []
    for S in range(1, 25):
        o = torch.empty(T, 32, device=DEV)
        gate_split_kernel[(triton.cdiv(T, 32),)](x, w32, o, T, K=4096, N=32, BM=32, BK=16, S=S, num_warps=4)
        res.append(f"S{S}:{int((o != ref).sum())}")
    print(f"M={T:6d} differing elements per split count: {' '.join(res)}", flush=True)
