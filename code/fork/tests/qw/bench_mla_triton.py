"""Triton strided batched GEMM for the MLA absorption matmuls vs production's torch.bmm (T=13824/1791)."""
import sys, itertools, torch, triton, triton.language as tl
dev = "cuda"
N, P, L, V = 32, 256, 512, 256


@triton.jit
def _bmm(A, B, C, M, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
         K: tl.constexpr, NN: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         ORDER: tl.constexpr):
    pid = tl.program_id(0)
    n_t: tl.constexpr = NN // BN
    m_t = tl.cdiv(M, BM)
    if ORDER == 0:      # n fastest, then m, then batch
        pn = pid % n_t
        pm = (pid // n_t) % m_t
        b = pid // (n_t * m_t)
    else:               # n fastest, then batch, then m (all heads of a token block together)
        pn = pid % n_t
        b = (pid // n_t) % NB
        pm = pid // (n_t * NB)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptr = A + b.to(tl.int64) * sab + rm[:, None].to(tl.int64) * sam + rk[None, :] * sak
    b_ptr = B + b.to(tl.int64) * sbb + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.0)
        bb = tl.load(b_ptr)
        acc = tl.dot(a, bb, acc)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    c_ptr = C + b.to(tl.int64) * scb + rm[:, None].to(tl.int64) * scm + rn[None, :] * scn
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rm[:, None] < M)


def tbmm(a, b, c, BM, BN, BK, warps, stages, order):
    NB, M, K = a.shape
    NN = b.shape[2]
    grid = (triton.cdiv(M, BM) * (NN // BN) * NB,)
    _bmm[grid](a, b, c, M, *a.stride(), *b.stride(), *c.stride(), K=K, NN=NN, NB=NB, BM=BM, BN=BN, BK=BK,
               ORDER=order, num_warps=warps, num_stages=stages)
    return c


def timeit(fn, it=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
    for x, y in ev:
        x.record(); fn(); y.record()
    torch.cuda.synchronize()
    ts = sorted(x.elapsed_time(y) for x, y in ev)
    return ts[len(ts) // 2]


g = torch.Generator(device=dev).manual_seed(0)
w = (torch.randn(N * (P + V), L, device=dev, generator=g) * 0.03).to(torch.bfloat16)
kvb = w.T.view(L, N, P + V)
W_UK, W_UV = kvb.split([P, V], dim=-1)
WUKT = W_UK.permute(1, 2, 0)
WUV = W_UV.transpose(0, 1)
quick = "--quick" in sys.argv
Ts = [int(x) for x in sys.argv[1:] if not x.startswith("--")] or [13824, 1791]
for T in Ts:
    q = torch.randn(T, N, P, device=dev, generator=g).to(torch.bfloat16)
    at = torch.randn(T, N, L, device=dev, generator=g).to(torch.bfloat16)
    ouk_ref = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
    torch.bmm(q.transpose(0, 1), WUKT, out=ouk_ref)
    ouv_ref = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
    torch.bmm(at.transpose(0, 1), WUV, out=ouv_ref.transpose(0, 1))
    r_uk = torch.einsum("tnp,npl->ntl", q.float(), WUKT.float())
    r_uv = torch.einsum("tnl,nlv->tnv", at.float(), WUV.float())
    t_uk = timeit(lambda: torch.bmm(q.transpose(0, 1), WUKT, out=ouk_ref))
    t_uv = timeit(lambda: torch.bmm(at.transpose(0, 1), WUV, out=ouv_ref.transpose(0, 1)))
    print(f"T={T} prod bmm: UK {t_uk:.3f} ms  UV {t_uv:.3f} ms  (sum {t_uk + t_uv:.3f})", flush=True)
    best = {}
    cfgs = list(itertools.product([64, 128], [64, 128, 256], [32, 64], [4, 8], [2, 3], [0, 1]))
    for (BM, BN, BK, wp, st, od) in cfgs:
        for tag in ("UK", "UV"):
            try:
                if tag == "UK":
                    o = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
                    fn = lambda: tbmm(q.transpose(0, 1), WUKT, o, BM, BN, BK, wp, st, od)
                    ref, refb, oo = r_uk, ouk_ref, o
                else:
                    o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
                    fn = lambda: tbmm(at.transpose(0, 1), WUV, o.transpose(0, 1), BM, BN, BK, wp, st, od)
                    ref, refb, oo = r_uv, ouv_ref, o
                if tag == "UV" and BN > V:
                    continue
                t = timeit(fn, it=10, warm=2)
                rel = ((oo.float() - ref).norm() / ref.norm()).item()
                relp = ((oo.float() - refb.float()).norm() / refb.float().norm()).item()
                if tag not in best or t < best[tag][0]:
                    best[tag] = (t, (BM, BN, BK, wp, st, od), rel, relp)
            except Exception as e:  # noqa: BLE001
                print("cfg fail", tag, (BM, BN, BK, wp, st, od), repr(e)[:120])
    for tag, (t, c, rel, relp) in best.items():
        print(f"T={T} triton {tag} best {t:.3f} ms cfg(BM,BN,BK,warps,stages,order)={c} relL2 vs fp32 {rel:.3e}"
              f" (vs prod bmm {relp:.3e})", flush=True)
    del q, at, r_uk, r_uv
    torch.cuda.empty_cache()
