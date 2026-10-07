"""MLA absorption GEMMs (W_UK / W_UV) at production shapes: which layouts/calls get cuBLAS off cutlass_80_wmma.
Production (mla_attention.py forward_impl / _v_up_proj, kv_b_proj BF16 unquantized -> W_UK_T / W_UV are views of it):
  W_UK:  torch.bmm(q_nope.transpose(0,1) [N,T,P], W_UK_T [N,P,L], out=[N,T,L])
  W_UV:  torch.bmm(attn_out.view(T,N,L).transpose(0,1) [N,T,L], W_UV [N,L,V], out=out.view(T,N,V).transpose(0,1))
"""
import sys, time, torch
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
N, P, L, V = 32, 256, 512, 256
Ts = [int(x) for x in (sys.argv[1:] or ["13824", "1791"])]
g = torch.Generator(device=dev).manual_seed(0)
w = (torch.randn(N * (P + V), L, device=dev, generator=g) * 0.03).to(torch.bfloat16)   # kv_b_proj.weight [out, in]
kvb = w.T.view(L, N, P + V)
W_UK, W_UV = kvb.split([P, V], dim=-1)
WUKT_prod = W_UK.permute(1, 2, 0)          # [N,P,L] strides (P+V)*L, L, 1
WUV_prod = W_UV.transpose(0, 1)            # [N,L,V] strides (P+V)*L, 1, L
print("W_UK_T strides", WUKT_prod.stride(), "W_UV strides", WUV_prod.stride(), flush=True)
WUKT_c = WUKT_prod.contiguous()
WUV_c = WUV_prod.contiguous()
WUV_cm = WUV_prod.transpose(1, 2).contiguous().transpose(1, 2)   # column-major per head, dense batch


def timeit(fn, it=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


def kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    return "; ".join(n[:70] for n in names)


for T in Ts:
    flop = 2 * N * T * P * L
    q = (torch.randn(T, N, P, device=dev, generator=g)).to(torch.bfloat16)
    a = (torch.randn(T, N, L, device=dev, generator=g)).to(torch.bfloat16)
    ref_uk = torch.einsum("tnp,npl->tnl", q.double(), WUKT_prod.double())
    ref_uv = torch.einsum("tnl,nlv->tnv", a.double(), WUV_prod.double())
    outs = {}

    def uk_prod():
        o = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
        torch.bmm(q.transpose(0, 1), WUKT_prod, out=o)
        outs["uk"] = o.transpose(0, 1)

    def uk_contigW():
        o = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
        torch.bmm(q.transpose(0, 1), WUKT_c, out=o)
        outs["uk"] = o.transpose(0, 1)

    def uk_perhead():
        o = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
        for h in range(N):
            torch.mm(q[:, h, :], WUKT_prod[h], out=o[h])
        outs["uk"] = o.transpose(0, 1)

    def uk_perhead_c():
        o = torch.empty(N, T, L, device=dev, dtype=torch.bfloat16)
        for h in range(N):
            torch.mm(q[:, h, :], WUKT_c[h], out=o[h])
        outs["uk"] = o.transpose(0, 1)

    def uk_perhead_tnl():   # output directly in [T,N,L]
        o = torch.empty(T, N, L, device=dev, dtype=torch.bfloat16)
        for h in range(N):
            torch.mm(q[:, h, :], WUKT_prod[h], out=o[:, h, :])
        outs["uk"] = o

    def uk_matmul_qcontig():
        qc = q.transpose(0, 1).contiguous()
        outs["uk"] = torch.bmm(qc, WUKT_c).transpose(0, 1)

    def uv_prod():
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        torch.bmm(a.transpose(0, 1), WUV_prod, out=o.transpose(0, 1))
        outs["uv"] = o

    def uv_contigW():
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        torch.bmm(a.transpose(0, 1), WUV_c, out=o.transpose(0, 1))
        outs["uv"] = o

    def uv_cmW():
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        torch.bmm(a.transpose(0, 1), WUV_cm, out=o.transpose(0, 1))
        outs["uv"] = o

    def uv_nbuf():   # contiguous [N,T,V] out then copy
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        t = torch.bmm(a.transpose(0, 1), WUV_prod)
        o.copy_(t.transpose(0, 1))
        outs["uv"] = o

    def uv_perhead():
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        for h in range(N):
            torch.mm(a[:, h, :], WUV_prod[h], out=o[:, h, :])
        outs["uv"] = o

    def uv_perhead_c():
        o = torch.empty(T, N, V, device=dev, dtype=torch.bfloat16)
        for h in range(N):
            torch.mm(a[:, h, :], WUV_c[h], out=o[:, h, :])
        outs["uv"] = o

    for name, fn, key, ref in [
        ("UK prod bmm", uk_prod, "uk", ref_uk), ("UK bmm contigW", uk_contigW, "uk", ref_uk),
        ("UK per-head mm", uk_perhead, "uk", ref_uk), ("UK per-head mm contigW", uk_perhead_c, "uk", ref_uk),
        ("UK per-head mm ->TNL", uk_perhead_tnl, "uk", ref_uk), ("UK q.contig+bmm", uk_matmul_qcontig, "uk", ref_uk),
        ("UV prod bmm", uv_prod, "uv", ref_uv), ("UV bmm contigW", uv_contigW, "uv", ref_uv),
        ("UV bmm colmajW", uv_cmW, "uv", ref_uv), ("UV bmm+copy", uv_nbuf, "uv", ref_uv),
        ("UV per-head mm", uv_perhead, "uv", ref_uv), ("UV per-head mm contigW", uv_perhead_c, "uv", ref_uv),
    ]:
        ms = timeit(fn)
        o = outs[key].double()
        rel = ((o - ref).norm() / ref.norm()).item()
        print(f"T={T:6d} {name:26s} {ms:7.3f} ms {flop / ms / 1e9:6.1f} TFLOPS relL2 {rel:.3e} | {kernels(fn)[:150]}",
              flush=True)
    del q, a, ref_uk, ref_uv
    torch.cuda.empty_cache()
