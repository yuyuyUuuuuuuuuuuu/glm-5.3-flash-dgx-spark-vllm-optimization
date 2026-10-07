"""nodeC micro-benchmark: smallops mhc_fused bf16-weight variants (cold weights: 89 distinct per graph) vs production's
mhc_fused_tilelang, M = 5, 6, 8, 16; also checks each variant bitwise against production on real-shaped data.
mode 0 = shipped staged kernel, mode NB = DIRECT (no shared-memory staging) with NB outputs per CTA."""
import hashlib, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.cpp_extension import load
from tf_exl3_moe import _cuda_include_shim
from bench_smallops import graph_of, ab
from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_tilelang
here = os.path.dirname(os.path.abspath(__file__))
tag = hashlib.sha256(open(f"{here}/mb_smallops.cu", "rb").read() + open(f"{here}/../kernels/smallops_kernels.cuh", "rb").read()).hexdigest()[:10]
inc = _cuda_include_shim()
E = load(name=f"mb_smallops_{tag}", sources=[f"{here}/mb_smallops.cu"], extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc])
dev = torch.device("cuda")
hc, H, n3, L = 4, 4096, 24, 89
ws = [(torch.randn(n3, hc * H, device=dev) * 0.02).bfloat16().float() for _ in range(L)]
wb = [w.bfloat16() for w in ws]
modes = [int(m) for m in os.environ.get("MB_MODES", "0,1,2,3,4").split(",")]
for M in [int(m) for m in os.environ.get("MB_MS", "5,6,8,16").split(",")]:
    S = 8 if M < 8 else 4
    tile_n = 2 if M < 8 else 3
    x = torch.randn(M, H, device=dev).bfloat16(); res = torch.randn(M, hc, H, device=dev).bfloat16()
    post = torch.rand(M, hc, device=dev); comb = torch.rand(M, 16, device=dev)
    ro = torch.empty_like(res)
    yp = torch.empty(S, M, n3, device=dev); rp = torch.empty(S, M, device=dev)
    # bitwise check of every mode vs production (weight 0)
    yr, rr, ror = torch.empty_like(yp), torch.empty_like(rp), torch.empty_like(res)
    mhc_fused_tilelang(comb.view(M, hc, hc), res, post, x, ws[0].view(n3, hc, H), yr, rr, ror, hc, H, n3,
                       tile_n=tile_n, n_splits=S)
    for mode in modes:
        E.mhc_fused_var(comb, post, res, x, wb[0], yp, rp, ro, mode)
        torch.cuda.synchronize()
        ok = torch.equal(yp.view(torch.int32), yr.view(torch.int32)) and torch.equal(rp.view(torch.int32), rr.view(torch.int32)) \
            and torch.equal(ro.view(torch.int16), ror.view(torch.int16))
        print(f"M={M} mode {mode}: bitwise {'OK' if ok else 'MISMATCH'}", flush=True)
    g = {}

    def prod():
        for i in range(L):
            mhc_fused_tilelang(comb.view(M, hc, hc), res, post, x, ws[i].view(n3, hc, H), yp, rp, ro, hc, H, n3,
                               tile_n=tile_n, n_splits=S)
    g["production mhc_fused_tilelang"] = graph_of(prod)
    for mode in modes:
        def f(mode=mode):
            for i in range(L):
                E.mhc_fused_var(comb, post, res, x, wb[i], yp, rp, ro, mode)
        g["shipped (staged)" if mode == 0 else (f"DIRECT NB={mode}" if mode < 5 else f"staged MC=8 NB={ {5: 2, 6: 3, 7: 1}[mode] }")] = graph_of(f)
    r = ab(g, L)
    print(f"== M={M} (S={S}), cold bf16 weights (89 distinct), us per call")
    for n, (m, sp) in r.items():
        print(f"   {n:<32s} {m:6.2f} us  spread {sp*100:4.1f}%", flush=True)
