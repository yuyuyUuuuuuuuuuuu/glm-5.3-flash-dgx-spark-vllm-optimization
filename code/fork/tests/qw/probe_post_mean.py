"""Probe: can a Triton kernel reproduce the tilelang mhc_post output bits and aten's mean(dim=1) bits?"""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/w")
DEV = "cuda"


@triton.jit
def _post_j(A, C, n, j: tl.constexpr, d, b0, b1, b2, b3, FMA: tl.constexpr):
    cj = tl.load(C + n * 4 + j)
    a0 = tl.load(A + n * 16 + 0 * 4 + j)
    a1 = tl.load(A + n * 16 + 1 * 4 + j)
    a2 = tl.load(A + n * 16 + 2 * 4 + j)
    a3 = tl.load(A + n * 16 + 3 * 4 + j)
    if FMA == 2:        # nvcc contracted c*d + a0*b0 as fma(c, d, a0*b0), then the fma chain
        x = tl.fma(cj, d, a0 * b0)
        x = tl.fma(a1, b1, x)
        x = tl.fma(a2, b2, x)
        x = tl.fma(a3, b3, x)
    elif FMA == 1:
        x = cj * d
        x = tl.fma(a0, b0, x)
        x = tl.fma(a1, b1, x)
        x = tl.fma(a2, b2, x)
        x = tl.fma(a3, b3, x)
    else:
        x = cj * d
        x = x + a0 * b0
        x = x + a1 * b1
        x = x + a2 * b2
        x = x + a3 * b3
    return x.to(tl.bfloat16)


@triton.jit
def post_mean_kernel(A, Bres, C, D, OUT, MEAN, H: tl.constexpr, BH: tl.constexpr, FMA: tl.constexpr,
                     MORDER: tl.constexpr, WRITE_FULL: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    offs = hb * BH + tl.arange(0, BH)
    d = tl.load(D + n * H + offs).to(tl.float32)
    b0 = tl.load(Bres + n * 4 * H + 0 * H + offs).to(tl.float32)
    b1 = tl.load(Bres + n * 4 * H + 1 * H + offs).to(tl.float32)
    b2 = tl.load(Bres + n * 4 * H + 2 * H + offs).to(tl.float32)
    b3 = tl.load(Bres + n * 4 * H + 3 * H + offs).to(tl.float32)
    o0 = _post_j(A, C, n, 0, d, b0, b1, b2, b3, FMA)
    o1 = _post_j(A, C, n, 1, d, b0, b1, b2, b3, FMA)
    o2 = _post_j(A, C, n, 2, d, b0, b1, b2, b3, FMA)
    o3 = _post_j(A, C, n, 3, d, b0, b1, b2, b3, FMA)
    if WRITE_FULL:
        tl.store(OUT + n * 4 * H + 0 * H + offs, o0)
        tl.store(OUT + n * 4 * H + 1 * H + offs, o1)
        tl.store(OUT + n * 4 * H + 2 * H + offs, o2)
        tl.store(OUT + n * 4 * H + 3 * H + offs, o3)
    f0 = o0.to(tl.float32)
    f1 = o1.to(tl.float32)
    f2 = o2.to(tl.float32)
    f3 = o3.to(tl.float32)
    if MORDER == 0:
        s = ((f0 + f1) + f2) + f3
    elif MORDER == 1:
        s = (f0 + f2) + (f1 + f3)
    else:
        s = (f0 + f1) + (f2 + f3)
    tl.store(MEAN + n * H + offs, (s * 0.25).to(tl.bfloat16))


@triton.jit
def mean_kernel(X, MEAN, H: tl.constexpr, BH: tl.constexpr, MORDER: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    f0 = tl.load(X + n * 4 * H + 0 * H + offs).to(tl.float32)
    f1 = tl.load(X + n * 4 * H + 1 * H + offs).to(tl.float32)
    f2 = tl.load(X + n * 4 * H + 2 * H + offs).to(tl.float32)
    f3 = tl.load(X + n * 4 * H + 3 * H + offs).to(tl.float32)
    if MORDER == 0:
        s = ((f0 + f1) + f2) + f3
    elif MORDER == 1:
        s = (f0 + f2) + (f1 + f3)
    else:
        s = (f0 + f1) + (f2 + f3)
    tl.store(MEAN + n * H + offs, (s * 0.25).to(tl.bfloat16))


def main():
    import vllm.model_executor.kernels.mhc.tilelang as TL
    H = 4096
    for T in (1791, 13824):
        g = torch.Generator(device=DEV).manual_seed(T)
        x = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
        # wide dynamic range: streams with exponents spread over ~2^-12..2^12
        res = (torch.randn(T, 4, H, device=DEV, generator=g) * torch.exp2(torch.randint(-12, 12, (T, 4, 1), device=DEV,
                                                                                         generator=g).float())).to(torch.bfloat16)
        post = torch.rand(T, 4, 1, device=DEV, generator=g) * 2
        comb = torch.softmax(torch.randn(T, 4, 4, device=DEV, generator=g) * 3, -1)
        ref = TL.mhc_post_tilelang(x, res, post, comb)
        mref = ref.mean(dim=1)
        for mo in (0, 1, 2):
            m2 = torch.empty(T, H, device=DEV, dtype=torch.bfloat16)
            mean_kernel[(T, H // 1024)](ref, m2, H=H, BH=1024, MORDER=mo, num_warps=4)
            print(f"T={T} mean order {mo} on production's post output: equal to aten mean {torch.equal(m2, mref)} "
                  f"(diff {(m2 != mref).sum().item()})", flush=True)
        for fma in (2, 1, 0):
            for mo in (0,):
                out = torch.empty_like(ref)
                mean = torch.empty(T, H, device=DEV, dtype=torch.bfloat16)
                post_mean_kernel[(T, H // 1024)](comb, res, post.view(T, 4), x, out, mean, H=H, BH=1024, FMA=fma,
                                                 MORDER=mo, WRITE_FULL=True, num_warps=4)
                print(f"T={T} fma={fma} morder={mo}: post bits equal {torch.equal(out, ref)} "
                      f"(diff {(out != ref).sum().item()}), mean bits equal {torch.equal(mean, mref)} "
                      f"(diff {(mean != mref).sum().item()}); mean-of-ref equal "
                      f"{torch.equal(mean, ref.mean(dim=1))}", flush=True)
        # timing: tilelang post + aten mean vs fused (write full + mean) vs mean only
        def t(fn, it=20):
            for _ in range(3): fn()
            torch.cuda.synchronize()
            e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
            for a, b in e:
                a.record(); fn(); b.record()
            torch.cuda.synchronize()
            v = sorted(a.elapsed_time(b) for a, b in e)
            return v[len(v) // 2]
        out = torch.empty_like(ref); mean = torch.empty(T, H, device=DEV, dtype=torch.bfloat16)
        for bh, nw in ((512, 4), (1024, 4), (1024, 8), (2048, 8), (4096, 8)):
            tf = t(lambda: post_mean_kernel[(T, H // bh)](comb, res, post.view(T, 4), x, out, mean, H=H, BH=bh, FMA=2,
                                                           MORDER=0, WRITE_FULL=True, num_warps=nw))
            tm = t(lambda: post_mean_kernel[(T, H // bh)](comb, res, post.view(T, 4), x, out, mean, H=H, BH=bh, FMA=2,
                                                           MORDER=0, WRITE_FULL=False, num_warps=nw))
            print(f"T={T} BH={bh} warps={nw}: fused post+mean {tf:.3f} ms, mean only {tm:.3f} ms", flush=True)
        tp = t(lambda: TL.mhc_post_tilelang(x, res, post, comb))
        tpm = t(lambda: TL.mhc_post_tilelang(x, res, post, comb).mean(dim=1))
        print(f"T={T} tilelang post {tp:.3f} ms, post + aten mean {tpm:.3f} ms", flush=True)


main()
