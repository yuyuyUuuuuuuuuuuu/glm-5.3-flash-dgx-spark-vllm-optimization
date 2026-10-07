"""GLM53_DEC_SMALLOPS: paired A/B timing of each replaced op vs production's, inside CUDA graphs (docs/DEC_SMALLOPS.md).

Every variant is captured into its own CUDA graph that runs the op the way a decode step does (per-layer weights, so
the weights stream cold from DRAM: mHC 89 distinct [24, 16384] weights = one step's worth); the graphs are replayed
alternately in ROUNDS rounds (order flipped every round; other GPU clients share nodeC) and the per-call medians and
spreads ((max - min) / median over rounds) are reported. Run under the shared lock:
  flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/bench_smallops.py [dconv] [mhc] ...
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glm53_smallops as SO  # noqa: E402

ROUNDS = int(os.environ.get("BENCH_ROUNDS", "11"))
REPS = int(os.environ.get("BENCH_REPS", "20"))


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def ab(graphs: dict, calls: int):
    """graphs: name -> CUDAGraph (each replay = `calls` op calls). Returns name -> (median us/call, spread)."""
    names = list(graphs)
    for n in names:
        for _ in range(3):
            graphs[n].replay()
    torch.cuda.synchronize()
    res = {n: [] for n in names}
    for r in range(ROUNDS):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(REPS):
                graphs[n].replay()
            e.record()
            torch.cuda.synchronize()
            res[n].append(s.elapsed_time(e) * 1000.0 / (REPS * calls))
    out = {}
    for n, v in res.items():
        v = sorted(v)
        med = v[len(v) // 2]
        out[n] = (med, (v[-1] - v[0]) / med)
    return out


def report(title, r, base):
    b = r[base][0]
    print(f"== {title}")
    for n, (med, spr) in r.items():
        print(f"   {n:<44s} {med:8.2f} us/call  spread {spr * 100:5.1f}%  vs {base}: {b - med:+7.2f} us "
              f"({b / med:5.2f}x)", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def bench_dconv(dev):
    from vllm.model_executor.models import qwen3_dflash2 as D
    H, gs, taps, G = 4096, 16, 2, 256
    LAY = 20            # one decode step: 5 drafter layers x 2 convs x (prepare + finish)
    for T in (8, 16, 32):
        xs = [torch.randn(T, H, device=dev).bfloat16() for _ in range(LAY)]
        cs = [(torch.randn(T, 2, taps, G, device=dev) * 0.3).bfloat16() for _ in range(LAY)]
        bs = [(torch.randn(taps, H, device=dev) * 0.3).bfloat16() for _ in range(LAY)]
        outs = [None] * LAY

        def prod():
            for i in range(LAY):
                outs[i] = D._grouped_conv(xs[i], cs[i][:, 0], bs[i], 8, G, gs, taps)

        def new():
            for i in range(LAY):
                outs[i] = SO.dconv(xs[i], cs[i][:, 0], bs[i], gs, 8)

        r = ab({"production _grouped_conv (10 kernels)": graph_of(prod), "dconv (1 kernel)": graph_of(new)}, LAY)
        report(f"DFlash2 grouped conv, T={T} ({LAY} calls per graph)", r, "production _grouped_conv (10 kernels)")


def bench_mhc(dev):
    from vllm.model_executor.kernels.mhc import tilelang as W
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (mhc_fused_tilelang,
                                                                    mhc_pre_big_fuse_with_norm_tilelang)
    hc, H, n3 = 4, 4096, 24
    L = 89                       # fused post+pre calls per decode step (45 layers x 2 - layer 0's standalone pre)
    ws = [(torch.randn(n3, hc * H, device=dev) * 0.02).bfloat16().float() for _ in range(L)]
    wb = [w.bfloat16().contiguous() for w in ws]
    scale = torch.rand(3, device=dev)
    base = torch.randn(n3, device=dev) * 0.1
    nw = (torch.rand(H, device=dev) + 0.5).bfloat16()
    for M in (5, 6, 8, 16):
        S = SO.mhc_splits(M)
        tile_n = 2 if M < 8 else 3
        x = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
        res = (torch.randn(M, hc, H, device=dev)).bfloat16()
        post = torch.sigmoid(torch.randn(M, hc, 1, device=dev)) * 2
        comb = torch.rand(M, hc, hc, device=dev)
        comb = comb / comb.sum(-1, keepdim=True)
        yp = torch.empty(S, M, n3, device=dev)
        rp = torch.empty(S, M, device=dev)
        ro = torch.empty_like(res)
        pm = torch.empty(M, hc, device=dev)
        cm = torch.empty(M, hc * hc, device=dev)
        li = torch.empty(M, H, device=dev, dtype=torch.bfloat16)

        def op_prod():
            for i in range(L):
                W.mhc_fused_post_pre_tilelang(x, res, post, comb, ws[i], scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20,
                                              1, 1, nw, 1e-5)

        def k_prod():
            for i in range(L):
                mhc_fused_tilelang(comb.view(M, hc, hc), res, post.view(M, hc), x, ws[i].view(n3, hc, H), yp, rp, ro,
                                   hc, H, n3, tile_n=tile_n, n_splits=S)

        def k_new32():
            for i in range(L):
                SO.mhc_fused(comb.view(M, 16), post.view(M, hc), res, x, ws[i], S, yp, rp, ro)

        def k_new16():
            for i in range(L):
                SO.mhc_fused(comb.view(M, 16), post.view(M, hc), res, x, wb[i], S, yp, rp, ro)

        def k_pre():
            for i in range(L):
                mhc_pre_big_fuse_with_norm_tilelang(yp, rp, scale, base, ro, pm, cm, li, nw, H, 1e-5, 1e-6, 1e-6,
                                                    2.0, 20, 1e-5, S, hc)

        def k_prod_warm():
            for i in range(L):
                mhc_fused_tilelang(comb.view(M, hc, hc), res, post.view(M, hc), x, ws[0].view(n3, hc, H), yp, rp, ro,
                                   hc, H, n3, tile_n=tile_n, n_splits=S)

        def k_new32_warm():
            for i in range(L):
                SO.mhc_fused(comb.view(M, 16), post.view(M, hc), res, x, ws[0], S, yp, rp, ro)

        def k_new16_warm():
            for i in range(L):
                SO.mhc_fused(comb.view(M, 16), post.view(M, hc), res, x, wb[0], S, yp, rp, ro)

        gr = {"production mhc_fused_tilelang": graph_of(k_prod), "mhc_fused fp32 weight": graph_of(k_new32),
              "production mhc_fused_tilelang, L2-warm weight": graph_of(k_prod_warm),
              "mhc_fused fp32, L2-warm weight": graph_of(k_new32_warm),
              "mhc_fused bf16, L2-warm weight": graph_of(k_new16_warm),
              "mhc_fused bf16 weight copy": graph_of(k_new16), "production pre_big_fuse_with_norm": graph_of(k_pre),
              "production whole op (fused + pre)": graph_of(op_prod)}
        r = ab(gr, L)
        report(f"mHC fused post+pre, M={M} (splits {S}; {L} distinct weights per graph = one step)", r,
               "production mhc_fused_tilelang")


def sleep_cycles_for(us: float) -> int:
    """torch.cuda._sleep cycles that take ~us microseconds on this GPU (measured)."""
    n = 100000
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(n)
    s.record()
    torch.cuda._sleep(n)
    e.record()
    torch.cuda.synchronize()
    per_us = n / (s.elapsed_time(e) * 1000.0)
    return int(us * per_us)


def bench_mhc_situ(dev):
    """The mHC op as it sits in a decode step: DRAM-bound GEMV (evict_first weights, like o_proj / the MoE) ->
    all-reduce (stand-in: a ~20 us spin on one SM = production's p10 AR, DRAM idle) -> mHC fused post+pre.
    Variants: production op / smallops fused kernel, each without and with the L2 prefetch of the layer's mHC weight
    forked onto a side stream right before the all-reduce."""
    from vllm.model_executor.kernels.mhc import tilelang as W
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_pre_big_fuse_with_norm_tilelang
    import glm53_bf16_gemv as G
    hc, H, n3 = 4, 4096, 24
    L = 89
    ws = [(torch.randn(n3, hc * H, device=dev) * 0.02).bfloat16().float() for _ in range(L)]
    gw = [(torch.randn(1024, 4096, device=dev) * 0.02).bfloat16() for _ in range(8)]     # 8 MiB each, 64 MiB > L2
    ctx = G.GemmCtx(1024, 4096, dev, max_splits=8)
    scale = torch.rand(3, device=dev)
    base = torch.randn(n3, device=dev) * 0.1
    nw = (torch.rand(H, device=dev) + 0.5).bfloat16()
    ar_us = float(os.environ.get("BENCH_AR_US", "20"))
    cyc = sleep_cycles_for(ar_us)
    side = torch.cuda.Stream()
    for M in (5, 8):
        S = SO.mhc_splits(M)
        x = (torch.randn(M, H, device=dev) * 0.5).bfloat16()
        gx = torch.randn(M, 4096, device=dev).bfloat16()
        res = (torch.randn(M, hc, H, device=dev)).bfloat16()
        post = torch.sigmoid(torch.randn(M, hc, 1, device=dev)) * 2
        comb = torch.rand(M, hc, hc, device=dev)
        comb = comb / comb.sum(-1, keepdim=True)
        yp = torch.empty(S, M, n3, device=dev)
        rp = torch.empty(S, M, device=dev)
        ro = torch.empty_like(res)
        pm = torch.empty(M, hc, device=dev)
        cm = torch.empty(M, hc * hc, device=dev)
        li = torch.empty(M, H, device=dev, dtype=torch.bfloat16)

        def mk(use_new, pf):
            def run():
                for i in range(L):
                    G.gemm(gx, gw[i % 8], ctx)
                    cur = torch.cuda.current_stream()
                    if pf:
                        side.wait_stream(cur)
                        with torch.cuda.stream(side):
                            SO.l2_prefetch(ws[i])
                    torch.cuda._sleep(cyc)
                    if pf:
                        cur.wait_stream(side)
                    if use_new:
                        SO.mhc_fused(comb.view(M, 16), post.view(M, hc), res, x, ws[i], S, yp, rp, ro)
                        mhc_pre_big_fuse_with_norm_tilelang(yp, rp, scale, base, ro, pm, cm, li, nw, H, 1e-5, 1e-6,
                                                            1e-6, 2.0, 20, 1e-5, S, hc)
                    else:
                        W.mhc_fused_post_pre_tilelang(x, res, post, comb, ws[i], scale, base, 1e-5, 1e-6, 1e-6, 2.0,
                                                      20, 1, 1, nw, 1e-5)
            return run

        def base_only():
            for i in range(L):
                G.gemm(gx, gw[i % 8], ctx)
                torch.cuda._sleep(cyc)

        gr = {"production op": graph_of(mk(False, False)), "production op + L2 prefetch": graph_of(mk(False, True)),
              "smallops fused + prod pre": graph_of(mk(True, False)),
              "smallops fused + prod pre + L2 prefetch": graph_of(mk(True, True)),
              "(GEMV + AR stand-in only)": graph_of(base_only)}
        r = ab(gr, L)
        report(f"mHC in a decode-like sequence, M={M}: GEMV 8 MiB -> AR stand-in {ar_us:.0f} us -> mHC op "
               f"(us per layer incl. GEMV + AR)", r, "production op")


def bench_tc(dev):
    """MLA decode BF16 GEMMs (11 layers' distinct weights per graph = one step's worth, cold from DRAM):
    production cuBLAS op vs tc_gemm configs (warps * 100 + stages)."""
    import torch.nn.functional as F
    L = 11
    cfgs = [int(c) for c in os.environ.get("BENCH_TC_CFGS", "104,106,108,204,206,208,404,406,408").split(",")]
    Ms = [int(m) for m in os.environ.get("BENCH_TC_MS", "5,6,8,16").split(",")]
    wq = [(torch.randn(4096, 1536, device=dev) * 0.02).bfloat16() for _ in range(L)]
    kvb = [(torch.randn(16384, 512, device=dev) * 0.02).bfloat16() for _ in range(L)]
    uk, uv = [], []
    for w in kvb:
        v = w.T.view(512, 32, 512)
        a, b = v.split([256, 256], dim=-1)
        uk.append(a.permute(1, 2, 0))
        uv.append(b.transpose(0, 1))
    LD = 3
    wd = [(torch.randn(5120, 4096, device=dev) * 0.02).bfloat16() for _ in range(LD)]
    for M in Ms:
        # indexer wq_b
        x = torch.randn(M, 1536, device=dev).bfloat16()
        y = torch.empty(M, 4096, device=dev, dtype=torch.bfloat16)

        def p_wq():
            for i in range(L):
                F.linear(x, wq[i])

        def t_wq(cfg):
            def f():
                for i in range(L):
                    SO.tc_gemm(wq[i].unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0), cfg)
            return f
        gr = {"production F.linear (cuBLAS)": graph_of(p_wq)}
        gr.update({f"tc_gemm cfg {c}": graph_of(t_wq(c)) for c in cfgs})
        report(f"indexer wq_b [4096, 1536] M={M} ({L} layers per graph, 12.6 MB each; floor at 250 GB/s 50.3 us)",
               ab(gr, L), "production F.linear (cuBLAS)")
        # W_UK
        q = torch.randn(M, 32 * 256, device=dev).bfloat16().view(M, 32, 256)
        qn = q.transpose(0, 1)
        o = qn.new_empty((32, M, 512))

        def p_uk():
            for i in range(L):
                torch.bmm(qn, uk[i], out=o)

        def t_uk(cfg):
            def f():
                for i in range(L):
                    SO.tc_gemm(uk[i].transpose(1, 2), qn, o, cfg)
            return f
        gr = {"production torch.bmm (cuBLAS)": graph_of(p_uk)}
        gr.update({f"tc_gemm cfg {c}": graph_of(t_uk(c)) for c in cfgs})
        report(f"MLA W_UK_T bmm (32, 256->512) M={M} (8 MB each; floor 33.6 us)", ab(gr, L),
               "production torch.bmm (cuBLAS)")
        # W_UV
        at = torch.randn(M, 32 * 512, device=dev).bfloat16()
        xv = at.view(-1, 32, 512).transpose(0, 1)
        ob = torch.empty(M, 32 * 256, device=dev, dtype=torch.bfloat16)
        ov = ob.view(-1, 32, 256).transpose(0, 1)

        def p_uv():
            for i in range(L):
                torch.bmm(xv, uv[i], out=ov)

        def t_uv(cfg):
            def f():
                for i in range(L):
                    SO.tc_gemm(uv[i].transpose(1, 2), xv, ov, cfg)
            return f
        gr = {"production torch.bmm (cuBLAS)": graph_of(p_uv)}
        gr.update({f"tc_gemm cfg {c}": graph_of(t_uv(c)) for c in cfgs})
        report(f"MLA W_UV bmm (32, 512->256) M={M} (8 MB each; floor 33.6 us)", ab(gr, L),
               "production torch.bmm (cuBLAS)")
        # drafter context K/V projection shape
        xd = torch.randn(M, 4096, device=dev).bfloat16()
        yd = torch.empty(M, 5120, device=dev, dtype=torch.bfloat16)

        def p_d():
            for i in range(LD):
                F.linear(xd, wd[i])

        def t_d(cfg):
            def f():
                for i in range(LD):
                    SO.tc_gemm(wd[i].unsqueeze(0), xd.unsqueeze(0), yd.unsqueeze(0), cfg)
            return f
        gr = {"production F.linear (cuBLAS)": graph_of(p_d)}
        gr.update({f"tc_gemm cfg {c}": graph_of(t_d(c)) for c in cfgs})
        report(f"[5120, 4096] linear M={M} (42 MB each; floor 168 us)", ab(gr, LD), "production F.linear (cuBLAS)")


def main():
    dev = torch.device("cuda", 0)
    SO.load_ext()
    which = set(sys.argv[1:]) or {"dconv", "mhc"}
    print(f"ext {SO.EXT_SOURCE}; rounds {ROUNDS} x {REPS} replays; {torch.cuda.get_device_name(0)}", flush=True)
    if "dconv" in which:
        bench_dconv(dev)
    if "mhc" in which:
        bench_mhc(dev)
    if "mhcsitu" in which:
        bench_mhc_situ(dev)
    if "tc" in which:
        bench_tc(dev)


if __name__ == "__main__":
    main()
