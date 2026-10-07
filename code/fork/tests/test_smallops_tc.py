"""GLM53_DEC_SMALLOPS tc_gemm vs the production BF16 GEMMs of the MLA decode path: bitwise equality per M.

  wq_b    indexer wq_b: F.linear(qr [M, 1536], W [4096, 1536])                     (cuBLAS 16x16_128x1_tn)
  W_UK    torch.bmm(q_nope (32, M, 256) view, W_UK_T (32, 256, 512) view, out=)   (cuBLAS 32x32_64x2_nn)
  W_UV    torch.bmm(x (32, M, 512) view, W_UV (32, 512, 256) view, out=view)      (cuBLAS 32x32_128x2_tn)
  draft   F.linear(x [M, 4096], W [5120, 4096])  (drafter context K/V projection shape, 16x16_128x1_tn)
Weights: real layer-11 / layer-45 wq_b and kv_b_proj (TP rank 0 rows) from GLM53_CKPT, laid out exactly as
MLAAttention.process_weights_after_loading does (views of kv_b_proj.weight.T); random bf16 for the drafter shape.
Also prints which kernel cuBLAS runs per M (torch.profiler), so a mismatch can be tied to a kernel switch.
Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4 TF_EXL3_JIT=1 tests/gpu_run.sh \
     python3 tests/test_smallops_tc.py [cfg ...]
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm53_smallops as SO  # noqa: E402
from test_smallops_kernels import ckpt_tensors  # noqa: E402

FAILS = []
MS = list(range(1, 17))


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def mla_views(kvb: torch.Tensor, H=32, P=256, V=256, L=512):
    """kv_b_proj weight rows of TP rank 0 [H*(P+V), L] -> (W_UK_T, W_UV) exactly as production builds them."""
    w = kvb.T                                     # get_and_maybe_dequant_weights(...).T
    w = w.view(L, H, P + V)
    W_UK, W_UV = w.split([P, V], dim=-1)
    return W_UK.permute(1, 2, 0), W_UV.transpose(0, 1)


def cublas_kernel(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    names = [n for n in names if "elementwise" not in n and "Memset" not in n]
    return "; ".join(sorted(set(n[:90] for n in names))) or "?"


def cases(dev, real):
    out = []
    gen = torch.Generator(device=dev).manual_seed(3)
    for lay in (11, 45):
        wq = real.get(f"model.language_model.layers.{lay}.self_attn.indexer.wq_b.weight")
        kvb = real.get(f"model.language_model.layers.{lay}.self_attn.kv_b_proj.weight")
        if wq is not None:
            wq = wq.to(dev)

            def mk_wq(M, wq=wq):
                x = (torch.randn(M, 1536, generator=gen, device=dev)).bfloat16()
                ref = lambda: F.linear(x, wq)                                   # noqa: E731
                y = torch.empty(M, 4096, device=dev, dtype=torch.bfloat16)
                mine = lambda cfg: (SO.tc_gemm(wq.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0), cfg), y)[1]  # noqa
                return ref, mine
            out.append((f"wq_b L{lay}", mk_wq))
        if kvb is not None:
            W_UK_T, W_UV = mla_views(kvb[:16384].to(dev).contiguous())

            def mk_uk(M, W_UK_T=W_UK_T):
                q = (torch.randn(M, 32 * 256, generator=gen, device=dev) * 0.5).bfloat16().view(M, 32, 256)
                qn = q.transpose(0, 1)                                          # (N, B, P)
                ref_out = qn.new_empty((32, M, 512))

                def ref():
                    torch.bmm(qn, W_UK_T, out=ref_out)
                    return ref_out
                y = qn.new_empty((32, M, 512))
                mine = lambda cfg: (SO.tc_gemm(W_UK_T.transpose(1, 2), qn, y, cfg), y)[1]  # noqa: E731
                return ref, mine

            def mk_uv(M, W_UV=W_UV):
                a = (torch.randn(M, 32 * 512, generator=gen, device=dev) * 0.1).bfloat16()
                x = a.view(-1, 32, 512).transpose(0, 1)
                ref_buf = torch.empty(M, 32 * 256, device=dev, dtype=torch.bfloat16)

                def ref():
                    o = ref_buf.view(-1, 32, 256)
                    torch.bmm(x, W_UV, out=o.transpose(0, 1))
                    return ref_buf
                buf = torch.empty(M, 32 * 256, device=dev, dtype=torch.bfloat16)

                def mine(cfg):
                    SO.tc_gemm(W_UV.transpose(1, 2), x, buf.view(-1, 32, 256).transpose(0, 1), cfg)
                    return buf
                return ref, mine
            out.append((f"W_UK L{lay}", mk_uk))
            out.append((f"W_UV L{lay}", mk_uv))
    wd = (torch.randn(5120, 4096, generator=gen, device=dev) * 0.02).bfloat16()

    def mk_d(M):
        x = torch.randn(M, 4096, generator=gen, device=dev).bfloat16()
        y = torch.empty(M, 5120, device=dev, dtype=torch.bfloat16)
        return (lambda: F.linear(x, wd)), (lambda cfg: (SO.tc_gemm(wd.unsqueeze(0), x.unsqueeze(0), y.unsqueeze(0),
                                                                   cfg), y)[1])
    out.append(("draft ctx-kv [5120,4096] random", mk_d))
    return out


def main():
    dev = torch.device("cuda", 0)
    SO.load_ext()
    print(f"ext {SO.EXT_SOURCE}; torch {torch.__version__}")
    cfgs = [int(a) for a in sys.argv[1:]] or [SO.TC_CFG_DEFAULT]
    names = [f"model.language_model.layers.{l}.self_attn.{t}" for l in (11, 45)
             for t in ("indexer.wq_b.weight", "kv_b_proj.weight")]
    real = ckpt_tensors(names)
    print(f"real tensors: {sorted(real)}")
    for name, mk in cases(dev, real):
        bad = []
        gemv = []
        kern = {}
        for M in MS:
            for trial in range(2):
                ref_fn, mine = mk(M)
                r = ref_fn().clone()
                if trial == 0:
                    kern[M] = cublas_kernel(ref_fn)
                for cfg in cfgs:
                    got = mine(cfg)
                    torch.cuda.synchronize()
                    if not torch.equal(r.view(torch.int16), got.view(torch.int16)):
                        d = (r.float() - got.float())
                        nd = int((r.view(torch.int16) != got.view(torch.int16)).sum())
                        rel = float(d.norm() / r.float().norm())
                        msg = f"M={M} cfg={cfg}: {nd}/{r.numel()} differ, rel_l2 {rel:.2e}"
                        # deploy-r16: where cuBLAS itself runs a SIMT gemv (F.linear at M = 1) there is no HMMA chain
                        # to reproduce; DEC_SMALLOPS.md §6 claims bitwise equality only against the wmma tensor-op
                        # kernels (wq_b M 2..16, W_UK / W_UV M 1..16). tc_gemm is not wired into production.
                        (gemv if "gemv" in kern[M] else bad).append(msg)
        byk = {}
        for M, k in kern.items():
            byk.setdefault(k, []).append(M)
        for k, ms in byk.items():
            print(f"   {name}: cuBLAS M={ms}: {k}")
        tc_ms = [M for M in MS if "gemv" not in kern[M]]
        check(not bad, f"tc_gemm {name} bitwise == production where cuBLAS runs a tensor-op kernel, M {tc_ms}, "
              f"cfgs {cfgs}" + ("" if not bad else f": {len(bad)} mismatches, first {bad[:4]}"))
        if gemv:
            print(f"INFO tc_gemm {name}: cuBLAS runs a SIMT gemv at M {[M for M in MS if 'gemv' in kern[M]]} "
                  f"(not an HMMA chain; not claimed bitwise): {gemv[:2]}")
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
