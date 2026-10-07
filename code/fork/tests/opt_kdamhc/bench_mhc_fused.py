"""opt-kdamhc: can the mHC post + prenorm GEMM be one pass at prefill sizes? (nodeC, production kernels)

Production's mhc_fused_post_pre_tilelang (vllm/model_executor/kernels/mhc/tilelang.py) has two branches:
  num_tokens <= 16 (decode): mhc_fused_tilelang = post-mapping + the prenorm dot products in ONE kernel (fp32 FMA on the
                             fp32 new residual, fn in fp32), then pre_big_fuse;
  num_tokens  > 16 (prefill): mhc_post_tilelang (writes residual_cur) -> deep_gemm tf32_hc_prenorm_gemm (RE-READS
                             residual_cur: 16 KB/token x 4 streams) -> pre_big_fuse (reads it again).
Under GLM53_MHC_SP a 13,824-token chunk runs 90 calls at 6,912 rows; the GEMM pass is ~1.1 ms of the ~5.0 ms per call.
This bench times, per M: the production prefill branch, and the decode branch's fused kernel at prefill M with several
(tile_n, n_splits); and reports how far the fused branch's outputs (post_mix / comb_mix / layer_input) are from the
prefill branch's (they are a different arithmetic: fp32 FMA on fp32 new_r vs tf32 MMA on bf16-rounded residual_cur).

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/bench_mhc_fused.py
"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mhc_sp"))
from bench_mhc_halving import make  # noqa: E402

H, NS = 4096, 4
H3 = 2 * NS + NS * NS


def tmed(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split, mhc_fused_tilelang, mhc_post_tilelang, mhc_pre_big_fuse_with_norm_tilelang)
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv
    print(f"gpu {torch.cuda.get_device_name(0)}, SMs {torch.cuda.get_device_properties(0).multi_processor_count}")
    for M in [int(v) for v in os.environ.get("MS", "6912,3456,13824,2232").split(",")]:
        w = make(M, seed=3)
        x, res, post, comb, fn = w["x"], w["res"], w["post"].view(M, NS), w["comb"], w["fn"]
        dev = x.device

        def outs(ns):
            return (torch.empty(ns, M, H3, dtype=torch.float32, device=dev), torch.empty(ns, M, dtype=torch.float32, device=dev))

        rc = torch.empty_like(res)
        pm_c = torch.empty(M, NS, dtype=torch.float32, device=dev)
        cm_c = torch.empty(M, NS * NS, dtype=torch.float32, device=dev)
        li = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        ns_p = compute_num_split(64, NS * H, cdiv(M, 64))
        g_mul, g_sq = outs(ns_p)

        def post_k():
            mhc_post_tilelang(comb, res, post, x, rc, NS, H)

        def gemm_k():
            tf32_hc_prenorm_gemm(rc.view(M, NS * H), fn, g_mul, g_sq, ns_p)

        def pre_k(gm, gs, nsp, o=(pm_c, cm_c, li)):
            mhc_pre_big_fuse_with_norm_tilelang(gm, gs, w["scale"], w["base"], rc, o[0], o[1], o[2], w["norm"], H,
                                                1e-5, 1e-6, 1e-6, 2.0, 20, 1e-5, nsp, NS)

        tp, tg, tb = tmed(post_k), tmed(gemm_k), tmed(lambda: pre_k(g_mul, g_sq, ns_p))
        post_k(); gemm_k(); pre_k(g_mul, g_sq, ns_p); torch.cuda.synchronize()
        ref = (rc.clone(), pm_c.clone(), cm_c.clone(), li.clone(), g_mul.sum(0).clone())
        print(f"M={M}: production prefill branch post {tp:.3f} + tf32 GEMM {tg:.3f} (n_splits {ns_p}) + pre_big_fuse "
              f"{tb:.3f} = {tp + tg + tb:.3f} ms", flush=True)
        for tile_n, nsp in ((24, 1), (12, 1), (8, 1), (6, 1), (24, 2), (12, 2), (8, 2), (3, 4)):
            rc2 = torch.empty_like(res)
            gm2, gs2 = outs(nsp)
            try:
                def fk():
                    mhc_fused_tilelang(comb, res, post, x, fn.view(H3, NS, H), gm2, gs2, rc2, NS, H, H3,
                                       tile_n=tile_n, n_splits=nsp)
                tf = tmed(fk, 7)
            except Exception as exc:  # noqa: BLE001
                print(f"   fused tile_n={tile_n} n_splits={nsp}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            fk(); torch.cuda.synchronize()
            o2 = (torch.empty_like(pm_c), torch.empty_like(cm_c), torch.empty_like(li))
            rc_save = rc.clone()
            rc.copy_(rc2)
            pre_k(gm2, gs2, nsp, o2); torch.cuda.synchronize()
            rc.copy_(rc_save)
            same_r = torch.equal(rc2, ref[0])
            dmul = ((gm2.sum(0) - ref[4]).norm() / ref[4].norm()).item()
            dli = ((o2[2].float() - ref[3].float()).norm() / ref[3].float().norm()).item()
            dpm = ((o2[0] - ref[1]).abs().max()).item()
            dcm = ((o2[1] - ref[2]).abs().max()).item()
            neq = (o2[2] != ref[3]).float().mean().item()
            print(f"   fused tile_n={tile_n:2d} n_splits={nsp}: {tf:.3f} ms (+pre {tb:.3f} = {tf + tb:.3f}; saves "
                  f"{tp + tg - tf:+.3f} ms/call); residual_cur bitwise {same_r}; gemm rel {dmul:.2e}; post_mix max abs "
                  f"{dpm:.2e}; comb_mix max abs {dcm:.2e}; layer_input rel {dli:.2e} ({100 * neq:.2f}% elems differ)",
                  flush=True)
        del w, rc, li


if __name__ == "__main__":
    main()
