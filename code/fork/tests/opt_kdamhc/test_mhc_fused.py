"""opt-kdamhc: glm53_mhc_fused.py through the REAL MHCFusedPostPreOp (production image, nodeC GPU). Exit 1 on failure.

F0  a non-finite first call (the engine's dummy profile run) is served by production's op and does not uninstall
F1  per M (6,912 SP shard, 3,456 SP2 sub-chunk, 13,824 TP, 2,232 tail shard, 37 ragged): the hooked op's residual_cur
    is BITWISE production's; post/comb/layer_input vs production's (tf32) and vs the decode-branch arithmetic
    (mhc_fused_tilelang logits through the same pre_big_fuse): the decode-consistent mode must match the latter in
    >= 99.9 % of layer_input elements
F2  decode-sized (T = 8) calls stay on production's op (bitwise)
F3  the SP2 helper route (out= row slices) == the plain hooked call bitwise
F4  time of the whole op, production vs hooked (median of 15)
Run: GPU_RUN_RO=$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/overlay:$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/site \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/test_mhc_fused.py
"""
import os
import statistics
import sys

os.environ.setdefault("GLM53_MHC_FUSED_JIT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
os.environ.setdefault("GLM53_MHC_FUSED_MIN_T", "17")      # the test also drives the 37-row ragged case
import torch  # noqa: E402
from bench_mhc_halving import make  # noqa: E402

FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


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


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    import glm53_mhc_fused as F
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
    from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_tilelang, mhc_pre_big_fuse_with_norm_tilelang
    F.install()
    with set_current_vllm_config(VllmConfig()):
        op = MHCFusedPostPreOp()
    prod = F.STATE["orig"]
    H = 4096

    def args(w, M):
        return (w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0, 20,
                1, 1, w["norm"], 1e-5)

    # F0
    w = make(6912, seed=1)
    bad = dict(w)
    bad["res"] = w["res"].clone()
    bad["res"][5, 0, 7] = float("nan")
    r = op.forward_cuda(*args(bad, 6912))
    ref0 = prod(op, *args(bad, 6912))
    torch.cuda.synchronize()
    # opt-kdamhc-rev: the self-check runs on deterministic synthetic rows, so a non-finite first call (the engine's
    # dummy profile run) is checked and served like any other; its finite rows equal production's bit for bit
    fin = torch.isfinite(ref0[0])
    check(F.STATE["installed"] and F.STATE["checked"] and F.STATE["calls"] == 1
          and torch.equal(r[0][fin], ref0[0][fin]) and torch.equal(torch.isfinite(r[0]), fin),
          "F0 non-finite first call: self-checked on synthetic rows, served, residual_cur == production's (NaN pattern too)")
    for M in (6912, 3456, 13824, 2232, 37):
        w = make(M, seed=3 + M)
        a = args(w, M)
        got = op.forward_cuda(*a)
        ref = prod(op, *a)
        torch.cuda.synchronize()
        check(F.STATE["installed"] and F.STATE["checked"], f"F1 M={M}: hook installed and checked")
        check(torch.equal(got[0], ref[0]), f"F1 M={M}: residual_cur bitwise == production")
        # decode-branch arithmetic through the same pre_big_fuse
        g_d = torch.empty(1, M, 24, device="cuda")
        s_d = torch.empty(1, M, device="cuda")
        rc_d = torch.empty_like(w["res"])
        mhc_fused_tilelang(w["comb"].contiguous(), w["res"], w["post"].view(M, 4).contiguous(), w["x"],
                           w["fn"].view(24, 4, H), g_d, s_d, rc_d, 4, H, 24, tile_n=12, n_splits=1)
        o_d = (torch.empty(M, 4, device="cuda"), torch.empty(M, 16, device="cuda"),
               torch.empty(M, H, dtype=torch.bfloat16, device="cuda"))
        mhc_pre_big_fuse_with_norm_tilelang(g_d, s_d, w["scale"], w["base"], rc_d, o_d[0], o_d[1], o_d[2], w["norm"],
                                            H, 1e-5, 1e-6, 1e-6, 2.0, 20, 1e-5, 1, 4)
        li, li_p, li_d = got[3], ref[3], o_d[2]
        same_d = 100 * (li == li_d).float().mean().item()
        same_p = 100 * (li == li_p).float().mean().item()
        print(f"   M={M}: layer_input rel vs production {rel(li, li_p):.2e} ({same_p:.1f} % equal), vs decode branch "
              f"{rel(li, li_d):.2e} ({same_d:.2f} % equal); post max |d| vs prod "
              f"{(got[1].view(M, 4) - ref[1].view(M, 4)).abs().max().item():.1e}", flush=True)
        if F.STATE["round_a"] == 0:
            check(same_d >= 99.9, f"F1 M={M}: decode-consistent layer_input (>= 99.9 % equal to the decode branch)")
        check(rel(li, li_p) < 1e-2, f"F1 M={M}: layer_input within 1e-2 of production")
        if M == 3456:
            # F3 the SP2 route: out= row slices of a bigger buffer
            S = 2 * M
            buf = (torch.empty(S, 4, H, dtype=torch.bfloat16, device="cuda"), torch.empty(S, 4, device="cuda"),
                   torch.empty(S, 16, device="cuda"), torch.empty(S, H, dtype=torch.bfloat16, device="cuda"))
            out = tuple(t[M:] for t in buf)
            F.fused_post_pre(w["x"], w["res"], w["post"].view(M, 4), w["comb"], w["fn"], w["scale"], w["base"], 1e-5,
                             1e-6, 1e-6, 2.0, 20, w["norm"], 1e-5, out=out)
            torch.cuda.synchronize()
            check(torch.equal(out[0], got[0]) and torch.equal(out[3], got[3].view(M, H))
                  and torch.equal(out[1], got[1].view(M, 4)) and torch.equal(out[2], got[2].view(M, 16)),
                  "F3 SP2 route (out= row slices) == the hooked op bitwise")
        tp = tmed(lambda: prod(op, *a))
        tf = tmed(lambda: op.forward_cuda(*a))
        print(f"   F4 M={M}: whole op production {tp:.3f} ms, hooked {tf:.3f} ms ({tf - tp:+.3f} ms/call)", flush=True)
    # F2
    w = make(8, seed=99)
    a = args(w, 8)
    p0 = F.STATE["prod"]
    got = op.forward_cuda(*a)
    ref = prod(op, *a)
    check(F.STATE["prod"] == p0 + 1 and all(torch.equal(x, y) for x, y in zip(got, ref)),
          "F2 T=8 (decode): production's op, bitwise")
    print(f"RESULT: {'ALL OK' if not FAIL else f'{len(FAIL)} FAILED'}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
