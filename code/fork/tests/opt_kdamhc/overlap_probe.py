"""opt-kdamhc: can an NCCL-like side-stream transfer hide behind COMPUTE-bound prefill kernels? (nodeC, one GB10)

tests/mhc2/contention_probe.py showed that a few-CTA copy (the stand-in for NCCL's wire-limited LL channels, 32-93 GB/s
alone) is starved while the DRAM-bound mHC runs (hidden 8-10 % of the shorter). That bounds GLM53_MHC_SP2's overlap.
This probe asks the same question for the compute-bound kernels that a deeper overlap would pair the collectives with:
  W8A8 custom CUTLASS GEMM (KDA in_proj [12576 x 4096], M = 13,824 / 6,912; persistent), Marlin W8A16 (shared-expert
  gate_up [2048 x 4096], M = 13,824; persistent), and the mHC sub-chunk again as the reference.
Side stream (priority -1): a Triton copy kernel with C CTAs streaming 3 x 28.3 MB (an RS/AG half-slice's DRAM footprint).
Reports: main kernel alone / concurrent, copy alone, both together vs serial, and the hidden share of the shorter.

Run: GPU_RUN_RO=$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/overlay:$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/site \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/overlap_probe.py
"""
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "mhc_sp")):
    sys.path.insert(0, p)
sys.path.append(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2/overlay"))
sys.path.append(os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2/site"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _copy(src, dst, n, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for it in range(iters):
        for start in range(pid * BLOCK, n, nprog * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def ev(fn, stream=None):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(stream); fn(); b.record(stream)
    return a, b


def probe(name, mainfn, src, dst, main_s, cs):
    n = src.numel()

    def cp(ctas, iters):
        _copy[(ctas,)](src, dst, n, iters, BLOCK=4096, num_warps=8)
    for _ in range(3):
        mainfn(); cp(8, 1)
    torch.cuda.synchronize()
    for ctas, iters in ((1, 1), (2, 2), (3, 3), (8, 3)):
        al, cl, ov, mo = [], [], [], []
        for _ in range(7):
            torch.cuda.synchronize()
            a, b = ev(mainfn); torch.cuda.synchronize(); al.append(a.elapsed_time(b))
            a, b = ev(lambda: cp(ctas, iters)); torch.cuda.synchronize(); cl.append(a.elapsed_time(b))
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            cs.wait_stream(main_s)
            with torch.cuda.stream(cs):
                cp(ctas, iters)
            m0, m1 = ev(mainfn)
            main_s.wait_stream(cs)
            t1.record()
            torch.cuda.synchronize()
            ov.append(t0.elapsed_time(t1)); mo.append(m0.elapsed_time(m1))
        A, C, O, M = (statistics.median(v) for v in (al, cl, ov, mo))
        gbs = 2 * n * 2 * iters / (C * 1e-3) / 1e9
        print(f"{name:34s} copy {ctas} CTAs x{iters} ({gbs:5.1f} GB/s alone, {C:6.3f} ms): main {A:.3f} -> {M:.3f} ms "
              f"(x{M / A:.3f}); together {O:.3f} vs serial {A + C:.3f} ms; hidden {100 * (A + C - O) / min(A, C):.0f}% "
              f"of the shorter", flush=True)


def main():
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    main_s, cs = torch.cuda.current_stream(), torch.cuda.Stream(priority=-1)
    slice_b = 6912 * 4096 * 2
    src = torch.empty(slice_b // 2, dtype=torch.bfloat16, device="cuda").normal_()
    dst = torch.empty_like(src)
    # mHC sub-chunk (reference: DRAM-bound)
    from bench_mhc_halving import make
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    w = make(3456, seed=9)
    probe("mHC fused post/pre 3,456 rows", lambda: mhc_fused_post_pre_tilelang(
        w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1,
        w["norm"], 1e-5), src, dst, main_s, cs)
    del w
    # W8A8 custom GEMM (compute-bound)
    import fp8_w8a8 as W
    from fp8_bench_common import marlin, quantize_like_prod
    W.ext(); W.STATE.custom = True
    g = torch.Generator(device="cuda").manual_seed(1)
    for (n, k, m, label) in ((12576, 4096, 13824, "W8A8 kda.in_proj M=13824"), (12576, 4096, 6912, "W8A8 kda.in_proj M=6912")):
        wt = (torch.randn(n, k, device="cuda", generator=g) * 0.02).bfloat16()
        layer, _, _ = quantize_like_prod(wt); del wt
        W.selftest(layer, n, k, None, "probe")
        x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
        q, s = W.quant_per_token(x)
        key = W._key(layer.weight, layer.weight_scale, k)
        w8 = W._scratch(n, k, x.device); W.repack(w8, layer.weight, n, k)
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        probe(label, lambda: W.custom_gemm(out, q, w8, s, W.ALPHA[key], n, k, m), src, dst, main_s, cs)
        del layer, x, q, s, out
    # Marlin W8A16 (shared-expert gate_up)
    n, k, m = 2048, 4096, 13824
    wt = (torch.randn(n, k, device="cuda", generator=g) * 0.02).bfloat16()
    layer, _, _ = quantize_like_prod(wt); del wt
    x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
    probe("Marlin shared.gate_up M=13824", lambda: marlin(layer, x, n, k), src, dst, main_s, cs)


if __name__ == "__main__":
    main()
