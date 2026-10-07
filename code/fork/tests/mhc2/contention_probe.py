"""mhc2: single-process DRAM/SM contention probe for the SP2 overlap (nodeC, one GB10).

The 2-process NCCL harness (nccl_sp2_pipeline.py) cannot measure contention as production sees it: its two ranks
time-slice ONE GPU (the peer's spinning NCCL kernels steal whole time slices) and its socket transport copies through
the kernel's loopback. Production: each rank owns its GPU; an in-flight RS/AG of a 3,456-row slice (28.3 MB) costs
the compute side 8 NCCL LL channel CTAs (of 48 SMs) plus the DRAM traffic of NCCL's host-staged no-GDR path
(~3x the slice: read input, write staging, read received staging -> write output; ~1.35 ms at the measured 21 GB/s
wire rate = ~60-65 GB/s of DRAM).

Low-CTA variants (1-3 CTAs) approximate NCCL's wire-limited DRAM rate (~60 GB/s); 8+ CTAs saturate DRAM (worst case).

This probe runs one production mHC sub-chunk (3,456 rows) on the main stream while a side stream runs an 8-CTA Triton
copy kernel that streams ~3 x 28.3 MB (the NCCL footprint stand-in), and reports the mHC slowdown and the combined time
vs the serial sum. Variants: copy volume x1/x3, CTAs 8/16.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc2/contention_probe.py
"""
import os
import statistics
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mhc_sp"))
from bench_mhc_halving import make  # noqa: E402


@triton.jit
def _copy(src, dst, n, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for it in range(iters):
        for start in range(pid * BLOCK, n, nprog * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def main():
    B = 3456
    w = make(B, seed=9)
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    mh = lambda: mhc_fused_post_pre_tilelang(w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"],
                                             1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, w["norm"], 1e-5)
    slice_b = 2 * B * 4096 * 2            # an RS input slice: 2B rows bf16 = 56.6 MB / 2
    src = torch.empty(slice_b // 2, dtype=torch.bfloat16, device="cuda").normal_()
    dst = torch.empty_like(src)
    n = src.numel()
    main_s, cs = torch.cuda.current_stream(), torch.cuda.Stream(priority=-1)

    def cp(ctas, iters):
        _copy[(ctas,)](src, dst, n, iters, BLOCK=4096, num_warps=8)

    def ev_time(fn, stream=None):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(stream)
        fn()
        b.record(stream)
        return a, b

    for _ in range(3):
        mh(); cp(8, 1)
    torch.cuda.synchronize()
    print(f"gpu {torch.cuda.get_device_name(0)}; mHC sub-chunk {B} rows; copy slice {slice_b / 2 / 1e6:.1f} MB read+write per iter")
    for ctas, iters in ((1, 1), (2, 2), (3, 3), (8, 1), (8, 3), (16, 3), (48, 3)):
        al, cl, ov, mo = [], [], [], []
        for _ in range(9):
            torch.cuda.synchronize()
            a, b = ev_time(mh); torch.cuda.synchronize(); al.append(a.elapsed_time(b))
            a, b = ev_time(lambda: cp(ctas, iters)); torch.cuda.synchronize(); cl.append(a.elapsed_time(b))
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            cs.wait_stream(main_s)
            with torch.cuda.stream(cs):
                cp(ctas, iters)
            m0, m1 = ev_time(mh)
            main_s.wait_stream(cs)
            t1.record()
            torch.cuda.synchronize()
            ov.append(t0.elapsed_time(t1)); mo.append(m0.elapsed_time(m1))
        A, C, O, M = (statistics.median(v) for v in (al, cl, ov, mo))
        gbs = 2 * n * 2 * iters / (C * 1e-3) / 1e9
        print(f"copy {ctas:2d} CTAs x{iters} ({gbs:5.1f} GB/s alone, {C:6.3f} ms): mHC alone {A:.3f} ms -> {M:.3f} ms "
              f"concurrent (x{M / A:.3f}); both together {O:.3f} ms vs serial {A + C:.3f} ms "
              f"(hidden {100 * (A + C - O) / min(A, C):.0f}% of the shorter)")


if __name__ == "__main__":
    main()
