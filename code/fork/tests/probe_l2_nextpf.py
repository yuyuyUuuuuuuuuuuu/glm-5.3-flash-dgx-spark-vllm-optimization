"""Probe: can the latency-bound tail of a decode layer (the all-reduce and the small kernels after it, ~40 us in
production) pull the first megabytes of the NEXT big GEMV's weight into the L2, so that GEMV reads them from L2?

Per "layer" (34 distinct KDA in_proj FP8 weights, (12576 -> Npad 12608) x 4096, 51.6 MB each, cold):
  [prefetch kernel: prefetch.global.L2 over the first X MB of this layer's weight]  (main stream, a few us to issue)
  [idle window: a 1-block spin kernel of W us, standing in for the all-reduce + mhc]
  [fp8_gemv in_proj at M = 5 with the production TABLE config]
CUDA graph of 34 layers, interleaved rounds per X; prints per-layer us and the saving vs X = 0.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.cpp_extension import load_inline

import harness as H
import moeglue_rig as MR

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void pf_kernel(const char* p, long long nbytes) {
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 128; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * 128)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + o));
}
__global__ void spin_kernel(long long ns) {
    long long t0 = clock64();
    // clock64 at ~1 GHz-ish; use globaltimer for wall time
    unsigned long long g0, g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while ((long long)(g1 - g0) < ns);
    (void)t0;
}
void prefetch(torch::Tensor t, long long nbytes, long long blocks) {
    nbytes = std::min<long long>(nbytes, t.numel() * t.element_size());
    if (nbytes <= 0) return;
    pf_kernel<<<(unsigned)blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const char*)t.data_ptr(), nbytes);
}
void spin(long long ns) { spin_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }
"""
CPP = "void prefetch(torch::Tensor t, long long nbytes, long long blocks); void spin(long long ns);"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", default="0,2,4,6,8,12,16")
    ap.add_argument("--idle", type=float, default=40.0)
    ap.add_argument("--layers", type=int, default=34)
    ap.add_argument("--rounds", type=int, default=41)
    a = ap.parse_args()
    H.gpu_guard(8.0)
    import fp8_gemv as F
    from fp8_bench_common import random_marlin_layers

    import os
    from tf_exl3_moe import _cuda_include_shim
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    inc = _cuda_include_shim()
    ext = load_inline("l2pf_probe", CPP, SRC, functions=["prefetch", "spin"], verbose=False,
                      extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc])
    dev = torch.device("cuda", 0)
    n, k = 12576, 4096
    lays = random_marlin_layers(n, k, a.layers, dev, seed=5)
    fe = F.ext()
    x = torch.randn(5, k, device=dev).to(torch.bfloat16)
    cfg = F.select_config(lays[0].weight.shape[1] // 4, k, 5)
    print(f"in_proj weight {lays[0].weight.numel() * 4 / 2**20:.1f} MiB per layer, cfg {cfg}, idle {a.idle} us", flush=True)
    graphs = {}
    for mb in [int(v) for v in a.mb.split(",")]:
        def fns(mb=mb):
            out = []
            for L in lays:
                def f(L=L, mb=mb):
                    if mb:
                        ext.prefetch(L.weight, mb << 20, 96)
                    ext.spin(int(a.idle * 1000))
                    return fe.fp8_gemv(x, L.weight, L.weight_scale.view(-1), None, n, k, *cfg)
                out.append(f)
            return out
        graphs[f"pf{mb}MB"], _ = MR.capture(fns(), dev)
    res = MR.ab_rounds(graphs, a.layers, a.rounds, 1)
    base = res["pf0MB"]
    for name, v in res.items():
        r = [p / q for p, q in zip(v, base)]
        print(f"{name}: {MR.med(v):.1f} us per layer (idle {a.idle} + gemv), saving vs no prefetch "
              f"{MR.med(base) - MR.med(v):+.1f} us, ratio median {MR.med(r):.4f} p25-p75 {MR.pct(r, .25):.4f}-"
              f"{MR.pct(r, .75):.4f}", flush=True)
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
