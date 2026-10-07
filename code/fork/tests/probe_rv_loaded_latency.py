"""Reviewer probe (docs/DEC_MOEGLUE.md risk 1): how much does the warm's DRAM traffic slow a latency-bound memory
access pattern (the kind an all-reduce's flag polling / DMA round trips are made of)? One thread chases a random
pointer cycle over 256 MiB (every hop a DRAM miss) and times itself with %globaltimer: alone, next to production's
l2_warm (16 blocks x 256 threads, 8 x 16 B in flight, 15.75 MiB per launch over 4 rotating 64 MiB-apart regions,
launched back to back so it covers the whole chase), and next to a 96-block full-bandwidth read for scale.
"""
from __future__ import annotations

import os
import statistics

import torch

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void chase_kernel(const unsigned* __restrict__ nxt, int hops, unsigned start, unsigned long long* out) {
    unsigned p = start;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = 0; i < hops; ++i) {
        unsigned v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(nxt + p));
        p = v;
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    out[0] = t1 - t0;
    out[1] = p;
}
void chase(torch::Tensor nxt, long long hops, long long start, torch::Tensor out) {
    chase_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((const unsigned*)nxt.data_ptr(), (int)hops,
        (unsigned)start, (unsigned long long*)out.data_ptr());
}
"""
CPP = "void chase(torch::Tensor nxt, long long hops, long long start, torch::Tensor out);"


def main():
    from torch.utils.cpp_extension import load_inline

    import harness as H
    import tf_exl3_moe as tfm
    from tf_exl3_moe import _cuda_include_shim

    H.gpu_guard(4.0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    inc = _cuda_include_shim()
    E = load_inline("rv_chase", CPP, SRC, functions=["chase"], verbose=False, extra_cuda_cflags=["-O3", *inc],
                    extra_cflags=["-O3", *inc])
    TF = tfm.load_ext()
    import l2touch
    LT = l2touch.ext()
    dev = torch.device("cuda", 0)
    lines = 2 ** 21                                    # 128 B lines -> 256 MiB
    perm = torch.randperm(lines, device=dev, dtype=torch.int64)
    nxt = torch.zeros(lines * 32, dtype=torch.int32, device=dev)
    nxt[perm * 32] = (torch.roll(perm, -1) * 32).to(torch.int32)
    out = torch.zeros(2, dtype=torch.int64, device=dev)
    load = torch.empty(4 * 64 * 2 ** 20, dtype=torch.uint8, device=dev)
    regs = [load[i * 64 * 2 ** 20: i * 64 * 2 ** 20 + int(15.75 * 2 ** 20)] for i in range(4)]
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    side = torch.cuda.Stream(device=dev)
    hops = 3000

    def one(mode):
        side.wait_stream(torch.cuda.current_stream())
        if mode != "alone":
            with torch.cuda.stream(side):
                for i in range(60):
                    if mode == "warm16":
                        TF.l2_warm([regs[i % 4]], 16, 8, sink)
                    else:
                        LT.touch(regs[i % 4], regs[i % 4].numel(), 96, 8 << 8, sink)
        s = int(torch.randint(0, lines, (1,)).item()) * 32
        E.chase(nxt, hops, s, out)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        return out[0].item() / hops

    for m in ("alone", "warm16", "bw96"):
        one(m)
    res = {m: [] for m in ("alone", "warm16", "bw96")}
    for r in range(15):
        for m in (("alone", "warm16", "bw96") if r % 2 == 0 else ("bw96", "warm16", "alone")):
            res[m].append(one(m))
    for m, v in res.items():
        print(f"{m:7s}: DRAM pointer-chase latency median {statistics.median(v):7.1f} ns/hop "
              f"(min {min(v):.1f}, max {max(v):.1f}, n={len(v)})", flush=True)
    a = statistics.median(res["alone"])
    print(f"warm16 / alone = x{statistics.median(res['warm16']) / a:.3f}; bw96 / alone = "
          f"x{statistics.median(res['bw96']) / a:.3f}")


if __name__ == "__main__":
    import harness as H
    H.run_main(main)
