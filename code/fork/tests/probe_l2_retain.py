"""Probe: does data brought into the GB10 L2 (24 MiB) in an idle window stay there and make the next read faster?

Eager, CUDA events around the timed read only. Per trial: flush the L2 (read 96 MiB of another buffer), then
  cold       spin W us                                   | timed read of X MiB
  warm       read X (plain loads)        spin W us       | timed read
  pf         prefetch.global.L2 over X   spin W us       | timed read
  pf_el      prefetch.global.L2::evict_last over X, spin | timed read
  bulkpf     cp.async.bulk.prefetch.L2 over X (TMA), spin| timed read
  touch_el   loads with an L2 evict_last policy, spin    | timed read
The timed read = plain 16 B loads over X MiB with 48 x 8 blocks of 256 threads.
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch
from torch.utils.cpp_extension import load_inline

import harness as H

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
__global__ void read_kernel(const uint4* __restrict__ p, long long n16, unsigned* out) {
    unsigned acc = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
__global__ void touch_el_kernel(const uint4* __restrict__ p, long long n16, unsigned* out) {
    unsigned acc = 0;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
        unsigned a, b, c, d;
        asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p + i), "l"(pol));
        acc ^= a ^ b ^ c ^ d;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
__global__ void pf_kernel(const char* p, long long nbytes, int el) {
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 128; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * 128) {
        if (el) asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + o));
        else asm volatile("prefetch.global.L2 [%0];" :: "l"(p + o));
    }
}
__global__ void bulkpf_kernel(const char* p, long long nbytes, int chunk) {
    for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * chunk; o < nbytes;
         o += (long long)gridDim.x * blockDim.x * chunk) {
        unsigned sz = (unsigned)min((long long)chunk, nbytes - o);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + o), "r"(sz) : "memory");
    }
}
__global__ void spin_kernel(long long ns) {
    unsigned long long g0, g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while ((long long)(g1 - g0) < ns);
}
static cudaStream_t cs() { return at::cuda::getCurrentCUDAStream(); }
void rd(torch::Tensor t, long long nbytes, torch::Tensor out, long long blocks) {
    read_kernel<<<(unsigned)blocks, 256, 0, cs()>>>((const uint4*)t.data_ptr(), nbytes / 16, (unsigned*)out.data_ptr());
}
void touch_el(torch::Tensor t, long long nbytes, torch::Tensor out, long long blocks) {
    touch_el_kernel<<<(unsigned)blocks, 256, 0, cs()>>>((const uint4*)t.data_ptr(), nbytes / 16, (unsigned*)out.data_ptr());
}
void pf(torch::Tensor t, long long nbytes, long long el, long long blocks) {
    pf_kernel<<<(unsigned)blocks, 256, 0, cs()>>>((const char*)t.data_ptr(), nbytes, (int)el);
}
void bulkpf(torch::Tensor t, long long nbytes, long long chunk, long long blocks) {
    bulkpf_kernel<<<(unsigned)blocks, 32, 0, cs()>>>((const char*)t.data_ptr(), nbytes, (int)chunk);
}
void spin(long long ns) { spin_kernel<<<1, 32, 0, cs()>>>(ns); }
"""
CPP = ("void rd(torch::Tensor t, long long nbytes, torch::Tensor out, long long blocks);"
       "void touch_el(torch::Tensor t, long long nbytes, torch::Tensor out, long long blocks);"
       "void pf(torch::Tensor t, long long nbytes, long long el, long long blocks);"
       "void bulkpf(torch::Tensor t, long long nbytes, long long chunk, long long blocks);"
       "void spin(long long ns);")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", default="2,4,8,12,16,20")
    ap.add_argument("--idle", type=float, default=40.0)
    ap.add_argument("--trials", type=int, default=25)
    a = ap.parse_args()
    H.gpu_guard(4.0)
    from tf_exl3_moe import _cuda_include_shim
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
    inc = _cuda_include_shim()
    ext = load_inline("l2retain_probe", CPP, SRC, functions=["rd", "touch_el", "pf", "bulkpf", "spin"],
                      verbose=False, extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc])
    dev = torch.device("cuda", 0)
    print(f"{torch.cuda.get_device_name(0)} L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB",
          flush=True)
    buf = torch.randint(0, 2**31 - 1, (32 << 20,), dtype=torch.int32, device=dev)       # 128 MiB, data region
    flush = torch.randint(0, 2**31 - 1, (24 << 20,), dtype=torch.int32, device=dev)     # 96 MiB
    out = torch.zeros(4, dtype=torch.int32, device=dev)
    B = 48 * 8
    idle_ns = int(a.idle * 1000)
    variants = ["cold", "warm", "pf", "pf_el", "bulkpf", "touch_el"]
    for mb in [int(v) for v in a.mb.split(",")]:
        nb = mb << 20
        res = {v: [] for v in variants}
        for t in range(a.trials):
            off = ((t * 7) % 8) * (8 << 20)          # a different 8 MiB-aligned window each trial
            reg = buf.view(torch.uint8)[off: off + nb]
            for v in (variants if t % 2 == 0 else variants[::-1]):
                ext.rd(flush, flush.numel() * 4, out, B)
                if v == "warm":
                    ext.rd(reg, nb, out, B)
                elif v == "pf":
                    ext.pf(reg, nb, 0, 96)
                elif v == "pf_el":
                    ext.pf(reg, nb, 1, 96)
                elif v == "bulkpf":
                    ext.bulkpf(reg, nb, 1 << 16, 8)
                elif v == "touch_el":
                    ext.touch_el(reg, nb, out, B)
                ext.spin(idle_ns)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                ext.rd(reg, nb, out, B)
                e.record()
                torch.cuda.synchronize()
                res[v].append(s.elapsed_time(e) * 1000.0)
        cold = statistics.median(res["cold"])
        line = [f"X={mb:2d} MiB:"]
        for v in variants:
            m = statistics.median(res[v])
            line.append(f"{v} {m:6.1f} us ({nb / m / 1e3:6.1f} GB/s, {cold - m:+6.1f})")
        print(" | ".join(line), flush=True)
    H.report_peak(4.0)


if __name__ == "__main__":
    H.run_main(main)
