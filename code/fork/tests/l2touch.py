"""Test-only helpers for the L2-warming experiments (docs/DEC_MOEGLUE.md): touch = read a tensor's bytes into the L2
(real 16 B loads, optional L2 evict_last policy, result discarded), spin = a 1-warp kernel that idles W us (stand-in
for the all-reduce / mhc window). JIT-built with load_inline (tests only)."""
from __future__ import annotations

import os

import torch

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
template <int U>
__global__ void touch_kernel(const uint4* __restrict__ p, long long n16, int el, unsigned* out) {
    unsigned acc = 0;
    uint64_t pol = 0;
    if (el) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i0 = (long long)blockIdx.x * blockDim.x + threadIdx.x; i0 < n16; i0 += stride * U) {
        unsigned r[U][4];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            long long i = i0 + u * stride;
            if (i < n16) {
                if (el)
                    asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                                 : "=r"(r[u][0]), "=r"(r[u][1]), "=r"(r[u][2]), "=r"(r[u][3]) : "l"(p + i), "l"(pol));
                else
                    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(r[u][0]), "=r"(r[u][1]), "=r"(r[u][2]), "=r"(r[u][3]) : "l"(p + i));
            } else {
                r[u][0] = r[u][1] = r[u][2] = r[u][3] = 0;
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= r[u][0] ^ r[u][1] ^ r[u][2] ^ r[u][3];
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
__global__ void spin_kernel(long long ns) {
    unsigned long long g0, g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1)); } while ((long long)(g1 - g0) < ns);
}
void touch(torch::Tensor t, long long nbytes, long long blocks, long long el, torch::Tensor out) {
    long long n = std::min<long long>(nbytes, t.numel() * t.element_size()) / 16;
    if (n <= 0) return;
    // el: bit 0 = L2 evict_last policy; bits 8..: unroll (1, 4 or 8 loads in flight per thread)
    int unroll = (int)(el >> 8), e = (int)(el & 1);
    auto s = at::cuda::getCurrentCUDAStream();
    auto P = (const uint4*)t.data_ptr();
    auto O = (unsigned*)out.data_ptr();
    if (unroll >= 8) touch_kernel<8><<<(unsigned)blocks, 256, 0, s>>>(P, n, e, O);
    else if (unroll >= 4) touch_kernel<4><<<(unsigned)blocks, 256, 0, s>>>(P, n, e, O);
    else touch_kernel<1><<<(unsigned)blocks, 256, 0, s>>>(P, n, e, O);
}
void spin(long long ns) { spin_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }
"""
CPP = ("void touch(torch::Tensor t, long long nbytes, long long blocks, long long el, torch::Tensor out);"
       "void spin(long long ns);")
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        from tf_exl3_moe import _cuda_include_shim
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = _cuda_include_shim()
        _EXT = load_inline("l2touch_test", CPP, SRC, functions=["touch", "spin"], verbose=False,
                           extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc])
    return _EXT
