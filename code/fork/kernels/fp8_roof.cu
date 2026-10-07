// GLM53_DEC_FP8ROOF helpers (fp8_roof.py): L2 prefetch of a read-only weight range from a side stream, so that the
// DRAM-idle window before a decode GEMV (all-reduce, mHC, norms, KDA recurrent, MLA attention) already streams the
// first bytes of the next weight into L2. Pure reads: the loaded values are discarded (never written anywhere; the
// only store is behind a data-dependent condition that the xor of the loads is a fixed 32-bit value AND a flag the
// caller never sets, so it cannot fire). No workspace, no atomics, no effect on any result: only on timing.
//
//   grid-stride over [p, p + nbytes) in 16-byte loads, U loads in flight per thread (ld.global.cg: cache in L2 only,
//   not L1), optionally with an L2 eviction-priority policy (createpolicy.fractional: 0 = normal, 1 = evict_last,
//   2 = evict_first -- the last one is only a control for the benchmarks: it must not help).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace fp8roof {

template <int POL>
__device__ __forceinline__ uint4 ld16(const uint4* p, uint64_t pol) {
  uint4 v;
  if constexpr (POL == 0) {
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else {
    asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  }
  return v;
}

template <int POL, int U>
__global__ void __launch_bounds__(256) l2_prefetch_kernel(const uint4* __restrict__ p, int64_t n16, int never,
                                                          unsigned* __restrict__ sink) {
  uint64_t pol = 0;
  if constexpr (POL == 1) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if constexpr (POL == 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  unsigned acc = 0;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = ld16<POL>(p + i + u * stride, pol);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = ld16<POL>(p + i, pol);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (never && acc == 0x9e3779b9u) sink[0] = acc;   // keeps the loads alive; `never` is always 0
}

}  // namespace fp8roof

// Prefetch bytes [offset, offset + nbytes) of tensor t (any dtype, contiguous, 16-byte aligned range) into L2 on the
// current stream. ctas CTAs of 256 threads; pol 0 normal / 1 evict_last / 2 evict_first.
void l2_prefetch(torch::Tensor t, int64_t offset, int64_t nbytes, int64_t ctas, int64_t pol) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "l2_prefetch: contiguous CUDA tensor expected");
  const int64_t total = t.numel() * t.element_size();
  TORCH_CHECK(offset >= 0 && nbytes >= 0 && offset + nbytes <= total, "l2_prefetch: range outside the tensor");
  const uintptr_t base = (uintptr_t)t.data_ptr() + (uintptr_t)offset;
  TORCH_CHECK((base & 15) == 0 && (nbytes & 15) == 0, "l2_prefetch: range must be 16-byte aligned");
  TORCH_CHECK(ctas >= 1 && ctas <= 1024 && pol >= 0 && pol <= 2, "l2_prefetch: bad ctas / pol");
  if (nbytes == 0) return;
  const at::cuda::CUDAGuard guard(t.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint4* p = reinterpret_cast<const uint4*>(base);
  const int64_t n16 = nbytes / 16;
  unsigned* sink = nullptr;
  switch (pol) {
    case 0: fp8roof::l2_prefetch_kernel<0, 4><<<(unsigned)ctas, 256, 0, stream>>>(p, n16, 0, sink); break;
    case 1: fp8roof::l2_prefetch_kernel<1, 4><<<(unsigned)ctas, 256, 0, stream>>>(p, n16, 0, sink); break;
    default: fp8roof::l2_prefetch_kernel<2, 4><<<(unsigned)ctas, 256, 0, stream>>>(p, n16, 0, sink); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("l2_prefetch", &l2_prefetch, "L2 prefetch of a read-only byte range (kernels/fp8_roof.cu)");
  m.attr("VERSION") = 1;
}
