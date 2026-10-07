// Marlin fp8 repack -> standard row-major fp8 [N, K] for the W8A8 prefill path (fp8_w8a8.py).
//
//   out[N, K] (row-major fp8) = the logical fp8 weight held by the Marlin repack Wq[K/16, 4*Npad], byte for byte.
//
// This is the fp8-direct re-layout the PF3000 Test C review asked for (docs/PF3000_KILLTESTS_FP8.md 6): a pure
// byte permutation, 2*N*K bytes of DRAM traffic, instead of the exact bf16 dequant + cast (6*N*K bytes, the
// measured 106.9 ms/chunk). The output feeds torch.ops._C.cutlass_scaled_mm as the fp8 B operand (transposed
// view) with production's own un-permuted per-channel scale; its fp8 values are the same bytes Marlin consumes,
// so the weights introduce no rounding of their own (the only new rounding is the per-token activation quant).
//
// Weight layout (kernels/fp8_gemv.cu:5-10, verified bit-exactly by tests/test_fp8_gemv.py::layout): tile (kt, nt)
// of 16 (k) x 64 (n) is 256 int32 at int32 offset (kt * Npad/64 + nt) * 256; inside the tile int32 j = 8*t + 2*w + h
// (t = 0..31, w = 0..3, h = 0..1) holds 4 fp8 bytes (little endian b0..b3) of column n = 64*nt + 16*w + t/4 + 8*h
// at rows k = 16*kt + 2*(t%4) + {0, 8, 1, 9}[b]. Padded columns (N <= n < Npad) are dropped: out has exactly N rows.
//
// Work split: one block of 256 threads covers FOUR consecutive k-tiles of one n-tile (thread = row_local * 4 +
// sub, sub = the k-tile): the warp's 16-byte stores stay inside 64-byte contiguous row segments (fully used 32-byte
// sectors) and the four int32 words a thread needs sit stride-8 within one 1 KiB tile, so the whole tile is read
// with fully used sectors as well. Rows are written only for n < N; every k-tile byte of a kept row is written.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"

namespace fp8w8 {

__device__ __forceinline__ uint32_t pick4(uint32_t a, uint32_t b, uint32_t sel) {
  return __byte_perm(a, b, sel);          // one byte out of {a.b0..b3, b.b0..b3} per result byte
}

__global__ void fp8_marlin_to_std_kernel(const uint32_t* __restrict__ Wq, uint32_t* __restrict__ Out,
                                         int64_t N, int64_t K, int ktiles, int ntiles, int64_t ld) {
  const int nt = blockIdx.x;
  const int kt0 = blockIdx.y * 4;
  const int row = threadIdx.x >> 2;                 // 0..63, the n within the tile
  const int sub = threadIdx.x & 3;                  // which of the four k-tiles
  const int kt = kt0 + sub;
  if (kt >= ktiles) return;
  const int n = nt * 64 + row;
  if (n >= N) return;

  // logical position inside the tile: n = 16*w + t/4 + 8*h  ->  w = n_local/16, h = (n_local%16)/8, t/4 = n_local%8
  const int nl = row;
  const int w = nl >> 4;
  const int h = (nl >> 3) & 1;
  const int t4 = nl & 7;                            // t = 4*t4 + tt (tt = 0..3)
  const int64_t tile = (int64_t)kt * ntiles + nt;
  const uint32_t* base = Wq + tile * 256 + 32 * t4 + 2 * w + h;
  const uint32_t w0 = base[0], w1 = base[8], w2 = base[16], w3 = base[24];   // tt = 0..3, k = 2*tt + {0,8,1,9}[b]
  // out bytes 2*tt <- b0, 2*tt+8 <- b1, 2*tt+1 <- b2, 2*tt+9 <- b3  (little endian uint32 assembly)
  const uint4 v = make_uint4(pick4(w0, w1, 0x6420), pick4(w2, w3, 0x6420),
                             pick4(w0, w1, 0x7531), pick4(w2, w3, 0x7531));
  // Out is uint32_t* only for the aligned-load idiom; the destination offset is a BYTE offset (K % 16 == 0 keeps
  // the 16-byte vector aligned)
  uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(Out) + (int64_t)n * ld + (int64_t)kt * 16);
  *dst = v;
}


// v2 (w8a82): the same permutation with the four 1 KiB tiles staged through shared memory by fully coalesced uint4
// loads (each warp reads 512 contiguous bytes) instead of four stride-8 int32 loads per thread; byte-identical
// output (tests/fp8_w8a8_unit.py L), measured closer to a plain copy of the same bytes.
__global__ void fp8_marlin_to_std_kernel_v2(const uint32_t* __restrict__ Wq, uint32_t* __restrict__ Out,
                                            int64_t N, int64_t K, int ktiles, int ntiles, int64_t ld) {
  __shared__ uint32_t tiles[4][256];
  const int nt = blockIdx.x;
  const int kt0 = blockIdx.y * 4;
  {
    const int sub = threadIdx.x >> 6;              // 64 threads x uint4 = one 1 KiB tile
    const int i = threadIdx.x & 63;
    const int kt = kt0 + sub;
    if (kt < ktiles) {
      const uint4* src = reinterpret_cast<const uint4*>(Wq + ((int64_t)kt * ntiles + nt) * 256);
      reinterpret_cast<uint4*>(tiles[sub])[i] = src[i];
    }
  }
  __syncthreads();
  const int row = threadIdx.x >> 2;
  const int sub = threadIdx.x & 3;
  const int kt = kt0 + sub;
  if (kt >= ktiles) return;
  const int n = nt * 64 + row;
  if (n >= N) return;
  const int w = row >> 4, h = (row >> 3) & 1, t4 = row & 7;
  const uint32_t* base = tiles[sub] + 32 * t4 + 2 * w + h;
  const uint32_t w0 = base[0], w1 = base[8], w2 = base[16], w3 = base[24];
  const uint4 v = make_uint4(pick4(w0, w1, 0x6420), pick4(w2, w3, 0x6420),
                             pick4(w0, w1, 0x7531), pick4(w2, w3, 0x7531));
  uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(Out) + (int64_t)n * ld + (int64_t)kt * 16);
  *dst = v;
}

void fp8_marlin_to_std(torch::Tensor& out, const torch::Tensor& wq, int64_t n, int64_t k, int64_t variant) {
  TORCH_CHECK(wq.is_cuda() && wq.is_contiguous() && wq.dtype() == torch::kInt32, "fp8_marlin_to_std: wq");
  // opt-dense: out may be a row-strided view [n, k] of a wider [n, ld] buffer (ld % 16 == 0, unit column stride):
  // the hi+lo GEMM operand [W | W_S] is written in place; ld == k is the original contiguous case, byte for byte.
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.stride(1) == 1 && out.stride(0) >= k && out.stride(0) % 16 == 0 &&
              out.dtype() == torch::kFloat8_e4m3fn, "fp8_marlin_to_std: out");
  TORCH_CHECK(k % 16 == 0, "fp8_marlin_to_std: K must be a multiple of 16");
  const int64_t ktiles = k / 16;
  const int64_t npad = wq.size(1) / 4;
  TORCH_CHECK(wq.size(0) * 16 == k && npad % 64 == 0 && npad >= n && npad - n < 128,
              "fp8_marlin_to_std: layout (wq ", wq.sizes(), ", n ", n, ")");
  TORCH_CHECK(out.size(0) == n && out.size(1) == k, "fp8_marlin_to_std: out shape");
  const at::cuda::OptionalCUDAGuard guard(wq.device());
  const int ntiles = (int)(npad / 64);
  const int ktblocks = (int)((ktiles + 3) / 4);
  dim3 grid(ntiles, ktblocks);
  if (variant == 2)
    fp8_marlin_to_std_kernel_v2<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(wq.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out.data_ptr()), n, k, (int)ktiles, ntiles, (int64_t)out.stride(0));
  else
    fp8_marlin_to_std_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(wq.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(out.data_ptr()), n, k, (int)ktiles, ntiles, (int64_t)out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace fp8w8

// ---------------------------------------------------------------------------------------------------------------
// W8A8 GEMM (w8a82): CUTLASS 4.x SM120 TMA warp-specialized persistent kernels with the EXACT epilogue arithmetic of
// vLLM's cutlass_scaled_mm ScaledEpilogue (bf16(sa[m] * (sb[n] * acc)), fp32 acc) - the outputs are bitwise equal to
// torch.ops._C.cutlass_scaled_mm at every production shape (tests/w8a82/cutlass_bench.py, fp8_w8a8.py self-test).
// What differs is the tile shape / schedule and the persistent scheduler's swizzle + raster order, chosen per shape
// (fp8_w8a8.py GEMM_TABLE): one call over the whole M (no 2048-row pieces) at 167-189 TFLOPS instead of 114-176.
namespace w8g {
using namespace cute;


template <class TileShape, bool Pingpong>
struct W8A8Gemm {
    using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignAB = 16;
  static constexpr int AlignD = 8;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cute::conditional_t<Pingpong, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                             cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using EpiSchedule = cute::conditional_t<Pingpong, cutlass::epilogue::TmaWarpSpecialized,
                                          cutlass::epilogue::TmaWarpSpecializedCooperative>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<Int<0>, Int<1>, Int<0>>>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVT0>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      EpiSchedule, EVT>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignAB, ElementB, LayoutB, AlignAB,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static void run(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
                  const torch::Tensor& sb, int swizzle, int raster, torch::Tensor& ws) {
    const int m = a.size(0), k = a.size(1), n = b.size(0);
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    typename EVT0::Arguments evt0{{static_cast<const float*>(sb.data_ptr()), 0.f, {}}, {}, {}};
    typename EVT::Arguments evt{{static_cast<const float*>(sa.data_ptr()), 0.f, {}}, evt0, {}};
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {static_cast<const ElementA*>(a.data_ptr()), sA, static_cast<const ElementB*>(b.data_ptr()), sB},
        {evt, nullptr, sD, static_cast<ElementD*>(out.data_ptr()), sD}};
    args.scheduler.max_swizzle_size = swizzle;
    using RO = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "w8a8_gemm: can_implement failed (", m, "x", n, "x", k, ")");
    size_t wsz = Gemm::get_workspace_size(args);
    TORCH_CHECK(wsz <= (size_t)ws.numel(), "workspace ", wsz);
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }
};


using C128 = W8A8Gemm<Shape<_128, _128, _128>, false>;
using P128 = W8A8Gemm<Shape<_128, _128, _128>, true>;
using C128x256 = W8A8Gemm<Shape<_128, _256, _64>, false>;
using C256x128 = W8A8Gemm<Shape<_256, _128, _64>, false>;
}  // namespace w8g

// cfg: 0 = cooperative 128x128x128, 1 = pingpong 128x128x128, 2 = cooperative 128x256x64, 3 = cooperative 256x128x64
// raster: 0 heuristic, 1 along M, 2 along N. a [M,K] fp8 row-major, b [N,K] fp8 row-major, sa fp32 [M], sb fp32 [>=N]
void fp8_w8a8_gemm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sa,
                   const torch::Tensor& sb, int64_t cfg, int64_t swizzle, int64_t raster, torch::Tensor& ws) {
  TORCH_CHECK(a.is_cuda() && a.dtype() == torch::kFloat8_e4m3fn && a.dim() == 2 && a.stride(1) == 1, "w8a8_gemm: a");
  TORCH_CHECK(b.is_cuda() && b.dtype() == torch::kFloat8_e4m3fn && b.is_contiguous() && b.size(1) == a.size(1),
              "w8a8_gemm: b");
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.size(0) == a.size(0) &&
              out.size(1) == b.size(0), "w8a8_gemm: out");
  TORCH_CHECK(a.is_contiguous(), "w8a8_gemm: a must be contiguous");
  TORCH_CHECK(sa.dtype() == torch::kFloat32 && sa.is_contiguous() && sa.numel() == a.size(0), "w8a8_gemm: sa");
  TORCH_CHECK(sb.dtype() == torch::kFloat32 && sb.is_contiguous() && sb.numel() >= b.size(0), "w8a8_gemm: sb");
  TORCH_CHECK(a.size(1) % 16 == 0 && b.size(0) % 8 == 0, "w8a8_gemm: alignment");
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous(), "w8a8_gemm: ws");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  const int sw = (int)swizzle, ro = (int)raster;
  switch (cfg) {
    case 0: w8g::C128::run(out, a, b, sa, sb, sw, ro, ws); break;
    case 1: w8g::P128::run(out, a, b, sa, sb, sw, ro, ws); break;
    case 2: w8g::C128x256::run(out, a, b, sa, sb, sw, ro, ws); break;
    case 3: w8g::C256x128::run(out, a, b, sa, sb, sw, ro, ws); break;
    default: TORCH_CHECK(false, "w8a8_gemm: cfg ", cfg);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_marlin_to_std", &fp8w8::fp8_marlin_to_std,
        "Marlin fp8 repack -> standard row-major fp8 [N, K], byte-exact (kernels/fp8_w8a8.cu); variant 1 = the w8a8 "
        "kernel, 2 = the shared-memory staged kernel (w8a82, default)",
        pybind11::arg("out"), pybind11::arg("wq"), pybind11::arg("n"), pybind11::arg("k"), pybind11::arg("variant") = 2);
  m.def("fp8_w8a8_gemm", &fp8_w8a8_gemm,
        "CUTLASS SM120 W8A8 GEMM, cutlass_scaled_mm epilogue arithmetic (kernels/fp8_w8a8.cu)");
  m.attr("VERSION") = 2;
  m.attr("REPACK_LD") = 1;   // opt-dense: fp8_marlin_to_std accepts a row-strided out
}
