// w8a82 probe: CUTLASS 4.x SM120 fp8 GEMM variants with the vLLM ScaledEpilogue arithmetic
//   out[M,N] bf16 = bf16( sa[m] * (sb[n] * acc) ),  acc = sum_k a[m,k] * b[n,k]  (fp32)
// a [M,K] fp8 row-major, b [N,K] fp8 row-major (= K-major B), sa fp32 [M], sb fp32 [N].
// Variants: kernel schedule (cooperative / pingpong) x CTA tile; runtime swizzle + raster order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement failed");
    size_t wsz = Gemm::get_workspace_size(args);
    TORCH_CHECK(wsz <= (size_t)ws.numel(), "workspace ", wsz);
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "run failed");
  }
};

using C128 = W8A8Gemm<Shape<_128, _128, _128>, false>;
using P128 = W8A8Gemm<Shape<_128, _128, _128>, true>;
using P64 = W8A8Gemm<Shape<_64, _128, _128>, true>;
using C128x256 = W8A8Gemm<Shape<_128, _256, _64>, false>;
using C256x128 = W8A8Gemm<Shape<_256, _128, _64>, false>;
using P64x256 = W8A8Gemm<Shape<_64, _256, _128>, true>;

void w8a8_gemm(int64_t cfg, torch::Tensor out, torch::Tensor a, torch::Tensor b, torch::Tensor sa, torch::Tensor sb,
          int64_t swizzle, int64_t raster, torch::Tensor ws) {
  const at::cuda::OptionalCUDAGuard guard(a.device());
  switch (cfg) {
    case 0: C128::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    case 1: P128::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    case 2: P64::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    case 3: C128x256::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    case 4: C256x128::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    case 5: P64x256::run(out, a, b, sa, sb, swizzle, raster, ws); break;
    default: TORCH_CHECK(false, "cfg");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &w8a8_gemm); }
