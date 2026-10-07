"""AOT build of the TF EXL3 kernels (docs/DESIGN.md §A12) + the vLLM plugin entry point (§D.4).

Build inside the production image (torch/CUDA of the image), without build isolation:
    python3 setup.py build_ext --inplace          # nodeC tests: tf_exl3_moe_ext*.so next to tf_exl3_moe.py
    pip install --no-build-isolation .            # image build: installs modules + extension + entry point
Target: sm_121a (GB10). TF_PARITY=1 (default) = exl3_moe precision parity; TF_PARITY=0 = NATIVE (experiments).
"""
import os
from pathlib import Path

from setuptools import setup

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

HERE = Path(__file__).resolve().parent


def cuda_include_shim() -> list[str]:
    """Same header shim as tf_exl3_moe._cuda_include_shim (cu13 pip headers without their crt/)."""
    base = Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include")
    tk = Path("/usr/local/cuda/include")
    if not base.is_dir():
        return []
    shim = Path(os.environ.get("TF_EXL3_SHIM", "/tmp/tf_exl3_shim"))
    shim.mkdir(parents=True, exist_ok=True)
    for f in list(base.glob("*.h")) + list(base.glob("*.hpp")):
        if not (tk / f.name).exists():
            link = shim / f.name
            if not link.exists():
                try:
                    link.symlink_to(f)
                except OSError:
                    pass
    return [f"-I{shim}"]


inc = cuda_include_shim()
_CUTLASS = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
CUTLASS_INC = [f"-I{_CUTLASS}/include", f"-I{_CUTLASS}/tools/util/include"]
parity = os.environ.get("TF_PARITY", "1")

setup(
    name="tf-exl3-moe",
    version="0.1.0",
    description="TensorFold EXL3 routed-expert kernels as a drop-in for exllamav3_ext.exl3_moe (vLLM decode)",
    license="AGPL-3.0-only",
    py_modules=["tf_exl3_moe", "integrate", "glm53_runtime", "fp8_gemv", "fp8_large_m_tl", "glm53_bf16_gemv",
                "glm53_gemv_install", "glm53_prefill_cap", "glm53_prefill_quickwins", "glm53_mla_prefill",
                "fp8_roof", "glm53_moeglue", "glm53_hostloop", "glm53_smallops", "glm53_smallops_install", "glm53_kda_lazy", "glm53_vtrim_stats", "glm53_spec_vtrim", "glm53_ar1shot", "glm53_dlmh", "glm53_mla_planpin"],
    # NOTE (r16n): fp8_w8a8 is deliberately NOT a py_module of the bundle site (the r16k glm53_dectrace
    # precedent: the kit's site/ stays byte-identical to the previous kit's with every switch unset). The module
    # ships in the bundle overlay (overlay/fp8_w8a8.py, a cmp-checked copy of the repo root module) and
    # overlay/patch_dense_w8a8.py installs it into site-packages ONLY when GLM53_DENSE_W8A8=1; the AOT kernel
    # tf_fp8_w8a8_ext below is built and shipped the same way (overlay/, not site/).
    ext_modules=[
        CUDAExtension(
            name="tf_exl3_moe_ext",
            sources=["kernels/exl3_bind.cpp", "kernels/exl3.cu"],
            extra_compile_args={
                "cxx": ["-O3", *inc],
                "nvcc": ["-O3", f"-DTF_PARITY={parity}", *inc],
            },
        ),
        CUDAExtension(   # FP8 Marlin-layout small-M GEMM (fp8_gemv.py, GLM53_FP8_GEMV)
            name="tf_fp8_gemv_ext",
            sources=["kernels/fp8_gemv.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
        CUDAExtension(   # GLM53_DEC_DLMH exact rescoring of lm_head column octets (glm53_dlmh.py)
            name="tf_dlmh_ext",
            sources=["kernels/dlmh_gemv.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
        CUDAExtension(   # FP8 Marlin-layout large-M path (fp8_gemv.py, GLM53_FP8_LARGE_M): exact dequant + epilogue
            name="tf_fp8_large_m_ext",
            sources=["kernels/fp8_large_m.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
        CUDAExtension(   # W8A8 prefill dense GEMMs (fp8_w8a8.py, GLM53_DENSE_W8A8): Marlin fp8 -> standard layout
            name="tf_fp8_w8a8_ext",
            sources=["kernels/fp8_w8a8.cu"],
            # w8a82: the CUTLASS 4.x SM120 GEMM uses the image's own CUTLASS headers (flashinfer's copy)
            extra_compile_args={"cxx": ["-O3", *inc, *CUTLASS_INC],
                                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-DNDEBUG", *inc,
                                         *CUTLASS_INC]},
        ),
        CUDAExtension(   # L2 prefetch of the next decode weight (fp8_roof.py, GLM53_DEC_FP8ROOF)
            name="tf_fp8_roof_ext",
            sources=["kernels/fp8_roof.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
        # small-M dense GEMMs of decode (docs/BF16_GEMV.md); used only when GLM53_BF16_GEMV=1
        CUDAExtension(
            name="glm53_gemv_ext",
            sources=["kernels/gemv_bf16.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
        # sparse-MLA prefill attention (docs/MLA_PREFILL.md); used only when GLM53_MLA_PREFILL is set
        CUDAExtension(
            name="glm53_mla_prefill_ext",
            sources=["kernels/mla_prefill/mla_prefill.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", "-use_fast_math", *inc]},
        ),
        # small decode ops outside the MoE (docs/DEC_SMALLOPS.md); used only when GLM53_DEC_SMALLOPS=1
        CUDAExtension(
            name="glm53_smallops_ext",
            sources=["kernels/smallops.cu"],
            extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    entry_points={"vllm.general_plugins": ["tf_exl3_moe = integrate:plugin_register"]},
)
