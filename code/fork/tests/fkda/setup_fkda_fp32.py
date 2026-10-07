"""Build vLLM's _flashkda_fp32_C from a FlashKDA checkout, outside vLLM's cmake.

Copy of pfkda's tests/pf3000/setup_pf3000.py (kindling-gx10's setup.py with the
nvcc gencode from FKDA_NVCC_ARCH, default sm_121a) with ONE further change: the
extension is named _flashkda_fp32_C instead of _flashkda_C, so the fp32-state
17a037d build registers torch.ops._flashkda_fp32_C.* and can coexist with the
image's pre-fix bf16 vllm/_flashkda_C (which nothing in the GLM-5.3-Flash path
imports, but which the same namespace would collide with). Kernel sources,
includes, macros and flags are unchanged (see build_flashkda_fp32.sh).
"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ninja compiles from its own build directory, so includes must be absolute.
HERE = Path(__file__).resolve().parent
K = HERE / "flashkda"
ARCH = os.environ.get("PF3000_NVCC_ARCH", "sm_121a")

setup(
    name="flashkda_c",
    ext_modules=[
        CUDAExtension(
            "_flashkda_fp32_C",
            sources=[
                "vllm-csrc/flashkda_registration.cpp",
                f"{K}/csrc/flash_kda.cpp",
                f"{K}/csrc/smxx/fwd_launch.cu",
            ],
            include_dirs=[
                f"{HERE}/vllm-csrc",
                f"{K}/csrc",
                f"{K}/cutlass/include",
                f"{K}/cutlass/examples/common",
                f"{K}/cutlass/tools/util/include",
            ],
            define_macros=[
                ("USE_CUDA", None),
                ("TORCH_TARGET_VERSION", "0x020B000000000000ULL"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    f"-gencode=arch=compute_{ARCH.lstrip('sm_')},code={ARCH}",
                    # torch adds these; vLLM's cmake removes them, and so does this.
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
