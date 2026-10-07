"""Build glm53_moe_e4m3_ext into overlay/ (next to overlay/glm53_moe_e4m3.py: the bundle ships both in its overlay dir and
overlay/patch_moe_e4m3.py copies them into site-packages only when GLM53_MOE_E4M3=1). Inside the production image:
   python3 tools/moee4m3/build.py [--verbose-ptxas] [--nvcc=FLAG ...] [--out=DIR (default overlay/)]
NO --use_fast_math / -ftz: the per-row quantization's division, the SiLU exponential and subnormal e4m3 follow IEEE
like torch. setup.py does NOT build this extension (an OFF kit's site/ stays the previous kit's byte for byte)."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
verbose = "--verbose-ptxas" in sys.argv
extra = [a[len("--nvcc="):] for a in sys.argv if a.startswith("--nvcc=")]
sys.path.insert(0, str(REPO))
from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

ns = {"__file__": str(REPO / "setup.py"), "__name__": "setup_prefix"}
exec(compile(open(REPO / "setup.py").read().split("inc = cuda_include_shim()")[0], "setup.py", "exec"), ns)
inc = ns["cuda_include_shim"]()
nvcc = ["-O3", "-lineinfo", *inc, *extra] + (["-Xptxas", "-v"] if verbose else [])
build = tempfile.mkdtemp(prefix="moee4m3-build-")
try:
    setup(name="glm53_moe_e4m3_ext", script_args=["build_ext", "--build-lib", build, "--build-temp", build + "/t"],
          ext_modules=[CUDAExtension(name="glm53_moe_e4m3_ext", sources=["kernels/moe_e4m3.cu"],
                                     extra_compile_args={"cxx": ["-O3", *inc], "nvcc": nvcc})],
          cmdclass={"build_ext": BuildExtension})
    so = next(Path(build).glob("glm53_moe_e4m3_ext*.so"))
    dst = next((Path(a[len("--out="):]) for a in sys.argv if a.startswith("--out=")), REPO / "overlay")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so, dst / so.name)
    print(f"built {dst / so.name}")
finally:
    shutil.rmtree(build, ignore_errors=True)
