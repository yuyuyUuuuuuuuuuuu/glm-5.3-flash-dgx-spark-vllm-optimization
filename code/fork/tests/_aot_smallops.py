# AOT build check of glm53_smallops_ext only (same flags as setup.py), into /w/.aot_smallops, then import it.
import os, sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
sys.argv = ["setup.py", "build_ext", "--build-lib", "/w/.aot_smallops", "--build-temp", "/tmp/aot_smallops_tmp"]
sys.path.insert(0, "/w")
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import importlib.util
spec = importlib.util.spec_from_file_location("fork_setup_shim", "/w/setup.py")
src = open("/w/setup.py").read()
ns = {"__file__": "/w/setup.py"}
exec(src.split("inc = cuda_include_shim()")[0], ns)
inc = ns["cuda_include_shim"]()
os.chdir("/w")
setup(name="aot-smallops-check", ext_modules=[CUDAExtension(name="glm53_smallops_ext", sources=["kernels/smallops.cu"],
      extra_compile_args={"cxx": ["-O3", *inc], "nvcc": ["-O3", *inc]})], cmdclass={"build_ext": BuildExtension})
sys.path.insert(0, "/w/.aot_smallops")
import glm53_smallops as SO
SO._EXT = None
print("AOT import:", SO.load_ext().__file__, "version", SO.load_ext().version(), SO.EXT_SOURCE)
