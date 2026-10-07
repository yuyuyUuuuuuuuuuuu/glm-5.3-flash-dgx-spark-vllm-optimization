"""Build tf_fp8_w8a8_ext (kernels/fp8_w8a8.cu, the REPACK_LD version) as an importable module for nodeC handoff runs:
same flags as setup.py; output /w/.optdense_build/tf_fp8_w8a8_ext.so (copied by the caller into overlay/ under the
AOT name). Not a kit build."""
import os, sys, shutil
sys.path.insert(0, "/w")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
import fp8_w8a8 as W
from torch.utils.cpp_extension import load
inc = W._include_shim()
cl = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
cinc = [f"-I{cl}/include", f"-I{cl}/tools/util/include"]
bd = "/w/.optdense_build"
os.makedirs(bd, exist_ok=True)
m = load(name="tf_fp8_w8a8_ext", sources=["/w/kernels/fp8_w8a8.cu"], build_directory=bd,
         extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DNDEBUG", *inc, *cinc],
         extra_cflags=["-O3", *inc, *cinc], verbose=False)
print("built", m.__file__, "VERSION", m.VERSION, "REPACK_LD", getattr(m, "REPACK_LD", None))
