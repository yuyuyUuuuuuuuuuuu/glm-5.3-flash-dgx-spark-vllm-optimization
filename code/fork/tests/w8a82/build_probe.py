import os, sys, time
sys.path.insert(0, "/w")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
import fp8_w8a8 as W
from torch.utils.cpp_extension import load
C = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
inc = W._include_shim() + [f"-I{C}/include", f"-I{C}/tools/util/include"]
t = time.time()
name = sys.argv[1] if len(sys.argv) > 1 else "cutlass_probe"
m = load(name=name, sources=[f"/w/tests/w8a82/{name}.cu"],
         extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG", "-DCUTLASS_ENABLE_GDC_FOR_SM100=0", *inc],
         extra_cflags=["-O3", *inc], verbose=False)
print("built", name, f"{time.time() - t:.0f}s", m)
