"""Does cuBLASLt (this image, GB10) offer a per-row alpha vector (CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_*) for a
bf16 GEMM with fp32 compute? That would fuse the FP8 per-channel scale into the GEMM epilogue. Log:
docs/logs/fp8_large_m/probe_cublaslt_alpha_vector.log."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import fp8_gemv as G  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
inc = G._include_shim()
E = load(name="tf_probe_cublaslt_alpha", sources=[str(Path(__file__).resolve().parent / "probe_cublaslt_alpha.cu")],
         extra_cflags=["-O3", *inc], extra_cuda_cflags=inc, extra_ldflags=["-lcublasLt"])
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("bf16 D algorithms:", E.algo_caps(False))
print("fp32 D algorithms:", E.algo_caps(True))
names = {0: "HOST", 1: "DEVICE", 2: "DEVICE_VECTOR", 3: "ALPHA_DEVICE_VECTOR_BETA_ZERO", 4: "ALPHA_DEVICE_VECTOR_BETA_HOST"}
for d32 in (False, True):
    for pm, nm in names.items():
        print(f"heuristic M=1791 N=12576 K=4096 D={'fp32' if d32 else 'bf16'} pointer mode {nm}:",
              E.heuristic(1791, 12576, 4096, d32, pm))
print("(status 15 = CUBLAS_STATUS_NOT_SUPPORTED; pointer_mode_mask 3 = HOST|DEVICE only)")
