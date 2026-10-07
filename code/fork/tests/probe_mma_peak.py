"""Tensor-core issue-rate ceiling on this GB10 (mma.sync bf16 m16n8k16 / fp8 m16n8k32, fp32 accumulate)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mla_prefill_common as C
ext = C.build_ext("mma_peak_ext", "kernels/mla_prefill/mma_peak.cu")
sms = torch.cuda.get_device_properties(0).multi_processor_count
print("SMs", sms)
for kind, name in ((0, "bf16 m16n8k16 f32acc"), (1, "e4m3 m16n8k32 f32acc"), (2, "f16 m16n8k16 f16acc")):
    for warps in (4, 8, 16):
        print(f"{name:22s} {warps:2d} warps/SM x1 CTA: {ext.run(kind, sms, warps * 32, 4000):7.1f} TFLOPS")
