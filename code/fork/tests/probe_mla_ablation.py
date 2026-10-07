"""Timing ablations of the exact MLA prefill kernel (wrong results by design; never used for parity).
GLM53_MLA_DBG bits: 1 skip fp8->bf16 conversion, 2 skip the S-exchange barriers, 4 skip the KV pipeline (IO warp
idle, math warps read whatever is in smem), 8 skip QK mma, 16 skip PV mma, 32 skip the whole key loop (per-CTA
prologue + epilogue only), 64 skip the softmax math (v3).  Usage: probe_mla_ablation.py 0 1 2 ..."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mla_prefill_common as C
case = C.Case(13824, 0, "sticky", seed=1)
flop = 4.0 * case.pairs() * 32 * 512
out = torch.empty(13824, 32, 512, dtype=torch.bfloat16, device="cuda")
for bits in [int(b) for b in sys.argv[1:]]:
    ext = C.build_ext(f"mla_dbg{bits}", "kernels/mla_prefill/mla_prefill.cu", [f"-DGLM53_MLA_DBG={bits}"] +
                      (["--ptxas-options=-v"] if bits == 0 else []))
    f = lambda: ext.run(case.q, case.cache.view(-1, 512), case.slots, case.valid, out, C.SM_SCALE, 1.0)
    med, best = C.cuda_time(f, 2, 8)
    print(f"DBG={bits:3d}: {med:7.3f} ms (best {best:7.3f})  {flop / med / 1e9:5.1f} TFLOPS", flush=True)
