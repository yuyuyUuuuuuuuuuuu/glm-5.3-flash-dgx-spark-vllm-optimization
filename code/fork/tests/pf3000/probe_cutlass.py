"""PF3000 probe: is vLLM's cutlass_scaled_mm usable on this GPU (GB10, sm_121) for fp8 e4m3 W8A8?
Prints the _C op schema, the support predicates, and runs one small GEMM (M=2048 pieces, per-token
activation scale) against an fp32 reference. Run 1: tests/gpu_run.sh python3 tests/pf3000/probe_cutlass.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard  # noqa: E402
import torch  # noqa: E402


def per_token_quant(x):
    amax = x.abs().amax(dim=1).float().clamp(min=1e-12)
    scale = (amax / 448.0).unsqueeze(1)
    q = (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def main():
    gpu_guard(6.0)
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}, {p.name}, cc {p.major}.{p.minor}")
    import vllm._custom_ops as ops
    print("cutlass_scaled_mm_supports_fp8:", ops.cutlass_scaled_mm_supports_fp8(p.major * 10 + p.minor))
    try:
        print("block_fp8:", ops.cutlass_scaled_mm_supports_block_fp8(p.major * 10 + p.minor))
    except Exception as e:
        print("block_fp8 probe failed:", repr(e))
    print("op schema:", torch.ops._C.cutlass_scaled_mm._schemas if hasattr(torch.ops._C.cutlass_scaled_mm, "_schemas")
          else [o for o in dir(torch.ops._C.cutlass_scaled_mm)])
    for name in ("per_token_group_quant_fp8", "scaled_fp8_quant", "cutlass_fp8_group_gemm"):
        print(f"ops.{name}:", hasattr(ops, name))

    g = torch.Generator(device="cuda").manual_seed(0)
    M, K, N = 2048, 4096, 12576
    x = (torch.randn(M, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    q, sa = per_token_quant(x)
    wb = w.to(torch.float8_e4m3fn)                       # standard layout [N,K] row-major; b = .t() -> col-major [K,N]
    sb = (wb.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12).unsqueeze(1)  # per output channel [N,1]
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    for call in (
        lambda: ops.cutlass_scaled_mm(q, wb.t(), sa, sb, torch.bfloat16),
        lambda: torch.ops._C.cutlass_scaled_mm(out, q, wb.t(), sa, sb),
    ):
        try:
            y = call()
            ref = torch.mm(x.float(), (wb.float() * sb).t())
            rel = (y.float() - ref).norm() / ref.norm()
            print(f"call OK: shape {tuple(y.shape)} dtype {y.dtype}, rel_l2 vs fp32 ref {rel.item():.3e}")
        except Exception as e:
            print("call FAILED:", repr(e)[:300])
    err = torch.cuda.synchronize() or torch.cuda.get_last_error() if False else None
    print("status: done")


run_main(main)
