"""opt-dense-rev (refuter): WHY did production's idx_gate check fail at the M=16384 profile run?
opt-dense assumes non-finite dummy activations. Probe the Triton head gate (glm53_prefill_quickwins.qw_head_gate) vs
production's torch.mm(x.float(), w32) at M = 16384 / 13824 on input families a profile run can produce:
normal, identical rows (embedding of token 0 repeated), zeros, bf16 subnormals, huge finite values, a NaN/inf row.
Prints bitwise equality, whether production's result is finite (-> opt-dense's 'nonfinite' verdict), else 'bad'.
Run: tests/gpu_run.sh python3 tests/optdense_rev/probe_gate_profile.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import glm53_prefill_quickwins as Q  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(5)
wb = (torch.randn(32, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16)
w32 = wb.t().contiguous().float()


def fam(name, M):
    if name == "normal":
        return (torch.randn(M, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
    if name == "identical_rows":
        r = (torch.randn(1, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
        return r.expand(M, 4096).contiguous()
    if name == "zeros":
        return torch.zeros(M, 4096, device=dev, dtype=torch.bfloat16)
    if name == "subnormal":
        x = (torch.randn(M, 4096, device=dev, generator=g) * 1e-39).to(torch.bfloat16)
        return x
    if name == "mixed_subnormal":
        x = (torch.randn(M, 4096, device=dev, generator=g) * 0.7)
        m = torch.rand(M, 4096, device=dev, generator=g) < 0.1
        x[m] = x[m] * 1e-39
        return x.to(torch.bfloat16)
    if name == "huge":
        return (torch.randn(M, 4096, device=dev, generator=g) * 1e36).to(torch.bfloat16)
    if name == "garbage_bits":
        return torch.randint(-32768, 32767, (M, 4096), device=dev, dtype=torch.int16, generator=g).view(torch.bfloat16)
    if name == "nan_row":
        x = (torch.randn(M, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
        x[3, 5] = float("nan")
        return x
    raise ValueError(name)


for M in (16384, 13824):
    for name in ("normal", "identical_rows", "zeros", "subnormal", "mixed_subnormal", "huge", "garbage_bits",
                 "nan_row"):
        x = fam(name, M)
        got = Q.qw_head_gate(x, w32)
        ref = torch.mm(x.float(), w32)
        fin = bool(torch.isfinite(ref).all())
        eq = torch.equal(got, ref)
        eqn = torch.equal(torch.nan_to_num(got, 1.0, 2.0, 3.0), torch.nan_to_num(ref, 1.0, 2.0, 3.0))
        ndiff = int((torch.nan_to_num(got, 1.0, 2.0, 3.0) != torch.nan_to_num(ref, 1.0, 2.0, 3.0)).sum())
        verdict = "nonfinite" if not fin else ("ok" if eq else "BAD")
        print(f"M={M} {name:16s} prod finite {fin!s:5s} torch.equal {eq!s:5s} equal-modulo-NaN {eqn!s:5s} "
              f"differing {ndiff:8d} -> opt-dense verdict {verdict}", flush=True)
