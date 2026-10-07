"""Candidate scale formulas vs the image's per-token scale (Triton variants of the division)."""
import torch, triton, triton.language as tl
import vllm._custom_ops as ops


@triton.jit
def k(X, OUT, K: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, K)
    x = tl.load(X + row * K + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    if MODE == 0:
        s = amax / 448.0
    elif MODE == 1:
        s = tl.math.div_rn(amax, 448.0)
    elif MODE == 2:
        r = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [tl.full([1], 448.0, tl.float32)],
                                      dtype=tl.float32, is_pure=True, pack=1)
        s = tl.sum(amax * r, axis=0)
    else:
        s = amax * 0.002232142857142857
    tl.store(OUT + row, s)


g = torch.Generator(device="cuda").manual_seed(1)
x = (torch.randn(4096, 4096, device="cuda", generator=g) * 0.5)
x[:, :16] *= 25
x = x.to(torch.bfloat16)
q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
s = s.float().reshape(-1)
for mode in range(4):
    o = torch.empty(4096, device="cuda")
    k[(4096,)](x, o, K=4096, MODE=mode)
    print("mode", mode, "scale rows differing", (o != s).sum().item())
amax = x.float().abs().amax(1)
print("torch amax/448 rows differing", (amax / 448.0 != s).sum().item(), "examples", s[:3].tolist(), (amax / 448)[:3].tolist())
