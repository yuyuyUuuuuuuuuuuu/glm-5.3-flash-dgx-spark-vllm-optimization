"""Which arithmetic reproduces the image's per-token fp8 quant bit for bit (scale formula, x/s vs x*(1/s), rounding)?"""
import sys, torch
sys.path.insert(0, "/w")
import vllm._custom_ops as ops
import fp8_w8a8 as W
g = torch.Generator(device="cuda").manual_seed(1)
x = (torch.randn(4096, 4096, device="cuda", generator=g) * 0.5)
x[:, :16] *= 25
x = x.to(torch.bfloat16)
q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
s = s.float().reshape(-1, 1)
xf = x.float()
amax = xf.abs().amax(1, keepdim=True)
for nm, sc in (("amax/448", amax / 448.0), ("amax*(1/448)", amax * (1.0 / 448.0)),
               ("max(amax/448,min)", torch.maximum(amax / 448.0, torch.tensor(1 / (448 * 512.), device="cuda")))):
    print(nm, "scale equal:", torch.equal(sc, s), "max rel", ((sc - s).abs() / s).max().item())
for nm, y in (("x/s", xf / s), ("x*(1/s)", xf * (1.0 / s))):
    qt = y.clamp(-448, 448).to(torch.float8_e4m3fn)
    print(nm, "torch q bytes differ:", (qt.view(torch.uint8) != q.view(torch.uint8)).sum().item())
S = torch.arange(0, 128, dtype=torch.int32, device="cuda")
for inv in (True, False):
    W.HILO_INV["v"] = inv
    a2, sa = W.hilo_quant(x, S)
    print("triton INV", inv, "scale equal", torch.equal(sa, s), "q bytes differ",
          (a2[:, :4096].contiguous().view(torch.uint8) != q.view(torch.uint8)).sum().item())
    y = (xf * (1.0 / sa)) if inv else (xf / sa)
    qt = y.clamp(-448, 448).to(torch.float8_e4m3fn)
    print("   triton vs torch same arithmetic: differ", (a2[:, :4096].contiguous().view(torch.uint8) != qt.view(torch.uint8)).sum().item())
