"""Probe: does a TileLang kernel (tvm_ffi adapter) launch on torch's current stream? And its generated CUDA source."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import tilelang  # noqa: E402
from probe_tilelang_gemm import make  # noqa: E402

N, K = 12288, 4096
k = tilelang.compile(make(N, K, 128, 256, 32, 4, 256, 10), out_idx=None, target="cuda")
src = k.get_kernel_source()
Path("/w/.ext").mkdir(exist_ok=True)
Path("/w/.ext/tl_gemm_scale.cu").write_text(src)
print("kernel source lines:", len(src.splitlines()))
print("\n".join(l for l in src.splitlines() if "#include" in l or "__global__" in l or "extern" in l)[:1500])
b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
s = torch.rand(N, device="cuda") * 1e-3
M = 4096
side = torch.cuda.Stream()
ok = True
for trial in range(5):
    a = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)           # ~ tens of ms of spin on the side stream
        a.normal_()                             # the input is written on the side stream, after the spin
        k(a, b, s, c)                           # must run after a.normal_() if it uses the current (side) stream
        ref = (torch.mm(a, b.t(), out_dtype=torch.float32) * s).to(torch.bfloat16)
    torch.cuda.synchronize()
    ok &= torch.equal(c, ref)
print("runs on torch's current stream (5 trials, input produced on a busy side stream):", ok)
