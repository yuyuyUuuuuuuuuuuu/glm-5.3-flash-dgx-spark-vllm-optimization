"""実物で確かめる: 同じ合成 trellis を ExLlamaV3 reconstruct と TensorFold unpack で復号し、ビット一致するか。
さらに線形層 y = x @ W の全経路(suh→had→W_q→had→svh)を ExLlamaV3 と TF 参照(float64)で比べる。"""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_format_ref as R
from prod_baseline import load_xl
import harness as H

H.gpu_guard(8.0)
xl = load_xl(); dev = "cuda"
g = torch.Generator().manual_seed(1)
failures = []
for (K, N) in ((256, 256), (4096, 768), (768, 4096)):
    tr = torch.randint(-2**15, 2**15 - 1, (K // 16, N // 16, 64), dtype=torch.int16, generator=g)
    w_xl = torch.empty((K, N), dtype=torch.half, device=dev)
    xl.reconstruct(w_xl, tr.to(dev), 4, True, False)
    w_tf = R.unpack(tr, 4)
    eq = torch.equal(w_xl.cpu(), w_tf)
    nd = int((w_xl.cpu() != w_tf).sum())
    print(f"reconstruct K={K} N={N}: bit-equal={eq}  differing={nd}/{K*N}")
    if not eq:
        failures.append(f"reconstruct K={K} N={N} not bit-equal")

    # 線形層全経路: ExLlamaV3 (had_r_128 + hgemm + had_r_128*svh) vs TF float64 参照
    suh = ((torch.randint(0, 2, (K,), generator=g) * 2 - 1).half())
    svh = (torch.rand(N, generator=g).half() * 0.02 + 0.01)
    x = torch.randn(3, K, generator=g).half()
    xh = torch.empty_like(x, device=dev)
    xl.had_r_128(x.to(dev), xh, suh.to(dev), None, 1.0)
    y = torch.empty((3, N), dtype=torch.half, device=dev)
    xl.hgemm(xh, w_xl, y)
    yo = torch.empty_like(y)
    xl.had_r_128(y, yo, None, svh.to(dev), 1.0)
    ref = R.forward(x, tr, suh, svh, 4)
    rel = float((yo.double().cpu() - ref).norm() / ref.norm())
    print(f"  linear y=xW  ExLlamaV3 vs TF-ref(float64): rel err {rel:.2e}")
    if not rel <= 1e-3:   # three fp16 roundings (xh, y, yo) ~ 2-3 u16; a layout error would be O(1)
        failures.append(f"linear K={K} N={N} rel err {rel:.2e} > 1e-3")

H.report_peak()
if failures:
    print("RESULT: FAIL", failures)
    sys.exit(1)
print("RESULT: PASS")
