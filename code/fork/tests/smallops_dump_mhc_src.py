"""Dump the CUDA source TileLang generates for production's decode mHC kernels (mhc_fused_tilelang,
mhc_pre_big_fuse_with_norm_tilelang) so the GLM53_DEC_SMALLOPS mHC kernel can replicate their arithmetic.
Run: tests/gpu_run.sh python3 tests/smallops_dump_mhc_src.py"""
import os
import torch
from vllm.model_executor.kernels.mhc import tilelang as W
from vllm.model_executor.kernels.mhc import tilelang_kernels as K

out = "/w/docs/logs/smallops/mhc_src"
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")
hc, H = 4, 4096
n3 = hc * (2 + hc)
print("ENABLE_PDL", K.ENABLE_PDL)
for M in (5, 8, 16):
    torch.manual_seed(0)
    x = torch.randn(M, H, device=dev).bfloat16()
    res = torch.randn(M, hc, H, device=dev).bfloat16()
    post = torch.rand(M, hc, 1, device=dev)
    comb = torch.rand(M, hc, hc, device=dev)
    fn = torch.randn(n3, hc * H, device=dev) * 0.01
    scale = torch.rand(3, device=dev)
    base = torch.randn(n3, device=dev) * 0.1
    nw = torch.rand(H, device=dev).bfloat16()
    r = W.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, nw, 1e-5)
    torch.cuda.synchronize()
tile_n = {5: 2, 8: 3, 16: 3}
for name, fn_ in (("mhc_fused_tilelang", K.mhc_fused_tilelang),
                  ("mhc_pre_big_fuse_with_norm_tilelang", K.mhc_pre_big_fuse_with_norm_tilelang)):
    for i, (key, kern) in enumerate(fn_._kernel_cache.items()):
        src = kern.get_kernel_source()
        path = f"{out}/{name}_{i}.cu"
        with open(path, "w") as f:
            f.write("// key: " + repr(key)[:2000].replace("\n", " ") + "\n")
            f.write(src)
        print(name, i, len(src), path, repr(key)[:300])
