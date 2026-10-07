"""ncu target: a few eager calls of production's mhc_fused_tilelang and the smallops mhc_fused (M from argv)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glm53_smallops as SO
from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_fused_tilelang, mhc_pre_big_fuse_with_norm_tilelang
dev = torch.device("cuda")
M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
hc, H, n3 = 4, 4096, 24
S = SO.mhc_splits(M); tile_n = 2 if M < 8 else 3
L = 40
ws = [(torch.randn(n3, hc * H, device=dev) * 0.02).bfloat16().float() for _ in range(L)]
wb = [w.bfloat16().contiguous() for w in ws]
x = torch.randn(M, H, device=dev).bfloat16(); res = torch.randn(M, hc, H, device=dev).bfloat16()
post = torch.rand(M, hc, device=dev); comb = torch.rand(M, hc, hc, device=dev)
yp = torch.empty(S, M, n3, device=dev); rp = torch.empty(S, M, device=dev); ro = torch.empty_like(res)
for i in range(L):
    mhc_fused_tilelang(comb, res, post, x, ws[i].view(n3, hc, H), yp, rp, ro, hc, H, n3, tile_n=tile_n, n_splits=S)
for i in range(L):
    SO.mhc_fused(comb.view(M, 16), post, res, x, ws[i], S, yp, rp, ro)
for i in range(L):
    SO.mhc_fused(comb.view(M, 16), post, res, x, wb[i], S, yp, rp, ro)
torch.cuda.synchronize()
print("done")
