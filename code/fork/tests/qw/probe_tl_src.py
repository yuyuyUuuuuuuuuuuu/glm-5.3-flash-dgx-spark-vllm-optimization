import sys, os, glob, torch
sys.path.insert(0, "/w")
import vllm.model_executor.kernels.mhc.tilelang as TL
T, H = 64, 4096
x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
res = torch.randn(T, 4, H, device="cuda").to(torch.bfloat16)
post = torch.rand(T, 4, 1, device="cuda")
comb = torch.rand(T, 4, 4, device="cuda")
TL.mhc_post_tilelang(x, res, post, comb)
torch.cuda.synchronize()
from vllm.model_executor.kernels.mhc import tilelang_kernels as K
k = K.mhc_post_tilelang
print(type(k), [a for a in dir(k) if not a.startswith("__")][:60])
for attr in ("get_kernel_source", "kernel_source"):
    pass
hits = []
for root in ("/tmp", os.path.expanduser("~")):
    for f in glob.glob(root + "/.tilelang/**/*", recursive=True):
        if f.endswith((".cu", ".cuh", ".py", ".json")) and os.path.isfile(f):
            try:
                t = open(f, errors="ignore").read()
            except Exception:
                continue
            if "mhc_post_tilelang_kernel" in t:
                hits.append(f)
print("hits", hits)
for f in [h for h in hits if h.endswith("device_kernel.cu")][:1]:
    t = open(f, errors="ignore").read()
    i = t.find("mhc_post_tilelang_kernel")
    print("=====", f)
    j = t.find("for (int i0_h"); print(t[j: j + 5000])
