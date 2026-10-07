"""Print the device CUDA source TileLang generates for mhc_post_tilelang (to mirror its arithmetic bit for bit)."""
import glob, os, sys
os.environ["TILELANG_CACHE_DIR"] = "/w/.tlcache_probe"
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mhc_sp"))
from bench_mhc_halving import make
from vllm.model_executor.kernels.mhc.tilelang_kernels import mhc_post_tilelang
M = 64
w = make(M, seed=3)
rc = torch.empty_like(w["res"])
mhc_post_tilelang(w["comb"], w["res"], w["post"].view(M, 4), w["x"], rc, 4, 4096)
torch.cuda.synchronize()
for k in dir(mhc_post_tilelang):
    if "cache" in k.lower():
        print("attr", k)
done = False
for obj in [getattr(mhc_post_tilelang, k) for k in dir(mhc_post_tilelang) if "cache" in k.lower()]:
    try:
        items = obj.values() if hasattr(obj, "values") else []
        for kern in items:
            src = kern.get_kernel_source()
            i = src.find("__global__")
            print(src[i:i + 6000]); done = True
    except Exception as e:
        print("attr err", e)
if not done:
    for f in glob.glob("/w/.tlcache_probe/**/*", recursive=True):
        if os.path.isfile(f) and f.endswith((".cu", ".cuh", ".c", ".txt")) or "kernel" in os.path.basename(f):
            try:
                s = open(f, errors="ignore").read()
            except Exception:
                continue
            if "mhc_post" in s and "__global__" in s:
                i = s.find("__global__")
                print("FILE", f); print(s[i:i + 6000]); break
