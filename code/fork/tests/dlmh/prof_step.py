"""kernel breakdown of the two-stage candidate head (T=7) with torch.profiler (CUPTI), eager after warmup."""
import sys
import torch
sys.path.insert(0, "/w"); sys.path.insert(0, "/w/tests/dlmh")
import common as Cm
import glm53_dlmh as D
dev = "cuda"
G = Cm.enable_fp8_gemv()
D.parse_env({"GLM53_DEC_DLMH": "1"})
VL = Cm.V // 2
lm = Cm.load_lm_bf16()
holder, fp8, _ = Cm.fp8_holder(lm[:VL].contiguous())
del lm, fp8
hd = D._Head()
hd.holder, hd.tp, hd.rank, hd.vloc, hd.org = holder, 2, 0, VL, Cm.V
hd.coarse_w, hd.coarse_s = D.build_coarse_from_marlin(holder.weight, holder.weight_scale, VL, Cm.H, D.CFG.group)
T = 7
hd.bufs[T] = ()  #
x = (torch.randn(T, Cm.H, device=dev) * 1.5).to(torch.bfloat16)
op = D.local_pack(x, hd).clone()
def ours():
    p = D.local_pack(x, hd)
    return D.merge(torch.cat([p, op], dim=-1), T, 16, 2, VL, Cm.V)
def prod():
    lg = Cm.prod_logits(G, holder, x)
    return torch.topk(torch.cat([lg, lg], -1), 16, dim=-1)
for _ in range(5):
    ours(); prod()
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
for name, fn in (("two-stage", ours), ("production", prod)):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    print("=====", name)
    for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:30]:
        print("%9.1f us/call  n=%4d  %s" % (e.device_time_total / 10, e.count, e.key[:100]))
