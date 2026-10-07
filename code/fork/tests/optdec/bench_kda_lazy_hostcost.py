"""decode4 review: HOST cost of the eager commit (the part nodeC's step bench hides behind the graph replay).

Production calls commit() from MambaHybridModelState.postprocess_state, i.e. on the host critical path between
sampling and the drafter launch (HOSTLOOP keeps the drafter-to-target host gap at ~0.04 ms). This measures the wall
time of commit() on the host (no device sync inside the timed region; the GPU is kept busy by a long dummy kernel
queued first so device time cannot show up in the host number), for the production shape (34 layers, nseq 8,
TMAX 8), num_sampled as a tensor, align on.
Usage: bench_kda_lazy_hostcost.py [iters]
"""
import sys
import time

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402

it = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
dev = "cuda"
NL, H, KD, V, NSEQ, TMAX = 34, 32, 128, 128, 8, 8
L._ORIG["frk"] = prod_frk
L.allocate(NL, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True
NB = 64
st = torch.zeros(NB, H, V, KD, dtype=torch.float32, device=dev)
for l in range(NL):
    a = torch.randn(H, device=dev).float()
    b = torch.randn(H * KD, device=dev).float()
    # distinct a_log tensors on one state tensor (shared-tensor layout)
    L.register(st, a, b)
assert len(L.ST.layers) == NL, len(L.ST.layers)
import os  # noqa: E402
os.environ["GLM53_DEC_KDA_LAZY_VERIFY"] = "0"
os.environ["GLM53_DEC_KDA_LAZY_VERIFY_EVERY"] = "0"
nreq = 1
num_sampled = torch.full((nreq,), 8, dtype=torch.int32, device=dev)
idx_map = torch.arange(nreq, dtype=torch.int32, device=dev)
ncomp = torch.full((16,), 1000, dtype=torch.int32, device=dev)
big = torch.randn(8192, 8192, device=dev)
for mode in ("noalign", "align"):
    al = mode == "align"
    for _ in range(50):
        L.commit(num_sampled, idx_map, nreq, num_computed=ncomp, block_size=4608, align=al)
    torch.cuda.synchronize()
    ts = []
    for i in range(it):
        if i % 50 == 0:
            torch.cuda.synchronize()
            for _ in range(4):
                big @ big          # keep the device busy so the host-side launch cost is what is timed
        t0 = time.perf_counter()
        L.commit(num_sampled, idx_map, nreq, num_computed=ncomp, block_size=4608, align=al)
        ts.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    ts.sort()
    print(f"commit host time ({mode}, 34 layers, nreq {nreq}): p50 {ts[len(ts)//2]*1e6:.1f} us  p90 "
          f"{ts[int(len(ts)*0.9)]*1e6:.1f} us  min {ts[0]*1e6:.1f} us", flush=True)
# the bare zero_ of the flags + a trivial triton-free torch op for scale
ts = []
for i in range(it):
    t0 = time.perf_counter()
    L.ST.meta[:NL, :, 0].zero_()
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"flags zero_ host time: p50 {ts[len(ts)//2]*1e6:.1f} us", flush=True)
