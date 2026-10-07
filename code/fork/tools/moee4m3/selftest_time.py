import os, sys, time
sys.path[:0] = ["/w", "/w/tests", "/w/tests/moee4m3"]
import torch
import prefill_cap_common as C
import harness as H
from real_layer import make_real_layer
H.gpu_guard(8.0)
prod = H.load_prod(); H.load_xl()
import glm53_moe_e4m3 as M
L = make_real_layer(prod, torch.device("cuda", 0))
for i in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = M.selftest(prod, L, C.LIMIT)
    torch.cuda.synchronize(); print(f"selftest {i}: {(time.perf_counter()-t0)*1000:.1f} ms {r}", flush=True)
