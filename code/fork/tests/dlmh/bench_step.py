"""DEC_DLMH speed: the drafter's candidate head per TP rank, production vs two-stage, inside CUDA graphs, real lm_head
shard (77440 x 4096), real drafter hidden rows, interleaved rounds (order flipped every round), median us per call.

production (rank 0 part): fp8_gemv lm_head [T, 77440] -> (all-gather: not on one GPU; the other rank's half is a
static tensor) -> torch.topk over [T, 154880]
two-stage (rank 0 part):  coarse int4 GEMV -> topk C -> gather T*C Marlin columns -> fp8_gemv on them -> pack
-> (all-gather: the other rank's pack static) -> -inf fill + scatter -> torch.topk over [T, 154880]
The all-gather itself differs too ([T, 77440] bf16 = 1.08 MB per rank vs [T, 2C] int32 = 7 KB): production's R15
trace has 86 us for it; it is added in the projection, not measured here.

  GPU_RUN_RO=<common.RO> tests/gpu_run.sh python3 tests/dlmh/bench_step.py
"""
import statistics as st
import sys

import torch

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/dlmh")
import common as Cm  # noqa: E402
import glm53_dlmh as D  # noqa: E402

dev = "cuda"
G = Cm.enable_fp8_gemv()
import os
D.parse_env({"GLM53_DEC_DLMH": "1", "GLM53_DEC_DLMH_C": os.environ.get("GLM53_DEC_DLMH_C", "128")})
VL = Cm.V // 2
lm = Cm.load_lm_bf16()
holder, fp8, _ = Cm.fp8_holder(lm[:VL].contiguous())
del lm, fp8
torch.cuda.empty_cache()
hd = D._Head()
hd.holder, hd.tp, hd.rank, hd.vloc, hd.org = holder, 2, 0, VL, Cm.V
hd.coarse_w, hd.coarse_s = D.build_coarse_from_marlin(holder.weight, holder.weight_scale, VL, Cm.H, D.CFG.group)
X = Cm.probe_hidden()["L0.6"]
cfg_all = {}


def run_case(T):
    hd.bufs[T] = D._vmaps(T, D.CFG.C, dev)
    x = X[:T].contiguous().clone()
    other_logits = torch.randn(T, VL, device=dev).to(torch.bfloat16)
    other_pack = D.local_pack(x, hd).clone()
    other_pack[:, : 8 * D.CFG.C] += VL                       # pretend rank 1's ids

    def prod():
        lg = Cm.prod_logits(G, holder, x)
        full = torch.cat([lg, other_logits], dim=-1)[..., : Cm.V]
        return torch.topk(full, 16, dim=-1)

    def ours():
        p = D.local_pack(x, hd)
        return D.merge(torch.cat([p, other_pack], dim=-1), T, 16, 2, VL, Cm.V)

    def parts():
        yc = torch.empty((T, VL), dtype=torch.float32, device=dev)
        return yc

    graphs = {}
    for name, fn in (("production", prod), ("two-stage", ours)):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(4):
                fn()
        graphs[name] = g
    # coarse GEMV alone and the exact part alone (attribution)
    yc = torch.empty((T, VL), dtype=torch.float32, device=dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4):
            D.coarse_gemv(x, hd.coarse_w, hd.coarse_s, yc, D.CFG.group)
    graphs["  coarse GEMV only"] = g
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = {k: [] for k in graphs}
    for r in range(15):
        items = list(graphs.items())
        for k, g in (items if r % 2 == 0 else items[::-1]):
            g.replay()
            torch.cuda.synchronize()
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            res[k].append(e0.elapsed_time(e1) * 1000 / 4)
    p = st.median(res["production"])
    for k, v in res.items():
        print(f"T={T:2d} {k:22s} median {st.median(v):8.1f} us  (min {min(v):7.1f}, max {max(v):7.1f})"
              + (f"  saving {p - st.median(v):7.1f} us" if k == "two-stage" else ""), flush=True)


for T in (7, 14):
    run_case(T)
