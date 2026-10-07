"""DEC_DLMH adversarial review checks (nodeC, production image, real lm_head shards, real DFlash2 drafter hidden rows).

R1 memory: persistent and TRANSIENT (peak) device memory of the coarse build at production's shard (77440 x 4096); the
   rollout doc only states the persistent +166.5 MiB, but the build runs in the profile run on hosts with 3-6 GB free
R2 tie stress: the probe rows scaled x1.5 / x2.5 / x4 (larger logits -> coarser bf16 grid -> many more exact ties at
   the 16th value) and mixed-regime batches (each request's 7 rows from a different confidence set), TP=2 emulated,
   C = 128: two-stage == production candidates + unary logits, every row; a mismatch is classified (tie / recall)
R3 the bench bias: bench_step.py's production arm pays a torch.cat of [T, 154880] bf16 that production's all-gather
   does not; its cost in a CUDA graph (T = 7, 14) is the overstatement of the saving

  GPU_RUN_RO=<common.RO> tests/gpu_run.sh python3 tests/dlmh/review_adv.py
"""
from __future__ import annotations

import statistics as st
import sys

import torch

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/dlmh")
import common as Cm  # noqa: E402
import glm53_dlmh as D  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 30 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
dev = "cuda"
G = Cm.enable_fp8_gemv()
D.parse_env({"GLM53_DEC_DLMH": "1"})
VL = Cm.V // 2
lm = Cm.load_lm_bf16()
heads = []
mem = []
for r in range(2):
    holder, fp8, _ = Cm.fp8_holder(lm[r * VL:(r + 1) * VL].contiguous())
    del fp8
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a0 = torch.cuda.memory_allocated()
    r0 = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    hd = D._Head()
    hd.holder, hd.tp, hd.rank, hd.vloc, hd.org = holder, 2, r, VL, Cm.V
    chunk = 4096 if r == 0 else D.BUILD_CHUNK_ROWS     # rank 0: the original chunk, rank 1: the module's setup chunk
    hd.coarse_w, hd.coarse_s = D.build_coarse_from_marlin(holder.weight, holder.weight_scale, VL, Cm.H, D.CFG.group,
                                                          chunk=chunk)
    torch.cuda.synchronize()
    mem.append(((torch.cuda.memory_allocated() - a0) / 2**20, (torch.cuda.max_memory_allocated() - a0) / 2**20,
                (torch.cuda.memory_reserved() - r0) / 2**20))
    hd.bufs = {7: (), 14: ()}
    heads.append(hd)
del lm
torch.cuda.empty_cache()
ref_w, ref_s = D.build_coarse_from_marlin(heads[1].holder.weight, heads[1].holder.weight_scale, VL, Cm.H, D.CFG.group,
                                         chunk=4096)
print(f"[dlmh-review] R1 coarse bytes chunk 4096 == chunk {D.BUILD_CHUNK_ROWS}: "
      f"{torch.equal(ref_w, heads[1].coarse_w) and torch.equal(ref_s.view(torch.int16), heads[1].coarse_s.view(torch.int16))}",
      flush=True)
del ref_w, ref_s
for r, (pers, peak, resv) in enumerate(mem):
    print(f"[dlmh-review] R1 rank {r} (chunk {4096 if r == 0 else D.BUILD_CHUNK_ROWS}) coarse build: persistent +{pers:.1f} MiB, peak during build +{peak:.1f} MiB, "
          f"allocator reserved after build (no empty_cache) +{resv:.1f} MiB", flush=True)

X = Cm.probe_hidden()


def prod_cands(xs):
    lg = torch.cat([Cm.prod_logits(G, hd.holder, xs) for hd in heads], dim=-1)[..., : Cm.V]
    v, i = torch.topk(lg, 16, dim=-1)
    return i, v, lg


def ours(xs):
    T = xs.shape[0]
    packs = torch.cat([D.local_pack(xs, hd) for hd in heads], dim=-1)
    return D.merge(packs, T, 16, 2, VL, Cm.V)


tot = bad = ties16 = 0
detail = []
names = list(X)
g = torch.Generator(device=dev).manual_seed(5)
cases = []
for sc in (1.5, 2.5, 4.0):
    for name in names:
        cases.append((f"{name}x{sc}", (X[name].float() * sc).to(torch.bfloat16)))
# mixed regimes: request j of a T=14 batch from set j, rows shuffled within the set
mix = []
for a in range(0, 448 - 7 + 1, 7):
    for j, name in enumerate(names):
        mix.append(X[name][a:a + 7])
cases.append(("mixed", torch.cat(mix, 0)))
for name, xs_all in cases:
    nb = 0
    for T in (7, 14):
        for a in range(0, xs_all.shape[0] - T + 1, T):
            xs = xs_all[a:a + T].contiguous()
            i0, v0, lg = prod_cands(xs)
            i1, v1 = ours(xs)
            d = ((i0 != i1) | (v0.view(torch.int16) != v1.view(torch.int16))).any(-1)
            kth = v0[:, -1:]
            ties16 += int(((lg >= kth).sum(-1) > 16).sum())
            tot += T
            nb += int(d.sum())
            for rr in d.nonzero().view(-1).tolist()[:2]:
                need = int((lg[rr] >= kth[rr]).sum())
                eq = int((lg[rr] == kth[rr]).sum())
                detail.append(f"{name} T={T} row {a + rr}: elements >= 16th {need}, tied at it {eq}, "
                              f"same set {set(i0[rr].tolist()) == set(i1[rr].tolist())}")
    bad += nb
    print(f"   R2 {name:18s} mismatched rows {nb}", flush=True)
for s in detail[:12]:
    print("   R2 miss:", s, flush=True)
print(f"[dlmh-review] R2 tie stress (C={D.CFG.C}): rows {tot}, rows with ties at the 16th value {ties16} "
      f"({100.0 * ties16 / max(tot, 1):.1f}%), mismatched rows {bad}", flush=True)

# R3: the torch.cat overstatement of bench_step's production arm
for T in (7, 14):
    a = torch.randn(T, VL, device=dev).to(torch.bfloat16)
    b = torch.randn(T, VL, device=dev).to(torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cat([a, b], dim=-1)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(4):
            torch.cat([a, b], dim=-1)[..., : Cm.V]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    res = []
    for _ in range(15):
        gr.replay()
        torch.cuda.synchronize()
        e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
        res.append(e0.elapsed_time(e1) * 1000 / 4)
    print(f"[dlmh-review] R3 T={T}: torch.cat [T, 154880] bf16 in bench_step's production arm costs median "
          f"{st.median(res):.1f} us (overstates the saving by about this much)", flush=True)
print("[dlmh-review] done", flush=True)
