"""DEC_DLMH exactness tests (nodeC, production image, real lm_head + real DFlash2 drafter hidden states).

U1 unpack of the Marlin FP8 layout == the e4m3 weight x bf16 scale (every element)
U2 coarse Triton GEMV == torch reference of the int4 copy
U3 gather + fp8_gemv on the gathered columns == the full fp8_gemv lm_head at those columns, bit for bit
   (T = 1, 5, 7, 8, 14, 16; random ids with duplicates; production's TABLE config for each T)
U4 torch.topk on a -inf-masked tensor that keeps every element >= the k-th value == torch.topk on the full tensor
   (ids and values), tie-heavy bf16 inputs
U5 end to end, TP=2 emulated (two 77440-row shards, all-gather = rank-major concat): candidates + unary logits ==
   production's (fp8_gemv per shard -> concat -> topk) on the real drafter's hidden states, every row, T = 7 and 14
U6 the same inside a CUDA graph (capture once, replay with new inputs) == eager production
U7 the installed DFlash2Qwen3ForCausalLM.compute_candidates (mode on / verify / off, setup + self-test) on a model
   stand-in with a 77440-row head: == the stock method; verify counters count 0 mismatches; T=21 -> production path

  GPU_RUN_RO=<common.RO> tests/gpu_run.sh python3 tests/dlmh/test_dlmh.py
"""
from __future__ import annotations

import sys
import types

import torch

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/dlmh")
import common as Cm  # noqa: E402
import glm53_dlmh as D  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 30 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
dev = "cuda"
G = Cm.enable_fp8_gemv()
D.parse_env({"GLM53_DEC_DLMH": "1"})
D.CFG.strict = True
FAILS = []


def check(name, ok, detail=""):
    print(f"[dlmh-test] {name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


lm = Cm.load_lm_bf16()
VL = Cm.V // 2
heads = []
for r in range(2):
    holder, fp8, sc = Cm.fp8_holder(lm[r * VL:(r + 1) * VL].contiguous())
    hd = D._Head()
    hd.holder, hd.tp, hd.rank, hd.vloc, hd.org = holder, 2, r, VL, Cm.V
    if r == 0:
        # U1
        bad = 0
        for a in range(0, VL, 8192):
            b = min(VL, a + 8192) // 64 * 64 if min(VL, a + 8192) != VL else VL
            u = D.unpack_marlin_fp8(holder.weight, holder.weight_scale, VL, Cm.H, a, b)
            ref = fp8[a:b].float() * sc[a:b].float()[:, None]
            bad += int((u != ref).sum())
        check("U1 unpack == e4m3 x scale", bad == 0, f"(mismatched elements {bad} of {VL * Cm.H})")
    del fp8
    hd.coarse_w, hd.coarse_s = D.build_coarse_from_marlin(holder.weight, holder.weight_scale, VL, Cm.H, D.CFG.group)
    for T in (7, 14):
        hd.bufs[T] = D._vmaps(T, D.CFG.C, dev)
    heads.append(hd)
del lm
torch.cuda.empty_cache()

# U2
x = (torch.randn(7, Cm.H, device=dev) * 1.5).to(torch.bfloat16)
y = torch.empty((7, VL), device=dev, dtype=torch.float32)
D.coarse_gemv(x, heads[0].coarse_w, heads[0].coarse_s, y, D.CFG.group)
ref = x.float() @ D.coarse_dequant(heads[0].coarse_w, heads[0].coarse_s, D.CFG.group).t()
rel = ((y - ref).norm() / ref.norm()).item()
check("U2 coarse GEMV vs reference", rel < 1e-5, f"(rel_l2 {rel:.2e})")
osc = torch.empty((7, VL // 8), device=dev, dtype=torch.float32)
D.coarse_octmax(x, heads[0].coarse_w, heads[0].coarse_s, osc, D.CFG.group)
ref_o = y.view(7, -1, 8, 8).amax(2).reshape(7, -1)
check("U2b fused octet-max coarse == octet max of the coarse GEMV", torch.equal(osc, ref_o),
      f"(max abs diff {(osc - ref_o).abs().max().item():.3e})")

# U3
h0 = heads[0].holder
g = torch.Generator(device=dev).manual_seed(11)
bad3 = 0
for T in (1, 5, 7, 8, 14, 16):
    cfg = G.select_config(VL, Cm.H, T)
    xs = (torch.randn(T, Cm.H, generator=g, device=dev) * 2).to(torch.bfloat16)
    full = Cm.prod_logits(G, h0, xs)
    for U in (64, 448, T * D.CFG.C):
        ids = torch.randint(0, VL, (U,), generator=g, device=dev, dtype=torch.int32)
        ids[: U // 8] = ids[U // 8: U // 4]             # duplicates
        ids[-1] = VL - 1
        ids[0] = 0
        dw = torch.empty((Cm.H // 16, 4 * U), dtype=torch.int32, device=dev)
        ds = torch.empty((U,), dtype=torch.bfloat16, device=dev)
        D.gather_marlin(h0.weight, h0.weight_scale, ids, dw, ds)
        yy = torch.empty((T, U), dtype=torch.bfloat16, device=dev)
        G.STATE.ext.fp8_gemv_out(yy, xs, dw, ds, None, U, Cm.H, *cfg, True)
        n = int((yy.view(torch.int16) != full[:, ids.long()].view(torch.int16)).sum())
        bad3 += n
        if n:
            print(f"   U3 T={T} U={U} cfg={cfg}: {n} mismatches", flush=True)
check("U3 gathered fp8_gemv == full head columns (bitwise)", bad3 == 0, f"(mismatches {bad3})")

# U3b: the octet-indirect kernel (kernels/dlmh_gemv.cu) reading the weight in place == fp8_gemv on the full weight
bad3b = 0
DX = D._dlmh_ext()
for T in (1, 5, 7, 8, 14, 16):
    cfg = G.select_config(VL, Cm.H, T)
    xs = (torch.randn(T, Cm.H, generator=g, device=dev) * 2).to(torch.bfloat16)
    full = Cm.prod_logits(G, h0, xs)
    for no in (8, 64, 7 * 64, 16 * 64):
        octs = torch.randint(0, VL // 8, (no,), generator=g, device=dev)
        octs[: no // 4] = octs[no // 4: no // 2]
        octs[0], octs[-1] = 0, VL // 8 - 1
        nv = 8 * no
        vc = torch.arange(nv, device=dev)
        o = octs[8 * (vc // 64) + vc % 8]
        src = 64 * (o // 8) + o % 8 + 8 * ((vc % 64) // 8)
        for warps in sorted({cfg[0], 4}):
            yy = torch.empty((T, nv), dtype=torch.bfloat16, device=dev)
            DX.dlmh_gemv_out(yy, xs, h0.weight, h0.weight_scale.view(-1), octs.contiguous(), nv, Cm.H, warps, cfg[1],
                             bool(cfg[3]))
            n = int((yy.view(torch.int16) != full[:, src].view(torch.int16)).sum())
            bad3b += n
            if n:
                print(f"   U3b T={T} octets={no} cfg={cfg} warps={warps}: {n} mismatches", flush=True)
check("U3b octet-indirect kernel == full head columns (bitwise)", bad3b == 0, f"(mismatches {bad3b})")

# U4
bad4 = 0
trials = 0
for t in range(300):
    T = 7
    levels = [4, 16, 64, 1024][t % 4]
    base = torch.randint(0, levels, (T, Cm.V), generator=g, device=dev).float() / levels * 8 - 4
    lg = base.to(torch.bfloat16)
    if t % 3 == 0:
        lg[:, torch.randint(0, Cm.V, (40,), generator=g, device=dev)] = lg.max()   # ties at the very top
    v0, i0 = torch.topk(lg, 16, dim=-1)
    kth = v0[:, -1:]
    keep = (lg >= kth) | (torch.rand(lg.shape, generator=g, device=dev) < 0.002)
    masked = torch.where(keep, lg, torch.full_like(lg, float("-inf")))
    v1, i1 = torch.topk(masked, 16, dim=-1)
    trials += 1
    bad4 += int(not (torch.equal(i0, i1) and torch.equal(v0.view(torch.int16), v1.view(torch.int16))))
check("U4 masked topk == full topk (tie-heavy)", bad4 == 0, f"({bad4} of {trials} trials differ)")

# U5 / U6
X = Cm.probe_hidden()


def prod_cands(xs):
    lg = torch.cat([Cm.prod_logits(G, hd.holder, xs) for hd in heads], dim=-1)[..., : Cm.V]
    v, i = torch.topk(lg, 16, dim=-1)
    return i, v


def ours(xs):
    T = xs.shape[0]
    packs = torch.cat([D.local_pack(xs, hd) for hd in heads], dim=-1)
    return D.merge(packs, T, 16, 2, VL, Cm.V)


res5 = {}
for Ctry in (64, 96, 128, 192):
    D.CFG.C = Ctry
    for hd in heads:
        hd.bufs = {T: D._vmaps(T, Ctry, dev) for T in (7, 14)}
    rows = bad5 = 0
    for name, xs_all in X.items():
        for T in (7, 14):
            for a in range(0, xs_all.shape[0] - T + 1, T):
                xs = xs_all[a:a + T].contiguous()
                i0, v0 = prod_cands(xs)
                i1, v1 = ours(xs)
                d = ((i0 != i1) | (v0.view(torch.int16) != v1.view(torch.int16))).any(-1)
                rows += T
                bad5 += int(d.sum())
    res5[Ctry] = (rows, bad5)
    print(f"   U5 C={Ctry} octets/row: rows {rows}, mismatched rows {bad5}", flush=True)
D.parse_env({"GLM53_DEC_DLMH": "1"})
for hd in heads:
    hd.bufs = {T: D._vmaps(T, D.CFG.C, dev) for T in (7, 14)}
check(f"U5 two-stage == production candidates (TP=2, real drafter hidden, default C={D.CFG.C})",
      res5[D.CFG.C][1] == 0, f"(rows {res5[D.CFG.C][0]}, mismatched {res5[D.CFG.C][1]}, sets {list(X)}; by C {res5})")

xs_static = X["L0.6"][:7].contiguous().clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    ours(xs_static)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    gi, gv = ours(xs_static)
bad6 = 0
for name in ("L0", "L0.9", "random_dir", "L0.45"):
    for a in (0, 70, 210):
        xs_static.copy_(X[name][a:a + 7])
        graph.replay()
        i0, v0 = prod_cands(xs_static)
        bad6 += int(not (torch.equal(i0, gi) and torch.equal(v0.view(torch.int16), gv.view(torch.int16))))
check("U6 CUDA-graph replay == production", bad6 == 0, f"({bad6} of 12 replays differ)")

# U7: the installed hook on the real vLLM class, model stand-in with one 77440-row head (TP=1, org = 77440)
from vllm.model_executor.models import qwen3_dflash2 as Q  # noqa: E402

stock = Q.DFlash2Qwen3ForCausalLM.compute_candidates
D._install_model()
hook = Q.DFlash2Qwen3ForCausalLM.compute_candidates


class _QM:
    def __init__(self, holder):
        self.holder = holder

    def apply(self, layer, x, bias=None):
        return Cm.prod_logits(G, self.holder, x)


class _Proc:
    soft_cap, scale, logits_as_input, use_all_gather = None, 1.0, False, True
    org_vocab_size = VL

    def __call__(self, lm_head, hs):
        return lm_head.quant_method.apply(lm_head, hs)[..., : self.org_vocab_size]


fake = types.SimpleNamespace()
fake.lm_head = types.SimpleNamespace(tp_size=1, glm53_fp8_head=heads[0].holder, quant_method=_QM(heads[0].holder))
fake.candidate_logits_processor = _Proc()
fake.model = types.SimpleNamespace(candidate_selector=types.SimpleNamespace(top_k=16))
D.HEAD.__init__()
bad7 = 0
for mode in ("1", "verify"):
    D.parse_env({"GLM53_DEC_DLMH": mode})
    for name in ("L0", "L0.75", "random_dir"):
        for T in (7, 14, 21):
            xs = X[name][:T].contiguous()
            c0, u0 = stock(fake, xs)
            c1, u1 = hook(fake, xs)
            bad7 += int(not (torch.equal(c0, c1) and torch.equal(u0.view(torch.int16), u1.view(torch.int16))))
torch.cuda.synchronize()
cnt = D.HEAD.counters.tolist()
check("U7 installed hook == stock compute_candidates (on / verify)", bad7 == 0 and D.HEAD.ready and cnt[2] == 0
      and cnt[0] == 6 and cnt[3] == 6, f"(differ {bad7}, counters calls/rows/mismatch/served {cnt}, "
      f"summary {D.summary()})")

print(f"[dlmh-test] {'ALL OK' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
sys.exit(1 if FAILS else 0)
