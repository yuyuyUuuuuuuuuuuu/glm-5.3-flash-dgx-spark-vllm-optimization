"""Unit tests of glm53_spec_vtrim (docs/SPEC_VTRIM.md) on nodeC's GPU, production image (tests/gpu_run.sh).

U1 nstar_kernel == nstar_reference (float64 torch) on random realized selector scores / candidates / drafts:
   greedy rows (draft = argmax), sampled rows (draft = a random candidate), temperatures 0 / 0.6 / 1.0 / 1.7,
   padded rows (sample_idx_mapping -1: n* untouched), TAU in {0, 0.05, 0.25, 0.6, 0.95, 1.0}, MIN in {0, 2};
   also the exactness-relevant invariant: n* does NOT depend on the draft at position n*+1 (re-drawing d_{n*+1}
   leaves n* unchanged) - the decision for row i never reads d_i.
U2 live_kernel == reference on random batch layouts (decode requests with 1 + K rows, K in {4, 5, 7}; prefill
   requests with one logits row; padded token rows after the batch keep LIVE = 1).
U3 the MoE hook: dead rows' routed ids become -1, live rows' ids untouched, prefill-sized calls untouched; the
   rejection hook masks exactly the rows with local pos >= 1 and > n*.
U4 the hooks are inert when the module is off (install(mode="off") installs nothing).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import glm53_spec_vtrim as VT  # noqa: E402

dev = torch.device("cuda")
FAIL = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def u1():
    print("== U1 nstar kernel vs reference")
    g = torch.Generator(device="cpu").manual_seed(7)
    S, K16 = 7, 16
    bad = 0
    total = 0
    indep_bad = 0
    for trial in range(60):
        R = int(torch.randint(1, 9, (1,), generator=g))
        max_reqs = 12
        scores = (torch.randn(R * S * K16, generator=g) * float(torch.rand(1, generator=g) * 4 + 0.3)).to(dev)
        cand = torch.stack([torch.randperm(154880, generator=g)[:K16] for _ in range(R * S)]).to(dev)
        greedy = bool(trial % 2)
        sc = scores.view(R * S, K16)
        if greedy:
            pick = sc.argmax(-1)
        else:
            pick = torch.randint(0, K16, (R * S,), generator=g).to(dev)
        toks = cand.gather(1, pick[:, None])[:, 0].contiguous()
        rs_of = torch.randperm(max_reqs, generator=g)[:R].to(torch.int32)
        sidx = rs_of.repeat_interleave(S).to(dev)
        if R > 1 and trial % 5 == 0:
            sidx[(R - 1) * S:] = -1                                    # a padded row
        temp = torch.tensor([[0.0, 0.6, 1.0, 1.7][(i + trial) % 4] for i in range(max_reqs)], device=dev)
        for tau in (0.0, 0.05, 0.25, 0.6, 0.95, 1.0):
            for mn in (0, 2):
                nst = torch.full((max_reqs,), -5, dtype=torch.int32, device=dev)
                VT.nstar_launch(scores, cand, toks, sidx, temp, nst, R, S, K16, tau, mn)
                ref = VT.nstar_reference(scores, cand, toks, sidx.cpu(), temp.cpu(), R, S, K16, tau, mn)
                got = nst.cpu()
                for rs in range(max_reqs):
                    want = ref.get(rs, -5)
                    total += 1
                    if int(got[rs]) != want:
                        # float32 kernel vs float64 reference: tolerate a mismatch only at a threshold tie
                        bad += 1
        # shadow grid: n* for every GRID threshold == the reference at that threshold
        grid = torch.tensor(VT.GRID, dtype=torch.float32, device=dev)
        ngrid = torch.full((max_reqs, len(VT.GRID)), -5, dtype=torch.int32, device=dev)
        nst = torch.full((max_reqs,), -5, dtype=torch.int32, device=dev)
        VT.nstar_launch(scores, cand, toks, sidx, temp, nst, R, S, K16, 0.25, 0, grid, ngrid)
        for gi, gt in enumerate(VT.GRID):
            ref = VT.nstar_reference(scores, cand, toks, sidx.cpu(), temp.cpu(), R, S, K16, gt, 0)
            for rs in range(max_reqs):
                total += 1
                bad += int(int(ngrid[rs, gi]) != ref.get(rs, -5))
        # independence: re-draw the draft right after n* (position n*+1) and recompute
        tau = 0.25
        nst = torch.full((max_reqs,), -5, dtype=torch.int32, device=dev)
        VT.nstar_launch(scores, cand, toks, sidx, temp, nst, R, S, K16, tau, 0)
        toks2 = toks.clone()
        for r in range(R):
            rs = int(sidx[r * S])
            if rs < 0:
                continue
            n = int(nst[rs])
            if n < S:
                toks2[r * S + n] = cand[r * S + n, (pick[r * S + n] + 1) % K16]
        nst2 = torch.full((max_reqs,), -5, dtype=torch.int32, device=dev)
        VT.nstar_launch(scores, cand, toks2, sidx, temp, nst2, R, S, K16, tau, 0)
        indep_bad += int((nst != nst2).sum())
    check(bad == 0, f"nstar kernel == float64 reference on {total} (request, tau, min) cases ({bad} differ)")
    check(indep_bad == 0, f"n* unchanged when the draft at position n*+1 is re-drawn (60 batches; {indep_bad} differ): "
          "the decision for row i never reads d_i")


def u2():
    print("== U2 live kernel vs reference")
    g = torch.Generator(device="cpu").manual_seed(11)
    bad = 0
    for trial in range(200):
        nreq = int(torch.randint(1, 9, (1,), generator=g))
        K = [4, 5, 7][trial % 3]
        lens, nls = [], []
        for b in range(nreq):
            if torch.rand(1, generator=g) < 0.2:
                ql = int(torch.randint(1, 40, (1,), generator=g)); nl = 1        # prefill chunk / plain decode
            else:
                ql = K + 1; nl = K + 1
            lens.append(ql); nls.append(nl)
        qsl = torch.zeros(nreq + 1, dtype=torch.int32); qsl[1:] = torch.cumsum(torch.tensor(lens), 0)
        cu = torch.zeros(nreq + 1, dtype=torch.int32); cu[1:] = torch.cumsum(torch.tensor(nls), 0)
        idx = torch.randperm(16, generator=g)[:nreq].to(torch.int32)
        nst = torch.randint(0, 9, (16,), generator=g).to(torch.int32)
        T = int(qsl[-1]) + 5
        live = torch.ones(T + 64, dtype=torch.uint8, device=dev)
        from vllm.triton_utils import triton
        VT._kernels()["live"][(nreq,)](live, idx.to(dev), qsl.to(dev), cu.to(dev), nst.to(dev),
                                         BLOCK=triton.next_power_of_2(K + 1), num_warps=1)
        ref = torch.ones(T + 64, dtype=torch.uint8)
        for b in range(nreq):
            nl = nls[b]
            if nl <= 1:
                continue
            end = int(qsl[b + 1])
            for j in range(1, nl):
                if j > int(nst[idx[b]]):
                    ref[end - nl + j] = 0
        bad += int(not torch.equal(live.cpu(), ref))
    check(bad == 0, f"live kernel == reference on 200 random batch layouts ({bad} differ)")


def u3():
    print("== U3 MoE and rejection hooks")
    prod = types.SimpleNamespace()
    seen = {}

    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit):
        seen["ids"] = ids.clone()
        return torch.zeros_like(x2d)

    apply_exl3_fused_moe._tf_exl3_apply_hook = True
    prod.apply_exl3_fused_moe = apply_exl3_fused_moe
    VT.ST.live = torch.ones(128, dtype=torch.uint8, device=dev)
    VT._hook_moe(prod)
    check(getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False),
          "the wrapper keeps integrate's _tf_exl3_apply_hook marker (its identity checks keep working)")
    B = 8
    ids = torch.randint(0, 288, (B, 8), device=dev)
    VT.ST.live[:B] = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1], dtype=torch.uint8, device=dev)
    prod.apply_exl3_fused_moe(torch.zeros(B, 16, device=dev), ids, None, None, None, None, 10.0)
    got = seen["ids"]
    want = ids.clone(); want[3:7] = -1
    check(torch.equal(got, want), "decode call: dead rows' routed ids -> -1, live rows untouched")
    big = torch.randint(0, 288, (100, 8), device=dev)
    VT.ST.live[:100] = 0
    prod.apply_exl3_fused_moe(torch.zeros(100, 16, device=dev), big, None, None, None, None, 10.0)
    check(torch.equal(seen["ids"], big), "prefill-sized call (> 64 tokens): ids untouched")
    VT.ST.live.fill_(1)
    # rejection hook masking
    rsm = types.SimpleNamespace()
    got_ds = {}

    def rejection_sample(target_logits, draft_logits, draft_sampled, cu, pos, idx_mapping, exp_map, exp_local, *a, **k):
        got_ds["ds"] = draft_sampled.clone()
        R = cu.shape[0] - 1
        return torch.zeros(R, 8, dtype=torch.int64, device=dev), torch.ones(R, dtype=torch.int32, device=dev)

    rsm.rejection_sample = rejection_sample
    VT.ST.mode = "on"
    VT.ST.nstar = torch.tensor([2, 0, 7, 3], dtype=torch.int32, device=dev)
    VT.ST.stats = torch.zeros(8, dtype=torch.int64, device=dev)
    VT.ST.hist = torch.zeros(512, dtype=torch.int64, device=dev)
    VT.ST.log_every = 0
    VT.ST.first_log = -1
    VT._hook_rejection(rsm)
    # two requests (req states 0 and 3), K = 4 -> 5 rows each
    exp_map = torch.tensor([0] * 5 + [3] * 5, dtype=torch.int32, device=dev)
    exp_local = torch.tensor(list(range(5)) * 2, dtype=torch.int32, device=dev)
    ds = torch.arange(100, 110, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 5, 10], dtype=torch.int32, device=dev)
    rsm.rejection_sample(None, None, ds, cu, None, torch.tensor([0, 3], dtype=torch.int32, device=dev), exp_map,
                         exp_local, None, None, 7)
    want = torch.tensor([100, 101, 102, -1, -1, 105, 106, 107, 108, -1], dtype=torch.int32, device=dev)
    check(torch.equal(got_ds["ds"], want), f"rejection hook masks rows with local pos > n* (n*=2 and 3): {got_ds['ds'].tolist()}")
    VT.ST.mode = "shadow"
    rsm.rejection_sample(None, None, ds, cu, None, torch.tensor([0, 3], dtype=torch.int32, device=dev), exp_map,
                         exp_local, None, None, 7)
    check(torch.equal(got_ds["ds"], ds), "shadow mode: drafts reach the sampler unchanged")
    st = VT.ST.stats.cpu().tolist()
    check(st[0] == 2 and st[2] == 16 and st[3] == 2 * ((4 - 2) + (4 - 3)), f"stats: steps/drafts/dead {st[:4]}")


def u5():
    print("== U5 fused stats kernel vs torch reference")
    g = torch.Generator(device="cpu").manual_seed(5)
    bad = 0
    for trial in range(300):
        R = int(torch.randint(1, 9, (1,), generator=g))
        nl = torch.tensor([1 if torch.rand(1, generator=g) < 0.2 else int(torch.randint(5, 9, (1,), generator=g))
                           for _ in range(R)], dtype=torch.int32)
        cu = torch.zeros(R + 1, dtype=torch.int32); cu[1:] = torch.cumsum(nl, 0)
        idx = torch.randperm(8, generator=g)[:R].to(torch.int32)
        ns = torch.tensor([int(torch.randint(1, int(x) + 1, (1,), generator=g)) for x in nl], dtype=torch.int32)
        nst = torch.randint(0, 9, (8,), generator=g).to(torch.int32)
        ngrid = torch.randint(0, 9, (8, len(VT.GRID)), generator=g).to(torch.int32)
        st = torch.zeros(8, dtype=torch.int64, device=dev); h = torch.zeros(512, dtype=torch.int64, device=dev)
        hg = torch.zeros(len(VT.GRID) * 512, dtype=torch.int64, device=dev)
        VT.stats_launch(cu.to(dev), idx.to(dev), ns.to(dev), nst.to(dev), st, h, ngrid.to(dev), hg)
        rst, rh, rhg = VT.stats_reference(cu, idx, ns, nst, ngrid)
        bad += int(not (torch.equal(st.cpu(), rst) and torch.equal(h.cpu(), rh) and torch.equal(hg.cpu(), rhg)))
    check(bad == 0, f"stats kernel == reference (stats, hist, per-threshold hists) on 300 random batches ({bad} differ)")


def u4():
    print("== U4 off = nothing installed")
    VT.ST.installed = False
    VT.ST.hooks = []
    r = VT.install(mode="off", tau=0.25, min_live=0, log_every=0, max_num_seqs=8, max_tokens=64)
    check(r is False and VT.ST.hooks == [], "install(mode='off') installs no hook")
    for raw, ok in (("", "off"), ("0", "off"), ("shadow", "shadow"), ("on", "on"), ("1", "on")):
        check(VT.parse_env({VT.ENV: raw})[0] == ok, f"parse {VT.ENV}={raw!r} -> {ok}")
    for raw in ("2", "yes", "maybe"):
        try:
            VT.parse_env({VT.ENV: raw}); check(False, f"{VT.ENV}={raw!r} rejected")
        except ValueError:
            check(True, f"{VT.ENV}={raw!r} rejected")
    for k, v in ((VT.ENV_TAU, "1.5"), (VT.ENV_TAU, "-0.1"), (VT.ENV_MIN, "8"), (VT.ENV_LOG, "-1")):
        try:
            VT.parse_env({VT.ENV: "on", k: v}); check(False, f"{k}={v} rejected")
        except ValueError:
            check(True, f"{k}={v} rejected")


u1(); u2(); u3(); u5(); u4()
print("\nALL PASSED" if not FAIL else f"\nFAILED: {len(FAIL)}")
for f in FAIL:
    print("  -", f)
sys.exit(1 if FAIL else 0)
