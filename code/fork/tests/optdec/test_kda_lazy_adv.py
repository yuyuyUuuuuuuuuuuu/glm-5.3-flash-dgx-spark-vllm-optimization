"""Adversarial review test for GLM53_DEC_KDA_LAZY v2 (decode4 review): production-shaped layer/tensor sharing plus the
step-to-step churn the engine produces, through the REAL wrapper + commit(), against production's kernel, bitwise.

  * 34 KDA layers on 11 shared state tensors (4 mamba groups: 11 + 11 + 11 + 1, layer l -> group l // 11, tensor
    l % 11), like production's _glm5_next_tensor_layout; each group has its own slot ids (disjoint blocks)
  * a pool of 10 persistent requests; each step draws a random subset (N in 1..8) in a RANDOM batch order (a request
    changes its batch row between steps, requests join/leave), random K in {4,5,7} per step, random A in 1..K+1
  * align mode: num_computed (post-step) chosen so that a block boundary falls inside the accepted rows for some
    requests (bias column) and not for others; idx_mapping maps rows -> request ids
  * every ~5th step is a mixed batch (_pure_spec_batch False): every layer takes production's path
  * self-check off (fast commit), then a second pass with the self-check on every commit
Checked after every step: every layer's verify output; for each layer/request the state in column A-1 (what the next
verify reads) and in the bias column (what the align post-copy reads) == production's per-row stores.
Run: tests/optdec/run_kda_lazy_tests.sh tests/optdec/test_kda_lazy_adv.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402

dev = "cuda"
H, KD, V = 32, 128, 128
NSEQ, TMAX = 8, 8
LB = -5.0
NT = 11
GROUPS = [11, 11, 11, 1]
LAYERS = [(gi, ti) for gi, n in enumerate(GROUPS) for ti in range(n)]   # layer -> (group, tensor)
LAYERS.sort(key=lambda x: (x[1], x[0]))      # forward order interleaves co-owners of one tensor
NL = len(LAYERS)
NG = len(GROUPS)
NREQ = 10
PAGE = H * V * KD + 4096
NBLK = 1 + NG * NREQ * TMAX
BS = 64                                       # align block size (any value; only the bias arithmetic matters)
FAIL = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not cond:
        FAIL.append(name)


def beq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


g = torch.Generator(device=dev).manual_seed(777)
A_LOG = [(torch.randn(H, device=dev, generator=g) * 0.5).float() for _ in range(NL)]
G_BIAS = [(torch.randn(H * KD, device=dev, generator=g) * 0.5).float() for _ in range(NL)]


def view(raw):
    return raw.as_strided((NBLK, H, V, KD), (PAGE, V * KD, KD, 1))


L._ORIG["frk"] = prod_frk
PURE = {"v": True}
L._pure_spec_batch = lambda: PURE["v"]
L.allocate(NL, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True


def run(steps, seed, verify_env):
    os.environ["GLM53_DEC_KDA_LAZY_VERIFY"], os.environ["GLM53_DEC_KDA_LAZY_VERIFY_EVERY"] = verify_env
    L.reset_layers()
    L.ST.repair = False
    L.COUNTERS["commits"] = 0
    rng = random.Random(seed)
    raws_p = [torch.empty(NBLK * PAGE, dtype=torch.float32, device=dev).normal_(0, 0.05, generator=g) for _ in range(NT)]
    raws_l = [r.clone() for r in raws_p]
    sp = [view(r) for r in raws_p]
    sl = [view(r) for r in raws_l]
    perm = torch.randperm(NBLK - 1, generator=torch.Generator().manual_seed(seed)) + 1
    # slot table per (group, request): TMAX consecutive-in-table block ids, disjoint across groups and requests
    tab = perm[: NG * NREQ * TMAX].view(NG, NREQ, TMAX).to(torch.int32).to(dev)
    nacc = [1] * NREQ
    ncomp = [rng.randint(100, 1000) for _ in range(NREQ)]
    bad_o = bad_s = bad_b = nb = mixed = 0
    for step in range(steps):
        N = rng.randint(1, NSEQ)
        rows = rng.sample(range(NREQ), N)                 # batch row -> request id (random order every step)
        Kd = rng.choice((4, 5, 7))
        T = Kd + 1
        PURE["v"] = (step % 5) != 4
        mixed += not PURE["v"]
        A = [rng.randint(1, T) for _ in rows]
        # post-step computed count; for ~half the requests put a block boundary inside the accepted rows
        nc_post = []
        for r, a in zip(rows, A):
            base = ncomp[r] + a
            if rng.random() < 0.5:
                b = rng.randint(0, a - 1)                 # want bias b: aligned = running + b, running = nc - a + 1
                running = (base // BS + 1) * BS - b       # choose running so that running + b is aligned
                nc = running + a - 1
            else:
                nc = base
            nc_post.append(nc)
        idx = torch.stack([tab[:, r] for r in rows], 1)   # [NG, N, TMAX]
        nacc_t = torch.tensor([nacc[r] for r in rows], device=dev, dtype=torch.int32)
        cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
        for l, (gi, ti) in enumerate(LAYERS):
            tot = N * T
            qkv = (torch.randn(1, tot, 3 * H * KD + 64, device=dev, generator=g) * 1.5).to(torch.bfloat16)
            q = qkv[..., : H * KD].view(1, tot, H, KD)
            k = qkv[..., H * KD: 2 * H * KD].view(1, tot, H, KD)
            v = qkv[..., 2 * H * KD: 3 * H * KD].view(1, tot, H, V)
            bproj = (torch.randn(1, tot, H + 96, device=dev, generator=g) * 2).to(torch.bfloat16)
            beta = bproj[..., 7:7 + H]
            gg = (torch.randn(1, tot, H, KD, device=dev, generator=g) * 2).to(torch.bfloat16)
            ix = idx[gi].contiguous()
            op = torch.empty(q.shape, dtype=q.dtype, device=dev)
            prod_frk(q=q, k=k, v=v, g=gg, beta=beta, initial_state=sp[ti], use_qk_l2norm_in_kernel=True,
                     cu_seqlens=cu, ssm_state_indices=ix, num_accepted_tokens=nacc_t, out=op, sigmoid_beta=True,
                     a_log=A_LOG[l], g_bias=G_BIAS[l], compute_gate=True, lower_bound=LB)
            ol = torch.empty(q.shape, dtype=q.dtype, device=dev)
            L.fused_recurrent_kda_lazy(q, k, v, gg, beta=beta, initial_state=sl[ti], use_qk_l2norm_in_kernel=True,
                                       cu_seqlens=cu, ssm_state_indices=ix, num_accepted_tokens=nacc_t, out=ol,
                                       sigmoid_beta=True, a_log=A_LOG[l], g_bias=G_BIAS[l], compute_gate=True,
                                       lower_bound=LB)
            bad_o += not beq(op, ol)
        At = torch.tensor(A, device=dev, dtype=torch.int32)
        idx_map = torch.tensor(rows, device=dev, dtype=torch.int32)
        ncomp_t = torch.zeros(NREQ, device=dev, dtype=torch.int32)
        for r, nc in zip(rows, nc_post):
            ncomp_t[r] = nc
        L.commit(At, idx_map, N, num_computed=ncomp_t, block_size=BS, align=True)
        torch.cuda.synchronize()
        for l, (gi, ti) in enumerate(LAYERS):
            for n, r in enumerate(rows):
                a = A[n]
                s = int(tab[gi, r, a - 1])
                bad_s += not beq(sl[ti][s], sp[ti][s])
                nc = nc_post[n]
                running = nc - a + 1
                aligned = nc // BS * BS
                if aligned >= running:
                    b = aligned - running
                    sb = int(tab[gi, r, b])
                    nb += 1
                    bad_b += not beq(sl[ti][sb], sp[ti][sb])
        for n, r in enumerate(rows):
            nacc[r] = A[n]
            ncomp[r] = nc_post[n]
    return bad_o, bad_s, bad_b, nb, mixed


bo, bs, bb, nb, mx = run(40, 5, ("0", "0"))
check(f"A1 34 layers / 11 shared tensors / 4 groups, churned batches, align, {mx} mixed steps: outputs bitwise",
      bo == 0, f"bad {bo}")
check("A1 ... column A-1 (next verify's initial state) bitwise == production", bs == 0, f"bad {bs}")
check(f"A1 ... align bias column bitwise == production ({nb} checks)", bb == 0 and nb > 0, f"bad {bb}")
check(f"A1 every layer registered ({len(L.ST.layers)} of {NL})", len(L.ST.layers) == NL)
m0 = L.COUNTERS["verify_mismatch"]
bo, bs, bb, nb, mx = run(15, 6, ("1000000", "1"))
check("A2 same with the self-check on every commit: no mismatch, states == production",
      bo == 0 and bs == 0 and bb == 0 and L.COUNTERS["verify_mismatch"] == m0 and not L.ST.repair,
      f"out {bo} state {bs} bias {bb} mismatches {L.COUNTERS['verify_mismatch'] - m0}")
print("counters", dict(L.COUNTERS))
print("RESULT:", "ALL OK" if not FAIL else f"FAIL {FAIL}")
sys.exit(1 if FAIL else 0)
