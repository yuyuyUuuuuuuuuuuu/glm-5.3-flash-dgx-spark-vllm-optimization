"""GLM53_DEC_KDA_LAZY with production's KV layout: several KDA layers SHARE one recurrent-state tensor.

GLM-5-Next's hybrid slot sharing (vllm/v1/core/kv_cache_utils.py `_glm5_next_tensor_layout`): one KV tensor per MLA
layer, co-owned by that MLA layer and ONE mamba (KDA) layer of EACH mamba group; the co-owners use disjoint block ids.
Production (45 layers: 34 KDA in groups of 11 + 11 MLA) therefore hands up to four KDA layers the SAME
`recurrent_state` view (same data_ptr, same stride), each layer with its own A_log / dt_bias and its own slot ids.
The handoff mini model (5 KDA + 5 MLA = one mamba group) and test_kda_lazy.py (one tensor per layer) never had that.

S1  forward-order chain through the REAL wrapper (fused_recurrent_kda_lazy) + commit(), NL layers on NT shared
    tensors, K in {4,5,7}, partial acceptance (A drawn in 1..K+1), N sequences: every layer's output and the state its
    next verify reads == production's kernel chain, bitwise. Fast commit only (no self-check).
S2  same with the self-check on every commit (the commit then writes production's bytes from the saved rows): passes
    and agrees with production.
S3  registration: every layer gets its own scratch slot (no "a_log / g_bias tensors changed" production fallback).
Run: tests/optdec/run_kda_lazy_tests.sh tests/optdec/test_kda_lazy_shared.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, "/w")
import glm53_kda_lazy as L  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as prod_frk  # noqa: E402
import vllm.third_party.flash_linear_attention.ops.fused_recurrent as FR  # noqa: E402

assert hasattr(FR, "token_stride"), "the strided-qkv backport must be bound over the image's fused_recurrent.py"
dev = "cuda"
H, KD, V = 32, 128, 128
NSEQ, TMAX = 8, 8
LB = -5.0
NG, NT = 3, 2                       # mamba groups x shared tensors -> NG * NT KDA layers
NL = NG * NT
PAGE = H * V * KD + 8192            # page stride > state size (the state rides inside the MLA page)
NBLK = 1 + NG * NSEQ * TMAX         # block 0 = null block; every group gets disjoint block ids
FAIL = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not cond:
        FAIL.append(name)


def beq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


g = torch.Generator(device=dev).manual_seed(4321)
# layer l: group l // NT, tensor l % NT (forward order interleaves the co-owners, like production's layer order)
A_LOG = [(torch.randn(H, device=dev, generator=g) * 0.5).float() for _ in range(NL)]
G_BIAS = [(torch.randn(H * KD, device=dev, generator=g) * 0.5).float() for _ in range(NL)]


def make_raw():
    raw = torch.zeros(NBLK * PAGE, dtype=torch.float32, device=dev)
    return raw


def view(raw):
    return raw.as_strided((NBLK, H, V, KD), (PAGE, V * KD, KD, 1))


def make_step(N, Ts):
    tot = sum(Ts)
    qkv = (torch.randn(1, tot, 3 * H * KD + 64, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    q = qkv[..., : H * KD].view(1, tot, H, KD)
    k = qkv[..., H * KD: 2 * H * KD].view(1, tot, H, KD)
    v = qkv[..., 2 * H * KD: 3 * H * KD].view(1, tot, H, V)
    bproj = (torch.randn(1, tot, H + 96, device=dev, generator=g) * 2).to(torch.bfloat16)
    beta = bproj[..., 7:7 + H]
    gg = (torch.randn(1, tot, H, KD, device=dev, generator=g) * 2).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(Ts).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    return q, k, v, gg, beta, cu


def group_slots(N, seed):
    perm = torch.randperm(NBLK - 1, generator=torch.Generator().manual_seed(seed)) + 1
    tabs = []
    for gi in range(NG):
        tabs.append(perm[gi * NSEQ * TMAX: gi * NSEQ * TMAX + N * TMAX].view(N, TMAX).to(torch.int32).to(dev))
    return tabs


L._ORIG["frk"] = prod_frk
L._pure_spec_batch = lambda: True               # the engine's forward context says: pure spec-verify batch
L.allocate(NL, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True


def run_chain(N, steps, seed, verify_env):
    os.environ["GLM53_DEC_KDA_LAZY_VERIFY"], os.environ["GLM53_DEC_KDA_LAZY_VERIFY_EVERY"] = verify_env
    L.reset_layers()
    L.ST.repair = False
    L.COUNTERS["commits"] = 0
    random.seed(seed)
    raws_p = [make_raw() for _ in range(NT)]
    for r in raws_p:
        r.normal_(0, 0.05, generator=g)
    raws_l = [r.clone() for r in raws_p]
    sp = [view(r) for r in raws_p]
    sl = [view(r) for r in raws_l]
    tabs = group_slots(N, seed)
    nacc = torch.ones(N, device=dev, dtype=torch.int32)
    bad_o = bad_s = bad_other = 0
    for step in range(steps):
        Kd = random.choice((4, 5, 7))
        T = Kd + 1
        A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
        for l in range(NL):
            gi, ti = l // NT, l % NT
            q, k, v, gg, beta, cu = make_step(N, [T] * N)
            op = torch.empty(q.shape, dtype=q.dtype, device=dev)
            prod_frk(q=q, k=k, v=v, g=gg, beta=beta, initial_state=sp[ti], use_qk_l2norm_in_kernel=True,
                     cu_seqlens=cu, ssm_state_indices=tabs[gi], num_accepted_tokens=nacc, out=op, sigmoid_beta=True,
                     a_log=A_LOG[l], g_bias=G_BIAS[l], compute_gate=True, lower_bound=LB)
            ol = torch.empty(q.shape, dtype=q.dtype, device=dev)
            L.fused_recurrent_kda_lazy(q, k, v, gg, beta=beta, initial_state=sl[ti], use_qk_l2norm_in_kernel=True,
                                       cu_seqlens=cu, ssm_state_indices=tabs[gi], num_accepted_tokens=nacc, out=ol,
                                       sigmoid_beta=True, a_log=A_LOG[l], g_bias=G_BIAS[l], compute_gate=True,
                                       lower_bound=LB)
            bad_o += not beq(op, ol)
        L.commit(A, None, N)
        torch.cuda.synchronize()
        # the state each layer's NEXT verify reads: column A-1 of its group's slot table, in its (shared) tensor
        for l in range(NL):
            gi, ti = l // NT, l % NT
            for n in range(N):
                s = int(tabs[gi][n, int(A[n]) - 1])
                bad_s += not beq(sl[ti][s], sp[ti][s])
        nacc = A.clone()
    # bytes outside every slot (page padding, null block) untouched
    for ti in range(NT):
        pad_p = raws_p[ti].view(NBLK, PAGE)[:, H * V * KD:]
        pad_l = raws_l[ti].view(NBLK, PAGE)[:, H * V * KD:]
        bad_other += not beq(pad_p, pad_l)
    return bad_o, bad_s, bad_other


tot_o = tot_s = tot_x = 0
for N, seed in ((1, 11), (2, 12), (4, 13), (8, 14)):
    bo, bs, bx = run_chain(N, 20, seed, ("0", "0"))
    tot_o += bo; tot_s += bs; tot_x += bx
    print(f"   S1 N={N}: output mismatches {bo}, next-initial-state mismatches {bs} (of {20 * NL * N}), pad {bx}")
check(f"S1 shared tensors ({NL} layers on {NT} tensors, {NG} groups), fast commit: outputs bitwise", tot_o == 0,
      f"bad {tot_o}")
check("S1 ... every layer's next initial state bitwise == production (partial acceptance)", tot_s == 0, f"bad {tot_s}")
check("S1 ... page padding untouched", tot_x == 0, f"bad {tot_x}")
check(f"S3 every layer registered its own scratch slot ({len(L.ST.layers)} of {NL})", len(L.ST.layers) == NL,
      f"layers {len(L.ST.layers)} lazy_calls {L.COUNTERS['lazy_calls']} prod_calls {L.COUNTERS['prod_calls']}")

m0 = L.COUNTERS["verify_mismatch"]
bo, bs, bx = run_chain(4, 12, 21, ("1000000", "1"))
check("S2 self-check on every commit: no mismatch, states == production", bs == 0 and bo == 0
      and L.COUNTERS["verify_mismatch"] == m0 and not L.ST.repair,
      f"out {bo} state {bs} mismatches {L.COUNTERS['verify_mismatch'] - m0} repair {L.ST.repair}")
print("counters", dict(L.COUNTERS))
print("RESULT:", "ALL OK" if not FAIL else f"FAIL {FAIL}")
sys.exit(1 if FAIL else 0)
