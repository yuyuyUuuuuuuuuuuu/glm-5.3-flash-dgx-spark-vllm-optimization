"""GLM53_DEC_KDA_LAZY bitwise tests against production's KDA spec-verify kernel (the image's fused_recurrent_kda with
the r16 strided-qkv backport bound over it, i.e. exactly what production runs; tests/optdec/run_kda_lazy_tests.sh).

T1  one step, many shapes: lazy verify output == production output (bitwise); lazy verify writes NO state byte;
    commit(A) writes column A-1 == production's column A-1 (bitwise), touches no other byte.
T2  multi-step chains (K in {4,5,7} per step, random acceptance, 1..8 sequences, strided q/k/v/beta views as in
    production): every step's output and the state the next step reads == production's.
T3  align boundary: the column the mamba align post-copy reads (bias) is committed == production; ALIGN_ALL commits
    every accepted column == production.
T4  CUDA graph: lazy verify captured + replayed with new inputs, eager commit == production.
T5  wrapper: a mixed batch / a shape it does not take -> production's function, and the layer's flags are cleared
    (a following commit writes nothing).
T6  scalar num_sampled (PP path) and num_sampled = 0 (treated as 1, like postprocess_state).
"""
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
NSLOT = 1 + NSEQ * TMAX * 2
LB = -5.0
FAIL = []


def check(name, cond, info=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not cond:
        FAIL.append(name)


def beq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a,
                                                                       b.view(torch.uint8) if b.dtype != torch.bool else b)


g = torch.Generator(device=dev).manual_seed(1234)
a_log = (torch.randn(H, device=dev, generator=g) * 0.5).float()
g_bias = (torch.randn(H * KD, device=dev, generator=g) * 0.5).float()
L.allocate(4, NSEQ, TMAX, H, KD, V, dev, (torch.bfloat16,) * 4)
L.ST.enabled = True


def make_step(N, Ts, strided=True):
    """q/k/v as column slices of a merged per-token buffer, beta a slice of a wider projection (production layout)."""
    tot = sum(Ts)
    qkv = (torch.randn(1, tot, 3 * H * KD + 64, device=dev, generator=g) * 1.5).to(torch.bfloat16)
    if strided:
        q = qkv[..., : H * KD].view(1, tot, H, KD)
        k = qkv[..., H * KD: 2 * H * KD].view(1, tot, H, KD)
        v = qkv[..., 2 * H * KD: 3 * H * KD].view(1, tot, H, V)
    else:
        q = qkv[..., : H * KD].reshape(1, tot, H, KD).contiguous()
        k = qkv[..., H * KD: 2 * H * KD].reshape(1, tot, H, KD).contiguous()
        v = qkv[..., 2 * H * KD: 3 * H * KD].reshape(1, tot, H, V).contiguous()
    bproj = (torch.randn(1, tot, H + 96, device=dev, generator=g) * 2).to(torch.bfloat16)
    beta = bproj[..., 7:7 + H] if strided else bproj[..., 7:7 + H].contiguous()
    gg = (torch.randn(1, tot, H, KD, device=dev, generator=g) * 2).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(Ts).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    return q, k, v, gg, beta, cu


def run_prod(state, q, k, v, gg, beta, cu, idx, nacc):
    out = torch.empty(q.shape, dtype=q.dtype, device=dev)
    prod_frk(q=q, k=k, v=v, g=gg, beta=beta, initial_state=state, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
             ssm_state_indices=idx, num_accepted_tokens=nacc, out=out, sigmoid_beta=True, a_log=a_log, g_bias=g_bias,
             compute_gate=True, lower_bound=LB)
    return out


def run_lazy(state, slot, q, k, v, gg, beta, cu, idx, nacc):
    out = torch.empty(q.shape, dtype=q.dtype, device=dev)
    L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, state, cu, idx, nacc, out, a_log, g_bias, LB, slot)
    return out


def fresh_state():
    return (torch.randn(NSLOT, H, V, KD, device=dev, generator=g) * 0.05).float()


def slots_for(N):
    perm = torch.randperm(NSLOT - 1, generator=torch.Generator().manual_seed(random.randint(0, 1 << 30)))[: N * TMAX] + 1
    return perm.view(N, TMAX).to(torch.int32).to(dev)


# --------------------------------------------------------------------------------------------------------------- T1
random.seed(7)
ncase = 0
bad_o = bad_untouched = bad_commit = bad_other = 0
for N in (1, 2, 3, 5, 8):
    for T in (1, 2, 4, 5, 6, 8):
        for strided in (True, False):
            for rep in range(2):
                st0 = fresh_state()
                idx = slots_for(N)
                nacc = torch.tensor([random.randint(1, TMAX) for _ in range(N)], device=dev, dtype=torch.int32)
                q, k, v, gg, beta, cu = make_step(N, [T] * N, strided)
                sp, sl = st0.clone(), st0.clone()
                op = run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
                slot = L.register(sl, a_log, g_bias)
                ol = run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
                torch.cuda.synchronize()
                bad_o += not beq(op, ol)
                bad_untouched += not beq(sl, st0)
                A = torch.tensor([random.randint(0, T) for _ in range(N)], device=dev, dtype=torch.int32)
                L.commit(A, None, N)
                torch.cuda.synchronize()
                exp = st0.clone()
                for n in range(N):
                    c = max(int(A[n]), 1) - 1
                    s = int(idx[n, c])
                    exp[s] = sp[s]
                okc = all(beq(sl[int(idx[n, max(int(A[n]), 1) - 1])], sp[int(idx[n, max(int(A[n]), 1) - 1])])
                          for n in range(N))
                bad_commit += not okc
                if not okc and bad_commit <= 12:
                    print(f"   T1 commit mismatch N={N} T={T} strided={strided} A={A.tolist()} nacc={nacc.tolist()}")
                bad_other += not beq(sl, exp)
                L.reset_layers()
                ncase += 1
check(f"T1 output bitwise ({ncase} cases)", bad_o == 0, f"bad {bad_o}")
check("T1 lazy verify writes no state byte", bad_untouched == 0, f"bad {bad_untouched}")
check("T1 commit column A-1 bitwise == production", bad_commit == 0, f"bad {bad_commit}")
check("T1 commit touches nothing else", bad_other == 0, f"bad {bad_other}")

# --------------------------------------------------------------------------------------------------------------- T2
bad = 0
steps = 0
for N in (1, 2, 4, 8):
    st0 = fresh_state()
    sp, sl = st0.clone(), st0.clone()
    L.reset_layers()
    slot = L.register(sl, a_log, g_bias)
    idx = slots_for(N)
    nacc = torch.ones(N, device=dev, dtype=torch.int32)
    for step in range(25):
        Kd = random.choice((4, 5, 7))
        T = Kd + 1
        q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
        op = run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
        ol = run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
        A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
        L.commit(A, None, N)
        torch.cuda.synchronize()
        ok = beq(op, ol) and all(beq(sl[int(idx[n, int(A[n]) - 1])], sp[int(idx[n, int(A[n]) - 1])]) for n in range(N))
        bad += not ok
        steps += 1
        nacc = A.clone()
        if step % 6 == 5:                       # the request's slots move (align block migration): new slot table
            new = slots_for(N)
            for n in range(N):
                c = int(nacc[n]) - 1
                sp[int(new[n, 0])] = sp[int(idx[n, c])]
                sl[int(new[n, 0])] = sl[int(idx[n, c])]
            idx = new
            nacc = torch.ones(N, device=dev, dtype=torch.int32)
check(f"T2 multi-step chains output + next initial state bitwise ({steps} steps)", bad == 0, f"bad {bad}")

# --------------------------------------------------------------------------------------------------------------- T3
bad = 0
cases = 0
BS = 64
for N in (1, 3, 8):
    for rep in range(6):
        st0 = fresh_state()
        sp, sl = st0.clone(), st0.clone()
        L.reset_layers()
        slot = L.register(sl, a_log, g_bias)
        idx = slots_for(N)
        nacc = torch.tensor([random.randint(1, TMAX) for _ in range(N)], device=dev, dtype=torch.int32)
        T = 8
        q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
        run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
        run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
        A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
        # post-step computed count so that a block boundary falls inside the accepted rows: bias = aligned - running
        idx_map = torch.arange(N, device=dev, dtype=torch.int32)
        biases = [random.randint(0, int(A[n]) - 1) for n in range(N)]
        ncomp = torch.tensor([BS * 10 + int(A[n]) - 1 - biases[n] for n in range(N)], device=dev, dtype=torch.int32)
        # running = nc - A + 1 ; aligned = nc // BS * BS = BS*10 iff nc - BS*10 in [0, BS)
        L.commit(A, idx_map, N, num_computed=ncomp, block_size=BS, align=True)
        torch.cuda.synchronize()
        for n in range(N):
            nc = int(ncomp[n]); a = int(A[n]); running = nc - a + 1; aligned = nc // BS * BS
            assert aligned >= running and aligned - running == biases[n]
            for c in {a - 1, biases[n]}:
                s = int(idx[n, c])
                bad += not beq(sl[s], sp[s])
            cases += 1
check(f"T3 align: bias column committed == production ({cases} requests)", bad == 0, f"bad {bad}")
bad = 0
for N in (2, 8):
    st0 = fresh_state()
    sp, sl = st0.clone(), st0.clone()
    L.reset_layers()
    slot = L.register(sl, a_log, g_bias)
    idx = slots_for(N)
    nacc = torch.ones(N, device=dev, dtype=torch.int32)
    q, k, v, gg, beta, cu = make_step(N, [8] * N, True)
    run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
    run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
    A = torch.tensor([random.randint(1, 8) for _ in range(N)], device=dev, dtype=torch.int32)
    L.commit(A, None, N, align=True, align_all=True)
    torch.cuda.synchronize()
    for n in range(N):
        for c in range(int(A[n])):
            bad += not beq(sl[int(idx[n, c])], sp[int(idx[n, c])])
check("T3 ALIGN_ALL: every accepted column == production", bad == 0, f"bad {bad}")

# --------------------------------------------------------------------------------------------------------------- T4
N, T = 4, 6
st0 = fresh_state()
sp, sl = st0.clone(), st0.clone()
L.reset_layers()
slot = L.register(sl, a_log, g_bias)
idx = slots_for(N)
nacc = torch.ones(N, device=dev, dtype=torch.int32)
q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
outb = torch.empty(q.shape, dtype=q.dtype, device=dev)
s_ = torch.cuda.Stream()
s_.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s_):
    L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, sl, cu, idx, nacc, outb, a_log, g_bias, LB, slot)
torch.cuda.current_stream().wait_stream(s_)
L.ST.meta.zero_()
sl.copy_(st0)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, sl, cu, idx, nacc, outb, a_log, g_bias, LB, slot)
bad = 0
for step in range(8):
    q2, k2, v2, gg2, beta2, _ = make_step(N, [T] * N, True)
    q.copy_(q2); k.copy_(k2); v.copy_(v2); gg.copy_(gg2); beta.copy_(beta2)
    op = run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
    graph.replay()
    A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
    L.commit(A, None, N)
    torch.cuda.synchronize()
    bad += not (beq(op, outb) and all(beq(sl[int(idx[n, int(A[n]) - 1])], sp[int(idx[n, int(A[n]) - 1])])
                                     for n in range(N)))
    nacc.copy_(A)
check("T4 CUDA graph replay + eager commit == production (8 steps)", bad == 0, f"bad {bad}")

# --------------------------------------------------------------------------------------------------------------- T5
L._ORIG["frk"] = prod_frk
st0 = fresh_state()
sl = st0.clone()
L.reset_layers()
slot = L.register(sl, a_log, g_bias)
N, T = 2, 5
idx = slots_for(N)
nacc = torch.ones(N, device=dev, dtype=torch.int32)
q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)       # leaves flags set
flags_before = int(L.ST.meta[slot, :N, 0].sum())
sp = st0.clone()
sl2 = st0.clone()
out = torch.empty(q.shape, dtype=q.dtype, device=dev)
# no forward context -> _pure_spec_batch() is None -> production path; flags of this layer cleared
L.ST.layers.clear(); L.ST.layers[L.layer_key(sl2, a_log)] = slot  # v2: per-layer key (state tensor, A_log)  # same anchor/offset bookkeeping is not needed: prod path
L.fused_recurrent_kda_lazy(q, k, v, gg, beta=beta, initial_state=sl2, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                           ssm_state_indices=idx, num_accepted_tokens=nacc, out=out, sigmoid_beta=True, a_log=a_log,
                           g_bias=g_bias, compute_gate=True, lower_bound=LB)
op = run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
torch.cuda.synchronize()
check("T5 wrapper outside a pure spec batch = production (output + per-row states)", beq(out, op) and beq(sl2, sp))
check("T5 ... and the layer's pending flags are cleared", flags_before == N and int(L.ST.meta[slot, :, 0].sum()) == 0,
      f"before {flags_before}")
snap = sl2.clone()
L.commit(torch.full((N,), 3, device=dev, dtype=torch.int32), None, N)
torch.cuda.synchronize()
check("T5 ... a following commit writes nothing", beq(sl2, snap))

# --------------------------------------------------------------------------------------------------------------- T6
bad = 0
for A_scalar in (0, 1, 3):
    st0 = fresh_state()
    sp, sl = st0.clone(), st0.clone()
    L.reset_layers()
    slot = L.register(sl, a_log, g_bias)
    N, T = 3, 5
    idx = slots_for(N)
    nacc = torch.tensor([2, 1, 5], device=dev, dtype=torch.int32)
    q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
    run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
    run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
    L.commit(A_scalar, None, N)
    torch.cuda.synchronize()
    c = max(A_scalar, 1) - 1
    bad += not all(beq(sl[int(idx[n, c])], sp[int(idx[n, c])]) for n in range(N))
check("T6 scalar num_sampled (0 -> 1, 1, 3) == production", bad == 0, f"bad {bad}")


# --------------------------------------------------------------------------------------------------------------- T7
# self-check (production's kernel recomputes the committed state from the saved rows) passes on a correct commit
L._ORIG["frk"] = prod_frk
import os as _os
_os.environ["GLM53_DEC_KDA_LAZY_VERIFY"] = "1000000"
L.ST.repair = False
v0, m0 = L.COUNTERS["verified"], L.COUNTERS["verify_mismatch"]
bad = 0
for N in (1, 3, 8):
    st0 = fresh_state()
    sp, sl = st0.clone(), st0.clone()
    L.reset_layers()
    slot = L.register(sl, a_log, g_bias)
    idx = slots_for(N)
    nacc = torch.tensor([random.randint(1, TMAX) for _ in range(N)], device=dev, dtype=torch.int32)
    for step in range(6):
        T = random.choice((5, 6, 8))
        q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
        run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
        run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
        A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
        L.commit(A, None, N)
        torch.cuda.synchronize()
        bad += not all(beq(sl[int(idx[n, int(A[n]) - 1])], sp[int(idx[n, int(A[n]) - 1])]) for n in range(N))
        nacc = A.clone()
check("T7 self-check runs and passes on correct commits", bad == 0 and L.COUNTERS["verify_mismatch"] == m0
      and L.COUNTERS["verified"] - v0 == 18 and not L.ST.repair,
      f"bad {bad} verified {L.COUNTERS['verified'] - v0} mismatches {L.COUNTERS['verify_mismatch'] - m0}")

# --------------------------------------------------------------------------------------------------------------- T8
# a wrong fast commit (corrupted gate table) is caught, this step repaired, later commits in repair mode == production
bad = 0
N = 4
st0 = fresh_state()
sp, sl = st0.clone(), st0.clone()
L.reset_layers()
slot = L.register(sl, a_log, g_bias)
L.ST.alog_tab[slot] += 0.25                       # the fast commit kernel now computes the wrong gate
idx = slots_for(N)
nacc = torch.ones(N, device=dev, dtype=torch.int32)
for step in range(5):
    T = 6
    q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
    run_prod(sp, q, k, v, gg, beta, cu, idx, nacc)
    run_lazy(sl, slot, q, k, v, gg, beta, cu, idx, nacc)
    A = torch.tensor([random.randint(1, T) for _ in range(N)], device=dev, dtype=torch.int32)
    L.commit(A, None, N)
    torch.cuda.synchronize()
    bad += not all(beq(sl[int(idx[n, int(A[n]) - 1])], sp[int(idx[n, int(A[n]) - 1])]) for n in range(N))
    nacc = A.clone()
check("T8 corrupted fast commit: detected, repaired, repair mode exact", bad == 0 and L.ST.repair
      and L.COUNTERS["verify_mismatch"] == m0 + 1, f"bad {bad} repair {L.ST.repair}")
L.ST.repair = False

# --------------------------------------------------------------------------------------------------------------- T9
# another gate lower bound (the commit uses the bound the lazy calls saw)
L._ORIG.pop("frk", None)
bad = 0
for LB2 in (-3.0, -7.5):
    st0 = fresh_state()
    sp, sl = st0.clone(), st0.clone()
    L.reset_layers()
    slot = L.register(sl, a_log, g_bias)
    L.ST.lb_seen = LB2
    N, T = 3, 6
    idx = slots_for(N)
    nacc = torch.tensor([1, 4, 6], device=dev, dtype=torch.int32)
    q, k, v, gg, beta, cu = make_step(N, [T] * N, True)
    outp = torch.empty(q.shape, dtype=q.dtype, device=dev)
    prod_frk(q=q, k=k, v=v, g=gg, beta=beta, initial_state=sp, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
             ssm_state_indices=idx, num_accepted_tokens=nacc, out=outp, sigmoid_beta=True, a_log=a_log, g_bias=g_bias,
             compute_gate=True, lower_bound=LB2)
    outl = torch.empty(q.shape, dtype=q.dtype, device=dev)
    L.lazy_verify(q, k, v, gg, beta, KD ** -0.5, sl, cu, idx, nacc, outl, a_log, g_bias, LB2, slot)
    A = torch.tensor([2, 6, 1], device=dev, dtype=torch.int32)
    L.commit(A, None, N)
    torch.cuda.synchronize()
    bad += not (beq(outp, outl) and all(beq(sl[int(idx[n, int(A[n]) - 1])], sp[int(idx[n, int(A[n]) - 1])])
                                        for n in range(N)))
check("T9 other gate lower bounds (-3, -7.5): output + commit == production", bad == 0, f"bad {bad}")
L.reset_layers()

print("RESULT:", "ALL OK" if not FAIL else f"FAIL {FAIL}")
sys.exit(1 if FAIL else 0)
