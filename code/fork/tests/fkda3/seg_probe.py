#!/usr/bin/env python3
"""FKDA3 probe: run one T-token FlashKDA call as consecutive S-token segments (state carried in fp32 through
final_state -> initial_state, one small workspace reused so K1's output stays L2-resident for K2). Reports time vs the
single call and whether out/final state are bit-identical to it."""
import statistics, sys, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/w/overlay")
import _flashkda_fp32_C  # noqa
op = torch.ops._flashkda_fp32_C
D, H = 128, 32
T = int(sys.argv[2]) if len(sys.argv) > 2 else 13824
g = torch.Generator(device="cpu").manual_seed(5)
rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
q, k, v, g1 = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D, sc=0.5)
beta = rn(1, T, 3 * H)[:, :, H:2 * H]
s0 = (torch.randn(1, H, D, D, generator=g) * 0.3).cuda()
A = (torch.randn(H, generator=g) * .2).cuda(); dtb = (torch.rand(H, D, generator=g) * 8 - 10).cuda()
out_ref = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda"); fs_ref = torch.empty(1, H, D, D, device="cuda")
ws_full = torch.empty(int(op.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
cu_full = torch.tensor([0, T], dtype=torch.int32, device="cuda")
def single():
    op.fwd(q, k, v, g1, beta, D ** -.5, out_ref, ws_full, A, dtb, -5.0, s0, fs_ref, cu_full, None, None)
def make_seg(S):
    ws = torch.empty(int(op.get_workspace_size(S, H, 1)), dtype=torch.uint8, device="cuda")
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    st = [torch.empty(1, H, D, D, device="cuda") for _ in range(2)]
    cus = {}
    segs = []
    for a in range(0, T, S):
        b = min(T, a + S)
        n = b - a
        if n not in cus:
            cus[n] = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        segs.append((a, b, cus[n]))
    def run():
        prev = s0
        for i, (a, b, cu) in enumerate(segs):
            nxt = st[i % 2]
            op.fwd(q[:, a:b], k[:, a:b], v[:, a:b], g1[:, a:b], beta[:, a:b], D ** -.5, out[:, a:b], ws, A, dtb, -5.0,
                   prev, nxt, cu, None, None)
            prev = nxt
        return prev
    return run, out
def med(fn, n=12):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
single(); torch.cuda.synchronize()
o0, f0 = out_ref.clone(), fs_ref.clone()
t0 = med(single)
print(f"T={T} single {t0:.3f} ms", flush=True)
for S in [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "128,256,384,512,768,1024,2048,4608").split(",")]:
    run, out = make_seg(S)
    fin = run(); torch.cuda.synchronize()
    same_o = torch.equal(out, o0); same_f = torch.equal(fin, f0)
    dmax = (out.float() - o0.float()).abs().max().item()
    t = med(run)
    print(f"  S={S:5d} segs={-(-T // S):3d} {t:.3f} ms ({t0 / t:.2f}x) out bit-identical {same_o} (max|d| {dmax:.2e}) "
          f"final-state bit-identical {same_f}", flush=True)
