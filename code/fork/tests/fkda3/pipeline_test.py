#!/usr/bin/env python3
"""FKDA3 opt-in K1->K2 pipeline (FKDA3_PIPELINE=1): correctness (bit-identical to the sequential path and to the
pipeline-free fkda3 build), race stress, fallbacks, and speed.
  P.1 bitwise: f3p(env unset) == f3p(env 1) == f3_all for T in {1,15,16,17,64,1000,1791,4608,13824}, zero/random state
  P.2 varlen N=2 (H*N=64 > 48-8: falls back to sequential) still bitwise equal
  P.3 race stress: --stress pipelined calls at T=13,824 (and alternating T) each compared bitwise to the reference
  P.4 timing (interleaved): f3_all vs f3p env-unset vs f3p env-1, per layer and per 13,824-token chunk (34 layers)
"""
import argparse
import os
import statistics
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stress", type=int, default=300)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--delays", default="0,5000,20000,50000")
a = ap.parse_args()
sys.path.insert(0, "/fkda/builds/f3p_all")
sys.path.insert(0, "/fkda/builds/f3_all")
import _fk3n_C  # noqa: E402,F401
import _fk3p_C  # noqa: E402,F401
P, R = torch.ops._fk3p_C, torch.ops._fk3n_C
D, H = 128, 32
FAIL = []


def check(c, m):
    print(("ok   " if c else "FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def mk(lens, seed, init):
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = sum(lens)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)  # noqa: E731
    x = dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, sc=0.5),
             b=rn(1, T, 3 * H)[:, :, H:2 * H],
             s0=(torch.zeros(len(lens), H, D, D, device="cuda") if init == "zero"
                 else (torch.randn(len(lens), H, D, D, generator=g) * 0.05).cuda()),
             A=(torch.randn(H, generator=g) * .2).cuda(), dtb=(torch.rand(H, D, generator=g) * 8 - 10).cuda(),
             cu=torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"))
    N = len(lens)
    x["ws"] = torch.empty(int(P.get_workspace_size(T, H, N)), dtype=torch.uint8, device="cuda")
    x["out"] = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    x["fs"] = torch.empty(N, H, D, D, device="cuda")
    return x


def run(op, x):
    x["out"].fill_(float("nan"))
    x["fs"].fill_(float("nan"))
    op.fwd(x["q"], x["k"], x["v"], x["g"], x["b"], D ** -.5, x["out"], x["ws"], x["A"], x["dtb"], -5.0, x["s0"],
           x["fs"], x["cu"], None, None)
    return x["out"].clone(), x["fs"].clone()


def env(on):
    if on:
        os.environ["FKDA3_PIPELINE"] = "1"
    else:
        os.environ.pop("FKDA3_PIPELINE", None)


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# P.1 / P.2
for lens in ([1], [15], [16], [17], [64], [1000], [1791], [4608], [13824], [700, 3000]):
    for init in ("zero", "rand"):
        x = mk(lens, sum(lens), init)
        env(False)
        ref = run(R, x)
        off = run(P, x)
        env(True)
        on = run(P, x)
        env(False)
        tag = "P.2" if len(lens) > 1 else "P.1"
        check(same(ref, off) and same(ref, on) and bool(torch.isfinite(on[0]).all()),
              f"{tag} lens={lens} init={init}: f3_all == f3p(off) == f3p(pipelined)")

# P.3 race stress
xs = [mk([13824], 77, "rand"), mk([4608], 78, "rand"), mk([1791], 79, "zero")]
env(False)
refs = [run(R, x) for x in xs]
env(True)
bad = 0
for i in range(a.stress):
    j = i % len(xs) if i % 2 else 0
    got = run(P, xs[j])
    if not same(got, refs[j]):
        bad += 1
torch.cuda.synchronize()
env(False)
check(bad == 0, f"P.3 race stress: {a.stress} pipelined calls bit-identical to the sequential reference ({bad} differ)")

# P.4 timing
for T in (13824, 4608, 1791):
    x = mk([T], 5, "rand")
    arms = {"f3_all": (R, False), "f3p_off": (P, False)}
    for dl in a.delays.split(","):
        arms[f"pipe_d{dl}"] = (P, dl)
    ts = {k: [] for k in arms}
    order = list(arms)
    for r in range(a.rounds):
        for name in (order if r % 2 == 0 else order[::-1]):
            op, on = arms[name]
            env(bool(on))
            if on and on is not True:
                os.environ["FKDA3_PIPELINE_DELAY_NS"] = on

            def fn():
                op.fwd(x["q"], x["k"], x["v"], x["g"], x["b"], D ** -.5, x["out"], x["ws"], x["A"], x["dtb"], -5.0,
                       x["s0"], x["fs"], x["cu"], None, None)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            for _ in range(10):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); fn(); e.record(); torch.cuda.synchronize()
                ts[name].append(s.elapsed_time(e))
    env(False)
    m = {k: statistics.median(v) for k, v in ts.items()}
    print(f"P.4 T={T}: " + " | ".join(f"{k} {v:.3f} ms" for k, v in m.items())
          + " | best pipeline saving " + f"{(m['f3p_off'] - min(v for k, v in m.items() if k.startswith('pipe'))):.3f} ms/layer = "
          f"{(m['f3p_off'] - min(v for k, v in m.items() if k.startswith('pipe'))) * 34:.1f} ms/chunk; f3p_off vs f3_all {(m['f3p_off'] - m['f3_all']) * 34:+.1f} ms/chunk",
          flush=True)
print("PIPELINE TEST:", "ALL OK" if not FAIL else f"{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
