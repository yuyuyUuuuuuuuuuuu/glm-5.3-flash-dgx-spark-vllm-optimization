#!/usr/bin/env python3
"""FKDA2: are two _flashkda_fp32_C builds bit-identical on the same inputs? (the reproducibility check behind the
precision A/B: build_variant.sh with every FKDA2_* switch 0 must reproduce the shipped overlay .so's outputs bit for
bit, so every difference of the fkda2 build is the patch). Single-sequence T=13824 with zero state, a 4-row varlen
call (1 / 15 / 200 / 4600 tokens, two with nonzero initial state), real-checkpoint A_log/dt_bias when staged.
Usage (inside tests/fkda/gpu_run.sh): check_bitwise.py --a <ext dir> --b <ext dir>   (exit 1 on any difference)"""
import argparse
import os
import subprocess
import sys

import torch

D, H = 128, 32


def inputs():
    g = torch.Generator(device="cpu").manual_seed(3)

    def rn(*s, scale=1.0):
        return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)
    cases = []
    for cu in ([0, 13824], [0, 1, 16, 216, 4816]):
        T, N = cu[-1], len(cu) - 1
        s0 = torch.zeros(N, H, D, D)
        if N > 1:
            s0[1] = torch.randn(H, D, D, generator=g) * 0.3
            s0[3] = torch.randn(H, D, D, generator=g) * 0.3
        A = torch.randn(H, generator=g) * 0.2
        dt = torch.rand(H, D, generator=g) * 8 - 10
        if os.path.exists("/pf3000/real_A_log.npy"):
            import numpy as np
            A = torch.from_numpy(np.load("/pf3000/real_A_log.npy")).float().reshape(-1)
            dt = torch.from_numpy(np.load("/pf3000/real_dt_bias.npy")).float().reshape(H, D)
        cases.append(dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, scale=0.5),
                          beta=rn(1, T, H), s0=s0, A=A, dt=dt, cu=torch.tensor(cu, dtype=torch.int32)))
    return cases


def run(ext, out):
    sys.path.insert(0, ext)
    import _flashkda_fp32_C as X  # noqa: F401
    res = []
    for c in inputs():
        c = {k: v.cuda() for k, v in c.items()}
        T, N = c["q"].shape[1], c["cu"].numel() - 1
        ws = torch.empty(int(torch.ops._flashkda_fp32_C.get_workspace_size(T, H, N)), dtype=torch.uint8, device="cuda")
        o = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        f = torch.empty(N, H, D, D, dtype=torch.float32, device="cuda")
        torch.ops._flashkda_fp32_C.fwd(c["q"], c["k"], c["v"], c["g"], c["beta"], D ** -0.5, o, ws, c["A"], c["dt"],
                                       -5.0, c["s0"], f, c["cu"], None, None)
        torch.cuda.synchronize()
        res.append((o.cpu(), f.cpu()))
    torch.save({"file": X.__file__, "res": res}, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--_run", nargs=2)
    a = ap.parse_args()
    if a._run:
        return run(*a._run)
    for tag, ext in (("a", a.a), ("b", a.b)):
        subprocess.run([sys.executable, __file__, "--_run", ext, f"/tmp/bw_{tag}.pt"], check=True)
    A, B = torch.load("/tmp/bw_a.pt"), torch.load("/tmp/bw_b.pt")
    print("a:", A["file"], "\nb:", B["file"])
    bad = 0
    for i, ((oa, fa), (ob, fb)) in enumerate(zip(A["res"], B["res"])):
        eo = torch.equal(oa.view(torch.int16), ob.view(torch.int16))
        ef = torch.equal(fa.view(torch.int32), fb.view(torch.int32))
        print(f"case {i}: out bit-equal {eo} (max|d| {float((oa.float() - ob.float()).abs().max()):.3g}), "
              f"final state bit-equal {ef} (max|d| {float((fa - fb).abs().max()):.3g})")
        bad += (not eo) + (not ef)
    print("BITWISE:", "ALL EQUAL" if not bad else f"{bad} DIFFER")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
