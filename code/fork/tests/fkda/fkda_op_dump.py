#!/usr/bin/env python3
"""FKDA check 1: the renamed _flashkda_fp32_C build is the same kernel as
pfkda's _flashkda_C (17a037d, 16-arg fp32-state fwd) and supports the varlen
(multi-sequence) cu_seqlens case the kill-test never measured.

One process per build (--ext {fp32,orig}) computes the op on identical inputs
(single sequence T=13824 AND a 3-sequence varlen call with per-sequence
initial states) and dumps outputs; the host script tests/fkda/check_rename.sh
compares the two dumps bitwise.

Also prints the registered schema so the record shows the namespace rename.
"""
import argparse

import torch

D = 128
LOWER_BOUND = -5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", choices=("fp32", "orig"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=13824)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    name = "_flashkda_fp32_C" if args.ext == "fp32" else "_flashkda_C"
    mod = __import__(name)
    print(f"imported {name}: fwd schema:")
    print("  ", torch.ops._flashkda_fp32_C.fwd.default._schema if args.ext == "fp32"
          else torch.ops._flashkda_C.fwd.default._schema)
    fwd = torch.ops._flashkda_fp32_C.fwd if args.ext == "fp32" else torch.ops._flashkda_C.fwd
    gw = torch.ops._flashkda_fp32_C.get_workspace_size if args.ext == "fp32" else torch.ops._flashkda_C.get_workspace_size

    H = 32
    g = torch.Generator(device="cpu").manual_seed(11)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).cuda().to(torch.bfloat16)

    T = args.tokens
    q, k, v, g1 = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D, scale=2.0)
    beta = rn(1, T, H)
    # realistic long-memory gate regime (kda_state_error.py --regime long):
    # dt_bias per channel U[-10,-2] and g1 ~ N(0, 0.5), so decays ~0.6..0.9998
    dtb = (torch.rand(H, D, generator=g) * 8 - 10).cuda().float()
    A_log = (torch.randn(H, generator=g) * 0.2).cuda().float()

    dump = {}
    for tag, cu, beta_arg in (
        ("single", [0, T], beta),
        # production's beta_ns is a column slice of the merged qkvbfg_a
        # projection, i.e. row-strided ([1, T, H] with token stride > H)
        ("varlen", [0, 5000, 7777, T], None),
    ):
        if beta_arg is None:
            beta_wide = rn(1, T, 2 * H)
            beta_arg = beta_wide[:, :, :H]
        cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
        N = len(cu) - 1
        s0 = (torch.randn(N, H, D, D, generator=g) * 0.3).cuda().float()
        s0[:, 0] = 0.0  # first sequence of each call starts from a fresh state
        ws = torch.empty(gw(T, H, N), dtype=torch.uint8, device="cuda")
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        fs = torch.empty(N, H, D, D, dtype=torch.float32, device="cuda")
        fwd(q.contiguous(), k.contiguous(), v.contiguous(), g1.contiguous(), beta_arg,
            D ** -0.5, out, ws, A_log, dtb, LOWER_BOUND, s0.contiguous(), fs, cu_t, None, None)
        torch.cuda.synchronize()
        dump[f"{tag}_out"] = out.cpu()
        dump[f"{tag}_fs"] = fs.cpu()
        dump[f"{tag}_s0"] = s0.cpu()
        print(f"{tag}: N={N} out {tuple(out.shape)} {out.dtype}, final_state {tuple(fs.shape)} {fs.dtype} "
              f"absmax {fs.abs().max().item():.4g}")
    torch.save({"A_log": A_log.cpu(), "dt_bias": dtb.cpu()}, args.out + ".params")
    torch.save(dump, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
