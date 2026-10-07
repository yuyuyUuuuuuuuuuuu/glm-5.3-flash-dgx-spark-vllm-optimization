#!/usr/bin/env python3
"""FKDA review check: does the incoming fp32 recurrent state survive a FlashKDA
call at fp32 precision, or is it rounded to bf16 once per call?

With beta ~ 0 (raw logit -30) the delta-rule write vanishes and the final state
is just the gated decay of the initial state, so any error vs the Triton chain
(fp32 state in registers) is precision lost on the carried state itself. A
per-call bf16 rounding would show ~2e-3 relative; fp32 carry shows ~1e-6.
Production hands the state across every 13,824-token prefill chunk, so a
per-call rounding accumulates over an 800k-token session (~60 chunks).
"""
import sys
from pathlib import Path

sys.path.insert(0, "/w/tests/fkda")
import wrapper_unit as wu  # noqa: E402


def main():
    import numpy as np
    import torch

    wu.install(Path("/usr/local/lib/python3.12/dist-packages"))
    import glm53_flashkda
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    layer = wu.FakeLayer()
    a = torch.from_numpy(np.load("/pf3000/real_A_log.npy")).float().cuda()
    d = torch.from_numpy(np.load("/pf3000/real_dt_bias.npy")).float().cuda()
    layer.A_log = a.reshape(layer.A_log.shape)
    layer.dt_bias = d.reshape(-1).contiguous()
    glm53_flashkda.configure(layer, wu.FakeCfg())
    H, D = 32, 128
    g = torch.Generator(device="cpu").manual_seed(3)
    for T in (1, 16, 64, 4608):
        q = (torch.randn(1, T, H, D, generator=g)).cuda().bfloat16()
        k = (torch.randn(1, T, H, D, generator=g)).cuda().bfloat16()
        v = (torch.randn(1, T, H, D, generator=g)).cuda().bfloat16()
        g1 = (torch.randn(1, T, H, D, generator=g) * 0.5).cuda().bfloat16()
        beta = torch.full((1, T, H), -30.0, device="cuda").bfloat16()   # sigmoid ~ 1e-13: no write
        s0 = (torch.randn(1, H, D, D, generator=g) * 0.3 + 1e-3).cuda().float()
        cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        _, fs = glm53_flashkda.chunk_prefill(layer, q, k, v, g1, beta, s0.clone(), cu)
        fs = fs.clone()
        _, ht = wu.triton_ref(layer, q.clone(), k.clone(), v.clone(), g1, beta, s0.clone(), cu)
        torch.cuda.synchronize()
        s0_bf = s0.bfloat16().float()
        _, ht_bf = wu.triton_ref(layer, q.clone(), k.clone(), v.clone(), g1, beta, s0_bf, cu)
        torch.cuda.synchronize()
        print(f"T={T:5d}: FlashKDA vs Triton state rel-RMS {wu.rel_rms(ht, fs):.3g}; "
              f"reference: Triton(bf16-rounded s0) vs Triton {wu.rel_rms(ht, ht_bf):.3g}; "
              f"state norm ratio fs/s0 {float(fs.norm() / s0.norm()):.4f}")


if __name__ == "__main__":
    main()
