#!/usr/bin/env python3
"""FKDA review check: error growth of the recurrent state across chained
prefill CHUNKS (the production handoff: each 13,824-token chunk's final state
is the next chunk's initial state), FlashKDA wrapper vs the Triton chain, both
graded against an fp64 per-token reference carried across the same chunks.

pfkda graded single calls up to 131k tokens (state fp32 between tiles inside
one call). state_carry_precision.py shows the carried-IN state is rounded to
bf16-level precision once per FlashKDA call, so the cross-chunk handoff is the
open accumulation path. Inputs: pfkda's "long" regime (g1 ~ N(0, 0.5)) with
the real checkpoint A_log/dt_bias (as /pf3000/real_regime.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/w/tests/fkda")
import wrapper_unit as wu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--T", type=int, default=13824)
    ap.add_argument("--ref", default="/fkda/kda_state_error.py")
    args = ap.parse_args()
    import importlib.util

    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("kse", args.ref)
    kse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kse)

    wu.install(Path("/usr/local/lib/python3.12/dist-packages"))
    import glm53_flashkda
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    layer = wu.FakeLayer()
    A = torch.from_numpy(np.load("/pf3000/real_A_log.npy")).float().cuda()
    B = torch.from_numpy(np.load("/pf3000/real_dt_bias.npy")).float().cuda()
    layer.A_log = A.reshape(layer.A_log.shape)
    layer.dt_bias = B.reshape(-1).contiguous()
    glm53_flashkda.configure(layer, wu.FakeCfg())
    H, D, T = 32, 128, args.T
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    s_ref = torch.zeros(1, H, D, D, dtype=torch.float64, device="cuda")
    s_fk = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
    s_tr = s_fk.clone()
    for c in range(args.chunks):
        q, k, v, g1, beta, _, _ = kse.make_inputs(T, H, torch.device("cuda"), seed=100 + c, regime="long")
        s_ref = kse.fp64_reference(q, k, v, g1, beta, A, B, s_ref)
        _, fs = glm53_flashkda.chunk_prefill(layer, q, k, v, g1, beta, s_fk, cu)
        s_fk = fs.clone()
        _, ht = wu.triton_ref(layer, q.clone(), k.clone(), v.clone(), g1, beta, s_tr, cu)
        s_tr = ht.float().clone()
        torch.cuda.synchronize()
        efk = kse.relstats(s_fk, s_ref)["rel_l2"]
        etr = kse.relstats(s_tr, s_ref)["rel_l2"]
        print(f"chunk {c + 1:2d} ({(c + 1) * T:7d} tok): state rel-L2 vs fp64  FlashKDA {efk:.4g}  "
              f"Triton {etr:.4g}  ratio {efk / etr:.3f}", flush=True)


if __name__ == "__main__":
    main()
