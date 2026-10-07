#!/usr/bin/env python3
"""FKDA3: kernel-level profile of the KDA prefill path around FlashKDA at production shapes
(H=32/rank, D=128, T=13,824, one sequence, fp32 initial state): the FlashKDA op (K1 prepare, K2 recurrence, the
in-op beta transpose), the layer-output merge copy kda.py does after the call, and o_norm (FusedRMSNormGated)."""
import argparse, json, sys, statistics
import torch
D = 128

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=13824)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--ext-dir", default="/w/overlay")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.ext_dir)
    import _flashkda_fp32_C  # noqa
    print("ext", _flashkda_fp32_C.__file__)
    T, H = a.T, a.H
    g = torch.Generator(device="cpu").manual_seed(1)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
    q, k, v = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D)
    g1 = rn(1, T, H, D, sc=0.5)
    beta = rn(1, T, 3 * H)[:, :, H:2 * H]
    s0 = (torch.randn(1, H, D, D, generator=g) * 0.3).cuda()
    A_log = (torch.randn(H, generator=g) * 0.2).cuda()
    dtb = (torch.rand(H, D, generator=g) * 8 - 10).cuda()
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    ws = torch.empty(int(torch.ops._flashkda_fp32_C.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    fs = torch.empty(1, H, D, D, dtype=torch.float32, device="cuda")
    core = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    g2 = rn(T, H, D)
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        onorm = FusedRMSNormGated(D, eps=1e-5, activation="sigmoid").cuda().to(torch.bfloat16)

    def op():
        torch.ops._flashkda_fp32_C.fwd(q, k, v, g1, beta, D ** -0.5, out, ws, A_log, dtb, -5.0, s0, fs, cu, None, None)

    def merge():
        core[0, :T] = out[0, :T]

    def norm():
        onorm(core, g2)

    def ev(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)
    res = {"op_ms": ev(op), "merge_copy_ms": ev(merge), "o_norm_ms": ev(norm)}
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(a.iters):
            op(); merge(); norm()
        torch.cuda.synchronize()
    ker = {}
    for e in p.events():
        if e.device_type.name == "CUDA":
            ker.setdefault(e.name[:90], []).append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
    res["kernels_us_per_iter"] = {k: sum(v) / a.iters for k, v in sorted(ker.items(), key=lambda kv: -sum(kv[1]))}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
