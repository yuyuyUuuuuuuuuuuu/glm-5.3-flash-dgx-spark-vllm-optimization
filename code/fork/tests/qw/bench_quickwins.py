"""GLM53_PREFILL_QUICKWINS: nodeC timing of each item, production statement vs fast path, production shapes (TP=2 rank:
32 MLA heads x (256 nope, 512 latent, 256 v), KDA 32 heads x 128 (q|k|v 3 x 4096 channels, in_proj width 12576, conv width
4), mHC 4 x 4096). Median of --iters CUDA-event timings after 3 warmups, rounds interleaved (prod, fast, prod, fast ...).
Per 13.8k chunk = per-call saving x calls per step (11 MLA layers, 34 KDA layers, 5 aux layers: prof5 trace).
"""
from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import types

import torch

sys.path.insert(0, "/w")
import glm53_prefill_quickwins as Q  # noqa: E402

DEV = "cuda"
CALLS = {"mla_bmm": 11, "mla_index": 11, "kda_conv": 34, "mhc_aux": 5, "mhc_mean": 1,   # mhc_mean: 5 aux + last
         "idx_gate": 11}


def timed(fn, iters):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def ab(prod, fast, iters, rounds=3):
    for _ in range(3):
        prod(); fast()
    torch.cuda.synchronize()
    tp, tf = [], []
    for _ in range(rounds):
        tp += timed(prod, iters)
        tf += timed(fast, iters)
    return statistics.median(tp), statistics.median(tf)


def kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type.name == "CUDA"]
    return len(ev), "; ".join(sorted({e.name[:48] for e in ev}))


def bench_mla_bmm(T, iters, g):
    N, P, L, V = 32, 256, 512, 256
    w = (torch.randn(N * (P + V), L, device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    W_UK, W_UV = w.T.view(L, N, P + V).split([P, V], dim=-1)
    WUKT, WUV = W_UK.permute(1, 2, 0), W_UV.transpose(0, 1)
    q = torch.randn(T, N, P, device=DEV, generator=g).to(torch.bfloat16)
    at = torch.randn(T, N, L, device=DEV, generator=g).to(torch.bfloat16)
    o1 = torch.empty(N, T, L, device=DEV, dtype=torch.bfloat16)
    o2 = torch.empty(T, N, V, device=DEV, dtype=torch.bfloat16)

    def prod():
        torch.bmm(q.transpose(0, 1), WUKT, out=o1)
        torch.bmm(at.transpose(0, 1), WUV, out=o2.transpose(0, 1))

    def fast():
        Q._qw_mla_bmm(q.transpose(0, 1), WUKT, o1)
        Q._qw_mla_bmm(at.transpose(0, 1), WUV, o2.transpose(0, 1))
    return prod, fast


def bench_mla_index(T, iters, g):
    sm = importlib.import_module(Q.M_SM90)
    conv = sm.triton_convert_req_index_to_global_index
    W, BS = 2048, 64
    gc = torch.Generator().manual_seed(1)
    ctx = [j + 1 for j in range(T)]
    rows = torch.full((T, W), -1, dtype=torch.int32)
    for i, c in enumerate(ctx):
        rows[i, :min(c, W)] = torch.arange(min(c, W), dtype=torch.int32) if c <= W else \
            torch.randperm(c, generator=gc)[:W].to(torch.int32)
    rows = rows.to(DEV)
    nb = (T + BS - 1) // BS + 1
    bt = torch.randperm(4 * nb, generator=gc)[:nb].to(torch.int32).view(1, nb).to(DEV)
    md = types.SimpleNamespace(req_id_per_token=torch.zeros(T, dtype=torch.int32, device=DEV), block_table=bt,
                               block_size=BS)
    st = types.SimpleNamespace(kv_indices=torch.zeros(16384 * W, dtype=torch.int32, device=DEV))

    def prod():
        slots, _ = conv(md.req_id_per_token[:T], md.block_table, rows, BLOCK_SIZE=BS, NUM_TOPK_TOKENS=W,
                        return_valid_counts=True)
        st.kv_indices[: T * W].copy_(slots.reshape(-1).clamp_(min=0).to(torch.int32))

    def fast():
        Q._qw_sm90_convert(md, rows, T, st, conv)
    return prod, fast


def bench_kda_conv(T, iters, g):
    conv_mod = importlib.import_module(Q.M_CONV)
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    P, WID = 4096, 4
    proj = torch.randn(T, 3 * P + 288, device=DEV, generator=g).to(torch.bfloat16)
    qkv = proj[:, : 3 * P]
    wgt = (torch.randn(3 * P, WID, device=DEV, generator=g) * 0.3).to(torch.bfloat16).contiguous()
    state = torch.randn(4, 3 * P, WID - 1, device=DEV, generator=g).to(torch.bfloat16)
    qsl_cpu = torch.tensor([0, T], dtype=torch.int32)
    nums, bptr, tptr = compute_causal_conv1d_metadata(qsl_cpu, device=torch.device(DEV))
    md = types.SimpleNamespace(nums_dict=nums, batch_ptr=bptr, token_chunk_offset_ptr=tptr)
    qsl = qsl_cpu.to(DEV)
    hi = torch.tensor([True], device=DEV)
    ci = torch.tensor([1], dtype=torch.int32, device=DEV)
    layer = types.SimpleNamespace(local_projection_size=P)

    def prod():   # merged conv + production's split, then FLA's q.contiguous() / k.contiguous() / v.contiguous()
        out = conv_mod.causal_conv1d_fn(qkv.transpose(0, 1), wgt, None, activation="silu", conv_states=state,
                                        has_initial_state=hi, cache_indices=ci, query_start_loc=qsl,
                                        metadata=md).transpose(0, 1)
        for t in out.split(P, dim=-1):
            t.contiguous()

    def fast():
        for t in Q._qw_kda_conv(layer, qkv, wgt, None, state, hi, ci, qsl, md, conv_mod.causal_conv1d_fn):
            t.contiguous()
    return prod, fast


def bench_mhc_aux(T, iters, g):
    import vllm.model_executor.kernels.mhc.tilelang as TL
    H, HC = 4096, 4
    fn = (torch.randn(24, HC * H, device=DEV, generator=g) * 0.02).float()
    hs = torch.tensor([0.9, 1.1, 0.7], device=DEV)
    hb = (torch.randn(24, device=DEV, generator=g) * 0.1).float()
    nw = torch.ones(H, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
    res = torch.randn(T, HC, H, device=DEV, generator=g).to(torch.bfloat16)
    post = torch.rand(T, HC, 1, device=DEV, generator=g)
    comb = torch.softmax(torch.randn(T, HC, HC, device=DEV, generator=g), -1)
    a = (1e-5, 1e-6, 1e-6, 2.0, 20)

    def prod():   # aux: post + mean, then the next layer's fused post -> pre
        full = TL.mhc_post_tilelang(x, res, post, comb)
        full.mean(dim=1)
        TL.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, hs, hb, *a, 1, 1, nw, 1e-5)

    def fast():   # aux: post + mean, then the next layer's standalone pre on the materialized streams
        full = TL.mhc_post_tilelang(x, res, post, comb)
        full.mean(dim=1)
        TL.mhc_pre_tilelang(full, fn, hs, hb, *a, 1, nw, 1e-5)
    return prod, fast


def bench_mhc_mean(T, iters, g):
    import vllm.model_executor.kernels.mhc.tilelang as TL
    H, HC = 4096, 4
    x = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
    res = torch.randn(T, HC, H, device=DEV, generator=g).to(torch.bfloat16)
    post = torch.rand(T, HC, 1, device=DEV, generator=g)
    comb = torch.softmax(torch.randn(T, HC, HC, device=DEV, generator=g), -1)

    def prod():   # 5 aux layers (post, mean) + the last layer (post, mean)
        for _ in range(6):
            TL.mhc_post_tilelang(x, res, post, comb).mean(dim=1)

    def fast():   # 5 aux layers (post + mean in one pass) + the last layer (mean only)
        for _ in range(5):
            Q.qw_post_mean(x, res, post, comb, True)
        Q.qw_post_mean(x, res, post, comb, False)
    return prod, fast


def bench_idx_gate(T, iters, g):
    import glm53_gemv_install as G
    G.register_ops()
    wb = (torch.randn(160, 4096, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w32 = wb[128:, :].t().contiguous().float()
    x = torch.randn(T, 4096, device=DEV, generator=g).to(torch.bfloat16)

    def prod():
        torch.mm(x.float(), w32)

    def fast():
        torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
    return prod, fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="13824,1791")
    ap.add_argument("--items", default=",".join(Q.ITEMS))
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    for m in {e[0] for v in Q.PLAN.values() for e in v}:
        importlib.import_module(m)
    r = Q.install(Q.ITEMS, 256)      # patches the production functions (kda_conv needs its conv-with-out variant)
    assert not r["pending"] and not Q._STATE["refused"], (r, Q._STATE["refused"])
    g = torch.Generator(device=DEV).manual_seed(0)
    fns = {"mla_bmm": bench_mla_bmm, "mla_index": bench_mla_index, "kda_conv": bench_kda_conv,
           "mhc_aux": bench_mhc_aux, "mhc_mean": bench_mhc_mean, "idx_gate": bench_idx_gate}
    tot = {}
    for T in (int(v) for v in args.ts.split(",")):
        for item in args.items.split(","):
            prod, fast = fns[item](T, args.iters, g)
            tp, tf = ab(prod, fast, args.iters)
            kp, kf = kernels(prod), kernels(fast)
            per_chunk = (tp - tf) * CALLS[item]
            tot[T] = tot.get(T, 0.0) + per_chunk
            print(f"T={T:6d} {item:9s} prod {tp:7.3f} ms ({kp[0]} kernels)  fast {tf:7.3f} ms ({kf[0]} kernels)  "
                  f"saving {tp - tf:6.3f} ms/call x {CALLS[item]} = {per_chunk:6.1f} ms/step", flush=True)
            print(f"           prod kernels: {kp[1]}\n           fast kernels: {kf[1]}", flush=True)
            del prod, fast
            torch.cuda.empty_cache()
    print(f"STATS {Q.STATS}")
    for T, v in tot.items():
        print(f"T={T}: all items {v:.1f} ms per step (rank 0 kernel time)")


if __name__ == "__main__":
    main()
