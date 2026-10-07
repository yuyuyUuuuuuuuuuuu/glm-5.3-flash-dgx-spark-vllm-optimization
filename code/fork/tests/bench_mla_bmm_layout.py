"""Applicability probe for PR #55736's MLA half on production's sparse MLA backend (NoPE; the backend is fa2 on sm_12x, fa3 only on Hopper).

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/bench_mla_bmm_layout.py

Upstream #55736 (a) skips the empty-rope `torch.cat(q)` in `flashinfer_mla_sparse.py` and (b) writes the
absorbed-query bmm into a token-major buffer in `mla_attention.py` so the MQA query arrives contiguous.
Production runs `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py`: (a) is structurally absent
(that impl requires the `(q_nope, q_pe)` tuple and passes the halves to the kernel separately), and (b)'s
effect can only be the layout the flashinfer MLA kernel is handed. This probe measures exactly that, standalone:

  1. the absorbed-query bmm, old (out into (N,B,L), handed over transposed) vs new (out written straight
     into the transposed view of a token-major (B,N,L) buffer): bitwise equality and graph time;
  2. flashinfer 0.6.18's BatchMLAPagedAttentionWrapper with the patched sm90 impl's backend selection
     (fa2 on sm_12x, fa3 on Hopper) run with the old transposed-view q_nope vs the new contiguous
     q_nope: output equality and kernel launches/time from torch.profiler.

Not a production MLA bench: a small synthetic top-k width and kv cache, just enough to plan and run the
real kernel with the model's real head geometry (N=64, qk_nope_head_dim=256, kv_lora_rank=512,
qk_rope_head_dim=0 = NoPE; batch 1, 8 tokens = the production verify width). If even this layout has no
measurable effect, part (b) cannot pay off on this backend and is not ported.
"""
from __future__ import annotations

import statistics
import sys

import torch

DEV = "cuda"
N, P, L, ROPE = int(__import__("os").environ.get("MLA_PROBE_N", "64")), 256, 512, 0  # N: 64 = unsharded, 32 = per rank at TP=2; production text_config: num_attention_heads, qk_nope_head_dim,
# kv_lora_rank, qk_rope_head_dim (NoPE)
B = 8  # 1 request x 8 verify tokens
TOPK = 64  # synthetic top-k width (production 2048); layout probe only
PAGES = 8192
ROUNDS, REPS = 15, 20


def build_wrapper(kv_dtype=torch.bfloat16):
    """_SM90State's construction (flashinfer_mla_sparse_sm90.py:200-222), standalone."""
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    return (
        BatchMLAPagedAttentionWrapper(
            ws,
            qo_indptr=torch.zeros(B + 1, dtype=torch.int32, device=DEV),
            kv_indptr=torch.zeros(B + 1, dtype=torch.int32, device=DEV),
            kv_indices=torch.zeros(B * TOPK, dtype=torch.int32, device=DEV),
            kv_len_arr=torch.full((B,), TOPK, dtype=torch.int32, device=DEV),
            use_cuda_graph=True,
            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),
        ),
        ws,
    )


def main() -> None:
    print(f"probe on {torch.cuda.get_device_name(0)}: N={N} P={P} L={L} ROPE={ROPE} B={B}")

    # ---- 1. the absorbed-query bmm, old vs new out layout -------------------------------
    # identical q / weights; only the out buffer's layout differs
    mqa_q_nope = torch.randn(N, B, P, dtype=torch.bfloat16, device=DEV)
    W = torch.randn(N, P, L, dtype=torch.bfloat16, device=DEV)
    bufN = torch.empty(N, B, L, dtype=torch.bfloat16, device=DEV)
    torch.bmm(mqa_q_nope, W, out=bufN)
    bufB = torch.empty(B, N, L, dtype=torch.bfloat16, device=DEV)
    torch.bmm(mqa_q_nope, W, out=bufB.transpose(0, 1))
    print(
        "- bmm bitwise: old ((N,B,L)-contiguous out) vs new (straight into the transposed view of a "
        f"(B,N,L) buffer) equal: {torch.equal(bufN.transpose(0, 1), bufB)}"
    )

    def bmm_old():
        out = torch.empty(N, B, L, dtype=torch.bfloat16, device=DEV)
        torch.bmm(mqa_q_nope, W, out=out)
        return out.transpose(0, 1)

    def bmm_new():
        out = torch.empty(B, N, L, dtype=torch.bfloat16, device=DEV)
        torch.bmm(mqa_q_nope, W, out=out.transpose(0, 1))
        return out

    def time_one(fn, label):
        ts = []
        torch.cuda.synchronize()
        fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        for _ in range(ROUNDS):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            g.replay()
            torch.cuda.synchronize()
            s.record()
            for _ in range(REPS):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / REPS)
        print(f"  {label}: {statistics.median(ts)*1e3:8.2f} us/launch (median of {ROUNDS}x{REPS})")

    time_one(bmm_old, "bmm old ((N,B,L)-contiguous out, handed over transposed)")
    time_one(bmm_new, "bmm new (straight into the transposed view of a (B,N,L) buffer)")

    # ---- 2. the production MLA kernel (fa2 on sm_12x) with old vs new q_nope layout ----
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except ImportError as exc:
        print(f"SKIP MLA probe: flashinfer not importable in this container ({exc})")
        return
    wrapper, ws = build_wrapper()
    kv = torch.randn(PAGES, 1, L + ROPE, dtype=torch.bfloat16, device=DEV)
    ckv = kv.reshape(-1, 1, L + ROPE)[..., :L]
    kpe = kv.reshape(-1, 1, L + ROPE)[..., L:]
    q_pe = torch.zeros(B, N, ROPE, dtype=torch.bfloat16, device=DEV)
    lens = torch.full((B,), TOPK, dtype=torch.int32)
    qo = torch.arange(B + 1, dtype=torch.int32)  # one row per token (decode)
    kv_indptr = torch.arange(B + 1, dtype=torch.int32) * TOPK
    kv_idx = torch.randint(0, PAGES, (B * TOPK,), dtype=torch.int32)
    wrapper.plan(
        qo,
        kv_indptr,
        kv_idx_dev := kv_idx.to(DEV),
        lens,
        N,
        L,
        ROPE,
        1,
        False,
        1.0 / (P**0.5),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    # one deterministic bmm result, handed to the kernel in the old and the new layout
    mqa_q_nope = torch.randn(N, B, P, dtype=torch.bfloat16, device=DEV)
    W = torch.randn(N, P, L, dtype=torch.bfloat16, device=DEV)
    bufN = torch.empty(N, B, L, dtype=torch.bfloat16, device=DEV)
    torch.bmm(mqa_q_nope, W, out=bufN)
    q_old = bufN.transpose(0, 1)                       # old: transposed view
    q_new = bufN.transpose(0, 1).contiguous()          # new: token-major contiguous
    outs = {}
    for name, q in (("old(transposed view)", q_old), ("new(contiguous)", q_new)):
        outs[name] = wrapper.run(q, q_pe, ckv, kpe)
    print(
        f"- output bitwise equal old vs new q layout: "
        f"{torch.equal(outs['old(transposed view)'], outs['new(contiguous)'])}"
    )

    # interleaved A/B (kills warm-up/order bias), then one profiler pass each for the kernel identity
    stats = {"old(transposed view)": [], "new(contiguous)": []}
    for _ in range(ROUNDS):
        for name, q in (("old(transposed view)", q_old), ("new(contiguous)", q_new)):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wrapper.run(q, q_pe, ckv, kpe)
            torch.cuda.synchronize()
            s.record()
            for _ in range(REPS):
                wrapper.run(q, q_pe, ckv, kpe)
            e.record()
            torch.cuda.synchronize()
            stats[name].append(s.elapsed_time(e) / REPS)
    for name, v in stats.items():
        print(f"  run {name}: {statistics.median(v)*1e3:8.2f} us/run (median of {ROUNDS}x{REPS})")
    from torch.profiler import ProfilerActivity, profile
    for name, q in (("old(transposed view)", q_old), ("new(contiguous)", q_new)):
        wrapper.run(q, q_pe, ckv, kpe)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            wrapper.run(q, q_pe, ckv, kpe)
            torch.cuda.synchronize()
        evs = [
            ev
            for ev in prof.key_averages()
            if ev.device_type == torch.autograd.DeviceType.CUDA and ev.self_device_time_total > 0
        ]
        tot = sum(ev.self_device_time_total for ev in evs)
        print(f"  kernels for {name}: {sum(ev.count for ev in evs)} launches, {tot:.1f} us")
        for ev in sorted(evs, key=lambda x: -x.self_device_time_total)[:5]:
            print(f"      {ev.self_device_time_total:8.1f} us x{ev.count:<3} {ev.key[:100]}")


if __name__ == "__main__":
    main()
