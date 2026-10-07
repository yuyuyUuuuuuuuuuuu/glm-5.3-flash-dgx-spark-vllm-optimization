"""Node1 A/B: production KDA decode with stock (four .contiguous() copies) vs the strided backport.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/bench_kda_strided_qkv.py

Measures, inside FULL CUDA graphs (production replays decode from graphs, so the captured work is
what production pays):
  - one KDA layer's recurrent call, batch 1, T = 1 / 5 / 8 tokens (8 = the production verify width,
    DFlash2 num_speculative_tokens=7 + 1)
  - a whole decode step's KDA cores: 34 layers (production GLM-5.3-Flash: 34 KDA of 45 layers),
    batch 1, 8 tokens — the per-step number the decision needs
  - CUDA kernel launches and GPU time per KDA layer call from torch.profiler, split kernel vs copies
Stock and patched run on the SAME static strided input buffers (identical bytes); graphs are replayed
in interleaved rounds, medians reported.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_strided_common as KC  # noqa: E402

import torch  # noqa: E402

DEV = "cuda"
H, K = 64, 128  # production GLM-5.3-Flash linear_attn_config: num_heads=64, head_dim=128
NLAYERS = 34  # production KDA layers of 45 (config linear_attn_config.kda_layers)
ROUNDS, REPS = 15, 20


def snap(src):
    return {n: (v.clone() if isinstance(v, torch.Tensor) else v) for n, v in src.items()}


def call(mod, cur, out=None):
    return KC.run_kda(mod, cur, out=out)


def build_step_graph(mod, inp: dict, num_layers: int):
    """FULL graph: `num_layers` sequential recurrent calls on the same static strided buffers,
    one out buffer per layer (production's spec path writes into per-layer output slices). The
    recurrent state flows in place, layer i+1 reads what layer i wrote, both sides identically."""
    T, HH, KK = inp["q"].shape[1], inp["q"].shape[2], inp["q"].shape[3]
    outs = [torch.empty(1, T, HH, KK, dtype=torch.bfloat16, device=DEV) for _ in range(num_layers)]

    def body():
        cur = dict(inp)
        for i in range(num_layers):
            outs[i], cur["initial_state"] = call(mod, cur, outs[i])
        return outs[-1]

    torch.cuda.synchronize()
    body()  # JIT / warm-up outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g


def time_graphs(graphs: dict, rounds=ROUNDS, reps=REPS) -> dict:
    out = {k: [] for k in graphs}
    for _ in range(rounds):
        for name, g in graphs.items():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            g.replay()  # warm replay, then the timed block (interleaved A/B)
            torch.cuda.synchronize()
            s.record()
            for _ in range(reps):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            out[name].append(s.elapsed_time(e) / reps)
    return {k: statistics.median(v) for k, v in out.items()}


def profile_call(mod, inp: dict, label: str):
    from torch.profiler import ProfilerActivity, profile

    def fresh():
        # keep the strided views (a clone of a column slice falls back to contiguous and would hide
        # the copies this counts); only the in-place-updated state gets its own buffer
        cur = dict(inp)
        cur["initial_state"] = inp["initial_state"].clone()
        return cur

    KC.run_kda(mod, fresh())  # JIT outside
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        KC.run_kda(mod, fresh())
        torch.cuda.synchronize()
    evs = [
        e
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]
    kern = sum(e.self_device_time_total for e in evs if "fused_recurrent" in e.key)
    copy = sum(e.self_device_time_total for e in evs if "fused_recurrent" not in e.key)
    cnt = sum(e.count for e in evs)
    copy_cnt = sum(e.count for e in evs if "fused_recurrent" not in e.key)
    print(
        f"    {label}: {cnt} launches ({copy_cnt} copies), GPU {kern:.1f} + copies {copy:.1f} = "
        f"{kern + copy:.1f} us per KDA layer call"
    )
    for e in sorted(evs, key=lambda x: -x.self_device_time_total):
        print(f"        {e.self_device_time_total:8.1f} us x{e.count:<3} {e.key[:110]}")
    return cnt, copy_cnt


def main() -> None:
    stock, patched, _, _ = KC.load_stock_and_patched()
    print(
        f"A/B on {torch.cuda.get_device_name(0)}; {NLAYERS} KDA layers/step; "
        f"{ROUNDS} interleaved rounds x {REPS} graph replays (medians)"
    )
    for ns, ql in ((1, 1), (1, 5), (1, 8)):
        T = ns * ql
        inp = KC.make_decode_inputs(ns, ql, H, K, DEV, seed=ns * 10 + ql)
        # the two paths must agree bitwise on these very buffers before anything is timed
        o_s, s_s = call(stock, snap(inp))
        o_p, s_p = call(patched, snap(inp))
        assert torch.equal(o_s, o_p), f"{ns}x{ql}: outputs diverged before timing"
        assert torch.equal(s_s, s_p), f"{ns}x{ql}: final states diverged before timing"

        one = time_graphs({"stock": build_step_graph(stock, inp, 1),
                           "patched": build_step_graph(patched, inp, 1)})
        step = time_graphs({"stock": build_step_graph(stock, inp, NLAYERS),
                            "patched": build_step_graph(patched, inp, NLAYERS)})

        print(f"- batch {ns}, verify width {ql} (T={T} tokens, H={H}, K={K})")
        print(
            f"    1 layer   : stock {one['stock']*1e3:8.2f} us | patched {one['patched']*1e3:8.2f} us"
            f" | delta {(one['patched']-one['stock'])*1e3:+7.2f} us"
        )
        print(
            f"    {NLAYERS}-layer step: stock {step['stock']:8.3f} ms | patched {step['patched']:8.3f} ms"
            f" | delta {(step['patched']-step['stock']):+.3f} ms/step"
        )
        profile_call(stock, inp, "stock  ")
        profile_call(patched, inp, "patched")


if __name__ == "__main__":
    main()
