"""Reviewer's nodeC check of GLM53_KDA_STRIDED_QKV at PRODUCTION PER-RANK shapes (TP=2: 32 KDA heads per rank).

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/review_kdaqkv_tp2.py [--timing-only|--no-timing]

Differences from tests/test_kda_strided_qkv.py / bench_kda_strided_qkv.py (implementer's, H=64 = the unsharded model):
  - H = 32 per rank (production: TP=2 on two GB10 nodes; in_proj_qkvbfg_a is 12576 wide per rank = 3*32*128 + 32 + 2*128,
    the width tests/r16/bk_rig.py uses and DEC_SMALLOPS.md's production trace shows as grid [1,16,32]).
  - Production's layout: causal_conv1d_update writes IN PLACE (out = x), so on a pure spec-verify step q/k/v AND beta are
    all column slices of ONE `projected` buffer with row stride 12576 (the implementer's rig used a separate [T, 3P]
    conv buffer). The mixed-step layout (index_select -> contiguous [n, 3P] conv, contiguous beta) is covered too.
  - Bitwise over the WHOLE state tensor (every slot, not only the written ones) and through out= (production passes
    core_attn_out[0, :n].unsqueeze(0)).
  - Production verify widths T = 5 / 6 / 8 (GLM53_ADAPTIVE_K_SET=4,5,7), batch 1..8.
  - A 34-layer FULL-graph step with PER-LAYER state / input / output buffers (production: every KDA layer has its own
    recurrent-state cache), stock vs patched graphs compared bitwise after re-filling the static inputs with new data
    (the graph must read live buffers, not baked values).
  - Timing: paired, interleaved, alternating order, 3 independent blocks; bare chain and a chain with in_proj/o_proj
    sized DRAM reads around each KDA core (51.5 MB + 16.8 MB per layer, distinct per layer) so the copies are measured
    in a production-like stream, not only back-to-back.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_strided_common as KC  # noqa: E402

import torch  # noqa: E402

from harness import Checks, run_main  # noqa: E402

DEV = "cuda"
H, D = 32, 128                     # per rank at TP=2
P = H * D                          # 4096
W = 3 * P + H + 2 * D              # 12576 = in_proj_qkvbfg_a per rank
NLAYERS = 34
LB = -5.0


def make_layer(num_seqs: int, qlen: int, gen: torch.Generator, layout: str = "inplace"):
    n = num_seqs * qlen
    if layout == "inplace":        # pure spec-verify step: conv wrote into `projected` in place
        projected = torch.randn(n, W, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEV)
        q, k, v = (projected[:, i * P:(i + 1) * P].reshape(1, -1, H, D) for i in range(3))
        beta = projected[:, 3 * P:3 * P + H].unsqueeze(0)
        keep = projected
    else:                          # mixed step: index_select -> contiguous [n, 3P] conv buffer, contiguous beta
        conv = torch.randn(n, 3 * P, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEV)
        q, k, v = (conv[:, i * P:(i + 1) * P].reshape(1, -1, H, D) for i in range(3))
        beta = torch.randn(1, n, H, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEV)
        keep = conv
    g = torch.randn(1, n, H, D, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEV)
    nslots = n + 2
    state = torch.randn(nslots, H, D, D, generator=gen, dtype=torch.float32).to(DEV) * 0.1
    slots = (torch.randperm(nslots - 1, generator=gen)[:n] + 1).to(torch.int32).to(DEV)
    inp = dict(
        q=q, k=k, v=v, g=g, beta=beta,
        a_log=(0.5 * torch.randn(H, generator=gen)).to(DEV),
        g_bias=(0.1 * torch.randn(H * D, generator=gen)).to(DEV),
        cu_seqlens=torch.arange(0, n + 1, qlen, dtype=torch.int32, device=DEV),
        initial_state=state,
        ssm_state_indices=slots if qlen == 1 else slots.view(num_seqs, qlen),
        _keep=keep,
    )
    if qlen > 1:
        inp["num_accepted_tokens"] = torch.randint(1, qlen + 1, (num_seqs,), generator=gen,
                                                   dtype=torch.int32).to(DEV)
    return inp


def run(mod, inp, out=None):
    return mod.fused_recurrent_kda(
        q=inp["q"], k=inp["k"], v=inp["v"], g=inp["g"], beta=inp["beta"],
        initial_state=inp["initial_state"], use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"], ssm_state_indices=inp["ssm_state_indices"],
        num_accepted_tokens=inp.get("num_accepted_tokens"), out=out, sigmoid_beta=True,
        a_log=inp["a_log"], g_bias=inp["g_bias"], compute_gate=True, lower_bound=LB)


def with_state(inp, state):
    d = dict(inp)
    d["initial_state"] = state
    return d


def bitwise(checks: Checks, stock, patched) -> None:
    gen = torch.Generator().manual_seed(1234)
    cases = [(1, 5, "inplace"), (1, 6, "inplace"), (1, 8, "inplace"), (2, 8, "inplace"), (8, 8, "inplace"),
             (8, 5, "inplace"), (3, 6, "mixed"), (1, 8, "mixed"), (1, 1, "inplace"), (4, 1, "inplace"),
             (4, 1, "mixed")]
    for ns, ql, layout in cases:
        inp = make_layer(ns, ql, gen, layout)
        n = ns * ql
        tag = f"H={H} {ns}x{ql} {layout}"
        noncontig = [nm for nm in ("q", "k", "v", "beta") if not inp[nm].is_contiguous()]
        s0 = inp["initial_state"].clone()
        st_s, st_p = s0.clone(), s0.clone()
        o_s, fs = run(stock, with_state(inp, st_s))
        o_p, fp = run(patched, with_state(inp, st_p))
        checks(fs.data_ptr() == st_s.data_ptr() and fp.data_ptr() == st_p.data_ptr(), f"[{tag}] state not in place")
        checks(torch.equal(o_s, o_p), f"[{tag}] output differs (bf16, bitwise)")
        checks(torch.equal(st_s, st_p), f"[{tag}] WHOLE state tensor differs (fp32, bitwise)")
        checks(not torch.equal(st_s, s0), f"[{tag}] vacuous: the kernel wrote no state")
        # out= path, as production passes core_attn_out[0, :n].unsqueeze(0)
        core = torch.full((1, n + 3, H, D), float("nan"), dtype=torch.bfloat16, device=DEV)
        st_o = s0.clone()
        o_o, _ = run(patched, with_state(inp, st_o), out=core[0, :n].unsqueeze(0))
        checks(o_o.data_ptr() == core.data_ptr(), f"[{tag}] out= not written in place")
        checks(torch.equal(core[0, :n], o_s[0]), f"[{tag}] out= path differs from stock")
        checks(bool(torch.isnan(core[0, n:].float()).all()), f"[{tag}] out= wrote past n")
        checks(torch.equal(st_o, st_s), f"[{tag}] out= path state differs")
        print(f"  ok   {tag}: non-contiguous inputs {noncontig or 'none'}; out/state/out= bitwise == stock", flush=True)


def build_chain(mod, layers, outs, fillers=None):
    def body():
        for i, L in enumerate(layers):
            if fillers is not None:
                x, w_in, w_o, y_in, y_o = fillers[i]
                torch.mm(x, w_in, out=y_in)
            run(mod, L, out=outs[i])
            if fillers is not None:
                torch.mm(outs[i].view(outs[i].shape[1], -1), w_o, out=y_o)
    return body


def capture(body):
    torch.cuda.synchronize()
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    torch.cuda.synchronize()
    return g


def full_graph(checks: Checks, stock, patched) -> None:
    gen = torch.Generator().manual_seed(77)
    for ns, ql in ((1, 8), (4, 6)):
        n = ns * ql
        L = [make_layer(ns, ql, gen) for _ in range(NLAYERS)]
        s0 = [x["initial_state"].clone() for x in L]
        Ls = [with_state(x, s.clone()) for x, s in zip(L, s0)]
        Lp = [with_state(x, s.clone()) for x, s in zip(L, s0)]
        outs_s = [torch.empty(1, n, H, D, dtype=torch.bfloat16, device=DEV) for _ in L]
        outs_p = [torch.empty(1, n, H, D, dtype=torch.bfloat16, device=DEV) for _ in L]
        gs = capture(build_chain(stock, Ls, outs_s))
        gp = capture(build_chain(patched, Lp, outs_p))
        for trial in range(2):
            # new live data in the static buffers (inputs and states), then replay both graphs
            fresh_states = []
            for x, a, b in zip(L, Ls, Lp):
                x["_keep"].copy_(torch.randn(x["_keep"].shape, generator=gen).to(torch.bfloat16))
                x["g"].copy_(torch.randn(x["g"].shape, generator=gen).to(torch.bfloat16))
                fresh = (0.1 * torch.randn(x["initial_state"].shape, generator=gen)).to(DEV)
                fresh_states.append(fresh)
                a["initial_state"].copy_(fresh)
                b["initial_state"].copy_(fresh)
            for o in outs_s + outs_p:
                o.fill_(float("nan"))
            gs.replay()
            gp.replay()
            torch.cuda.synchronize()
            ok_o = all(torch.equal(a, b) for a, b in zip(outs_s, outs_p))
            ok_s = all(torch.equal(a["initial_state"], b["initial_state"]) for a, b in zip(Ls, Lp))
            no_nan = all(not torch.isnan(o.float()).any() for o in outs_p)
            checks(ok_o and no_nan, f"[FULL {ns}x{ql} trial {trial}] stock vs patched graph outputs differ / NaN left")
            checks(ok_s, f"[FULL {ns}x{ql} trial {trial}] stock vs patched graph states differ")
            # the patched replay must equal an eager patched run on the SAME new data (not values baked at capture)
            ok_e = True
            for i, x in enumerate(L):
                st = fresh_states[i].clone()
                o_e = torch.empty(1, n, H, D, dtype=torch.bfloat16, device=DEV)
                run(patched, with_state(x, st), out=o_e)
                ok_e &= torch.equal(o_e, outs_p[i]) and torch.equal(st, Lp[i]["initial_state"])
            torch.cuda.synchronize()
            checks(ok_e, f"[FULL {ns}x{ql} trial {trial}] patched replay != eager patched on the refilled buffers")
            print(f"  ok   FULL {NLAYERS}-layer graph {ns}x{ql} trial {trial}: stock == patched bitwise "
                  f"(outputs {ok_o}, states {ok_s}); replay == eager on refilled data {ok_e}", flush=True)
        del gs, gp


def kernel_count(checks: Checks, stock, patched) -> None:
    from torch.profiler import ProfilerActivity, profile
    gen = torch.Generator().manual_seed(5)
    for ql in (5, 8):
        L = make_layer(1, ql, gen)
        res = {}
        for name, mod in (("stock", stock), ("patched", patched)):
            Lx = with_state(L, L["initial_state"].clone())
            out = torch.empty(1, ql, H, D, dtype=torch.bfloat16, device=DEV)
            g = capture(lambda: run(mod, Lx, out=out))
            g.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                g.replay()
                torch.cuda.synchronize()
            evs = [e for e in prof.key_averages()
                   if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
            res[name] = sum(e.count for e in evs)
            print(f"    {name} T={ql}: {res[name]} kernels in the replay: "
                  + "; ".join(f"{e.key[:48]} x{e.count} {e.self_device_time_total:.1f}us" for e in evs), flush=True)
        checks(res["stock"] == 5 and res["patched"] == 1,
               f"T={ql}: kernels per layer stock {res['stock']} (want 5) patched {res['patched']} (want 1)")


def timing(checks: Checks, stock, patched, fill: bool, widths=(5, 6, 8), blocks=3, rounds=12, reps=10) -> None:
    gen = torch.Generator().manual_seed(99)
    for ql in widths:
        L = [make_layer(1, ql, gen) for _ in range(NLAYERS)]
        s0 = [x["initial_state"].clone() for x in L]
        Ls = [with_state(x, s.clone()) for x, s in zip(L, s0)]
        Lp = [with_state(x, s.clone()) for x, s in zip(L, s0)]
        outs_s = [torch.empty(1, ql, H, D, dtype=torch.bfloat16, device=DEV) for _ in L]
        outs_p = [torch.empty(1, ql, H, D, dtype=torch.bfloat16, device=DEV) for _ in L]
        fillers = None
        if fill:
            # per layer: in_proj-sized read (FP8 12576x4096 = 51.5 MB -> bf16 4096x6288) and o_proj-sized read
            # (FP8 4096x4096 = 16.8 MB -> bf16 4096x2048); distinct per layer so nothing stays in L2
            fillers = []
            for _ in range(NLAYERS):
                x = torch.randn(ql, 4096, device=DEV, dtype=torch.bfloat16)
                w_in = torch.randn(4096, 6288, device=DEV, dtype=torch.bfloat16)
                w_o = torch.randn(4096, 2048, device=DEV, dtype=torch.bfloat16)
                fillers.append((x, w_in, w_o, torch.empty(ql, 6288, device=DEV, dtype=torch.bfloat16),
                                torch.empty(ql, 2048, device=DEV, dtype=torch.bfloat16)))
        gs = capture(build_chain(stock, Ls, outs_s, fillers))
        gp = capture(build_chain(patched, Lp, outs_p, fillers))
        graphs = {"stock": gs, "patched": gp}
        block_deltas, arms = [], {"stock": [], "patched": []}
        for b in range(blocks):
            diffs = []
            for r in range(rounds):
                order = ("stock", "patched") if (r + b) % 2 == 0 else ("patched", "stock")
                t = {}
                for name in order:
                    g = graphs[name]
                    g.replay()
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(reps):
                        g.replay()
                    e.record()
                    torch.cuda.synchronize()
                    t[name] = s.elapsed_time(e) / reps
                    arms[name].append(t[name])
                diffs.append(t["patched"] - t["stock"])
            block_deltas.append(statistics.median(diffs))
        ms = statistics.median(arms["stock"])
        mp = statistics.median(arms["patched"])
        print(f"  {'filler' if fill else 'bare  '} T={ql} H={H} {NLAYERS} layers: stock {ms:.3f} ms, patched {mp:.3f} ms; "
              f"paired delta per block (median of {rounds}): "
              + ", ".join(f"{d:+.3f}" for d in block_deltas) + f" ms/step; per layer "
              f"{statistics.median(block_deltas) / NLAYERS * 1e3:+.2f} us", flush=True)
        del gs, gp, fillers
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    checks = Checks()
    stock, patched, _, _ = KC.load_stock_and_patched()
    print(f"device {torch.cuda.get_device_name(0)}; per-rank H={H} D={D} W={W}", flush=True)
    if not a.timing_only:
        print("- bitwise, production per-rank layouts", flush=True)
        bitwise(checks, stock, patched)
        print("- FULL CUDA graphs, 34 layers, per-layer buffers", flush=True)
        full_graph(checks, stock, patched)
        print("- kernels per layer in a FULL replay", flush=True)
        kernel_count(checks, stock, patched)
    if not a.no_timing:
        print("- timing (paired, interleaved, alternating order; 3 blocks)", flush=True)
        timing(checks, stock, patched, fill=False)
        timing(checks, stock, patched, fill=True, widths=(5, 8))
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    checks.summary()


if __name__ == "__main__":
    run_main(main)
