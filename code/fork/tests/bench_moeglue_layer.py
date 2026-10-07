"""Production-shaped decode MoE layer, paired A/B inside CUDA graphs (docs/DEC_MOEGLUE.md).

Usage: tests/gpu_run.sh python3 -u tests/bench_moeglue_layer.py [--modes base,glue,...] [--T 1,5,8] [--prof]
Modes: base = production (K2 apply path, shared expert forked after the router), glue = GLM53_DEC_MOEGLUE path,
       base_pf / glue_pf = the same with the shared expert forked before the router (emulation of lever E).
Per T: 3 synthetic EXL3 layers x 4 routings (corr40 for T >= 5, rand for T < 5) = 12 layer calls per graph, cold
weights; rounds interleaved (order rotated / flipped); median per-layer-call us, spread, ratio vs base per round.
--prof: torch.profiler over 3 replays of each graph -> per-kernel median us and the per-layer main-stream span.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import statistics

import torch

import harness as H
import moeglue_rig as MR


def kernel_table(gr, calls, dev):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            gr.replay()
        torch.cuda.synchronize()
    agg = collections.defaultdict(list)
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = re.sub(r"\(.*", "", ev.name.replace("void ", "").replace("(anonymous namespace)::", ""))[:60]
        agg[name].append(ev.device_time if hasattr(ev, "device_time") else ev.cuda_time)
    rows = []
    for nm, v in agg.items():
        rows.append((sum(v) / (3 * calls), len(v) / (3 * calls), statistics.median(v), nm))
    rows.sort(reverse=True)
    return rows


def layer_spans(gr, calls):
    """Kineto trace of 3 replays: per layer call (router GEMV start -> end of the add), the pieces of the main-stream
    timeline: pre = router start -> first grouped start, mid = gate/up grouped end -> down grouped start, post = down
    grouped end -> end of the layer (the add). Medians over all calls."""
    import gzip, json, tempfile
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            gr.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        ev = json.load(open(f.name))["traceEvents"]
    k = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
    starts = [i for i, e in enumerate(k) if "gemm_bf16_kernel" in e["name"]]
    spans, pre, mid, post, grp = [], [], [], [], []
    for a, b in zip(starts, starts[1:] + [len(k)]):
        seg = k[a:b]
        g_ = [e for e in seg if "grouped_kernel" in e["name"]]
        adds = [e for e in seg if "CUDAFunctor_add" in e["name"] or "add_kernel" in e["name"]]
        if len(g_) != 2 or not adds:
            continue
        end = adds[-1]["ts"] + adds[-1]["dur"]
        spans.append(end - seg[0]["ts"])
        pre.append(g_[0]["ts"] - seg[0]["ts"])
        mid.append(g_[1]["ts"] - (g_[0]["ts"] + g_[0]["dur"]))
        post.append(end - (g_[1]["ts"] + g_[1]["dur"]))
        grp.append(g_[0]["dur"] + g_[1]["dur"])
    md = lambda v: statistics.median(v) if v else float("nan")
    return dict(n=len(spans), span=md(spans), pre=md(pre), grouped=md(grp), mid=md(mid), post=md(post))


def dump_layer(gr, calls, nlayers):
    """Kernel timeline (start offset from the layer's router GEMV start, duration, stream) of `nlayers` layer calls
    from the middle of the second replay."""
    import json, tempfile
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            gr.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        ev = json.load(open(f.name))["traceEvents"]
    k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"],
               key=lambda e: e["ts"])
    starts = [i for i, e in enumerate(k) if "gemm_bf16_kernel" in e["name"]]
    mid = calls + calls // 2
    streams = {}
    for li in range(mid, min(mid + nlayers, len(starts) - 1)):
        a, b = starts[li], starts[li + 1]
        t0 = k[a]["ts"]
        print(f"      --- layer call {li}", flush=True)
        for e in k[a:b]:
            s = e.get("args", {}).get("stream")
            sid = streams.setdefault(s, len(streams))
            nm = re.sub(r"\(.*", "", e["name"].replace("void ", "").replace("(anonymous namespace)::", ""))[:56]
            print(f"      {e['ts'] - t0:8.1f} {e['dur']:7.1f}  s{sid}  {nm}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="base,glue")
    ap.add_argument("--T", default="1,5,8")
    ap.add_argument("--kind", default="auto")
    ap.add_argument("--rounds", type=int, default=151)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--layers", type=int, default=42)
    ap.add_argument("--trellis", type=int, default=3)
    ap.add_argument("--window", type=float, default=0.0, help="spin (us) before every layer call in every mode")
    ap.add_argument("--mhc", action="store_true", help="production's mhc_fused_post_pre after the window spin")
    ap.add_argument("--dump", type=int, default=0, help="print the kernel timeline of this many layer calls")
    a = ap.parse_args()
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    print(f"K2 apply hook {'on' if rep.get('apply_hook') else 'off'}", flush=True)
    modes = a.modes.split(",")
    glue = None
    if any(m.startswith("glue") or m.startswith("sdlate") for m in modes):
        import glm53_moeglue as MG
        r2 = MG.install(prodmod=prod, force=True)
        print(f"moeglue install: {r2}", flush=True)
        glue = MG
    layers = MR.make_layers(prod, dev, a.trellis, a.layers)
    rig = MR.Rig(prod, layers, dev)
    print(f"{torch.cuda.get_device_name(0)}; {a.layers} layers; modes {modes}", flush=True)
    orig_apply = getattr(prod.apply_exl3_experts, "_glm53_moeglue_orig", prod.apply_exl3_experts)

    def apply_for(mode):
        if mode.startswith("sdlate"):
            pf = "nopf" not in mode

            def f(x, ids, w, layer, ev, pf=pf):
                return tf.apply_glue(x, ids, w, layer, layer._exl3_inners, None, MR.LIMIT, pf, None, ev)
            return f
        if mode.startswith("glue"):
            pf = "nopf" not in mode

            def f(x, ids, w, layer, pf=pf):
                glue.CFG.prefetch = pf
                return prod.apply_exl3_experts(x, ids, w, layer, limit=MR.LIMIT)
            return f
        return lambda x, ids, w, layer: orig_apply(x, ids, w, layer, limit=MR.LIMIT)

    for T in [int(t) for t in a.T.split(",")]:
        kind = a.kind if a.kind != "auto" else ("corr40" if T >= 5 else "rand")
        sets = MR.make_sets(kind, T, a.layers, 1, dev, seed=4242 + T)
        order = sets                                   # one call per layer, in layer order (a decode step)
        distinct = statistics.mean(int(torch.unique(s[2]).numel()) for s in order)
        graphs, outs = {}, {}
        main_hi = torch.cuda.Stream(device=dev, priority=-5)
        for m in modes:
            ap_ = apply_for(m)
            tags = set(m.split("+")[1:])
            pf = m.split("+")[0].endswith("_pf")
            sl = m.startswith("sdlate")
            fns = [lambda s=s, ap_=ap_, pf=pf, sl=sl: rig.call(s[0], s[1], s[2], s[3], apply=ap_, prefork=pf, sd_late=sl)
                   for s in order]
            rig.aux_prio = "hp" in tags                  # +hp: the shared expert's stream at high priority
            rig.window_us = next((float(t[1:]) for t in tags if t.startswith("w")), a.window)   # +w40: 40 us window
            rig.touch = next((t[1:] for t in tags if t.startswith("t")), "")                 # +trg: touch r, g
            rig.touch_blocks = next((int(t[1:]) for t in tags if t.startswith("b")), 96)     # +b48: touch blocks
            rig.touch_el = (1 if "el" in tags else 0) | (next((int(t[1:]) for t in tags if t.startswith("u")), 1) << 8)
            rig.mhc = "mhc" in tags or a.mhc                 # +mhc: production's mhc_fused_post_pre after the spin
            rig.warm_prod = "W" in tags                      # +W: production's l2_warm (f, r, g, d; 16 blocks)
            rig.warm_blocks = next((int(t[2:]) for t in tags if t.startswith("Wb")), 16)
            rig.warm_set = next((t[2:] for t in tags if t.startswith("Ws")), "frgd")          # +Wsfrg: regions
            if any(t.startswith("Wb") or t.startswith("Ws") for t in tags):
                rig.warm_prod = True
            graphs[m], outs[m] = MR.capture(fns, dev, stream=main_hi if "mhp" in tags else None)
            rig.aux_prio, rig.window_us, rig.touch, rig.touch_el, rig.mhc = False, 0.0, "", 0, False
            rig.warm_prod = False
        # what was timed: every mode's outputs vs base (bf16 layer outputs)
        for m in modes:
            graphs[m].replay()
        torch.cuda.synchronize()
        chk = []
        for m in modes:
            if m == modes[0]:
                continue
            worst, bit = 0.0, True
            for o, o0 in zip(outs[m], outs[modes[0]]):
                d = (o.float() - o0.float()).norm().item() / max(o0.float().norm().item(), 1e-30)
                worst = max(worst, d)
                bit &= bool(torch.equal(o, o0))
            chk.append(f"{m} vs {modes[0]}: rel_l2 max {worst:.2e} bitwise {bit}")
        res = MR.ab_rounds(graphs, len(order), a.rounds, a.reps)
        base = modes[0]
        line = [f"{kind} T={T} distinct {distinct:.1f}:"]
        for m in modes:
            ratios = [x / y for x, y in zip(res[m], res[base])]
            line.append(f"{m} {MR.med(res[m]):.1f} us (min {min(res[m]):.1f}, p10-p90 {MR.pct(res[m], .1):.1f}-"
                        f"{MR.pct(res[m], .9):.1f}; ratio median x{MR.med(ratios):.4f}, p25-p75 "
                        f"{MR.pct(ratios, .25):.4f}-{MR.pct(ratios, .75):.4f})")
        print(" | ".join(line), flush=True)
        for c in chk:
            print("   check:", c, flush=True)
        for m in modes:
            sp = layer_spans(graphs[m], len(order))
            print(f"   timeline [{m}] ({sp['n']} layer calls, medians): span {sp['span']:.1f} = pre {sp['pre']:.1f} + "
                  f"grouped g/u+d {sp['grouped']:.1f} + mid {sp['mid']:.1f} + post {sp['post']:.1f} us", flush=True)
        if a.dump:
            for m in modes:
                dump_layer(graphs[m], len(order), a.dump)
        if a.prof:
            for m in modes:
                rows = kernel_table(graphs[m], len(order), dev)
                tot = sum(r[0] for r in rows)
                print(f"   kernels [{m}] per layer call: sum {tot:.1f} us", flush=True)
                for us, n, mdn, nm in rows[:24]:
                    print(f"      {us:8.1f} us  x{n:4.2f}  median {mdn:7.1f}  {nm}", flush=True)
        del graphs, outs
        torch.cuda.synchronize()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
