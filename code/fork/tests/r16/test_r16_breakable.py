"""[sidestream] GLM53_DEC_FP8ROOF / GLM53_DEC_MOEGLUE_WARM under vLLM's BREAKABLE CUDA-graph capture (production's
PIECEWISE graphs; the first deploy-r16 boot failed in "Capturing CUDA graphs (PIECEWISE): 0/23" with
breakable_cudagraph.py add_eager -> _end_segment -> capture_end: "capturing stream has unjoined work").
Stack: tests/r16/bk_rig.py (vLLM's own Glm5NextModel / DecoderLayer / KDA / MLA-wrapper / MoE forwards on shells,
production FP8 linears + fp8_gemv + fp8_roof hooks, real EXL3 routed experts via Exl3MoEMethod.apply -> moeglue join ->
TF K2, moeglue warm_post_load wiring, quickwins mhc_mean + kda_conv installed; eager breaks = vLLM's real decorator).
Every case runs in its own interpreter (a failed capture leaves CUDA in an undefined state).

R16_BK_EXPECT=before (the kit production runs now, 0501c3e, bound over /w/fp8_roof.py and /w/glm53_moeglue.py):
  B.1  PIECEWISE M=8, fp8roof (all triggers): capture fails with "capturing stream has unjoined work" at the first KDA
       layer's eager break (traceback: kda.py forward -> breakable_cudagraph wrapper -> add_eager -> _end_segment ->
       capture_end), the pending fork = t1 (in_proj(0) -> o_proj(0))
  B.2  the same with production's own (quickwins-recompiled, decorated) Glm5NextLinearAttention._forward
  B.3  PIECEWISE M=8, moeglue warm only: capture ok, replay == eager bitwise x3 (the warm window has no break)
  B.4  PIECEWISE M=8, both: fails as B.1
  B.5  each trigger alone: t1 and t4 fail (join behind an eager break), t2 and t3 capture (and replay bitwise)
  B.6  FULL M=8, both (the regime nodeC's earlier tests covered): capture ok, replay bitwise
default (R16_BK_EXPECT=after, this tree):
  A.1  installs: the three production break points are decorated; roof guard hooked, t1/t4 in bk_skip; warm guard
       hooked; warm wired 5 MoE sublayers; one wrapper per module in BreakableCUDAGraphCapture's chain
  A.2  eager M=5, off / roof / warm / both: forks t1 6 t2 8 t3 7 t4 2 warm 5 (unchanged: no capture), bitwise equal
  A.3  PIECEWISE M in 1,5,8,16,32,64, both on: capture ok; forks per capture t2 8 t3 7 warm 5, t1 / t4 0 (skipped 6 / 2),
       nothing joined at a break or at the capture end, no pending fork; replay == eager (both off) bitwise x3 with
       fresh inputs; counters frozen during replay
  A.4  FULL M in 1,5,8,16,64 after the PIECEWISE captures (production's order), both on: forks as eager (t1 6 t2 8 t3 7
       t4 2 warm 5), no skip; replay bitwise; M=65 eager: no fork
  A.5  PIECEWISE M=8: roof only, warm only, and each trigger alone: capture ok, replay bitwise
  A.6  production's own KDA _forward (kda_core=real): PIECEWISE and FULL capture ok, replays run
  A.7  guard (static skip emptied): t1 and t4 fork, are joined at the break (1 each), learned (WARNING once each),
       the rest of the capture skips them; replay bitwise; a second capture skips them all
  A.8  a Python error inside the capture with prefetches / a warm pending (injected in layer 1's mHC; in layer 4's MoE
       between o_proj's fork and the join): the capture raises THAT error (no unjoined-stream error), no pending fork
       is left behind, and a new capture + replay is bitwise
  B.7  (before) a Python error inside such a capture with same-segment forks pending (t2 / t3 + warm) ends as
       "capturing stream has unjoined work" and leaves the forks pending (stale)
  B.8  (before) with GLM53_PREFILL_QUICKWINS off (vLLM's stock forwards, production's state since 2026-09-29): roof
       (also with the stock KDA _forward) and t4 fail the same way, the warm alone captures
  A.9  the same with quickwins off: eager bitwise; PIECEWISE M 1 / 8 / 64 and FULL M 8 / 64 with both on capture and
       replay bitwise (t1 / t4 skipped only in PIECEWISE); warm / t1 / t4 alone; the stock KDA _forward
Timing (R16_BK_TIMING=1, case after:timing; =noqw: quickwins off; nodeC, indicative): per-forward replay time of FULL
and breakable PIECEWISE graphs per configuration (off, off2, roof, t1..t4 alone, warm, both, both without t2, roof /
both without t1 and t4 = what PIECEWISE keeps; PIECEWISE also roof with t1 / t4 forked and joined at the break),
docs/DEC_FP8ROOF.md section 12.
Run: GPU_RUN_BIND=<overlay exl3.py> tests/r16/gpu.sh python3 -u tests/r16/test_r16_breakable.py
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
MARK = "CASE_RESULT "


# ------------------------------------------------------------------------------------------------------------------
# child side
def _tb_facts(tb: str) -> dict:
    return {"unjoined": "unjoined work" in tb, "add_eager": "in add_eager" in tb, "end_segment": "in _end_segment" in tb,
            "capture_end": "capture_end" in tb, "kda_forward": "glm5next/nvidia/kda.py" in tb,
            "bk_wrapper": "breakable_cudagraph.py" in tb and "in wrapper" in tb,
            "first": next((ln for ln in tb.splitlines() if _EXC.match(ln)), "")[:300],
            "last": next((ln for ln in reversed(tb.splitlines()) if _EXC.match(ln)), "")[:300]}


_EXC = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception)(: |$)")


def _snap(rig) -> dict:
    return {"counters": rig.counters(), "roof": dict(rig.R.COUNTERS), "mg": dict(rig.MG.COUNTERS),
            "pending": [str(k) for k in rig.R.STATE.pending],
            "warm_pending": {str(k): v for k, v in rig.MG.WARM.pending.items() if v}}


def _bitwise_pw(rig, inp, cfg, n=3, seed=100, wrapper=None) -> bool:
    import torch
    same = True
    before = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS))
    for s in range(n):
        inp["inputs_embeds"].copy_(rig.inputs(inp["inputs_embeds"].shape[0], seed=seed + s)["inputs_embeds"])
        y = rig.replay_pw(inp, wrapper).clone()
        frozen = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS)) == before
        rig.setcfg(False, False)
        ye = rig.eager(inp)
        rig.setcfg(*cfg)
        torch.cuda.synchronize()
        same &= bool(torch.equal(y, ye)) and frozen
        before = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS))
    return same


def _bitwise_full(rig, gr, out, inp, cfg, n=3, seed=200) -> bool:
    import torch
    same = True
    before = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS))
    for s in range(n):
        inp["inputs_embeds"].copy_(rig.inputs(inp["inputs_embeds"].shape[0], seed=seed + s)["inputs_embeds"])
        gr.replay()
        y = out.clone()
        frozen = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS)) == before
        rig.setcfg(False, False)
        ye = rig.eager(inp)
        rig.setcfg(*cfg)
        torch.cuda.synchronize()
        same &= bool(torch.equal(y, ye)) and frozen
        before = (dict(rig.R.COUNTERS), dict(rig.MG.COUNTERS))
    return same


def _pw_case(rig, cfg, M, bitwise=True) -> dict:
    """V2 runner order: eager warm-up (counted apart), then the breakable capture (counted), then replays."""
    rig.setcfg(*cfg)
    rig.clear()
    inp = rig.inputs(M)
    warm = {}
    wrapper = rig.new_wrapper()
    rig.last = (wrapper, inp)

    def after_warmup():
        warm.update(_snap(rig))
        rig.clear()
    res = {"cfg": [cfg[0], cfg[1], sorted(cfg[2]) if cfg[2] is not None else "all"], "M": M}
    try:
        rig.capture_pw(inp, wrapper=wrapper, after_warmup=after_warmup)
        import torch
        torch.cuda.synchronize()
        res.update(ok=True, **_snap(rig), warmup=warm)
        entry = wrapper.entries.get(next(iter(wrapper.entries)))
        res["segments"] = {"graphs": entry.capture.num_graphs, "eager_breaks": entry.capture.num_eager_breaks}
        if bitwise:
            res["bitwise"] = _bitwise_pw(rig, inp, cfg, wrapper=wrapper)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        res.update(ok=False, tb=_tb_facts(tb), **_snap(rig), warmup=warm)
        print(tb, flush=True)
    return res


def _full_case(rig, cfg, M, bitwise=True) -> dict:
    rig.setcfg(*cfg)
    rig.clear()
    inp = rig.inputs(M)
    warm = {}

    def after_warmup():
        warm.update(_snap(rig))
        rig.clear()
    res = {"cfg": [cfg[0], cfg[1], sorted(cfg[2]) if cfg[2] is not None else "all"], "M": M}
    try:
        gr, out = rig.capture_full(inp, after_warmup=after_warmup)
        import torch
        torch.cuda.synchronize()
        res.update(ok=True, **_snap(rig), warmup=warm)
        if bitwise:
            res["bitwise"] = _bitwise_full(rig, gr, out, inp, cfg)
        del gr
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        res.update(ok=False, tb=_tb_facts(tb), **_snap(rig), warmup=warm)
        print(tb, flush=True)
    return res


def child(case: str) -> dict:
    import torch
    import harness as H
    import bk_rig
    H.gpu_guard(8.0)
    full_case = case
    noqw = case.endswith(":noqw")            # production's state since 2026-09-29: GLM53_PREFILL_QUICKWINS unset
    if noqw:
        case = case[: -len(":noqw")]
    rig = bk_rig.Rig(kda_core="real" if case.endswith(":realkda") else "standin", quickwins=not noqw)
    from vllm.models.glm5next.nvidia import kda as kda_mod
    origin = {"Glm5NextModel.forward": rig.gm.Glm5NextModel.forward, "Glm5NextDecoderLayer.forward":
              rig.gm.Glm5NextDecoderLayer.forward, "Glm5NextLinearAttention._forward":
              kda_mod.Glm5NextLinearAttention._forward}
    out = {"case": full_case, "decorated": rig.decorated, "wired": rig.wout["wired"], "rrep_pf": rig.rrep.get("pf"),
           "mrep_warm": rig.mrep.get("warm"), "qrep": rig.qrep, "quickwins": rig.quickwins,
           "qw_recompiled": {k: bool(getattr(f, "_glm53_qw", False)) for k, f in origin.items()}}
    R = rig.R
    ALL = None
    if case.split(":", 1)[1] in ("inject-roof", "inject-warm"):
        # triggers t2 / t3 (+ warm): the ones whose fork and join share a segment, so the kit's code captures too
        T23 = (True, True, ("t2", "t3"))
        where = (1, "mhc") if case.endswith("roof") else (4, "moe")
        # a graph kept alive in vLLM's global pool first, as in production (a capture that fails discards its own
        # segments; with no other graph in the pool the next capture_begin would trip PyTorch's allocator assert)
        out["first"] = _pw_case(rig, T23, 5)
        first_w, first_inp = rig.last
        rig.setcfg(*T23)
        rig.clear()
        inp = rig.inputs(8)

        def arm():
            out["pending_at_arm"] = _snap(rig)
            rig.inject = where
        w = rig.new_wrapper()
        try:
            rig.capture_pw(inp, wrapper=w, after_warmup=arm)
            out["raised"] = None
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            out["raised"] = _tb_facts(tb)
            out["raised"]["injected"] = "injected failure" in tb
            print(tb, flush=True)
        rig.inject = None
        out["after_fail"] = _snap(rig)
        if case.startswith("after:"):
            rig.clear()
            out["retry"] = _pw_case(rig, (True, True, ALL), 8)
            out["first_replay"] = _bitwise_pw(rig, first_inp, T23, seed=300, wrapper=first_w)
        out["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
        return out
    if case == "after:stock":
        # quickwins off (vLLM's stock forwards): the same break positions, the same fix
        M = 5
        x = rig.inputs(M)
        outs, cnts = {}, {}
        for nm, cfg in {"off": (False, False, ALL), "both": (True, True, ALL)}.items():
            rig.setcfg(*cfg)
            rig.clear()
            outs[nm] = rig.eager(x).clone()
            torch.cuda.synchronize()
            cnts[nm] = _snap(rig)
        out["eager"] = {"counts": cnts, "bitwise": torch.equal(outs["both"], outs["off"])}
        out["pw"] = {M: _pw_case(rig, (True, True, ALL), M) for M in (1, 8, 64)}
        out["full"] = {M: _full_case(rig, (True, True, ALL), M) for M in (8, 64)}
        out["pw_single"] = {nm: _pw_case(rig, cfg, 8) for nm, cfg in
                            {"warm": (False, True, ALL), "t1": (True, False, ("t1",)),
                             "t4": (True, False, ("t4",))}.items()}
        out["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
        return out
    if case.startswith("before:"):
        name = case.split(":", 1)[1]
        if name in ("roof", "roof:realkda"):
            out["pw"] = _pw_case(rig, (True, False, ALL), 8, bitwise=False)
        elif name == "warm":
            out["pw"] = _pw_case(rig, (False, True, ALL), 8)
        elif name == "both":
            out["pw"] = _pw_case(rig, (True, True, ALL), 8, bitwise=False)
        elif name in ("t1", "t2", "t3", "t4"):
            out["pw"] = _pw_case(rig, (True, False, (name,)), 8)
        elif name == "full":
            out["full"] = _full_case(rig, (True, True, ALL), 8)
        return out
    if case == "after:main":
        BK = rig.BK.BreakableCUDAGraphCapture
        chain, fn = [], BK.add_eager
        while fn is not None:
            chain.append([k for k in ("_glm53_roof_seg", "_glm53_warm_seg") if getattr(fn, k, False)])
            fn = getattr(fn, "_glm53_seg_orig", None)
        out["guard"] = {"roof": R.STATE.bk_guard, "roof_skip": sorted(R.STATE.bk_skip), "warm": rig.MG.WARM.bk_guard,
                        "chain": chain, "rrep": {k: rig.rrep.get(k) for k in ("bk_guard", "bk_skip")}}
        # A.2 eager
        M = 5
        x = rig.inputs(M)
        outs, cnts = {}, {}
        for nm, cfg in {"off": (False, False, ALL), "roof": (True, False, ALL), "warm": (False, True, ALL),
                        "both": (True, True, ALL)}.items():
            rig.setcfg(*cfg)
            rig.clear()
            outs[nm] = rig.eager(x).clone()
            torch.cuda.synchronize()
            cnts[nm] = _snap(rig)
        out["eager"] = {"counts": cnts, "bitwise": all(torch.equal(outs[n], outs["off"]) for n in outs)}
        # A.3 PIECEWISE, both on, every size
        out["pw"] = {M: _pw_case(rig, (True, True, ALL), M) for M in (1, 5, 8, 16, 32, 64)}
        # A.4 FULL after the PIECEWISE captures
        out["full"] = {M: _full_case(rig, (True, True, ALL), M) for M in (1, 5, 8, 16, 64)}
        rig.setcfg(True, True)
        rig.clear()
        rig.eager(rig.inputs(65))
        torch.cuda.synchronize()
        out["m65"] = _snap(rig)
        # A.5 single features / triggers
        out["pw_single"] = {nm: _pw_case(rig, cfg, 8) for nm, cfg in
                            {"roof": (True, False, ALL), "warm": (False, True, ALL), "t1": (True, False, ("t1",)),
                             "t2": (True, False, ("t2",)), "t3": (True, False, ("t3",)),
                             "t4": (True, False, ("t4",))}.items()}
        out["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
        H.report_peak(8.0)
        return out
    if case == "after:main:realkda":
        out["pw"] = _pw_case(rig, (True, True, ALL), 8, bitwise=False)
        out["full"] = _full_case(rig, (True, True, ALL), 8, bitwise=False)
        if out["pw"]["ok"]:
            inp = rig.inputs(8)
            rig.setcfg(True, True)
            w = rig.new_wrapper()
            rig.capture_pw(inp, wrapper=w)
            for _ in range(3):
                rig.replay_pw(inp, w)
            torch.cuda.synchronize()
            out["replays_ran"] = True
        out["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
        return out
    if case == "after:learn":
        R.STATE.bk_skip = set()
        out["pw1"] = _pw_case(rig, (True, True, ALL), 8)
        out["skip_after"] = sorted(R.STATE.bk_skip)
        out["pw2"] = _pw_case(rig, (True, True, ALL), 5)
        out["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
        return out
    if case == "after:timing":
        return timing(rig)
    raise SystemExit(f"unknown case {case}")


def timing(rig) -> dict:
    """Per-forward replay time, FULL vs PIECEWISE (breakable), per configuration; latency-bound windows are clock spins at
    the R15 trace's medians (tests/roof/sim_step.py's 'spin' mode): KDA conv 7 + recurrent 24 us + 8 MiB state writes,
    indexer 40 + MLA attention 110 us + 16 MiB reads, AR 25 + mHC 6 us around each mHC, topk 3 + rot_in 24 + epilogue 8
    us in the MoE; routed experts topk 8 (real EXL3, 32 experts). Interleaved rounds, medians, paired ratios vs off."""
    import torch
    rig.topk = 8
    rig.spin_on = True
    cyc = rig.calibrate_spin()
    R = rig.R
    cfgs = {"off": (False, False, None), "off2": (False, False, None), "roof": (True, False, None),
            "t1": (True, False, ("t1",)), "t2": (True, False, ("t2",)), "t3": (True, False, ("t3",)),
            "t4": (True, False, ("t4",)), "warm": (False, True, None), "both": (True, True, None),
            "both_no_t2": (True, True, ("t0", "t1", "t3", "t4", "t5")),
            # what a breakable PIECEWISE graph keeps of fp8roof (t1 / t4 not forked there), in both regimes: FULL
            # roof vs roof_t23 is the t1 + t4 gain that PIECEWISE graphs give up
            "roof_t23": (True, False, ("t0", "t2", "t3", "t5")), "both_t23": (True, True, ("t0", "t2", "t3", "t5"))}
    rounds = int(os.environ.get("R16_BK_ROUNDS", "15"))
    reps = 10
    res = {"cyc_per_us": cyc, "rounds": rounds}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for M in [int(v) for v in os.environ.get("R16_BK_TIMING_M", "5,32").split(",")]:
        graphs = {}
        for mode in ("full", "pw"):
            for nm, cfg in cfgs.items():
                rig.setcfg(*cfg)
                R.STATE.bk_skip = set(R.BREAK_CROSS) & set(R.STATE.triggers)
                R.STATE.bk_learn = True
                rig.clear()
                inp = rig.inputs(M)
                if mode == "full":
                    gr, _o = rig.capture_full(inp)
                    graphs[(mode, nm)] = (lambda gr=gr: gr.replay())
                else:
                    w = rig.new_wrapper()
                    rig.capture_pw(inp, wrapper=w)
                    graphs[(mode, nm)] = (lambda inp=inp, w=w: rig.replay_pw(inp, w))
                res.setdefault("forks", {})[f"{mode}:{nm}:M{M}"] = rig.counters()
            if mode == "pw":                   # the alternative policy: fork t1 / t4 anyway, join at the break
                nm = "roof_joinbreak"
                rig.setcfg(True, False)
                R.STATE.bk_skip = set()
                R.STATE.bk_learn = False
                rig.clear()
                inp = rig.inputs(M)
                w = rig.new_wrapper()
                rig.capture_pw(inp, wrapper=w)
                graphs[(mode, nm)] = (lambda inp=inp, w=w: rig.replay_pw(inp, w))
                res.setdefault("forks", {})[f"{mode}:{nm}:M{M}"] = dict(rig.counters(),
                                                                          joined_at_break=R.COUNTERS.get("joined_at_break", 0))
                R.STATE.bk_learn = True
        rig.setcfg(True, True)
        keys = list(graphs)
        ts = {k: [] for k in keys}
        for r in range(rounds):
            order = keys[r % len(keys):] + keys[:r % len(keys)]
            if r % 2:
                order = order[::-1]
            for k in order:
                graphs[k]()
                torch.cuda.synchronize()
                ev[0].record()
                for _ in range(reps):
                    graphs[k]()
                ev[1].record()
                torch.cuda.synchronize()
                ts[k].append(ev[0].elapsed_time(ev[1]) * 1000 / reps)
        for mode in ("full", "pw"):
            base = ts[(mode, "off")]
            for k in keys:
                if k[0] != mode:
                    continue
                rat = sorted(a / b for a, b in zip(ts[k], base))
                row = {"us": statistics.median(ts[k]), "ratio": statistics.median(rat), "p10": rat[len(rat) // 10],
                       "p90": rat[9 * len(rat) // 10], "saving_us": statistics.median(base) - statistics.median(ts[k])}
                res.setdefault("t", {})[f"{mode}:{k[1]}:M{M}"] = row
                print(f"T M={M} {mode:4s} {k[1]:14s} {row['us']:9.1f} us/forward  saving {row['saving_us']:7.1f} us  "
                      f"ratio vs off {row['ratio']:.4f} [p10 {row['p10']:.4f}, p90 {row['p90']:.4f}]", flush=True)
        del graphs
        torch.cuda.empty_cache()
    res["warnings"] = [m for lv, nm, m in rig.cap.records if lv >= 30]
    return res


# ------------------------------------------------------------------------------------------------------------------
# parent side
def run(case: str) -> dict:
    t = int(os.environ.get("R16_BK_CASE_TIMEOUT", "1800"))
    p = subprocess.run([sys.executable, "-u", __file__, "--case", case], capture_output=True, text=True, timeout=t)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(MARK)]
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-25:])
    if not lines:
        print(f"--- case {case}: no result (rc {p.returncode})\n{tail}", flush=True)
        return {"case": case, "crashed": True, "rc": p.returncode}
    r = json.loads(lines[-1][len(MARK):])
    r["rc"] = p.returncode
    if r.get("crashed"):
        print(f"--- case {case} crashed:\n" + "\n".join(p.stdout.splitlines()[-40:]), flush=True)
    elif os.environ.get("R16_BK_VERBOSE"):
        print(tail, flush=True)
    return r


def fork_counts(c: dict) -> dict:
    return {k: c.get(k, 0) for k in ("t0", "t1", "t2", "t3", "t4", "t5", "warm")}


def main_before(ck) -> None:
    exp = {"roof": False, "roof:realkda": False, "warm": True, "both": False, "t1": False, "t2": True, "t3": True,
           "t4": False, "roof:noqw": False, "roof:realkda:noqw": False, "warm:noqw": True, "t4:noqw": False}
    for name, ok in exp.items():
        r = run("before:" + name)
        if name.endswith(":noqw"):
            ck(r.get("quickwins") is False and not any(r.get("qw_recompiled", {}).values()),
               f"B {name}: quickwins off, vLLM's stock forwards: {r.get('qw_recompiled')}")
        elif not r.get("crashed"):
            ck(r.get("qw_recompiled", {}).get("Glm5NextDecoderLayer.forward"), f"B {name}: quickwins recompiled forwards")
        pw = r.get("pw", {})
        ck(not r.get("crashed"), f"B {name}: child crashed")
        facts = pw.get("tb", {})
        if ok:
            ck(pw.get("ok") and pw.get("bitwise"), f"B {name}: expected capture ok + bitwise, got {pw}")
            print(f"B {name:13s}: PIECEWISE capture ok (segments {pw.get('segments')}), forks {fork_counts(pw.get('counters', {}))}"
                  f", replay == eager bitwise {pw.get('bitwise')}", flush=True)
        else:
            base = name.replace(":noqw", "")
            ck(pw.get("ok") is False and facts.get("unjoined") and facts.get("add_eager") and facts.get("end_segment")
               and facts.get("capture_end") and facts.get("kda_forward" if base in ("roof", "roof:realkda", "both", "t1")
                                                          else "bk_wrapper"),
               f"B {name}: expected the unjoined-stream failure at an eager break, got {pw}")
            print(f"B {name:13s}: PIECEWISE capture FAILED as production did: {facts.get('first')!r}; traceback via "
                  f"kda.py forward {facts.get('kda_forward')}, breakable wrapper {facts.get('bk_wrapper')}, add_eager "
                  f"{facts.get('add_eager')} -> _end_segment {facts.get('end_segment')} -> capture_end; pending "
                  f"fork(s) at the failure {pw.get('pending')}", flush=True)
        if name in ("roof", "roof:noqw", "roof:realkda:noqw"):
            ck(pw.get("pending") == ["('kda_o', 0)"], f"B {name}: the pending fork is t1 of layer 0: {pw.get('pending')}")
        if name == "t4:noqw":
            ck(pw.get("pending") == ["('mla_o', 3)"], f"B {name}: the pending fork is t4 of layer 3: {pw.get('pending')}")
            ck(all(r.get("decorated", {}).values()), f"B production break points decorated: {r.get('decorated')}")
    for case, stale, wstale in (("before:inject-roof", ["('kda_in', 1)"], {}),
                                ("before:inject-warm", ["('sh_gu', 4)"], {"0": "cap"})):
        r = run(case)
        rs, af = r.get("raised") or {}, r.get("after_fail", {})
        ck(r.get("first", {}).get("ok") and rs.get("injected") and rs.get("unjoined") and af.get("pending") == stale
           and af.get("warm_pending") == wstale,
           f"B {case}: expected the injected error masked by an unjoined-stream error and a stale fork: {r}")
        print(f"B {case}: a Python error inside the capture (t2/t3 + warm: same-segment triggers only) ends as "
              f"{rs.get('last')!r} after the unjoined-stream error {rs.get('unjoined')} (the injected {rs.get('first')!r} "
              f"is only the first link of the chain); stale forks left behind: roof {af.get('pending')} warm "
              f"{af.get('warm_pending')}", flush=True)
    r = run("before:full")
    f = r.get("full", {})
    ck(f.get("ok") and f.get("bitwise"), f"B full: FULL capture ok + bitwise: {f}")
    print(f"B full         : FULL capture ok, forks {fork_counts(f.get('counters', {}))}, replay bitwise "
          f"{f.get('bitwise')}", flush=True)


def main_after(ck) -> None:
    import bk_rig
    want = {k: v for k, v in bk_rig.WANT.items()}
    want_pw = dict(want, t1=0, t4=0)
    r = run("after:main")
    ck(not r.get("crashed"), "A main crashed")
    if r.get("crashed"):
        return
    g = r["guard"]
    ck(all(r["decorated"].values()), f"A.1 production break points decorated: {r['decorated']}")
    ck(g["roof"] == "hooked" and g["warm"] == "hooked" and g["roof_skip"] == ["t1", "t4"] and
       sorted(sum(g["chain"], [])) == ["_glm53_roof_seg", "_glm53_warm_seg"], f"A.1 guards {g}")
    ck(r["wired"] == 5 and r["rrep_pf"] and r["mrep_warm"], f"A.1 installs: wired {r['wired']}")
    print(f"A.1 installs: break points decorated {r['decorated']}; roof guard {g['roof']} skip {g['roof_skip']}, warm "
          f"guard {g['warm']}, add_eager chain {g['chain']}; warm wired {r['wired']}", flush=True)
    e = r["eager"]
    for nm, (ro, wa) in {"off": (0, 0), "roof": (1, 0), "warm": (0, 1), "both": (1, 1)}.items():
        exp = {k: (v if (k == "warm" and wa) or (k != "warm" and ro) else 0) for k, v in want.items()}
        ck(fork_counts(e["counts"][nm]["counters"]) == exp and not e["counts"][nm]["pending"],
           f"A.2 eager {nm}: {e['counts'][nm]}")
    ck(e["bitwise"], "A.2 eager outputs differ")
    print(f"A.2 eager M=5: forks both {fork_counts(e['counts']['both']['counters'])}, off/roof/warm/both bitwise "
          f"{e['bitwise']}", flush=True)
    for M, p in r["pw"].items():
        c = fork_counts(p.get("counters", {}))
        rf = p.get("roof", {})
        ok = (p.get("ok") and c == want_pw and rf.get("bk_skipped_t1") == 6 and rf.get("bk_skipped_t4") == 2 and
              not any(k.startswith(("joined_at", "dropped_at")) for k in rf) and
              not any(k.startswith("warm_joined_at") for k in p.get("mg", {})) and not p.get("pending") and
              not p.get("warm_pending") and p.get("bitwise"))
        ck(ok, f"A.3 PIECEWISE M={M}: {p}")
        print(f"A.3 PIECEWISE M={M:2}: capture ok {p.get('ok')} (segments {p.get('segments')}), forks {c}, skipped t1 "
              f"{rf.get('bk_skipped_t1')} t4 {rf.get('bk_skipped_t4')}, replay == eager bitwise x3 {p.get('bitwise')}",
              flush=True)
    for M, p in r["full"].items():
        c = fork_counts(p.get("counters", {}))
        ok = p.get("ok") and c == want and not any(k.startswith("bk_") for k in p.get("roof", {})) and p.get("bitwise")
        ck(ok, f"A.4 FULL M={M}: {p}")
        print(f"A.4 FULL      M={M:2}: capture ok {p.get('ok')}, forks {c}, replay bitwise {p.get('bitwise')}",
              flush=True)
    ck(sum(fork_counts(r["m65"]["counters"]).values()) == 0, f"A.4 M=65 forked: {r['m65']}")
    for nm, p in r["pw_single"].items():
        ck(p.get("ok") and p.get("bitwise") and not p.get("pending"), f"A.5 PIECEWISE {nm}: {p}")
        print(f"A.5 PIECEWISE {nm:4s} M=8: capture ok {p.get('ok')}, forks {fork_counts(p.get('counters', {}))}, "
              f"replay bitwise {p.get('bitwise')}", flush=True)
    ck(not r["warnings"], f"A no WARNING: {r['warnings'][:4]}")
    r = run("after:main:realkda")
    ck(not r.get("crashed") and r["pw"]["ok"] and r["full"]["ok"] and r.get("replays_ran") and not r["warnings"],
       f"A.6 production KDA _forward: {r}")
    print(f"A.6 production KDA _forward: PIECEWISE ok {r.get('pw', {}).get('ok')} (segments "
          f"{r.get('pw', {}).get('segments')}, forks {fork_counts(r.get('pw', {}).get('counters', {}))}), FULL ok "
          f"{r.get('full', {}).get('ok')}, replays ran {r.get('replays_ran')}", flush=True)
    r = run("after:stock:noqw")
    ck(not r.get("crashed") and r.get("quickwins") is False and not any(r.get("qw_recompiled", {}).values()),
       f"A.9 quickwins off: stock forwards {r.get('qw_recompiled')}")
    if not r.get("crashed"):
        e = r["eager"]
        ck(e["bitwise"] and fork_counts(e["counts"]["both"]["counters"]) == want, f"A.9 eager: {e}")
        for M, p in r["pw"].items():
            c = fork_counts(p.get("counters", {}))
            rf = p.get("roof", {})
            ck(p.get("ok") and c == want_pw and rf.get("bk_skipped_t1") == 6 and rf.get("bk_skipped_t4") == 2 and
               not any(k.startswith(("joined_at", "dropped_at")) for k in rf) and not p.get("pending") and
               not p.get("warm_pending") and p.get("bitwise"), f"A.9 PIECEWISE M={M}: {p}")
            print(f"A.9 quickwins off, PIECEWISE M={M:2}: capture ok {p.get('ok')} (segments {p.get('segments')}), forks "
                  f"{c}, replay == eager bitwise x3 {p.get('bitwise')}", flush=True)
        for M, p in r["full"].items():
            c = fork_counts(p.get("counters", {}))
            ck(p.get("ok") and c == want and p.get("bitwise"), f"A.9 FULL M={M}: {p}")
            print(f"A.9 quickwins off, FULL      M={M:2}: capture ok {p.get('ok')}, forks {c}, replay bitwise "
                  f"{p.get('bitwise')}", flush=True)
        for nm, p in r["pw_single"].items():
            ck(p.get("ok") and p.get("bitwise") and not p.get("pending"), f"A.9 PIECEWISE {nm}: {p}")
            print(f"A.9 quickwins off, PIECEWISE {nm:4s} M=8: capture ok {p.get('ok')}, forks "
                  f"{fork_counts(p.get('counters', {}))}, replay bitwise {p.get('bitwise')}", flush=True)
        ck(not r["warnings"], f"A.9 no WARNING: {r['warnings'][:4]}")
    r = run("after:main:realkda:noqw")
    ck(not r.get("crashed") and r["pw"]["ok"] and r["full"]["ok"] and r.get("replays_ran") and not r["warnings"]
       and not any(r.get("qw_recompiled", {}).values()), f"A.9 production's stock KDA _forward: {r}")
    print(f"A.9 quickwins off, vLLM's stock KDA _forward (kda_core=real): PIECEWISE ok {r.get('pw', {}).get('ok')}, "
          f"FULL ok {r.get('full', {}).get('ok')}, replays ran {r.get('replays_ran')}", flush=True)
    r = run("after:learn")
    p1, p2 = r.get("pw1", {}), r.get("pw2", {})
    rf1 = p1.get("roof", {})
    ok = (p1.get("ok") and p1.get("bitwise") and rf1.get("joined_at_break_t1") == 1 and
          rf1.get("joined_at_break_t4") == 1 and rf1.get("bk_skipped_t1") == 5 and rf1.get("bk_skipped_t4") == 1 and
          r.get("skip_after") == ["t1", "t4"] and len([w for w in r.get("warnings", []) if "breakable CUDA-graph break" in w]) == 2
          and p2.get("ok") and p2.get("roof", {}).get("bk_skipped_t1") == 6 and "joined_at_break" not in p2.get("roof", {}))
    ck(ok, f"A.7 guard: {r}")
    print(f"A.7 guard (static skip emptied): capture ok {p1.get('ok')}, joined at the break t1 "
          f"{rf1.get('joined_at_break_t1')} t4 {rf1.get('joined_at_break_t4')}, then skipped t1 {rf1.get('bk_skipped_t1')} "
          f"t4 {rf1.get('bk_skipped_t4')}; learned {r.get('skip_after')}; replay bitwise {p1.get('bitwise')}; second "
          f"capture skips all {p2.get('roof', {}).get('bk_skipped_t1')}/{p2.get('roof', {}).get('bk_skipped_t4')}; "
          f"WARNINGs {len(r.get('warnings', []))}", flush=True)
    for case in ("after:inject-roof", "after:inject-warm"):
        r = run(case)
        rs = r.get("raised") or {}
        af = r.get("after_fail", {})
        rt = r.get("retry", {})
        ok = (r.get("first", {}).get("ok") and rs.get("injected") and not rs.get("unjoined") and not af.get("pending")
              and not af.get("warm_pending") and rt.get("ok") and rt.get("bitwise") and r.get("first_replay")
              and not r.get("warnings"))
        ck(ok, f"A.8 {case}: {r}")
        print(f"A.8 {case}: capture raised the injected error {rs.get('injected')} (unjoined-stream error "
              f"{rs.get('unjoined')}); joined at the capture end roof {af.get('roof', {}).get('joined_at_capture_end', 0)}"
              f" warm {af.get('mg', {}).get('warm_joined_at_capture_end', 0)}; pending after: roof {af.get('pending')} "
              f"warm {af.get('warm_pending')}; new capture ok {rt.get('ok')} bitwise {rt.get('bitwise')}; the graph "
              f"captured before still replays bitwise {r.get('first_replay')}", flush=True)
    if os.environ.get("R16_BK_TIMING"):
        r = run("after:timing" + (":noqw" if os.environ["R16_BK_TIMING"] == "noqw" else ""))
        ck(not r.get("crashed") and not r.get("warnings"), f"T timing: {r.get('warnings')}")
        for k, v in sorted(r.get("forks", {}).items()):
            print(f"T forks {k}: {v}", flush=True)
        for k, v in r.get("t", {}).items():
            print(f"T {k:26s} {v['us']:9.1f} us  saving {v['saving_us']:7.1f} us  ratio {v['ratio']:.4f} "
                  f"[{v['p10']:.4f}, {v['p90']:.4f}]", flush=True)


def main():
    import harness as H
    ck = H.Checks()
    mode = os.environ.get("R16_BK_EXPECT", "after")
    import fp8_roof
    import glm53_moeglue
    print(f"R16_BK_EXPECT={mode}: fp8_roof {fp8_roof.__file__} (breakable guard in module: "
          f"{hasattr(fp8_roof, 'BREAK_CROSS')}), glm53_moeglue {glm53_moeglue.__file__} (guard: "
          f"{hasattr(glm53_moeglue, '_bk_hook')})", flush=True)
    if mode == "before":
        main_before(ck)
    else:
        main_after(ck)
    ck.summary()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        try:
            res = child(sys.argv[2])
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            res = {"case": sys.argv[2], "crashed": True}
        print(MARK + json.dumps(res, default=str), flush=True)
        os._exit(0)
    import harness as H
    H.run_main(main)
