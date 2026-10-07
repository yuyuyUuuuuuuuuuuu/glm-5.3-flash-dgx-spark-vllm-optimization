#!/usr/bin/env python3
"""opt-w8a8layers: GLM53_DENSE_W8A8_SKIP_LAYERS (fp8_w8a8.py) - parsing, selection, served-layer counts on production's
45-layer layout, load-time behaviour and the per-call path; unset must be bit-for-bit the r16z2rev behaviour.

Runs on the host (a minimal torch stub is injected when torch is absent: the tested code never touches a tensor) and in
the production image (tests/gpu_run.sh python3 tests/w8a8layers/test_skip_layers.py, real torch). Also checks that the
repo-root fp8_w8a8.py and overlay/fp8_w8a8.py are the same bytes.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
try:
    import torch  # noqa: F401
    STUB = False
except ImportError:                     # host: the filter logic is pure Python
    STUB = True
    t = types.ModuleType("torch")
    t.Tensor = object
    t.compiler = types.SimpleNamespace(is_compiling=lambda: False)
    t.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False, is_available=lambda: False)
    sys.modules["torch"] = t

spec = importlib.util.spec_from_file_location("fp8_w8a8_under_test", REPO / "overlay" / "fp8_w8a8.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

OK = FAIL = 0


def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


class M:                                 # a Glm53DenseFp8Method stand-in (group, prefix, ready)
    def __init__(self, group, prefix):
        self.group, self.prefix, self.ready = group, prefix, True


# ---------------------------------------------------------------- production's 45-layer dense-FP8 layout
KDA = ("in_proj_qkvbfg_a", "o_proj", "f_b_proj", "g_b_proj")
MLA = ("fused_qkv_a_proj", "q_b_proj", "o_proj")
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}


def prod_layout():
    out = []
    for L in range(45):
        if L in MLA_LAYERS:
            out += [M("mla", f"model.layers.{L}.self_attn.{p}") for p in MLA]
        else:
            out += [M("kda", f"model.layers.{L}.self_attn.{p}") for p in KDA]
        if L < 3:
            out += [M("dense", f"model.layers.{L}.mlp.{p}") for p in ("gate_up_proj", "down_proj")]
        else:
            out += [M("shared", f"model.layers.{L}.mlp.shared_experts.{p}") for p in ("gate_up_proj", "down_proj")]
    return out


def orig_selected(method):              # r16z2rev's selected(), verbatim
    if W.CFG.only is None:
        return True
    pre = getattr(method, "prefix", "") or ""
    return f"{method.group}.{pre.rsplit('.', 1)[-1]}" in W.CFG.only


def setcfg(only=None, skip=""):
    W.CFG.only = frozenset(only.split(",")) if only else None
    W.CFG.skip = W.parse_skip(skip)


def count(layout):
    return sum(1 for m in layout if W.selected(m))


def main():
    # -- 0. the two copies of the module are the same bytes
    check("root == overlay", (REPO / "fp8_w8a8.py").read_bytes() == (REPO / "overlay" / "fp8_w8a8.py").read_bytes())

    # -- 1. parsing
    good = {"": None, "   ": None, "0-2": ((0, 2, None),), "5": ((5, 5, None),),
            "0-2,3:mla.o_proj": ((0, 2, None), (3, 3, "mla.o_proj")),
            " 7-9:shared , 1:kda.f_b_proj ": ((7, 9, "shared"), (1, 1, "kda.f_b_proj")),
            "0-44:kda.g_b_proj": ((0, 44, "kda.g_b_proj"),), "4095": ((4095, 4095, None),),
            "0,,1": ((0, 0, None), (1, 1, None))}
    for raw, want in good.items():
        try:
            got = W.parse_skip(raw)
        except ValueError as e:
            got = e
        check(f"parse ok {raw!r}", got == want, f"got {got!r}")
    for p in sorted(W.PROJ_NAMES) + sorted(W.SKIP_EXTRA_NAMES) + sorted(W.SKIP_GROUPS):
        check(f"parse name {p}", W.parse_skip(f"3:{p}") == ((3, 3, p),))
    bad = ["a", "3-1", "0-4096", "4096", "1:foo", "1:", "1-2-3", "-1", ",", " , ", "1:mla.o", "1:attn",
           "1:kda.in_proj", "L1", "1.5", "1 2", "12345", "1:MLA.O_PROJ"]
    for raw in bad:
        try:
            W.parse_skip(raw)
            check(f"parse refuses {raw!r}", False, "accepted")
        except ValueError as e:
            check(f"parse refuses {raw!r} (message)", "item" in str(e) or "no item" in str(e), str(e))
    check("layer_of", (W.layer_of("model.layers.12.self_attn.o_proj"), W.layer_of("layers.0.mlp.down_proj"),
                       W.layer_of("model.norm"), W.layer_of("model.mylayers.3.x"), W.layer_of("")) ==
          (12, 0, None, None, None))

    # -- 2. unset == r16z2rev's selected(), over every projection x every ONLY setting tried in the A/Bs
    layout = prod_layout()
    extra = [M("draft", "model.fc"), M("kda", "model.layers.0.self_attn.unknown"), M("mla", "noindex.o_proj")]
    onlys = [None, "kda.in_proj_qkvbfg_a,mla.o_proj", "kda.in_proj_qkvbfg_a", "mla.o_proj,shared.down_proj",
             ",".join(sorted(W.PROJ_NAMES))]
    for only in onlys:
        setcfg(only, "")
        check(f"unset == r16z2rev (ONLY={only})", all(W.selected(m) == orig_selected(m) for m in layout + extra))
        check(f"unset never skips (ONLY={only})", not any(W.skipped(m) for m in layout + extra))

    # -- 3. served-layer counts on production's layout (259 = the 'all' A/B arm, 45 = 'sub')
    want = {(None, ""): 259, ("kda.in_proj_qkvbfg_a,mla.o_proj", ""): 45,
            (None, "0-2"): 259 - 3 * 6, (None, "0-2:dense"): 259 - 6, (None, "0-2:kda"): 259 - 12,
            (None, "0-44:kda.f_b_proj,0-44:kda.g_b_proj"): 259 - 68,
            (None, "3"): 259 - 5, (None, "3:mla.o_proj"): 258, (None, "0-44"): 0, (None, "45-100"): 259,
            ("kda.in_proj_qkvbfg_a,mla.o_proj", "0-2"): 42, ("kda.in_proj_qkvbfg_a,mla.o_proj", "3-44:shared"): 45,
            (None, "0-5,3-4"): 259 - (3 * 6 + 5 + 6 + 6), (None, "0-2,0-2:dense"): 259 - 18}
    for (only, skip), n in want.items():
        setcfg(only, skip)
        got = count(layout)
        check(f"count ONLY={only} SKIP={skip!r} = {n}", got == n, f"got {got}")
    # exhaustive single-layer exclusion: exactly that layer's projections leave, nothing else changes
    for L in range(45):
        setcfg(None, str(L))
        changed = [m for m in layout if not W.selected(m)]
        check(f"skip {L} removes exactly layer {L}", changed and all(W.layer_of(m.prefix) == L for m in changed) and
              len(changed) == sum(1 for m in layout if W.layer_of(m.prefix) == L))
    # layer x projection: every (layer, name) item removes exactly one projection
    for L, nm in itertools.product((0, 2, 3, 4, 43, 44), sorted(W.PROJ_NAMES | W.SKIP_EXTRA_NAMES)):
        setcfg(None, f"{L}:{nm}")
        gone = [m for m in layout if not W.selected(m)]
        exists = any(W.layer_of(m.prefix) == L and f"{m.group}.{m.prefix.rsplit('.', 1)[-1]}" == nm for m in layout)
        check(f"skip {L}:{nm}", len(gone) == (1 if exists else 0), f"{len(gone)} gone")
    setcfg(None, "")

    # -- 4. load time: a skipped layer is not self-tested and is counted; unset = the r16z2rev calls exactly
    calls = []
    W.selftest = lambda layer, n, k, bias=None, label="": calls.append(label) or (True, "stub")
    W.STATE.enabled, W.STATE.groups = True, frozenset({"dense", "kda", "mla", "shared"})
    pwal = W._make_pwal(lambda self, layer: None)

    class Lay:
        glm53_fp8_n, glm53_fp8_k, bias = 4096, 4096, None

    for skip, n_test, n_skip in (("", 259, 0), ("0-2", 241, 18), ("0-44", 0, 259)):
        setcfg(None, skip)
        calls.clear()
        W.COUNTERS.clear()
        for m in layout:
            pwal(m, Lay())
        check(f"pwal SKIP={skip!r}: {n_test} self-tests", len(calls) == n_test, f"got {len(calls)}")
        check(f"pwal SKIP={skip!r}: skip counter {n_skip}", W.COUNTERS.get("skip_layers_at_load", 0) == n_skip,
              f"got {W.COUNTERS}")
    setcfg("kda.in_proj_qkvbfg_a,mla.o_proj", "")
    calls.clear()
    W.COUNTERS.clear()
    for m in layout:
        pwal(m, Lay())
    check("pwal ONLY=sub unset skip: 45 self-tests, no counter", len(calls) == 45 and not W.COUNTERS)

    # -- 5. per call: a skipped projection goes straight to production's apply, a kept one to try_w8a8
    seen = []
    W.try_w8a8 = lambda x, layer, bias, n, k, pre=None, hilo=0: seen.append("w8a8") or "Y"
    apply = W._make_apply(lambda self, layer, x, bias=None: seen.append("prod") or "P")
    setcfg(None, "0-2:dense,3")
    for pre, grp, want_path in (("model.layers.0.mlp.down_proj", "dense", "prod"),
                                ("model.layers.0.self_attn.o_proj", "kda", "w8a8"),
                                ("model.layers.3.self_attn.o_proj", "mla", "prod"),
                                ("model.layers.3.mlp.shared_experts.down_proj", "shared", "prod"),
                                ("model.layers.4.mlp.shared_experts.down_proj", "shared", "w8a8")):
        seen.clear()
        y = apply(M(grp, pre), Lay(), object())
        check(f"apply {pre}", seen == [want_path] and y == ("Y" if want_path == "w8a8" else "P"), f"{seen}")
    setcfg(None, "")
    seen.clear()
    apply(M("dense", "model.layers.0.mlp.down_proj"), Lay(), object())
    check("apply unset: served", seen == ["w8a8"])

    # -- 6. skip_text round trip (the install log line)
    setcfg(None, "0-2, 3:mla.o_proj ,9-9:shared")
    check("skip_text", W.skip_text() == "0-2,3:mla.o_proj,9:shared", W.skip_text())
    check("skip_text reparse", W.parse_skip(W.skip_text()) == W.CFG.skip)
    setcfg(None, "")
    check("skip_text unset", W.skip_text() == "none")

    print(f"test_skip_layers ({'host, torch stub' if STUB else 'image torch'}): {OK} ok, {FAIL} failed")
    print("ALL OK" if FAIL == 0 else "FAILED")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
