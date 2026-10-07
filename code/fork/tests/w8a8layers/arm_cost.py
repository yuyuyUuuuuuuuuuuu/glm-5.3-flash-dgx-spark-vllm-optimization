#!/usr/bin/env python3
"""Speed model of a W8A8 production arm: GLM53_DENSE_W8A8_ONLY / _SKIP_LAYERS evaluated with the module's own filter
(overlay/fp8_w8a8.py selected()) over production's 45-layer dense-FP8 layout, priced with the per-call savings at
M = 13,824 measured on nodeC (docs/DEPLOY_R16Z.md r16z2rev table; production minus w8a82 wall, prof_new_13824.log;
kda.f_b/g_b ~0.03 ms = the residual of the 'all' row). Time per 32k request (2 x 13,824 + 4,289) is extrapolated from
the production A/B (2026-10-02, MoE e4m3 on): base 2,453 tok/s, sub 2,620 (359 ms/chunk), all 2,752 (551 ms/chunk),
interpolating seconds-saved per ms/chunk linearly between the two measured arms.
Usage: arm_cost.py "<label>|<ONLY or ->|<SKIP or ->" ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_skip_layers as T  # noqa: E402  (loads overlay/fp8_w8a8.py, torch stub on the host)

W = T.W
PER_CALL = {"kda.in_proj_qkvbfg_a": 7.45, "kda.o_proj": 2.08, "kda.f_b_proj": 0.03, "kda.g_b_proj": 0.03,
            "mla.fused_qkv_a_proj": 0.55, "mla.q_b_proj": 2.31, "mla.o_proj": 9.61, "dense.gate_up_proj": 6.80,
            "dense.down_proj": 3.51, "shared.gate_up_proj": 0.675, "shared.down_proj": 0.675}
TOK = 2 * 13824 + 4289
T_BASE = TOK / 2453
SAVE = {359.0: T_BASE - TOK / 2620, 551.0: T_BASE - TOK / 2752}


def secs_saved(ms):
    (a, sa), (b, sb) = sorted(SAVE.items())
    k = sa / a + (ms - a) * ((sb / b - sa / a) / (b - a)) if ms > 0 else 0.0
    return ms * k


def arm(only, skip):
    T.setcfg(None if only in ("", "-") else only, "" if skip in ("", "-") else skip)
    ms, n = 0.0, 0
    for m in T.prod_layout():
        if W.selected(m):
            ms += PER_CALL[f"{m.group}.{m.prefix.rsplit('.', 1)[-1]}"]
            n += 1
    return ms, n


def main():
    print(f"{'arm':28s} {'GEMMs':>5s} {'ms/chunk':>8s} {'s/32k':>6s} {'tok/s':>6s}")
    for spec in sys.argv[1:]:
        lab, only, skip = (spec.split("|") + ["-", "-"])[:3]
        ms, n = arm(only, skip)
        s = secs_saved(ms)
        print(f"{lab:28s} {n:5d} {ms:8.1f} {T_BASE - s:6.2f} {TOK / (T_BASE - s):6.0f}")


if __name__ == "__main__":
    main()
