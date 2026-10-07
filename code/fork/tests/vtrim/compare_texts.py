#!/usr/bin/env python3
"""Temp-0 output check for a production A/B arm against base bench JSONs (bench_decode.py --out files).
Production's base is NOT deterministic at temp 0 (MoE atomics; prose gave 10 distinct texts in 10 runs), so "identical"
is judged distributionally: the common-prefix length (chars) of every arm text with every base text vs the same
statistic among the base texts themselves, plus exact-match counts. A decode-state bug (e.g. GLM53_DEC_KDA_LAZY on
2026-10-04: garbled Japanese, coding 4.13 -> 2.73 acc/step) shows as arm-vs-base prefixes far below base-vs-base and
as text anomalies (doubled CJK characters).
Usage: compare_texts.py <base_w.json> <arm_w.json> [more base/arm pairs ...]"""
import itertools, json, re, statistics as st, sys


def cp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


args = sys.argv[1:]
for bpath, apath in zip(args[0::2], args[1::2]):
    B = [r["text"] for r in json.load(open(bpath))["runs"]]
    A = [r["text"] for r in json.load(open(apath))["runs"]]
    bb = [cp(x, y) for x, y in itertools.combinations(B, 2)]
    ab = [cp(x, y) for x in A for y in B]
    exact = sum(1 for x in A if x in B)
    dbl = sum(len(re.findall(r"([぀-ヿ一-鿿])\1", t)) for t in A)
    dblb = sum(len(re.findall(r"([぀-ヿ一-鿿])\1", t)) for t in B)
    print(f"{apath} vs {bpath}: base-vs-base prefix median {st.median(bb):.0f} (min {min(bb)}), arm-vs-base median "
          f"{st.median(ab):.0f} (min {min(ab)}); arm texts found verbatim in base {exact}/{len(A)}; doubled CJK chars "
          f"arm {dbl} / base {dblb}")
