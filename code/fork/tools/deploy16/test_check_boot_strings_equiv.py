#!/usr/bin/env python3
"""opt-kittools: differential test of check_boot_strings.printable (split-based) against printable_reference (the
previous one-regex implementation). Both must give the same verdict for
  A. synthetic chunk lists over a tiny alphabet (placeholders of every kind, empty chunks, repeated chunks) and markers
     built from them (exact renderings, renderings with a wrong placeholder value, random edits, random strings);
  B. the real markers of a boot_checks.sh and mutations of them (a dropped / inserted / replaced character, a
     truncated or extended marker, two markers joined) against the real chunk lists of a kit.
Usage: test_check_boot_strings_equiv.py [<boot_checks.sh> <kit dir> <prod launcher dir> [n_real_mutations]]
Exit 0 = every verdict equal (prints the counts of True / False verdicts exercised)."""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_boot_strings as C  # noqa: E402

PHS = ["%s", "%d", "%5.2f", "%r", "{}", "{x}", "%%", "%i", "%g"]


def synth(rng: random.Random):
    alpha = "ab -:1x"
    nlit = rng.randint(1, 4)
    parts = []
    for k in range(nlit):
        parts.append("".join(rng.choice(alpha) for _ in range(rng.randint(0, 6))))
        if k < nlit - 1:
            parts.append(rng.choice(PHS))
    text = "".join(parts)
    sp = C.PH.split(text)
    return text, (sp[0::2], [C.ph_regex(x) for x in sp[1::2]])


def render(chunks, phs, rng):
    out = ""
    for t, c in enumerate(chunks):
        out += c
        if t < len(phs):
            pr = phs[t]
            out += rng.choice({r"-?\d+": ["12", "-3", "0"], r"[-+0-9.eEinfa]+": ["1.5", "inf", "-2e3"],
                               "%": ["%"], r"\S*": ["", "v", "a:b", "x1"]}.get(pr, ["?"]))
    return out


def part_a(n: int, seed: int = 1) -> tuple[int, int]:
    rng = random.Random(seed)
    t = f = 0
    for case in range(n):
        cls = [synth(rng)[1] for _ in range(rng.randint(1, 3))]
        chunks, phs = rng.choice(cls)
        full = render(chunks, phs, rng)
        kind = rng.randrange(5)
        if kind == 0:
            mk = full
        elif kind == 1 and full:
            a = rng.randrange(len(full)); b = rng.randrange(a, len(full) + 1); mk = full[a:b]
        elif kind == 2 and full:
            i = rng.randrange(len(full)); mk = full[:i] + rng.choice("ab -:1xZ ") + full[i + 1:]
        elif kind == 3:
            mk = "".join(rng.choice("ab -:1xZ%") for _ in range(rng.randint(1, 8)))
        else:
            mk = full + rng.choice(["", " ", "Z", "a"])
        C._MID_CACHE.clear(); C._PAT_CACHE.clear()   # both caches are keyed by the index into ONE chunk list set
        a1 = C.printable(mk, cls)
        a2 = C.printable_reference(mk, cls)
        if a1 != a2:
            print(f"MISMATCH synthetic case {case}: marker {mk!r} chunk lists {cls!r}: new {a1} reference {a2}")
            sys.exit(1)
        t += a1; f += not a1
    return t, f


_CL = None


def _both(mk):
    return mk, C.printable(mk, _CL), C.printable_reference(mk, _CL)


def part_b(sh: Path, kit: Path, prod: Path, nmut: int) -> tuple[int, int]:
    global _CL
    files = sorted(kit.glob("site/*.py")) + sorted(kit.glob("overlay/*.py")) + sorted(kit.glob("launcher/overlay/*.py")) \
        + [p for p in sorted(prod.glob("overlay/*.py")) if not (kit / "launcher/overlay" / p.name).exists()]
    _CL = [c for f in files for c in C.literals(f)] + C.shell_literals(kit / "launcher/start.sh")
    ms = [s for _, s in C.markers(sh.read_text())]
    rng = random.Random(7)
    cand = []
    for _ in range(nmut):
        s = rng.choice(ms)
        k = rng.randrange(6)
        if k == 0 and len(s) > 1:
            i = rng.randrange(len(s)); s = s[:i] + s[i + 1:]
        elif k == 1:
            i = rng.randrange(len(s) + 1); s = s[:i] + rng.choice("7Zq ") + s[i:]
        elif k == 2 and s:
            i = rng.randrange(len(s)); s = s[:i] + ("X" if s[i] != "X" else "Y") + s[i + 1:]
        elif k == 3 and len(s) > 4:
            s = s[: rng.randrange(2, len(s))]
        elif k == 4:
            s = s + rng.choice([" extra", "9", ")"])
        else:
            s = s + " " + rng.choice(ms)
        cand.append(s)
    cand += rng.sample(ms, min(len(ms), max(10, nmut // 4)))
    with mp.get_context("fork").Pool(int(os.environ.get("EQ_WORKERS", "4"))) as pool:
        res = list(pool.imap_unordered(_both, cand, chunksize=1))
    t = f = 0
    for mk, a1, a2 in res:
        if a1 != a2:
            print(f"MISMATCH real marker {mk!r}: new {a1} reference {a2}")
            sys.exit(1)
        t += a1; f += not a1
    return t, f


def main() -> int:
    t, f = part_a(int(os.environ.get("N_SYNTH", "20000")))
    print(f"A synthetic: {t + f} cases equal ({t} printable, {f} not)")
    if len(sys.argv) >= 4:
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 120
        t, f = part_b(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), n)
        print(f"B real markers + mutations: {t + f} cases equal ({t} printable, {f} not)")
    print("check_boot_strings equivalence: ALL EQUAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
