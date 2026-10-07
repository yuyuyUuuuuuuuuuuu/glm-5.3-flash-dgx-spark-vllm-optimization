"""Build the speed tables of docs/STATUS.md and docs/OPTIMIZATION.md from the committed logs (docs/logs/), so every
number in those tables can be traced to a line of a log in the repository.

  python3 tests/doc_tables.py [docs/logs]        (host or container; no GPU, standard library only)
  python3 tests/doc_tables.py --e5 LOG [LOG ...] E.5 table of each bench_ab_decode.log given, and (several logs) the
                                                 apply-graph production/TF ratio of each side by side

Inputs:
  run_all/bench_ab_decode.log         E.5 of the final tests/run_all.sh (production exl3_moe vs TF, 9 alternating
                                      rounds, median us per call)
  ab/<name>.log                       tests/bench_variant_ab.py runs (paired B/A, median of per-round ratios), one
                                      configuration line per (routing, T, repeat)
Output: markdown tables on stdout.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

CEIL = 250.0
PER_EXPERT_MIB = 6.0

E5_ROW = re.compile(
    r"^(rand|corr40)\s+(\d+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)")
E5_RATIOS = re.compile(r"^  (rand|corr40)\s+T=\s*(\d+) total .* per-round-median speedups \(bare eager, apply eager, "
                       r"bare graph, apply graph\) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)")
AB_ROW = re.compile(r"^(rand|corr40)\s+T=\s*(\d+) distinct\s+([\d.]+) empty-seg\s+([\d.]+) \| A\s+([\d.]+) us .*? B\s+"
                    r"([\d.]+) us .*?\| B/A ([\d.]+) \| spread A\s+([\d.]+)% B\s+([\d.]+)% \| (.*?) (True|False) \| "
                    r"E\.1 worst ([\d.e+-]+)")


def e5(path: Path) -> list[dict]:
    rows, ratios = [], {}
    for line in path.read_text().splitlines():
        m = E5_ROW.match(line)
        if m:
            g = m.groups()
            rows.append(dict(kind=g[0], T=int(g[1]), distinct=float(g[2]), be_x=float(g[4]), be_t=float(g[5]),
                             bg_x=float(g[7]), bg_t=float(g[8]), ae_x=float(g[10]), ae_t=float(g[11]),
                             ag_x=float(g[13]), ag_t=float(g[14]), gbps=float(g[16]), pct=float(g[17]),
                             floor=float(g[18])))
        m = E5_RATIOS.match(line)
        if m:
            ratios[(m.group(1), int(m.group(2)))] = tuple(float(v) for v in m.groups()[2:])
    for r in rows:
        r["prm"] = ratios.get((r["kind"], r["T"]))
    return rows


def ab(path: Path) -> dict:
    """{(kind, T): [row of repeat 0, row of repeat 1, ...]} plus the header."""
    out: dict = {}
    head = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("mode "):
            head.append(line)
        m = AB_ROW.match(line)
        if m:
            g = m.groups()
            out.setdefault((g[0], int(g[1])), []).append(dict(
                distinct=float(g[2]), ta=float(g[4]), tb=float(g[5]), ratio=float(g[6]), sa=float(g[7]),
                sb=float(g[8]), check=g[9], ok=g[10] == "True", worst=float(g[11])))
    return {"rows": out, "head": head}


def table_e5(rows: list[dict]) -> str:
    s = ["| routing | T | distinct | floor (µs) | apply graph: XL / TF (µs) | × | bare graph: XL / TF (µs) | × | "
         "TF GB/s (% of 250) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s.append(f"| {r['kind']} | {r['T']} | {r['distinct']:.1f} | {r['floor']:.1f} | {r['ag_x']:.1f} / {r['ag_t']:.1f} "
                 f"| {r['ag_x'] / r['ag_t']:.2f} | {r['bg_x']:.1f} / {r['bg_t']:.1f} | {r['bg_x'] / r['bg_t']:.2f} | "
                 f"{r['gbps']:.1f} ({r['pct']:.1f}%) |")
    return "\n".join(s)


def table_ab(d: dict, label_a: str, label_b: str) -> str:
    reps = max(len(v) for v in d["rows"].values())
    s = [f"| routing | T | distinct | {label_a} (µs) | {label_b} (µs) | B/A run 1 | " +
         " | ".join(f"B/A run {i + 1}" for i in range(1, reps)) + " | spread A / B (run 1) | check |",
         "|---" * (7 + reps - 1) + "|"]
    for (kind, T), rs in d["rows"].items():
        s.append(f"| {kind} | {T} | {rs[0]['distinct']:.1f} | {rs[0]['ta']:.1f} | {rs[0]['tb']:.1f} | "
                 + " | ".join(f"{r['ratio']:.3f}" for r in rs) + f" | {rs[0]['sa']:.1f}% / {rs[0]['sb']:.1f}% | "
                 f"{rs[0]['check']} {all(r['ok'] for r in rs)}, E.1 worst {max(r['worst'] for r in rs):.1e} |")
    return "\n".join(s)


def ratios_line(d: dict) -> str:
    out = []
    for kind in ("rand", "corr40"):
        ks = [(T, rs) for (k, T), rs in d["rows"].items() if k == kind]
        if ks:
            out.append(f"{kind:6s} " + "  ".join(f"T{T}:" + "/".join(f"{r['ratio']:.3f}" for r in rs) for T, rs in ks))
    return "\n".join(out)


def spread_of(d: dict, kinds=("rand", "corr40"), tmin=0, tmax=10 ** 9) -> tuple[float, float, float, int]:
    v = [r["ratio"] for (k, T), rs in d["rows"].items() if k in kinds and tmin <= T <= tmax for r in rs]
    return min(v), statistics.median(v), max(v), len(v)


def table_status(rows: list[dict], full: dict) -> str:
    """docs/STATUS.md's subset: E.5 (production vs TF) and the paired master -> final ratios side by side."""
    keep = {("rand", t) for t in (1, 2, 4, 8, 16, 32, 64, 96, 128)} | {("corr40", t) for t in (5, 8, 16, 32, 64)}
    s = ["| 経路 | T | 本番 apply: 本番 / TF (µs) | 倍率 | 呼び出し単体 graph: 本番 / TF (µs) | 倍率 | TF の GB/s(天井比) | "
         "floor (µs) | master の TF に対する時間比(対比較 2 回) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        k = (r["kind"], r["T"])
        if k not in keep or k not in full:
            continue
        f = full[k]
        s.append(f"| {r['kind']} | {r['T']} | {r['ag_x']:.1f} / {r['ag_t']:.1f} | {r['ag_x'] / r['ag_t']:.2f} | "
                 f"{r['bg_x']:.1f} / {r['bg_t']:.1f} | {r['bg_x'] / r['bg_t']:.2f} | {r['gbps']:.1f} ({r['pct']:.1f}%) | "
                 f"{r['floor']:.1f} | " + " / ".join(f"{x['ratio']:.3f}" for x in f) + " |")
    return "\n".join(s)


def e5_logs(paths: list[str]) -> None:
    runs = []
    for p in paths:
        rows = e5(Path(p))
        base = next((l for l in Path(p).read_text().splitlines() if l.startswith("baseline (XL):")), "baseline (XL): ?")
        mod = next((l for l in Path(p).read_text().splitlines() if l.startswith("production module:")), "")
        print(f"## E.5 ({p})\n\n{mod}\n{base}\n")
        print(table_e5(rows))
        print()
        runs.append({(r["kind"], r["T"]): r for r in rows})
    if len(runs) > 1:
        keys = [k for k in runs[0] if all(k in r for r in runs[1:])]
        print("| routing | T | " + " | ".join(f"apply graph XL / TF (µs), × [{i + 1}]" for i in range(len(runs))) + " |")
        print("|---|---|" + "---|" * len(runs))
        for k in keys:
            print(f"| {k[0]} | {k[1]} | " + " | ".join(
                f"{r[k]['ag_x']:.1f} / {r[k]['ag_t']:.1f}, {r[k]['ag_x'] / r[k]['ag_t']:.2f}" for r in runs) + " |")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--e5":
        e5_logs(sys.argv[2:])
        return
    base = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "logs")
    rows = e5(base / "run_all" / "bench_ab_decode.log")
    print("## E.5 (run_all/bench_ab_decode.log)\n")
    print(table_e5(rows))
    prod = [r for r in rows if r["kind"] == "corr40" and 5 <= r["T"] <= 64]
    agx = [r["ag_x"] / r["ag_t"] for r in prod]
    print(f"\ncorr40 T=5..64 apply-graph XL/TF: {min(agx):.2f}..{max(agx):.2f}; "
          f"% of ceiling {min(r['pct'] for r in prod):.1f}..{max(r['pct'] for r in prod):.1f}")
    rr = [r for r in rows if r["kind"] == "rand" and 5 <= r["T"] <= 64]
    print(f"rand T=5..64 apply-graph XL/TF: {min(r['ag_x'] / r['ag_t'] for r in rr):.2f}.."
          f"{max(r['ag_x'] / r['ag_t'] for r in rr):.2f}")
    for name, la, lb in (("full_master_vs_final", "A = master TF", "B = final TF"),
                         ("null_0_vs_0", "A = shipped", "B = shipped")):
        p = base / "ab" / f"{name}.log"
        if not p.exists():
            continue
        d = ab(p)
        print(f"\n## ab/{name}.log\n")
        print("\n".join(d["head"]))
        print()
        print(table_ab(d, la, lb))
        lo, med, hi, n = spread_of(d)
        print(f"\nall B/A: min {lo:.3f} median {med:.3f} max {hi:.3f} over {n}")
        dev = sorted(abs(r["ratio"] - 1) for rs in d["rows"].values() for r in rs)
        q = lambda f: dev[min(len(dev) - 1, int(round(f * (len(dev) - 1))))]
        print(f"|B/A - 1|: median {q(0.5) * 100:.2f}%, 90th percentile {q(0.9) * 100:.2f}%, 95th {q(0.95) * 100:.2f}%, "
              f"max {dev[-1] * 100:.2f}%; configurations whose two runs are both below 1 - 0.8%: "
              f"{sum(all(r['ratio'] < 0.992 for r in rs) for rs in d['rows'].values())} of {len(d['rows'])}, both "
              f"above 1 + 0.8%: {sum(all(r['ratio'] > 1.008 for r in rs) for rs in d['rows'].values())}")
        for kinds, t0, t1 in ((("corr40",), 5, 64), (("rand",), 5, 64), (("rand", "corr40"), 1, 4),
                              (("rand", "corr40"), 96, 128)):
            if any(k in kinds and t0 <= T <= t1 for (k, T) in d["rows"]):
                lo, med, hi, n = spread_of(d, kinds, t0, t1)
                print(f"{'+'.join(kinds)} T={t0}..{t1}: min {lo:.3f} median {med:.3f} max {hi:.3f} over {n}")
    pf = base / "ab" / "full_master_vs_final.log"
    if pf.exists():
        print("\n## docs/STATUS.md table (run_all/bench_ab_decode.log + ab/full_master_vs_final.log)\n")
        print(table_status(rows, ab(pf)["rows"]))
    print("\n## ladder (B/A per T, run 1 / run 2)\n")
    for name in ("k1_ord_1_vs_2", "k3_evictfirst_2_vs_6", "k6_table1_6_vs_9", "c2_table6_9_vs_12",
                 "k3b_discard_12_vs_0", "k2_apply_0_vs_1", "null_0_vs_0"):
        p = base / "ab" / f"{name}.log"
        if not p.exists():
            continue
        d = ab(p)
        ok = all(r["ok"] for rs in d["rows"].values() for r in rs)
        lo, med, hi, n = spread_of(d)
        gain = [f"{k} T{T}" for (k, T), rs in d["rows"].items() if len(rs) > 1 and all(r["ratio"] < 0.992 for r in rs)]
        loss = [f"{k} T{T}" for (k, T), rs in d["rows"].items() if len(rs) > 1 and all(r["ratio"] > 1.008 for r in rs)]
        print(f"### {name}  (checks {ok}; B/A min {lo:.3f} median {med:.3f} max {hi:.3f} over {n}; both runs < 0.992: "
              f"{len(gain)} of {len(d['rows'])}, both > 1.008: {len(loss)} {loss if loss else ''})")
        print("```")
        print(ratios_line(d))
        print("```")


if __name__ == "__main__":
    main()
