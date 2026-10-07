#!/usr/bin/env python3
"""usage: bench_check.py <label> -> output-correctness check of the 4 bench_round workloads
(structured exact, coding exec, prose/ja doubled-token garble heuristic)."""
import json, os, re, sys
L = sys.argv[1]; B = os.path.join(os.environ.get("DEPLOY_DIR") or os.path.expanduser("~/tf-exl3-deploy"), "bench")
want = " ".join(str(i) for i in range(1, 201))
def load(w):
    try: return [r["text"] for r in json.load(open(f"{B}/{L}_{w}.json"))["runs"]]
    except Exception: return None
out = []
T = load("structured")
if T: out.append(f"structured exact {sum(want.startswith(t.strip()) or t.strip().startswith(want) for t in T)}/{len(T)}")
T = load("coding")
if T:
    ok = 0
    for t in T:
        m = re.search(r"```python\n(.*?)```", t, re.S)
        try:
            ns = {}; exec(m.group(1), ns); ok += ns["clamp_range"]([-10, 5, 60, 25, 0, 50, 51]) == [0, 5, 50, 25, 0, 50, 50]
        except Exception: pass
    out.append(f"coding pass {ok}/{len(T)}")
T = load("ja")
if T:
    hard = [i for i, t in enumerate(T) if re.search(r"([のがをにはでとも一-龥])\1", t)]
    soft = sorted({t[m.start()-3:m.end()+3].replace("\n", " ") for t in T for m in re.finditer(r"([ァ-ヶぁ-ん])\1", t)})
    out.append(f"ja particle/kanji-doubled runs {len(hard)}/{len(T)}" + (f" {hard}" if hard else "") + (f" kana-doubles(check) {soft[:6]}" if soft else ""))
    out.append(f"ja distinct {len(set(T))}/{len(T)}")
for w, pat in (("prose", r"\b(\w{3,})\s+\1\b"),):
    T = load(w)
    if T:
        bad = [i for i, t in enumerate(T) if re.search(pat, t)]
        out.append(f"{w} doubled-token runs {len(bad)}/{len(T)}" + (f" {bad}" if bad else ""))
        out.append(f"{w} distinct {len(set(T))}/{len(T)}")
print(f"[{L}] check: " + " | ".join(out))
