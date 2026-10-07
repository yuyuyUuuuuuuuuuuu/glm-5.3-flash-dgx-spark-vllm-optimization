#!/usr/bin/env python3
"""Fill the generated tables of docs/BLOCK_VERIFY.md from the committed logs (EXACT_TABLE / RB_TABLE from
docs/logs/blockverify/exact_*.json via exact_table.py, GAIN_TABLE from estimate_gain.log). Idempotent: a filled doc is
regenerated between the <!-- gen:NAME --> markers."""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs/BLOCK_VERIFY.md"
LOG = REPO / "docs/logs/blockverify"
tables = subprocess.run([sys.executable, str(REPO / "tools/blockverify/exact_table.py"), str(LOG)], check=True,
                        capture_output=True, text=True).stdout.strip().split("\n\n")
exact, rb = tables[0], tables[1]
gain = json.loads([l for l in (LOG / "estimate_gain.log").read_text().splitlines() if l.startswith("{")][-1])
g = ["| family | n | tokens/step standard | tokens/step block | gain % (SE) | standard unconditional acceptance per position |",
     "|---|---|---|---|---|---|"]
for fam, r in gain.items():
    for n in (4, 5, 7):
        x = r[f"n{n}"]
        g.append(f"| {fam} | {n} | {x['std_tokens_per_step']:.3f} | {x['block_tokens_per_step']:.3f} | "
                 f"{x['gain_pct']:+.2f} ({x['gain_pct_se']:.2f}) | {', '.join(f'{v:.2f}' for v in x['std_uncond_per_pos'])} |")
blocks = {"EXACT_TABLE": exact, "RB_TABLE": rb, "GAIN_TABLE": "\n".join(g)}
s = DOC.read_text()
for name, body in blocks.items():
    new = f"<!-- gen:{name} -->\n{body}\n<!-- /gen:{name} -->"
    if f"<!-- gen:{name} -->" in s:
        s = re.sub(rf"<!-- gen:{name} -->.*?<!-- /gen:{name} -->", lambda _m: new, s, flags=re.S)
    else:
        assert s.count(f"\n{name}\n") == 1, name
        s = s.replace(f"\n{name}\n", f"\n{new}\n")
DOC.write_text(s)
print("filled", ", ".join(blocks))
