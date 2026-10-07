"""Path hygiene of the published tree (host, CPU only; the last step of tests/run_all.sh).

The published code refers to host directories only through tests/paths.sh's variables (TF_EXL3_MODELS, TF_EXL3_ASSETS,
TF_EXL3_KITS) or ~-relative defaults. This fails on what would break on another machine:
  - a placeholder that stands for a machine-specific value (${HOME}, ${WORKER_HOME}, ${HEAD_IP}, ${WORKER_IP},
    ${WORKER_USER}, <LAN_IP>) inside a Python string that is not a docstring: Python does not expand it, so the path or
    address is wrong at run time (shell code that Python strings generate keeps its other ${VAR}s; those are fine);
  - an absolute /home/<name>/ or /Users/<name>/ path in code, scripts or configuration (docs and logs excluded).
Usage: check_paths.py [code dir, default this tree's code/]. Exit 1 on any finding.
"""
from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

CODE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent.parent  # code/
_VARS = "HOME|WORKER_HOME|HEAD_IP|WORKER_IP|WORKER_USER"
PLACEHOLDER = re.compile(r"\$\{(?:" + _VARS + r")\}|<" + "LAN_IP>")
ABS_HOME = re.compile(r"(?<![\w$])/(?:home|Users)/[a-z_][a-z0-9_.-]*/")
TEXT = {".py", ".sh", ".toml", ".cfg", ".json", ".example", ".r16", ""}
fails: list[str] = []


def py_strings(path: Path):
    src = path.read_text()
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, t in enumerate(toks):
        if t.type not in (tokenize.STRING, tokenize.FSTRING_MIDDLE):
            continue
        prev = next((x for x in reversed(toks[:i]) if x.type not in (tokenize.NL, tokenize.COMMENT)), None)
        nxt = next((x for x in toks[i + 1:] if x.type not in (tokenize.NL, tokenize.COMMENT)), None)
        doc = (t.type == tokenize.STRING and (prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENCODING)) and nxt is not None and nxt.type in (tokenize.NEWLINE,
               tokenize.ENDMARKER))
        if not doc:
            yield t.start[0], t.string


n_py = n_text = 0
for p in sorted(CODE.rglob("*")):
    if not p.is_file() or "/docs/" in p.as_posix() or p.name.endswith((".md", ".log", ".patch", ".txt")):
        continue
    if p.suffix == ".py":
        n_py += 1
        # f-strings: "${HOME}" there is "$" + the value of a variable HOME, so look at the raw source line too
        lines = p.read_text().splitlines()
        for ln, s in py_strings(p):
            if PLACEHOLDER.search(s):
                fails.append(f"{p.relative_to(CODE)}:{ln}: unexpanded placeholder in a Python string: {s[:100]}")
        for ln, line in enumerate(lines, 1):
            if re.search(r'f["\'][^"\']*\$\{(?:' + _VARS + r')\}', line):
                fails.append(f"{p.relative_to(CODE)}:{ln}: shell placeholder inside an f-string: {line.strip()[:100]}")
    if p.suffix in TEXT:
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        n_text += 1
        for ln, line in enumerate(text.splitlines(), 1):
            if ABS_HOME.search(line):
                fails.append(f"{p.relative_to(CODE)}:{ln}: absolute home path: {line.strip()[:100]}")

for f in fails:
    print("FAIL " + f)
print(f"checked {n_py} Python files, {n_text} code / script / config files under {CODE.name}/: "
      f"{'OK' if not fails else f'{len(fails)} finding(s)'}")
sys.exit(1 if fails else 0)
