#!/usr/bin/env python3
"""deploy-r16: every marker boot_checks.sh greps for must be printable by the code that ships, or a check is vacuous
(a `never` string no code prints always passes) or impossible (a `need` string no code prints always fails).

Sources: the kit's site/*.py and overlay/*.py, launcher/overlay/*.py and launcher/start.sh, plus the production
launcher overlays the new launcher keeps (prod-launcher/overlay/*.py). A marker is printable when it is a substring of
some rendering of a string literal (%-format, str.format and f-string placeholders render as anything; adjacent
literals are one constant). `need` markers are also looked up verbatim in real logs given as extra arguments
(nodeC test logs and the in-container census of the kit), reported as seen / not seen (not a failure: several lines
only exist after model load in production).
Usage: check_boot_strings.py <boot_checks.sh> <kit dir> <prod launcher dir> [log files...]
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

PH = re.compile(r"(%[-+ #0]*\d*(?:\.\d+)?[sdrgfxi%]|\{[^{}]*\})")


def ph_regex(ph: str) -> str:
    """what a placeholder can render inside a marker: integers for %d, numbers for %f/%g, and for %s / %r / {} /
    f-string fields a value WITHOUT whitespace (a marker whose words would have to come from a placeholder is not
    evidence that the code prints it)."""
    if ph.endswith(("d", "i")) and ph.startswith("%"):
        return r"-?\d+"
    if ph.endswith(("f", "g")) and ph.startswith("%"):
        return r"[-+0-9.eEinfa]+"
    if ph == "%%":
        return "%"
    return r"\S*"


def literals(path: Path) -> list[list[str]]:
    """string literals of a Python file as chunk lists (placeholders split them)."""
    out = []
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = PH.split(node.value)          # literal, placeholder, literal, ...
            out.append((parts[0::2], [ph_regex(x) for x in parts[1::2]]))
        elif isinstance(node, ast.JoinedStr):
            chunks, cur = [], ""
            for v in node.values:
                if isinstance(v, ast.Constant):
                    cur += str(v.value)
                else:
                    chunks.append(cur); cur = ""
            chunks.append(cur)
            out.append((chunks, [r"\S*"] * (len(chunks) - 1)))
    return out


def shell_literals(path: Path) -> list[list[str]]:
    out = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', path.read_text(errors="replace")):
        chunks = re.split(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*", m.group(1))
        out.append((chunks, [r"\S*"] * (len(chunks) - 1)))
    return out


_PAT_CACHE: dict = {}
_CL = None               # the chunk lists, set before the fork so the pool workers inherit them


def _verdict(args):
    """per-marker verdict (independent, so the markers can be checked in parallel workers; the output in main()
    keeps the markers() order)."""
    kind, s = args
    code = printable(s, _CL)
    vllm_own = s.startswith("GPU KV cache size")
    return kind, s, code, vllm_own   # (chunk-list index, i, j) -> pattern STRING (depends only on the chunk list, not the
                        # marker: compiled/cached once instead of rebuilt for every marker x candidate)


def printable(marker: str, chunk_lists) -> bool:
    """marker printable by some literal: marker == chunks[i] substring, or
    marker == suffix(chunks[i]) ph_i chunks[i+1] ... ph_{j-1} prefix(chunks[j]) for some i < j.
    opt-kittools: decided by splitting the marker instead of one regex with an alternation of EVERY suffix of
    chunks[i] and EVERY prefix of chunks[j] (that pattern is quadratic in the chunk length - docstrings made it
    megabytes and the run 8 minutes). The regex fullmatches iff some split p <= q has marker[:p] a suffix of
    chunks[i], marker[p:q] fullmatching the middle (ph_i chunks[i+1] ... chunks[j-1] ph_{j-1}) and marker[q:] a prefix
    of chunks[j] - exactly what is tested here. printable_reference() is the previous regex implementation, kept
    for the differential test (tools/deploy16/test_check_boot_strings_equiv.py)."""
    toks = set(re.findall(r"[A-Za-z_]{3,}", marker))
    m = len(marker)
    suf_cache: dict = {}
    pre_cache: dict = {}

    def suffix_splits(a: str):   # p with marker[:p] a suffix of a
        r = suf_cache.get(a)
        if r is None:
            r = [p for p in range(min(m, len(a)) + 1) if a.endswith(marker[:p])]
            suf_cache[a] = r
        return r

    def prefix_splits(b: str):   # q with marker[q:] a prefix of b
        r = pre_cache.get(b)
        if r is None:
            r = [q for q in range(max(0, m - len(b)), m + 1) if b.startswith(marker[q:])]
            pre_cache[b] = r
        return r

    for idx, (chunks, phs) in enumerate(chunk_lists):
        joined = "".join(chunks)
        if toks and not any(t in joined for t in toks):
            continue
        n = len(chunks)
        for i in range(n):
            if marker in chunks[i]:
                return True
            ps = None
            for j in range(i + 1, n):
                if j > i + 1 and chunks[j - 1] not in marker:
                    break          # every full middle chunk must occur in the marker (later j need it too)
                key = (idx, i, j)
                mid = _MID_CACHE.get(key)
                if mid is None:
                    mid = re.compile("".join("(?:" + phs[t - 1] + ")" + re.escape(chunks[t]) for t in range(i + 1, j))
                                     + "(?:" + phs[j - 1] + ")", re.S)
                    if len(_MID_CACHE) < 200000:
                        _MID_CACHE[key] = mid
                if ps is None:
                    ps = suffix_splits(chunks[i])
                qs = prefix_splits(chunks[j])
                for p in ps:
                    for q in qs:
                        if q >= p and mid.fullmatch(marker, p, q):
                            return True
    return False


_MID_CACHE: dict = {}


def printable_reference(marker: str, chunk_lists) -> bool:
    # prefilter (speed only): some identifier-like token of the marker must occur in the literal's text
    toks = set(re.findall(r"[A-Za-z_]{3,}", marker))
    for idx, (chunks, phs) in enumerate(chunk_lists):
        joined = "".join(chunks)
        if toks and not any(t in joined for t in toks):
            continue
        n = len(chunks)
        for i in range(n):
            if marker in chunks[i]:
                return True
            for j in range(i + 1, n):
                if j > i + 1 and chunks[j - 1] not in marker:
                    break          # every full middle chunk must occur in the marker (later j need it too)
                # marker = suffix(chunks[i]) ph_i chunks[i+1] ... ph_{j-1} prefix(chunks[j])
                key = (idx, i, j)
                pat = _PAT_CACHE.get(key)
                if pat is None:
                    pre = "(?:" + "|".join(re.escape(chunks[i][k:]) for k in range(len(chunks[i]) + 1)) + ")"
                    mid = "".join("(?:" + phs[t - 1] + ")" + re.escape(chunks[t]) for t in range(i + 1, j))
                    post = "(?:" + phs[j - 1] + ")(?:" + "|".join(re.escape(chunks[j][:k]) for k in range(len(chunks[j]) + 1)) + ")"
                    pat = pre + mid + post
                    if len(_PAT_CACHE) < 200000 and len(pat) <= 1000000:
                        _PAT_CACHE[key] = pat
                if re.fullmatch(pat, marker, re.S):
                    return True
    return False


def markers(sh: str) -> list[tuple[str, str]]:
    res = []
    for m in re.finditer(r'^\s*(need|never) \S+ "\$\w+"(?: \d+)? "((?:[^"\\]|\\.)*)"', sh, re.M):
        res.append((m.group(1), m.group(2)))
    # for-loops: `for it in a b c; do need ... "... $it ..."` and `for bad in "x" "y"; do never ... "$bad"`
    for m in re.finditer(r'for (\w+) in (.*?); do\n?(.*?)(?:done|\n)', sh, re.S):
        var, items, body = m.group(1), m.group(2), m.group(3)
        vals = re.findall(r'"((?:[^"\\]|\\.)*)"', items) or items.split()
        for kind in ("need", "never"):
            for b in re.finditer(rf'{kind} \S+ "\$\w+"(?: \d+)? "((?:[^"\\]|\\.)*)"', body):
                tpl = b.group(1)
                if f"${var}" in tpl:
                    res += [(kind, tpl.replace(f"${var}", v)) for v in vals]
    return [(k, s.replace('\\"', '"')) for k, s in res if "$" not in s]


def main() -> int:
    sh_path, kit, prod = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    logs = [Path(p) for p in sys.argv[4:]]
    files = sorted(kit.glob("site/*.py")) + sorted(kit.glob("overlay/*.py")) + sorted(kit.glob("launcher/overlay/*.py")) \
        + [p for p in sorted(prod.glob("overlay/*.py")) if not (kit / "launcher/overlay" / p.name).exists()]
    chunk_lists = [c for f in files for c in literals(f)] + shell_literals(kit / "launcher/start.sh")
    # vLLM's own lines (GPU KV cache size) come from the image: accept them from the logs only
    log_text = "\n".join(p.read_text(errors="replace") for p in logs if p.is_file())
    bad = 0
    ms = markers(sh_path.read_text())

    results = []
    global _CL
    workers = min(8, os.cpu_count() or 1, len(ms) or 1)
    if workers > 1 and len(ms) > 8:
        import multiprocessing as mp
        ctx = mp.get_context("fork")           # fork: the workers inherit chunk_lists copy-on-write
        _CL = chunk_lists
        with ctx.Pool(workers) as pool:
            results = list(pool.imap(_verdict, ms, chunksize=1))
        _CL = None
    else:
        _CL = chunk_lists                      # (was `verdict(m)`: a NameError on this serial path)
        results = [_verdict(m) for m in ms]
    for kind, s, code, vllm_own in results:
        seen = s in log_text
        ok = code or vllm_own
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {kind:5s} {'code' if code else ('vllm' if vllm_own else '----')} "
              f"{'seen-in-log' if seen else '           '} {s}")
    print(f"{len(ms)} markers, {bad} not printable by the shipped code; {sum(s in log_text for _, s in ms)} seen verbatim "
          f"in {len([p for p in logs if p.is_file()])} log files")
    return 1 if bad or not ms else 0


if __name__ == "__main__":
    sys.exit(main())
