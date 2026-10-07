#!/usr/bin/env python3
"""Build tools/klh/fixtures/klh_v1.json.gz (token ids, so nodeA needs no tokenizer). Runs on nodeC with a python that
has `tokenizers` and `jinja2` (e.g. uv venv); reads the production tokenizer.json (nodeA's snapshot copy) and the
GLM chat template. Every source is local or public domain:
  ql0..ql2   quality_long.py's three texts, token for token (asserted: 13,802 / 12,343 / 15,954 tokens)
  lp3..lp5   three more windows of the same corpus (/usr/share/common-licenses + /usr/lib/python3.12/*.py)
  ja-lit0/1  Natsume Soseki "Kokoro" (Aozora Bunko 773, public domain), ruby/annotations stripped
  ja-man     the Japanese man pages of this host (manpages-ja), rendered by groff
  tool0/1    synthetic tool-calling transcripts in GLM-5.3's chat format (the model's chat_template.jinja): tool schemas,
             assistant <think> + <tool_call> turns, tool responses carrying REAL file contents / listings / grep lines
  code-c     C headers of this host (/usr/include)
  dec fixtures: ~4,000-token prompts (sparse indexer active: ctx > 2,048) + 512 generated positions each;
             traj "greedy" (frozen by `klh.py make-traj` on production) or "real" (the text's own continuation).
usage: make_fixtures.py --tokenizer tokenizer.json --template chat_template.jinja --kokoro kokoro.txt --out F.json.gz
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import re
import subprocess

from tokenizers import Tokenizer

QL_LENS = (13802, 12343, 15954)


def corpus():
    files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
    return "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)


def window(t, tag, span, margin):
    s = random.Random(tag).randrange(0, len(t) - margin)
    return t[s:s + span]


def kokoro(path):
    raw = open(path, "rb").read().decode("cp932")
    body = raw.split("-------------------------------------------------------", 2)[-1]
    body = body.split("底本：")[0]
    body = re.sub(r"《[^》]*》", "", body)
    body = re.sub(r"［＃[^］]*］", "", body)
    body = body.replace("｜", "")
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def man_ja():
    pages = []
    for sec in ("man1", "man5", "man8"):
        pages += sorted(glob.glob("/usr/share/man/ja/%s/*.gz" % sec))
    out = []
    for p in pages:
        try:
            src = subprocess.run(["zcat", p], capture_output=True, check=True).stdout
            txt = subprocess.run(["groff", "-Tutf8", "-mandoc", "-Kutf8", "-P-c", "-rLL=160n"], input=src,
                                 capture_output=True).stdout
            txt = subprocess.run(["col", "-bx"], input=txt, capture_output=True).stdout.decode("utf-8", "ignore")
        except Exception:
            continue
        txt = re.sub(r"[ \t]{2,}", " ", txt)
        out.append(txt.strip())
    return "\n\n".join(out)


def c_headers():
    names = ["stdio.h", "stdlib.h", "string.h", "unistd.h", "pthread.h", "signal.h", "fcntl.h", "time.h", "math.h",
             "sys/socket.h", "netinet/in.h", "sys/stat.h", "dirent.h", "errno.h", "locale.h", "wchar.h", "stdint.h",
             "inttypes.h", "sys/mman.h", "sched.h", "poll.h", "termios.h", "search.h", "regex.h", "glob.h"]
    out = []
    for n in names:
        for base in ("/usr/include", "/usr/include/aarch64-linux-gnu", "/usr/include/x86_64-linux-gnu"):
            p = os.path.join(base, n)
            if os.path.exists(p):
                out.append("/* ==== %s ==== */\n%s" % (n, open(p, errors="ignore").read()))
                break
    return "\n\n".join(out)


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "absolute path"},
                    "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search files for a regular expression.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                    "max_results": {"type": "integer"}}, "required": ["pattern", "path"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "Run a short Python snippet and return stdout.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "write_note", "description": "Append a markdown note for the user.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["title", "body"]}}},
]
ASK = [
    "{f} の {fn} がどう動いているか調べて、要点を日本語でまとめてください。",
    "Can you look at {f} and explain what {fn} does, including edge cases?",
    "{f} の中で {pat} を使っている箇所を全部探して、使い方の違いを表にしてください。",
    "I think there is a bug around {fn} in {f}. Please read the code and tell me whether the behaviour is intended.",
    "このディレクトリ {d} の構成を見て、{f} が何のためのモジュールか説明して。",
    "Write a short note summarising how {f} handles errors, then check it with a quick experiment.",
]
THINK = [
    "ユーザーは {f} の {fn} について知りたがっている。まずファイルを読んで定義を確認する。",
    "The user wants an explanation of {fn}. I should read {f} first, then look for callers with grep.",
    "{pat} の使用箇所を探す必要がある。grep で {d} を検索してから該当ファイルを読む。",
    "Before answering I need the actual source. Let me list {d} to confirm the file exists.",
    "結果を確かめるために、小さな Python スニペットを実行して挙動を確認する。",
    "I have enough context now. Summarise the findings and write a note for the user.",
]
ANSWER = [
    "{f} の {fn} は、入力を検証したうえで内部状態を更新します。例外は呼び出し側に伝播し、境界条件では空の結果を返します。",
    "Summary: `{fn}` in `{f}` is a thin wrapper; the real work happens in the helper it calls. Edge cases are handled "
    "by the early returns at the top of the function.",
    "調査の結果、{pat} は {f} で主に設定値の読み込みに使われています。詳細はノートにまとめました。",
]


def py_files():
    return sorted(glob.glob("/usr/lib/python3.12/*.py"))


def funcs_of(src):
    return re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\(", src, re.M) or ["main"]


def snippet(src, rng, n=60):
    lines = src.splitlines()
    if len(lines) <= n:
        return "\n".join(lines), 1, len(lines)
    s = rng.randrange(0, len(lines) - n)
    return "\n".join(lines[s:s + n]), s + 1, s + n


def tool_transcript(template, seed, budget_chars):
    import jinja2
    rng = random.Random("klh-tool-%d" % seed)
    files = py_files()
    msgs = [{"role": "system", "content": "You are a careful coding assistant working in a Linux workspace. "
                                          "Answer in the user's language."}]
    total = 0
    turn = 0
    while total < budget_chars:
        f = rng.choice(files)
        src = open(f, errors="ignore").read()
        fn = rng.choice(funcs_of(src))
        pat = rng.choice(["import", "raise", "self\\.", "return None", "isinstance", "with open", "lambda"])
        d = os.path.dirname(f)
        ctx = {"f": f, "fn": fn, "pat": pat, "d": d}
        msgs.append({"role": "user", "content": rng.choice(ASK).format(**ctx)})
        for step in range(rng.randint(2, 4)):
            kind = rng.choice(["read_file", "read_file", "grep", "list_dir", "run_python"])
            if kind == "read_file":
                body, a, b = snippet(src, rng)
                args = {"path": f, "start_line": a, "end_line": b}
                out = body
            elif kind == "grep":
                hits = [("%s:%d:%s" % (os.path.basename(f), i + 1, l)) for i, l in enumerate(src.splitlines())
                        if re.search(pat, l)][:25]
                args = {"pattern": pat, "path": d, "max_results": 25}
                out = "\n".join(hits) or "(no matches)"
            elif kind == "list_dir":
                ents = sorted(os.listdir(d))
                s = rng.randrange(0, max(1, len(ents) - 40))
                args = {"path": d}
                out = "\n".join(ents[s:s + 40])
            else:
                code = "import %s\nprint(%s.__name__, len(dir(%s)))" % ((os.path.basename(f)[:-3],) * 3)
                args = {"code": code}
                out = json.dumps({"stdout": "%s %d\n" % (os.path.basename(f)[:-3], rng.randint(20, 300)),
                                  "exit_code": 0})
            msgs.append({"role": "assistant", "content": "", "reasoning_content": rng.choice(THINK).format(**ctx),
                         "tool_calls": [{"id": "call_%d_%d" % (turn, step), "type": "function",
                                         "function": {"name": kind, "arguments": args}}]})
            msgs.append({"role": "tool", "tool_call_id": "call_%d_%d" % (turn, step), "content": out})
        msgs.append({"role": "assistant", "content": rng.choice(ANSWER).format(**ctx),
                     "reasoning_content": rng.choice(THINK).format(**ctx)})
        turn += 1
        total = sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, ensure_ascii=True: json.dumps(v, ensure_ascii=ensure_ascii)
    tpl = env.from_string(open(template).read())
    full = tpl.render(messages=msgs, tools=TOOLS, add_generation_prompt=False, clear_thinking=False)
    # the generation prompt version: everything up to (and including) the LAST user turn + <|assistant|><think>
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    gen = tpl.render(messages=msgs[:last_user + 1], tools=TOOLS, add_generation_prompt=True, clear_thinking=False)
    return full, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--kokoro", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tk = Tokenizer.from_file(a.tokenizer)
    enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    for sp in ("<|assistant|>", "<tool_call>", "<|observation|>", "[gMASK]"):
        assert len(enc(sp)) == 1, sp
    T = corpus()
    pf, dec = [], []
    for i in range(3):
        ids = enc(window(T, "qlong-%d" % i, 60000, 70000))
        assert len(ids) == QL_LENS[i], (i, len(ids))
        pf.append({"name": "ql%d" % i, "group": "lic-py", "ids": ids, "min_pos": 1024, "quick": i == 0})
    for i in range(3, 6):
        pf.append({"name": "lp%d" % i, "group": "lic-py", "ids": enc(window(T, "klh-lp-%d" % i, 60000, 70000)),
                   "min_pos": 1024})
    K = kokoro(a.kokoro)
    n = len(K)
    pf.append({"name": "ja-lit0", "group": "ja", "ids": enc(K[int(0.05 * n):int(0.05 * n) + 19000]), "min_pos": 1024,
               "quick": True})
    pf.append({"name": "ja-lit1", "group": "ja", "ids": enc(K[int(0.55 * n):int(0.55 * n) + 19000]), "min_pos": 1024})
    M = man_ja()
    pf.append({"name": "ja-man", "group": "ja", "ids": enc(M[:26000]), "min_pos": 1024})
    for s in (0, 1):
        full, _ = tool_transcript(a.template, s, 52000)
        pf.append({"name": "tool%d" % s, "group": "tool", "ids": enc(full), "min_pos": 1024, "quick": s == 0})
    C = c_headers()
    pf.append({"name": "code-c", "group": "code", "ids": enc(C[:48000]), "min_pos": 1024})
    # decode fixtures: 4,000-token prompts, 512 positions
    P = 4000
    G = 512
    w = enc(window(T, "klh-dec-lp-0", 40000, 70000))
    dec.append({"name": "d-lp0", "group": "lic-py", "prompt": w[:P], "gen": G, "traj": "greedy", "quick": True})
    w = enc(window(T, "klh-dec-lp-1", 40000, 70000))
    dec.append({"name": "d-lp1", "group": "lic-py", "prompt": w[:P], "gen": G, "traj": "real", "real": w[P:P + G]})
    w = enc(K[int(0.80 * n):int(0.80 * n) + 12000])
    dec.append({"name": "d-ja", "group": "ja", "prompt": w[:P], "gen": G, "traj": "greedy"})
    w = enc(M[30000:60000])
    dec.append({"name": "d-jaman", "group": "ja", "prompt": w[:P], "gen": G, "traj": "greedy"})
    _, gp = tool_transcript(a.template, 2, 24000)
    dec.append({"name": "d-tool", "group": "tool", "prompt": enc(gp), "gen": G, "traj": "greedy", "quick": True})
    w = enc(C[48000:90000])
    dec.append({"name": "d-code", "group": "code", "prompt": w[:P], "gen": G, "traj": "greedy"})
    for d in dec:
        if d["traj"] == "real":
            assert len(d["real"]) == G, d["name"]
    fx = {"version": 1, "tokenizer_sha256": hashlib.sha256(open(a.tokenizer, "rb").read()).hexdigest(),
          "template_sha256": hashlib.sha256(open(a.template, "rb").read()).hexdigest(),
          "kokoro_sha256": hashlib.sha256(open(a.kokoro, "rb").read()).hexdigest(),
          "built": "make_fixtures.py", "pf": pf, "dec": dec}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.GzipFile(a.out, "wb", mtime=0) as f:
        f.write(json.dumps(fx, separators=(",", ":")).encode())
    for d in pf:
        print("pf  %-8s %-7s %6d tokens (scored from position %d)" % (d["name"], d["group"], len(d["ids"]), d["min_pos"]))
    for d in dec:
        print("dec %-8s %-7s prompt %5d + %d (%s)" % (d["name"], d["group"], len(d["prompt"]), d["gen"], d["traj"]))
    print("total pf tokens %d, dec prompt tokens %d" % (sum(len(d["ids"]) for d in pf),
                                                         sum(len(d["prompt"]) + d["gen"] for d in dec)))


if __name__ == "__main__":
    main()
