#!/usr/bin/env python3
"""CPU tests of overlay/patch_kpool_tail_ring.py itself (host python, no torch): install states, fail-closed paths,
idempotency, composition with patch_kpool_tail_seed_stride.py in both re-run orders, the bundle registration, and
that the patched decode kernel/wrapper are upstream #58454's code (AST equal, docstrings aside).

    python3 tests/kpoolring/test_patch_kpool_tail_ring.py [vllm source dir]

The vllm source dir defaults to the read-only extracted image sources ($TF_EXL3_ASSETS/vllm-src/vllm,
byte-identical to the image's dist-packages/vllm for the three target files).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "vllm-src/vllm"))
FILES = {
    "GLM53_KPOOL_COMPRESS_PY": "models/glm5next/nvidia/ops/kpool_compress.py",
    "GLM53_SPARSE_INDEXER_KPOOL_PY": "model_executor/layers/sparse_attn_indexer_kpool.py",
    "GLM53_GLM5NEXT_ATTENTION_PY": "models/glm5next/nvidia/attention.py",
}
RING = REPO / "overlay/patch_kpool_tail_ring.py"
SEED = REPO / "overlay/patch_kpool_tail_seed_stride.py"
UPSTREAM = REPO / "tests/kpoolring/ref/kpool_compress_vllm58454.py"
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def fresh() -> tuple[Path, dict]:
    d = Path(tempfile.mkdtemp(prefix="kpoolring-cpu-"))
    env = dict(os.environ)
    for var, rel in FILES.items():
        dst = d / Path(rel).name
        shutil.copy(SRC / rel, dst)
        env[var] = str(dst)
    return d, env


def run(script: Path, env: dict, ring: str | None = "1") -> subprocess.CompletedProcess:
    e = dict(env)
    e.pop("GLM53_KPOOL_RING", None)
    if ring is not None:
        e["GLM53_KPOOL_RING"] = ring
    return subprocess.run([sys.executable, str(script)], env=e, capture_output=True, text=True)


def shas(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.py"))}


spec = importlib.util.spec_from_file_location("ring_patch", RING)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# ---- ring size formula (upstream test table + production)
table = {k: M.ring_slots(4, k) for k in (0, 1, 4, 5, 7, 13)}
check(table == {0: 4, 1: 8, 4: 8, 5: 16, 7: 16, 13: 32}, f"ring_slots(4, k) = {table}")
check(all(4608 % M.ring_slots(4, k) == 0 for k in range(0, 60)), "every ring up to k=59 divides the 4608 block")

# ---- pinned pristine region
src_kc = (SRC / FILES["GLM53_KPOOL_COMPRESS_PY"]).read_text()
check(M._sha(M.split_region(src_kc)[1]) == M.REGION_PRISTINE_SHA, "image decode region matches the pinned sha256")

# ---- 1. without the #57477 seed fix: fail closed, nothing written
d, env = fresh()
before = shas(d)
r = run(RING, env)
check(r.returncode != 0 and "patch_kpool_tail_seed_stride.py first" in (r.stdout + r.stderr) and shas(d) == before,
      f"no seed-stride -> rc={r.returncode}, files untouched: {(r.stdout + r.stderr).strip()[-140:]}")

# ---- 2. env values other than exactly 1 refuse (the bundle only runs it for a non-empty value)
for val in ("0", "yes", " 1x", "true"):
    r = run(RING, env, ring=val)
    check(r.returncode != 0 and shas(d) == before, f"GLM53_KPOOL_RING={val!r} refused (rc={r.returncode})")

# ---- 3. seed-stride then ring: patched; second runs of both, in both orders, are byte-identical no-ops
r1 = run(SEED, env)
check(r1.returncode == 0 and "patched" in r1.stdout, f"seed-stride: {r1.stdout.strip()}")
r2 = run(RING, env)
check(r2.returncode == 0 and r2.stdout.count(": patched") == 3, f"ring: {r2.stdout.strip()}")
after = shas(d)
for order in ((RING, SEED), (SEED, RING)):
    outs = [run(s, env) for s in order]
    check(all(o.returncode == 0 and "already present" in o.stdout for o in outs) and shas(d) == after,
          "re-run " + " -> ".join(s.name for s in order) + ": " + " | ".join(o.stdout.strip() for o in outs))

# ---- 4. patched decode kernel + wrapper are upstream #58454's code (AST, docstrings stripped)
def fn_ast(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                node.body = body[1:]
            return ast.dump(node, include_attributes=False)
    raise KeyError(name)


patched_kc = Path(env["GLM53_KPOOL_COMPRESS_PY"]).read_text()
up_kc = UPSTREAM.read_text()
for name in ("_kpool_decode_update_batched_kernel", "kpool_decode_update_and_maybe_write_cache_batched"):
    same = fn_ast(patched_kc, name) == fn_ast(up_kc, name)
    check(same, f"{name}: AST-equal to upstream 2617fe93")
    stock_same = fn_ast(src_kc, name) == fn_ast(up_kc, name)
    check(not stock_same, f"{name}: stock image differs from upstream (sanity)")
# the seed kernel keeps seed-stride's text; upstream's differs only by KPOOL->RING in the addressing
check(fn_ast(patched_kc, "_kpool_tail_seed_kernel") != fn_ast(up_kc, "_kpool_tail_seed_kernel"),
      "seed kernel intentionally NOT rewritten (owned by patch_kpool_tail_seed_stride.py; ring passed at the call site)")
idx_text = Path(env["GLM53_SPARSE_INDEXER_KPOOL_PY"]).read_text()
calls = [n for n in ast.walk(ast.parse(idx_text)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == "kpool_seed_tail_cache"]
check(len(calls) == 1 and ast.unparse(calls[0].args[4]) == "tail_kv_cache.shape[2]",
      f"seed call kpool argument: {ast.unparse(calls[0].args[4]) if calls else None}")
attn_text = Path(env["GLM53_GLM5NEXT_ATTENTION_PY"]).read_text()
check("block_size=ring," in attn_text and "sliding_window=ring," in attn_text
      and "self.cache_config.block_size % ring == 0" in attn_text, "tail spec: block_size / sliding_window = ring, assert")
for label, path in (("kpool_compress", "GLM53_KPOOL_COMPRESS_PY"), ("indexer", "GLM53_SPARSE_INDEXER_KPOOL_PY"),
                    ("attention", "GLM53_GLM5NEXT_ATTENTION_PY")):
    try:
        compile(Path(env[path]).read_text(), path, "exec")
        check(True, f"{label} compiles")
    except SyntaxError as exc:
        check(False, f"{label}: {exc}")

# ---- 5. drift in any one file: fail closed, NO file written (preflight before write)
for var, needle, repl in (
    ("GLM53_GLM5NEXT_ATTENTION_PY", "sliding_window=self._index_kpool,", "sliding_window=self._index_kpool ,"),
    ("GLM53_SPARSE_INDEXER_KPOOL_PY", "                            index_kpool,\n                            head_dim,\n",
     "                            index_kpool,  # drift\n                            head_dim,\n"),
    ("GLM53_KPOOL_COMPRESS_PY", "        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE\n",
     "        block = tl.maximum(tail_slot, 0).to(tl.int64) // (POOL_SIZE)\n"),
):
    d2, env2 = fresh()
    run(SEED, env2)
    p = Path(env2[var])
    t = p.read_text()
    assert t.count(needle) == 1, (var, needle)
    p.write_text(t.replace(needle, repl))
    b2 = shas(d2)
    r = run(RING, env2)
    check(r.returncode != 0 and shas(d2) == b2, f"drift in {p.name}: rc={r.returncode}, no file written; "
          f"{(r.stdout + r.stderr).strip()[-110:]}")

# ---- 6. partial / inconsistent installs are refused
d3, env3 = fresh()
run(SEED, env3)
run(RING, env3)
p = Path(env3["GLM53_KPOOL_COMPRESS_PY"])
p.write_text(p.read_text().replace("        phys_slot = safe_pos % RING\n", "        phys_slot = safe_pos % POOL_SIZE\n"))
r = run(RING, env3)
check(r.returncode != 0 and "partial/inconsistent" in (r.stdout + r.stderr), "hand-edited patched region refused")

# ---- 7. the bundle registers it after seed-stride and skips it when unset
bspec = importlib.util.spec_from_file_location("bundle", REPO / "overlay/patch_tf_bundle.py")
B = importlib.util.module_from_spec(bspec)
bspec.loader.exec_module(B)
names = [f for f, _ in B.PATCHES]
check(("patch_kpool_tail_ring.py", "GLM53_KPOOL_RING") in B.PATCHES
      and names.index("patch_kpool_tail_seed_stride.py") < names.index("patch_kpool_tail_ring.py"),
      f"bundle PATCHES order: {names}")
os.environ.pop("GLM53_KPOOL_RING", None)
import contextlib
import io

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    B.run_patch("patch_kpool_tail_ring.py", "GLM53_KPOOL_RING")
check("unset -> skipped (stock)" in buf.getvalue(), f"bundle with GLM53_KPOOL_RING unset: {buf.getvalue().strip()}")

print(f"== {'ALL OK' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
