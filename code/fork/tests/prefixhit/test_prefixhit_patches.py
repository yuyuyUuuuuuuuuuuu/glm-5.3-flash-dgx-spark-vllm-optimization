#!/usr/bin/env python3
"""CPU unit tests for the prefixhit overlays (no torch, no GPU).

  python3 tests/prefixhit/test_prefixhit_patches.py [path/to/pristine/mamba_hybrid.py]

Default source: the image's file (inside the container) or $PH_MAMBA_HYBRID_SRC. Checks, on a scratch copy:
  1. patch_mamba_align_seed: the stock add_request seeds a prefix hit at 27648 in column 47 when the worker sees
     the engine core's recomputed cache_config.block_size (576) -- the bug -- and in column 5 after the patch (also 5
     with block_size == mamba_block_size == 4608, i.e. production's mp worker, before and after); idempotent.
  2. patch_kpool_tail_positions: prepare_attn's build_attn_metadata call gains positions=input_batch.positions (and
     nothing else changes); idempotent; both patches compose in either order.
"""
import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PH_MAMBA_HYBRID_SRC", "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"))
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "overlay" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def method_src(text, cls, meth):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == meth:
                    return ast.get_source_segment(text, f), f
    raise KeyError(meth)


def seed_column(text, num_computed, block_size, mamba_block_size):
    src, _ = method_src(text, "MambaHybridModelState", "add_request")
    rec = {}

    class Cell:
        def __init__(self, key):
            self.key = key

        def fill_(self, v):
            rec[self.key] = v

    class Buf:
        def __init__(self, key):
            self.key = key

        def __getitem__(self, i):
            return Cell(self.key)

    class Base:
        def add_request(self, req_index, new_req_data):
            pass

    ns = {"Base": Base}
    exec("from __future__ import annotations\nclass Stub(Base):\n" + textwrap.indent(textwrap.dedent(src), "    "), ns)
    s = ns["Stub"]()
    s.num_accepted_tokens_gpu = Buf("acc")
    s._mamba_state_idx_gpu = Buf("idx")
    s._align_mode = True
    s.cache_config = type("C", (), {"block_size": block_size, "mamba_block_size": mamba_block_size})()
    s.add_request(0, type("D", (), {"num_computed_tokens": num_computed})())
    return rec["idx"]


def build_call_kwargs(text):
    _, f = method_src(text, "MambaHybridModelState", "prepare_attn")
    for node in ast.walk(f):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_attn_metadata":
            return {k.arg: ast.unparse(k.value) for k in node.keywords}
    raise KeyError("build_attn_metadata")


def run_patch(name, env, path):
    e = dict(os.environ, GLM53_MAMBA_HYBRID_PY=str(path), **{env: "1"})
    r = subprocess.run([sys.executable, str(HERE / "overlay" / f"{name}.py")], env=e, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    pristine = SRC.read_text()
    tmp = Path(tempfile.mkdtemp(prefix="ph-"))
    # ---- 1. seed
    p = tmp / "seed.py"
    shutil.copy(SRC, p)
    c_bug = seed_column(pristine, 27648, 576, 4608)
    c_mp = seed_column(pristine, 27648, 4608, 4608)
    check(c_bug == 47, f"stock seed with the engine core's recomputed block_size 576: column {c_bug} (bug: != 5)")
    check(c_mp == 5, f"stock seed with block_size 4608 (production mp worker): column {c_mp}")
    rc, out = run_patch("patch_mamba_align_seed", "GLM53_MAMBA_ALIGN_SEED", p)
    check(rc == 0 and "patched" in out, "seed patch applies: " + out.strip().splitlines()[-1])
    t = p.read_text()
    for nc, want in ((27648, 5), (32256, 6), (36864, 7), (0, -1), (4609, 1)):
        got = seed_column(t, nc, 576, 4608)
        check(got == want, f"patched seed num_computed {nc} block_size 576 mamba 4608 -> column {got} (want {want})")
    check(seed_column(t, 27648, 4608, None) == 5, "patched seed falls back to block_size when mamba_block_size unset")
    rc, out = run_patch("patch_mamba_align_seed", "GLM53_MAMBA_ALIGN_SEED", p)
    check(rc == 0 and "already present" in out and p.read_text() == t, "seed patch idempotent")
    # ---- 2. tail positions
    q = tmp / "pos.py"
    shutil.copy(SRC, q)
    kw0 = build_call_kwargs(pristine)
    check("positions" not in kw0, "stock prepare_attn does not pass positions (the bug)")
    rc, out = run_patch("patch_kpool_tail_positions", "GLM53_KPOOL_TAIL_POSITIONS", q)
    check(rc == 0 and "patched" in out, "positions patch applies")
    kw1 = build_call_kwargs(q.read_text())
    check(kw1.get("positions") == "input_batch.positions", f"patched prepare_attn passes positions={kw1.get('positions')}")
    check({k: v for k, v in kw1.items() if k != "positions"} == kw0, "no other build_attn_metadata argument changed")
    t2 = q.read_text()
    rc, out = run_patch("patch_kpool_tail_positions", "GLM53_KPOOL_TAIL_POSITIONS", q)
    check(rc == 0 and "already present" in out and q.read_text() == t2, "positions patch idempotent")
    # ---- composition in both orders gives the same file
    a, b = tmp / "ab.py", tmp / "ba.py"
    shutil.copy(SRC, a); shutil.copy(SRC, b)
    run_patch("patch_mamba_align_seed", "GLM53_MAMBA_ALIGN_SEED", a)
    run_patch("patch_kpool_tail_positions", "GLM53_KPOOL_TAIL_POSITIONS", a)
    run_patch("patch_kpool_tail_positions", "GLM53_KPOOL_TAIL_POSITIONS", b)
    run_patch("patch_mamba_align_seed", "GLM53_MAMBA_ALIGN_SEED", b)
    check(a.read_text() == b.read_text(), "both patches compose in either order")
    ast.parse(a.read_text())
    # ---- 0 = stock
    z = tmp / "zero.py"
    shutil.copy(SRC, z)
    for name, env in (("patch_mamba_align_seed", "GLM53_MAMBA_ALIGN_SEED"),
                      ("patch_kpool_tail_positions", "GLM53_KPOOL_TAIL_POSITIONS")):
        e = dict(os.environ, GLM53_MAMBA_HYBRID_PY=str(z), **{env: "0"})
        r = subprocess.run([sys.executable, str(HERE / "overlay" / f"{name}.py")], env=e, capture_output=True, text=True)
        check(r.returncode == 0 and z.read_text() == pristine, f"{env}=0 leaves the file untouched")
    shutil.rmtree(tmp)
    print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
