"""Additive, anchored edit of the GLM launcher start.sh: pass GLM53_KPOOL_RING to BOTH rank containers.

The bundle (overlay/patch_tf_bundle.py) runs overlay/patch_kpool_tail_ring.py inside each container only when
GLM53_KPOOL_RING is non-empty in THAT container's environment, so the variable must reach
  - the head:   an explicit `-e GLM53_KPOOL_RING="${GLM53_KPOOL_RING:-}"` in the head `docker run`
                (inserted right after the GLM53_KPOOL_SEED_STRIDE line), and
  - the worker: the serve_env_names list (the worker `docker run` is built from it over ssh).
A rank that misses it would keep the 4-slot ring while the other rank sizes a 16-slot one: the KV-cache spec is
computed per worker, so the two ranks would disagree on the tail group's block size.

Usage (on nodeA, by the operator):  python3 edit_start_env_kpoolring.py [--apply] [path/to/start.sh]
Default path ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/start.sh. Default is a dry run that prints the diff. Every anchor must
occur exactly once or nothing is written; already-applied is a no-op; --apply keeps a timestamped backup and runs
`bash -n` on the result before replacing the file.
"""
import difflib
import os
import subprocess
import sys
import tempfile
import time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
P = args[0] if args else os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/start.sh")
s0 = open(P).read()

LIST_OLD = " GLM53_PREFILL_FUSED_CAP GLM53_FP8_LARGE_M GLM53_KPOOL_SEED_STRIDE; do  # [tf-exl3-fork]\n"
LIST_NEW = " GLM53_PREFILL_FUSED_CAP GLM53_FP8_LARGE_M GLM53_KPOOL_SEED_STRIDE GLM53_KPOOL_RING; do  # [tf-exl3-fork]\n"
HEAD_ANCHOR = '        -e GLM53_KPOOL_SEED_STRIDE="${GLM53_KPOOL_SEED_STRIDE:-}" \\\n'
HEAD_LINE = '        -e GLM53_KPOOL_RING="${GLM53_KPOOL_RING:-}" \\\n'

if s0.count(LIST_NEW) == 1 and s0.count(HEAD_ANCHOR + HEAD_LINE) == 1:
    print("already applied (worker list and head -e both present) - nothing to do")
    sys.exit(0)
if "GLM53_KPOOL_RING" in s0:
    sys.exit("ABORT: GLM53_KPOOL_RING appears in start.sh but not in the expected two places; edit by hand")
for name, anchor in (("worker serve_env_names list", LIST_OLD), ("head -e GLM53_KPOOL_SEED_STRIDE", HEAD_ANCHOR)):
    n = s0.count(anchor)
    if n != 1:
        sys.exit(f"ABORT: {name} anchor found {n}x (need 1): {anchor.strip()[:100]}")
s = s0.replace(LIST_OLD, LIST_NEW, 1).replace(HEAD_ANCHOR, HEAD_ANCHOR + HEAD_LINE, 1)
d = list(difflib.unified_diff(s0.splitlines(True), s.splitlines(True), "start.sh", "start.sh+kpoolring", n=1))
sys.stdout.writelines(d)
with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
    f.write(s)
    tmpname = f.name
r = subprocess.run(["bash", "-n", tmpname], capture_output=True, text=True)
os.unlink(tmpname)
if r.returncode:
    sys.exit(f"ABORT: bash -n failed on the edited script: {r.stderr.strip()}")
print("\nbash -n: ok")
if "--apply" in sys.argv:
    bak = P + ".bak-kpoolring-" + time.strftime("%Y%m%d-%H%M%S")
    open(bak, "w").write(s0)
    os.chmod(bak, 0o644)
    tmp = P + ".tmp-kpoolring"
    open(tmp, "w").write(s)
    os.chmod(tmp, os.stat(P).st_mode)
    os.replace(tmp, P)
    print(f"applied; backup: {bak}")
else:
    print("dry run (no changes). re-run with --apply")
