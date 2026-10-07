"""Additive, anchored edit of the GLM launcher start.sh for the tf-exl3-fork bundle.
Usage on nodeA:  python3 - [--apply] < edit_start_sh.py      (default: dry run, prints the diff)
Every anchor must occur exactly once or nothing is written. A timestamped backup is kept on --apply.
All additions are inert unless TF_EXL3_MOE / GLM53_SPEC_RESAMPLE_INDEPENDENT / GLM53_DRAFT_* are set in .env."""
import difflib, os, sys, time
P = os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/start.sh")
MARK = "[tf-exl3-fork]"
s0 = open(P).read()
if MARK in s0:
    print("already applied (marker present) - nothing to do"); sys.exit(0)
ins = [  # (anchor, text inserted AFTER the anchor)
 ('DEFAULT_TOKENS_PATCH_HOST="${DEFAULT_TOKENS_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_default_max_new_tokens.py}"\n',
  '# [tf-exl3-fork] bundle: TensorFold EXL3 MoE kernels + resample-noise fix + drafter FP8 (inert unless its knobs are set)\n'
  'TF_BUNDLE_PATCH_HOST="${TF_BUNDLE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_tf_bundle.py}"\n'
  'TF_BUNDLE_DIR_HOST="${TF_BUNDLE_DIR_HOST:-$SCRIPT_DIR/overlay/tf}"\n'),
 ('        "$DENSE_FP8_PATCH_HOST|[glm53-dense-fp8]|$main_guard"\n',
  '        "$TF_BUNDLE_PATCH_HOST|[glm53-tf-bundle]|$main_guard"\n'),
 ('    [ -f "$DENSE_FP8_PATCH_HOST" ] || die "$DENSE_FP8_PATCH_HOST missing"\n',
  '    [ -f "$TF_BUNDLE_PATCH_HOST" ] || die "$TF_BUNDLE_PATCH_HOST missing"  # [tf-exl3-fork]\n'
  '    [ -d "$TF_BUNDLE_DIR_HOST/site" ] || die "$TF_BUNDLE_DIR_HOST/site missing"\n'),
 ('    scp -q -o BatchMode=yes "$DENSE_FP8_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_dense_fp8.py"\n',
  '    [ -f "$TF_BUNDLE_PATCH_HOST" ] || die "missing $TF_BUNDLE_PATCH_HOST"  # [tf-exl3-fork]\n'
  '    scp -q -o BatchMode=yes "$TF_BUNDLE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_tf_bundle.py"\n'
  '    worker_ssh "rm -rf /tmp/glm53-tf"\n'
  '    scp -q -r -o BatchMode=yes "$TF_BUNDLE_DIR_HOST" "${WORKER_SSH}:/tmp/glm53-tf"\n'),
 ('    patch_dense_fp8.py\n', '    patch_tf_bundle.py\n'),
 ("        -v '/tmp/patch_dense_fp8.py:/opt/glm53/patch_dense_fp8.py:ro' \\\n",
  "        -v '/tmp/patch_tf_bundle.py:/opt/glm53/patch_tf_bundle.py:ro' \\\n"
  "        -v '/tmp/glm53-tf:/opt/glm53/tf:ro' \\\n"),
 ('        -v "$DENSE_FP8_PATCH_HOST:/opt/glm53/patch_dense_fp8.py:ro" \\\n',
  '        -v "$TF_BUNDLE_PATCH_HOST:/opt/glm53/patch_tf_bundle.py:ro" \\\n'
  '        -v "$TF_BUNDLE_DIR_HOST:/opt/glm53/tf:ro" \\\n'),
]
rep = [('             GLM53_COOP_GEOMETRY; do\n',
        '             GLM53_COOP_GEOMETRY \\\n'
        '             TF_EXL3_MOE TF_EXL3_TOKENS GLM53_SPEC_RESAMPLE_INDEPENDENT GLM53_DRAFT_FP8 GLM53_DRAFT_LMHEAD_FP8; do  # [tf-exl3-fork]\n')]
s = s0
for a, t in ins:
    n = s.count(a)
    if n != 1: sys.exit(f"ABORT: anchor found {n}x (need 1): {a.strip()[:90]}")
    s = s.replace(a, a + t, 1)
for a, t in rep:
    n = s.count(a)
    if n != 1: sys.exit(f"ABORT: anchor found {n}x (need 1): {a.strip()[:90]}")
    s = s.replace(a, t, 1)
d = list(difflib.unified_diff(s0.splitlines(True), s.splitlines(True), "start.sh", "start.sh+tf", n=0))
sys.stdout.writelines(d)
added = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
removed = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
print(f"\n== +{added} / -{removed} lines")
if "--apply" in sys.argv:
    bak = P + ".bak-tf-" + time.strftime("%Y%m%d-%H%M%S")
    open(bak, "w").write(s0); os.chmod(bak, 0o644)
    tmp = P + ".tmp-tf"; open(tmp, "w").write(s); os.chmod(tmp, os.stat(P).st_mode); os.replace(tmp, P)
    print(f"applied; backup: {bak}")
else:
    print("dry run (no changes). re-run with --apply")
