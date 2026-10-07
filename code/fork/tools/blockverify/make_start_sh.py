#!/usr/bin/env python3
"""blockverify launcher start.sh = the deploy-r16 kit's start.sh (production's current, sha256 b721de12...) + the
GLM53_REJECTION_METHOD switch:
  E1 the DFlash2 --speculative-config JSON, in BOTH inner scripts (head + worker), takes
     "rejection_sample_method" from GLM53_REJECTION_METHOD (unset/empty -> "standard" = today's JSON byte for byte;
     anything but standard|block raises inside the container before vllm serve starts)
  E2 validate_numeric_config: empty/unset = standard; otherwise exactly standard|block; block additionally needs
     SPEC_METHOD=dflash, GLM53_SPEC_RESAMPLE_INDEPENDENT=1, overlay/tf/overlay/patch_spec_block_keys.py and a
     patch_tf_bundle.py that runs it (a partial install would otherwise run stock, biased block mode)
  E3 head docker run: -e GLM53_REJECTION_METHOD="${GLM53_REJECTION_METHOD:-}" right after GLM53_SPEC_RESAMPLE_INDEPENDENT
  E4 worker serve_env_names loop: GLM53_REJECTION_METHOD right after the GLM53_SPEC_RESAMPLE_INDEPENDENT line
  E5 a note next to the other [tf-exl3-fork] passthrough notes
Every anchor must occur the stated number of times or nothing is written; the result must pass `bash -n`, differ from
the input only by these edits, and contain the knob exactly once in the head -e list and once in the worker list.
Usage: make_start_sh.py <kit start.sh> <out start.sh>"""
import difflib
import hashlib
import re
import subprocess
import sys
from pathlib import Path

KIT_SHA = "b721de12e341cb698556bb6543a89a34eca70d8d0ef285812b4f04636b4c5261"
KNOB = "GLM53_REJECTION_METHOD"

E1_OLD = '"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}\n'
E1_NEW = ('"draft_sample_method":"probabilistic","rejection_sample_method":(os.environ.get("GLM53_REJECTION_METHOD") '
          'or "standard").strip()}\n'
          'if spec["rejection_sample_method"] not in ("standard","block"):\n'
          '    raise SystemExit("GLM53_REJECTION_METHOD must be standard or block (got: %r)" % '
          'os.environ.get("GLM53_REJECTION_METHOD"))\n')
E2_OLD = "    _glm53_validate_mixed_prefill || return\n"
E2_NEW = """    # [blockverify] rejection sampling of the DFlash2 --speculative-config, on both ranks (docs/BLOCK_VERIFY.md).
    # Unset/empty = standard (today). block is exact only with the resample-noise fix and the row-keyed DFlash2
    # drafter/verifier (overlay/tf/overlay/patch_spec_block_keys.py, run by patch_tf_bundle.py).
    if [ -n "${GLM53_REJECTION_METHOD:-}" ]; then
        _glm53_validate_enum GLM53_REJECTION_METHOD "$GLM53_REJECTION_METHOD" standard block || return
    fi
    if [ "${GLM53_REJECTION_METHOD:-}" = "block" ]; then
        if [ "$SPEC_METHOD" != "dflash" ]; then
            echo "GLM53_REJECTION_METHOD=block requires SPEC_METHOD=dflash (got: $SPEC_METHOD)" >&2
            return 2
        fi
        if [ "${GLM53_SPEC_RESAMPLE_INDEPENDENT:-}" != "1" ]; then
            echo "GLM53_REJECTION_METHOD=block requires GLM53_SPEC_RESAMPLE_INDEPENDENT=1 (got: ${GLM53_SPEC_RESAMPLE_INDEPENDENT:-unset})" >&2
            return 2
        fi
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_spec_block_keys.py" ]; then
            echo "GLM53_REJECTION_METHOD=block requires $TF_BUNDLE_DIR_HOST/overlay/patch_spec_block_keys.py" >&2
            return 2
        fi
        # a pre-blockverify patch_tf_bundle.py never runs the overlay: the ranks would verify with stock block mode,
        # which is biased across steps (docs/BLOCK_VERIFY.md section 0)
        if ! grep -qF '("patch_spec_block_keys.py", "GLM53_REJECTION_METHOD")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_REJECTION_METHOD=block requires $TF_BUNDLE_PATCH_HOST to run patch_spec_block_keys.py (the blockverify patch_tf_bundle.py)" >&2
            return 2
        fi
    fi
    _glm53_validate_mixed_prefill || return
"""
E3_OLD = '        -e GLM53_SPEC_RESAMPLE_INDEPENDENT="${GLM53_SPEC_RESAMPLE_INDEPENDENT:-}" \\\n'
E3_NEW = E3_OLD + '        -e GLM53_REJECTION_METHOD="${GLM53_REJECTION_METHOD:-}" \\\n'
E4_OLD = ("             TF_EXL3_MOE TF_EXL3_TOKENS GLM53_SPEC_RESAMPLE_INDEPENDENT GLM53_DRAFT_FP8 "
          "GLM53_DRAFT_LMHEAD_FP8 \\\n")
E4_NEW = E4_OLD + "             GLM53_REJECTION_METHOD \\\n"
E5_OLD = "# [tf-exl3-fork] bundle: TensorFold EXL3 MoE kernels + resample-noise fix + drafter FP8 (inert unless its knobs are set)\n"
E5_NEW = ("# [blockverify] GLM53_REJECTION_METHOD=standard|block (unset/empty = standard) -> rejection_sample_method of the\n"
          "#   DFlash2 --speculative-config in BOTH inner scripts; forwarded to both ranks (head -e after\n"
          "#   GLM53_SPEC_RESAMPLE_INDEPENDENT, worker serve_env_names). block also runs the bundle overlay\n"
          "#   patch_spec_block_keys.py (patch_tf_bundle.py) on both ranks. docs/BLOCK_VERIFY.md of tf-exl3-fork.\n") + E5_OLD
EDITS = (("E1 spec JSON", E1_OLD, E1_NEW, 2), ("E2 validation", E2_OLD, E2_NEW, 1), ("E3 head -e", E3_OLD, E3_NEW, 1),
         ("E4 worker list", E4_OLD, E4_NEW, 1), ("E5 note", E5_OLD, E5_NEW, 1))


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    s0 = src.read_bytes()
    if hashlib.sha256(s0).hexdigest() != KIT_SHA:
        sys.exit(f"ABORT: {src} is not the deploy-r16 kit start.sh {KIT_SHA[:16]}")
    s = s0.decode()
    if re.search(rf"\b{KNOB}\b", s):
        sys.exit(f"ABORT: {KNOB} already in {src}")
    for name, old, _new, n in EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: {name}: anchor found {c}x (need {n})")
    s2 = s
    for _name, old, new, n in EDITS:
        s2 = s2.replace(old, new, n)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=s2.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n: {r.stderr.decode()}")
    head = s2.count(f'        -e {KNOB}="${{{KNOB}:-}}" \\\n')
    worker = s2[s2.index("local -a serve_env_names=()"):s2.index('serve_env+=" -e $v=')].count(KNOB)
    spec = s2.count('os.environ.get("GLM53_REJECTION_METHOD") or "standard"')
    if (head, worker, spec) != (1, 1, 2):
        sys.exit(f"ABORT: knob placement head={head} worker={worker} spec={spec} (need 1, 1, 2)")
    # only the edits changed: removing the new text gives the input back
    back = s2
    for _name, old, new, n in EDITS:
        back = back.replace(new, old, n)
    if back != s:
        sys.exit("ABORT: result differs from the input outside the edits")
    out.write_text(s2)
    out.chmod(0o755)
    diff = "".join(difflib.unified_diff(s.splitlines(True), s2.splitlines(True), "a/start.sh", "b/start.sh", n=3))
    out.with_name("start.sh.blockverify.patch").write_text(diff)
    print(f"wrote {out} (sha256 {hashlib.sha256(s2.encode()).hexdigest()[:16]}) and start.sh.blockverify.patch; "
          f"bash -n OK; {KNOB}: head -e x{head}, worker list x{worker}, spec JSON x{spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
