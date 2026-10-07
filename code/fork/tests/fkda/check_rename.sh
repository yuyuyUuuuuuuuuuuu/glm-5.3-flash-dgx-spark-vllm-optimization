#!/usr/bin/env bash
# FKDA check 1 driver: dump the fp32 build's outputs and pfkda's original
# _flashkda_C outputs on identical inputs, then compare them.
# Each dump is its own GPU process (two extensions, same torch.library
# namespace, cannot be imported together). Runs under the GPU lock.
#   flock /tmp/tf-gpu-bench.lock tests/fkda/check_rename.sh
# fkda2 (docs/KDA_FLASHKDA2.md): the fp32 build is no longer only a renamed
# 17a037d. The r16o kit ships the fkda2 PRECISION build (overlay/
# _flashkda_fp32_C.abi3.so sha256 dd1788c24f10af2c... = FKDA2_FP32_DECAY/U/OUT),
# which rounds the decayed q/k, u and out in fp32 once instead of the stock
# build's bf16 chains, so it deviates from pfkda's _flashkda_C by a
# bf16-rounding-class amount instead of being bit-equal. Expectations therefore
# follow the build that was dumped (sha256 of $OUT/_flashkda_fp32_C.abi3.so, the
# file PYTHONPATH=/fkda/out imports):
#   dd1788c24f10af2c... (fkda2 precision): shapes/dtypes exact, all finite, and
#     per-output rel-L2 <= $REL_L2_MAX / rel-Linf <= $REL_INF_MAX (measured on
#     these very inputs: worst rel-L2 7.5e-3, worst rel-Linf 8.6e-3; a stock
#     build is bit-equal, so anything in between means an unintended source
#     change). Bit-equality is NOT expected.
#   anything else (the stock fkda build c286213f..., or tests/fkda2
#     build_variant.sh v_stock = 5c3a9a78...): bit-equal, the r16n expectation
#     (KDA_FLASHKDA2.md 3: v_stock is bit-identical to the shipped c286213f).
# FKDA_RENAME_EXPECT=bitwise|precision overrides (auto = decide by the sha).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${FKDA_SCRATCH:-/tmp/fkda}/out"
REL_L2_MAX="${FKDA_RENAME_REL_L2_MAX:-0.03}"
REL_INF_MAX="${FKDA_RENAME_REL_INF_MAX:-0.05}"
FKDA2_SHA="dd1788c24f10af2c90c776b355ef44991b786dd8a2bc708459f81603c58cd484"
EXPECT="${FKDA_RENAME_EXPECT:-auto}"
case "$EXPECT" in
  auto)
    if [ "$(sha256sum "$OUT/_flashkda_fp32_C.abi3.so" | cut -d' ' -f1)" = "$FKDA2_SHA" ]; then
      EXPECT=precision
    else
      EXPECT=bitwise
    fi ;;
  bitwise|precision) ;;
  *) echo "FKDA_RENAME_EXPECT must be auto|bitwise|precision (got $EXPECT)" >&2; exit 2 ;;
esac
echo "check_rename: fp32 build sha256 $(sha256sum "$OUT/_flashkda_fp32_C.abi3.so" | cut -c1-16), expectation: $EXPECT"
run() { flock /tmp/tf-gpu-bench.lock "$REPO/tests/fkda/gpu_run.sh" -c "python3 /w/tests/fkda/fkda_op_dump.py --ext $1 --out /fkda/out/dump_$1.pt"; }
run fp32
run orig
# the compare needs torch (not installed on the host) -- CPU-only container, the
# helper as a FILE (docker_cpu.sh does not forward stdin: a `python3 -` heredoc
# would exit 0 without comparing anything)
"$REPO/tests/fkda/docker_cpu.sh" -c \
  "python3 /w/tests/fkda/_cmp_dump.py /fkda/out/dump_fp32.pt /fkda/out/dump_orig.pt $EXPECT $REL_L2_MAX $REL_INF_MAX"
