#!/usr/bin/env bash
# Run tests/blockverify/test_block_exactness.py config by config on nodeC (production image, <= 8 GiB, under
# /tmp/tf-gpu-bench.lock via tests/r16/gpu.sh, released between configs). Logs + JSON in docs/logs/blockverify/.
# Usage: tests/blockverify/run_exactness.sh [config ...]   (default: prod stress textbook vocab)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$REPO/docs/logs/blockverify"
mkdir -p "$LOG"
rc_all=0
for cfg in "${@:-prod stress textbook vocab}"; do
  for c in $cfg; do
    GPU_RUN_ENV="BLOCKVERIFY_SCALE=${BLOCKVERIFY_SCALE:-1.0};BLOCKVERIFY_ONLY=$c;BLOCKVERIFY_JSON=/w/docs/logs/blockverify/exact_$c.json" \
      GPU_TIMEOUT="${GPU_TIMEOUT:-5400}" "$REPO/tests/r16/gpu.sh" python3 -u tests/blockverify/test_block_exactness.py \
      > "$LOG/exact_$c.log" 2>&1
    rc=$?
    echo "$c rc=$rc $(tail -1 "$LOG/exact_$c.log")"
    [ "$rc" -eq 0 ] || rc_all=1
  done
done
exit "$rc_all"
