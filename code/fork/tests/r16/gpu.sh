#!/usr/bin/env bash
# [deploy-r16] tests/gpu_run.sh under the shared GPU lock of nodeC (every GPU job serializes its GPU work through
# /tmp/tf-gpu-bench.lock), retried while another tf-exl3 GPU container runs (gpu_run exit 4) or host memory is short
# (exit 3). Mounts / env come from the caller (GPU_RUN_BIND, GPU_RUN_BIND_DIR, GPU_RUN_RO, GPU_RUN_ENV,
# GPU_RUN_ENV_EXTRA). Per-command timeout GPU_TIMEOUT (default 3600 s).
# Usage: tests/r16/gpu.sh python3 -u tests/foo.py [args]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${GPU_LOCK:-/tmp/tf-gpu-bench.lock}"
for i in $(seq 1 "${GPU_RETRY_MAX:-120}"); do
  flock "$LOCK" timeout "${GPU_TIMEOUT:-3600}" "$REPO/tests/gpu_run.sh" "$@"
  rc=$?
  [ "$rc" -ne 4 ] && [ "$rc" -ne 3 ] && exit "$rc"
  sleep 20
done
echo "r16/gpu.sh: gave up waiting for the GPU" >&2
exit 99
