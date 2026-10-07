#!/usr/bin/env bash
# tests/gpu_run.sh with retries while another GPU container runs (exit 4): up to GPU_RETRY_MAX tries, 30-60 s apart.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
max="${GPU_RETRY_MAX:-40}"
for ((i = 1; i <= max; i++)); do
  "${REPO}/tests/gpu_run.sh" "$@"
  rc=$?
  if [ "$rc" -ne 4 ] && [ "$rc" -ne 3 ]; then exit "$rc"; fi
  sleep $((30 + RANDOM % 31))
done
echo "gpu_retry: GPU still busy after ${max} tries" >&2
exit 4
