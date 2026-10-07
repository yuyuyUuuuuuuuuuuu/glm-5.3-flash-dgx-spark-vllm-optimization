#!/usr/bin/env bash
# [dec-hostloop] run one command in the production image with the production-mounted vLLM/flashinfer overrides
# (flashinfer 0.6.18, patched cuda.py, patched flashinfer_mla_sparse_sm90.py = what glm53-exl3-head/worker mount),
# serialized with other GPU jobs through /tmp/tf-gpu-bench.lock and retried while another tf-exl3 GPU container runs.
# Usage: tests/hostloop_gpu.sh python3 tests/test_hostloop.py [args]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP=/usr/local/lib/python3.12/dist-packages
A=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}
export GPU_RUN_BIND="${GPU_RUN_BIND-$A/vllm-patches/cuda.py.exl3.patched=$SP/vllm/platforms/cuda.py;$A/vllm-patches/flashinfer_mla_sparse_sm90.py.patched=$SP/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py}"
export GPU_RUN_BIND_DIR="${GPU_RUN_BIND_DIR-$A/fi618/flashinfer=$SP/flashinfer;$A/fi618/flashinfer_cubin=$SP/flashinfer_cubin}"
export GPU_RUN_ENV="${GPU_RUN_ENV-FLASHINFER_DISABLE_VERSION_CHECK=1}"
for i in $(seq 1 90); do
  flock /tmp/tf-gpu-bench.lock timeout "${HOSTLOOP_TIMEOUT:-1200}" "$REPO/tests/gpu_run.sh" "$@"
  rc=$?
  [ "$rc" -ne 4 ] && [ "$rc" -ne 3 ] && exit "$rc"
  sleep 20
done
echo "hostloop_gpu: gave up waiting for the GPU" >&2; exit 99
