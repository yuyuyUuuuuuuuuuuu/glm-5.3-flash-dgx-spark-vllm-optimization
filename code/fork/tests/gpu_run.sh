#!/usr/bin/env bash
# Run one command inside the production image with the GPU, under nodeC's host rules:
#   - refuse unless >= 40 GB of host memory is available (free -g, "available" column)
#   - refuse if another tf-exl3 GPU container is already running (one at a time)
#   - --rm, --network none, no privileges, only this repo mounted at /w
# Usage: tests/gpu_run.sh python3 tests/foo.py [args]    |    tests/gpu_run.sh bash -c '...'
#   GPU_RUN_RO="/dir1:/dir2" additionally mounts those host directories read-only at the same path
#   (downloaded weights under ${TF_EXL3_MODELS:-$HOME/models}/, never the repo's parent).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=paths.sh
source "${REPO}/tests/paths.sh"   # TF_EXL3_MODELS / TF_EXL3_ASSETS / TF_EXL3_KITS, passed into the container below
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
avail=$(free -g | awk 'NR==2{print $7}')
if [ "${avail}" -lt 40 ]; then
  echo "gpu_run: host available memory ${avail} GB < 40 GB, refusing to start a GPU container" >&2
  exit 3
fi
if [ -n "$(docker ps -q --filter name=tf-exl3-gpu-)" ]; then
  echo "gpu_run: another tf-exl3 GPU container is running, refusing (one at a time)" >&2
  exit 4
fi
entry="$1"; shift
ro=()
IFS=: read -r -a ro_dirs <<< "${GPU_RUN_RO:-}"
for d in "${ro_dirs[@]}"; do
  [ -n "$d" ] || continue
  [ -d "$d" ] || { echo "gpu_run: GPU_RUN_RO entry $d is not a directory" >&2; exit 5; }
  ro+=(-v "$d:$d:ro")
done
# GPU_RUN_BIND="host_path=container_path;..." bind files or directories read-only at another path
# (used to test against the LIVE production overlay, e.g. overlay exl3.py over the image's quantization/exl3.py)
IFS=';' read -r -a binds <<< "${GPU_RUN_BIND:-}"
for b in "${binds[@]}"; do
  [ -n "$b" ] || continue
  src="${b%%=*}"; dst="${b#*=}"
  [ -e "$src" ] || { echo "gpu_run: GPU_RUN_BIND source $src does not exist" >&2; exit 6; }
  ro+=(-v "$src:$dst:ro")
done
# GPU_RUN_BIND_DIR="host_dir=container_path;..." bind directories read-only at another path (production mounts
# flashinfer 0.6.18 over site-packages with SM90_KV=1); GPU_RUN_ENV="NAME=value;..." extra container env
IFS=';' read -r -a dbinds <<< "${GPU_RUN_BIND_DIR:-}"
for b in "${dbinds[@]}"; do
  [ -n "$b" ] || continue
  src="${b%%=*}"; dst="${b#*=}"
  [ -d "$src" ] || { echo "gpu_run: GPU_RUN_BIND_DIR source $src is not a directory" >&2; exit 7; }
  ro+=(-v "$src:$dst:ro")
done
# GPU_RUN_ENV="K=V;K2=V2" passes extra environment variables (e.g. FLASHINFER_DISABLE_VERSION_CHECK=1);
# GPU_RUN_ENV_EXTRA (same syntax) is appended after it, so a flag set (tests/r16/flags.sh) survives the mount helpers
# (tests/mla_env.sh, tests/qw/prod_env.sh) that overwrite GPU_RUN_ENV
envs=()
IFS=';' read -r -a env_kv <<< "${GPU_RUN_ENV:-};${GPU_RUN_ENV_EXTRA:-}"
for kv in "${env_kv[@]}"; do
  [ -n "$kv" ] || continue
  envs+=(-e "$kv")
done
# GPU_RUN_SHM="2g" sets --shm-size (NCCL multi-process tests need more than docker's 64 MB /dev/shm)
shm=(); [ -n "${GPU_RUN_SHM:-}" ] && shm=(--shm-size "${GPU_RUN_SHM}")
exec docker run --rm --gpus all --network none --name "tf-exl3-gpu-$$" "${shm[@]}" \
  -u "$(id -u):$(id -g)" -v "${REPO}:/w" -w /w "${ro[@]}" \
  -e TORCH_EXTENSIONS_DIR=/w/.ext -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e TF_EXL3_JIT="${TF_EXL3_JIT:-0}" "${envs[@]}" \
  -e TF_EXL3_MODELS="${TF_EXL3_MODELS}" -e TF_EXL3_ASSETS="${TF_EXL3_ASSETS}" -e TF_EXL3_KITS="${TF_EXL3_KITS}" \
  -v "${GPU_GUARD_DIR:-$REPO/tests/gpu_guard}:/opt/gpuguard:ro" -e PYTHONPATH=/opt/gpuguard -e GPU_MEM_CAP_GB="${GPU_MEM_CAP_GB:-40}" --memory "${GPU_RUN_MEM:-64g}" \
  --entrypoint "${entry}" "${IMAGE}" "$@"
