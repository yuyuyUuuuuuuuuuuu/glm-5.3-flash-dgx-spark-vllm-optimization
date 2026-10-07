#!/usr/bin/env bash
# FKDA GPU runner: production image + this repo (/w) + both FlashKDA builds on
# PYTHONPATH: pfkda's 16-arg _flashkda_C (/pf3000/out) and the fp32-renamed
# _flashkda_fp32_C (/fkda/out). Same host rules as tests/pf3000/gpu_run_pf3000.sh
# (>= 40 GB available, one container at a time, --rm, --network none) and the
# same nodeC gpu-guard (torch CUDA cap GPU_MEM_CAP_GB=40, --memory 64g).
# Call it under the GPU lock, around the docker run only:
#   flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh python3 ...
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
GUARD="${GPU_GUARD_DIR:-$REPO/tests/gpu_guard}"
[ -d "$GUARD" ] || { echo "gpu_run.sh: gpu-guard dir $GUARD missing" >&2; exit 5; }
avail=$(free -g | awk 'NR==2{print $7}')
if [ "${avail}" -lt 40 ]; then
  echo "gpu_run.sh: host available memory ${avail} GB < 40 GB, refusing" >&2
  exit 3
fi
if [ -n "$(docker ps -q --filter name=tf-exl3-gpu-)" ] \
   || [ -n "$(docker ps -q --filter name=pf3000-)" ] \
   || [ -n "$(docker ps -q --filter name=fkda-)" ]; then
  echo "gpu_run.sh: another GPU container is running, refusing" >&2
  exit 4
fi
# FKDA_USER=root: the unit rigs install the overlay into dist-packages, so the
# container must run as root (the launcher's containers do too); the default is
# this host's uid (outputs stay owned by the caller)
USERARG=(-u "$(id -u):$(id -g)")
[ -z "${FKDA_USER:-}" ] || USERARG=(-u "$FKDA_USER")
exec docker run --rm --gpus all --network none --name "fkda-gpu-$$" \
  "${USERARG[@]}" \
  -v "${REPO}:/w" -v "${FKDA_SCRATCH:-/tmp/fkda}:/fkda" -v "${FKDA_SRC:-/tmp/pf3000}:/pf3000" -w /w \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/tmp/ext \
  -e PYTHONPATH="/pf3000/out:/fkda/out:/opt/gpuguard" \
  -e GPU_MEM_CAP_GB="${GPU_MEM_CAP_GB:-40}" -e GPU_GUARD_VERBOSE="${GPU_GUARD_VERBOSE:-1}" \
  -v "${GUARD}:/opt/gpuguard:ro" \
  --memory "${GPU_RUN_MEM:-64g}" \
  --entrypoint bash "${IMAGE}" "$@"
