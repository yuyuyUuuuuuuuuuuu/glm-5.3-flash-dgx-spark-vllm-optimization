#!/usr/bin/env bash
# FKDA CPU runner: the production image, CPU only (no --gpus, so no GPU lock),
# this repo at /w and the FKDA scratch at /fkda. For the _flashkda_fp32_C build
# (build_flashkda_fp32.sh) and the host-side comparison dumps. Same refusal
# rules as tests/pf3000/gpu_run_pf3000.sh minus the GPU checks (host memory
# still matters: the compile peaks ~4 GB).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
SCR="${FKDA_SCRATCH:-/tmp/fkda}"
SRC="${FKDA_SRC:-/tmp/pf3000}"
mkdir -p "$SCR/out"
avail=$(free -g | awk 'NR==2{print $7}')
if [ "${avail}" -lt 20 ]; then
  echo "docker_cpu.sh: host available memory ${avail} GB < 20 GB, refusing" >&2
  exit 3
fi
# FKDA_CPU_USER=root: the install checks write into dist-packages (the
# launcher's containers run as root too); default = this host's uid
USERARG=(-u "$(id -u):$(id -g)")
[ -z "${FKDA_CPU_USER:-}" ] || USERARG=(-u "$FKDA_CPU_USER")
exec docker run --rm --network none --name "fkda-cpu-$$" \
  "${USERARG[@]}" \
  -v "${REPO}:/w" -v "${SCR}:/fkda" -v "${SRC}:/fkda/in:ro" -w /w \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/tmp/ext \
  --entrypoint bash "${IMAGE}" "$@"
