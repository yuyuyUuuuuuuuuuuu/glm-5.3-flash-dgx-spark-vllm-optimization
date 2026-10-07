#!/usr/bin/env bash
# Build glm53_moe_e4m3_ext in the production image WITHOUT a GPU (no GPU lock needed): compile only.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
exec docker run --rm --network none --name "opt-moe-build-$$" -u "$(id -u):$(id -g)" -v "$REPO:/w" -w /w \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_CUDA_ARCH_LIST=12.1a -e MAX_JOBS=4 --memory 16g \
  --entrypoint python3 "$IMAGE" tools/moee4m3/build.py "$@"
