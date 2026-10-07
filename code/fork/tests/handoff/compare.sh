#!/usr/bin/env bash
# tests/handoff/compare.py in a CPU-only throwaway container of the production image (the host has no torch).
# Usage: tests/handoff/compare.sh <ref run dir> <run dir> [compare.py args]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A="$(cd "$1" && pwd)"; B="$(cd "$2" && pwd)"; shift 2
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
exec docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/w:ro" -v "$A:/ref:ro" -v "$B:/run:ro" --entrypoint python3 "$IMG" /w/tests/handoff/compare.py /ref /run "$@"
