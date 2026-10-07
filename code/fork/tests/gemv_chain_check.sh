#!/usr/bin/env bash
# Apply the launcher's runtime overlay chain (GLM53_OVERLAY_ORDER of <tree>/start.sh, the in-container
# `python3 /opt/glm53/<patch>` loop) to the image's files, then print the fingerprints of the production functions
# GLM53_BF16_GEMV relies on (tools/gemv_fingerprints.py). They must equal the ones glm53_gemv_install.py was
# verified against, i.e. the launcher chain does not touch them. CPU only, --network none, --rm.
# Usage: tests/gemv_chain_check.sh <launcher tree> <env file (KEY=VALUE lines, no secrets)>
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="${1:?launcher tree}"; ENVF="${2:?env file}"
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$TREE/start.sh")
[ -n "$order" ] || { echo "no GLM53_OVERLAY_ORDER in $TREE/start.sh"; exit 2; }
echo "image $(docker image inspect "$IMG" --format '{{.Id}}' | cut -c8-19); tree $TREE; order: $(echo $order)"
args=(-v "$TREE/overlay/exl3.py:/opt/glm53/exl3.py:ro" -v "$REPO:/w:ro")
for f in "$TREE"/overlay/*.py; do args+=(-v "$f:/opt/glm53/$(basename "$f"):ro"); done
docker run --rm --network none --memory 16g --name "tf-exl3-chain-$$" --env-file "$ENVF" -e PYTHONDONTWRITEBYTECODE=1 \
  -e ORDER="$order" "${args[@]}" --entrypoint bash "$IMG" -c '
fail=0
for p in $ORDER; do
  if [ -f /opt/glm53/$p ]; then out=$(python3 /opt/glm53/$p 2>&1); rc=$?; echo "rc=$rc $p"; [ $rc -ne 0 ] && { echo "$out" | tail -n 3; fail=1; }
  else echo "absent $p"; fi
done
python3 /w/tools/gemv_fingerprints.py 2>&1 | grep -v "WARNING\|W0928\|warn"
exit $fail'
