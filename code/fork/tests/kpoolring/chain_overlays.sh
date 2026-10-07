#!/usr/bin/env bash
# Replay the production launcher's runtime overlay chain (GLM53_OVERLAY_ORDER of the launcher start.sh, the exact
# in-container `python3 /opt/glm53/<patch>` loop both ranks run) on a fresh --rm container of the production image,
# with THIS worktree's overlay/ as the bundle's overlay dir (site/ and tools/ from the launcher's shipped bundle) and the
# production env (env_nonsecret.txt) plus GLM53_KPOOL_RING=<1|empty>. Then:
#   - every launcher patch exits 0 (patch_tf_bundle.py prints each sub-patch's line);
#   - unified diff of the three ring targets + block_table.py against the pristine image files;
#   - the whole chain a second time: every patch exits 0 and no file under vllm/ changes (idempotent);
#   - tests/kpoolring/chain_check.py (imports incl. PD connectors, tail spec at production settings, KV accounting).
# CPU only, --network none, runs as root inside the throwaway container like the launcher's inner script does.
# Nothing is written outside the container except stdout. No production access.
# Usage: tests/kpoolring/chain_overlays.sh <on|off> [launcher dir, default ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:?on|off}"; L="${2:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher}"
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
case "$MODE" in on) RING=1;; off) RING=;; *) echo "mode must be on|off"; exit 2;; esac
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$L/start.sh")
[ -n "$order" ] || { echo "no GLM53_OVERLAY_ORDER in $L/start.sh"; exit 2; }
envf=$(mktemp); trap 'rm -f "$envf"' EXIT
# KEY=VALUE lines only, inline comments stripped (docker --env-file keeps them verbatim); never printed
grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$L/env_nonsecret.txt" | sed -E 's/[[:space:]]+#.*$//' | grep -v '^GLM53_KPOOL_RING=' > "$envf"
echo "image $(docker image inspect "$IMG" --format '{{.Id}}' | cut -c8-23); mode=$MODE GLM53_KPOOL_RING='$RING'"
echo "launcher $L: start.sh $(sha256sum "$L/start.sh" | cut -c1-16); env vars: $(cut -d= -f1 "$envf" | tr '\n' ' ')"
echo "worktree $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo +dirty); bundle overlay dir:"
for f in "$REPO"/overlay/*.py; do echo "  $(sha256sum "$f" | cut -c1-16) $(basename "$f")"; done
args=(-v "$L/overlay/exl3.py:/opt/glm53/exl3.py:ro" -v "$L/overlay/tf:/opt/glm53/tf:ro"
      -v "$REPO/overlay:/opt/glm53/tf/overlay:ro" -v "$REPO/overlay/patch_tf_bundle.py:/opt/glm53/patch_tf_bundle.py:ro"
      -v "$REPO:/w:ro")
for p in $order; do
  [ "$p" = patch_tf_bundle.py ] && continue
  [ -f "$L/overlay/$p" ] && args+=(-v "$L/overlay/$p:/opt/glm53/$p:ro")
done
docker run --rm --network none --memory 16g --name "tf-exl3-kpoolring-chain-$$" --env-file "$envf" \
  -e GLM53_KPOOL_RING="$RING" -e KPOOLRING_MODE="$MODE" -e ORDER="$(echo $order)" -e PYTHONDONTWRITEBYTECODE=1 \
  "${args[@]}" --entrypoint bash "$IMG" -c '
SP=/usr/local/lib/python3.12/dist-packages
T="vllm/models/glm5next/nvidia/attention.py vllm/models/glm5next/nvidia/ops/kpool_compress.py vllm/model_executor/layers/sparse_attn_indexer_kpool.py vllm/v1/worker/block_table.py"
mkdir -p /tmp/pristine; for f in $T; do mkdir -p /tmp/pristine/$(dirname $f); cp $SP/$f /tmp/pristine/$f; done
fail=0
run_chain() {
  for p in $ORDER; do
    if [ -f /opt/glm53/$p ]; then out=$(python3 /opt/glm53/$p 2>&1); rc=$?
      case $p in patch_tf_bundle.py) echo "rc=$rc $p"; echo "$out" | grep -v "installed .* entries" | sed "s/^/    /";; *) echo "rc=$rc $p";; esac
      [ $rc -ne 0 ] && { echo "$out" | tail -n 5 | sed "s/^/    /"; fail=1; }
    else echo "absent $p"; fi
  done
}
echo "== chain run 1"; run_chain
for f in $T; do echo "== diff pristine -> after chain: $f ($(sha256sum /tmp/pristine/$f | cut -c1-12) -> $(sha256sum $SP/$f | cut -c1-12))"; diff -u /tmp/pristine/$f $SP/$f | sed "1,2d"; done
(cd $SP && find vllm -name "*.py" -newer /tmp/pristine -type f | sort | xargs sha256sum) > /tmp/after1.sha
echo "== files under vllm/ changed by the chain: $(wc -l < /tmp/after1.sha)"; sed "s/^/    /" /tmp/after1.sha
(cd $SP && find vllm -name "*.py" -type f | sort | xargs sha256sum) > /tmp/all1.sha
echo "== chain run 2 (idempotency)"; run_chain
(cd $SP && find vllm -name "*.py" -type f | sort | xargs sha256sum) > /tmp/all2.sha
if cmp -s /tmp/all1.sha /tmp/all2.sha; then echo "second run: vllm/*.py byte-identical ($(wc -l < /tmp/all2.sha) files)"; else echo "second run CHANGED files:"; diff /tmp/all1.sha /tmp/all2.sha; fail=1; fi
echo "== chain_check.py"; cd /tmp && python3 /w/tests/kpoolring/chain_check.py 2>&1 | grep -vE "^(W|I)[0-9]{4} |Warning|warn\(" ; [ ${PIPESTATUS[0]} -ne 0 ] && fail=1
echo "chain result mode=$KPOOLRING_MODE: $([ $fail -eq 0 ] && echo ALL OK || echo FAILED)"
exit $fail'
