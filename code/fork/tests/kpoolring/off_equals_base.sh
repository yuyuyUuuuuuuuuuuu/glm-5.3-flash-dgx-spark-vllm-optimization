#!/usr/bin/env bash
# Default-off check: with GLM53_KPOOL_RING unset, the production overlay chain must leave every vllm/*.py of the
# image byte-identical to what the same chain produces with a base ref's bundle overlay dir (default deploy-r15, what
# production runs). Runs tests/kpoolring/chain_overlays.sh off twice (this worktree, then `git archive <base> overlay`
# in a temp dir) and diffs the "files under vllm/ changed by the chain" sha256 lists. CPU only, no production access.
# Usage: tests/kpoolring/off_equals_base.sh [base ref, default deploy-r15] [launcher dir]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${1:-deploy-r15}"; L="${2:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/base/tests"
git -C "$REPO" archive "$BASE" overlay | tar -x -C "$tmp/base"
cp -r "$REPO/tests/kpoolring" "$tmp/base/tests/"
git -C "$tmp/base" init -q
lists() { awk '/files under vllm\/ changed by the chain/{f=1} /== chain run 2/{f=0} f' "$1"; }
bash "$REPO/tests/kpoolring/chain_overlays.sh" off "$L" > "$tmp/head.log" 2>&1; rc_head=$?
bash "$tmp/base/tests/kpoolring/chain_overlays.sh" off "$L" > "$tmp/base.log" 2>&1; rc_base=$?
echo "worktree $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo +dirty) vs base $BASE ($(git -C "$REPO" rev-parse --short "$BASE"))"
echo "chain off exit: worktree $rc_head, base $rc_base"
for f in head base; do grep -E "^\s+\[glm53-tf-bundle\]" "$tmp/$f.log" | head -n 6 | sed "s/^/  $f: /"; done
lists "$tmp/head.log" > "$tmp/head.sha"; lists "$tmp/base.log" > "$tmp/base.sha"
if [ "$rc_head" -eq 0 ] && [ "$rc_base" -eq 0 ] && [ -s "$tmp/head.sha" ] && cmp -s "$tmp/head.sha" "$tmp/base.sha"; then
  echo "default-off: $(head -n1 "$tmp/head.sha" | sed 's/== //') -- identical sha256 for every one of them"
  sed 's/^/  /' "$tmp/head.sha" | tail -n +2
  echo "RESULT: OFF == $BASE (byte-identical)"; exit 0
fi
echo "RESULT: DIFFERENT"; diff "$tmp/base.sha" "$tmp/head.sha"; exit 1
