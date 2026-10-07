#!/usr/bin/env bash
# The prefill -> decode state-handoff check (docs/HANDOFF.md). Three engine runs of the production image on nodeC
# (tests/handoff/run.sh: production-composed container, --network none, flock /tmp/tf-gpu-bench.lock, < 16 GiB GPU),
# each asserted by tests/handoff/check.py:
#   qw_all       GLM53_PREFILL_QUICKWINS=all, prompts 200 / 14400 (a 13824-token chunk without initial state, so
#                idx_gate's M >= 10240 path runs, then a 576-token chunk with initial state) / 3001 (one chunk):
#                every item bit-identical to production in outputs, every KV byte and the side buffers
#   r16_prefill  GLM53_PREFILL_QUICKWINS=all + GLM53_MLA_PREFILL=1, prompts 200 / 6000 (4608 + 1392) / 3001: the
#                kernel leaves the FA2 wrapper's kv_indices exactly as production does, stays within 0.5 % of fp32,
#                and decode's past-the-step reads (production's FA2 plan) only see the request's own slots
#   mla_defect   the same with the deploy-r16 behaviour (--mla-kv-indices 0): the check must see foreign slots
# The 200-token request goes through production FA2 (< 256 tokens) and leaves its slots in the process-wide
# kv_indices, as other traffic does in production. Needs the mini model (python3 tests/handoff/build_mini.py).
# Usage: tests/handoff/check.sh [out dir, default tests/logs/handoff]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO}/tests/logs/handoff}"
mkdir -p "$OUT"
cd "$REPO"
rc=0
one() {   # one <label> <check mode> <check args> -- <run.sh args...>
  local label="$1" mode="$2" cargs="$3"; shift 4
  rm -rf "${OUT:?}/${label}"
  echo "=== handoff ${label} ($(date +%H:%M:%S))"
  if ! tests/handoff/run.sh "$OUT" "$label" "$@"; then echo "--- handoff ${label}: RUN FAILED"; rc=1; return; fi
  # shellcheck disable=SC2086
  python3 tests/handoff/check.py "$mode" "$OUT/$label" $cargs || rc=1
}
one qw_all exact "--expect-fast kda_conv_fast,mla_bmm_fast,mla_index_fast,mhc_aux_reuse,mhc_mean_aux_fast,mhc_mean_final_fast,idx_gate_fast" -- \
  GLM53_PREFILL_QUICKWINS=all -- --prompts 200,14400,3001 --gen 16 --mla-probe 0
one r16_prefill mla "--expect-fast kda_conv_fast,mla_bmm_fast,mhc_aux_reuse,mhc_mean_aux_fast,mhc_mean_final_fast,mla_prefill_calls" -- \
  GLM53_PREFILL_QUICKWINS=all GLM53_MLA_PREFILL=1 -- --prompts 200,6000,3001 --gen 24
one mla_defect mla-defect "" -- GLM53_MLA_PREFILL=1 -- --prompts 200,6000,3001 --gen 24 --mla-kv-indices 0
echo "handoff check: $([ $rc = 0 ] && echo ALL PASSED || echo FAILED) (runs in $OUT)"
exit $rc
