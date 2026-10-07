#!/usr/bin/env bash
# GLM53_SPEC_VTRIM on the REAL engine (tests/handoff/run.sh: production image, the launcher overlay chain, this
# worktree's bundle, the EXL3-MoE handoff mini model + the real DFlash2 drafter, adaptive K {4,5,7}), greedy:
#   off_a, off_b   : the feature unset (twice: the cross-process noise floor of the greedy outputs)
#   shadow30       : GLM53_SPEC_VTRIM=shadow (log only; outputs must match off)
#   on30           : GLM53_SPEC_VTRIM=on TAU 0.3
#   on100          : GLM53_SPEC_VTRIM=on TAU 1.0 (n* = 0 every step: only the anchor row is verified)
# Usage: tests/vtrim/run_engine_ab.sh <out dir> [labels...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"; shift
LABELS="${*:-off_a off_b shadow30 on30 on100 syn_off syn_on30}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-full}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z6rev}"
ARGS=(-- --model "$HANDOFF_MODEL" --prompts "${VT_PROMPTS:-700,1500,3001}" --gen "${VT_GEN:-96}" --shadow 0 --topk-probe 0 --mla-probe 0 --kv-bytes 2147483648)
for l in $LABELS; do
  ARGS_L=()
  case $l in
    off_g64) KV=(); ARGS_L=(--gen 64);;   # base, shorter generations: does request i's prefill depend on request i-1's decode?
    off_*) KV=();;
    shadow30) KV=(GLM53_SPEC_VTRIM=shadow GLM53_SPEC_VTRIM_TAU=0.3 GLM53_SPEC_VTRIM_LOG=40);;
    on30) KV=(GLM53_SPEC_VTRIM=on GLM53_SPEC_VTRIM_TAU=0.3 GLM53_SPEC_VTRIM_LOG=40);;
    on100*) KV=(GLM53_SPEC_VTRIM=on GLM53_SPEC_VTRIM_TAU=1.0 GLM53_SPEC_VTRIM_LOG=40);;
    # vLLM's synthetic acceptance (drafts accepted at these per-position rates regardless of the token): the mini
    # model accepts ~0 drafts on its own, these two exercise dead rows that WOULD have been accepted (robustness,
    # stats, MoE skip in the captured graph; outputs are not comparable in synthetic mode)
    syn_off) KV=(HANDOFF_SPEC_SYNTHETIC=0.95,0.9,0.85,0.8,0.7,0.6,0.5);;
    syn_on30) KV=(HANDOFF_SPEC_SYNTHETIC=0.95,0.9,0.85,0.8,0.7,0.6,0.5 GLM53_SPEC_VTRIM=on GLM53_SPEC_VTRIM_TAU=0.3 GLM53_SPEC_VTRIM_LOG=40);;
    *) echo "unknown label $l"; exit 2;;
  esac
  # the EXL3-MoE mini model's hybrid scheduler block is 9216 tokens: run.sh's 32256 retention interval is refused
  "$REPO/tests/handoff/run.sh" "$OUT" "$l" "VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VT_RET:-36864}" "${KV[@]}" "${ARGS[@]}" "${ARGS_L[@]}" > "$OUT/$l.log" 2>&1
  echo "$l rc=$? $(grep -c 'glm53-spec-vtrim' "$OUT/$l.log") vtrim lines"
done
