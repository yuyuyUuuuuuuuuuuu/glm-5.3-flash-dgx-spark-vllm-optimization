#!/usr/bin/env bash
# opt-w8a8layers: one engine boot of tests/w8a8layers/dvp_driver.py on the handoff MoE mini (TP=2 rank-0 shapes),
# production prefill config (MoE e4m3, FlashKDA, MLA prefill, quick wins, fused cap) + GLM53_DENSE_W8A8=1.
# run.sh takes /tmp/tf-gpu-bench.lock itself (do not wrap in flock).
# Usage: tests/w8a8layers/run_dvp.sh <label> [NAME=value ...] -- <dvp_driver args>
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-w8a8layers/runs}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2rev}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-tp2r0}"
lab="$1"; shift
BASE="GLM53_MOE_E4M3=1 GLM53_PREFILL_FUSED_CAP=1 GLM53_KDA_FLASHKDA=1 GLM53_MLA_PREFILL=1 GLM53_PREFILL_QUICKWINS=all GLM53_DENSE_W8A8=1"
exec "$R/tests/handoff/run.sh" "$OUT" "$lab" HANDOFF_DRIVER=/w/tests/w8a8layers/dvp_driver.py $BASE "$@"
