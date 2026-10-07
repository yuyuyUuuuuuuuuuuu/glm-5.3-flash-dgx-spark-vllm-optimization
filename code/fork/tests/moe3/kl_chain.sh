#!/usr/bin/env bash
# moe3 KL chain on the handoff MoE mini at TP=2 rank-0 shapes (real engine, production-composed container, nodeC):
#   off (reference) -> off2 (A/A) -> fused16 (GLM53_MOE_FUSED16=1) -> e4m3 (all MoE layers) -> e4m3 layer subsets
# run.sh takes /tmp/tf-gpu-bench.lock itself (do not wrap this script in flock).
# Usage: tests/moe3/kl_chain.sh [out dir] [arms...]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/moe3/kl}"
shift || true
ARMS="${*:-off off2 fused16 e4m3}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16x}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-tp2r0}"
D=HANDOFF_DRIVER=/w/tests/moe3/kl_driver.py
CAP=GLM53_PREFILL_FUSED_CAP=1
for a in $ARMS; do
  case $a in
    off)     "$REPO/tests/handoff/run.sh" "$OUT" off $D $CAP -- --save 1 ;;
    off2)    "$REPO/tests/handoff/run.sh" "$OUT" off2 $D $CAP -- --ref /out/off ;;
    fused16) "$REPO/tests/handoff/run.sh" "$OUT" fused16 $D $CAP GLM53_MOE_FUSED16=1 -- --save 1 --ref /out/off ;;
    e4m3)    "$REPO/tests/handoff/run.sh" "$OUT" e4m3 $D $CAP GLM53_MOE_E4M3=1 -- --ref /out/off ;;
    e4m3f)   "$REPO/tests/handoff/run.sh" "$OUT" e4m3f $D $CAP GLM53_MOE_E4M3=1 GLM53_MOE_FUSED16=1 -- --ref /out/off,/out/fused16 ;;
    e4m3d16) "$REPO/tests/handoff/run.sh" "$OUT" e4m3d16 $D $CAP GLM53_MOE_E4M3=1 GLM53_MOE_E4M3_DOWN=f16 -- --ref /out/off ;;
    e4m3_D*) sel="${a#e4m3_D}"; sel="${sel//_/,}"
             "$REPO/tests/handoff/run.sh" "$OUT" "$a" $D $CAP GLM53_MOE_E4M3=1 "GLM53_MOE_E4M3_DOWN_LAYERS=$sel" \
               -- --ref /out/off ;;
    e4m3_L*) sel="${a#e4m3_L}"; sel="${sel//_/,}"
             "$REPO/tests/handoff/run.sh" "$OUT" "$a" $D $CAP GLM53_MOE_E4M3=1 GLM53_MOE_FUSED16=1 \
               "GLM53_MOE_E4M3_LAYERS=$sel" -- --ref /out/off ;;
  esac
done
