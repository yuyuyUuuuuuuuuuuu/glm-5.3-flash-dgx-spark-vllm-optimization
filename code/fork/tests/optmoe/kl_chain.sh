#!/usr/bin/env bash
# opt-moe KL chain on the handoff MoE mini at TP=2 rank-0 shapes (real engine, production-composed container, nodeC;
# tests/moe3/kl_driver.py, kit r16z2 = production's): off (reference) -> e4m3 (fp32 accumulator, = production since
# 10-02) -> e4m3 again (A/A of the e4m3 arm) -> e4m3 + GLM53_MOE_E4M3_ACC=bf16 (+ its A/A e4m3bx) -> + FOLD_SHARED=1
# (e4m3bf; e4m3/e4m3b/e4m3bf ran before TOKGATHER existed = the per-pair gather) -> TOKGATHER on (default since
# then): e4m3t (fp32 acc), e4m3bft (ACC=bf16 + FOLD) -> + DOWN=f16 ...
# run.sh takes /tmp/tf-gpu-bench.lock itself (do not wrap this script in flock).
# Usage: tests/optmoe/kl_chain.sh [out dir] [arms...]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-moe/kl}"
shift || true
ARMS="${*:-off e4m3 e4m3b e4m3x e4m3bx}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-tp2r0}"
D=HANDOFF_DRIVER=/w/tests/moe3/kl_driver.py
CAP=GLM53_PREFILL_FUSED_CAP=1
E=GLM53_MOE_E4M3=1
B=GLM53_MOE_E4M3_ACC=bf16
for a in $ARMS; do
  case $a in
    off)      "$REPO/tests/handoff/run.sh" "$OUT" off $D $CAP -- --save 1 ;;
    e4m3)     "$REPO/tests/handoff/run.sh" "$OUT" e4m3 $D $CAP $E -- --save 1 --ref /out/off ;;
    e4m3x)    "$REPO/tests/handoff/run.sh" "$OUT" e4m3x $D $CAP $E -- --ref /out/off,/out/e4m3 ;;
    e4m3b)    "$REPO/tests/handoff/run.sh" "$OUT" e4m3b $D $CAP $E $B -- --save 1 --ref /out/off,/out/e4m3 ;;
    e4m3bx)   "$REPO/tests/handoff/run.sh" "$OUT" e4m3bx $D $CAP $E $B -- --ref /out/off,/out/e4m3,/out/e4m3b ;;
    e4m3bf)   "$REPO/tests/handoff/run.sh" "$OUT" e4m3bf $D $CAP $E $B GLM53_MOE_E4M3_FOLD_SHARED=1 -- --ref /out/off,/out/e4m3,/out/e4m3b ;;
    e4m3bft)  "$REPO/tests/handoff/run.sh" "$OUT" e4m3bft $D $CAP $E $B GLM53_MOE_E4M3_FOLD_SHARED=1 -- --ref /out/off,/out/e4m3,/out/e4m3b ;;
    e4m3t)    "$REPO/tests/handoff/run.sh" "$OUT" e4m3t $D $CAP $E -- --ref /out/off,/out/e4m3 ;;
    e4m3nt)   "$REPO/tests/handoff/run.sh" "$OUT" e4m3nt $D $CAP $E GLM53_MOE_E4M3_TOKGATHER=0 -- --ref /out/off,/out/e4m3 ;;
    d16)      "$REPO/tests/handoff/run.sh" "$OUT" d16 $D $CAP $E GLM53_MOE_E4M3_DOWN=f16 -- --ref /out/off,/out/e4m3 ;;
    d16b)     "$REPO/tests/handoff/run.sh" "$OUT" d16b $D $CAP $E GLM53_MOE_E4M3_DOWN=f16 $B -- --ref /out/off,/out/e4m3 ;;
  esac
done
