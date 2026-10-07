#!/usr/bin/env bash
# KL chain on the handoff mini (w8a8 config variant): bf16 dense (ref0) -> production FP8 (ref1) -> A/A -> W8A8 on
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/w8a82/kl}"
shift || true
ARMS="${*:-bf16 off off2 on}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16n}"
export HANDOFF_MODEL=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini-w8a8
D=HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py
for a in $ARMS; do
  case $a in
    bf16) "$REPO/tests/handoff/run.sh" "$OUT" bf16 $D GLM53_DENSE_FP8=off -- --save 1 ;;
    off)  "$REPO/tests/handoff/run.sh" "$OUT" off $D -- --save 1 --ref /out/bf16 ;;
    off2) "$REPO/tests/handoff/run.sh" "$OUT" off2 $D -- --ref /out/off ;;
    on)   "$REPO/tests/handoff/run.sh" "$OUT" on $D GLM53_DENSE_W8A8=1 -- --ref /out/off,/out/bf16 --stats 1 ;;
    on*)  "$REPO/tests/handoff/run.sh" "$OUT" "$a" $D GLM53_DENSE_W8A8=1 ${ARM_ENV:-} -- --ref /out/off,/out/bf16 --stats 1 ;;
  esac
done
