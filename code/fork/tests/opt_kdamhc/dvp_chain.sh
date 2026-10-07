#!/usr/bin/env bash
# decode-vs-prefill consistency + prefill KL of the fused mHC post+prenorm kernel on the handoff mini (nodeC).
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-kdamhc/dvp}
export HANDOFF_KIT=${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}
export HANDOFF_MODEL=${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini-w8a8}
D=HANDOFF_DRIVER=/w/tests/opt_kdamhc/dvp_driver.py
F="GLM53_PREFILL_QUICKWINS=all GLM53_MLA_PREFILL=1 GLM53_KDA_FLASHKDA=1 GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj"
A="--model $HANDOFF_MODEL ${DVP_ARGS:-}"
[ -s $OUT/base/dvp.jsonl ] || "$R/tests/handoff/run.sh" "$OUT" base $D $F -- $A
# arms: base_b = repeat of base (noise floor); fk3 = the fkda3 FlashKDA build (GLM53_KDA_FLASHKDA_V=3, direct output);
# fused0/fused1 = the fused mHC post+prenorm kernel (tests/opt_kdamhc/mhc_hook.py); fk3_fused0 = both
for arm in ${ARMS:-base_b fk3 fused0 fused1}; do
  [ -s $OUT/$arm/dvp.jsonl ] && continue
  m=""; x=""
  case $arm in fused*) m="--mhc $arm";; fk3_fused*) m="--mhc ${arm#fk3_}"; x="GLM53_KDA_FLASHKDA_V=3";; fk3) x="GLM53_KDA_FLASHKDA_V=3";; esac
  "$R/tests/handoff/run.sh" "$OUT" $arm $D $F $x -- $A $m --ref /out/base
done
grep -h DVP $OUT/*/dvp.log
echo DVP-CHAIN-DONE
