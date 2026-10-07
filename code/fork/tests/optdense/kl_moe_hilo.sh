#!/usr/bin/env bash
# opt-dense: KL of the outlier-channel hi+lo PROTOTYPE (GLM53_DENSE_W8A8_HILO) on the handoff MoE mini, vs the same
# W8A8-off reference as kl_moe_w8a8.sh (OUT/off). run.sh takes the GPU lock per run.
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-dense/kl_moe}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-tp2r0}"
D=HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py
BASE="GLM53_MOE_E4M3=1 GLM53_PREFILL_FUSED_CAP=1 GLM53_DENSE_W8A8=1"
MA="--model $HANDOFF_MODEL --ref /out/off"
SUB=kda.in_proj_qkvbfg_a,mla.o_proj
ALLX=kda.in_proj_qkvbfg_a,kda.o_proj,mla.q_b_proj,mla.o_proj,shared.gate_up_proj,shared.down_proj,dense.gate_up_proj
HA=kda.o_proj:256,shared.down_proj:128,dense.down_proj:512,mla.q_b_proj:256,mla.o_proj:512
HB=$HA,kda.in_proj_qkvbfg_a:512,mla.fused_qkv_a_proj:512,shared.gate_up_proj:512,dense.gate_up_proj:512
HS=mla.o_proj:512
for a in ${ARMS:-all_hA allx_hA all_hA_first all_hB sub_hS}; do
  [ -s "$OUT/$a/kl.jsonl" ] && continue
  case $a in
    all_hA)       "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_HILO=$HA -- $MA ;;
    all_hA_first) "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_HILO=$HA GLM53_DENSE_W8A8_HILO_SEL=first -- $MA ;;
    all_hB)       "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_HILO=$HB -- $MA ;;
    allx_hA)      "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_ONLY=$ALLX GLM53_DENSE_W8A8_HILO=$HA -- $MA ;;
    allx_hA_first) "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_ONLY=$ALLX GLM53_DENSE_W8A8_HILO=$HA GLM53_DENSE_W8A8_HILO_SEL=first -- $MA ;;
    sub_hS_first) "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_ONLY=$SUB GLM53_DENSE_W8A8_HILO=$HS GLM53_DENSE_W8A8_HILO_SEL=first -- $MA ;;
    allx_hA_fast) "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_ONLY=$ALLX GLM53_DENSE_W8A8_HILO=$HA GLM53_DENSE_W8A8_HILO_SEL=first -- $MA ;;  # with a REPACK_LD overlay ext
    sub_hS)       "$R/tests/handoff/run.sh" "$OUT" $a $D $BASE GLM53_DENSE_W8A8_ONLY=$SUB GLM53_DENSE_W8A8_HILO=$HS -- $MA ;;
  esac
done
echo KLHILO-DONE
