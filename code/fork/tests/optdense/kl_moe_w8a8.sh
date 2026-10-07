#!/usr/bin/env bash
# opt-dense: per-projection W8A8 KL on the handoff MoE mini (TP=2 rank-0 shapes; 5 KDA + 5 MLA, 3 dense-MLP + 7 MoE
# layers WITH the shared expert = the shared.* projections the w8a8 mini cannot measure), production MoE config
# (GLM53_MOE_E4M3=1, GLM53_PREFILL_FUSED_CAP=1) in every arm, KL vs the W8A8-off arm (tests/w8a82/kl_driver.py).
# run.sh takes /tmp/tf-gpu-bench.lock itself per run (do not wrap this script in flock).
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-dense/kl_moe}"
export HANDOFF_KIT="${HANDOFF_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}"
export HANDOFF_MODEL="${HANDOFF_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-moe-mini-tp2r0}"
D=HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py
BASE="GLM53_MOE_E4M3=1 GLM53_PREFILL_FUSED_CAP=1"
MA="--model $HANDOFF_MODEL"
SUB=kda.in_proj_qkvbfg_a,mla.o_proj
ALLX=kda.in_proj_qkvbfg_a,kda.o_proj,mla.q_b_proj,mla.o_proj,shared.gate_up_proj,shared.down_proj,dense.gate_up_proj
NOSH=kda.in_proj_qkvbfg_a,kda.o_proj,mla.fused_qkv_a_proj,mla.q_b_proj,mla.o_proj,dense.gate_up_proj,dense.down_proj
for a in ${ARMS:-off off2 sub all shared_gate_up shared_down allx nosh}; do
  [ -s "$OUT/$a/kl.jsonl" ] && continue
  case $a in
    off)  [ -s "$OUT/off/logprobs.pt" ] || "$R/tests/handoff/run.sh" "$OUT" off $D $BASE -- $MA --save 1 ;;
    off2) "$R/tests/handoff/run.sh" "$OUT" off2 $D $BASE -- $MA --ref /out/off ;;
    all)  "$R/tests/handoff/run.sh" "$OUT" all $D $BASE GLM53_DENSE_W8A8=1 -- $MA --ref /out/off ;;
    sub)  "$R/tests/handoff/run.sh" "$OUT" sub $D $BASE GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=$SUB -- $MA --ref /out/off ;;
    allx) "$R/tests/handoff/run.sh" "$OUT" allx $D $BASE GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=$ALLX -- $MA --ref /out/off ;;
    nosh) "$R/tests/handoff/run.sh" "$OUT" nosh $D $BASE GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=$NOSH -- $MA --ref /out/off ;;
    shared_gate_up|shared_down|dense_gate_up|dense_down|kda_o_proj|mla_q_b_proj|mla_fused_qkv_a_proj)
          p="${a%%_*}.${a#*_}"; case $p in *_proj) ;; *) p="${p}_proj" ;; esac   # shared_down -> shared.down_proj
          "$R/tests/handoff/run.sh" "$OUT" "$a" $D $BASE GLM53_DENSE_W8A8=1 "GLM53_DENSE_W8A8_ONLY=$p" -- $MA --ref /out/off ;;
  esac
done
echo KLMOE-DONE
