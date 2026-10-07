#!/usr/bin/env bash
# per-projection W8A8 KL on the handoff mini (r16z2rev review): off reference, then one arm per PROJ_NAME (+ all)
set -u
R=${TF_EXL3_KITS:-$HOME}/tf-exl3-fork-wt-r16z2rev
OUT=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/w8a8-rev/kl
export HANDOFF_KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16n
export HANDOFF_MODEL=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini-w8a8
D=HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py
[ -s $OUT/off/logprobs.pt ] || "$R/tests/handoff/run.sh" "$OUT" off $D -- --save 1
for p in ${ARMS:-kda.in_proj_qkvbfg_a kda.o_proj mla.fused_qkv_a_proj mla.q_b_proj mla.o_proj dense.gate_up_proj dense.down_proj all}; do
  lab=p_${p//./_}
  [ -s $OUT/$lab/kl.jsonl ] && continue
  if [ $p = all ]; then "$R/tests/handoff/run.sh" "$OUT" $lab $D GLM53_DENSE_W8A8=1 -- --ref /out/off
  else "$R/tests/handoff/run.sh" "$OUT" $lab $D GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=$p -- --ref /out/off; fi
done
echo KLPROJ-DONE
