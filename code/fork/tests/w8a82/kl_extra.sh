#!/usr/bin/env bash
set -u
until grep -q KLPROJ-DONE ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/w8a8-rev/kl_proj.log; do sleep 30; done
R=${TF_EXL3_KITS:-$HOME}/tf-exl3-fork-wt-r16z2rev; OUT=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/w8a8-rev/kl
export HANDOFF_KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16n HANDOFF_MODEL=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini-w8a8
D=HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py
"$R/tests/handoff/run.sh" "$OUT" t_inproj_tail288 $D GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a GLM53_DENSE_W8A8_EXACT_TAIL=12576:288 -- --ref /out/off
"$R/tests/handoff/run.sh" "$OUT" p_sub $D GLM53_DENSE_W8A8=1 GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj -- --ref /out/off
echo KLEXTRA-DONE
