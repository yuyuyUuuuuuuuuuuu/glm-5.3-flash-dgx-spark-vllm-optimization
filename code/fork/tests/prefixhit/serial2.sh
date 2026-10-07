#!/usr/bin/env bash
set -u
O=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/adv/runs
M=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini2g
W=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/wt-adv
export HANDOFF_MODEL=$M HANDOFF_KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z6rev
COMMON="--model $M --shadow 0 --topk-probe 0 --mla-probe 0 --kv-bytes 4294967296 --gpu-util 0.2 --max-model-len 40960"
PH="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 HANDOFF_B_LENS=27660,27712,32260,32268,32320,36900 HANDOFF_B_REPEAT=1 HANDOFF_B_GEN=4 HANDOFF_MP=1"
run() { label=$1; kv=$2; a=$3; $W/tests/handoff/run.sh $O $label $kv -- $COMMON $a > $O/$label.log 2>&1; echo "$label rc=$? $(date +%T)" >> $O/rc.txt; }
until grep -q E2eagerbase $O/rc.txt 2>/dev/null; do sleep 30; done
run R14mpfix "$PH GLM53_KPOOL_TAIL_POSITIONS=2" "--prompts 37000 --gen 2"
run R15mpring "$PH GLM53_KPOOL_RING=1" "--prompts 37000 --gen 2"
run R16mpringfix "$PH GLM53_KPOOL_RING=1 GLM53_KPOOL_TAIL_POSITIONS=2" "--prompts 37000 --gen 2"
