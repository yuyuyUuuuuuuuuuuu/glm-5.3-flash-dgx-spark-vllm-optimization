#!/usr/bin/env bash
# prefixhit nodeC mini-rig runs. Usage: go.sh <label> <A0 len> <lens csv> [extra KV ...]
set -u
O=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/runs
M=${PH_MODEL:-${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini2g}
W=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/wt
export HANDOFF_MODEL=$M HANDOFF_KIT=${PH_KIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z6rev}
label=$1; a0=$2; lens=$3; shift 3
C="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 HANDOFF_B_LENS=$lens HANDOFF_B_REPEAT=1 HANDOFF_B_GEN=${PH_GEN:-4}"
COMMON="--model $M --shadow 0 --topk-probe 0 --mla-probe 0 --kv-bytes 4294967296 --gpu-util 0.2 --max-model-len 40960 --consistency 0"
$W/tests/handoff/run.sh $O $label $C "$@" -- $COMMON --prompts $a0 --gen 2 > $O/$label.log 2>&1
echo "$label rc=$?" >> $O/rc.txt
