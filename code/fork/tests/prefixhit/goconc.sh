#!/usr/bin/env bash
# concurrency pooled-key test: go.sh-like. Usage: goconc.sh <label> <prompts csv> <batch_a> [KV...]
set -u
O=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/runs
M=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini2g
W=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/wt
export HANDOFF_MODEL=$M HANDOFF_KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z6rev
label=$1; prompts=$2; ba=$3; shift 3
C="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 GLM53_MAMBA_ALIGN_SEED=1 HANDOFF_CONC_TEST=1 HANDOFF_SPEC_SYNTHETIC=0.9,0.85,0.8,0.7,0.6,0.5,0.4"
COMMON="--model $M --shadow 0 --topk-probe 0 --mla-probe 0 --kv-bytes 4294967296 --gpu-util 0.2 --max-model-len 40960 --consistency 0"
$W/tests/handoff/run.sh $O $label $C "$@" -- $COMMON --prompts $prompts --batch-a $ba --gen 96 > $O/$label.log 2>&1
echo "$label rc=$?" >> $O/rc.txt
