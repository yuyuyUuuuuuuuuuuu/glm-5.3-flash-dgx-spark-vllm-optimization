#!/usr/bin/env bash
# serial adv runs; run.sh queues on flock /tmp/tf-gpu-bench.lock itself
set -u
O=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/adv/runs
M=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-handoff-mini2g
W=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit/wt-adv
export HANDOFF_MODEL=$M HANDOFF_KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z6rev
COMMON="--model $M --shadow 0 --topk-probe 0 --mla-probe 0 --kv-bytes 4294967296 --gpu-util 0.2 --max-model-len 40960"
SYN="0.9,0.85,0.8,0.7,0.6,0.5,0.4"
CB="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 GLM53_MAMBA_ALIGN_SEED=1 HANDOFF_CONC_TEST=1 HANDOFF_SPEC_SYNTHETIC=$SYN GLM53_KPOOL_RING=1 HANDOFF_TAILPTR=1"
CA="--prompts 6000,7001 --batch-a 2 --gen 96"
PH="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 HANDOFF_B_LENS=27660,32260,32268,32320,36900 HANDOFF_B_REPEAT=1 HANDOFF_B_GEN=4"
DEC="GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 HANDOFF_SPEC_SYNTHETIC=$SYN HANDOFF_B_SALT=A HANDOFF_B_PAIR=1 HANDOFF_MP=1 GLM53_KPOOL_RING=1 GLM53_DEC_KDA_LAZY=1"
DA="--prompts 32240,4590 --batch-a 2 --gen 64 --consistency 40"
run() { label=$1; kv=$2; a=$3; $W/tests/handoff/run.sh $O $label $kv -- $COMMON $a > $O/$label.log 2>&1; echo "$label rc=$? $(date +%T)" >> $O/rc.txt; }
run F1graphfix2 "$CB GLM53_KPOOL_TAIL_POSITIONS=2" "$CA"
run F0graphfix1 "$CB GLM53_KPOOL_TAIL_POSITIONS=1" "$CA"
run E1eagerfix "$CB GLM53_KPOOL_TAIL_POSITIONS=1 HANDOFF_EAGER=1" "$CA"
run R1x "$PH" "--prompts 37000 --gen 2"
run R13x "$PH HANDOFF_MP=1" "--prompts 37000 --gen 2"
run P1mp "$DEC" "$DA"
run P2mpfix "$DEC GLM53_KPOOL_TAIL_POSITIONS=2" "$DA"
run E2eagerbase "$CB HANDOFF_EAGER=1" "$CA"
