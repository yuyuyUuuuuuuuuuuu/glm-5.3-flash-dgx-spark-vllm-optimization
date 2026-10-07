#!/usr/bin/env bash
# inside the container (/w): per-trigger ablation, and robustness of the simulator gain over M and the dirty-L2 volume
cd /w
L=tests/logs/roof
env ROOF_SIM_CONFIGS="off;all;t1;t2;t3;t4" python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_ablation_m5.log 2>&1; echo "sim_ablation rc $?"
for args in "19 1 10 16 160 13" "19 8 10 16 160 13" "19 16 10 16 160 13" "19 5 0 16 160 13" "19 5 16 16 160 13"; do
  set -- $args
  env ROOF_SIM_CONFIGS="off;all" python3 -u tests/roof/sim_step.py "$@" > $L/sim_m$2_d$3.log 2>&1; echo "sim M=$2 D=$3 rc $?"
done
