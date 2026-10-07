#!/usr/bin/env bash
# inside the container (/w): final simulator runs + the roof test; logs -> tests/logs/roof/
cd /w
L=tests/logs/roof
env ROOF_SIM_CONFIGS="off;all;v1:all" ROOF_SIM_PROFILE=6 python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_final_m5.log 2>&1; echo "sim_final_m5 rc $?"
env ROOF_SIM_CONFIGS="off;all;t1;t2;t3;t4" python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_ablation_m5.log 2>&1; echo "sim_ablation rc $?"
python3 -u tests/test_fp8_roof.py > $L/test_fp8_roof.log 2>&1; echo "test_fp8_roof rc $?"
