#!/usr/bin/env bash
# inside the container (/w): the roof test + the final simulators; logs -> tests/logs/roof/
cd /w
L=tests/logs/roof
python3 -u tests/test_fp8_roof.py > $L/test_fp8_roof.log 2>&1; echo "test_fp8_roof rc $?"; grep -E "^RESULT|checks:|CHECK FAILED" $L/test_fp8_roof.log | tail -8
env ROOF_LOOP_CONFIGS="off;on:t0,t5;t5:t5;t0:t0;big:t0,t5:t0=16,t5=24" python3 -u tests/roof/sim_loop.py 5 1000 8 4 15 4 > $L/sim_loop_m5.log 2>&1; echo "sim_loop rc $?"
env ROOF_SIM_CONFIGS="off;all;v1:all" ROOF_SIM_PROFILE=6 python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_final_m5.log 2>&1; echo "sim_final_m5 rc $?"
