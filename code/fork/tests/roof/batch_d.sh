#!/usr/bin/env bash
# inside the container (/w): budget / CTA re-tune and the per-trigger ablation on the simulator with the attention
# pre-block windows (a repeat of the ablation: the first one ran while another GPU user was active, spread 14-18 %)
cd /w
L=tests/logs/roof
env ROOF_SIM_CONFIGS="off;all;t1;t2;t3;t4" python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_ablation_m5.log 2>&1; echo "sim_ablation rc $?"
env ROOF_SIM_CONFIGS="off;all;t3x16:all:t3=16;t1x16:all:t1=16;big:all:t1=16,t3=16,t4=24;c12:all::12;c6:all::6;pol1:all:::1" python3 -u tests/roof/sim_step.py 19 5 10 16 160 15 > $L/sim_tune_m5.log 2>&1; echo "sim_tune rc $?"
