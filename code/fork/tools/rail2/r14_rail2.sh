#!/usr/bin/env bash
# R14: bring up the second PCIe-half RoCE rails while GLM is stopped, measure them, switch NCCL to two HCAs, drop the
# NCCL tuner, start. Falls back to the single rail automatically if GLM does not come up healthy.
# The 10.0.100.x / 10.0.102.x / 10.0.103.x worker addresses below are example rail subnets (see glm53-rail2); edit them.
set -uo pipefail
L=R14; W=${WORKER_USER}@${WORKER_IP}
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
say(){ echo "[$L $(date +%T)] $*"; }
t0=$(date +%s)
while :; do m=$(curl -s -m 5 localhost:8888/metrics)
  r=$(echo "$m" | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}'); w=$(echo "$m" | awk '/^vllm:num_requests_waiting\{/{s+=$2} END{print s+0}')
  [ "$r" = 0 ] && [ "$w" = 0 ] && break
  [ $(( $(date +%s) - t0 )) -ge 900 ] && { say "busy for 900 s; not restarting"; exit 3; }; sleep 5; done
say "idle -> stop"; ./start.sh stop 2>&1 | tail -2; sleep 20
say "rail2 up"; sudo -n /usr/local/sbin/glm53-rail2 up | sed 's/^/  nodeA /'; ssh -o BatchMode=yes $W "sudo -n /usr/local/sbin/glm53-rail2 up" | sed 's/^/  nodeB /'
for ip in 10.0.102.3 10.0.103.3; do printf "  ping %s: " $ip; ping -c 2 -W 1 $ip >/dev/null 2>&1 && echo ok || echo FAIL; done
bw(){ # $1 label, then pairs dev:ip:port ; prints per-pair and total Gb/s
  local label=$1; shift; local srv="" pids=() i=0
  for p in "$@"; do IFS=: read -r d ip port <<< "$p"; srv="$srv timeout 25 ib_write_bw -d $d -x 3 -p $port -s 1048576 -q 4 -D 5 -F --report_gbits >/dev/null 2>&1 &"; done
  ssh -o BatchMode=yes $W "$srv wait" & local sp=$!; sleep 2
  for p in "$@"; do IFS=: read -r d ip port <<< "$p"; ( timeout 25 ib_write_bw -d $d -x 3 -p $port -s 1048576 -q 4 -D 5 -F --report_gbits $ip 2>/dev/null | awk '/^ *1048576/{print $4}' > /tmp/bw_$i ) & pids+=($!); i=$((i+1)); done
  wait "${pids[@]}"; wait $sp
  local tot=0 out=""; for j in $(seq 0 $((i-1))); do v=$(cat /tmp/bw_$j 2>/dev/null); out="$out ${v:-?}"; tot=$(awk -v a=$tot -v b=${v:-0} 'BEGIN{print a+b}'); done
  say "  ib_write_bw $label: per-link$out Gb/s, total $tot Gb/s"; echo $tot
}
A=$(bw "rocep1s0f1 alone" rocep1s0f1:10.0.100.3:18515 | tail -1)
B=$(bw "roceP2p1s0f1 alone" roceP2p1s0f1:10.0.102.3:18516 | tail -1)
C=$(bw "same cable (p1s0f1 + P2p1s0f1)" rocep1s0f1:10.0.100.3:18515 roceP2p1s0f1:10.0.102.3:18516 | tail -1)
D=$(bw "two cables (p1s0f1 + P2p1s0f0)" rocep1s0f1:10.0.100.3:18515 roceP2p1s0f0:10.0.103.3:18517 | tail -1)
if awk -v c=$C -v d=$D 'BEGIN{exit !(d > 1.1*c)}'; then HCA=rocep1s0f1,roceP2p1s0f0; MERGE=0; else HCA=rocep1s0f1,roceP2p1s0f1; MERGE=1; fi
say "choose NCCL_IB_HCA=$HCA MERGE_NICS=$MERGE (single $A, same-cable $C, two-cable $D Gb/s)"
cp -p .env .env.bak-tf-R14-$(date +%H%M%S)
sed -i -E "s/^HEAD_CX7_IB=.*/HEAD_CX7_IB=$HCA/; s/^WORKER_CX7_IB=.*/WORKER_CX7_IB=$HCA/; s/^NCCL_TUNER_PLUGIN=.*/NCCL_TUNER_PLUGIN=/; s/^NCCL_TUNER_CONFIG_FILE=.*/NCCL_TUNER_CONFIG_FILE=/" .env
grep -q '^NCCL_IB_MERGE_NICS=' .env && sed -i -E "s/^NCCL_IB_MERGE_NICS=.*/NCCL_IB_MERGE_NICS=$MERGE/" .env || printf '%s\n' "# [tf-exl3-fork] R14 dual PCIe-half rail" "NCCL_IB_MERGE_NICS=$MERGE" >> .env
~/tf-exl3-deploy/fix_gids.sh || say "GID detection failed"
say "start"; SKIP_BUILD=1 ./start.sh start 2>&1 | tail -2
ok=0; for i in $(seq 1 144); do curl -s -m 3 -o /dev/null -w '%{http_code}' localhost:8888/health | grep -q 200 && { ok=1; break; }; sleep 5; done
if [ $ok = 1 ]; then say "healthy with dual rail"; exit 0; fi
say "NOT healthy in 12 min -> fallback to the single rail"
docker logs glm53-exl3-head 2>&1 | grep -iE "nccl|error" | tail -5
./start.sh stop 2>&1 | tail -1; sleep 20
sed -i -E "s/^HEAD_CX7_IB=.*/HEAD_CX7_IB=rocep1s0f1/; s/^WORKER_CX7_IB=.*/WORKER_CX7_IB=rocep1s0f1/; s/^NCCL_IB_MERGE_NICS=.*/NCCL_IB_MERGE_NICS=0/" .env
~/tf-exl3-deploy/fix_gids.sh; SKIP_BUILD=1 ./start.sh start 2>&1 | tail -2
for i in $(seq 1 144); do curl -s -m 3 -o /dev/null -w '%{http_code}' localhost:8888/health | grep -q 200 && { say "healthy on single rail (fallback)"; exit 2; }; sleep 5; done
say "STILL NOT HEALTHY"; exit 1
