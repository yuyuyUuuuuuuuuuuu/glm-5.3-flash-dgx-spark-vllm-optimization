#!/usr/bin/env bash
# usage: restart2.sh <label> [bw]
#   stop -> 25s -> memprep on both nodes (defragment free memory before the new cudaMalloc's; see memprep.py)
#   -> (bw: per-chunk fresh-allocation read bandwidth on both nodes) -> GID auto-fix -> SKIP_BUILD=1 start
set -uo pipefail
# Production runs with ABLIT (o_proj transplant) ON - owner 2026-10-05. start.sh resets ABLIT=0 after sourcing .env and only
# honours a caller export, so every restart through this script exports it (ABLIT=0 restart2.sh ... for a stock-weights start).
export ABLIT="${ABLIT:-1}"
L="$1"; BW="${2:-}"
IMG=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
W="${WORKER_SSH:?set WORKER_SSH=<user>@<worker address>}"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/tf-exl3-deploy}"
cd "${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
say(){ echo "[$L $(date +%T)] $*"; }
say "running=$(curl -s -m 5 localhost:8888/metrics | awk '/^vllm:num_requests_running/{print $2}')"
# never cut a user request: wait (up to IDLE_WAIT_S, default 900 s) until nothing is running or waiting
t0=$(date +%s)
while :; do
  m=$(curl -s -m 5 localhost:8888/metrics)
  r=$(echo "$m" | awk "/^vllm:num_requests_running/{s+=\$2} END{print s+0}")
  w=$(echo "$m" | awk "/^vllm:num_requests_waiting\\{/{s+=\$2} END{print s+0}")
  [ "$r" = 0 ] && [ "$w" = 0 ] && break
  if [ $(( $(date +%s) - t0 )) -ge "${IDLE_WAIT_S:-900}" ]; then say "still busy (running=$r waiting=$w) after ${IDLE_WAIT_S:-900}s - NOT restarting"; exit 3; fi
  sleep 5
done
say "idle (running=0 waiting=0) -> stop"; ./start.sh stop 2>&1 | tail -3
sleep 25
say "containers nodeA: $(docker ps --format '{{.Names}}' | grep glm53 || echo none) | nodeB: $(ssh -o BatchMode=yes $W "docker ps --format '{{.Names}}' | grep glm53 || echo none")"
say "mem avail GB nodeA=$(free -g | awk 'NR==2{print $7}') nodeB=$(ssh -o BatchMode=yes $W "free -g | awk 'NR==2{print \$7}'")"
if [ "${MEMPREP:-0}" = 1 ]; then  # start.sh runs memprep itself when GLM53_MEMPREP=1
  say "memprep (both nodes in parallel)"
  python3 "$DEPLOY_DIR/memprep.py" --floor-gib "${MEMPREP_FLOOR_GIB:-4}" > /tmp/memprep-nodeA.log 2>&1 &
  ssh -o BatchMode=yes $W "python3 ~/tf-exl3-deploy/memprep.py --floor-gib ${MEMPREP_FLOOR_GIB:-4}" > /tmp/memprep-nodeB.log 2>&1 &
  wait
  cat /tmp/memprep-nodeA.log /tmp/memprep-nodeB.log
fi
if [ "$BW" = bw ]; then
  cmd="timeout 300 docker run --rm --gpus all --network none --memory 20g -v /tmp/tf-bw:/w -w /w -e HOME=/tmp -e TORCH_EXTENSIONS_DIR=/tmp/ext --entrypoint python3 $IMG -u tests/bw_pagesize.py 0 8 0 2>&1 | grep -E 'cudaMalloc chunk'"
  say "fresh-allocation bandwidth per 1 GiB chunk (nodeA | nodeB in parallel)"
  bash -c "$cmd" > /tmp/bw-nodeA.log 2>&1 &
  ssh -o BatchMode=yes $W "$cmd" > /tmp/bw-nodeB.log 2>&1 &
  wait
  echo "-- nodeA"; cat /tmp/bw-nodeA.log; echo "-- nodeB"; cat /tmp/bw-nodeB.log
fi
# second PCIe-half rail: re-apply while GLM is stopped (down -> trimmed RX rings -> up) on both nodes
say "rail2: $(sudo -n /usr/local/sbin/glm53-rail2 boot 2>&1 | grep -E "^enP2p1s0f0np0" ) | nodeB: $(ssh -o BatchMode=yes $W "sudo -n /usr/local/sbin/glm53-rail2 boot 2>&1 | grep -E ^enP2p1s0f0np0")"
say "gid check"; "$DEPLOY_DIR/fix_gids.sh" || { say "GID detection FAILED - starting with .env as is"; }
say "start (ABLIT=$ABLIT)"; SKIP_BUILD=1 ./start.sh start 2>&1 | tail -5
say "start returned"
