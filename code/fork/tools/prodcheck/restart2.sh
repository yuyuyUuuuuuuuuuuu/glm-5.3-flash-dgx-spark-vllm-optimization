#!/usr/bin/env bash
# usage: restart2.sh <label> [bw]
#   stop -> 25s -> memprep on both nodes (defragment free memory before the new cudaMalloc's; see memprep.py)
#   -> (bw: per-chunk fresh-allocation read bandwidth on both nodes) -> GID auto-fix -> SKIP_BUILD=1 start
set -uo pipefail
L="$1"; BW="${2:-}"
IMG=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
W=${WORKER_USER}@${WORKER_IP}
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
say(){ echo "[$L $(date +%T)] $*"; }
say "running=$(curl -s -m 5 localhost:8888/metrics | awk '/^vllm:num_requests_running/{print $2}')"
say "stop"; ./start.sh stop 2>&1 | tail -3
sleep 25
say "containers nodeA: $(docker ps --format '{{.Names}}' | grep glm53 || echo none) | nodeB: $(ssh -o BatchMode=yes $W "docker ps --format '{{.Names}}' | grep glm53 || echo none")"
say "mem avail GB nodeA=$(free -g | awk 'NR==2{print $7}') nodeB=$(ssh -o BatchMode=yes $W "free -g | awk 'NR==2{print \$7}'")"
if [ "${MEMPREP:-1}" = 1 ]; then
  say "memprep (both nodes in parallel)"
  python3 ~/tf-exl3-deploy/memprep.py --floor-gib "${MEMPREP_FLOOR_GIB:-4}" > /tmp/memprep-nodeA.log 2>&1 &
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
say "gid check"; ~/tf-exl3-deploy/fix_gids.sh || { say "GID detection FAILED - starting with .env as is"; }
say "start"; SKIP_BUILD=1 ./start.sh start 2>&1 | tail -5
say "start returned"
