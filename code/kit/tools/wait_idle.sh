#!/usr/bin/env bash
# deploy-r16 (nodeA, operator): wait until the GLM server is idle (vllm:num_requests_running == 0 AND
# vllm:num_requests_waiting == 0 on 3 consecutive polls, 5 s apart), at most MAX_WAIT seconds (default 900).
# Exit 0 = idle (restart now), 1 = still busy after MAX_WAIT (do NOT restart), 2 = metrics unreachable.
# Reads only /metrics; never prints a credential.   Usage: wait_idle.sh [port]
set -uo pipefail
PORT="${1:-8888}"; MAX="${MAX_WAIT:-900}"; t0=$(date +%s); ok=0
while :; do
  m=$(curl -sf -m 5 "http://127.0.0.1:${PORT}/metrics") || { echo "wait_idle: /metrics unreachable"; exit 2; }
  # an answer without both gauges (an error page, a server still starting) would sum to 0 = a false "idle"
  # (here-strings, not pipes: under pipefail an early-exiting grep -q can SIGPIPE the writer)
  grep -q '^vllm:num_requests_running' <<< "$m" && grep -qE '^vllm:num_requests_waiting[{ ]' <<< "$m" \
    || { echo "wait_idle: /metrics has no vllm:num_requests_running / num_requests_waiting gauge - not treating it as idle"; exit 2; }
  run=$(echo "$m" | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
  wai=$(echo "$m" | awk '/^vllm:num_requests_waiting[{ ]/{s+=$2} END{print s+0}')
  if [ "${run%.*}" = 0 ] && [ "${wai%.*}" = 0 ]; then ok=$((ok + 1)); else ok=0; fi
  echo "wait_idle $(date +%T): running=$run waiting=$wai idle_polls=$ok"
  [ "$ok" -ge 3 ] && exit 0
  [ $(( $(date +%s) - t0 )) -ge "$MAX" ] && { echo "wait_idle: still busy after ${MAX}s - not restarting"; exit 1; }
  sleep 5
done
