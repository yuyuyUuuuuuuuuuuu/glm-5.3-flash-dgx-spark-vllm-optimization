#!/usr/bin/env bash
# usage: bench_round.sh <label>   (runs prose/structured/coding x10 on the local GLM, prints a one-line summary each)
set -uo pipefail
cd "${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
D="${DEPLOY_DIR:-$HOME/tf-exl3-deploy}"
export VLLM_API_KEY="$(grep '^VLLM_API_KEY=' .env | cut -d= -f2- | tr -d "\"'")"
L="$1"; mkdir -p "$D/bench"
snap(){ echo "$1 $(date +%T) running=$(curl -s -m 5 localhost:8888/metrics | awk '/^vllm:num_requests_running/{print $2}') $(grep -E 'pgmigrate_fail|compact_stall' /proc/vmstat | tr '\n' ' ') proactive=$(cat /proc/sys/vm/compaction_proactiveness) avail_gb=$(free -g | awk 'NR==2{print $7}')"; }
snap "[$L] start"
for w in prose structured coding ja; do
  a=""; b=tests/bench_decode.py; [ "$w" != prose ] && a="--$w"
  [ "$w" = ja ] && { a="--skip-coherence"; b="$D/bench_decode_ja.py"; }
  timeout 1500 python3 $b --phase "$L-$w" $a --out "$D/bench/${L}_${w}.json" --runs 10 >/dev/null 2>&1
  python3 - "$L" "$w" "$D" <<'PY'
import json,sys
L,w,D=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(f"{D}/bench/{L}_{w}.json"))
g=lambda k:(round(d[k],2) if isinstance(d.get(k),float) else d.get(k))
print(f"[{L}] {w:10s} tok/s med {g('tok_s_median')} (min {g('tok_s_min')} max {g('tok_s_max')}) acc/step {g('accepted_per_step_median')} ms/step {g('decode_ms_per_draft_step_median')}")
PY
done
snap "[$L] end"
