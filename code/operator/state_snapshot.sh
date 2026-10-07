#!/usr/bin/env bash
# Read-only node state snapshot for the "bandwidth decays with uptime" investigation. Usage: state_snapshot.sh <label>
# Writes ~/tf-exl3-deploy/state/<host>-<label>-<ts>.txt and prints a one-line summary.
set -u
L="${1:-snap}"; D=~/tf-exl3-deploy/state; mkdir -p "$D"; F="$D/$(hostname)-$L-$(date +%Y%m%d-%H%M%S).txt"
{
echo "## host $(hostname) label $L date $(date '+%F %T') boot $(uptime -s) kernel $(uname -r)"
echo "## cmdline"; cat /proc/cmdline
echo "## meminfo"; grep -E 'MemTotal|MemFree|MemAvailable|Cached|AnonHugePages|ShmemHugePages|FileHugePages|HugePages_|Hugepagesize|CmaTotal|CmaFree' /proc/meminfo
echo "## buddyinfo"; cat /proc/buddyinfo
echo "## thp"; for f in enabled defrag khugepaged/defrag; do echo "$f=$(cat /sys/kernel/mm/transparent_hugepage/$f 2>/dev/null)"; done
echo "## sysctl"; for k in compaction_proactiveness extfrag_threshold watermark_scale_factor min_free_kbytes zone_reclaim_mode; do echo "$k=$(cat /proc/sys/vm/$k 2>/dev/null)"; done
echo "## vmstat"; grep -E '^(pgmigrate|compact_|thp_|numa_|pgfault|pgmajfault|nr_free_pages|nr_anon_transparent)' /proc/vmstat
echo "## devfreq"; for d in /sys/class/devfreq/*; do [ -d "$d" ] && echo "$(basename $d) cur=$(cat $d/cur_freq 2>/dev/null) min=$(cat $d/min_freq 2>/dev/null) max=$(cat $d/max_freq 2>/dev/null) gov=$(cat $d/governor 2>/dev/null)"; done
echo "## cpufreq"; for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do echo "$(basename $(dirname $c)) cur=$(cat $c/scaling_cur_freq 2>/dev/null) max=$(cat $c/scaling_max_freq 2>/dev/null) gov=$(cat $c/scaling_governor 2>/dev/null)"; done | sort -V
echo "## thermal"; for t in /sys/class/thermal/thermal_zone*; do echo "$(cat $t/type 2>/dev/null)=$(cat $t/temp 2>/dev/null)"; done
echo "## nvidia-smi"; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE 2>&1 | grep -vE '^\s*$' | head -80
echo "## kthreads cpu"; ps -eo comm,time --sort=-time | grep -E 'kcompactd|khugepaged|kswapd|nvidia|irq/' | head -10
} > "$F" 2>&1
echo "$(hostname) $L: $(grep -E '^## host' "$F" | cut -c4-) | avail=$(awk '/MemAvailable/{printf "%.0fG",$2/1048576}' /proc/meminfo) | devfreq: $(ls /sys/class/devfreq 2>/dev/null | tr '\n' ' ') -> $F"

# one-line trend summary (CSV) for the bandwidth-decay watch
S=~/tf-exl3-deploy/state/summary.csv
[ -f "$S" ] || echo "ts,host,uptime_h,avail_gb,pgmigrate_success,pgmigrate_fail,compact_stall,sw_power_cap_us,gpu_power_w,gpu_temp_c,itl_sum_s,itl_count,spec_accepted,spec_drafts" > "$S"
vm(){ awk -v k="$1" '$1==k{print $2}' /proc/vmstat; }
spc=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | awk -F: '/SW Power Capping/{gsub(/[^0-9]/,"",$2); print $2; exit}')
pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
tc=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
M=$(curl -s -m 5 localhost:8888/metrics 2>/dev/null)
ms(){ echo "$M" | awk -v k="$1" 'index($1,k)==1{s+=$2} END{if(s!="")printf "%.6f",s}'; }
up=$(awk '{printf "%.2f",$1/3600}' /proc/uptime)
echo "$(date '+%F %T'),$(hostname),$up,$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo),$(vm pgmigrate_success),$(vm pgmigrate_fail),$(vm compact_stall),$spc,$pw,$tc,$(ms vllm:inter_token_latency_seconds_sum),$(ms vllm:inter_token_latency_seconds_count),$(ms vllm:spec_decode_num_accepted_tokens_total),$(ms vllm:spec_decode_num_draft_tokens_total)" >> "$S"

# v2 trend (2026-09-28): free-memory contiguity + an idle-only GLM decode probe (node with the launcher only).
# free_ge8m_gib = free memory in buddy blocks >= 8 MiB (what a fresh cudaMalloc can get contiguously);
# probe = launcher tests/bench_decode.py prose x3, ONLY when no request is running or waiting (ms per draft step is
# the bandwidth-sensitive number; it would drift up if the served model's memory got slower).
S2=~/tf-exl3-deploy/state/summary2.csv
[ -f "$S2" ] || echo "ts,host,uptime_h,avail_gb,free_ge2m_gib,free_ge8m_gib,pgmigrate_fail,compact_stall,probe_ms_step,probe_tok_s,probe_acc_step" > "$S2"
fge(){ awk -v o0="$1" '{for(i=5;i<=NF;i++){o=i-5; if(o>=o0) s+=$i*(4096*2^o)}} END{printf "%.2f", s/2^30}' /proc/buddyinfo; }
pms=""; pts=""; pacc=""
LD=~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
if [ -f "$LD/tests/bench_decode.py" ] && [ -n "$M" ]; then
  run=$(echo "$M" | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
  wai=$(echo "$M" | awk '/^vllm:num_requests_waiting\{/{s+=$2} END{print s+0}')
  if [ "$run" = 0 ] && [ "$wai" = 0 ]; then
    ( cd "$LD" && VLLM_API_KEY="$(grep '^VLLM_API_KEY=' .env | cut -d= -f2- | tr -d "\"'")" \
        timeout 180 python3 tests/bench_decode.py --phase probe --runs 3 --out /tmp/glm53-probe.json >/dev/null 2>&1 )
    if [ -f /tmp/glm53-probe.json ]; then
      read -r pms pts pacc < <(python3 -c 'import json;d=json.load(open("/tmp/glm53-probe.json"));f=lambda k:(d.get(k) if d.get(k) is not None else "");print(f("decode_ms_per_draft_step_median"),f("tok_s_median"),f("accepted_per_step_median"))' 2>/dev/null)
      rm -f /tmp/glm53-probe.json
    fi
  fi
fi
echo "$(date '+%F %T'),$(hostname),$up,$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo),$(fge 9),$(fge 11),$(vm pgmigrate_fail),$(vm compact_stall),$pms,$pts,$pacc" >> "$S2"
find ~/tf-exl3-deploy/state -name "*.txt" -mtime +30 -delete 2>/dev/null
