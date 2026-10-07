#!/usr/bin/env bash
# Post-restart measurement set for a release label (idle production only; each probe refuses/aborts when busy).
# usage: r16_measure.sh <label>     output: ~/tf-exl3-deploy/measure/<label>.log
set -uo pipefail
L="${1:?label}"; D="${DEPLOY_DIR:-$HOME/tf-exl3-deploy}"; mkdir -p $D/measure; LOG=$D/measure/$L.log
cd $D
{
echo "== [$L] $(date '+%F %T') mem nodeA: $(free -g | awk 'NR==2{print "avail "$7"G"}') swap $(free -m | awk 'NR==3{print $3"M"}') | nodeB: $(ssh -o BatchMode=yes "${WORKER_SSH:?set WORKER_SSH}" "free -g | awk 'NR==2{print \"avail \"\$7\"G\"}'")"
echo "== decode (bench_decode x10)"; ./bench_round.sh "$L"
echo "== decode stream gaps"; python3 stall_probe.py "$L-2k" 2000 1500; python3 stall_probe.py "$L-40k" 40000 1500
echo "== prefill real text"; python3 prefill_probe_real.py "$L" 8500,15300 3
echo "== prefill random 24k"; python3 prefill_probe.py "$L" 24000 3 2>&1 | tail -2
echo "== APC 40k (repeat + turn2)"; python3 apc_probe.py 40000
echo "== APC 97k repeat (fresh 591 case)"; python3 longctx_probe.py "$L-apc97k" 100000 64
echo "== kpool decode-vs-prefill"; for _k in 1 2 3 4 5 6 7 8 9 10; do python3 kpool_decode_consistency.py $D/kpool-after-$L.json && break; echo "kpool: busy, retry $_k"; sleep 30; done; python3 kpool_decode_consistency.py --compare $D/kpool-before-r15.json $D/kpool-after-$L.json
echo "== quality long vs R15"; python3 quality_long.py QL_$L.json 2>&1 | tail -1; python3 quality_long.py --compare QL_R15a.json QL_$L.json 2>&1 | tail -1
echo "== quality short vs R15"; VLLM_API_KEY="$(grep '^VLLM_API_KEY=' "${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}/.env" | cut -d= -f2- | tr -d "\"'")" python3 quality_probe.py Q_$L.json 2>&1 | tail -1; VLLM_API_KEY="$(grep '^VLLM_API_KEY=' "${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}/.env" | cut -d= -f2- | tr -d "\"'")" python3 quality_probe.py --compare Q_R15a.json Q_$L.json 2>&1 | tail -2
echo "== klh"; KLH_BASE="$(echo "$L" | sed -E "s/^envab-[^-]+-/envab-base-/")" KLH_REF="${KLH_REF:-$D/klh/ref_v2.klh.gz}" bash "$D/klh/klh_measure.sh" "$L" --preset gate 2>&1 | grep -E "^(KLH|KLH-CMP|KLH-GATE)|klh:"
echo "== [$L] done $(date '+%F %T') mem nodeA: $(free -g | awk 'NR==2{print "avail "$7"G"}') swap $(free -m | awk 'NR==3{print $3"M"}')"
} 2>&1 | tee -a $LOG
