#!/usr/bin/env bash
# nodeA, OPERATOR-run, production booted with GLM53_SPEC_VTRIM=shadow (+ GLM53_SPEC_VTRIM_LOG=20): runs the 4 decode
# workloads of ~/tf-exl3-deploy/bench_round.sh one by one under the prodbench lock and projects GLM53_SPEC_VTRIM=on
# for each workload and each shadow-grid threshold from the histogram difference (rank 0 writes it into the bind-
# mounted vLLM cache dir every GLM53_SPEC_VTRIM_LOG verify steps). Outputs and texts are production's untouched ones
# (shadow changes nothing), so the run doubles as a base arm.
# Usage: shadow_bench.sh <label> [runs]   (copy tests/vtrim/project_hist.py next to it)
set -uo pipefail
L="$1"; RUNS="${2:-10}"
H="${SPEC_VTRIM_HIST:-$HOME/.cache/vllm-glm53-flash/glm53_spec_vtrim_hist.json}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/tf-exl3-deploy/bench/shadow-$L"; mkdir -p "$OUT"
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
export VLLM_API_KEY="$(grep '^VLLM_API_KEY=' .env | cut -d= -f2- | tr -d "\"'")"
declare -A C0=([structured]=52.6 [prose]=53.7 [coding]=57.1 [ja]=50.1)
for w in structured prose coding ja; do
  a=""; b=tests/bench_decode.py; [ "$w" != prose ] && a="--$w"
  [ "$w" = ja ] && { a="--skip-coherence"; b=${HOME}/tf-exl3-deploy/bench_decode_ja.py; }
  sleep 3; BEF=(); cp "$H" "$OUT/${w}_before.json" 2>/dev/null && BEF=(--before "$OUT/${w}_before.json")
  flock ${HOME}/tf-exl3-assets-prodbench.lock timeout 1500 python3 $b --phase "shadow-$L-$w" $a \
    --out "$OUT/${w}.json" --runs "$RUNS" > /dev/null 2>&1
  sleep 3; cp "$H" "$OUT/${w}_after.json"
  echo "== $w"; python3 "$HERE/project_hist.py" "$OUT/${w}_after.json" "${BEF[@]}" --c0 "${C0[$w]}"
done
