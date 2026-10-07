#!/usr/bin/env bash
# Run the drafter tests on nodeC (one GPU container at a time, <= 8 GiB each, weights read-only).
# Logs: docs/logs/drafter/<test>.log. Exit non-zero if any test fails.
#   A  test_spec_resample_noise.py   real Triton kernels, no weights (~3 min)
#   B1/C1 bench_drafter_fp8.py        DFlash2 + target lm_head weights (~3 min)
#   B/C test_drafter_fp8_build.py     real vLLM classes, drafter build x3 (~2 min)
#   review_counterfactual.py          host, CPU only: pre-review patches (a4d51fa) vs current (~10 s)
# DRAFTER_TESTS="a.py b.py" runs a subset (default: all four).
# GPU_RUN_BIND (passed through to tests/gpu_run.sh) runs the GPU tests against another production file, e.g. the
# launcher overlay exl3.py production installs over the image's (docs/ref/prod_live/overlay_exl3.py):
#   GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" \
#   DRAFTER_LOG_DIR=docs/logs/drafter_live tests/drafter/run_drafter_tests.sh
# (review_counterfactual.py runs on the host and is not affected by the bind.)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRAFT=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-DFlash2-dc77ff1c
TGT=${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-EXL3-TR3-4bpw-partial
SEL="${DRAFTER_TESTS:-test_spec_resample_noise.py bench_drafter_fp8.py test_drafter_fp8_build.py review_counterfactual.py}"
LOGD="${DRAFTER_LOG_DIR:-$REPO/docs/logs/drafter}"
case "$LOGD" in /*) ;; *) LOGD="$REPO/$LOGD";; esac
mkdir -p "$LOGD"
echo "drafter tests: GPU_RUN_BIND=${GPU_RUN_BIND:-<none: the image modules>} logs=$LOGD"
rc=0
run() {
  local name="$1" ro="$2"
  [[ " $SEL " == *" $name "* ]] || return 0
  echo "== $name"
  local log="$LOGD/${name%.py}.log" r
  if [ "$ro" = "host" ]; then
    python3 -u "$REPO/tests/drafter/$name" >"$log" 2>&1
  else
    GPU_RUN_RO="$ro" "$REPO/tests/gpu_run.sh" python3 -u "tests/drafter/$name" >"$log" 2>&1
  fi
  r=$?
  tail -n 1 "$log"
  [ $r -eq 0 ] || { echo "   -> exit $r"; rc=1; }
}
run test_spec_resample_noise.py ""
run bench_drafter_fp8.py "$DRAFT:$TGT/lm_head"
run test_drafter_fp8_build.py "$DRAFT:$TGT"
run review_counterfactual.py host
exit $rc
