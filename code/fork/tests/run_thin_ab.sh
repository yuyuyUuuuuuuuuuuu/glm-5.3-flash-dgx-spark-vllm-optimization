#!/usr/bin/env bash
# E.5 against production's thin-decode kernel (launcher GLM53_EXL3_MOE_FAST=1), reproducible from this repo:
#   tests/run_thin_ab.sh <shared|indep> [thin-decode .so] [log]
# shared = gate/up share their input rotation (the overlay aliases the up_suh table and the thin kernel skips one
# Hadamard), indep = they do not. Binds read-only through tests/gpu_run.sh (GPU_RUN_BIND):
#   the thin-decode exllamav3_ext over the image's, and the launcher overlay exl3.py over the image's
#   quantization/exl3.py (the module production imports, docs/STATUS.md "本番の exl3.py").
# Refuses unless the .so is the build nodeC measured (sha256 below; provenance docs/ref/launcher_0924/PROVENANCE.txt).
# The log starts with the exact command, git HEAD, image ID and every bound file's sha256, then
# tests/probe_native_kernel.py (which native kernel exl3_moe launched: must be glm53_exl3_moe_fast_kernel<4, 256, ...>
# with the expected shared_input flag, else the run stops), then tests/bench_ab_decode.py.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
MODE="${1:?usage: tests/run_thin_ab.sh <shared|indep> [thin .so] [log]}"
SO="${2:-${REPO}/artifacts/thin_decode/exllamav3_ext.thin.so}"
LOG="${3:-${REPO}/docs/logs/prod_live/bench_ab_decode_thin_${MODE}.log}"
WANT=50e915073d39356ec8cbd6d451ffaf897c2476831d035de2194dcc920578509f
case "${MODE}" in shared) SH=1 ;; indep) SH=0 ;; *) echo "mode must be shared or indep" >&2; exit 2 ;; esac
[ -f "${SO}" ] || { echo "thin-decode .so ${SO} not found (docs/ref/launcher_0924/PROVENANCE.txt)" >&2; exit 2; }
have=$(sha256sum "${SO}" | cut -d' ' -f1)
[ "${have}" = "${WANT}" ] || { echo "thin-decode .so sha256 ${have} is not the measured build ${WANT}" >&2; exit 2; }
SPK=/usr/local/lib/python3.12/dist-packages
LIVE="${REPO}/docs/ref/prod_live/overlay_exl3.py"
BIND="${SO}=${SPK}/exllamav3_ext.cpython-312-aarch64-linux-gnu.so;${LIVE}=${SPK}/vllm/model_executor/layers/quantization/exl3.py"
CMD="export GLM53_EXL3_MOE_FAST=1 TF_EXL3_BENCH_SHARED_SUH=${SH}; python3 -u tests/probe_native_kernel.py && python3 -u tests/bench_ab_decode.py"
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
{
  echo "command: GPU_RUN_BIND=\"${BIND}\" tests/gpu_run.sh bash -c '${CMD}'"
  echo "wrapper: tests/run_thin_ab.sh ${MODE}; git HEAD $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!docs' || echo ' + uncommitted code changes'); image ${IMG} $(docker image inspect "${IMG}" --format '{{.Id}}' | cut -c8-19); $(date '+%F %T')"
  echo "bind $(sha256sum "${SO}" | cut -c1-64) ${SO} -> ${SPK}/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
  echo "bind $(sha256sum "${LIVE}" | cut -c1-64) ${LIVE} -> ${SPK}/vllm/model_executor/layers/quantization/exl3.py"
} > "${LOG}"
set +e
GPU_RUN_BIND="${BIND}" tests/gpu_run.sh bash -c "${CMD}" >> "${LOG}" 2>&1
rc=$?
echo "exit=${rc}" >> "${LOG}"
grep -v "exl3 e2 diag" "${LOG}" | grep -E "^command|^wrapper|^bind|exl3_moe launched|suggested|RESULT|checks:" | cut -c1-240
exit ${rc}
