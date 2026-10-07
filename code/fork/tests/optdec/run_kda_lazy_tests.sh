#!/usr/bin/env bash
# Bitwise tests of GLM53_DEC_KDA_LAZY on nodeC in the production image, against production's KDA recurrent kernel:
# the image's fused_recurrent.py / kda.py with overlay/patch_kda_strided_qkv.py applied (= production since r16k),
# bound read-only over the image's files. Usage: tests/optdec/run_kda_lazy_tests.sh [test.py]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${VLLM_SRC:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/vllm-src/vllm}/third_party/flash_linear_attention/ops"
FLA="${KDA_LAZY_FLA:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-decode/fla}"
mkdir -p "$FLA"
cp "$SRC/fused_recurrent.py" "$FLA/fused_recurrent.py"; cp "$SRC/kda.py" "$FLA/kda.py"
GLM53_KDA_STRIDED_QKV=1 GLM53_FLA_FUSED_RECURRENT_PY="$FLA/fused_recurrent.py" GLM53_FLA_KDA_PY="$FLA/kda.py" \
  python3 "$REPO/overlay/patch_kda_strided_qkv.py"
SP=/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops
export GPU_RUN_BIND="$FLA/fused_recurrent.py=$SP/fused_recurrent.py;$FLA/kda.py=$SP/kda.py"
T="${1:-tests/optdec/test_kda_lazy.py}"; shift || true
exec flock /tmp/tf-gpu-bench.lock "$REPO/tests/gpu_run.sh" python3 "/w/$T" "$@"
