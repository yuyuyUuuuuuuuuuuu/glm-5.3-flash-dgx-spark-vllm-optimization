# source me: production module set for tests/gpu_run.sh (image + launcher overlay exl3.py + SM90_KV=1 mounts:
# flashinfer 0.6.18, patched vllm/platforms/cuda.py and flashinfer_mla_sparse_sm90.py, FLASHINFER_DISABLE_VERSION_CHECK=1;
# launcher start.sh:2081-2104). Assets: ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}.
_R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_A="${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}"
_SP=/usr/local/lib/python3.12/dist-packages
export GPU_RUN_BIND="${_R}/docs/ref/prod_live/overlay_exl3.py=${_SP}/vllm/model_executor/layers/quantization/exl3.py;${_A}/vllm-patches/cuda.py.exl3.patched=${_SP}/vllm/platforms/cuda.py;${_A}/vllm-patches/flashinfer_mla_sparse_sm90.py.patched=${_SP}/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
export GPU_RUN_BIND_DIR="${_A}/fi618/flashinfer=${_SP}/flashinfer;${_A}/fi618/flashinfer_cubin=${_SP}/flashinfer_cubin"
export GPU_RUN_ENV="FLASHINFER_DISABLE_VERSION_CHECK=1"
