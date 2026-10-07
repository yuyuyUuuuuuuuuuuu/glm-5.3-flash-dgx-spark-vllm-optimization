# Source me: production module set for the sparse-MLA prefill tests (SM90_KV=1 as in production start.sh:2081-2104).
# image + launcher overlay exl3.py + flashinfer 0.6.18 + patched cuda.py / flashinfer_mla_sparse_sm90.py.
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_SP=/usr/local/lib/python3.12/dist-packages
_A="${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}"
_SM90="${MLA_SM90_PY:-$_A/vllm-patches/flashinfer_mla_sparse_sm90.py.patched}"
export GPU_RUN_BIND="$_REPO/docs/ref/prod_live/overlay_exl3.py=$_SP/vllm/model_executor/layers/quantization/exl3.py;$_A/fi618/flashinfer=$_SP/flashinfer;$_A/fi618/flashinfer_cubin=$_SP/flashinfer_cubin;$_A/vllm-patches/cuda.py.exl3.patched=$_SP/vllm/platforms/cuda.py;$_SM90=$_SP/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
export GPU_RUN_ENV="FLASHINFER_DISABLE_VERSION_CHECK=1;FLASHINFER_WORKSPACE_BASE=/w/.fi_ws;FLASHINFER_CUDA_ARCH_LIST=12.1a;TF_EXL3_SHIM=/w/.ext/shim"
