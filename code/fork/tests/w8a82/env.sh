# source me: production overlay exl3.py + partial checkpoint for the w8a82 benches
export GPU_RUN_ENV="TF_EXL3_JIT=1"
export GPU_RUN_RO="${TF_EXL3_MODELS:-$HOME/models}/GLM-5.3-Flash-EXL3-TR3-4bpw-partial"
export GPU_RUN_BIND="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py"
