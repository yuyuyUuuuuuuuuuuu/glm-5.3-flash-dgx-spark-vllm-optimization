# Source me: the host directories the tests read, each overridable (REPRODUCE.md section 13). tests/run_all.sh and
# tests/gpu_run.sh source this file; gpu_run.sh passes the three values into the container (where HOME=/tmp), so Python
# code inside and outside containers resolves the same paths.
#   TF_EXL3_MODELS  model checkpoints and mini checkpoints      (default ~/models)
#   TF_EXL3_ASSETS  test assets: extracted image sources, patched vLLM files, flashinfer 0.6.18, launcher copy
#                                                                (default ~/tf-exl3-assets)
#   TF_EXL3_KITS    the directory holding deploy kits (tf-exl3-deploy16*) and branch worktrees (default ~)
: "${TF_EXL3_MODELS:=${HOME}/models}"
: "${TF_EXL3_ASSETS:=${HOME}/tf-exl3-assets}"
: "${TF_EXL3_KITS:=${HOME}}"
export TF_EXL3_MODELS TF_EXL3_ASSETS TF_EXL3_KITS
