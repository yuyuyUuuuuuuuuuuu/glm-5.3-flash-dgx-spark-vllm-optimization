#!/usr/bin/env bash
# Rebuild the test inputs that can be derived from public sources (REPRODUCE.md section 13). CPU only, no GPU, no
# container is started (the image's files are copied out of a created, never started, container).
# Writes under $TF_EXL3_MODELS and $TF_EXL3_ASSETS (tests/paths.sh) and checks every result against
# tests/assets/models.sha256 / tests/assets/assets.sha256.
# Usage: tests/derive_assets.sh [step ...]      steps (default: all, in this order):
#   models        hf download of the public checkpoints the tests read (about 44 GB):
#                   GLM-5.3-Flash-Uncensored-NVFP4  thebriangao/GLM-5.3-Flash-Uncensored-NVFP4 @ 59a99c95, 8 of 62 shards
#                   GLM-5.3-Flash-DFlash2-dc77ff1c  incoai/GLM-5.3-Flash-DFlash2 @ dc77ff1c (CC BY-NC-ND 4.0)
#                   GLM-OCR/tokenizer.json          zai-org/GLM-OCR @ 2e85a628 (the prompt tokenizer of tests/handoff)
#   vllm-src      the image's vllm package -> $TF_EXL3_ASSETS/vllm-src/vllm
#   vllm-patches  code/vllm-patches/*.patch applied to it -> $TF_EXL3_ASSETS/vllm-patches/*.patched
#   launcher      the public launcher at the pinned commit -> $TF_EXL3_ASSETS/prod-launcher (tests/handoff reads its
#                 overlay/ only; the non-secret env it used is tests/handoff/env_nonsecret.txt)
#   mini          tests/handoff/build_mini.py -> $TF_EXL3_MODELS/GLM-5.3-Flash-handoff-mini (+ -dflash2), ~8 GB RAM
#   fi618         flashinfer-python + flashinfer-cubin 0.6.18.dev20260819 (git 61a6c651) copied out of the public image
#                 ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6... -> $TF_EXL3_ASSETS/fi618 (the overlay production
#                 mounts with SM90_KV=1; byte-identical to production's: 95,185 files, aggregate sha256 below)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO}/tests/paths.sh"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
PATCHES="${REPO}/../vllm-patches"
LAUNCHER_URL=https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
LAUNCHER_COMMIT=0f49cfdbaa131286eb592cd6ebfa048f3aa85c4e
SP=/usr/local/lib/python3.12/dist-packages
FI618_IMAGE="${FI618_IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6}"
FI618_SUM=c60e3671cb78ee1e5ecf943cb3dc6054c81c381bc8085b432ad0d596e607a5e1   # sha256 of the sorted per-file sha256 list
steps=("$@"); [ ${#steps[@]} -gt 0 ] || steps=(models vllm-src vllm-patches launcher mini fi618)

check() {   # check <root> <sha file> <path prefix>: verify the lines of <sha file> that start with <path prefix>
  local root="$1" list="$2" pre="$3"
  (cd "$root" && grep -E "  ${pre}" "$list" | sha256sum -c --quiet -) && echo "derive: ${pre} sha256 OK"
}

for s in "${steps[@]}"; do
  case "$s" in
  models)
    command -v hf >/dev/null || { echo "derive: needs the hf CLI (pip install -U huggingface_hub)"; exit 2; }
    hf download thebriangao/GLM-5.3-Flash-Uncensored-NVFP4 --revision 59a99c95e6ea1142be39af0e617bdfcec0766052 \
      --local-dir "${TF_EXL3_MODELS}/GLM-5.3-Flash-Uncensored-NVFP4" config.json generation_config.json \
      model-0000{1..8}-of-00062.safetensors
    hf download incoai/GLM-5.3-Flash-DFlash2 --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 \
      --local-dir "${TF_EXL3_MODELS}/GLM-5.3-Flash-DFlash2-dc77ff1c" config.json model.safetensors
    hf download zai-org/GLM-OCR --revision 2e85a62840ccac27daa451df36c736c4636b8628 \
      --local-dir "${TF_EXL3_MODELS}/GLM-OCR" tokenizer.json
    for p in GLM-5.3-Flash-Uncensored-NVFP4/ GLM-5.3-Flash-DFlash2-dc77ff1c/ GLM-OCR/; do
      check "${TF_EXL3_MODELS}" "${REPO}/tests/assets/models.sha256" "$p"; done ;;
  vllm-src)
    mkdir -p "${TF_EXL3_ASSETS}/vllm-src"
    [ ! -e "${TF_EXL3_ASSETS}/vllm-src/vllm" ] || { echo "derive: ${TF_EXL3_ASSETS}/vllm-src/vllm exists, kept"; continue; }
    cid=$(docker create --network none "${IMAGE}" true)
    docker cp "${cid}:${SP}/vllm" "${TF_EXL3_ASSETS}/vllm-src/vllm" > /dev/null; docker rm "${cid}" > /dev/null
    echo "derive: ${TF_EXL3_ASSETS}/vllm-src/vllm from $(docker image inspect "${IMAGE}" --format '{{.Id}}' | cut -c1-19)" ;;
  vllm-patches)
    V="${TF_EXL3_ASSETS}/vllm-src/vllm"; O="${TF_EXL3_ASSETS}/vllm-patches"; mkdir -p "$O"
    [ -f "$V/platforms/cuda.py" ] || { echo "derive: run the vllm-src step first"; exit 2; }
    patch -s -o "$O/cuda.py.exl3.patched" "$V/platforms/cuda.py" < "${PATCHES}/cuda.py.patch"
    patch -s -o "$O/flashinfer_mla_sparse_sm90.py.patched" "$V/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py" \
      < "${PATCHES}/flashinfer_mla_sparse_sm90.patch"
    check "${TF_EXL3_ASSETS}" "${REPO}/tests/assets/assets.sha256" vllm-patches/ ;;
  launcher)
    L="${TF_EXL3_ASSETS}/prod-launcher"
    [ ! -e "$L" ] || { echo "derive: $L exists, kept"; continue; }
    git init -q "$L.git-tmp"; git -C "$L.git-tmp" fetch -q --depth 1 "${LAUNCHER_URL}" "${LAUNCHER_COMMIT}"
    git -C "$L.git-tmp" checkout -q FETCH_HEAD; mkdir -p "$L"; cp -a "$L.git-tmp/overlay" "$L/overlay"; rm -rf "$L.git-tmp"
    echo "derive: $L/overlay from the launcher at ${LAUNCHER_COMMIT:0:7}" ;;
  mini)
    python3 "${REPO}/tests/handoff/build_mini.py" "${TF_EXL3_MODELS}/GLM-5.3-Flash-handoff-mini"
    check "${TF_EXL3_MODELS}" "${REPO}/tests/assets/models.sha256" GLM-5.3-Flash-handoff-mini/ ;;
  fi618)
    F="${TF_EXL3_ASSETS}/fi618"
    [ ! -e "$F" ] || { echo "derive: $F exists, kept"; continue; }
    mkdir -p "$F"; cid=$(docker create --network none "${FI618_IMAGE}" true)
    for d in flashinfer flashinfer_cubin flashinfer_python-0.6.18.dev20260819.dist-info \
             flashinfer_cubin-0.6.18.dev20260819.dist-info; do
      docker cp "${cid}:${SP}/$d" "$F/$d" > /dev/null; done
    docker rm "${cid}" > /dev/null
    got=$(cd "$F" && find flashinfer flashinfer_cubin -type f ! -name '*.pyc' ! -path '*__pycache__*' -print0 | sort -z \
          | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
    [ "$got" = "${FI618_SUM}" ] && echo "derive: fi618 sha256 OK (identical to production's overlay)" \
      || { echo "derive: fi618 aggregate sha256 ${got} != ${FI618_SUM}"; exit 3; } ;;
  *) echo "derive: unknown step $s"; exit 2 ;;
  esac
done
