#!/usr/bin/env bash
# Build every AOT extension the kit ships, inside the pinned serving image (CPU-only containers, --network none,
# your uid), stage the .so files into code/kit/{site,overlay}/ and regenerate code/kit/MANIFEST.sha256.
#   FKDA_SRC=<dir staged by fetch_flashkda_sources.sh> code/build/build_all.sh
# Needs docker and the image below (pull it by digest first). Peak host memory is a few GB per nvcc job: lower MAX_JOBS
# on a box that is also serving. No GPU is used. The resulting .so files are NOT byte-identical to production's (nvcc
# embeds temporary file names); code/kit/PRODUCTION_BINARIES.sha256 lists production's bytes for reference.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="$REPO/fork"; KIT="$REPO/kit"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d}"
FKDA_SRC="${FKDA_SRC:?set FKDA_SRC=<dir> (run code/build/fetch_flashkda_sources.sh <dir> first)}"
SCR="${BUILD_SCRATCH:-$REPO/build/.scratch}"
mkdir -p "$SCR/fkda/out"
run() {
  docker run --rm --network none -u "$(id -u):$(id -g)" -v "$FORK:/w" -w /w \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/tmp/ext -e MAX_JOBS="${MAX_JOBS:-8}" \
    --entrypoint bash "$IMAGE" -c "$1"
}
echo "== 1/4 setup.py extensions (TF_PARITY=1 = production arithmetic)"
run 'python3 setup.py build_ext --inplace --force'
echo "== 2/4 overlay-only extensions (e4m3 routed-MoE prefill, fused mHC post+prenorm)"
run 'python3 tools/moee4m3/build.py && python3 tools/mhcfused/build.py'
echo "== 3/4 FlashKDA 17a037d fp32-state build (_flashkda_fp32_C, the GLM53_KDA_FLASHKDA_V=1 build production runs)"
FKDA_SRC="$FKDA_SRC" FKDA_SCRATCH="$SCR/fkda" TF_EXL3_IMAGE="$IMAGE" bash "$FORK/tests/fkda/build_flashkda_fp32.sh"
echo "== 4/4 stage into the kit"
SFX=cpython-312-aarch64-linux-gnu.so
for n in tf_exl3_moe_ext tf_fp8_gemv_ext tf_dlmh_ext tf_fp8_large_m_ext tf_fp8_roof_ext glm53_gemv_ext \
         glm53_mla_prefill_ext glm53_smallops_ext; do
  install -m 0755 "$FORK/$n.$SFX" "$KIT/site/$n.$SFX"
done
install -m 0755 "$FORK/tf_fp8_w8a8_ext.$SFX" "$KIT/overlay/tf_fp8_w8a8_ext.$SFX"
install -m 0755 "$FORK/overlay/glm53_moe_e4m3_ext.$SFX" "$KIT/overlay/glm53_moe_e4m3_ext.$SFX"
install -m 0755 "$FORK/overlay/glm53_mhc_fused_ext.$SFX" "$KIT/overlay/glm53_mhc_fused_ext.$SFX"
install -m 0755 "$SCR/fkda/out/_flashkda_fp32_C.abi3.so" "$KIT/overlay/_flashkda_fp32_C.abi3.so"
# GLM53_KDA_FLASHKDA_V=2|3 (not used in production) need _flashkda_fp32_C2/_C3 from tests/fkda2|fkda3/build_variant.sh,
# and their wrappers refuse any build whose sha256 differs from the one they were validated with (see docs/KDA_FLASHKDA3.md).
bash "$KIT/tools/make_manifest.sh"
echo "built and staged; next: REPRODUCE.md step 'Install the kit'"
