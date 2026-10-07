#!/usr/bin/env bash
# Stage the FlashKDA build inputs exactly as the production build used them (needs git, curl and network):
#   <dest>/fk/          FlashKDA @ 17a037d (vllm-project/FlashKDA), with CUTLASS @ 5c149f52 in fk/cutlass
#   <dest>/vllm-csrc/   flashkda_registration.cpp + core/registration.h from vLLM @ ddd6fbca (the registration shim)
# Then: FKDA_SRC=<dest> FKDA_SCRATCH=<scratch> code/fork/tests/fkda/build_flashkda_fp32.sh  (or code/build/build_all.sh)
# FlashKDA is MIT (Copyright (c) 2026 MoonshotAI), CUTLASS is BSD-3-Clause, vLLM is Apache-2.0. Nothing is vendored here.
set -euo pipefail
DEST="${1:?usage: fetch_flashkda_sources.sh <dest dir>}"
FK_REV=17a037d98da546deb4591e967cf961a43c034d8b
CUTLASS_REV=5c149f52a436782210263fb2f19b354443a61c6a
VLLM_REV=ddd6fbca148a867aad1fcab7ec72f582b9977db4
mkdir -p "$DEST/vllm-csrc/core"
if [ ! -d "$DEST/fk/.git" ]; then git clone --quiet https://github.com/vllm-project/FlashKDA.git "$DEST/fk"; fi
git -C "$DEST/fk" -c advice.detachedHead=false checkout --quiet "$FK_REV"
if [ ! -d "$DEST/fk/cutlass/.git" ] && [ ! -f "$DEST/fk/cutlass/.git" ]; then
  rm -rf "$DEST/fk/cutlass"; git clone --quiet https://github.com/NVIDIA/cutlass.git "$DEST/fk/cutlass"
fi
git -C "$DEST/fk/cutlass" -c advice.detachedHead=false checkout --quiet "$CUTLASS_REV"
R=https://raw.githubusercontent.com/vllm-project/vllm/$VLLM_REV/csrc
curl -fsSL -o "$DEST/vllm-csrc/flashkda_registration.cpp" "$R/flashkda_registration.cpp"
curl -fsSL -o "$DEST/vllm-csrc/core/registration.h" "$R/core/registration.h"
# the two shim files production's build used (sha256 of the staged copies)
(cd "$DEST/vllm-csrc" && sha256sum -c --quiet - <<'SUMS'
949db0aba6302707e7500fb237dba2f9482418e0d2eadbc2394eab360cf755e6  flashkda_registration.cpp
58b98cd4792f18baae3d316af7a91440ec991ee75d2641d2bc805bcc8cd1c02e  core/registration.h
SUMS
)
echo "staged: FlashKDA $(git -C "$DEST/fk" rev-parse --short HEAD), CUTLASS $(git -C "$DEST/fk/cutlass" rev-parse --short HEAD), vLLM csrc @ ${VLLM_REV:0:8}"
