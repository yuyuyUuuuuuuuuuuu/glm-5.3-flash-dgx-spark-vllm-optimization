#!/usr/bin/env bash
# FKDA: build FlashKDA 17a037d (fp32 recurrent state, vllm#58846) as
# _flashkda_fp32_C for sm_121a (GB10), inside the production image. CPU-only
# compile (no --gpus): does not take /tmp/tf-gpu-bench.lock. Copy of pfkda's
# tests/pf3000/build_flashkda.sh with two changes:
#   1. the extension (and with it the torch.library namespace) is renamed
#      _flashkda_C -> _flashkda_fp32_C: the fp32-state build must be able to
#      coexist with the image's pre-fix bf16 vllm/_flashkda_C (same namespace).
#      The rename is applied to a COPY of the registration shim (the only place
#      the namespace appears outside the extension name itself); the FlashKDA
#      kernel sources are untouched, so the built kernel is 17a037d bit for bit.
#   2. staging/output live under /tmp/fkda (FKDA_SCRATCH), not /tmp/pf3000.
#
# Inputs (host, already staged by pfkda; network was only used for their git
# fetches):
#   $SRC/fk/         FlashKDA @ 17a037d (+ cutlass @ 5c149f5 in fk/cutlass)
#   $SRC/vllm-csrc/  flashkda_registration.cpp + core/registration.h @ ddd6fbca
# Output: $OUT/_flashkda_fp32_C.abi3.so (+ BUILD_ARCH / *_REF records)
# Run:  tests/fkda/build_flashkda_fp32.sh            (uses tests/fkda/docker_cpu.sh)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${FKDA_SRC:-/tmp/pf3000}"
SCR="${FKDA_SCRATCH:-/tmp/fkda}"
OUT="$SCR/out"
mkdir -p "$OUT"
cat > "$SCR/_stage_build.sh" <<'EOS'
set -euo pipefail
BUILD=/fkda/build
rm -rf "$BUILD" && mkdir -p "$BUILD" /fkda/out
cp -a /fkda/in/fk "$BUILD"/flashkda
rm -rf /fkda/reg "$BUILD"/vllm-csrc && cp -a /fkda/in/vllm-csrc /fkda/reg
# the rename: the torch.library namespace and the registered python module name
grep -c "_flashkda_C" /fkda/reg/flashkda_registration.cpp >/dev/null
sed -i 's/_flashkda_C/_flashkda_fp32_C/g' /fkda/reg/flashkda_registration.cpp
! grep -q "_flashkda_C" /fkda/reg/flashkda_registration.cpp
cp -a /fkda/reg "$BUILD"/vllm-csrc
cp /w/tests/fkda/setup_fkda_fp32.py "$BUILD"/setup.py
cd "$BUILD"
ARCH="${FKDA_NVCC_ARCH:-sm_121a}"
printf '__global__ void k(){}\nint main(){return 0;}\n' > /tmp/arch_probe.cu
if ! nvcc -arch="$ARCH" -c /tmp/arch_probe.cu -o /tmp/arch_probe.o >/dev/null 2>&1; then
  echo "NOTE: nvcc rejected $ARCH, falling back to sm_120f" | tee /fkda/out/BUILD_ARCH_NOTE.txt
  ARCH=sm_120f
fi
echo "building _flashkda_fp32_C for $ARCH"
export PF3000_NVCC_ARCH="$ARCH"
export MAX_JOBS="${MAX_JOBS:-16}"
python3 setup.py build_ext --inplace 2>&1 | tee /fkda/out/build_ext.log
SO=$(ls _flashkda_fp32_C*.so | head -1)
cp "$SO" /fkda/out/_flashkda_fp32_C.abi3.so
echo "17a037d98da546deb4591e967cf961a43c034d8b" > /fkda/out/FLASHKDA_REF
echo "5c149f52a436782210263fb2f19b354443a61c6a" > /fkda/out/CUTLASS_REF
echo "ddd6fbca148a867aad1fcab7ec72f582b9977db4" > /fkda/out/VLLM_REF
echo "flashkda_registration.cpp @ ddd6fbca, namespace _flashkda_C -> _flashkda_fp32_C" > /fkda/out/REGISTRATION_NOTE
echo "$ARCH" > /fkda/out/BUILD_ARCH
ls -la /fkda/out/
EOS
exec "$REPO/tests/fkda/docker_cpu.sh" -c 'bash /fkda/_stage_build.sh'
