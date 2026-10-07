#!/usr/bin/env bash
# FKDA2: build a _flashkda_fp32_C variant = FlashKDA 17a037d + tests/fkda2/patch_flashkda_precision.py with the given
# -D switches, exactly the fkda build otherwise (tests/fkda/build_flashkda_fp32.sh: same registration-shim rename,
# same setup flags, sm_121a, CPU-only compile in the production image, no GPU lock).
# Usage: tests/fkda2/build_variant.sh <name> [-DFKDA2_X=0 ...]
# Sources: $FKDA2_SRC/{fk,vllm-csrc} (persistent copy of pfkda's /tmp/pf3000 staging, csrc identical to
# fkda-evidence/flashkda-src). Output: $FKDA2_BUILDS/<name>/_flashkda_fp32_C.abi3.so + DEFS + build_ext.log
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="$1"; shift
DEFS="$*"
SRC="${FKDA2_SRC:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/fkda2/src}"
BUILDS="${FKDA2_BUILDS:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/fkda2/builds}"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
mkdir -p "$BUILDS/$NAME"
avail=$(free -g | awk 'NR==2{print $7}')
[ "$avail" -ge 20 ] || { echo "build_variant: host available ${avail} GB < 20" >&2; exit 3; }
cat > "$BUILDS/$NAME/_build.sh" <<EOS
set -euo pipefail
B=/tmp/b && rm -rf \$B && mkdir -p \$B
cp -a /src/fk \$B/flashkda
python3 /w/tests/fkda2/patch_flashkda_precision.py \$B/flashkda
cp -a /src/vllm-csrc \$B/vllm-csrc
sed -i 's/_flashkda_C/_flashkda_fp32_C/g' \$B/vllm-csrc/flashkda_registration.cpp
! grep -q "_flashkda_C" \$B/vllm-csrc/flashkda_registration.cpp
cp /w/tests/fkda/setup_fkda_fp32.py \$B/setup.py
# the variant switches go to nvcc (appended to the unchanged flag list)
python3 - \$B/setup.py "$DEFS" <<'PY'
import sys
p, defs = sys.argv[1], sys.argv[2].split()
s = open(p).read()
anchor = '"-U__CUDA_NO_HALF2_OPERATORS__",'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "".join(f' "{d}",' for d in defs))
open(p, "w").write(s)
PY
cd \$B
export PF3000_NVCC_ARCH=sm_121a MAX_JOBS=\${MAX_JOBS:-16}
python3 setup.py build_ext --inplace > /out/build_ext.log 2>&1 || { tail -40 /out/build_ext.log; exit 1; }
cp _flashkda_fp32_C*.so /out/_flashkda_fp32_C.abi3.so
cp -a \$B/flashkda/csrc /out/csrc
echo "$DEFS" > /out/DEFS
sha256sum /out/_flashkda_fp32_C.abi3.so
EOS
exec docker run --rm --network none --name "fkda2-cpu-$$" -u "$(id -u):$(id -g)" \
  -v "$REPO:/w:ro" -v "$SRC:/src:ro" -v "$BUILDS/$NAME:/out" -w /w \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/tmp/ext \
  --entrypoint bash "$IMAGE" /out/_build.sh
