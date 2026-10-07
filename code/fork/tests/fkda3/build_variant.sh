#!/usr/bin/env bash
# FKDA3: build a FlashKDA variant = FlashKDA 17a037d + tests/fkda2/patch_flashkda_precision.py (the fkda2 precision
# edits, default switches) + optionally tests/fkda3/patch_flashkda_fkda3.py, with the given -D switches, under a chosen
# extension/op namespace (default _flashkda_fp32_C; a different name lets two builds load in one process for overlap
# experiments). Same flags/arch/image as tests/fkda2/build_variant.sh (CPU-only compile, no GPU lock).
# Usage: tests/fkda3/build_variant.sh <name> <modname> <fkda3: 0 none | 1 precision patch | 2 + pipeline experiment> [-DX=Y ...]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="$1"; MOD="$2"; P3="$3"; shift 3
DEFS="$*"
SRC="${FKDA2_SRC:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/fkda2/src}"
BUILDS="${FKDA3_BUILDS:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/fkda3/builds}"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
mkdir -p "$BUILDS/$NAME"
avail=$(free -g | awk 'NR==2{print $7}')
[ "$avail" -ge 20 ] || { echo "build_variant: host available ${avail} GB < 20" >&2; exit 3; }
cat > "$BUILDS/$NAME/_build.sh" <<EOS
set -euo pipefail
B=/tmp/b && rm -rf \$B && mkdir -p \$B
cp -a /src/fk \$B/flashkda
python3 /w/tests/fkda2/patch_flashkda_precision.py \$B/flashkda
if [ "$P3" -ge 1 ]; then python3 /w/tests/fkda3/patch_flashkda_fkda3.py \$B/flashkda; fi
if [ "$P3" -ge 2 ]; then python3 /w/tests/fkda3/patch_flashkda_fkda3_pipeline.py \$B/flashkda; fi
cp -a /src/vllm-csrc \$B/vllm-csrc
sed -i 's/_flashkda_C/$MOD/g' \$B/vllm-csrc/flashkda_registration.cpp
cp /w/tests/fkda/setup_fkda_fp32.py \$B/setup.py
python3 - \$B/setup.py "$DEFS" "$MOD" <<'PY'
import sys
p, defs, mod = sys.argv[1], sys.argv[2].split(), sys.argv[3]
s = open(p).read()
anchor = '"-U__CUDA_NO_HALF2_OPERATORS__",'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "".join(f' "{d}",' for d in defs))
cxx = '"cxx": ["-O3"],'
assert s.count(cxx) == 1
s = s.replace(cxx, '"cxx": ["-O3"' + "".join(f', "{d}"' for d in defs) + '],')
assert s.count('"_flashkda_fp32_C",') == 1
s = s.replace('"_flashkda_fp32_C",', f'"{mod}",')
open(p, "w").write(s)
PY
cd \$B
export PF3000_NVCC_ARCH=sm_121a MAX_JOBS=\${MAX_JOBS:-16}
python3 setup.py build_ext --inplace > /out/build_ext.log 2>&1 || { tail -40 /out/build_ext.log; exit 1; }
cp ${MOD}*.so /out/$MOD.abi3.so
cp -a \$B/flashkda/csrc /out/csrc
echo "$DEFS p3=$P3 mod=$MOD" > /out/DEFS
sha256sum /out/$MOD.abi3.so
EOS
exec docker run --rm --network none --name "fkda3-cpu-$NAME-$$" -u "$(id -u):$(id -g)" \
  -v "$REPO:/w:ro" -v "$SRC:/src:ro" -v "$BUILDS/$NAME:/out" -w /w \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/tmp/ext \
  --entrypoint bash "$IMAGE" /out/_build.sh
