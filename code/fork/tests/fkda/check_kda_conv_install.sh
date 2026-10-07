#!/usr/bin/env bash
# FKDA review B1 end-to-end (CPU-only, no GPU lock): with the bundle composed
# the way the launcher does it (site/ installed, then the overlays), run
# patch_flashkda.py with GLM53_KDA_FLASHKDA=1 and then the REAL quickwins
# runtime path: import the patched kda.py and let
# glm53_prefill_quickwins.install_item("kda_conv", ...) transplant
# Glm5NextLinearAttention._forward -- which succeeds only if the
# extend_quickwins_allowlist fingerprint landed in the installed VERIFIED set.
# Also proves the OFF path: with GLM53_KDA_FLASHKDA=0 nothing is touched.
# r16z: an optional build argument (fkda | fkda2 | fkda3, default the shipped fkda build) stages that build's
# overlay pair and sets GLM53_KDA_FLASHKDA_V; the expected patched _forward fingerprint follows the kda.py call
# text (fkda/fkda2: 9715f9b548cfa694 = production's; fkda3: eb0e8dedaee6deb6, the direct-output call).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VER="${1:-fkda}"
case "$VER" in
  fkda)  PAIR="glm53_flashkda.py _flashkda_fp32_C.abi3.so";   VVAL="";  EXP_FP=9715f9b548cfa694; OUT_RE=0;;
  fkda2) PAIR="glm53_flashkda2.py _flashkda_fp32_C2.abi3.so"; VVAL=2;   EXP_FP=9715f9b548cfa694; OUT_RE=0;;
  fkda3) PAIR="glm53_flashkda3.py _flashkda_fp32_C3.abi3.so"; VVAL=3;   EXP_FP=eb0e8dedaee6deb6; OUT_RE=1;;
  *) echo "check_kda_conv_install.sh: build must be fkda, fkda2 or fkda3 (got $VER)" >&2; exit 2;;
esac
"${REPO}/tests/fkda/docker_cpu.sh" -c '
set -e
SP=/usr/local/lib/python3.12/dist-packages
# compose: install_site() would copy the bundle modules into dist-packages
cp /w/glm53_prefill_quickwins.py "$SP/"
mkdir -p /tmp/fk/stage && cp /w/overlay/patch_flashkda.py /tmp/fk/stage/ && for f in '"$PAIR"'; do cp "/w/overlay/$f" /tmp/fk/stage/; done
VV='"$VVAL"'
if [ -n "$VV" ]; then export GLM53_KDA_FLASHKDA_V=$VV; fi
# ---- OFF: 0 -> nothing touched (the module and kda.py stay stock)
GLM53_KDA_FLASHKDA=0 GLM53_TF_OVERLAY=/tmp/fk/stage GLM53_SITEPKG=$SP python3 /tmp/fk/stage/patch_flashkda.py
grep -c "1e4f45149fceddf5" "$SP/glm53_prefill_quickwins.py" | grep -qx 1
grep -qF "frozenset({\"1e4f45149fceddf5\"})" "$SP/glm53_prefill_quickwins.py"
! grep -q "glm53_flashkda" "$SP/vllm/models/glm5next/nvidia/kda.py"
echo "OFF: kda.py + quickwins module untouched"
# ---- ON: install, then the real runtime transplant
GLM53_KDA_FLASHKDA=1 GLM53_TF_OVERLAY=/tmp/fk/stage GLM53_SITEPKG=$SP python3 /tmp/fk/stage/patch_flashkda.py
export EXPECT_OUT='"$OUT_RE"'
cd /tmp && python3 - <<"PY"   # not /w: the engine imports glm53_prefill_quickwins from dist-packages
import glm53_prefill_quickwins as Q
import vllm.models.glm5next.nvidia.kda as K
fp = Q.source_fingerprint(K.Glm5NextLinearAttention._forward)
assert fp == "'"$EXP_FP"'", fp   # fkda/fkda2: production call text; fkda3: the direct-output call
assert fp in Q._fps("kda_conv", "Glm5NextLinearAttention._forward"), sorted(Q._fps("kda_conv", "Glm5NextLinearAttention._forward"))
Q.install_item("kda_conv", K)
assert "kda_conv" in Q._STATE["installed"], Q._STATE["installed"]
import inspect
src = inspect.getsource(K.Glm5NextLinearAttention._forward)
assert "_glm53_qw_kda_conv" in src
import os
if os.environ.get("EXPECT_OUT") == "1":
    assert "out=(None if use_spec else core_attn_out[:, :num_actual_tokens])" in src   # the fkda3 call text
else:
    assert "out=(None if use_spec" not in src                                          # fkda/fkda2: no out= kwarg
assert Q.STATS["kda_conv_prod"] == 0 and Q.STATS["kda_conv_fast"] == 0
print("RUNTIME: kda_conv transplanted onto the FlashKDA-patched _forward (fingerprint", fp, "accepted)")
print("installed:", Q._STATE["installed"]["kda_conv"])
PY
' 2>&1 | grep -v "^W0\|^INFO"
echo "KDA_CONV INSTALL E2E ($VER): ALL OK"
echo "KDA_CONV INSTALL E2E: ALL OK"
