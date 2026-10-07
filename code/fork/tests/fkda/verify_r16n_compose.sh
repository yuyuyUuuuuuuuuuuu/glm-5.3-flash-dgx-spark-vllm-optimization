#!/usr/bin/env bash
# Adversarial r16n verification (CPU only, no GPU lock): compose a KIT exactly the way the launcher does
# (kit site/ -> /opt/glm53/tf/site, kit overlay/ -> /opt/glm53/tf/overlay, kit launcher/overlay/patch_tf_bundle.py)
# inside the production image, then exercise the REAL quickwins runtime path on the composed tree.
#   A  FLASHKDA=1 + every other r16 switch on (kdaqkv, kpooldown, block, capture install): bundle ok, kda_conv
#      transplants onto the FlashKDA-patched _forward, the transplanted source holds BOTH edits in the right order
#   B  re-run the bundle in the SAME container fs (docker restart / a second bundle pass): must stay idempotent
#   C  FLASHKDA unset: the quickwins module and kda.py are stock
# Usage: tests/fkda/verify_r16n_compose.sh <kit dir>
set -uo pipefail
KIT="$(cd "${1:?kit dir}" && pwd)"
IMAGE="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
run() {  # $1 = case script
  docker run --rm --network none --name "fkda-cpu-verify-$$-$RANDOM" -u root \
    -v "$KIT/site:/opt/glm53/tf/site:ro" -v "$KIT/overlay:/opt/glm53/tf/overlay:ro" \
    -v "$KIT/launcher/overlay/patch_tf_bundle.py:/opt/glm53/patch_tf_bundle.py:ro" \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$IMAGE" -c "$1" 2>&1 \
    | grep -v '^W1\|^WARNING\|^INFO\|No module named .vllm._C'
}
PYCHECK='
import inspect, sys
import glm53_prefill_quickwins as Q
import vllm.models.glm5next.nvidia.kda as K
fp = Q.source_fingerprint(K.Glm5NextLinearAttention._forward)
print("forward fp", fp, "in VERIFIED:", fp in Q._fps("kda_conv", "Glm5NextLinearAttention._forward"))
Q.install_item("kda_conv", K)
src = inspect.getsource(K.Glm5NextLinearAttention._forward)
i_conv = src.find("_glm53_qw_kda_conv(")
i_fk = src.find("glm53_flashkda.chunk_prefill(")
i_tr = src.find("chunk_kda_with_fused_gate(")
i_stock_conv = src.find("qkv_ns = causal_conv1d_fn(")
print("transplanted: qw_conv@%d flashkda@%d triton_else@%d stock_prefill_conv@%d" % (i_conv, i_fk, i_tr, i_stock_conv))
assert i_conv > 0 and i_fk > i_conv and i_tr > i_fk and i_stock_conv < 0, "combined function wrong"
assert src.count("_glm53_qw_kda_conv(") == 1 and src.count("glm53_flashkda.chunk_prefill(") == 1
# the decode-branch conv (causal_conv1d_update) must remain for num_decodes
assert src.count("causal_conv1d_update(") == 2, src.count("causal_conv1d_update(")
print("COMBINED OK")
'
echo "== A: FLASHKDA=1 + kdaqkv + kpooldown + block, QUICKWINS runtime"
run "set -e
export GLM53_KDA_FLASHKDA=1 GLM53_KDA_STRIDED_QKV=1 GLM53_KPOOL_DROP_LOWEST=1 GLM53_KPOOL_SEED_STRIDE=1 GLM53_KPOOL_RING=1 \
  GLM53_SPEC_RESAMPLE_INDEPENDENT=1 GLM53_REJECTION_METHOD=block GLM53_DRAFT_FP8=layers,fc
python3 /opt/glm53/patch_tf_bundle.py | grep -v 'installed .* entries'
cd /tmp && python3 -c '$PYCHECK'
echo '-- B: second bundle pass in the same fs (docker restart)'
if python3 /opt/glm53/patch_tf_bundle.py > /tmp/b2.log 2>&1; then echo 'B second pass rc=0'; grep -a 'flashkda' /tmp/b2.log; cd /tmp && python3 -c '$PYCHECK'; else echo 'B second pass FAILED rc='\$?; tail -3 /tmp/b2.log; fi
"
echo "== C: FLASHKDA unset -> quickwins module + kda.py stock"
run "set -e
python3 /opt/glm53/patch_tf_bundle.py | grep -a flashkda
SP=/usr/local/lib/python3.12/dist-packages
grep -c 'frozenset({\"1e4f45149fceddf5\"})' \$SP/glm53_prefill_quickwins.py
cmp \$SP/glm53_prefill_quickwins.py /opt/glm53/tf/site/glm53_prefill_quickwins.py && echo 'C quickwins == kit site byte for byte'
! grep -q glm53_flashkda \$SP/vllm/models/glm5next/nvidia/kda.py && echo 'C kda.py stock'
! ls \$SP/_flashkda_fp32_C* 2>/dev/null && echo 'C no extension installed'
"
