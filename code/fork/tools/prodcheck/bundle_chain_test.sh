#!/usr/bin/env bash
# Replay the LIVE launcher overlay chain (GLM53_OVERLAY_ORDER of the copied df864b5 start.sh) with patch_tf_bundle.py
# inserted after patch_dense_fp8.py, exactly as the edited start.sh will run it in the container, then check the result.
# Usage: bundle_chain_test.sh <mode: on|off>
set -uo pipefail
REPO=${HOME}/tf-exl3-fork; TREE=$REPO/artifacts/launcher-live-df864b5/tree; ENVF=$REPO/artifacts/launcher-live-df864b5/env.nosecrets
B=$REPO/artifacts/bundle; SO=$REPO/docs/ref/prod_live/so/exllamav3_ext.prod.so
IMG=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor; SP=/usr/local/lib/python3.12/dist-packages
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$TREE/start.sh")
full=""; for p in $order; do full="$full $p"; [ "$p" = patch_dense_fp8.py ] && full="$full patch_tf_bundle.py"; done
args=(-v "$TREE/overlay/exl3.py:/opt/glm53/exl3.py:ro" -v "$B/patch_tf_bundle.py:/opt/glm53/patch_tf_bundle.py:ro" -v "$B/tf:/opt/glm53/tf:ro" -v "$SO:$SP/exllamav3_ext.cpython-312-aarch64-linux-gnu.so:ro")
for p in $order; do [ -f "$TREE/overlay/$p" ] && args+=(-v "$TREE/overlay/$p:/opt/glm53/$p:ro"); done
[ -d "$TREE/ablit" ] && args+=(-v "$TREE/ablit:/opt/glm53/ablit:ro")
if [ "$1" = on ]; then envs=(-e TF_EXL3_MOE=1 -e GLM53_SPEC_RESAMPLE_INDEPENDENT=1 -e GLM53_DRAFT_FP8= -e GLM53_DRAFT_LMHEAD_FP8=)
else envs=(-e TF_EXL3_MOE= -e GLM53_SPEC_RESAMPLE_INDEPENDENT= -e GLM53_DRAFT_FP8= -e GLM53_DRAFT_LMHEAD_FP8=); fi
docker run --rm --gpus all --network none --memory 16g --name "tf-exl3-gpu-chain-$$" --env-file "$ENVF" "${envs[@]}" \
  -e ORDER="$full" -e PYTHONDONTWRITEBYTECODE=1 "${args[@]}" --entrypoint bash "$IMG" -c '
fail=0
for p in $ORDER; do
  if [ -f /opt/glm53/$p ]; then out=$(python3 /opt/glm53/$p 2>&1); rc=$?
    case $p in patch_tf_bundle.py) echo "$out" | sed "s/^/    /";; esac
    [ $rc -ne 0 ] && { echo "rc=$rc $p"; echo "$out" | tail -5; fail=1; }
  else echo "absent $p"; fi
done
echo "chain: $([ $fail = 0 ] && echo ALL rc=0 || echo FAILURES)"
echo "exl3.py sha: $(sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py | cut -c1-16)"
echo "resample patch marker in rejection_sampler_utils: $(grep -c "glm53-resample-noise" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)"
cd /tmp && python3 - <<PY 2>&1 | grep -v "^W09"
import os
from vllm.plugins import load_general_plugins
load_general_plugins()
import exllamav3_ext, vllm.model_executor.layers.quantization.exl3 as q
print("TF_EXL3_MOE=%r dispatcher=%s build_hook=%s" % (os.environ.get("TF_EXL3_MOE"), bool(getattr(exllamav3_ext.exl3_moe, "_tf_exl3_dispatch", False)), bool(getattr(q.build_exl3_fused_state, "_tf_exl3_hook", False))))
PY
exit $fail'
