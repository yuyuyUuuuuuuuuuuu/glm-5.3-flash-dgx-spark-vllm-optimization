#!/usr/bin/env bash
# review: compose the r16msp tree (S1 env + GLM53_MHC_SP=1) with the production overlay order, then run
# patch_tf_bundle.py a SECOND time in the same container; copy out the three edited files after each pass.
set -uo pipefail
REPO=${TF_EXL3_KITS:-$HOME}/tf-exl3-fork-wt-mhcsp
KIT=${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16msp
PROD=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher; AS=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}
IMG=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
SP=/usr/local/lib/python3.12/dist-packages
OUT="$1"; mkdir -p "$OUT"; SPV="${2:-1}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
L=$tmp/new; mkdir -p $L; cp -a "$PROD/start.sh" "$PROD/overlay" "$L/"
rm -rf "$L/overlay/tf/site" "$L/overlay/tf/overlay"; cp -a "$KIT/site" "$L/overlay/tf/site"; cp -a "$KIT/overlay" "$L/overlay/tf/overlay"
cp -p "$KIT"/launcher/overlay/*.py "$L/overlay/"; cp -p "$KIT/launcher/start.sh" "$L/start.sh"
envkeys() { grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1" | sed -E 's/[[:space:]]+#.*$//'; }
r16names=$(grep -E '^[A-Z0-9_]+=' "$KIT/env.r16" | cut -d= -f1 | paste -sd'|')
envkeys "$PROD/env_nonsecret.txt" | grep -vE "^($r16names|GLM53_REJECTION_METHOD)=" > "$tmp/base.env"
{ cat "$tmp/base.env"; envkeys "$KIT/env.r16" | grep -vE '^(GLM53_DEC_FP8ROOF|GLM53_DEC_MOEGLUE_WARM|GLM53_DEC_HOSTLOOP|GLM53_KPOOL_RING)='; } > "$tmp/S.env"
[ "${S2:-0}" = 1 ] && { cat "$tmp/base.env"; envkeys "$KIT/env.r16"; } > "$tmp/S.env"
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$L/start.sh")
args=(-v "$L/overlay/exl3.py:/opt/glm53/exl3.py:ro" -v "$L/overlay/tf:/opt/glm53/tf:ro"
      -v "$AS/fi618/flashinfer:$SP/flashinfer:ro" -v "$AS/fi618/flashinfer_cubin:$SP/flashinfer_cubin:ro"
      -v "$AS/vllm-patches/cuda.py.exl3.patched:$SP/vllm/platforms/cuda.py:ro"
      -v "$AS/vllm-patches/flashinfer_mla_sparse_sm90.py.patched:$SP/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:ro")
for p in $order; do [ -f "$L/overlay/$p" ] && args+=(-v "$L/overlay/$p:/opt/glm53/$p:ro"); done
flock /tmp/tf-gpu-bench.lock docker run --rm --network none --memory 16g --name "tf-exl3-review-$$" --env-file "$tmp/S.env" \
  -e GLM53_REJECTION_METHOD= -e GLM53_KDA_STRIDED_QKV= -e GLM53_KPOOL_DROP_LOWEST= -e GLM53_MHC_SP="$SPV" -e GLM53_APC_PRIOR_CHECKPOINT=0 \
  -e ORDER="$(echo $order)" -e PYTHONDONTWRITEBYTECODE=1 "${args[@]}" -v "$OUT:/out" --entrypoint bash "$IMG" -c '
SP=/usr/local/lib/python3.12/dist-packages; fail=0
for p in $ORDER; do
  if [ -f /opt/glm53/$p ]; then out=$(python3 /opt/glm53/$p 2>&1); rc=$?; echo "rc=$rc $p"
    case $p in patch_tf_bundle.py) echo "$out" | grep -v "installed .* entries" | sed "s/^/    /";; esac
    [ $rc -ne 0 ] && { echo "$out" | tail -n 5; fail=1; }
  fi
done
mkdir -p /out/p1 /out/p2
cp $SP/vllm/models/glm5next/nvidia/model.py $SP/glm53_prefill_quickwins.py $SP/glm53_moeglue.py $SP/integrate.py /out/p1/
echo "=== second bundle pass"
python3 /opt/glm53/patch_tf_bundle.py 2>&1 | grep -v "installed .* entries"; echo "rc2=$?"
cp $SP/vllm/models/glm5next/nvidia/model.py $SP/glm53_prefill_quickwins.py $SP/glm53_moeglue.py /out/p2/
echo "chain fail=$fail"; chmod -R a+rwX /out' > "$OUT/compose.log" 2>&1
echo "rc=$?"
