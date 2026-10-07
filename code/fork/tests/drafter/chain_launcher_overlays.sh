#!/usr/bin/env bash
# Apply the launcher's runtime overlay chain (GLM53_OVERLAY_ORDER of <tree>/start.sh, the exact in-container
# `python3 /opt/glm53/<patch>` loop of the inner start script) to the image's files, with this repo's three
# patches inserted after patch_dense_fp8.py as docs/DRAFTER_IMPL.md "Production install" prescribes, then check:
#   - every launcher patch and ours exits 0; ours print "stock sha256 match" (their target files were not
#     touched by the launcher chain) and are idempotent on a second run;
#   - vllm/model_executor/layers/quantization/exl3.py after the chain is byte-identical to the launcher overlay
#     (patch_dense_fp8.py installs /opt/glm53/exl3.py wholesale over the image's module);
#   - the patched modules import, and the drafter FP8 config built by patch_drafter_fp8.py gets the INSTALLED
#     exl3.py's Glm53DenseFp8Method with the right constructor shape.
# CPU only, --network none, --rm; nothing leaves the container except the log. No production access.
# Usage: tests/drafter/chain_launcher_overlays.sh <launcher tree> <env file (KEY=VALUE lines, no secrets)>
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE="${1:?launcher tree}"; ENVF="${2:?env file}"
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
LIVE="$REPO/docs/ref/prod_live/overlay_exl3.py"
want=$(sha256sum "$LIVE" | cut -d' ' -f1)
have=$(sha256sum "$TREE/overlay/exl3.py" | cut -d' ' -f1)
echo "image $(docker image inspect "$IMG" --format '{{.Id}}' | cut -c8-19); tree $TREE"
# the env file is recorded by path, sha256 and variable names only (never its values)
echo "env file $ENVF sha256 $(sha256sum "$ENVF" | cut -c1-64); $(grep -cE '^[A-Za-z_][A-Za-z0-9_]*=' "$ENVF") variables: $(grep -oE '^[A-Za-z_][A-Za-z0-9_]*' "$ENVF" | tr '\n' ' ')"
echo "tree overlay/exl3.py sha256 $have; docs/ref/prod_live/overlay_exl3.py $want"
[ "$have" = "$want" ] || { echo "the tree's exl3.py is not the live overlay; refusing"; exit 2; }
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$TREE/start.sh")
[ -n "$order" ] || { echo "no GLM53_OVERLAY_ORDER in $TREE/start.sh"; exit 2; }
ours="patch_spec_resample_noise.py patch_drafter_fp8.py patch_drafter_lmhead_fp8.py"
full=""
for p in $order; do full="$full $p"; [ "$p" = patch_dense_fp8.py ] && full="$full $ours"; done
echo "order:$full"
args=(-v "$TREE/overlay/exl3.py:/opt/glm53/exl3.py:ro")
for p in $order; do
  if [ -f "$TREE/overlay/$p" ]; then args+=(-v "$TREE/overlay/$p:/opt/glm53/$p:ro"); echo "  $(sha256sum "$TREE/overlay/$p" | cut -c1-12) $p"; fi
done
for p in $ours; do args+=(-v "$REPO/overlay/$p:/opt/glm53/$p:ro"); echo "  $(sha256sum "$REPO/overlay/$p" | cut -c1-12) $p (this repo)"; done
docker run --rm --network none --memory 16g --name "tf-exl3-chain-$$" --env-file "$ENVF" \
  -e GLM53_SPEC_RESAMPLE_INDEPENDENT=1 -e GLM53_DRAFT_FP8=1 -e GLM53_DRAFT_LMHEAD_FP8=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e ORDER="$full" -e OURS="$ours" -e WANT="$want" "${args[@]}" --entrypoint bash "$IMG" -c '
set -u
fail=0
for p in $ORDER; do
  if [ -f /opt/glm53/$p ]; then
    out=$(python3 /opt/glm53/$p 2>&1); rc=$?
    case " $OURS " in *" $p "*) echo "rc=$rc $p: $(echo "$out" | tail -n 1)";; *) echo "rc=$rc $p";; esac
    [ $rc -ne 0 ] && { echo "$out" | tail -n 5; fail=1; }
  else
    echo "absent $p"
  fi
done
Q=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
got=$(sha256sum $Q | cut -d" " -f1)
echo "installed quantization/exl3.py sha256 $got (live overlay: $([ "$got" = "$WANT" ] && echo yes || echo NO))"
[ "$got" = "$WANT" ] || fail=1
for p in $OURS; do out=$(python3 /opt/glm53/$p 2>&1); rc=$?; echo "second run rc=$rc: $(echo "$out" | tail -n 1)"; [ $rc -ne 0 ] && fail=1; done
python3 - <<PY || fail=1
import inspect, torch
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as R
import vllm.model_executor.models.qwen3_dflash as QD
import vllm.model_executor.models.qwen3_dflash2 as QD2
import vllm.v1.worker.gpu.spec_decode.dflash.utils as DU
import vllm.model_executor.layers.quantization.exl3 as EXL3
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
print("imports OK; installed Glm53DenseFp8Method" + str(inspect.signature(EXL3.Glm53DenseFp8Method.__init__)),
      "; mx path", hasattr(EXL3, "mx_enabled"), "; fast path", hasattr(EXL3, "exl3_moe_fast_requested"))
assert "glm53_candidate_head" in inspect.getsource(QD2) and hasattr(DU, "_glm53_attach_candidate_head")
assert QD._GLM53_DRAFT_FP8_INSTALLED == "layers", QD._GLM53_DRAFT_FP8_INSTALLED
cfg = QD._glm53_draft_fp8_quant_config(None)
class FakeLinear(LinearBase):
    pass
lin = FakeLinear.__new__(FakeLinear)
torch.nn.Module.__init__(lin)
m = cfg.get_quant_method(lin, "model.layers.45.mlp.down_proj")
m2 = cfg.get_quant_method(lin, "model.layers.45.self_attn.kernel_projection")
ok = type(m) is EXL3.Glm53DenseFp8Method and m.group == "draft" and getattr(m, "prefix", None) == "model.layers.45.mlp.down_proj" \
     and type(m2) is UnquantizedLinearMethod
print("drafter FP8 config -> %s(group=%r, prefix=%r); other linear -> %s: %s" % (type(m).__name__, m.group,
      getattr(m, "prefix", None), type(m2).__name__, "OK" if ok else "WRONG"))
assert ok
print("rejection_sampler_utils independent-noise switch:", R._GLM53_RESAMPLE_INDEPENDENT if hasattr(R, "_GLM53_RESAMPLE_INDEPENDENT") else "?")
PY
echo "chain result: $([ $fail -eq 0 ] && echo ALL OK || echo FAILED)"
exit $fail'
