#!/usr/bin/env bash
# State-handoff harness (nodeC only, production image, --network none, no production access, no secret): runs
# tests/handoff/run_engine.py once per feature configuration, each in a fresh --rm container composed EXACTLY the way
# both production ranks compose theirs after the R16 rollout:
#   - the launcher's in-container overlay loop (GLM53_OVERLAY_ORDER of the kit's start.sh, every patch from the
#     read-only production launcher copy, the 3 APC overlays + patch_tf_bundle.py of the kit) with THIS worktree's
#     bundle as /opt/glm53/tf (site = the worktree's modules + the kit's AOT .so + dist-info; overlay = worktree overlay/)
#   - production's SM90_KV mounts (flashinfer 0.6.18, patched cuda.py / flashinfer_mla_sparse_sm90.py)
#   - production's non-secret env of 09-28 (tests/handoff/env_nonsecret.txt, HANDOFF_ENV) + the R16 launcher APC flags; every R16 feature knob EMPTY
#     except what the configuration sets (the launcher forwards unset knobs as "")
# The container runs as root like the launcher's inner script (the overlay loop edits site-packages); its outputs are
# chowned back to the caller. GPU work is serialized with every other GPU job through flock /tmp/tf-gpu-bench.lock and
# refused while another tf-exl3 GPU container runs or host memory is short (tests/gpu_run.sh's rules).
# Usage: tests/handoff/run.sh <out dir> <label> [NAME=value ...] [-- run_engine.py args]
#   e.g. tests/handoff/run.sh /tmp/h off
#        tests/handoff/run.sh /tmp/h qw_kda_conv GLM53_PREFILL_QUICKWINS=kda_conv -- --consistency 40
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"; LABEL="$2"; shift 2
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
declare -a KV=() PYARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; PYARGS=("$@"); break; fi
  KV+=("$1"); shift
done
PROD="${PROD_LAUNCHER:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher}"
# the kit: HANDOFF_KIT, else this repository's code/kit (its start.sh, launcher overlays and dist-info), else a kit dir
if [ -n "${HANDOFF_KIT:-}" ]; then KIT="${HANDOFF_KIT}"
elif [ -f "$REPO/../kit/launcher/start.sh" ]; then KIT="$(cd "$REPO/../kit" && pwd)"
else KIT="${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16"; fi
AS="${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}"
IMG="${TF_EXL3_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
M=${TF_EXL3_MODELS:-$HOME/models}
MINI="${HANDOFF_MODEL:-$M/GLM-5.3-Flash-handoff-mini}"
CACHE="${HANDOFF_CACHE:-${HOME}/.cache/tf-exl3-handoff}"
SP=/usr/local/lib/python3.12/dist-packages
LOCK="${GPU_LOCK:-/tmp/tf-gpu-bench.lock}"
[ -f "$MINI/model.safetensors" ] || { echo "no $MINI (python3 tests/handoff/build_mini.py)"; exit 2; }
for d in "$PROD/overlay" "$AS/fi618/flashinfer" "$AS/fi618/flashinfer_cubin" "$AS/vllm-patches/cuda.py.exl3.patched" \
         "$M/GLM-5.3-Flash-DFlash2-dc77ff1c/model.safetensors" "$M/GLM-OCR/tokenizer.json" "$KIT/launcher/start.sh"; do
  [ -e "$d" ] || { echo "no $d (REPRODUCE.md section 13; tests/derive_assets.sh)"; exit 2; }
done
(cd "$MINI" && sha256sum --quiet -c MANIFEST.sha256) || { echo "$MINI MANIFEST mismatch"; exit 2; }
mkdir -p "$CACHE"/{triton,tilelang,vllm,fi_ws,torch_ext}

# ---- stage the post-rollout launcher tree with THIS worktree's bundle
tmp=$(mktemp -d "${TMPDIR:-/tmp}/handoff-stage.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
L="$tmp/launcher"; mkdir -p "$L/overlay/tf/site" "$L/overlay/tf/overlay"
cp -a "$PROD/overlay/." "$L/overlay/"
rm -rf "$L/overlay/tf/site" "$L/overlay/tf/overlay"; mkdir -p "$L/overlay/tf/site" "$L/overlay/tf/overlay"
mods=$(python3 - "$REPO/setup.py" <<'PY'
import ast, re, sys
print(" ".join(ast.literal_eval(re.search(r"py_modules=(\[.*?\])", open(sys.argv[1]).read(), re.S).group(1))))
PY
)
for m in $mods; do cp -p "$REPO/$m.py" "$L/overlay/tf/site/"; done
for so in "$KIT"/site/*.so; do
  b=$(basename "$so")
  # opt-decodekit: only a NON-EMPTY worktree build overrides the kit's (a GPU_RUN_BIND of a single .so onto /w leaves a
  # 0-byte root-owned mount point in the worktree, which silently disabled GLM53_MLA_PREFILL in every later run)
  if [ -s "$REPO/$b" ]; then cp -p "$REPO/$b" "$L/overlay/tf/site/"; echo "NOTE: site/$b from the worktree (not the kit)"
  else [ -e "$REPO/$b" ] && echo "NOTE: ignoring the EMPTY worktree $b (the kit's is used)"; cp -p "$so" "$L/overlay/tf/site/"; fi
done
# decode5: a worktree AOT extension the kit does not ship yet (tf_dlmh_ext for GLM53_DEC_DLMH) is staged too
for so in "$REPO"/*.so; do
  b=$(basename "$so"); [ -s "$so" ] && [ ! -e "$KIT/site/$b" ] && { cp -p "$so" "$L/overlay/tf/site/"; echo "NOTE: site/$b from the worktree (new, not in the kit)"; }
done
for so in "$L"/overlay/tf/site/*.so; do [ -s "$so" ] || { echo "ABORT: empty $so in the staged site/"; exit 3; }; done
cp -a "$KIT/site/tf_exl3_moe-0.1.0.dist-info" "$L/overlay/tf/site/"
for f in "$REPO"/overlay/*.py; do [ "$(basename "$f")" = patch_tf_bundle.py ] || cp -p "$f" "$L/overlay/tf/overlay/"; done
# bundle overlay kernels/extensions (w8a8 AOT ext; glm53_moe_e4m3_ext: GLM53_MOE_E4M3 / GLM53_MOE_FUSED16 install it from here when enabled)
for f in "$REPO"/overlay/*.so; do [ -e "$f" ] && cp -p "$f" "$L/overlay/tf/overlay/"; done
# r16m: the FlashKDA extension the flashkda overlay installs when GLM53_KDA_FLASHKDA=1 (absent = the switch cannot
# be turned on in this run; tests/fkda/build_flashkda_fp32.sh builds it into /tmp/fkda/out)
cp -p "$REPO"/overlay/_flashkda_fp32_C.abi3.so "$L/overlay/tf/overlay/" 2>/dev/null || echo "NOTE: no overlay/_flashkda_fp32_C.abi3.so; GLM53_KDA_FLASHKDA=1 runs will fail closed"
cp -p "$REPO/overlay/patch_tf_bundle.py" "$L/overlay/"
cp -p "$KIT"/launcher/overlay/patch_{hybrid_prefix_hit,mamba_align_chunking,apc_per_group_retention}.py "$L/overlay/"
cp -p "$KIT/launcher/start.sh" "$L/start.sh"
order=$(awk '/^GLM53_OVERLAY_ORDER=\(/{f=1;next} f&&/^\)/{exit} f{print $1}' "$L/start.sh")
[ -n "$order" ] || { echo "no GLM53_OVERLAY_ORDER"; exit 2; }

# ---- env: production non-secret env + APC flags; every R16 knob empty; then this configuration
envf="$tmp/env"
grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${HANDOFF_ENV:-$REPO/tests/handoff/env_nonsecret.txt}" | sed -E 's/[[:space:]]+#.*$//' > "$envf"
source "$REPO/tests/r16/flags.sh"
for k in $R16_KNOBS; do grep -v "^$k=" "$envf" > "$envf.1"; mv "$envf.1" "$envf"; echo "$k=" >> "$envf"; done
echo "$R16_LAUNCHER_ENV" | tr ';' '\n' >> "$envf"
cat >> "$envf" <<EOF
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32256
VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=0
GLM53_APC_PRIOR_CHECKPOINT=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FLASHINFER_DISABLE_VERSION_CHECK=1
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_CUDA_ARCH_LIST=12.1a
VLLM_NO_USAGE_STATS=1
DO_NOT_TRACK=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TRITON_CACHE_DIR=/cache/triton
TILELANG_CACHE_DIR=/cache/tilelang
VLLM_CACHE_ROOT=/cache/vllm
FLASHINFER_WORKSPACE_BASE=/cache/fi_ws
TORCH_EXTENSIONS_DIR=/cache/torch_ext
PYTHONDONTWRITEBYTECODE=1
EOF
for kv in "${KV[@]}"; do k="${kv%%=*}"; grep -v "^$k=" "$envf" > "$envf.1"; mv "$envf.1" "$envf"; echo "$kv" >> "$envf"; done
mkdir -p "$OUT/$LABEL"
grep -E '^(GLM53_|TF_EXL3|VLLM_PREFIX|SPEC|DFLASH)' "$envf" > "$OUT/$LABEL/env.txt"
echo "handoff run $LABEL: worktree $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo +dirty); image $(docker image inspect "$IMG" --format '{{.Id}}' | cut -c8-23); config: ${KV[*]:-<baseline>}"

args=(-v "$L/overlay/exl3.py:/opt/glm53/exl3.py:ro" -v "$L/overlay/tf:/opt/glm53/tf:ro"
      -v "$AS/fi618/flashinfer:$SP/flashinfer:ro" -v "$AS/fi618/flashinfer_cubin:$SP/flashinfer_cubin:ro"
      -v "$AS/vllm-patches/cuda.py.exl3.patched:$SP/vllm/platforms/cuda.py:ro"
      -v "$AS/vllm-patches/flashinfer_mla_sparse_sm90.py.patched:$SP/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:ro"
      -v "$REPO:/w:ro" -v "$OUT:/out" -v "$CACHE:/cache"
      -v "$MINI:$MINI:ro" -v "$MINI-dflash2:$MINI-dflash2:ro" -v "$M/GLM-5.3-Flash-DFlash2-dc77ff1c:$M/GLM-5.3-Flash-DFlash2-dc77ff1c:ro"
      -v "$M/GLM-OCR:$M/GLM-OCR:ro")
for p in $order; do [ -f "$L/overlay/$p" ] && args+=(-v "$L/overlay/$p:/opt/glm53/$p:ro"); done

cat > "$tmp/inner.sh" <<'INNER'
fail=0
for p in $ORDER; do
  if [ -f /opt/glm53/$p ]; then out=$(python3 /opt/glm53/$p 2>&1); rc=$?
    [ $rc -ne 0 ] && { echo "chain: rc=$rc $p"; echo "$out" | tail -n 8; fail=1; }
    case $p in patch_tf_bundle.py) echo "$out" | grep -v "installed .* entries" | sed "s/^/  /";; esac
  fi
done
[ $fail = 0 ] || exit 90
echo "chain: every patch rc=0 ($(echo $ORDER | wc -w) patches)"
cd /tmp && python3 -u "${HANDOFF_DRIVER:-/w/tests/handoff/run_engine.py}" --out /out/$LABEL --label $LABEL "$@" 2>&1 | grep --line-buffered -vE "^(W|I)[0-9]{4} |exl3 e2 diag"
rc=${PIPESTATUS[0]}
chown -R $HOST_UID:$HOST_GID /out/$LABEL /cache 2>/dev/null
exit $rc
INNER
{
  echo 'avail=$(free -g | awk '"'"'NR==2{print $7}'"'"')'
  echo '[ "$avail" -ge 40 ] || { echo "handoff: host available ${avail} GB < 40"; exit 3; }'
  echo '[ -z "$(docker ps -q --filter name=tf-exl3-gpu-)" ] || exit 4'
  printf 'exec docker run --rm --gpus all --network none --memory 32g --shm-size 8g --name tf-exl3-gpu-handoff-%s' "$$"
  # nodeC GPU guard (as tests/gpu_run.sh, 2026-09-30): PyTorch CUDA allocations capped per process (GPU_MEM_CAP_GB)
  printf ' %q' -v "${GPU_GUARD_DIR:-$REPO/tests/gpu_guard}:/opt/gpuguard:ro" -e PYTHONPATH=/opt/gpuguard -e "TF_EXL3_MODELS=$M" \
    -e "GPU_MEM_CAP_GB=${GPU_MEM_CAP_GB:-40}"
  printf ' %q' --env-file "$envf" -e "ORDER=$(echo $order)" -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    -e "LABEL=$LABEL" "${args[@]}" -v "$tmp/inner.sh:/inner.sh:ro" --entrypoint bash "$IMG" /inner.sh "${PYARGS[@]}"
  echo
} > "$tmp/go.sh"
for i in $(seq 1 "${GPU_RETRY_MAX:-180}"); do
  flock "$LOCK" timeout "${GPU_TIMEOUT:-5400}" bash "$tmp/go.sh" > "$OUT/$LABEL/container.log" 2>&1
  rc=$?
  [ "$rc" -ne 4 ] && [ "$rc" -ne 3 ] && break
  sleep 20
done
tail -n 5 "$OUT/$LABEL/container.log"
echo "handoff run $LABEL: rc=$rc (log $OUT/$LABEL/container.log)"
exit $rc
