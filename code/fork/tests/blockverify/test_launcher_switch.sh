#!/usr/bin/env bash
# The GLM53_REJECTION_METHOD switch of launcher/start.sh (host only; nothing is started, no ssh, no docker).
#   L1 launcher/start.sh == tools/blockverify/make_start_sh.py applied to the deploy-r16 kit start.sh (b721de12);
#      bash -n; the committed start.sh.blockverify.patch is that diff
#   L2 the knob reaches both ranks exactly like GLM53_SPEC_RESAMPLE_INDEPENDENT: one head `-e` line right after it,
#      one worker serve_env_names entry (so GLM53_EXTRA_ENV cannot shadow it), and the spec JSON of BOTH inner
#      scripts reads it
#   L3 the in-container spec snippet (both copies, extracted from start.sh): unset / empty / standard give the kit's
#      JSON byte for byte (default = today); block gives rejection_sample_method "block"; anything else exits non-zero,
#      and under the inner script's `set -euo pipefail` the ARGS line then aborts before vllm serve
#   L4 the launcher's own validate_numeric_config on production's non-secret .env (functions only, main not run):
#      unset / empty / standard / block accepted; bogus value, block with GLM53_SPEC_RESAMPLE_INDEPENDENT!=1, block
#      with SPEC_METHOD!=dflash, block without overlay/tf/overlay/patch_spec_block_keys.py refused, block with the
#      kit's (pre-blockverify) overlay/patch_tf_bundle.py refused (partial install = stock biased block mode)
# Usage: tests/blockverify/test_launcher_switch.sh    (exit 1 on any failure)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIT="${KIT_DIR:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16}"
PROD_ENV="${PROD_ENV:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher/env_nonsecret.txt}"
NEW="$REPO/launcher/start.sh"
fail=0
ck() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fail=1; fi; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/bv-launcher.XXXXXX"); trap 'rm -rf "$tmp"' EXIT

# ---- L1
python3 "$REPO/tools/blockverify/make_start_sh.py" "$KIT/launcher/start.sh" "$tmp/start.sh" > "$tmp/gen.txt" 2>&1
cmp -s "$tmp/start.sh" "$NEW"; ck $? "L1 launcher/start.sh is reproducible from the kit start.sh ($(cat "$tmp/gen.txt"))"
cmp -s "$tmp/start.sh.blockverify.patch" "$REPO/launcher/start.sh.blockverify.patch"; ck $? "L1 start.sh.blockverify.patch is that diff"
bash -n "$NEW"; ck $? "L1 bash -n launcher/start.sh"
( cd "$tmp" && cp "$KIT/launcher/start.sh" k.sh && patch -s k.sh < "$REPO/launcher/start.sh.blockverify.patch" && cmp -s k.sh "$NEW" )
ck $? "L1 the patch applies to the kit start.sh with patch(1) and gives launcher/start.sh"

# ---- L2
n_head=$(grep -c '^        -e GLM53_REJECTION_METHOD="${GLM53_REJECTION_METHOD:-}" \\$' "$NEW")
prev=$(grep -B1 '^        -e GLM53_REJECTION_METHOD=' "$NEW" | head -1)
[ "$n_head" = 1 ] && [[ "$prev" == *'-e GLM53_SPEC_RESAMPLE_INDEPENDENT='* ]]; ck $? "L2 head docker run: one -e line, right after GLM53_SPEC_RESAMPLE_INDEPENDENT"
worker=$(sed -n '/local -a serve_env_names=()/,/serve_env+=" -e \$v=/p' "$NEW")
[ "$(grep -o '\bGLM53_REJECTION_METHOD\b' <<< "$worker" | wc -l)" = 1 ] && grep -q 'GLM53_SPEC_RESAMPLE_INDEPENDENT' <<< "$worker"
ck $? "L2 worker serve_env_names loop: one entry (and GLM53_EXTRA_ENV's owned set is read off that list)"
[ "$(grep -c 'os.environ.get("GLM53_REJECTION_METHOD") or "standard"' "$NEW")" = 2 ]; ck $? "L2 both inner scripts' spec JSON read it"

# ---- L3: the snippet between `python3 -S -c '` and `')")` of each dflash branch
python3 - "$NEW" "$KIT/launcher/start.sh" "$tmp" <<'PY'
import re, sys
from pathlib import Path
new, kit, out = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text(), Path(sys.argv[3])
pat = re.compile(r"""ARGS\+=\(--speculative-config "\$\(python3 -S -c '(.*?)'\)"\)""", re.S)
sn, sk = pat.findall(new), pat.findall(kit)
assert len(sn) == 2 and len(sk) == 2, (len(sn), len(sk))
for i, s in enumerate(sn):
    (out / f"snip_new_{i}.py").write_text(s)
for i, s in enumerate(sk):
    (out / f"snip_kit_{i}.py").write_text(s)
PY
ck $? "L3 extracted the two spec snippets of the new and the kit start.sh"
base_env=(env -i PATH="$PATH" DFLASH_MODEL_DIR=/models/dflash2 DFLASH_TOKENS=7 DFLASH_DRAFT_TP=2)
for i in 0 1; do
  kit_json=$("${base_env[@]}" python3 -S -c "$(cat "$tmp/snip_kit_$i.py")")
  for case in unset empty standard; do
    case $case in unset) e=();; empty) e=(GLM53_REJECTION_METHOD=);; standard) e=(GLM53_REJECTION_METHOD=standard);; esac
    j=$("${base_env[@]}" "${e[@]}" python3 -S -c "$(cat "$tmp/snip_new_$i.py")")
    [ "$j" = "$kit_json" ]; ck $? "L3 snippet $i, GLM53_REJECTION_METHOD $case: JSON byte-identical to the kit's ($j)"
  done
  j=$("${base_env[@]}" GLM53_REJECTION_METHOD=block python3 -S -c "$(cat "$tmp/snip_new_$i.py")")
  [ "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["rejection_sample_method"])' "$j")" = block ] \
    && [ "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.pop("rejection_sample_method"); print(json.dumps(d,sort_keys=True))' "$j")" \
       = "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.pop("rejection_sample_method"); print(json.dumps(d,sort_keys=True))' "$kit_json")" ]
  ck $? "L3 snippet $i, block: rejection_sample_method=block, every other field as the kit's"
  out=$("${base_env[@]}" GLM53_REJECTION_METHOD=blok python3 -S -c "$(cat "$tmp/snip_new_$i.py")" 2>&1); rc=$?
  [ $rc != 0 ] && grep -q "must be standard or block" <<< "$out"; ck $? "L3 snippet $i, bogus value: exit $rc ($out)"
  out=$("${base_env[@]}" GLM53_REJECTION_METHOD=blok bash -c 'set -euo pipefail; ARGS=(); ARGS+=(--speculative-config "$(python3 -S -c "$1")"); echo "REACHED vllm serve"' _ "$(cat "$tmp/snip_new_$i.py")" 2>&1)
  ! grep -q "REACHED" <<< "$out"; ck $? "L3 snippet $i, bogus value under the inner script's set -euo pipefail: vllm serve is never reached"
done

# ---- L4: validate_numeric_config (functions only; production's non-secret .env)
mk_cfg() {  # $1 = dir, then .env lines to append
  local d="$1"; shift
  mkdir -p "$d/overlay/tf/overlay"
  sed '$d' "$NEW" > "$d/check_config.sh"
  printf 'validate_numeric_config && echo "validate_numeric_config: OK"\n' >> "$d/check_config.sh"
  cp "$PROD_ENV" "$d/.env"
  cp "$REPO/overlay/patch_spec_block_keys.py" "$d/overlay/tf/overlay/"
  cp "$REPO/launcher/overlay/patch_tf_bundle.py" "$d/overlay/"
  local l; for l in "$@"; do echo "$l" >> "$d/.env"; done
}
run_cfg() { env -i PATH="$PATH" HOME="$tmp" timeout 60 bash "$1/check_config.sh" 2>&1; }
i=0
for spec in "OK|" "OK|GLM53_REJECTION_METHOD=" "OK|GLM53_REJECTION_METHOD=standard" "OK|GLM53_REJECTION_METHOD=block" \
            "must be one of: standard block|GLM53_REJECTION_METHOD=blok" \
            "requires GLM53_SPEC_RESAMPLE_INDEPENDENT=1|GLM53_REJECTION_METHOD=block;GLM53_SPEC_RESAMPLE_INDEPENDENT=0" \
            "requires SPEC_METHOD=dflash|GLM53_REJECTION_METHOD=block;SPEC_METHOD=mtp" \
            "requires .*patch_spec_block_keys.py|GLM53_REJECTION_METHOD=block;NO_OVERLAY" \
            "requires .*patch_tf_bundle.py to run patch_spec_block_keys.py|GLM53_REJECTION_METHOD=block;KIT_BUNDLE" \
            "OK|GLM53_REJECTION_METHOD=standard;KIT_BUNDLE"; do
  i=$((i + 1)); want="${spec%%|*}"; lines="${spec#*|}"
  d="$tmp/cfg$i"; IFS=';' read -r -a arr <<< "$lines"
  extra=(); no_overlay=0; kit_bundle=0
  for l in "${arr[@]}"; do
    case "$l" in NO_OVERLAY) no_overlay=1;; KIT_BUNDLE) kit_bundle=1;; "") ;; *) extra+=("$l");; esac
  done
  mk_cfg "$d" "${extra[@]}"
  [ $no_overlay = 1 ] && rm "$d/overlay/tf/overlay/patch_spec_block_keys.py"
  [ $kit_bundle = 1 ] && cp "$KIT/launcher/overlay/patch_tf_bundle.py" "$d/overlay/patch_tf_bundle.py"
  out=$(run_cfg "$d")
  if [ "$want" = OK ]; then
    grep -q "validate_numeric_config: OK" <<< "$out"; ck $? "L4 .env + [${lines:-nothing}]: accepted"
  else
    ! grep -q "validate_numeric_config: OK" <<< "$out" && grep -Eq "$want" <<< "$out"
    ck $? "L4 .env + [$lines]: refused ($(grep -E "$want" <<< "$out" | head -1))"
  fi
done
[ $fail = 0 ] && echo "ALL PASSED" || echo "FAILED"
exit $fail
