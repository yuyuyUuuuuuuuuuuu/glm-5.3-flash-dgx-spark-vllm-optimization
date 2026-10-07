#!/usr/bin/env bash
# [opt-decodekit] host test of tools/deploy16/env_r16.sh's launcher guard (every edit is checked against the INSTALLED
# start.sh's validate_numeric_config; a .env it would refuse is restored and refused, rc 2) and the mhcsp cascade.
# Scratch launcher = prod-launcher (production's launcher files) + this branch's start.sh / patch_tf_bundle.py + the
# production kit's (r16z2) bundle overlay + this branch's exactlens overlay files; .env = env_nonsecret.txt + dummy key
# + production's 2026-10-02 GLM53 switch lines. Host only, nothing outside the temp dir is written.
#   G1 on mhcsp + on mhcsp2: accepted (rc 0), both lines appended
#   G2 off mhcsp with SP2=1 set: rc 0, BOTH lines gone (the cascade), and the installed start.sh accepts the result
#   G3 (control, the hole) the same edit by hand (only GLM53_MHC_SP removed): the start.sh validation REFUSES it
#   G4 a .env edit the start.sh refuses through env_r16 (on mhcsp2 forced past env_r16's own prerequisite check by
#      a start.sh whose patch_tf_bundle.py lacks the sp2 registration -> refused by env_r16's own check) and a launcher
#      refusal env_r16 has no own check for (GLM53_MHC_SP=1 + a bundle without patch_mhc_sp.py registration while an
#      unrelated feature is switched): rc 2, .env byte-identical, the refusal reason printed, no secret printed
#   G5 off/on of a feature with no dependency (exactlens) keeps working (rc 0), .env byte-identical after off+on+off
#   G6 a start.sh without validate_numeric_config: info line, no check, rc 0
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROD=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher
PKIT=${PKIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}
tmp=$(mktemp -d -p ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-decodekit/tmp); trap 'rm -rf "$tmp"' EXIT
fail=0; ck() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
L="$tmp/launcher"; K="$tmp/kit"
mkdir -p "$L" "$K/tools"
cp -a "$PROD/overlay" "$L/"; rm -rf "$L/overlay/tf"; mkdir -p "$L/overlay/tf"
cp -a "$PKIT/site" "$L/overlay/tf/site"; cp -a "$PKIT/overlay" "$L/overlay/tf/overlay"
cp -p "$REPO/overlay/patch_mla_exactlens.py" "$REPO/overlay/glm53_mla_exactlens.py" "$L/overlay/tf/overlay/"
cp -p "$REPO/launcher/overlay/patch_tf_bundle.py" "$L/overlay/patch_tf_bundle.py"
cp -p "$REPO/launcher/start.sh" "$L/start.sh"
cp -p "$REPO/tools/deploy16/env_r16.sh" "$K/tools/"; cp -p "$REPO/tools/deploy16/env.r16" "$K/env.r16"
{ grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$PROD/env_nonsecret.txt"; echo "VLLM_API_KEY=dummy-secret-value"
  printf '%s\n' GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 GLM53_MHC_SP=1 GLM53_MOE_E4M3=1 GLM53_DENSE_W8A8=1 \
    GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj; } > "$L/.env"
# production's R16 lines (env.r16 value lines) so the launcher sees the deployed state
grep -E '^[A-Z0-9_]+=' "$K/env.r16" | while IFS= read -r line; do grep -qE "^${line%%=*}=" "$L/.env" || echo "$line" >> "$L/.env"; done
sed -i -E '/^GLM53_DEC_FP8ROOF=|^GLM53_DEC_MOEGLUE_WARM=/d' "$L/.env"
export LAUNCHER_DIR="$L"
run() { bash "$K/tools/env_r16.sh" "$@" > "$tmp/out" 2>&1; }
validate() {  # validate <launcher dir>: the installed start.sh's own validation of that tree + .env
  local c="$tmp/cfg"; rm -rf "$c"; mkdir -p "$c"; sed '$d' "$1/start.sh" > "$c/check.sh"; cp -p "$1/.env" "$c/.env"
  printf 'validate_numeric_config && echo "validate_numeric_config: OK"\n' >> "$c/check.sh"
  (cd "$c" && env -i PATH="$PATH" HOME="$tmp" TF_BUNDLE_PATCH_HOST="$1/overlay/patch_tf_bundle.py" TF_BUNDLE_DIR_HOST="$1/overlay/tf" \
     timeout 120 bash "$c/check.sh" 2>&1) > "$tmp/vout"; grep -qx "validate_numeric_config: OK" "$tmp/vout"; }
validate "$L"; ck $? "G0 the scratch launcher (production's switch state) passes the start.sh validation"
cp -p "$L/.env" "$tmp/env0"
run on mhcsp2; r=$?
diff "$tmp/env0" "$L/.env" > "$tmp/d1"; [ $r = 0 ] && [ "$(grep -c '^>' "$tmp/d1")" = 1 ] && grep -qx '> GLM53_MHC_SP2=1' "$tmp/d1"
ck $? "G1 on mhcsp2 (SP=1 already set): rc $r, exactly GLM53_MHC_SP2=1 appended"
cp -p "$L/.env" "$tmp/env1"
grep -vE '^GLM53_MHC_SP=' "$tmp/env1" > "$tmp/env_hand"; cp -a "$L" "$tmp/Lh"; cp -p "$tmp/env_hand" "$tmp/Lh/.env"
! validate "$tmp/Lh" && grep -q 'GLM53_MHC_SP2=1 requires GLM53_MHC_SP=1' "$tmp/vout"
ck $? "G3 (control: the hole) GLM53_MHC_SP removed with SP2=1 left: the start.sh validation refuses it (after both ranks are stopped)"
rm -rf "$tmp/Lh"
run off mhcsp; r=$?
[ $r = 0 ] && ! grep -qE '^GLM53_MHC_SP2?=' "$L/.env" && validate "$L"
ck $? "G2 off mhcsp with SP2=1: rc $r, GLM53_MHC_SP and GLM53_MHC_SP2 both removed, the start.sh accepts the result"
cp -p "$tmp/env1" "$L/.env"
# G4: a refusal only the launcher knows: the bundle script without the sp registration (a half-applied kit), then an
# unrelated switch is flipped -> env_r16 must refuse and restore
cp -p "$L/overlay/patch_tf_bundle.py" "$tmp/ptb"; sed -i '/("patch_mhc_sp.py", "GLM53_MHC_SP")/d' "$L/overlay/patch_tf_bundle.py"
cp -p "$L/.env" "$tmp/env4"; run on exactlens; r=$?
[ $r = 2 ] && cmp -s "$L/.env" "$tmp/env4" && grep -q 'refused, .env restored' "$tmp/out" && grep -q 'patch_mhc_sp.py' "$tmp/out" \
  && ! grep -q 'dummy-secret-value' "$tmp/out"
ck $? "G4 a switch flip on a tree the start.sh refuses (half-applied bundle script): rc $r, .env byte-identical, reason printed, no secret"
sed -n 1,8p "$tmp/out" | sed 's/^/  G4| /'
cp -p "$tmp/ptb" "$L/overlay/patch_tf_bundle.py"
cp -p "$L/.env" "$tmp/env5"; run on exactlens; r1=$?; run off exactlens; r2=$?
[ $r1 = 0 ] && [ $r2 = 0 ] && cmp -s "$L/.env" "$tmp/env5"
ck $? "G5 on/off exactlens (no dependency): rc $r1/$r2, .env byte-identical after on+off"
head -c -0 /dev/null; sed -i 's/^validate_numeric_config() {/validate_numeric_config_renamed() {/' "$L/start.sh"
run on exactlens; r=$?
[ $r = 0 ] && grep -q 'was not pre-validated' "$tmp/out" && grep -qx 'GLM53_MLA_EXACT_LENS=1' "$L/.env"
ck $? "G6 a start.sh without validate_numeric_config: rc $r, info line, the edit made"
[ $fail = 0 ] && echo "test_env_r16_guard: ALL OK" || { echo "test_env_r16_guard: FAILURES"; exit 1; }
