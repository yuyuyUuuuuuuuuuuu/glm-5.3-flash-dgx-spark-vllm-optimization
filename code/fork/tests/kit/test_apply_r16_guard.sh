#!/usr/bin/env bash
# [opt-decodekit] host test of apply_r16.sh's pre-install launcher check: the kit's start.sh validate_numeric_config
# must accept the resulting tree + .env, else ABORT before anything is written. Fake kit = a copy of the production kit
# (r16z2, PKIT) with this branch's tools/apply_r16.sh (MANIFEST line refreshed); scratch launcher = that kit applied
# (prod-launcher's other launcher overlays) + .env = env_nonsecret.txt + dummy key + production's 2026-10-02 switches.
#   A1 production's .env: dry run rc 0, "accepts the resulting tree + .env"
#   A2 GLM53_DENSE_W8A8_ONLY with a name the kit's fp8_w8a8.py does not serve (a #knob apply never validated itself):
#      dry run and --apply refuse (rc != 0), the name not printed, nothing written (.env, tree, no backup dir)
#   A3 GLM53_MHC_SP2=1 without GLM53_MHC_SP=1 (each value passes apply's own 0|1 switch check): refused, nothing written
#   A4 (control) the original apply_r16.sh of the production kit accepts A2/A3 (the hole): rc 0 on the dry run
#   A5 production's .env --apply on the applied tree: rc 0 "nothing to do"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROD=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher
PKIT=${PKIT:-${TF_EXL3_KITS:-$HOME}/tf-exl3-deploy16.r16z2}
tmp=$(mktemp -d -p ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-decodekit/tmp); trap 'rm -rf "$tmp"' EXIT
fail=0; ck() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
K="$tmp/kit"; cp -a "$PKIT" "$K"; cp -p "$REPO/tools/deploy16/apply_r16.sh" "$K/tools/apply_r16.sh"
cp -p "$REPO/tools/deploy16/revert_r16.sh" "$K/tools/revert_r16.sh"
for f in apply_r16.sh revert_r16.sh; do
  sed -i -E "s#^[0-9a-f]{64}  \./tools/$f\$#$(sha256sum "$K/tools/$f" | cut -c1-64)  ./tools/$f#" "$K/MANIFEST.sha256"
done
(cd "$K" && sha256sum --quiet -c MANIFEST.sha256) || { echo "FAIL fake kit MANIFEST"; exit 1; }
L="$tmp/launcher"; mkdir -p "$L/overlay/tf"; cp -a "$PROD/overlay/." "$L/overlay/"; rm -rf "$L/overlay/tf"; mkdir -p "$L/overlay/tf"
cp -a "$K/site" "$L/overlay/tf/site"; cp -a "$K/overlay" "$L/overlay/tf/overlay"; cp -p "$K/launcher/start.sh" "$L/start.sh"
cp -p "$K"/launcher/overlay/*.py "$L/overlay/"
{ grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$PROD/env_nonsecret.txt"; echo "VLLM_API_KEY=dummy-secret-value"
  printf '%s\n' GLM53_KDA_STRIDED_QKV=1 GLM53_KDA_FLASHKDA=1 GLM53_MHC_SP=1 GLM53_MOE_E4M3=1 GLM53_DENSE_W8A8=1 \
    GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,mla.o_proj; } > "$L/.env"
grep -E '^[A-Z0-9_]+=' "$K/env.r16" | while IFS= read -r line; do grep -qE "^${line%%=*}=" "$L/.env" || echo "$line" >> "$L/.env"; done
sed -i -E '/^GLM53_DEC_FP8ROOF=|^GLM53_DEC_MOEGLUE_WARM=/d' "$L/.env"
cp -p "$L/.env" "$tmp/env0"
export LAUNCHER_DIR="$L" BACKUP_ROOT="$tmp/bk"
snap() { (cd "$L" && find . -type f ! -name '.env.bak-*' | sort | xargs sha256sum); }
snap > "$tmp/s0"
bash "$K/tools/apply_r16.sh" > "$tmp/out" 2>&1; r=$?
[ $r = 0 ] && grep -q "accepts the resulting tree + .env" "$tmp/out"
ck $? "A1 production's .env on the applied tree: dry run rc $r, the kit's start.sh accepts it"
[ $r = 0 ] || sed -n 1,30p "$tmp/out" | sed 's/^/  A1| /'
for case in A2 A3; do
  cp -p "$tmp/env0" "$L/.env"
  if [ $case = A2 ]; then sed -i -E 's/^GLM53_DENSE_W8A8_ONLY=.*/GLM53_DENSE_W8A8_ONLY=kda.in_proj_qkvbfg_a,bogusgroup.secretproj/' "$L/.env"
  else sed -i -E '/^GLM53_MHC_SP=/d' "$L/.env"; echo GLM53_MHC_SP2=1 >> "$L/.env"; fi
  cp -p "$L/.env" "$tmp/envc"; snap > "$tmp/sc"
  bash "$K/tools/apply_r16.sh" > "$tmp/out" 2>&1; r1=$?
  bash "$K/tools/apply_r16.sh" --apply > "$tmp/out2" 2>&1; r2=$?
  [ $r1 != 0 ] && [ $r2 != 0 ] && grep -q "would refuse the resulting tree" "$tmp/out" && cmp -s "$L/.env" "$tmp/envc" \
    && diff <(snap) "$tmp/sc" > /dev/null && [ ! -d "$tmp/bk" ] && ! grep -qE 'bogusgroup|secretproj|dummy-secret' "$tmp/out" "$tmp/out2"
  ck $? "$case refused by the dry run (rc $r1) and --apply (rc $r2), nothing written, no value printed"
  grep -E "^  |ABORT" "$tmp/out" | head -3 | sed "s/^/  $case| /"
  bash "$PKIT/tools/apply_r16.sh" > "$tmp/out3" 2>&1; r3=$?
  [ $r3 = 0 ]; ck $? "A4 ($case control) the production kit's own apply_r16.sh accepts it (rc $r3): the hole"
done
cp -p "$tmp/env0" "$L/.env"
bash "$K/tools/apply_r16.sh" --apply > "$tmp/out" 2>&1; r=$?
[ $r = 0 ] && grep -q "nothing to do" "$tmp/out" && diff <(snap) "$tmp/s0" > /dev/null
ck $? "A5 production's .env --apply on the applied tree: rc $r, nothing to do, nothing changed"
# R1 (revert_r16) a full revert removes the tracer lines env_r16.sh 'on trace' writes, keeps every non-R16 line
cp -p "$tmp/env0" "$L/.env"; printf '%s\n' GLM53_DEC_TRACE=/root/.cache/vllm/glm53-dectrace GLM53_DEC_TRACE_MAX_STEPS=20000 >> "$L/.env"
bash "$K/tools/revert_r16.sh" --from-kit --apply > "$tmp/out" 2>&1; r=$?
nonr16() { grep -vE "^($(grep -E '^[A-Z0-9_]+=' "$K/env.r16" | cut -d= -f1 | paste -sd'|')|GLM53_(KDA_STRIDED_QKV|KDA_FLASHKDA|MHC_SP|MOE_E4M3|DENSE_W8A8|DENSE_W8A8_ONLY))=" "$1"; }
[ $r = 0 ] && ! grep -q '^GLM53_DEC_TRACE' "$L/.env" && diff <(nonr16 "$tmp/env0") <(nonr16 "$L/.env") > /dev/null
ck $? "R1 revert --from-kit: rc $r, the GLM53_DEC_TRACE* lines removed, every non-R16 .env line kept"
[ $r = 0 ] || tail -5 "$tmp/out" | sed 's/^/  R1| /'
cp -p "$tmp/env0" "$L/.env"; printf '%s\n' GLM53_DEC_TRACE=/root/.cache/vllm/glm53-dectrace >> "$L/.env"
bash "$PKIT/tools/revert_r16.sh" --from-kit --apply > "$tmp/out" 2>&1; r=$?
[ $r = 0 ] && grep -q '^GLM53_DEC_TRACE=' "$L/.env"
ck $? "R1 (control) the production kit's revert_r16.sh keeps the GLM53_DEC_TRACE line (rc $r): the leftover"
[ $fail = 0 ] && echo "test_apply_r16_guard: ALL OK" || { echo "test_apply_r16_guard: FAILURES"; exit 1; }
