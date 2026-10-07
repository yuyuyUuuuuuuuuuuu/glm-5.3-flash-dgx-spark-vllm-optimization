#!/usr/bin/env bash
# deploy-r16 (nodeA, operator; r16z3rev): run the INSTALLED start.sh's own config validation - exactly what
# `./start.sh start` refuses - WITHOUT starting or stopping anything. Use it after every hand edit of .env (the moe3
# layer lists GLM53_MOE_E4M3_LAYERS / GLM53_MOE_E4M3_DOWN_LAYERS are not env_r16.sh features) and BEFORE
# ~/tf-exl3-deploy/restart2.sh: restart2.sh runs `./start.sh stop` first and `./start.sh start` (where the validation
# lives) only afterwards, so a value start.sh refuses leaves BOTH ranks stopped, and restart2.sh still exits 0.
#   tools/check_config.sh                      validate the launcher tree + .env as they are
#   tools/check_config.sh NAME=value NAME= ... the same with these values applied on top of .env (NAME= = unset)
# rc 0 = start.sh would accept it; rc 1 = refused (start.sh's own message, which never prints a list value or a
# credential, is shown masked); rc 2 = usage / not a launcher tree. Nothing is written except a temporary copy of the
# start.sh functions in the launcher dir (removed on exit); .env is read by start.sh's own preamble only.
# VALIDATE_CHAIN (default: what main runs before start) may be narrowed, e.g. VALIDATE_CHAIN=validate_numeric_config.
set -uo pipefail
L="${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
CHAIN=${VALIDATE_CHAIN:-validate_numeric_config configure_capture_sizes configure_kv_cache_memory validate_overlay_artifacts validate_thin_ext_so}
[ -f "$L/start.sh" ] && [ -f "$L/.env" ] || { echo "check_config: $L is not the launcher tree"; exit 2; }
[ "$(tail -n 1 "$L/start.sh")" = 'main "$@"' ] || { echo "check_config: $L/start.sh does not end in main \"\$@\" (unknown launcher)"; exit 2; }
ex=""
for kv in "$@"; do
  # r16z5rev: ':' allowed (GLM53_DENSE_W8A8_SKIP_LAYERS 0-44:mla.o_proj / GLM53_DENSE_W8A8_HILO kda.o_proj:256 were refused)
  [[ "$kv" =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.,:-]*)$ ]] || { echo "check_config: '$kv' is not NAME=value with a [A-Za-z0-9_.,:-] value"; exit 2; }
  n=${BASH_REMATCH[1]}; v=${BASH_REMATCH[2]}
  [[ "$n" =~ (KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL) ]] && { echo "check_config: $n is not a switch"; exit 2; }
  if [ -z "$v" ]; then ex+="unset $n; "; else ex+="export $kv; "; fi
done
T="$L/.check-config-$$.sh"; trap 'rm -f "$T"' EXIT
sed '$d' "$L/start.sh" > "$T"
printf '%s %s && echo CHECK-CONFIG-OK\n' "$ex" "$(echo $CHAIN | sed 's/ / \&\& /g')" >> "$T"
out=$(env -i PATH="$PATH" HOME="$HOME" USER="${USER:-$(id -un)}" bash "$T" 2>&1)
if printf '%s\n' "$out" | grep -qx 'CHECK-CONFIG-OK'; then echo "check_config: OK - the installed start.sh accepts this configuration${*:+ (with $# override(s))}"; exit 0; fi
echo "check_config: REFUSED by the installed start.sh (restart2.sh would stop both ranks and then fail to start):"
printf '%s\n' "$out" | grep -v CHECK-CONFIG-OK | tail -n 3 | sed -E 's#(key|token|secret|password)([=: ][^ ]*)#\1=***#gi' | cut -c1-240 | sed 's/^/  /'
exit 1
