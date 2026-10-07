#!/usr/bin/env bash
# deploy-r16 (nodeA, operator): install the kit into the launcher tree. Nothing is restarted here.
#   tools/apply_r16.sh            dry run: kit MANIFEST, production files == revert-r15/ (the bytes the kit was built
#                                 against) or == the previous deploy-r16 kit (prev-r16/MANIFEST.sha256, when the kit
#                                 carries one: an update of R16) or == this kit, .env prerequisites and which env.r16
#                                 lines would be added
#   tools/apply_r16.sh --apply    the same checks, then backup -> install -> .env lines -> post-install verification
#   FRESH_INSTALL=1 tools/apply_r16.sh [--apply]   first install into a launcher checkout that has no overlay/tf/ yet
#                                 (the public launcher at the commit REPRODUCE.md names): the state check below is
#                                 skipped, every env.r16 line is added, nothing of overlay/tf/ needs to exist before.
# .env: from the deploy-r15 state every env.r16 line is added. As an UPDATE (production at the previous deploy-r16 kit,
# or already at this kit, and the kit carries prev-r16/ENV_ADD) only the lines named in prev-r16/ENV_ADD are added (the
# features this update fixes); every other env.r16 line stays exactly as production has it, present or absent (an
# env-only revert made with tools/env_r16.sh off <feature> is kept), and .env is only appended to. An EMPTY ENV_ADD (r16j,
# built with PREV_ENV_ADD=none) = the update appends nothing to .env.
# Operator switches (env.r16 `#switch NAME values`, r16j: GLM53_REJECTION_METHOD standard|block) are never added: unset =
# the first value. This script only reports them and refuses (before writing anything) a value start.sh would refuse
# after restart2.sh has stopped both ranks: not one of the values, or block without GLM53_SPEC_RESAMPLE_INDEPENDENT=1.
# Backup: ${BACKUP_ROOT:-~/tf-exl3-deploy}/backup-r16-<ts>/ (overlay/tf/site, overlay/tf/overlay, start.sh, the 4
# replaced launcher overlays, .env; BACKUP.sha256). tools/revert_r16.sh <that dir> restores it.
# The running containers are not affected (both ranks read these files only when start.sh starts them; start.sh
# copies overlay/tf and the patches to nodeB itself). Never prints a value from .env (names only).
set -euo pipefail
export LC_ALL=C
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
BK_ROOT="${BACKUP_ROOT:-$HOME/tf-exl3-deploy}"
LOV="patch_tf_bundle.py patch_hybrid_prefix_hit.py patch_mamba_align_chunking.py patch_apc_per_group_retention.py"
mode=dry; [ "${1:-}" = --apply ] && mode=apply
say() { echo "[apply_r16 $(date +%T)] $*"; }
die() { echo "[apply_r16] ABORT: $*" >&2; exit 1; }
fresh=""; if [ "${FRESH_INSTALL:-0}" = 1 ]; then fresh=1; fi
{ [ -d "$L/overlay/tf" ] || [ -n "$fresh" ]; } && [ -f "$L/start.sh" ] && [ -f "$L/.env" ] || die "$L is not the launcher tree"

# 1. the kit itself
(cd "$KIT" && sha256sum --quiet -c MANIFEST.sha256) || die "kit MANIFEST.sha256 does not verify"
# the directories that get installed must hold exactly the MANIFEST's files: a copy into an older kit dir without
# `rsync --delete` would leave stale files that sha256sum -c never sees, and site/ is copied into site-packages as is
extra=$(cd "$KIT" && comm -13 <(sed -E 's#^[0-9a-f]{64}  (\./)?##' MANIFEST.sha256 | sort) \
        <(find site overlay launcher -type f | sort))
[ -z "$extra" ] || die "kit site/ overlay/ launcher/ hold files not in MANIFEST.sha256 (copy the kit with rsync -a --delete): $(echo $extra | cut -c1-300)"
say "kit $KIT: MANIFEST ok ($(wc -l < "$KIT/MANIFEST.sha256") files), deploy-r16 $(cut -c1-12 "$KIT/SOURCE_COMMIT"), launcher-apc $(cut -c1-12 "$KIT/SOURCE_COMMIT_LAUNCHER")"

# 2. production must be exactly what the kit was built against (else the kit's start.sh would drop a newer edit)
state() {   # state <ref dir with site/ overlay/ launcher/> -> 0 if the launcher tree equals it
  local r="$1" f
  diff -rq "$r/site" "$L/overlay/tf/site" > /dev/null 2>&1 || return 1
  diff -rq "$r/overlay" "$L/overlay/tf/overlay" > /dev/null 2>&1 || return 1
  cmp -s "$r/launcher/start.sh" "$L/start.sh" || return 1
  for f in $LOV; do cmp -s "$r/launcher/overlay/$f" "$L/overlay/$f" || return 1; done
  return 0
}
# the previous deploy-r16 kit, from its MANIFEST lines (kit paths -> launcher paths); no file beyond them in site/ overlay/
prevmap() { sed -E 's#  \./site/#  overlay/tf/site/#; s#  \./overlay/#  overlay/tf/overlay/#; s#  \./launcher/start\.sh$#  start.sh#; s#  \./launcher/overlay/#  overlay/#' "$KIT/prev-r16/MANIFEST.sha256"; }
prev_state() {
  [ -f "$KIT/prev-r16/MANIFEST.sha256" ] || return 1
  (cd "$L" && prevmap | sha256sum --quiet --status -c -) 2>/dev/null || return 1
  [ -z "$(comm -13 <(prevmap | sed -E 's#^[0-9a-f]{64}  ##' | grep -E '^overlay/tf/(site|overlay)/' | sort) \
         <(cd "$L" && find overlay/tf/site overlay/tf/overlay -type f | sort))" ] || return 1
  return 0
}
upd=""
if [ -n "$fresh" ]; then
  [ ! -e "$L/overlay/tf/site" ] && [ ! -e "$L/overlay/tf/overlay" ] \
    || die "FRESH_INSTALL=1 but $L/overlay/tf/ already holds site/ or overlay/: not a first install"
  say "fresh install: no overlay/tf/ yet; the kit's start.sh and 4 launcher overlays replace the launcher's own"
elif state "$KIT/revert-r15"; then
  say "production launcher files == revert-r15/ (deploy-r15 bundle, start.sh ed7dcd8e, APC baseline overlays)"
elif prev_state; then
  upd=1
  say "production launcher files == the previous deploy-r16 kit $(cut -c1-12 "$KIT/prev-r16/SOURCE_COMMIT") (prev-r16/MANIFEST.sha256); this kit replaces:"
  join -j 2 <(sed -E 's#  \./#  #' "$KIT/prev-r16/MANIFEST.sha256" | awk '{print $1, $2}' | sort -k2) \
            <(grep -E '  \./(site|overlay|launcher)/' "$KIT/MANIFEST.sha256" | sed -E 's#  \./#  #' | awk '{print $1, $2}' | sort -k2) \
    | awk '$2 != $3 {print "  " $1}'
  comm -13 <(sed -E 's#^[0-9a-f]{64}  \./##' "$KIT/prev-r16/MANIFEST.sha256" | sort) \
           <(grep -E '  \./(site|overlay|launcher)/' "$KIT/MANIFEST.sha256" | sed -E 's#^[0-9a-f]{64}  \./##' | sort) | sed 's/^/  (new) /'
  comm -23 <(sed -E 's#^[0-9a-f]{64}  \./##' "$KIT/prev-r16/MANIFEST.sha256" | sort) \
           <(grep -E '  \./(site|overlay|launcher)/' "$KIT/MANIFEST.sha256" | sed -E 's#^[0-9a-f]{64}  \./##' | sort) | sed 's/^/  (removed) /'
elif state "$KIT"; then
  say "production launcher files already == this kit (applied before); only .env is checked below"
  already=1; [ ! -f "$KIT/prev-r16/ENV_ADD" ] || upd=1
else
  for d in site overlay; do diff -rq "$KIT/revert-r15/$d" "$L/overlay/tf/$d" 2>&1 | sed "s/^/  $d: /" | head -5 || true; done
  cmp -s "$KIT/revert-r15/launcher/start.sh" "$L/start.sh" || echo "  start.sh: sha256 $(sha256sum "$L/start.sh" | cut -c1-16) (kit built against ed7dcd8e13ec5239)"
  for f in $LOV; do cmp -s "$KIT/revert-r15/launcher/overlay/$f" "$L/overlay/$f" || echo "  overlay/$f differs"; done
  [ -f "$KIT/prev-r16/MANIFEST.sha256" ] && echo "  (and not the previous deploy-r16 kit $(cut -c1-12 "$KIT/prev-r16/SOURCE_COMMIT") either)"
  die "the launcher tree changed since the kit was built: rebuild the kit (tools/deploy16/make_kit.sh) from the current files"
fi

# 3. .env: prerequisites (non-secret flag values) and the lines to add
envval() { grep -E "^$1=" "$L/.env" | tail -n 1 | cut -d= -f2- | sed -E "s/^[\"']//; s/[\"']$//; s/[[:space:]]+#.*$//"; }
for kv in TF_EXL3_MOE=1 GLM53_FP8_GEMV=1 GLM53_KPOOL_SEED_STRIDE=1 SPEC_METHOD=dflash; do
  [ "$(envval "${kv%%=*}")" = "${kv#*=}" ] || die ".env must have ${kv} (GLM53_DEC_FP8ROOF needs FP8_GEMV, KPOOL_RING needs SEED_STRIDE, DRAFT_KV_COMPACT needs dflash)"
done
say ".env prerequisites ok: TF_EXL3_MOE=1 GLM53_FP8_GEMV=1 GLM53_KPOOL_SEED_STRIDE=1 SPEC_METHOD=dflash"
# the env.r16 lines this run adds: all of them (from deploy-r15), or only prev-r16/ENV_ADD's names (an R16 update)
if [ -n "$upd" ] && [ -f "$KIT/prev-r16/ENV_ADD" ]; then
  addnames=$(grep -E '^[A-Z0-9_]+$' "$KIT/prev-r16/ENV_ADD" | paste -sd'|' || true)
  # every line of ENV_ADD must be a name (an empty file = nothing to add; anything else is a broken kit)
  [ "$(grep -cvE '^[A-Z0-9_]+$' "$KIT/prev-r16/ENV_ADD" || true)" = 0 ] || die "prev-r16/ENV_ADD holds a line that is not a variable name"
  for n in $(echo "$addnames" | tr '|' ' '); do grep -qE "^$n=" "$KIT/env.r16" || die "prev-r16/ENV_ADD: $n is not in env.r16"; done
  kept=""; unset_=""
  for n in $(grep -E '^[A-Z0-9_]+=' "$KIT/env.r16" | cut -d= -f1); do
    [[ "|$addnames|" == *"|$n|"* ]] && continue
    if grep -qE "^$n=" "$L/.env"; then kept+="$n "; else unset_+="$n "; fi
  done
  if [ -n "$addnames" ]; then
    say "R16 update: only ${addnames//|/ } may be added; the other env.r16 lines stay as production has them (not touched): present ${kept:-none}; absent ${unset_:-none}"
  else
    say "R16 update: nothing is added to .env (prev-r16/ENV_ADD is empty); every env.r16 line stays as production has it (not touched): present ${kept:-none}; absent ${unset_:-none}"
  fi
else
  addnames=$(grep -E '^[A-Z0-9_]+=' "$KIT/env.r16" | cut -d= -f1 | paste -sd'|')
fi
add=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  n="${line%%=*}"; v="${line#*=}"
  if grep -qE "^$n=" "$L/.env"; then
    [ "$(envval "$n")" = "$v" ] || die ".env already sets $n to another value: decide by hand (value not printed)"
    say ".env already has $n (same value)"
  else
    add+=("$line")
  fi
done < <([ -z "$addnames" ] || grep -E "^($addnames)=" "$KIT/env.r16")
say ".env lines to add: ${#add[@]} (${add[*]:-none})"
# operator switches: never added; reported, and a value the kit's start.sh would refuse is refused here (nothing written)
while read -r sw vals; do
  [ -n "$sw" ] || continue
  v=$(envval "$sw" || true)
  if [ -z "$v" ]; then
    say "operator switch $sw: unset (= ${vals%%|*}); not added by this script"
  elif [[ "|$vals|" == *"|$v|"* ]]; then
    say "operator switch $sw: $v (kept as is)"
  else
    die ".env sets the operator switch $sw to a value that is not one of $vals (not printed): the kit's start.sh would refuse it after restart2.sh has stopped both ranks; fix .env first"
  fi
done < <(sed -nE 's/^#switch ([A-Z0-9_]+) ([a-z0-9|_]+).*$/\1 \2/p' "$KIT/env.r16")   # (r16z2: the values may carry _ and ., e.g. custom|cutlass_mm)
if grep -qE '^#switch GLM53_REJECTION_METHOD ' "$KIT/env.r16"; then
  pre="SPEC_METHOD=dflash $([ "$(envval SPEC_METHOD)" = dflash ] && echo ok || echo MISSING), GLM53_SPEC_RESAMPLE_INDEPENDENT=1 $([ "$(envval GLM53_SPEC_RESAMPLE_INDEPENDENT)" = 1 ] && echo ok || echo MISSING)"
  if [ "$(envval GLM53_REJECTION_METHOD)" = block ]; then
    [[ "$pre" != *MISSING* ]] || die "GLM53_REJECTION_METHOD=block needs $pre (start.sh refuses it after both ranks are stopped)"
  fi
  say "block arm prerequisites in .env (ROLLOUT.md; tools/env_r16.sh on blockverify checks them again): $pre"
fi
# [opt-decodekit] the kit's start.sh must accept the tree + .env it is about to run with: its own validate_numeric_config
# (pure reads: switch values, overlay files, bundle registrations, W8A8_ONLY names against the kit's fp8_w8a8.py ...)
# on the .env-to-be (current .env + the lines this run adds) with the KIT's bundle (TF_BUNDLE_DIR_HOST=$KIT: site/ +
# overlay/, the kit's patch_tf_bundle.py). Otherwise start.sh refuses only after restart2.sh has stopped both ranks.
# Refused -> ABORT before anything is written (dry run: the same verdict). Run in a 700 dir next to .env, env -i.
kstart="$KIT/launcher/start.sh"
if grep -q '^validate_numeric_config() {' "$kstart" && [ "$(tail -n 1 "$kstart")" = 'main "$@"' ]; then
  ck=$(mktemp -d "$L/.apply_r16-check.XXXXXX"); chmod 700 "$ck"
  sed '$d' "$kstart" > "$ck/check.sh"; printf '%s\n' 'validate_numeric_config && echo "apply_r16-launcher-check: OK"' >> "$ck/check.sh"
  { cat "$L/.env"; [ -z "$(tail -c1 "$L/.env")" ] || echo; [ "${#add[@]}" -eq 0 ] || printf '%s\n' "${add[@]}"; } > "$ck/.env"; chmod 600 "$ck/.env"
  # PYTHONDONTWRITEBYTECODE: the validation may import the kit's overlay modules (it did, once, and the kit then
  # failed its own MANIFEST cleanliness check with an untracked overlay/__pycache__/*.pyc)
  ckout=$(cd "$ck" && env -i PATH="$PATH" HOME="${HOME:-/tmp}" PYTHONDONTWRITEBYTECODE=1 \
          TF_BUNDLE_PATCH_HOST="$KIT/launcher/overlay/patch_tf_bundle.py" \
          TF_BUNDLE_DIR_HOST="$KIT" timeout 120 bash "$ck/check.sh" 2>&1) || true
  rm -rf "$ck"
  if printf '%s\n' "$ckout" | grep -qx 'apply_r16-launcher-check: OK'; then
    say "the kit's start.sh accepts the resulting tree + .env (validate_numeric_config)"
  else
    printf '%s\n' "$ckout" | grep -vE '^$' | sed -E 's#(key|token|secret|password)([^ =:]*)[=:][^ ]*#\1\2=***#gi' | tail -n 5 | sed 's/^/  /' >&2
    die "the kit's start.sh would refuse the resulting tree + .env (above; it refuses only after restart2.sh has stopped both ranks): fix .env first, nothing written"
  fi
else
  say "info: the kit's start.sh has no validate_numeric_config/main tail; the resulting .env was not pre-validated"
fi
[ "$mode" = apply ] || { say "dry run only; run with --apply to back up and install"; exit 0; }

if [ -n "${already:-}" ] && [ "${#add[@]}" -eq 0 ]; then say "nothing to do: files and .env already carry the kit"; exit 0; fi

# 4. backup (a new directory every time; an existing one is never written into)
ts=$(date +%Y%m%d-%H%M%S); B="$BK_ROOT/backup-r16-$ts"; i=1
while [ -e "$B" ]; do B="$BK_ROOT/backup-r16-$ts-$i"; i=$((i + 1)); done
mkdir -p "$BK_ROOT"; mkdir "$B"; mkdir -p "$B/overlay/tf"
for d in site overlay; do [ ! -d "$L/overlay/tf/$d" ] || cp -a "$L/overlay/tf/$d" "$B/overlay/tf/"; done
cp -p "$L/start.sh" "$B/start.sh"
for f in $LOV; do [ ! -f "$L/overlay/$f" ] || cp -p "$L/overlay/$f" "$B/overlay/$f"; done
cp -p "$L/.env" "$B/.env"; chmod 600 "$B/.env"
echo "$L" > "$B/LAUNCHER_DIR"
(cd "$B" && find . -type f ! -name BACKUP.sha256 ! -name LAUNCHER_DIR | sort | xargs sha256sum > BACKUP.sha256)
say "backup: $B ($(wc -l < "$B/BACKUP.sha256") files)"

# 5. install (new dirs staged next to the old ones, then swapped)
if [ -z "${already:-}" ]; then
  mkdir -p "$L/overlay/tf"; rm -rf "$L/overlay/tf/site.r16-new" "$L/overlay/tf/overlay.r16-new"
  cp -a "$KIT/site" "$L/overlay/tf/site.r16-new"; cp -a "$KIT/overlay" "$L/overlay/tf/overlay.r16-new"
  rm -rf "$L/overlay/tf/site" "$L/overlay/tf/overlay"
  mv "$L/overlay/tf/site.r16-new" "$L/overlay/tf/site"; mv "$L/overlay/tf/overlay.r16-new" "$L/overlay/tf/overlay"
  for f in $LOV; do cp -p "$KIT/launcher/overlay/$f" "$L/overlay/$f"; done
  cp -p "$KIT/launcher/start.sh" "$L/start.sh"
fi
if [ "${#add[@]}" -gt 0 ]; then
  [ -z "$(tail -c1 "$L/.env")" ] || echo >> "$L/.env"
  { echo "# deploy-r16 ($ts; kit $(cut -c1-12 "$KIT/SOURCE_COMMIT")) - tools/revert_r16.sh $B restores the previous .env lines"
    printf '%s\n' "${add[@]}"; } >> "$L/.env"
fi

# 6. verify
state "$KIT" || die "post-install: the launcher tree does not equal the kit (backup kept at $B)"
bash -n "$L/start.sh" || die "post-install: bash -n start.sh failed"
for line in $([ -z "$addnames" ] || grep -E "^($addnames)=" "$KIT/env.r16"); do
  [ "$(grep -cE "^${line%%=*}=" "$L/.env")" = 1 ] && [ "$(envval "${line%%=*}")" = "${line#*=}" ] || die "post-install: .env ${line%%=*} not set exactly once"
done
# .env was only appended to: the backup's .env is a byte prefix of the new one (every other line, flag or not, unchanged)
cmp -s -n "$(stat -c %s "$B/.env")" "$B/.env" "$L/.env" || die "post-install: .env lines before the appended block changed (backup kept at $B)"
say "installed: overlay/tf/site ($(ls "$L/overlay/tf/site" | wc -l) entries), overlay/tf/overlay ($(ls "$L/overlay/tf/overlay" | wc -l)), start.sh $(sha256sum "$L/start.sh" | cut -c1-16), 4 launcher overlays, .env +${#add[@]} lines"
say "next: the idle-gated restart of ROLLOUT.md (tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh <label>; r16j: step 2, switch unset)"
