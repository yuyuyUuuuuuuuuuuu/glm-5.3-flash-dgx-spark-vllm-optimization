#!/usr/bin/env bash
# deploy-r16 (nodeA, operator): FULL revert to the deploy-r15 bundle and the previous launcher files. Nothing is restarted.
#   tools/revert_r16.sh <backup dir> [--apply]   restore from apply_r16.sh's backup (default: dry run, shows what differs)
#   tools/revert_r16.sh --from-kit [--apply]     restore the files from the kit's revert-r15/ (production as copied on
#                                                2026-09-28: deploy-r15 bundle, start.sh ed7dcd8e, APC baseline overlays)
# .env: every line whose NAME is in env.r16 (and apply_r16's comment lines) is removed, then the backup's own lines of
# that kind are put back (none for a backup of the deploy-r15 state; the deployed R16 lines for a backup taken by an
# update from a previous deploy-r16 kit); other .env lines are kept, and the script reports (names only) where .env
# still differs from the backup's .env. The operator switches of env.r16 (`#switch`, r16j: GLM53_REJECTION_METHOD) are
# handled like env.r16 lines: removed, and put back only if the backup's .env had them (a backup taken by the r16j update
# from r16i has none: the switch goes, i.e. standard, as r16i's start.sh does not know it anyway).
# Then: tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh r15-revert
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
LOV="patch_tf_bundle.py patch_hybrid_prefix_hit.py patch_mamba_align_chunking.py patch_apc_per_group_retention.py"
say() { echo "[revert_r16 $(date +%T)] $*"; }
die() { echo "[revert_r16] ABORT: $*" >&2; exit 1; }
src="${1:?usage: revert_r16.sh <backup dir>|--from-kit [--apply]}"; mode=dry; [ "${2:-}" = --apply ] && mode=apply
if [ "$src" = --from-kit ]; then
  R="$KIT/revert-r15"; (cd "$R" && sha256sum --quiet -c REVERT.sha256) || die "revert-r15/REVERT.sha256 does not verify"
  SITE="$R/site"; OVL="$R/overlay"; START="$R/launcher/start.sh"; LO="$R/launcher/overlay"; BENV=""
else
  R="$(cd "$src" && pwd)"; (cd "$R" && sha256sum --quiet -c BACKUP.sha256) || die "$R/BACKUP.sha256 does not verify"
  SITE="$R/overlay/tf/site"; OVL="$R/overlay/tf/overlay"; START="$R/start.sh"; LO="$R/overlay"; BENV="$R/.env"
fi
say "restore source $R"
diff -rq "$SITE" "$L/overlay/tf/site" | sed 's/^/  site: /' | head -30 || true
diff -rq "$OVL" "$L/overlay/tf/overlay" | sed 's/^/  overlay: /' || true
cmp -s "$START" "$L/start.sh" || say "start.sh: $(sha256sum "$L/start.sh" | cut -c1-16) -> $(sha256sum "$START" | cut -c1-16)"
for f in $LOV; do cmp -s "$LO/$f" "$L/overlay/$f" || say "overlay/$f: $(sha256sum "$L/overlay/$f" | cut -c1-16) -> $(sha256sum "$LO/$f" | cut -c1-16)"; done
# r16z2rev: the free-form `#knob` lines (GLM53_DENSE_W8A8_ONLY) are operator lines like the switches: without them a
# set projection filter survived the revert (inert on the older start.sh, but silently re-armed by the next r16z2 apply)
# opt-decodekit: + every GLM53_DEC_TRACE* line (tools/env_r16.sh on trace writes them; measurement-only, never an env.r16
# line): they survived a full revert and re-armed the tracer at the next R16 apply whose start.sh forwards them
names=$({ grep -E '^[A-Z0-9_]+=' "$KIT/env.r16" | cut -d= -f1; sed -nE 's/^#((opt)?switch|knob) ([A-Z0-9_]+) .*/\3/p' "$KIT/env.r16"; echo 'GLM53_DEC_TRACE[A-Z0-9_]*'; } | paste -sd'|')
say ".env lines to remove: $(grep -cE "^($names)=|^# deploy-r16 \(" "$L/.env" || true) ($(grep -E "^($names)=" "$L/.env" | cut -d= -f1 | tr '\n' ' '))"
keep=""; [ -z "$BENV" ] || keep=$(grep -E "^($names)=|^# deploy-r16 \(" "$BENV" || true)
say "the backup's own env.r16 lines put back: $(printf '%s' "$keep" | grep -cE "^($names)=" || true) ($(printf '%s\n' "$keep" | grep -E "^($names)=" | cut -d= -f1 | tr '\n' ' '))"
[ "$mode" = apply ] || { say "dry run only; add --apply"; exit 0; }
cp -p "$L/.env" "$L/.env.bak-r16revert-$(date +%Y%m%d-%H%M%S)"
rm -rf "$L/overlay/tf/site.revert" "$L/overlay/tf/overlay.revert"
cp -a "$SITE" "$L/overlay/tf/site.revert"; cp -a "$OVL" "$L/overlay/tf/overlay.revert"
rm -rf "$L/overlay/tf/site" "$L/overlay/tf/overlay"
mv "$L/overlay/tf/site.revert" "$L/overlay/tf/site"; mv "$L/overlay/tf/overlay.revert" "$L/overlay/tf/overlay"
cp -p "$START" "$L/start.sh"; for f in $LOV; do cp -p "$LO/$f" "$L/overlay/$f"; done
# r16z5rev: when .env differs from the backup's ONLY in env.r16-named / deploy-r16 comment lines (the usual case:
# apply_r16 appended, env_r16 / the A/B flipped switches), put the backup's .env back byte for byte - the remove-and-
# append below moves the backup's own env.r16 lines (and e.g. a comment above GLM53_DEC_HOSTLOOP_WAKE) to the end
if [ -n "$BENV" ] && cmp -s <(grep -vE "^($names)=|^# deploy-r16 \(" "$L/.env") <(grep -vE "^($names)=|^# deploy-r16 \(" "$BENV"); then
  cp -p "$BENV" "$L/.env.r16revert-new" && chmod --reference="$L/.env" "$L/.env.r16revert-new" && mv "$L/.env.r16revert-new" "$L/.env"
else
  sed -i -E "/^($names)=/d; /^# deploy-r16 \(/d" "$L/.env"
  if [ -n "$keep" ]; then [ -z "$(tail -c1 "$L/.env")" ] || echo >> "$L/.env"; printf '%s\n' "$keep" >> "$L/.env"; fi
fi
diff -rq "$SITE" "$L/overlay/tf/site" >/dev/null && diff -rq "$OVL" "$L/overlay/tf/overlay" >/dev/null && cmp -s "$START" "$L/start.sh" \
  || die "post-restore: files differ from $R"
for f in $LOV; do cmp -s "$LO/$f" "$L/overlay/$f" || die "post-restore: overlay/$f differs"; done
bash -n "$L/start.sh" || die "post-restore: bash -n start.sh failed"
if [ -n "$BENV" ]; then
  if cmp -s "$BENV" "$L/.env"; then say ".env == the backup's .env"
  else say ".env differs from the backup's in (names only): $(diff <(cut -d= -f1 "$BENV") <(cut -d= -f1 "$L/.env") | grep -E '^[<>]' | tr '\n' ' ')(or values)"; fi
fi
say "restored; next: tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh r15-revert"
