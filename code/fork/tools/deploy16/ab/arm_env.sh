#!/usr/bin/env bash
# GENERIC production switch-flip A/B on a deploy-r16 kit (r16z3rev: the reviewed replacement of r16z3's arm_env.sh).
# One arm = one named set of .env lines of the kit's operator switches/knobs:
#   KNOBS="GLM53_X GLM53_Y"          the names the A/B owns. Each must be a `#switch`/`#knob` of the kit's env.r16
#                                    (anything else - in particular a credential-like name - is refused before
#                                    anything is touched). base removes them from .env; arms may only set these.
#   ARMS="base <name> ..."           "base" MUST come first, exactly once (= production with every KNOB line removed;
#                                    production may not set a KNOB itself unless BASE_DROPS_PROD=1). Names: [a-z0-9_].
#   ARM_<name>="GLM53_X=1 GLM53_Y=13-44"   the arm's lines; ARM_<name> or ARM_<NAME> (upper-cased) is accepted.
#                                    Values: [A-Za-z0-9_.,-] only (no spaces); #switch values must be one of the
#                                    listed ones; the two moe3 layer lists must be comma lists/ranges of 3..44.
#   PROOF='glm53_moe_fused16 active:'  a fixed string every non-base arm must show in BOTH ranks' logs after its
#                                    measurement; PROOF_<name>=<string> overrides it per arm ("-" = none for that arm).
#                                    (r16z8) several strings separated by '|' (a combined arm, one per feature): EVERY
#                                    one must be in both ranks' logs, e.g. PROOF='serving confirmed (mode on): graph
#                                    replays|graph-captured one-shot all-reduces' (dlmh + ar1shot).
#   NEVER='glm53_kda_lazy: self-check:'  (r16z6rev) a fixed string that must NOT appear in EITHER rank's log after a
#                                    non-base arm's measurement (a refusal that can only show up under traffic, e.g. the
#                                    kdalazy self-check 'differ' line = repair mode, ~5 ms/step slower); NEVER_<name>
#                                    overrides it per arm ("-" = none). boot_checks runs only BEFORE the measurement.
#                                    (r16z8) several strings separated by '|': NONE may appear on either rank.
#   ARM_PROBE='<command>'            (r16z7, optional) run on nodeA in EVERY arm (base included) after the measurement and
#                                    the gate, e.g. the concurrency-quality probe:
#                                    ARM_PROBE='flock -w 900 ~/tf-exl3-assets-prodbench.lock python3
#                                    ~/<kit>/tools/conc_quality_probe.py --out ~/tf-exl3-deploy/conc/{label}.json'
#                                    ({label} = envab-<arm>-<ts>; characters [A-Za-z0-9_./~{}=:, -] only; run under
#                                    `timeout ${PROBE_TIMEOUT:-900}`). Its output goes to $OUT/probe-<arm>.log, its last
#                                    line starting with ${PROBE_LINE:-CONCQ} to the arm's log and $OUT/probe-summary-<ts>.txt.
#                                    Informational: a failing probe does not invalidate the arm (judge it by hand).
#   KEEP=auto|base                   auto (default): keep the fastest arm that passes every gate; base: always end on
#                                    base (sensitivity / probe studies such as DEPLOY_R16Z3 section 4's group A/B,
#                                    whose arms are measurements, not candidates).
#   PREV_KIT_NAME=<kit on nodeA>     swap mode: production runs that kit (its plain launcher start.sh must be the one
#                                    $KIT_NAME's prev-r16/MANIFEST lists); PRE_START_SHA = the first 8 hex of the live
#                                    start.sh.
#                                    Without PREV_KIT_NAME production must already run $KIT_NAME (checked:
#                                    start.sh == the kit's one, overlay/tf + the 4 launcher overlays ==
#                                    the kit's).
# Judgement (arm_w8a8.rev.sh's): an arm passes with long KL <= LMAX, dvp KL <= DMAX, long top-1 >= TMIN, gate 32k
# median > GMARGIN x base AND > the best so far, decode structured tok/s >= DECMIN x base, EVERY field numeric.
#
# Review fixes against r16z3's arm_env.sh (each one is exercised by tools/deploy16/ab/mock/run_env_mock.sh):
#  E1 the documented invocations (ARM_exact10=..., ARM_f16exact=...) could never work: the script only read the
#     upper-cased name, and it validated each arm only when the loop reached it - AFTER the kit swap and the base
#     measurement, so the documented run swapped the kit, restarted, measured base and then exited 2 mid-A/B. Every
#     arm (name, lines, KNOBS membership, values) is now validated before anything is paused, copied or written.
#  E2 without "base" first, the judgement had nothing to compare and skipped the final restart: production was left
#     on the LAST arm, measured but never judged (it could fail LMAX). base is now mandatory and first.
#  E3 restart2.sh runs `./start.sh stop` and only then `./start.sh start`, and start.sh validates its config only on
#     start: an arm value start.sh refuses (a layer list with 2 or 45, FUSED16=2, ...) stopped BOTH ranks and left
#     production down (restart2 still returns 0) until the 15-min health timeout. Every arm is now validated by the
#     INSTALLED start.sh itself (its functions, main not run, the arm's values applied) once before the first restart
#     and again right before each restart; a refused value restarts nothing. A head container missing right after
#     restart2 is detected at once (no 15-min wait) and recovered to base.
#  E4 a refused FIRST restart (production busy) stopped the A/B with the new kit installed on disk under the old
#     runtime and .env stripped of the KNOB lines: the next restart by anyone would have booted an untested state.
#     The pre-A/B files (revert_r16 + start.sh/.env backups) are now put back without a restart.
#  E5 a refused back-to-base / final restart left production on a failed or unjudged arm (only a message, rc 0).
#     Recovery and final restarts now retry (RECOVER_TRIES x IDLE_WAIT_S); if production stays busy, .env is left at
#     the TARGET state (so the next restart heals) and the script exits 10 with the state spelled out.
#  E6 the restart verification read the head's env only; now head AND worker env == the arm (R2).
#  E7 bootok accepted a boot_checks run that died half-way (no MISS line printed) and ignored EVERY MISS/BAD line that
#     merely contained "idx_gate"; now the final "boot_checks: ALL OK|PROBLEMS" line is required and only the known
#     benign "BAD ... idx_gate result differs" row is ignored.
#  E8 gate rows counted when the server was idle BEFORE the probe even if another request ran during it; a row now
#     needs both its "before" and "after" lines at running=0.0 waiting=0.0.
#  E9 restore_all said "restored healthy" when its restart had been refused (only /health was checked) and never
#     checked the restored start.sh; it now verifies start.sh == PRE sha, StartedAt changed, both ranks' KNOB env ==
#     the pre-A/B snapshot, and exits 11 when the runtime could not be restarted.
#  E10 setenv's status was always 0 (it ended in `echo`); a failed .env write now stops the step.
#  E12 KNOBS were free-form: a typo or e.g. VLLM_API_KEY in KNOBS made base delete that .env line and runenv print
#     its value. KNOBS must be env.r16 #switch/#knob names; a credential-like name is refused.
#  E13 judgement compared an arm with GMARGIN x the best arm so far instead of GMARGIN x base AND the best (the
#     documented rule).
#  E14 PREV_KIT_NAME was used blind (the documented r16z2rev does not exist on nodeA - the A/B aborted after copying
#     the kit); the previous kit's plain start.sh must now be the one the kit's prev-r16/MANIFEST lists, checked
#     before anything is written. A lock file prevents two concurrent runs.
# r16z5rev (review of the r16z5 kit):
#  V1 the value class [A-Za-z0-9_.,-] had no ':' - every GLM53_DENSE_W8A8_SKIP_LAYERS / _HILO value (0-44:mla.o_proj,
#     kda.o_proj:256) was refused as "not NAME=VALUE" before its validator ran: the documented arms (e), A1, A2, A4
#     could never start. ':' is allowed now (setenv / vcheck / want handle it like any other value character).
#  V2 KIT_NAME defaulted to tf-exl3-deploy16.r16z3 (an existing older kit on nodeC): a run without KIT_NAME swapped
#     to - or flipped .env against - the wrong kit. The default is now the kit this script ships in (its parent dir
#     with env.r16 + MANIFEST.sha256); from the repo copy KIT_NAME is mandatory.
#  V3 base was "production with every KNOB line removed": a KNOB production sets (GLM53_DENSE_W8A8_ONLY=sub) could
#     only be dropped from base too (BASE_DROPS_PROD=1 - base then is NOT production, and KEEP=base / a losing arm
#     left production on the stripped config, unjudged). Now base = production's own KNOB lines (unchanged), an arm
#     overrides them, and "NAME=" (empty) in an arm UNSETS a KNOB (A1: GLM53_DENSE_W8A8_ONLY= ..._SKIP_LAYERS=...).
#     BASE_DROPS_PROD=1 keeps the old stripping semantics.
#  V4 in swap mode base = the NEW kit with production's switches (arm (a), the deliberate default changes) and it was
#     never judged: a base that failed the quality limits stayed in production (KEEP=base, or no arm better). Base is
#     now gated in swap mode: long KL <= BASE_LMAX, dvp <= BASE_DMAX, top-1 >= BASE_TMIN (defaults LMAX/DMAX/TMIN) and,
#     when set, gate 32k >= BASE_GMIN; a failing base restores the pre-A/B kit (exit 14).
# r16z8 review (2026-10-05):
#  A1 ABLIT: start.sh clears a .env ABLIT (ABLIT=0 right after reading .env) and honours only a CALLER-exported ABLIT=1;
#     restart2.sh hands its own environment to `./start.sh start`. Production was restarted with ABLIT=1 on 2026-10-05
#     09:44 JST (the owner: every measurement with ABLIT=1), and every restart2.sh this script ran carried no ABLIT: each
#     arm booted ABLIT=0 (the A/B measured a model production does not serve) and the run ENDED with production's
#     ablit switched off. Now the value both running containers carry (RABL; head != worker or not 0/1 -> abort before
#     anything is touched) is passed as ABLIT=$RABL to every restart2.sh (arms, recovery, restore) and to the installed
#     start.sh's validation, and each restart must show it on BOTH ranks (else the arm is broken, rc 5 / restore exit 12).
#     An operator ABLIT=0|1 must equal the live value (switching ablit is not an A/B arm).
set -u
OUT=${OUT:-$HOME/tf-exl3-assets/env-ab}; G=GLM-5.3-Flash-EXL3-2x-DGX-Sparks
HOMEDIR=${HOMEDIR:-$HOME}
# V2: the kit this script ships in (<kit>/tools/arm_env_ab.sh) unless KIT_NAME names another; the repo copy needs it
_SELFKIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)
if [ -z "${KIT_NAME:-}" ] && [ -f "$_SELFKIT/env.r16" ] && [ -f "$_SELFKIT/MANIFEST.sha256" ]; then KIT_NAME=$(basename "$_SELFKIT"); fi
[ -n "${KIT_NAME:-}" ] || { echo "[env-ab] ABORT: KIT_NAME=<the kit dir name under $HOMEDIR> is required (run the kit's tools/arm_env_ab.sh, or set it)"; exit 2; }
K=$KIT_NAME; KP=${PREV_KIT_NAME:-}
TS=$(date +%Y%m%d-%H%M%S)
LMAX=${LMAX:-0.0165}; DMAX=${DMAX:-0.0230}; TMIN=${TMIN:-98.3}; GMARGIN=${GMARGIN:-1.02}; DECMIN=${DECMIN:-0.97}
BASE_LMAX=${BASE_LMAX:-$LMAX}; BASE_DMAX=${BASE_DMAX:-$DMAX}; BASE_TMIN=${BASE_TMIN:-$TMIN}; BASE_GMIN=${BASE_GMIN:-}
VCHARS='A-Za-z0-9_.,:-'   # V1: the value characters an arm / base line may carry (':' for SKIP_LAYERS / HILO)
ARMS=${ARMS:-}; KNOBS=${KNOBS:-}; PROOF=${PROOF:-}; NEVER=${NEVER:-}; KEEP=${KEEP:-auto}
PRE_START_SHA=${PRE_START_SHA:-}
IDLE_WAIT_S=${IDLE_WAIT_S:-900}; RECOVER_TRIES=${RECOVER_TRIES:-4}
VALIDATE_CHAIN=${VALIDATE_CHAIN:-validate_numeric_config configure_capture_sizes configure_kv_cache_memory validate_overlay_artifacts validate_thin_ext_so}
HEAD_C=glm53-exl3-head; WORK_C=glm53-exl3-worker; W=${WORKER_SSH:?set WORKER_SSH=<user>@<worker address>}
GATE=${GATE:-$HOME/tf-exl3-deploy/gate_prefill_32k.py}
say(){ echo "[env-ab $(date +%T)] $*"; }
die(){ say "ABORT: $1"; exit "${2:-2}"; }
HEAD_SSH=${HEAD_SSH:-nodeA}   # ssh alias of the head node (this script runs on a third machine)
n2(){ ssh -o BatchMode=yes "$HEAD_SSH" "set -o pipefail; $*"; }
isnum(){ [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

# ------------------------------------------------------------------ local preflight (nothing remote, nothing paused)
LK="$HOMEDIR/$K"; ENVR="$LK/env.r16"
[ -f "$ENVR" ] || die "local kit $LK has no env.r16"
(cd "$LK" && sha256sum --quiet -c MANIFEST.sha256) || die "local kit $LK MANIFEST does not verify"
[ -n "$KNOBS" ] || die 'KNOBS="NAME NAME" is required'
[ -n "$ARMS" ] || die 'ARMS="base <name> ..." is required'
OWNED=" $(sed -nE 's/^#((opt)?switch|knob) ([A-Z0-9_]+) .*/\3/p' "$ENVR" | tr '\n' ' ') "
for n in $KNOBS; do
  [[ "$n" =~ ^[A-Z0-9_]+$ ]] || die "KNOBS: '$n' is not a NAME"
  [[ "$n" =~ (KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE|SESSION|PRIVATE) ]] && die "KNOBS: $n looks like a credential"
  [[ "$OWNED" == *" $n "* ]] || die "KNOBS: $n is not a #switch/#knob of $ENVR (the A/B only flips operator switches/knobs)"
done
KRE=$(echo $KNOBS | tr ' ' '\n' | paste -sd'|')
swvals(){ sed -nE "s/^#(opt)?switch $1 ([^ ]+).*/\2/p" "$ENVR" | head -1; }   # "0|1" / "" for a #knob
laylist(){ # a moe3 layer list: comma list of N or A-B, 3 <= N, A <= B <= 44
  local p a b IFS=,; [ -n "$1" ] || return 1
  for p in $1; do
    if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then a=$((10#${BASH_REMATCH[1]})); b=$((10#${BASH_REMATCH[2]})); [ $a -ge 3 ] && [ $a -le $b ] && [ $b -le 44 ] || return 1
    elif [[ "$p" =~ ^[0-9]+$ ]]; then a=$((10#$p)); [ $a -ge 3 ] && [ $a -le 44 ] || return 1
    else return 1; fi
  done; return 0; }
hilolist(){ # the hi+lo value: comma list of <group>.<proj>:<channels>, channels %16==0 in 16..2048
  local p nm ch IFS=,; [ -n "$1" ] || return 1
  for p in $1; do
    nm="${p%%:*}"; ch="${p#*:}"
    [ "$ch" != "$p" ] || return 1
    [[ "$nm" =~ ^(kda|mla|shared|dense)\.[a-z_]+$ ]] || return 1
    [[ "$nm" == draft.fc ]] && return 1
    [[ "$ch" =~ ^[0-9]+$ ]] || return 1
    [ $((10#$ch % 16)) -eq 0 ] && [ $((10#$ch)) -ge 16 ] && [ $((10#$ch)) -le 2048 ] || return 1
  done; return 0; }
skiplist(){ # the skip-layers value: comma list of <layer>[:<name>], layer N or A-B (<=4095), name a group/projection word
  local p rng nm IFS=,; [ -n "$1" ] || return 1
  for p in $1; do
    rng="${p%%:*}"; nm="${p#*:}"
    if [[ "$rng" =~ ^([0-9]{1,4})-([0-9]{1,4})$ ]]; then
      [ $((10#${BASH_REMATCH[1]})) -le $((10#${BASH_REMATCH[2]})) ] && [ $((10#${BASH_REMATCH[2]})) -le 4095 ] || return 1
    elif [[ "$rng" =~ ^[0-9]{1,4}$ ]]; then
      [ $((10#$rng)) -le 4095 ] || return 1
    else return 1; fi
    [ "$nm" = "$p" ] || [[ "$nm" =~ ^(kda|mla|dense|shared)(\.[a-z_]+)?$ ]] || return 1
    # r16z5rev V7: the name must be one the kit's fp8_w8a8.parse_skip accepts - the start.sh checks only the layer
    # part, and an unknown name (mla.bogus) makes the module refuse the WHOLE W8A8 install at boot (production's dense
    # FP8 path for every projection, incl. production's own ONLY=sub): a wasted restart pair caught only by boot_checks
    [ "$nm" = "$p" ] || [[ " $SKIPNAMES " == *" $nm "* ]] || return 1
  done; return 0; }
# the names parse_skip accepts (PROJ_NAMES | SKIP_EXTRA_NAMES | SKIP_GROUPS of the kit's overlay/fp8_w8a8.py)
SKIPNAMES=$(python3 - "$LK/overlay/fp8_w8a8.py" 2>/dev/null <<'PY'
import ast, sys
want = {"PROJ_NAMES", "SKIP_EXTRA_NAMES", "SKIP_GROUPS"}
names = set()
for node in ast.parse(open(sys.argv[1]).read()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in want:
        names |= set(ast.literal_eval(node.value.args[0]))
print(" ".join(sorted(names)))
PY
)
armvar(){ if declare -p "ARM_$1" >/dev/null 2>&1; then echo "ARM_$1"; else echo "ARM_$(echo "$1" | tr '[:lower:]' '[:upper:]')"; fi; }
armraw(){ [ "$1" = base ] && return 0; local v; v=$(armvar "$1"); echo "${!v:-}"; }
BASEL=""   # V3: base's KNOB lines = production's own (filled by the remote preflight; empty with BASE_DROPS_PROD=1)
armenv(){ # the EFFECTIVE lines of an arm: base's lines, overridden by the arm's; NAME= (empty) unsets NAME
  [ "$1" = base ] && { echo "$BASEL"; return 0; }
  local raw kv out=""; raw=$(armraw "$1")
  for kv in $BASEL; do [[ " $raw " == *" ${kv%%=*}="* ]] || out+="$kv "; done
  for kv in $raw; do [ -n "${kv#*=}" ] && out+="$kv "; done
  echo "$out"; }
proofof(){ [ "$1" = base ] && return 0; local v="PROOF_$1"; local u; u="PROOF_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  if [ -n "${!v+x}" ]; then echo "${!v}"; elif [ -n "${!u+x}" ]; then echo "${!u}"; else echo "$PROOF"; fi; }
neverof(){ [ "$1" = base ] && return 0; local v="NEVER_$1"; local u; u="NEVER_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  if [ -n "${!v+x}" ]; then echo "${!v}"; elif [ -n "${!u+x}" ]; then echo "${!u}"; else echo "$NEVER"; fi; }
set -- $ARMS; [ "$1" = base ] || die "ARMS must start with base (got '$1'): without a base measurement nothing can be judged and production would be left on the last arm"
[ "$(echo " $ARMS " | grep -o ' base ' | wc -l)" = 1 ] || die "base must appear exactly once in ARMS"
[ -z "${ARM_BASE+x}${ARM_base+x}" ] || die "ARM_base must not be set (base = production with every KNOB removed)"
case "$KEEP" in auto|base) ;; *) die "KEEP must be auto or base";; esac
seen=" "
for A in $ARMS; do
  [[ "$A" =~ ^[a-z0-9_]+$ ]] || die "arm name '$A': use [a-z0-9_]"
  [ "$A" != none ] || die "arm name 'none' is reserved"
  [[ "$seen" == *" $A "* ]] && die "arm $A appears twice"; seen+="$A "
  [ "$A" = base ] && continue
  lines=$(armraw "$A"); [ -n "$lines" ] || die "arm $A has no ARM_$A / $(armvar "$A") line"
  names=" "
  for kv in $lines; do
    [[ "$kv" =~ ^([A-Z0-9_]+)=([$VCHARS]*)$ ]] || die "arm $A: '$kv' is not NAME=VALUE with a [$VCHARS] value"
    n=${BASH_REMATCH[1]}; v=${BASH_REMATCH[2]}
    [[ " $KNOBS " == *" $n "* ]] || die "arm $A dials $n, which is not in KNOBS ($KNOBS)"
    [[ "$names" == *" $n "* ]] && die "arm $A sets $n twice"; names+="$n "
    [ -n "$v" ] || continue      # V3: NAME= unsets the KNOB for this arm (base keeps production's value)
    vs=$(swvals "$n")
    if [ -n "$vs" ]; then [[ "|$vs|" == *"|$v|"* ]] || die "arm $A: $n must be one of $vs (value not accepted by the kit)"; fi
    case "$n" in
      GLM53_MOE_E4M3_LAYERS|GLM53_MOE_E4M3_DOWN_LAYERS) laylist "$v" || die "arm $A: $n must be a comma list of MoE layers / ranges within 3..44";;
      # r16z5 knobs (validated here like the kit's start.sh validates them - before any restart is attempted):
      GLM53_MLA_PREFILL_KV_ROWS) [[ "$v" == all || "$v" =~ ^[0-9]+$ ]] || die "arm $A: GLM53_MLA_PREFILL_KV_ROWS must be 'all' or a non-negative integer";;
      GLM53_MHC_FUSED_ROUND_A) [[ "$v" == 0 || "$v" == 1 ]] || die "arm $A: GLM53_MHC_FUSED_ROUND_A must be 0 or 1";;
      GLM53_MHC_FUSED_CFG) [[ "$v" =~ ^[0-9]+$ ]] || die "arm $A: GLM53_MHC_FUSED_CFG must be a decimal kernel cfg";;
      GLM53_DENSE_W8A8_HILO) hilolist "$v" || die "arm $A: GLM53_DENSE_W8A8_HILO must be a comma list of <group>.<proj>:<channels> (channels a multiple of 16 in 16..2048)";;
      GLM53_DENSE_W8A8_SKIP_LAYERS) skiplist "$v" || die "arm $A: GLM53_DENSE_W8A8_SKIP_LAYERS must be a comma list of <layer>[:<name>] (N or A-B <= 4095; name one of: ${SKIPNAMES:-<unreadable: the kit fp8_w8a8.py>})";;
    esac
  done
  [ "$(proofof "$A")" = "-" ] || [ -n "$(proofof "$A")" ] || say "note: arm $A has no PROOF string (it is judged without a served-on-both-ranks check)"
  case "$(proofof "$A")$(neverof "$A")" in *"'"*) die "arm $A: a PROOF/NEVER string must not contain a single quote";; esac
done
if [ -n "$KP" ]; then
  [ -n "$PRE_START_SHA" ] && [[ "$PRE_START_SHA" =~ ^[0-9a-f]{8}$ ]] || die "swap mode needs PRE_START_SHA (8 hex of the live start.sh)"
  KPSHA=$(awk '$2=="./launcher/start.sh"{print $1}' "$LK/prev-r16/MANIFEST.sha256" 2>/dev/null)
  [ -n "$KPSHA" ] || die "$K has no prev-r16/MANIFEST launcher/start.sh entry: it is not an update kit"
fi
if [ -n "${ARM_PROBE:-}" ]; then   # r16z7: validated before anything is touched (it runs on nodeA inside double quotes)
  [[ "$ARM_PROBE" =~ ^[A-Za-z0-9_./~{}=:,\ -]+$ ]] || die "ARM_PROBE may only contain [A-Za-z0-9_./~{}=:, -] (no quotes, \$, ;, |, backticks)"
  [[ "${PROBE_LINE:-CONCQ}" =~ ^[A-Za-z0-9_-]+$ ]] || die "PROBE_LINE must be a word"
  [[ "${PROBE_TIMEOUT:-900}" =~ ^[0-9]+$ ]] || die "PROBE_TIMEOUT must be seconds"
fi
mkdir -p "$OUT"; exec 9> "$OUT/.arm_env.lock"; flock -n 9 || die "another arm_env.sh run holds $OUT/.arm_env.lock"
rm -f "$OUT"/res-*.txt

# ------------------------------------------------------------------ remote helpers
health(){ for i in $(seq 1 ${1:-120}); do [ "$(n2 "curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8888/health")" = 200 ] && return 0; sleep ${HEALTH_SLEEP:-10}; done; return 1; }
# both ranks' KNOB env as "head:NAME=v ... | worker:NAME=v ..." (sorted, exactly the KNOB names; values are 0/1 flags,
# enum words and layer lists: env.r16 #switch/#knob values, not secrets)
envof(){ n2 "docker inspect $HEAD_C --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^($KRE)=' | LC_ALL=C sort | tr '\n' ' '; echo -n '| '; ssh -o BatchMode=yes $W \"docker inspect $WORK_C --format '{{range .Config.Env}}{{println .}}{{end}}'\" | grep -E '^($KRE)=' | LC_ALL=C sort | tr '\n' ' '" 2>/dev/null; }
want(){ # <arm> -> what envof must print for it (the kit start.sh forwards every KNOB, unset as NAME=)
  local n kv v out=""
  for n in $KNOBS; do v=""; for kv in $(armenv "$1"); do [ "${kv%%=*}" = "$n" ] && v="${kv#*=}"; done; out+="$n=$v"$'\n'; done
  out=$(printf '%s' "$out" | LC_ALL=C sort | tr '\n' ' '); echo "$out| $out"; }
# A1: both containers' ABLIT as "<head>|<worker>" (0/1: not a secret)
ablof(){ n2 "h=\$(docker inspect $HEAD_C --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^ABLIT=' | tail -n 1 | cut -d= -f2-); w=\$(ssh -o BatchMode=yes $W \"docker inspect $WORK_C --format '{{range .Config.Env}}{{println .}}{{end}}'\" | grep -E '^ABLIT=' | tail -n 1 | cut -d= -f2-); echo \"\$h|\$w\"" 2>/dev/null; }
RABL=0   # A1: the ABLIT every restart passes (set by the remote preflight from the running containers)
started(){ n2 "docker inspect $HEAD_C --format '{{.State.StartedAt}}'" 2>/dev/null; }
headup(){ [ "$(n2 "docker inspect $HEAD_C --format '{{.State.Running}}'" 2>/dev/null)" = true ]; }
setenv(){ # <arm>: .env's KNOB lines become the arm's effective lines; rc = the remote rc
  # r16z5rev V5: a KNOB line production already has is rewritten IN PLACE (the first one; later duplicates dropped),
  # a new one is appended, an unset one removed - so base / KEEP=base ends with production's .env byte for byte (the
  # old delete-then-append moved production's own KNOB lines, e.g. GLM53_DENSE_W8A8_ONLY, to the end). Written via a
  # temp file + rename with .env's mode kept (it holds the API key); a missing trailing newline is not glued to.
  local spec="" n kv v
  for n in $KNOBS; do v=""; for kv in $(armenv "$1"); do [ "${kv%%=*}" = "$n" ] && v="${kv#*=}"; done; spec+="$n=$v "; done
  # base (= production's own KNOB lines) with nothing else in .env changed since the pre-A/B backup: put the backup
  # back byte for byte (an arm that UNSET a production KNOB lost its line position)
  local bk=""; [ "$1" = base ] && [ "${BASE_DROPS_PROD:-0}" != 1 ] && bk=".env.bak-pre-envab-$TS"
  n2 "cd ~/$G && if [ -n '$bk' ] && [ -f '$bk' ] && cmp -s <(grep -vE '^($KRE)=' .env) <(grep -vE '^($KRE)=' '$bk'); then cp -p '$bk' .env && echo env-now: base = the pre-A/B .env; else python3 - $spec <<'PY' && echo env-now: \$(grep -oE \"^($KRE)=\" .env | tr '\n' ' ')
import os, sys
want = dict(a.split('=', 1) for a in sys.argv[1:])
nl = chr(10)
src = open('.env').read()
lines = src.split(nl)
if src.endswith(nl):
    lines = lines[:-1]
out, done = [], set()
for ln in lines:
    n = ln.split('=', 1)[0] if '=' in ln else None
    if n in want:
        if want[n] and n not in done:
            out.append(n + '=' + want[n])
            done.add(n)
        continue
    out.append(ln)
for n in sorted(want):
    if want[n] and n not in done:
        out.append(n + '=' + want[n])
tmp = '.env.envab-tmp'
with open(tmp, 'w') as f:
    f.write(nl.join(out) + nl)
os.chmod(tmp, os.stat('.env').st_mode & 0o7777)
os.replace(tmp, '.env')
PY
fi"; }
# the INSTALLED start.sh's own validation (its functions; main not run) with the arm's KNOB values applied on top of
# .env: exactly what `./start.sh start` would refuse after restart2.sh has already stopped both ranks
vcheck(){ local a=$1 kv ex="unset $KNOBS;"
  for kv in $(armenv "$a"); do ex+=" export $kv;"; done
  local chain; chain=$(echo $VALIDATE_CHAIN | sed 's/ / \&\& /g')
  local o; o=$(n2 "cd ~/$G && [ \"\$(tail -n 1 start.sh)\" = 'main \"\$@\"' ] && sed '\$d' start.sh > .envab-check-$TS.sh && echo '$ex $chain && echo ENVAB-VALIDATE-OK' >> .envab-check-$TS.sh; r=\$?; [ \$r = 0 ] && ABLIT=$RABL bash .envab-check-$TS.sh 2>&1 | tail -n 3; rm -f .envab-check-$TS.sh" 2>&1)
  if echo "$o" | grep -q '^ENVAB-VALIDATE-OK$'; then return 0; fi
  say "arm $a: the installed start.sh REFUSES it (it would have stopped both ranks first): $(echo "$o" | grep -v ENVAB | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; return 1; }
CUR=none; CURBAD=0   # the arm the containers run (none = the pre-A/B runtime); CURBAD=1: it failed a check
restart_to(){ # <arm> <label>: rc 0 ok / 3 refused (busy; nothing restarted) / 4 refused config (nothing restarted) / 5 broken
  local A=$1 L=$2 s0 rc got
  vcheck "$A" || return 4
  setenv "$A" || { say "writing .env for $A failed"; return 4; }
  s0=$(started)
  n2 "ABLIT=$RABL IDLE_WAIT_S=$IDLE_WAIT_S bash ~/tf-exl3-deploy/restart2.sh $L 2>&1 | tail -n 3"; rc=$?
  if ! headup; then say "after restart2 (rc $rc) the head container is NOT running: production is DOWN"; CUR=$A; CURBAD=1; return 5; fi
  if [ $rc = 3 ] || [ "$(started)" = "$s0" ]; then say "restart to $A REFUSED/not done (rc $rc; production busy)"; return 3; fi
  CUR=$A; CURBAD=1
  health 90 || { say "$A NOT HEALTHY"; return 5; }
  got=$(envof)
  [ "$got" = "$(want $A)" ] || { say "runtime env mismatch for $A: got [$got] want [$(want $A)]"; return 5; }
  got=$(ablof); [ "$got" = "$RABL|$RABL" ] || { say "runtime ABLIT mismatch for $A: got [$got] want [$RABL|$RABL] (head|worker)"; return 5; }
  CURBAD=0; return 0; }
bootok(){ local BC; BC=$(n2 "bash ~/${BCKIT:-$K}/tools/boot_checks.sh $* 2>&1")
  echo "$BC" | grep -E '^(MISS|BAD) ' | head -25
  echo "$BC" | grep -E '^(ok|MISS|BAD) ' | grep -E "$KRE" | head -14
  echo "$BC" | grep -qE '^boot_checks: (ALL OK|PROBLEMS)$' || { say "boot_checks did not run to its end"; return 1; }
  echo "$BC" | grep -qE '^ok +\[mhcsp-pair\]' || { say "no ok [mhcsp-pair]"; return 1; }
  echo "$BC" | grep -E '^(MISS|BAD) ' | grep -vE '^BAD +\[(head|worker)\] \([0-9]+\) idx_gate result differs$' | grep -q . && return 1; return 0; }
PRE_ENV=""
restore_files(){ # the pre-A/B start.sh/.env (+ revert_r16 to the pre-A/B backup in swap mode); no restart
  n2 "cd ~/$G && B=\$(ls -dt ~/tf-exl3-deploy/backup-r16-* 2>/dev/null | head -n 1) && if [ -n \"$KP\" ] && [ -n \"\$B\" ] && [ \"\$B\" != \"$PREB\" ]; then bash ~/$K/tools/revert_r16.sh \$B --apply 2>&1 | tail -n 2; fi && cp -p start.sh.bak-pre-envab-$TS start.sh && cp -p .env.bak-pre-envab-$TS .env && [ \$(sha256sum start.sh | cut -c1-8) = $PRESHA ] && cmp -s .env .env.bak-pre-envab-$TS && echo restore-files-ok" \
    && return 0
  say "RESTORE FILES FAILED - operator needed (backups: ~/$G/start.sh.bak-pre-envab-$TS / .env.bak-pre-envab-$TS, pre-A/B backup dir $PREB)"; return 1; }
restore_all(){ # files back + restart + verify; exits
  say "RESTORE the pre-A/B state (${KP:-$K})"
  restore_files || exit 12
  local s0 t rc; s0=$(started)
  for t in $(seq 1 $RECOVER_TRIES); do
    n2 "ABLIT=$RABL IDLE_WAIT_S=$IDLE_WAIT_S bash ~/tf-exl3-deploy/restart2.sh restore-envab-$TS 2>&1 | tail -n 3"; rc=$?
    { [ $rc = 3 ] || { headup && [ "$(started)" = "$s0" ]; }; } || break
    say "restore restart refused (busy), try $t/$RECOVER_TRIES"; done
  if headup && [ "$(started)" = "$s0" ]; then say "RESTORE: files restored but production stayed busy - the runtime is still $CUR (the next restart boots the restored files)"; exit 11; fi
  health 90 || { say "RESTORE NOT HEALTHY - operator needed"; exit 12; }
  [ "$(envof)" = "$PRE_ENV" ] || { say "RESTORE: runtime KNOB env [$(envof)] != pre-A/B [$PRE_ENV] - operator needed"; exit 12; }
  [ "$(ablof)" = "$RABL|$RABL" ] || { say "RESTORE: runtime ABLIT [$(ablof)] != pre-A/B [$RABL|$RABL] (head|worker) - operator needed"; exit 12; }
  BCKIT=${KP:-$K} bootok > /dev/null && say "restored: healthy, pre-A/B env on both ranks, boot_checks ok" || say "restored: healthy, pre-A/B env; boot_checks of ${KP:-$K} reports problems (see above)"
  exit "${1:-9}"; }
settle(){ # <target> <label>: get production onto <target> (retry while busy); 0 ok, else exits
  local T=$1 L=$2 t rc
  for t in $(seq 1 $RECOVER_TRIES); do
    restart_to "$T" "$L-$t"; rc=$?
    [ $rc = 3 ] || break; say "restart to $T refused (busy), try $t/$RECOVER_TRIES"; done
  if [ $rc = 0 ] && bootok > /dev/null; then return 0; fi
  if [ $rc = 3 ]; then
    setenv "$T" > /dev/null
    say "PRODUCTION STAYED BUSY: the runtime is still arm $CUR$([ $CURBAD = 1 ] && echo ' (which FAILED a check)'), .env is set to $T so the next restart boots $T: run 'ABLIT=$RABL ~/tf-exl3-deploy/restart2.sh <label>' + boot_checks on an idle window"; exit 10; fi
  [ "$T" != base ] && { say "$T did not come up cleanly (rc $rc) -> base"; settle base "$L-base"; return 0; }
  say "base did not come up cleanly (rc $rc) -> full restore"; restore_all 9; }


# ------------------------------------------------------------------ remote preflight + (swap) install
PREB=$(n2 "ls -dt ~/tf-exl3-deploy/backup-r16-* 2>/dev/null | head -n 1")
PRESHA=$(n2 "sha256sum ~/$G/start.sh | cut -c1-8") || die "cannot read nodeA start.sh"
PRE_ENV=$(envof); [ -n "$PRE_ENV" ] || die "cannot read the containers' env"
PRE_ABL=$(ablof) || die "cannot read the containers' ABLIT"; _ah=${PRE_ABL%%|*}; _aw=${PRE_ABL#*|}   # A1
[ "$_ah" = "$_aw" ] || die "the head and worker containers carry different ABLIT values ('$_ah' / '$_aw'): fix production first"
case "$_ah" in 0|1) ;; "") _ah=0;; *) die "the containers' ABLIT is neither 0 nor 1: decide by hand";; esac
[ -z "${ABLIT:-}" ] || [ "$ABLIT" = "$_ah" ] || die "ABLIT=$ABLIT was given but production runs ABLIT=$_ah: an A/B keeps production's ablit state (restart production with the wanted ABLIT first, or omit ABLIT)"
RABL=$_ah; say "ablit: production runs ABLIT=$RABL on both ranks; every restart of this A/B passes ABLIT=$RABL (start.sh honours only a caller-exported ABLIT)"
started > /dev/null && headup || die "the head container is not running: not starting an A/B on a down production"
prodset=$(n2 "grep -E '^($KRE)=.' ~/$G/.env | cut -d= -f1 | tr '\n' ' '")
if [ -n "${prodset// /}" ] && [ "${BASE_DROPS_PROD:-0}" != 1 ]; then
  # V3: base keeps production's own KNOB lines (the last line of each name, as start.sh's .env read sees it); the
  # values are env.r16 switch/knob values (not secrets) and must be plain [$VCHARS] words (no quotes / comments)
  for n in $prodset; do
    pl=$(n2 "grep -E '^$n=' ~/$G/.env | tail -n 1") || die "cannot read production's $n line"
    [[ "$pl" =~ ^$n=([$VCHARS]+)$ ]] || die "production's .env $n line is not a plain NAME=[$VCHARS] value (quotes, spaces or a trailing comment): normalize it by hand first (value not printed)"
    BASEL+="$pl "
  done
  say "base = production's KNOB lines as they are: $prodset(an arm overrides them; NAME= in an arm unsets one)"
  for A in $ARMS; do for kv in $(armenv "$A"); do n=${kv%%=*}; v=${kv#*=}
    case "$n" in GLM53_MOE_E4M3_LAYERS|GLM53_MOE_E4M3_DOWN_LAYERS) laylist "$v" || die "arm $A (with base's lines): $n is not a valid layer list";;
      GLM53_DENSE_W8A8_HILO) hilolist "$v" || die "arm $A (with base's lines): $n is not a valid hi+lo list";;
      GLM53_DENSE_W8A8_SKIP_LAYERS) skiplist "$v" || die "arm $A (with base's lines): $n is not a valid skip list";; esac
    vs=$(swvals "$n"); [ -z "$vs" ] || [[ "|$vs|" == *"|$v|"* ]] || die "arm $A (with base's lines): $n must be one of $vs"; done; done
fi
if [ -n "$KP" ]; then
  [ "$PRESHA" = "$PRE_START_SHA" ] || die "live start.sh is $PRESHA, not PRE_START_SHA=$PRE_START_SHA" 3
  # The previous kit's launcher/start.sh that $K's prev-r16/MANIFEST lists must exist on nodeA as ~/$KP/... verbatim.
  if [ "${KPSHA:0:8}" != "$PRE_START_SHA" ]; then
    n2 "[ \$(sha256sum ~/$KP/launcher/start.sh | cut -d' ' -f1) = $KPSHA ]" || die "~/$KP/launcher/start.sh on nodeA is not the plain start.sh $K was built against (${KPSHA:0:16}): wrong PREV_KIT_NAME" 3
  fi
  # ~/$KP must be the kit $K was built from: MANIFEST ok and its start.sh == KPSHA.
  n2 "cd ~/$KP && sha256sum --quiet -c MANIFEST.sha256 && [ \$(sha256sum launcher/start.sh | cut -d' ' -f1) = $KPSHA ]" \
    || die "~/$KP on nodeA is not the previous kit $K's prev-r16/MANIFEST names (missing, MANIFEST fails, or its start.sh is not ${KPSHA:0:16}): wrong PREV_KIT_NAME" 3
  say "copy kit + swap from $KP"
  tar -C $HOMEDIR -czf $OUT/kit-$K.tgz $K && scp -q $OUT/kit-$K.tgz "$HEAD_SSH":$HOMEDIR/ && n2 "rm -rf ~/$K && tar -C ~ -xzf ~/kit-$K.tgz && cd ~/$K && sha256sum --quiet -c MANIFEST.sha256 && echo MANIFEST-ok" || die "kit copy failed" 2
  n2 "cd ~/$G && cp -p start.sh start.sh.bak-pre-envab-$TS && cp -p .env .env.bak-pre-envab-$TS && echo backups-ok" || die "backups failed" 3
  n2 "cd ~/$G && if [ \$(sha256sum start.sh | cut -d' ' -f1) != $KPSHA ]; then cp ~/$KP/launcher/start.sh start.sh; fi; bash ~/$K/tools/apply_r16.sh 2>&1 | tail -n 20" \
    || { say "apply dry run failed"; restore_files; exit 4; }
  n2 "cd ~/$G && bash ~/$K/tools/apply_r16.sh --apply 2>&1 | tail -n 6" || { say "apply failed"; restore_files; exit 4; }
else
  say "no PREV_KIT_NAME: production must already run $K; only .env is flipped"
  n2 "cd ~/$K && sha256sum --quiet -c MANIFEST.sha256 && cd ~/$G && s=\$(sha256sum start.sh | cut -d' ' -f1) && [ \$s = \$(sha256sum ~/$K/launcher/start.sh | cut -d' ' -f1) ] && diff -rq ~/$K/site overlay/tf/site > /dev/null && diff -rq ~/$K/overlay overlay/tf/overlay > /dev/null && for f in ~/$K/launcher/overlay/*.py; do cmp -s \$f overlay/\$(basename \$f) || exit 1; done && echo kit-installed-ok" \
    || die "production does not run $K (start.sh / overlay/tf / launcher overlays differ, or the kit is missing on nodeA)" 3
  n2 "cd ~/$G && cp -p start.sh start.sh.bak-pre-envab-$TS && cp -p .env .env.bak-pre-envab-$TS && echo backups-ok" || die "backups failed" 3
fi
for A in $ARMS; do vcheck "$A" || { say "arm $A refused by the installed start.sh - nothing was restarted; putting the pre-A/B files back"; restore_files; exit 4; }; done
say "every arm accepted by the installed start.sh ($ARMS)"

# ------------------------------------------------------------------ the arms
arm(){ # <name> -> $OUT/res-<name>.txt : g32 long ltop dvp dec
  local A=$1 L="envab-$A-$TS" rc; say "=== arm $A"
  restart_to $A $L; rc=$?; [ $rc = 0 ] || return $rc
  bootok || { CURBAD=1; say "$A boot abort"; return 7; }
  local M; M=$(n2 "bash ~/tf-exl3-deploy/r16_measure.sh $L 2>&1"); echo "$M" | tail -n 40
  local long ltop dvp dec; dec=$(echo "$M" | grep -oP 'structured tok/s med \K[0-9.]+' | tail -n 1); long=$(echo "$M" | grep -oP 'long-context .*KL mean \K[0-9.]+' | tail -n 1); ltop=$(echo "$M" | grep -oP 'long-context .*top-1 agree \K[0-9.]+' | tail -n 1); dvp=$(echo "$M" | grep -oP 'KL mean before [0-9.]+ -> after \K[0-9.]+' | tail -n 1)
  local GT; GT=$(for s in 71 72 73 74 75 76; do python3 "$GATE" 32k $s 2>&1; done)
  echo "$GT" | grep -E '^(before|after| +[0-9]+ )'
  # a 32k row counts only if the server ran nothing else right before AND right after it
  local g32 n32; read g32 n32 < <(echo "$GT" | awk '/^before /{idle=($3=="running=0.0" && $4=="waiting=0.0"); v=""} /^ +[0-9]+ +[0-9.]+ +[0-9]+ /{if($1<64000 && idle) v=$3} /^after /{if(v!="" && $3=="running=0.0" && $4=="waiting=0.0") print v; v=""}' \
       | sort -n | awk '{a[NR]=$1}END{if(NR==0)print "x 0"; else print (a[int((NR+1)/2)]+a[int(NR/2)+1])/2, NR}')
  if [ -n "${ARM_PROBE:-}" ]; then   # r16z7: the operator's per-arm probe (informational, never invalidates the arm)
    local PO prc pl; PO=$(n2 "timeout ${PROBE_TIMEOUT:-900} ${ARM_PROBE//\{label\}/$L} 2>&1"); prc=$?
    printf '%s\n' "$PO" > "$OUT/probe-$A.log"
    pl=$(printf '%s\n' "$PO" | grep -E "^${PROBE_LINE:-CONCQ} " | tail -n 1)
    say "PROBE $A (rc $prc): ${pl:-<no ${PROBE_LINE:-CONCQ} line; see $OUT/probe-$A.log>}"
    echo "$A rc=$prc ${pl:-none}" >> "$OUT/probe-summary-$TS.txt"
  fi
  health 6 || { CURBAD=1; say "$A unhealthy after measurement"; return 6; }
  local pf; pf=$(proofof "$A")
  if [ "$A" != base ] && [ -n "$pf" ] && [ "$pf" != - ]; then   # the arm's path must have served on BOTH ranks (R5)
    # r16z8: PROOF may list several fixed strings separated by '|' (a combined arm: one per feature); EVERY one must
    # be in BOTH ranks' logs (an array, not a read loop: n2's ssh would eat the loop's stdin)
    local sv pfs pp; IFS='|' read -r -a pfs <<< "$pf"
    for pp in "${pfs[@]}"; do
      [ -n "$pp" ] || continue
      sv=$(n2 "h=\$(docker logs $HEAD_C 2>&1 | grep -cF -- '$pp'); w=\$(ssh -o BatchMode=yes $W \"docker logs $WORK_C 2>&1 | grep -cF -- '$pp'\"); echo \$h \$w")
      set -- $sv; { [ "${1:-0}" -ge 1 ] 2>/dev/null && [ "${2:-0}" -ge 1 ] 2>/dev/null; } || { say "$A: '$pp' head=${1:-?} worker=${2:-?} (never served on a rank) -> arm invalid"; return 9; }
    done
  fi
  local nv; nv=$(neverof "$A")
  if [ "$A" != base ] && [ -n "$nv" ] && [ "$nv" != - ]; then   # r16z6rev: a refusal that only shows under traffic
    local nvc nvs nn; IFS='|' read -r -a nvs <<< "$nv"   # r16z8: several fixed strings separated by '|', NONE may appear
    for nn in "${nvs[@]}"; do
      [ -n "$nn" ] || continue
      nvc=$(n2 "h=\$(docker logs $HEAD_C 2>&1 | grep -cF -- '$nn'); w=\$(ssh -o BatchMode=yes $W \"docker logs $WORK_C 2>&1 | grep -cF -- '$nn'\"); echo \$h \$w")
      set -- $nvc; { [ "${1:-x}" = 0 ] && [ "${2:-x}" = 0 ]; } || { CURBAD=1; say "$A: NEVER '$nn' head=${1:-?} worker=${2:-?} (appeared during the measurement, or unreadable) -> arm invalid"; return 9; }
    done
  fi
  for v in "$g32" "$long" "$ltop" "$dvp" "$dec"; do isnum "$v" || { say "$A: a measurement is missing (g32=$g32 long=$long top1=$ltop dvp=$dvp dec=$dec) -> arm invalid"; return 8; }; done
  [ "$n32" -ge 3 ] || { say "$A: only $n32 idle 32k gate samples -> arm invalid"; return 8; }
  echo "$g32 $long $ltop $dvp $dec" > $OUT/res-$A.txt; say "RESULT $A: gate32k-median=$g32 (n=$n32) long=$long top1=$ltop dvp=$dvp decode-structured=$dec"; return 0; }
basegate(){ # V4: swap mode - base IS the new kit (arm (a)); it must pass the absolute quality limits itself
  local g l t d dec; read g l t d dec < $OUT/res-base.txt
  local ok; ok=$(awk -v l=$l -v d=$d -v t=$t -v g=$g -v LM=$BASE_LMAX -v DM=$BASE_DMAX -v TM=$BASE_TMIN -v GM="${BASE_GMIN:-0}" \
       'BEGIN{print (l+0<=LM+0 && d+0<=DM+0 && t+0>=TM+0 && g+0>=GM+0)?1:0}')
  say "base gate (swap mode: base = $K with production's switches): long $l <= $BASE_LMAX, dvp $d <= $BASE_DMAX, top1 $t >= $BASE_TMIN${BASE_GMIN:+, gate $g >= $BASE_GMIN} -> $ok"
  [ "$ok" = 1 ]; }
for A in $ARMS; do
  arm $A; rc=$?
  if [ $rc = 0 ] && [ "$A" = base ] && [ -n "$KP" ]; then
    basegate || { say "the new kit's base FAILED its gate: back to the pre-A/B kit ${KP}"; restore_all 14; }
  fi
  [ $rc = 0 ] && continue
  if [ "$A" = base ]; then
    if [ $rc = 3 ] && [ "$CUR" = none ]; then say "the first restart was refused (busy): nothing restarted -> the pre-A/B files go back, A/B not run"; restore_files || exit 12; exit 3; fi
    say "base arm failed (rc $rc)"; restore_all 8
  fi
  if [ $rc = 3 ] || [ $rc = 4 ]; then
    say "restart to $A not done (rc $rc): production still runs $CUR; A/B stopped here"
    setenv "$CUR" > /dev/null || say "WARNING: .env could not be set back to $CUR"
    break
  fi
  say "$A failed (rc $rc) -> back to base"; settle base envab-base-$TS
done
# ------------------------------------------------------------------ judgement + final state
[ -s $OUT/res-base.txt ] || { say "no base result - restoring"; restore_all 8; }
read BG _ _ _ BDEC < $OUT/res-base.txt; BEST=base; BESTG=$BG
for A in $ARMS; do [ "$A" = base ] && continue; [ -s $OUT/res-$A.txt ] || continue; read g l t d dec < $OUT/res-$A.txt
  ok=$(awk -v l=$l -v d=$d -v t=$t -v g=$g -v bg=$BG -v best=$BESTG -v LM=$LMAX -v DM=$DMAX -v TM=$TMIN -v GM=$GMARGIN -v dec=$dec -v bdec=$BDEC -v DD=$DECMIN \
       'BEGIN{print (l+0<=LM+0 && d+0<=DM+0 && t+0>=TM+0 && g+0>GM*bg && g+0>best+0 && dec+0>=DD*bdec)?1:0}')
  say "judge $A: gate $g long $l top1 $t dvp $d dec $dec -> $ok"; [ "$ok" = 1 ] && { BEST=$A; BESTG=$g; }; done
[ "$KEEP" = base ] && { say "KEEP=base: the best arm was $BEST; production ends on base"; BEST=base; }
say "FINAL keep: $BEST (running now: $CUR)"
if [ "$BEST" != "$CUR" ] || [ $CURBAD = 1 ]; then settle "$BEST" envab-final-$BEST-$TS; fi
say "done: runtime [$(envof)] (expected arm $CUR = [$(want $CUR)])"
[ "$(envof)" = "$(want $CUR)" ] || { say "WARNING: runtime env != the expected arm"; exit 13; }
