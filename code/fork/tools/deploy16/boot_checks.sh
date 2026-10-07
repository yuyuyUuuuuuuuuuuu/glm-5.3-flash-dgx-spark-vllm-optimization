#!/usr/bin/env bash
# deploy-r16 (nodeA, operator): per-rank boot-log checks after the R16 restart. Reads the head container's logs locally
# and the worker's on nodeB over ssh (W, as ~/tf-exl3-deploy/restart2.sh does). Prints only the matched marker lines,
# masked; exit 1 if an expected line is missing or a refusal line is present on either rank.
# Every string below is printed by the code that ships (bundle modules, overlays, launcher-apc overlays); the plugin-load
# lines were also seen in nodeC's in-container census of this kit (tools/deploy16/kit_chain.sh K.4 census C).
# Usage: boot_checks.sh [--after-traffic]   (--after-traffic also requires the first-use lines: send one prompt of
#        >= 4k tokens and a few decode turns first)
#   R16_ENV=auto (default)  each R16 feature is checked in the state the HEAD container's env gives it (the kit's
#                           env.r16 value(s) = on: its lines required, its refusals absent; its line(s) absent = off:
#                           its install lines must be absent, e.g. after tools/env_r16.sh off <feature>); the worker's
#                           container must carry exactly the head's env.r16 lines (the R1 incident); a flag with a value
#                           that is not the kit's -> MISS (decide by hand)
#   R16_ENV=on              every feature on (every env.r16 line in both containers): the full R16 state
#   R16_ENV=off             every feature off (all env.r16 feature lines removed): the "skipped (stock)" / inert lines
# A container env line with an EMPTY value (NAME=) is the same as no line: the launcher forwards every knob it knows to
# both ranks, unset ones as "" (= the module default), so the containers of a feature that is off carry `NAME=`.
# r16j operator switch GLM53_REJECTION_METHOD (env.r16 `#switch`; checked in every R16_ENV mode, by the HEAD container's
# env; unset/empty = standard): both containers must carry the same value; block -> on each rank the bundle ran
# patch_spec_block_keys.py (both files patched) and the DFlash2 drafter uses the row keys, the head's `launching: vllm
# serve` line has "rejection_sample_method":"block" (--after-traffic: also the verifier's one-time block-keys line);
# standard/unset -> no block-keys line anywhere, "rejection_sample_method":"standard" on the head, and each rank's bundle
# printed its skip line for the overlay (which proves the r16j bundle script runs there). Never, in any state, the
# resample-noise patch's "block is not exact" WARNING. Containers that carry NO GLM53_REJECTION_METHOD line at all
# (not even empty) were started by a start.sh without the switch (r16i or older, e.g. after a full revert): standard,
# no block-keys line allowed, the skip line not required (that bundle script does not print it).
# r16k operator switch GLM53_KDA_STRIDED_QKV (env.r16 `#switch`, kdaqkv; by the HEAD container's env; unset/empty/0 =
# stock): both containers must carry the same value; 1 -> on each rank the bundle patched ops/fused_recurrent.py and
# ops/kda.py; unset/empty -> each rank's bundle printed its skip line, 0 -> the overlay's "=0: stock" line, and no rank
# may carry a "patched" line. Neither container carrying the variable = a start.sh without the knob (r16j or older):
# stock, no patched line allowed (an r16j start.sh ignores a GLM53_KDA_STRIDED_QKV line of .env).
# r16l operator switch GLM53_KPOOL_DROP_LOWEST (env.r16 `#switch`, kpooldown; by the HEAD container's env; unset/empty/0
# = stock): both containers must carry the same value; 1 -> on each rank the bundle patched
# model_executor/layers/sparse_attn_indexer_kpool.py; unset/empty -> each rank's bundle printed its skip line, 0 -> the
# overlay's "=0: stock" line, and no rank may carry a "patched" line. Neither container carrying the variable = a
# start.sh without the knob (r16k or older): stock, no patched line allowed.
# r16n operator switch GLM53_KDA_FLASHKDA (env.r16 `#switch`, flashkda; by the HEAD container's env; unset/empty/0 =
# stock): both containers must carry the same value; 1 -> on each rank the bundle installed glm53_flashkda.py +
# _flashkda_fp32_C.abi3.so and patched models/glm5next/nvidia/kda.py (the KDA chunked prefill then logs its one-time
# FlashKDA line at model build); unset/empty -> each rank's bundle printed its skip line, 0 -> the overlay's
# "=0: stock" line, and no rank may carry a "patched" line. Neither container carrying the variable = a start.sh
# without the knob (r16l or older): stock, no patched line allowed.
# r16y operator switch GLM53_DENSE_W8A8 (env.r16 `#switch`, w8a8; by the HEAD container's env; unset/empty/0 =
# stock): both containers must carry the same value; 1 -> on each rank the bundle overlay patch_dense_w8a8.py
# installed fp8_w8a8.py + tf_fp8_w8a8_ext into site-packages, armed the site integrate.py AND the plugin printed
# its install line (the feature is active at the first eager prefill call >= 512 rows, so --after-traffic
# additionally requires the "first call served" line); unset/empty -> each rank's bundle printed its skip line,
# 0 -> the overlay's "=0: stock" line (any installed copy removed, integrate.py disarmed), and no rank may carry an
# installed/active line. Neither container carrying the variable = a start.sh without the switch (r16n or older):
# stock, no installed line allowed.
# r16msp operator switch GLM53_MHC_SP (env.r16 `#switch`, mhcsp; by the HEAD container's env; unset/empty/0 = stock):
# both containers must carry the SAME value and, at 1, BOTH ranks' logs must carry "model.py: patched;" BEFORE every
# traffic check (D-A, adversarial review 2: the SP prefill is a paired reduce-scatter/all-gather collective at TP=2 -
# one rank on = the residual stream is silently corrupted, ABORT); 1 -> on each rank the bundle patched
# vllm/models/glm5next/nvidia/model.py and extended the quickwins/moeglue fingerprint tables; unset/empty -> each
# rank's bundle printed its skip line, 0 -> the overlay's "=0: stock" line, and no rank may carry a "patched" line.
# Neither container carrying the variable = a start.sh without the knob (r16l or older): stock, no patched line.
# r16e4 operator switch GLM53_MOE_E4M3 (env.r16 `#switch`, moee4m3; by the HEAD container's env; unset/empty/0 =
# stock): both containers must carry the same value; 1 -> on each rank the bundle installed glm53_moe_e4m3 + its
# extension and armed integrate.py, the plugin installed, the load-time self-test passed, the summary says
# "42/42 layers served, 0 fell back" and the first prefill call ran on the e4m3 kernels (vLLM's profile run is
# one); never a self-test FAILED / raised / layer-not-served / DECODE BOUND EXCEEDED line; unset/empty -> each rank's
# bundle printed its skip line, 0 -> the overlay's "nothing installed" line, and no install line anywhere. Neither
# container carrying the variable = a start.sh without the knob (r16l or older): stock, no install line allowed.
# r16z operator switch GLM53_KDA_FLASHKDA_V (env.r16 `#switch`, fkda-v; by the HEAD container's env; unset/empty/1
# = the SHIPPED r16x FlashKDA build): both containers must carry the same value; the value only matters when
# GLM53_KDA_FLASHKDA=1, and decides WHICH build the bundle stages (all three under the site names
# glm53_flashkda.py / _flashkda_fp32_C.abi3.so): unset/1 -> the shipped build (the model-build line names no
# extension sha); 2 -> the fkda2 precision build (staging glm53_flashkda2.py sha256 dd1788c2..., boot line ends
# "extension sha256 dd1788c2... = fkda2 precision build"); 3 -> the fkda3 build (staging glm53_flashkda3.py
# sha256 d98adc4a..., boot line ends "extension sha256 d98adc4a... = fkda3 precision build"). The staging or
# boot line of the WRONG build anywhere is a MISS (a rank would serve a different kernel than its peer).
# r16z2 operator switch GLM53_DENSE_W8A8_GEMM + knob GLM53_DENSE_W8A8_ONLY (env.r16 `#switch`/`#knob`, w8a82; by the
# HEAD container's env; only read when GLM53_DENSE_W8A8=1): both containers must carry the same value. GEMM
# unset/custom -> each rank's install line names the custom GEMM ("custom CUTLASS SM120 GEMM, one call per M") and
# each served shape's self-test detail carries "== cutlass_scaled_mm" (bitwise == cutlass_scaled_mm, checked per shape at
# load); cutlass_mm -> the custom-GEMM lines must be absent. Never, in any state, "custom GEMM output differs"
# (a shape whose bitwise check failed falls back to cutlass_mm with a WARNING - a real mismatch is a BAD).
# ONLY (a comma list of fp8_w8a8.py's PROJ_NAMES, unset = every served projection): the two ranks must agree.
# r16z2 operator switch GLM53_MOE_E4M3_DOWN (env.r16 `#switch`, moe2; by the HEAD container's env; unset/empty/e4m3 =
# the original spec): both containers must carry the same value; f16 -> with GLM53_MOE_E4M3=1 each rank's install
# line reads "(down projection: fp16 (GLM53_MOE_E4M3_DOWN=f16))"; unset/e4m3 -> that line must be absent.
# r16z operator switch GLM53_MHC_SP2 (env.r16 `#switch`, mhcsp2; by the HEAD container's env; unset/empty/0 = the
# r16x SP result): both containers must carry the same value; 1 -> on each rank the bundle ran patch_mhc_sp2.py
# (model.py patched AGAIN on top of the SP tree, fingerprints extended). Requires GLM53_MHC_SP=1 on BOTH ranks
# (the pipelined SP extends the r16x SP patch of the same model.py): SP2=1 with SP!=1 is a MISS before every
# traffic check, like mhcsp D-A. mhcsp2 is the same PAIRED-collective feature as mhcsp (a rank with SP2 on and a
# peer without it = mismatched collective counts = hang/corruption): the [mhcsp2-pair] gate requires the sp2
# patched line in BOTH ranks' logs BEFORE every traffic check.
# r16z3 operator switch GLM53_MOE_FUSED16 (env.r16 `#switch`, fused16; by the HEAD container's env; unset/empty/0 =
# stock): both containers must carry the same value; 1 -> on each rank the bundle installed glm53_moe_fused16.py +
# the shared e4m3 extension, armed integrate.py (the block BEFORE glm53_moe_e4m3's), the plugin installed, the
# load-time self-test passed, the summary says "42/42 layers served, 0 fell back (none); self-test FAILED 0, raised
# 0" and the first eligible prefill call ran on a P16 schedule (vLLM's profile run is one); never a self-test
# FAILED / raised / layer-not-served / not-installed line. unset/empty -> each rank's bundle printed its skip line,
# 0 -> the patch's "nothing installed (stock)" line, and no install line anywhere. Neither container carrying the
# variable = a start.sh without the switch (r16z2 or older): stock, no install line allowed.
# r16z3 operator knobs GLM53_MOE_E4M3_LAYERS / GLM53_MOE_E4M3_DOWN_LAYERS (env.r16 `#knob`, moe3; only read when
# GLM53_MOE_E4M3=1): free-form comma lists of the model's MoE layer indices (the start.sh validated the syntax and
# the 3..44 range before any rank was stopped), so boot_checks checks EQUALITY between the ranks and, for
# GLM53_MOE_E4M3_LAYERS with GLM53_MOE_E4M3=1, that the e4m3 summary serves EXACTLY the listed layers:
# "K/K layers served, 0 fell back" + "; GLM53_MOE_E4M3_LAYERS: K indices selected, 42-K layers unselected
# (production's path)" - fewer served than listed (a half-applied state, e.g. one layer's self-test failed) is a
# MISS. Unset on both ranks: the "42/42 layers served" summary, and the summary suffix must be absent.
# r16z4 operator switches GLM53_MOE_E4M3_ACC / GLM53_MOE_E4M3_FOLD_SHARED / GLM53_MOE_E4M3_TOKGATHER (env.r16
# `#switch`, moe-opt; only read when GLM53_MOE_E4M3=1): both containers must carry the same value (else MISS),
# re-checked by opt-moe-rev's observability lines: FOLD=1 requires the "fold: first served call folded" INFO in
# BOTH ranks' logs BEFORE every traffic check (a rank whose served calls do not fold serves different MoE
# arithmetic than its peer) and never the not-folded / NOT-active WARNINGs; TOKGATHER on requires the summary's
# "token gather K/K served layers" per rank (the profile run qualifies the layers at boot).
# ACC unset/f32 -> the summary's "GLM53_MOE_E4M3_ACC=bf16 (bf16 accumulator)" suffix must be ABSENT; bf16 -> the
# suffix REQUIRED (the summary is checked as a whole below). FOLD unset/0 -> the "(routed sum accumulated into the
# shared experts' output)" suffix absent; 1 -> the suffix REQUIRED, and the module's own fallback WARNING
# "GLM53_MOE_E4M3_FOLD_SHARED=1 NOT active" is a MISS in every state (the fold silently not folding = the operator's
# dial lying). TOKGATHER unset/1 -> the summary's "token gather K/K served layers" REQUIRED (K = the served layer
# count; a layer that does not qualify falls back to the per-pair gather = a half-applied pair, MISS); 0 -> the
# "; GLM53_MOE_E4M3_TOKGATHER=0 (per-pair gather)" suffix REQUIRED and no "token gather" clause. Neither container
# carrying a knob = a start.sh without it (r16z3 or older): stock - no ACC/FOLD suffix and NO "token gather" clause
# (the r16z3 module predates it). The opt-moe dials change no collective: unlike mhcsp/sp2/fp8ag there is no pair
# gate, only the env equality + summary agreement checked above (a rank with ACC=bf16 and a peer with f32 would
# serve different MoE arithmetic per rank = the env-equality MISS).
# r16z4 operator switch GLM53_MLA_PREFILL_FUSED_INDEX (env.r16 `#switch`, mla-fused-index; only read when
# GLM53_MLA_PREFILL=1): both containers must carry the same value; unset/1 -> with --after-traffic the mla calls
# line must carry "fused index pass:" (the one-pass index write); 0 -> NO "fused index pass" in any state (the
# revert lever working). Neither container carrying it = a start.sh without the knob (r16z3 or older: the site
# module there has no fused pass) -> the line must be absent too.
# r16z4 operator switch GLM53_DENSE_W8A8_FP8AG (env.r16 `#switch`, w8a8-fp8ag; only read when GLM53_DENSE_W8A8=1):
# both containers must carry the same value; 1 with W8A8=1 is a PAIRED collective (per-layer CPU MIN vote over the
# TP group): the [fp8ag-pair] gate BEFORE every traffic check requires "FP8 all-gather installed" in BOTH ranks'
# logs (a rank whose W8A8 install was refused never wraps sp_all_gather and never joins the vote; its peer BLOCKS
# at the first sequence-parallel forward = boot hang), and the opt-dense-rev refusal ERROR ("this rank will not
# join") is a MISS in every state. FP8AG=1 without W8A8=1 is inert (nothing reads it) = info only. unset/0 -> no
# "FP8 all-gather installed" line anywhere.
# r16z4 operator knob GLM53_DENSE_W8A8_HILO + switch GLM53_DENSE_W8A8_HILO_SEL (env.r16 `#knob`/`#switch`,
# w8a8-hilo; only read when GLM53_DENSE_W8A8=1): both containers must carry the same values (equality only - the
# start.sh validated the "<group>.<proj>:<channels>" syntax and the 16..2048/16-multiple channel rule, and that
# draft.fc is not in the list, before any rank was stopped); with HILO set and --after-traffic each rank's log must
# carry the "hi+lo [..." channel-set-frozen line (with HILO_SEL=first the set freezes at the first real served
# call; deterministic afterwards); HILO unset -> no hi+lo line in any state.
# r16z5 operator knob/switches (opt-kdamhc-rev / opt-decodekit / opt-w8a8layers / opt-moe2-rev; by the HEAD
# container's env; unset = stock unless a deliberate default says otherwise):
# - GLM53_MLA_PREFILL_KV_ROWS (kdamhc; only read when GLM53_MLA_PREFILL=1): unset = the LIMITED write-back (the
#   deliberate default: only the rows an FA2 call can read; the install line says "first N rows"), 'all' = the r16
#   full write-back (the revert lever, "every row"). Equality between the ranks (the value is free-form; the
#   start.sh validated it).
# - GLM53_MHC_FUSED (kdamhc; DEFAULT OFF): 1 = the fused mHC post+prenorm GEMM; a PAIRED feature - the
#   [mhcfused-pair] gate BEFORE every traffic check requires the applied line on BOTH ranks (the mHC outputs feed
#   the same all-reduce; a one-rank install = different mHC arithmetic per rank); never a self-check/uninstall
#   line. _ROUND_A (0|1, default 0) and _CFG (decimal, default 9) dial the kernel, equality only.
# - GLM53_DENSE_W8A8_SKIP_LAYERS (w8a8layers; only read when GLM53_DENSE_W8A8=1): unset = no layer excluded; else
#   a comma list of '<layer>[:<name>]' applied after ONLY (equality only, names not printed; the start.sh
#   validated the syntax). With a value, each rank's install line must name the layer filter ("layer filter
#   GLM53_DENSE_W8A8_SKIP_LAYERS: excluded") and the load summary counts skip_layers_at_load.
# - GLM53_MOE_E4M3_MAINLOOP (moe2-rev; only read when GLM53_MOE_E4M3=1; DEFAULT OFF = SASS-identical to
#   opt-moe-rev): 1 = the lean fused mainloop - the install line's "; mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)"
#   and the summary suffix on BOTH ranks; unset/0 with the r16z5 module -> the suffix must be absent.
# r16z6 operator knobs (opt-decode; env.r16 `#knob`, site py_modules; by the HEAD container's env; unset/0 = stock):
# - GLM53_DEC_KDA_LAZY + _VERIFY + _VERIFY_EVERY (kdalazy; docs/DEC_KDA_LAZY.md): equality between the ranks; 1 ->
#   the site module glm53_kda_lazy.py installs ('glm53_kda_lazy on <model>: N KDA layers, ...') and, once the first
#   decode steps ran, the self-check line 'fast commit == production's kernel byte for byte' on BOTH ranks - the
#   self-check 'differ' line is a REFUSAL (the commit fell into repair mode: exact but ~5 ms/step SLOWER, roll
#   back), as are 'not wired on' / 'wiring failed' (production's stores silently stay). unset/0 -> the module's
#   'plugin loaded ... -> off, production kernels unchanged' line REQUIRED (the module is this kit's site/), and no
#   install line anywhere.
# - GLM53_DEC_VTRIM_STATS + _CAP + _FILE (vtrim; docs/OPT_DECODE.md 4): equality between the ranks; a positive
#   integer -> the log-only collector installs ('glm53_vtrim_stats on: one record per request and verify step');
#   unset/0 -> 'plugin loaded ... -> off' required, no collector line. Its .npy lives in the bind-mounted vLLM
#   cache dir; no token ids are recorded, so no secret can leak through this row's printed lines.

# r16z7 operator switches GLM53_KPOOL_TAIL_POSITIONS (0|2) / GLM53_MAMBA_ALIGN_SEED (0|1) (prefixhit-adv; env.r16
# `#switch`, bundle overlays; by the HEAD container's env): both containers must carry the same value (a one-rank tail fix
# = the TP halves' pooled indexer keys differ = silently degraded quality). =2 -> on each rank the overlay patched
# indexer.py IN PLACE (the FULL-graph half) and mamba_hybrid.py (positions) - the full-path lines the in-container overlay
# loop prints, e.g. "[glm53-kpool-tail-positions] /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/
# indexer.py: patched (in-place persistent tail slots)" - and the [kpooltail-pair] gate needs both lines on BOTH ranks
# before any traffic; =1 (FULL decode graphs never read it; the kit's start.sh refuses it) and other values -> MISS;
# unset -> the bundle's skip line, 0 -> the overlay's stock line, and no patched line anywhere. ALIGN_SEED=1 -> the
# add_request line. Neither container carrying a name = a start.sh without it (r16z6sv or older): stock.
set -uo pipefail
W="${W:-${WORKER_SSH:?set WORKER_SSH=<user>@<worker address> (or W=...)}}"; HEAD_C="${CONTAINER_HEAD:-glm53-exl3-head}"; WORK_C="${CONTAINER_WORKER:-glm53-exl3-worker}"
PORT="${PORT:-8888}"
mask() { sed -E 's#(key|token|secret|password|authorization|bearer)([=: "][^ ,]*)#\1=***#gi' | cut -c1-240; }
H=$(docker logs "$HEAD_C" 2>&1); WK=$(ssh -o BatchMode=yes "$W" "docker logs $WORK_C 2>&1")
[ -n "$H" ] && [ -n "$WK" ] || { echo "boot_checks: could not read the logs of both ranks"; exit 2; }
fail=0
MODE="${R16_ENV:-auto}"
case "$MODE" in on|off|auto) ;; *) echo "boot_checks: R16_ENV must be auto, on or off (got: $MODE)"; exit 2;; esac
# ---- the containers' env (names of env.r16 only, whose values are 0/1/all/dconv flags; the R1 incident: a variable
# that reached one rank only)
ENVF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.r16"
fmt='{{range .Config.Env}}{{println .}}{{end}}'
[ -f "$ENVF" ] || { echo "boot_checks: $ENVF missing"; exit 2; }
names=$(grep -E '^[A-Z0-9_]+=' "$ENVF" | cut -d= -f1 | paste -sd'|')
he=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E "^($names)=")
we=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E "^($names)=")
kitval() { grep -E "^$1=" "$ENVF" | tail -n 1 | cut -d= -f2-; }
envline() { printf '%s\n' "$1" | grep -E "^$2=." | tail -n 1; }     # envline <env lines> <name> -> NAME=value or "" (NAME= = unset)
# features (tools/env_r16.sh's names; apc is checked below from the head's own flags)
FEATS="kpoolring quickwins mla fp8roof moeglue hostloop smallops"
fnames() { case "$1" in kpoolring) echo GLM53_KPOOL_RING;; quickwins) echo GLM53_PREFILL_QUICKWINS;; mla) echo GLM53_MLA_PREFILL;;
  fp8roof) echo GLM53_DEC_FP8ROOF;; moeglue) echo GLM53_DEC_MOEGLUE_WARM;; hostloop) echo GLM53_DEC_HOSTLOOP;;
  smallops) echo "GLM53_DEC_SMALLOPS GLM53_DEC_SMALLOPS_KINDS";; esac; }
fstate() {   # fstate <feature> -> on | off | other, from the head container's env
  local n l on=1 off=1
  for n in $(fnames "$1"); do
    l=$(envline "$he" "$n")
    if [ -z "$l" ]; then on=0; else off=0; [ "${l#*=}" = "$(kitval "$n")" ] || on=0; fi
  done
  if [ $on = 1 ]; then echo on; elif [ $off = 1 ]; then echo off; else echo other; fi
}
declare -A ST
for f in $FEATS; do
  case "$MODE" in on|off) ST[$f]=$MODE;; auto) ST[$f]=$(fstate "$f");; esac
done
if [ "$MODE" = on ]; then
  while IFS= read -r line; do
    for rank in head worker; do
      if [ $rank = head ]; then e="$he"; else e="$we"; fi
      if printf '%s\n' "$e" | grep -qxF -- "$line"; then echo "ok   [$rank] container env $line"
      else echo "MISS [$rank] container env $line (got: $(printf '%s\n' "$e" | grep -E "^${line%%=*}=" || echo none))"; fail=1; fi
    done
  done < <(grep -E '^[A-Z0-9_]+=' "$ENVF")
elif [ "$MODE" = auto ]; then
  for n in $(echo "$names" | tr '|' ' '); do
    hl=$(envline "$he" "$n"); wl=$(envline "$we" "$n")
    if [ "$hl" = "$wl" ]; then echo "ok   [head=worker] container env ${hl:-$n unset}"
    else echo "MISS [worker] container env ${hl:-$n unset} (got: ${wl:-none})"; fail=1; fi
  done
  for f in $FEATS; do
    if [ "${ST[$f]}" = other ]; then
      echo "MISS [head] feature $f: container env $(for n in $(fnames "$f"); do echo -n "$(envline "$he" "$n" || true) "; done)is neither the kit's value(s) nor unset: decide by hand"; fail=1
    fi
  done
  echo "info features by the head container's env: $(for f in $FEATS; do echo -n "$f=${ST[$f]} "; done)"
fi
# ---- r16j operator switch GLM53_REJECTION_METHOD (env.r16 #switch): head == worker, value standard|block (else MISS)
SWREJ=""; REJ=standard; REJV=""; PRESW=""
if grep -qE '^#switch GLM53_REJECTION_METHOD ' "$ENVF"; then
  SWREJ=1
  rejh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_REJECTION_METHOD=' | tail -n 1)
  rejw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_REJECTION_METHOD=' | tail -n 1)
  REJV=$(printf '%s' "$rejh_l" | cut -d= -f2-); rejw=$(printf '%s' "$rejw_l" | cut -d= -f2-)
  show() { case "$1" in "") echo "<unset> (standard)";; standard|block) echo "$1";; *) echo "<not standard|block; not printed>";; esac; }
  if [ -z "$rejh_l" ] && [ -z "$rejw_l" ]; then
    PRESW=1
    echo "info neither container carries GLM53_REJECTION_METHOD: started by a start.sh without the switch (r16i or older) -> standard; block-keys lines must be absent"
  elif [ -z "$rejh_l" ] || [ -z "$rejw_l" ]; then
    echo "MISS [$([ -z "$rejh_l" ] && echo head || echo worker)] container env has no GLM53_REJECTION_METHOD line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$REJV" = "$rejw" ]; then echo "ok   [head=worker] container env GLM53_REJECTION_METHOD $(show "$REJV")"
  else echo "MISS [worker] container env GLM53_REJECTION_METHOD $(show "$REJV") (got: $(show "$rejw"))"; fail=1; fi
  case "$REJV" in
    ""|standard) REJ=standard;;
    block) REJ=block;;
    *) REJ=other; echo "MISS [head] container env GLM53_REJECTION_METHOD is neither standard nor block: decide by hand"; fail=1;;
  esac
  launch=$(printf '%s\n' "$H" | grep -F '[glm53-exl3-head] launching: vllm serve' | tail -n 1)
  if [ "$REJ" != other ]; then
    case "$launch" in
      *"\"rejection_sample_method\":\"$REJ\""*) echo "ok   [head] launching: vllm serve ... --speculative-config {... \"rejection_sample_method\":\"$REJ\" ...}";;
      *) echo "MISS [head] the head's last 'launching: vllm serve' line has no \"rejection_sample_method\":\"$REJ\"$([ -n "$launch" ] || echo ' (no launching line in the head log)')"; fail=1;;
    esac
  fi
  echo "info rejection sampling by the head container's env: $REJ"
fi
# ---- r16k operator switch GLM53_KDA_STRIDED_QKV (env.r16 #switch, kdaqkv): head == worker, value empty/0/1 (else MISS)
SWKDA=""; KDA=off; KDAV=""; PREKDA=""
if grep -qE '^#switch GLM53_KDA_STRIDED_QKV ' "$ENVF"; then
  SWKDA=1
  kdah_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_KDA_STRIDED_QKV=' | tail -n 1)
  kdaw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_KDA_STRIDED_QKV=' | tail -n 1)
  KDAV=$(printf '%s' "$kdah_l" | cut -d= -f2-); kdaw=$(printf '%s' "$kdaw_l" | cut -d= -f2-)
  kshow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$kdah_l" ] && [ -z "$kdaw_l" ]; then
    PREKDA=1
    echo "info neither container carries GLM53_KDA_STRIDED_QKV: started by a start.sh without the knob (r16j or older) -> stock; kdaqkv patched lines must be absent"
  elif [ -z "$kdah_l" ] || [ -z "$kdaw_l" ]; then
    echo "MISS [$([ -z "$kdah_l" ] && echo head || echo worker)] container env has no GLM53_KDA_STRIDED_QKV line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$KDAV" = "$kdaw" ]; then echo "ok   [head=worker] container env GLM53_KDA_STRIDED_QKV $(kshow "$KDAV")"
  else echo "MISS [worker] container env GLM53_KDA_STRIDED_QKV $(kshow "$KDAV") (got: $(kshow "$kdaw"))"; fail=1; fi
  case "$KDAV" in
    ""|0) KDA=off;;
    1) KDA=on;;
    *) KDA=other; echo "MISS [head] container env GLM53_KDA_STRIDED_QKV is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info kdaqkv (KDA strided decode inputs) by the head container's env: $KDA"
fi
# ---- r16l operator switch GLM53_KPOOL_DROP_LOWEST (env.r16 #switch, kpooldown): head == worker, value empty/0/1 (else MISS)
SWDROP=""; DROP=off; DROPV=""; PREDROP=""
if grep -qE '^#switch GLM53_KPOOL_DROP_LOWEST ' "$ENVF"; then
  SWDROP=1
  drh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_KPOOL_DROP_LOWEST=' | tail -n 1)
  drw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_KPOOL_DROP_LOWEST=' | tail -n 1)
  DROPV=$(printf '%s' "$drh_l" | cut -d= -f2-); drw=$(printf '%s' "$drw_l" | cut -d= -f2-)
  dshow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$drh_l" ] && [ -z "$drw_l" ]; then
    PREDROP=1
    echo "info neither container carries GLM53_KPOOL_DROP_LOWEST: started by a start.sh without the knob (r16k or older) -> stock; kpooldown patched lines must be absent"
  elif [ -z "$drh_l" ] || [ -z "$drw_l" ]; then
    echo "MISS [$([ -z "$drh_l" ] && echo head || echo worker)] container env has no GLM53_KPOOL_DROP_LOWEST line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$DROPV" = "$drw" ]; then echo "ok   [head=worker] container env GLM53_KPOOL_DROP_LOWEST $(dshow "$DROPV")"
  else echo "MISS [worker] container env GLM53_KPOOL_DROP_LOWEST $(dshow "$DROPV") (got: $(dshow "$drw"))"; fail=1; fi
  case "$DROPV" in
    ""|0) DROP=off;;
    1) DROP=on;;
    *) DROP=other; echo "MISS [head] container env GLM53_KPOOL_DROP_LOWEST is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info kpooldown (drop the lowest-scored kpool) by the head container's env: $DROP"
fi
# ---- r16z2x operator switch GLM53_MLA_EXACT_LENS (env.r16 #switch, exactlens; docs/MLA_EXACT_LENS.md): head == worker,
# value empty/0/1 (else MISS)
SWXL=""; XL=off; XLV=""; PREXL=""
if grep -qE '^#switch GLM53_MLA_EXACT_LENS ' "$ENVF"; then
  SWXL=1
  xlh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MLA_EXACT_LENS=' | tail -n 1)
  xlw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MLA_EXACT_LENS=' | tail -n 1)
  XLV=$(printf '%s' "$xlh_l" | cut -d= -f2-); xlw=$(printf '%s' "$xlw_l" | cut -d= -f2-)
  xshow() { case "$1" in "") echo "<unset> (production's FA2 plan)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$xlh_l" ] && [ -z "$xlw_l" ]; then
    PREXL=1
    echo "info neither container carries GLM53_MLA_EXACT_LENS: started by a start.sh without the knob (r16z2 or older) -> production's plan; exactlens lines must be absent"
  elif [ -z "$xlh_l" ] || [ -z "$xlw_l" ]; then
    echo "MISS [$([ -z "$xlh_l" ] && echo head || echo worker)] container env has no GLM53_MLA_EXACT_LENS line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$XLV" = "$xlw" ]; then echo "ok   [head=worker] container env GLM53_MLA_EXACT_LENS $(xshow "$XLV")"
  else echo "MISS [worker] container env GLM53_MLA_EXACT_LENS $(xshow "$XLV") (got: $(xshow "$xlw"))"; fail=1; fi
  case "$XLV" in
    ""|0) XL=off;;
    1) XL=on;;
    *) XL=other; echo "MISS [head] container env GLM53_MLA_EXACT_LENS is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info exactlens (FA2 sparse-MLA plan with the exact selected-key counts) by the head container's env: $XL"
fi
# ---- r16n operator switch GLM53_KDA_FLASHKDA (env.r16 `#switch`, flashkda; r16n): head == worker, value empty/0/1 (else MISS)
SWFK=""; FK=off; FKV=""; PREFK=""
if grep -qE '^#switch GLM53_KDA_FLASHKDA ' "$ENVF"; then
  SWFK=1
  fkh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_KDA_FLASHKDA=' | tail -n 1)
  fkw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_KDA_FLASHKDA=' | tail -n 1)
  FKV=$(printf '%s' "$fkh_l" | cut -d= -f2-); fkw=$(printf '%s' "$fkw_l" | cut -d= -f2-)
  fshow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$fkh_l" ] && [ -z "$fkw_l" ]; then
    PREFK=1
    echo "info neither container carries GLM53_KDA_FLASHKDA: started by a start.sh without the knob (r16l or older) -> stock; flashkda patched lines must be absent"
  elif [ -z "$fkh_l" ] || [ -z "$fkw_l" ]; then
    echo "MISS [$([ -z "$fkh_l" ] && echo head || echo worker)] container env has no GLM53_KDA_FLASHKDA line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$FKV" = "$fkw" ]; then echo "ok   [head=worker] container env GLM53_KDA_FLASHKDA $(fshow "$FKV")"
  else echo "MISS [worker] container env GLM53_KDA_FLASHKDA $(fshow "$FKV") (got: $(fshow "$fkw"))"; fail=1; fi
  case "$FKV" in
    ""|0) FK=off;;
    1) FK=on;;
    *) FK=other; echo "MISS [head] container env GLM53_KDA_FLASHKDA is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info flashkda (FlashKDA 17a037d KDA chunked prefill) by the head container's env: $FK"
fi
# ---- r16y operator switch GLM53_DENSE_W8A8 (env.r16 #switch, w8a8): head == worker, value empty/0/1 (else MISS)
SWW8A=""; W8A=off; W8AV=""; PREW8A=""
if grep -qE '^#switch GLM53_DENSE_W8A8 ' "$ENVF"; then
  SWW8A=1
  wah_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DENSE_W8A8=' | tail -n 1)
  waw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_DENSE_W8A8=' | tail -n 1)
  W8AV=$(printf '%s' "$wah_l" | cut -d= -f2-); waw=$(printf '%s' "$waw_l" | cut -d= -f2-)
  washow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$wah_l" ] && [ -z "$waw_l" ]; then
    PREW8A=1
    echo "info neither container carries GLM53_DENSE_W8A8: started by a start.sh without the switch (r16n or older) -> stock; w8a8 lines must be absent"
  elif [ -z "$wah_l" ] || [ -z "$waw_l" ]; then
    echo "MISS [$([ -z "$wah_l" ] && echo head || echo worker)] container env has no GLM53_DENSE_W8A8 line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$W8AV" = "$waw" ]; then echo "ok   [head=worker] container env GLM53_DENSE_W8A8 $(washow "$W8AV")"
  else echo "MISS [worker] container env GLM53_DENSE_W8A8 $(washow "$W8AV") (got: $(washow "$waw"))"; fail=1; fi
  case "$W8AV" in
    ""|0) W8A=off;;
    1) W8A=on;;
    *) W8A=other; echo "MISS [head] container env GLM53_DENSE_W8A8 is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info w8a8 (W8A8 prefill dense GEMMs) by the head container's env: $W8A"
fi
# ---- r16msp operator switch GLM53_MHC_SP (env.r16 #switch, mhcsp): head == worker, value empty/0/1 (else MISS)
SWSP=""; SP=off; SPV=""; PRESP=""
if grep -qE '^#switch GLM53_MHC_SP ' "$ENVF"; then
  SWSP=1
  drh_s=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MHC_SP=' | tail -n 1)
  drw_s=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MHC_SP=' | tail -n 1)
  SPV=$(printf '%s' "$drh_s" | cut -d= -f2-); drws=$(printf '%s' "$drw_s" | cut -d= -f2-)
  dshow_sp() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$drh_s" ] && [ -z "$drw_s" ]; then
    PRESP=1
    echo "info neither container carries GLM53_MHC_SP: started by a start.sh without the knob (r16l or older) -> stock; mhcsp patched lines must be absent"
  elif [ -z "$drh_s" ] || [ -z "$drw_s" ]; then
    echo "MISS [$([ -z "$drh_s" ] && echo head || echo worker)] container env has no GLM53_MHC_SP line while the other rank has one (two different start.sh?) (mhcsp D-A: one rank on = silent corruption, ABORT)"; fail=1
  elif [ "$SPV" = "$drws" ]; then echo "ok   [head=worker] container env GLM53_MHC_SP $(dshow_sp "$SPV")"
  else echo "MISS [worker] container env GLM53_MHC_SP $(dshow_sp "$SPV") (got: $(dshow_sp "$drws")) (mhcsp D-A: one rank on = silent RS+AG corruption, ABORT)"; fail=1; fi
  case "$SPV" in
    ""|0) SP=off;;
    1) SP=on;;
    *) SP=other; echo "MISS [head] container env GLM53_MHC_SP is neither empty, 0 nor 1: decide by hand (mhcsp D-A: the ranks must agree, ABORT)"; fail=1;;
  esac
  echo "info mhcsp (sequence-parallel mHC prefill) by the head container's env: $SP"
fi
# ---- r16e4 operator switch GLM53_MOE_E4M3 (env.r16 #switch, moee4m3): head == worker, value empty/0/1 (else MISS)
SWE4=""; E4=off; E4V=""; PREE4=""
if grep -qE '^#switch GLM53_MOE_E4M3 ' "$ENVF"; then
  SWE4=1
  e4h_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MOE_E4M3=' | tail -n 1)
  e4w_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MOE_E4M3=' | tail -n 1)
  E4V=$(printf '%s' "$e4h_l" | cut -d= -f2-); e4w=$(printf '%s' "$e4w_l" | cut -d= -f2-)
  eshow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$e4h_l" ] && [ -z "$e4w_l" ]; then
    PREE4=1
    echo "info neither container carries GLM53_MOE_E4M3: started by a start.sh without the knob (r16l or older) -> stock; moee4m3 install lines must be absent"
  elif [ -z "$e4h_l" ] || [ -z "$e4w_l" ]; then
    echo "MISS [$([ -z "$e4h_l" ] && echo head || echo worker)] container env has no GLM53_MOE_E4M3 line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$E4V" = "$e4w" ]; then echo "ok   [head=worker] container env GLM53_MOE_E4M3 $(eshow "$E4V")"
  else echo "MISS [worker] container env GLM53_MOE_E4M3 $(eshow "$E4V") (got: $(eshow "$e4w"))"; fail=1; fi
  case "$E4V" in
    ""|0) E4=off;;
    1) E4=on;;
    *) E4=other; echo "MISS [head] container env GLM53_MOE_E4M3 is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info moee4m3 (e4m3 routed-MoE prefill) by the head container's env: $E4"
fi
# ---- r16z operator switch GLM53_KDA_FLASHKDA_V (fkda-v; WHICH FlashKDA build, only read with GLM53_KDA_FLASHKDA=1):
# head == worker, value empty/1/2/3 (else MISS)
SWFV=""; FVV=""
if grep -qE '^#switch GLM53_KDA_FLASHKDA_V ' "$ENVF"; then
  SWFV=1
  fvh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_KDA_FLASHKDA_V=' | tail -n 1)
  fvw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_KDA_FLASHKDA_V=' | tail -n 1)
  FVV=$(printf '%s' "$fvh_l" | cut -d= -f2-); fvw=$(printf '%s' "$fvw_l" | cut -d= -f2-)
  fvshow() { case "$1" in "") echo "<unset> (the shipped r16x build)";; 1) echo "1 (the shipped r16x build)";; 2) echo "2 (the fkda2 precision build)";; 3) echo "3 (the fkda3 build)";; *) echo "<not unset|1|2|3; not printed>";; esac; }
  if [ -z "$fvh_l" ] && [ -z "$fvw_l" ]; then
    echo "info neither container carries GLM53_KDA_FLASHKDA_V: started by a start.sh without the switch (r16x or older) -> the shipped build; fkda2/fkda3 staging lines must be absent"
  elif [ -z "$fvh_l" ] || [ -z "$fvw_l" ]; then
    echo "MISS [$([ -z "$fvh_l" ] && echo head || echo worker)] container env has no GLM53_KDA_FLASHKDA_V line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$FVV" = "$fvw" ]; then echo "ok   [head=worker] container env GLM53_KDA_FLASHKDA_V $(fvshow "$FVV")"
  else echo "MISS [worker] container env GLM53_KDA_FLASHKDA_V $(fvshow "$FVV") (got: $(fvshow "$fvw"))"; fail=1; fi
  case "$FVV" in
    ""|1|2|3) ;;
    *) echo "MISS [head] container env GLM53_KDA_FLASHKDA_V is neither unset, 1, 2 nor 3: decide by hand"; fail=1;;
  esac
  echo "info fkda-v (WHICH FlashKDA build; only read with GLM53_KDA_FLASHKDA=1) by the head container's env: $(case "$FVV" in ""|1) echo fkda;; 2) echo fkda2;; 3) echo fkda3;; *) echo other;; esac)"
fi
# ---- r16z operator switch GLM53_MHC_SP2 (mhcsp2; pipelined SP, needs GLM53_MHC_SP=1): head == worker, value empty/0/1 (else MISS)
SWSP2=""; SP2=off; SP2V=""; PRESP2=""
if grep -qE '^#switch GLM53_MHC_SP2 ' "$ENVF"; then
  SWSP2=1
  s2h_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MHC_SP2=' | tail -n 1)
  s2w_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MHC_SP2=' | tail -n 1)
  SP2V=$(printf '%s' "$s2h_l" | cut -d= -f2-); s2w=$(printf '%s' "$s2w_l" | cut -d= -f2-)
  dshow_sp2() { case "$1" in "") echo "<unset> (the r16x SP result)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$s2h_l" ] && [ -z "$s2w_l" ]; then
    PRESP2=1
    echo "info neither container carries GLM53_MHC_SP2: started by a start.sh without the switch (r16x or older) -> the r16x SP result; mhcsp2 patched lines must be absent"
  elif [ -z "$s2h_l" ] || [ -z "$s2w_l" ]; then
    echo "MISS [$([ -z "$s2h_l" ] && echo head || echo worker)] container env has no GLM53_MHC_SP2 line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$SP2V" = "$s2w" ]; then echo "ok   [head=worker] container env GLM53_MHC_SP2 $(dshow_sp2 "$SP2V")"
  else echo "MISS [worker] container env GLM53_MHC_SP2 $(dshow_sp2 "$SP2V") (got: $(dshow_sp2 "$s2w")) (mhcsp2: one rank on = mismatched collective counts, ABORT)"; fail=1; fi
  case "$SP2V" in
    ""|0) SP2=off;;
    1) SP2=on;;
    *) SP2=other; echo "MISS [head] container env GLM53_MHC_SP2 is neither empty, 0 nor 1: decide by hand (mhcsp2: the ranks must agree, ABORT)"; fail=1;;
  esac
  if [ "$SP2" = on ] && [ "$SP" != on ]; then
    echo "MISS [mhcsp2] GLM53_MHC_SP2=1 needs GLM53_MHC_SP=1 (head env: $(dshow_sp "$SPV")): the pipelined SP extends the r16x SP patch of the same model.py; ABORT before traffic"; fail=1
  fi
  echo "info mhcsp2 (pipelined SP prefill) by the head container's env: $SP2"
fi
# ---- r16z2 operator switch GLM53_DENSE_W8A8_GEMM + knob GLM53_DENSE_W8A8_ONLY (w8a82; only read with
# GLM53_DENSE_W8A8=1): head == worker, GEMM value empty/custom/cutlass_mm (else MISS), ONLY equality (its value is a
# free-form comma list: equality only, the start.sh validated the names before any rank was stopped)
SWG=""; WGV=""; WGV2=""; WOV=""; PREW82=""
if grep -qE '^#switch GLM53_DENSE_W8A8_GEMM ' "$ENVF"; then
  SWG=1
  wgh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DENSE_W8A8_GEMM=' | tail -n 1)
  wgw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_DENSE_W8A8_GEMM=' | tail -n 1)
  wo_h=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DENSE_W8A8_ONLY=' | tail -n 1)
  wo_w=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_DENSE_W8A8_ONLY=' | tail -n 1)
  WGV=$(printf '%s' "$wgh_l" | cut -d= -f2-); wgw=$(printf '%s' "$wgw_l" | cut -d= -f2-)
  WOV=$(printf '%s' "$wo_h" | cut -d= -f2-); wow=$(printf '%s' "$wo_w" | cut -d= -f2-)
  wgshow() { case "$1" in "") echo "<unset> (custom, the kit's CUTLASS GEMM)";; custom) echo "custom";; cutlass_mm) echo "cutlass_mm (the image's cutlass_scaled_mm)";; *) echo "<not unset|custom|cutlass_mm; not printed>";; esac; }
  if [ -z "$wgh_l" ] && [ -z "$wgw_l" ]; then
    PREW82=1
    echo "info neither container carries GLM53_DENSE_W8A8_GEMM: started by a start.sh without the switch (r16z or older) -> custom; w8a82 lines must be absent"
  elif [ -z "$wgh_l" ] || [ -z "$wgw_l" ]; then
    echo "MISS [$([ -z "$wgh_l" ] && echo head || echo worker)] container env has no GLM53_DENSE_W8A8_GEMM line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$WGV" = "$wgw" ]; then echo "ok   [head=worker] container env GLM53_DENSE_W8A8_GEMM $(wgshow "$WGV")"
  else echo "MISS [worker] container env GLM53_DENSE_W8A8_GEMM $(wgshow "$WGV") (got: $(wgshow "$wgw"))"; fail=1; fi
  case "$WGV" in
    ""|custom|cutlass_mm) ;;
    *) echo "MISS [head] container env GLM53_DENSE_W8A8_GEMM is neither unset, custom nor cutlass_mm: decide by hand"; fail=1;;
  esac
  if { [ -z "$wo_h" ] && [ -z "$wo_w" ]; } || { [ -n "$wo_h" ] && [ -n "$wo_w" ] && [ -z "$WOV" ] && [ -z "$wow" ]; }; then
    echo "ok   [head=worker] container env GLM53_DENSE_W8A8_ONLY <unset> (every served projection)"
  elif [ -z "$wo_h" ] || [ -z "$wo_w" ] || [ "$WOV" != "$wow" ]; then
    echo "MISS [worker] container env GLM53_DENSE_W8A8_ONLY differs between the ranks (a projection filter on one rank only = a half-served pair; values not printed)"; fail=1
  else
    echo "ok   [head=worker] container env GLM53_DENSE_W8A8_ONLY set (${#WOV} chars; names not printed)"
  fi
  echo "info w8a82 (W8A8 GEMM backend + projection filter) by the head container's env: gemm=$(case "$WGV" in ""|custom) echo custom;; cutlass_mm) echo cutlass_mm;; *) echo other;; esac), only=$([ -n "$WOV" ] && echo set || echo all)"
fi
# ---- r16z2 operator switch GLM53_MOE_E4M3_DOWN (moe2; only read with GLM53_MOE_E4M3=1): head == worker, value
# empty/e4m3/f16 (else MISS)
SWED=""; EDV=""; PREED=""
if grep -qE '^#switch GLM53_MOE_E4M3_DOWN ' "$ENVF"; then
  SWED=1
  edh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MOE_E4M3_DOWN=' | tail -n 1)
  edw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MOE_E4M3_DOWN=' | tail -n 1)
  EDV=$(printf '%s' "$edh_l" | cut -d= -f2-); edw=$(printf '%s' "$edw_l" | cut -d= -f2-)
  edshow() { case "$1" in "") echo "<unset> (e4m3, the original spec)";; e4m3) echo "e4m3";; f16) echo "f16 (the fused variant 16)";; *) echo "<not unset|e4m3|f16; not printed>";; esac; }
  if [ -z "$edh_l" ] && [ -z "$edw_l" ]; then
    PREED=1
    echo "info neither container carries GLM53_MOE_E4M3_DOWN: started by a start.sh without the switch (r16z or older) -> e4m3; the fp16 down lines must be absent"
  elif [ -z "$edh_l" ] || [ -z "$edw_l" ]; then
    echo "MISS [$([ -z "$edh_l" ] && echo head || echo worker)] container env has no GLM53_MOE_E4M3_DOWN line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$EDV" = "$edw" ]; then echo "ok   [head=worker] container env GLM53_MOE_E4M3_DOWN $(edshow "$EDV")"
  else echo "MISS [worker] container env GLM53_MOE_E4M3_DOWN $(edshow "$EDV") (got: $(edshow "$edw"))"; fail=1; fi
  case "$EDV" in
    ""|e4m3|f16) ;;
    *) echo "MISS [head] container env GLM53_MOE_E4M3_DOWN is neither unset, e4m3 nor f16: decide by hand"; fail=1;;
  esac
  echo "info moe2 (e4m3 down width) by the head container's env: $(case "$EDV" in ""|e4m3) echo e4m3;; f16) echo f16;; *) echo other;; esac)"
fi
# ---- r16z3 operator switch GLM53_MOE_FUSED16 (moe3): head == worker, value empty/0/1 (else MISS)
SWFF=""; FF=off; FFV=""; PREFF=""
if grep -qE '^#switch GLM53_MOE_FUSED16 ' "$ENVF"; then
  SWFF=1
  ffh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MOE_FUSED16=' | tail -n 1)
  ffw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MOE_FUSED16=' | tail -n 1)
  FFV=$(printf '%s' "$ffh_l" | cut -d= -f2-); ffw=$(printf '%s' "$ffw_l" | cut -d= -f2-)
  ffshow() { case "$1" in "") echo "<unset> (stock)";; 0|1) echo "$1";; *) echo "<not 0|1; not printed>";; esac; }
  if [ -z "$ffh_l" ] && [ -z "$ffw_l" ]; then
    PREFF=1
    echo "info neither container carries GLM53_MOE_FUSED16: started by a start.sh without the switch (r16z2 or older) -> stock; fused16 install lines must be absent"
  elif [ -z "$ffh_l" ] || [ -z "$ffw_l" ]; then
    echo "MISS [$([ -z "$ffh_l" ] && echo head || echo worker)] container env has no GLM53_MOE_FUSED16 line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$FFV" = "$ffw" ]; then echo "ok   [head=worker] container env GLM53_MOE_FUSED16 $(ffshow "$FFV")"
  else echo "MISS [worker] container env GLM53_MOE_FUSED16 $(ffshow "$FFV") (got: $(ffshow "$ffw"))"; fail=1; fi
  case "$FFV" in
    ""|0) FF=off;;
    1) FF=on;;
    *) FF=other; echo "MISS [head] container env GLM53_MOE_FUSED16 is neither empty, 0 nor 1: decide by hand"; fail=1;;
  esac
  echo "info fused16 (P16 routed-MoE prefill, production arithmetic) by the head container's env: $FF"
fi
# ---- r16z3 operator knobs GLM53_MOE_E4M3_LAYERS / GLM53_MOE_E4M3_DOWN_LAYERS (moe3; only read with GLM53_MOE_E4M3=1):
# free-form comma lists of MoE layer indices -> equality between the ranks (the start.sh validated the syntax/range)
SWLS=""; LSV=""; SWLD=""; LDV=""
if grep -qE '^#knob GLM53_MOE_E4M3_LAYERS ' "$ENVF"; then
  SWLS=1
  lsh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MOE_E4M3_LAYERS=' | tail -n 1)
  lsw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MOE_E4M3_LAYERS=' | tail -n 1)
  LSV=$(printf '%s' "$lsh_l" | cut -d= -f2-); lsw=$(printf '%s' "$lsw_l" | cut -d= -f2-)
  if { [ -z "$lsh_l" ] && [ -z "$lsw_l" ]; } || { [ -n "$lsh_l" ] && [ -n "$lsw_l" ] && [ -z "$LSV" ] && [ -z "$lsw" ]; }; then
    echo "ok   [head=worker] container env GLM53_MOE_E4M3_LAYERS <unset> (every MoE layer on e4m3)"
  elif [ -z "$lsh_l" ] || [ -z "$lsw_l" ] || [ "$LSV" != "$lsw" ]; then
    echo "MISS [worker] container env GLM53_MOE_E4M3_LAYERS differs between the ranks (a layer list on one rank only = a half-applied pair; values not printed)"; fail=1
  else
    echo "ok   [head=worker] container env GLM53_MOE_E4M3_LAYERS set (${#LSV} chars; names not printed)"
  fi
  echo "info moe3 e4m3 layer selection by the head container's env: $([ -n "$LSV" ] && echo set || echo all)"
fi
if grep -qE '^#knob GLM53_MOE_E4M3_DOWN_LAYERS ' "$ENVF"; then
  SWLD=1
  ldh_l=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_MOE_E4M3_DOWN_LAYERS=' | tail -n 1)
  ldw_l=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_MOE_E4M3_DOWN_LAYERS=' | tail -n 1)
  LDV=$(printf '%s' "$ldh_l" | cut -d= -f2-); ldw=$(printf '%s' "$ldw_l" | cut -d= -f2-)
  if { [ -z "$ldh_l" ] && [ -z "$ldw_l" ]; } || { [ -n "$ldh_l" ] && [ -n "$ldw_l" ] && [ -z "$LDV" ] && [ -z "$ldw" ]; }; then
    echo "ok   [head=worker] container env GLM53_MOE_E4M3_DOWN_LAYERS <unset> (GLM53_MOE_E4M3_DOWN decides)"
  elif [ -z "$ldh_l" ] || [ -z "$ldw_l" ] || [ "$LDV" != "$ldw" ]; then
    echo "MISS [worker] container env GLM53_MOE_E4M3_DOWN_LAYERS differs between the ranks (a layer list on one rank only = a half-applied pair; values not printed)"; fail=1
  else
    echo "ok   [head=worker] container env GLM53_MOE_E4M3_DOWN_LAYERS set (${#LDV} chars; names not printed)"
  fi
  echo "info moe3 f16-down layer selection by the head container's env: $([ -n "$LDV" ] && echo set || echo GLM53_MOE_E4M3_DOWN)"
fi
# ---- r16z4 operator switches GLM53_MOE_E4M3_ACC / _FOLD_SHARED / _TOKGATHER (moe-opt; only read with
# GLM53_MOE_E4M3=1) and GLM53_MLA_PREFILL_FUSED_INDEX (mla-fused-index; only read with GLM53_MLA_PREFILL=1), and
# GLM53_DENSE_W8A8_FP8AG + knob GLM53_DENSE_W8A8_HILO + switch _HILO_SEL (w8a8 fp8ag/hilo; only read with
# GLM53_DENSE_W8A8=1): head == worker, enum values (else MISS); HILO equality only (free-form list).
E4ACCV=""; E4FOLDV=""; E4TGV=""; MLAFIV=""; FP8AGV=""; HILOV=""; HISELV=""
PRE_ACC=""; PRE_FOLD=""; PRE_TG=""; PRE_MLAFI=""; PRE_FP8AG=""; PREZ4=""
EQV=""; EQPRE=""; EQERR=""
eqswitch() {   # eqswitch <NAME> <allowed shell-case pattern, e.g. ''|0|1>: sets EQV (the head's value; "" = unset),
  # EQPRE=1 (neither container carries the name = a start.sh without the knob) or EQERR=1 (present on one rank
  # only / differs between the ranks = MISS), and prints the row; a value outside <allowed> is NOT printed
  EQV=""; EQPRE=""; EQERR=""
  local n="$1" h w
  eqshow() { case "$1" in "") echo "<unset>";; *) case " $3 " in *" $1 "*) echo "$1";; *) echo "<not $4; not printed>";; esac;; esac; }
  h=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E "^$n=" | tail -n 1)
  w=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E "^$n=" | tail -n 1)
  if [ -z "$h" ] && [ -z "$w" ]; then
    EQPRE=1
    echo "info neither container carries $n: started by a start.sh without the knob (r16z3 or older) -> stock"
  elif [ -z "$h" ] || [ -z "$w" ]; then
    EQERR=1
    echo "MISS [$([ -z "$h" ] && echo head || echo worker)] container env has no $n line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "${h#*=}" = "${w#*=}" ]; then
    EQV="${h#*=}"
    echo "ok   [head=worker] container env $n $(eqshow "$EQV" "$2" "$3")"
  else
    EQERR=1
    echo "MISS [worker] container env $n $(eqshow "${h#*=}" "$2" "$3") (got: $(eqshow "${w#*=}" "$2" "$3"))"; fail=1
  fi
}
eqswitch GLM53_MOE_E4M3_ACC "f32 bf16" 'unset, f32 or bf16'; E4ACCV="$EQV"; PRE_ACC="$EQPRE"
case "$E4ACCV" in ""|f32|bf16) ;; *) echo "MISS [head] container env GLM53_MOE_E4M3_ACC is neither unset, f32 nor bf16: decide by hand"; fail=1;; esac
echo "info moe-opt accumulator by the head container's env: $(case "$E4ACCV" in ""|f32) echo f32;; bf16) echo bf16;; *) echo other;; esac)"
eqswitch GLM53_MOE_E4M3_FOLD_SHARED "0 1" 'unset, 0 or 1'; E4FOLDV="$EQV"; PRE_FOLD="$EQPRE"
case "$E4FOLDV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MOE_E4M3_FOLD_SHARED is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
echo "info moe-opt fold by the head container's env: $(case "$E4FOLDV" in 1) echo on;; *) echo off;; esac)"
eqswitch GLM53_MOE_E4M3_TOKGATHER "0 1" 'unset, 0 or 1'; E4TGV="$EQV"; PRE_TG="$EQPRE"
case "$E4TGV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MOE_E4M3_TOKGATHER is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
echo "info moe-opt token gather by the head container's env: $(case "$E4TGV" in 0) echo "off (per-pair gather)";; *) echo "on (the designed default)";; esac)"
eqswitch GLM53_MLA_PREFILL_FUSED_INDEX "0 1" 'unset, 0 or 1'; MLAFIV="$EQV"; PRE_MLAFI="$EQPRE"
case "$MLAFIV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MLA_PREFILL_FUSED_INDEX is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
echo "info mla-fused-index by the head container's env: $(case "$MLAFIV" in 0) echo "off (production's index chain)";; *) echo "on (the deliberate default)";; esac)"
eqswitch GLM53_DENSE_W8A8_FP8AG "0 1" 'unset, 0 or 1'; FP8AGV="$EQV"; PRE_FP8AG="$EQPRE"
case "$FP8AGV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_DENSE_W8A8_FP8AG is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
echo "info w8a8-fp8ag by the head container's env: $(case "$FP8AGV" in 1) echo on;; *) echo "off (the bf16 gather)";; esac)$( [ "$FP8AGV" = 1 ] && [ "$W8A" != on ] && echo " (INERT: GLM53_DENSE_W8A8 is not 1, nothing reads the knob)")"
h=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DENSE_W8A8_HILO=' | tail -n 1)
w=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_DENSE_W8A8_HILO=' | tail -n 1)
if [ -z "$h" ] && [ -z "$w" ]; then
  echo "info neither container carries GLM53_DENSE_W8A8_HILO: started by a start.sh without the knob (r16z3 or older) -> off"
elif [ -z "$h" ] || [ -z "$w" ]; then
  echo "MISS [$([ -z "$h" ] && echo head || echo worker)] container env has no GLM53_DENSE_W8A8_HILO line while the other rank has one (two different start.sh?)"; fail=1
elif [ "${h#*=}" = "${w#*=}" ]; then
  echo "ok   [head=worker] container env GLM53_DENSE_W8A8_HILO $([ -n "${h#*=}" ] && echo "set (${#h} chars incl. the name; the list is not printed)" || echo "<unset> (off)")"
else
  echo "MISS [worker] container env GLM53_DENSE_W8A8_HILO differs between the ranks (a hi+lo channel set on one rank only = a half-served pair; values not printed)"; fail=1
fi
HILOV="${h#*=}"
case "$HILOV" in "") ;; *) echo "info w8a8-hilo channels by the head container's env: set (${#HILOV} chars; names not printed)";; esac
eqswitch GLM53_DENSE_W8A8_HILO_SEL "call first" 'unset, call or first'; HISELV="$EQV"
case "$HISELV" in ""|call|first) ;; *) echo "MISS [head] container env GLM53_DENSE_W8A8_HILO_SEL is neither unset, call nor first: decide by hand"; fail=1;; esac
echo "info w8a8-hilo channel-set selection by the head container's env: $(case "$HISELV" in first) echo first;; *) echo call;; esac)"
# ---- r16z5 knobs: GLM53_MLA_PREFILL_KV_ROWS (kdamhc; free-form 'all'|N -> equality only, the start.sh validated
# the value), the mhc-fused triple (GLM53_MHC_FUSED + _ROUND_A + _CFG), GLM53_DENSE_W8A8_SKIP_LAYERS
# (w8a8layers; free-form list -> equality only) and GLM53_MOE_E4M3_MAINLOOP (moe2-rev; 0|1).
eqswitch GLM53_MLA_PREFILL_KV_ROWS; KRVV="$EQV"; PRE_KRV="$EQPRE"
echo "info kdamhc kv_rows by the head container's env: $(case "$KRVV" in "") echo "LIMITED (the deliberate default)";; all) echo "all (the revert lever)";; *) echo "set (a row count)";; esac)"
eqswitch GLM53_MHC_FUSED '|0|1' 'unset, 0 or 1'; MFCV="$EQV"; PRE_MFC="$EQPRE"
case "$MFCV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MHC_FUSED is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
v2=$(eqswitch GLM53_MHC_FUSED_ROUND_A '|0|1' 'unset, 0 or 1'); MFRV="$EQV"
v3=$(eqswitch GLM53_MHC_FUSED_CFG ''); MFCFGV="$EQV"
echo "info kdamhc mhc-fused by the head container's env: $(case "$MFCV" in 1) echo on;; *) echo off;; esac) round_a=$([ -n "$MFRV" ] && echo "$MFRV" || echo 0) cfg=$([ -n "$MFCFGV" ] && echo "$MFCFGV" || echo 9)"
if [ "$MFCV" = 1 ]; then
  case "$MFRV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MHC_FUSED_ROUND_A is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
  case "$MFCFGV" in ""|*[!0-9]*) [ -z "$MFCFGV" ] || { echo "MISS [head] container env GLM53_MHC_FUSED_CFG is not a decimal cfg: decide by hand"; fail=1; };; esac
fi
[ "$MFCV" = 1 ] || { [ -z "$MFRV" ] || echo "info kdamhc mhc-fused dials present with MHC_FUSED off (inert: nothing reads them)"; }
SKV=""; eqswitch_skip=""
h=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DENSE_W8A8_SKIP_LAYERS=' | tail -n 1)
w=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E '^GLM53_DENSE_W8A8_SKIP_LAYERS=' | tail -n 1)
if [ -z "$h" ] && [ -z "$w" ]; then
  echo "info neither container carries GLM53_DENSE_W8A8_SKIP_LAYERS: started by a start.sh without the knob (r16z4 or older) -> no layer excluded"
elif [ -z "$h" ] || [ -z "$w" ]; then
  echo "MISS [$([ -z "$h" ] && echo head || echo worker)] container env has no GLM53_DENSE_W8A8_SKIP_LAYERS line while the other rank has one (two different start.sh?)"; fail=1
elif [ "${h#*=}" = "${w#*=}" ]; then
  echo "ok   [head=worker] container env GLM53_DENSE_W8A8_SKIP_LAYERS $([ -n "${h#*=}" ] && echo "set (${#h} chars; the list is not printed)" || echo "<unset> (no layer excluded)")"
else
  echo "MISS [worker] container env GLM53_DENSE_W8A8_SKIP_LAYERS differs between the ranks (a layer filter on one rank only = a half-served pair; values not printed)"; fail=1
fi
SKV="${h#*=}"
eqswitch GLM53_MOE_E4M3_MAINLOOP '|0|1' 'unset, 0 or 1'; MSV="$EQV"; PRE_MS="$EQPRE"
case "$MSV" in ""|0|1) ;; *) echo "MISS [head] container env GLM53_MOE_E4M3_MAINLOOP is neither unset, 0 nor 1: decide by hand"; fail=1;; esac
echo "info moe2-rev mainloop by the head container's env: $(case "$MSV" in 1) echo "lean (=1)";; *) echo "shipped (unset/0)";; esac)"
# ---- the [mhcfused-pair] pre-traffic gate: the fused mHC replaces the mHC post+prenorm op whose outputs feed the
# SAME all-reduce on both ranks (and the replicated non-SP steps run both ranks' rows): a rank whose install was
# refused (or whose rank-consistent self-check uninstalled the patch) must never serve alone. Hard requirement
# BEFORE every traffic check: the armed line in BOTH ranks' logs.
if [ "$MFCV" = 1 ]; then
  gh=$(printf '%s\n' "$H" | grep -cF -- "patch_mhc_fused.py: applied (GLM53_MHC_FUSED=1)")
  gw=$(printf '%s\n' "$WK" | grep -cF -- "patch_mhc_fused.py: applied (GLM53_MHC_FUSED=1)")
  if [ "$gh" -ge 1 ] && [ "$gw" -ge 1 ]; then echo "ok   [mhcfused-pair] GLM53_MHC_FUSED=1: the patch ran on BOTH ranks (head + worker)"
  else echo "MISS [mhcfused-pair] GLM53_MHC_FUSED=1 needs the applied line on BOTH ranks (head $gh, worker $gw): a one-rank fused mHC = different mHC arithmetic per rank; ABORT before traffic"; fail=1; fi
fi
# ---- fp8ag pair gate (r16z4, docs/OPT_DENSE.md): the fp8 all-gather is a PAIRED collective (the per-layer CPU MIN
# vote over the TP group). A rank with GLM53_DENSE_W8A8_FP8AG=1 whose W8A8 install was REFUSED never wraps
# sp_all_gather and never joins the vote: its peer (which installed it) BLOCKS in the vote at the first
# sequence-parallel forward = boot hang, not a fallback (the module logs the refusal at ERROR, opt-dense-rev).
# Hard requirement, checked BEFORE every traffic check: W8A8=1 (checked above), both values equal (checked above),
# and "FP8 all-gather installed" in BOTH ranks' logs. ABORT = align GLM53_DENSE_W8A8 / _FP8AG on both ranks
# (tools/env_r16.sh) and restart; do not serve traffic in this state.
if [ -n "$FP8AGV" ] && [ "$FP8AGV" = 1 ] && [ "$W8A" = on ]; then
  gh=$(printf '%s\n' "$H" | grep -cF -- "FP8 all-gather installed")
  gw=$(printf '%s\n' "$WK" | grep -cF -- "FP8 all-gather installed")
  if [ "$gh" -ge 1 ] && [ "$gw" -ge 1 ]; then echo "ok   [fp8ag-pair] GLM53_DENSE_W8A8_FP8AG=1: FP8 all-gather installed on BOTH ranks (head + worker)"
  else echo "MISS [fp8ag-pair] GLM53_DENSE_W8A8_FP8AG=1 needs 'FP8 all-gather installed' on BOTH ranks (head $gh, worker $gw): a rank without it does not join the per-layer vote and its peer hangs at the first sequence-parallel forward; ABORT before traffic"; fail=1; fi
fi
# layer_count <value>: the number of DISTINCT layer indices a comma list / range set names (empty = 0)
layer_count() { printf '%s' "$1" | tr -d '[:space:]' | awk -F, '{
  n = 0; for (i = 1; i <= NF; i++) { p = $i; if (p == "") continue;
    if (p ~ /-/) { split(p, r, "-"); for (j = r[1]; j <= r[2]; j++) if (!(j in seen)) { seen[j] = 1; n++ } }
    else if (!(p in seen)) { seen[p] = 1; n++ } } print n; }'; }
# ---- hostloop's measurement / wake knobs (not env.r16 lines; the kit's start.sh forwards them to both ranks): the
# hostloop boot line names them ("meter every N steps" / "meter off", "wake <value>" / "wake off"), so the expected
# line follows the HEAD container's values, and the worker must carry the same ones (a knob on one rank only = MISS).
# GLM53_DEC_HOSTLOOP_METER=<N> is the production gap / eager-collective meter, GLM53_DEC_HOSTLOOP_WAKE=auto the cold
# NCCL host-node fix (docs/DEC_HOSTLOOP.md §4-§5, docs/DEC_HOSTGAP.md §2).
hlv() {   # hlv head|worker NAME -> value ("" = unset or empty)
  if [ "$1" = head ]; then docker inspect "$HEAD_C" --format "$fmt"
  else ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'"; fi | grep -E "^$2=" | tail -n 1 | cut -d= -f2-
}
HL_METER=off; HL_WAKE=off
for n in GLM53_DEC_HOSTLOOP_METER GLM53_DEC_HOSTLOOP_WAKE; do
  vh=$(hlv head "$n" || true); vw=$(hlv worker "$n" || true)
  if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env $n=${vh:-<unset>}"
  else echo "MISS [worker] container env $n=${vh:-<unset>} (got: ${vw:-<unset>})"; fail=1; fi
  lv=$(printf '%s' "$vh" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  case "$n" in
    GLM53_DEC_HOSTLOOP_METER) case "$lv" in ""|0) ;; *) HL_METER="every $vh steps";; esac;;
    GLM53_DEC_HOSTLOOP_WAKE) case "$lv" in ""|0|off|false|no) ;; *) HL_WAKE="$(printf '%s' "$vh" | tr -d '[:space:]')";; esac;;
  esac
done
# ---- r16z6 operator knobs (opt-decode; env.r16 `#knob`, site py_modules; the kit's start.sh forwards them to both
# ranks): GLM53_DEC_KDA_LAZY (+ its self-check dials _VERIFY / _VERIFY_EVERY) and GLM53_DEC_VTRIM_STATS (+ _CAP /
# _FILE). Equality between the ranks (the dials are integers / a path; the start.sh validated them), classified by
# the HEAD container's value: KDA_LAZY 1 = the one-state-per-step commit (docs/DEC_KDA_LAZY.md), VTRIM_STATS a
# positive integer = the log-only collector (docs/OPT_DECODE.md 4); unset/0 = off for both.
SWKL=""; SWVT=""; KL=off; VT=off
grep -qE '^#knob GLM53_DEC_KDA_LAZY ' "$ENVF" && SWKL=1
grep -qE '^#knob GLM53_DEC_VTRIM_STATS ' "$ENVF" && SWVT=1
if [ -n "$SWKL" ] || [ -n "$SWVT" ]; then
  kmask() { case "$1" in "") echo "<unset>";; *[!0-9]*) echo "<not printed>";; *) echo "$1";; esac; }
  for n in GLM53_DEC_KDA_LAZY GLM53_DEC_KDA_LAZY_VERIFY GLM53_DEC_KDA_LAZY_VERIFY_EVERY \
           GLM53_DEC_VTRIM_STATS GLM53_DEC_VTRIM_STATS_CAP GLM53_DEC_VTRIM_STATS_FILE; do
    vh=$(hlv head "$n" || true); vw=$(hlv worker "$n" || true)
    if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env $n=$(kmask "$vh" || true)"
    else echo "MISS [worker] container env $n=$(kmask "$vh" || true) (got: $(kmask "$vw" || true))"; fail=1; fi
  done
fi
if [ -n "$SWKL" ]; then
  KLV=$(hlv head GLM53_DEC_KDA_LAZY || true)
  case "$KLV" in 1) KL=on;; ""|0) KL=off;; *) echo "MISS [head] container env GLM53_DEC_KDA_LAZY is neither unset/0/1: decide by hand"; fail=1;; esac
fi
if [ -n "$SWVT" ]; then
  VTV=$(hlv head GLM53_DEC_VTRIM_STATS || true)
  case "$VTV" in ""|0) VT=off;; *) VT=on;; esac
fi
if [ -n "$SWKL" ] || [ -n "$SWVT" ]; then
  echo "info kdalazy/vtrim by the head container's env: GLM53_DEC_KDA_LAZY=$KL GLM53_DEC_VTRIM_STATS=$VT"
fi
# ---- r16z6sv operator knobs (decode4 track B; env.r16 `#knob`, site py_module glm53_spec_vtrim.py; the kit's start.sh
# forwards them to both ranks): GLM53_SPEC_VTRIM (+ _TAU / _MIN / _LOG). Equality between the ranks is REQUIRED (mode on
# on one rank only would make the ranks accept different drafts), classified by the HEAD container's value.
SWSV=""; SV=off
grep -qE '^#knob GLM53_SPEC_VTRIM ' "$ENVF" && SWSV=1
if [ -n "$SWSV" ]; then
  svmask() { case "$1" in "") echo "<unset>";; off|0|shadow|on|1) echo "$1";; *[!0-9.]*) echo "<not printed>";; *) echo "$1";; esac; }
  for n in GLM53_SPEC_VTRIM GLM53_SPEC_VTRIM_TAU GLM53_SPEC_VTRIM_MIN GLM53_SPEC_VTRIM_LOG; do
    vh=$(hlv head "$n" || true); vw=$(hlv worker "$n" || true)
    if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env $n=$(svmask "$vh" || true)"
    else echo "MISS [worker] container env $n=$(svmask "$vh" || true) (got: $(svmask "$vw" || true))"; fail=1; fi
  done
  SVV=$(hlv head GLM53_SPEC_VTRIM || true)
  case "$SVV" in ""|off|0) SV=off;; shadow) SV=shadow;; on|1) SV=on;; *) echo "MISS [head] container env GLM53_SPEC_VTRIM is neither unset/off/0/shadow/on/1: decide by hand"; fail=1;; esac
  echo "info specvtrim by the head container's env: GLM53_SPEC_VTRIM=$SV"
fi
# ---- r16z7 operator switches (prefixhit-adv; env.r16 `#switch`, bundle overlays patch_kpool_tail_positions.py /
# patch_mamba_align_seed.py; the kit's start.sh forwards both to both ranks): GLM53_KPOOL_TAIL_POSITIONS (unset/0 =
# stock, 2 = per-request tail rings written into the persistent slot buffer; 1 is refused by the kit's start.sh)
# and GLM53_MAMBA_ALIGN_SEED (unset/0 = stock, 1). head == worker REQUIRED (a one-rank tail fix makes the ranks'
# pooled indexer keys differ = silently degraded quality), classified by the HEAD container's value. Neither
# container carrying a name = a start.sh without the switch (r16z6sv or older): stock, no patched line allowed.
SWPT=""; PT=off; PTV=""; PREPT=""; SWAS=""; AS=off; ASV=""; PREAS=""
ptshow() { case "$1" in "") echo "<unset> (stock)";; 0|1|2) echo "$1";; *) echo "<not 0|1|2; not printed>";; esac; }
for __sw in GLM53_KPOOL_TAIL_POSITIONS GLM53_MAMBA_ALIGN_SEED; do
  grep -qE "^#switch $__sw " "$ENVF" || continue
  __hl=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E "^$__sw=" | tail -n 1)
  __wl=$(ssh -o BatchMode=yes "$W" "docker inspect $WORK_C --format '$fmt'" | grep -E "^$__sw=" | tail -n 1)
  __hv=$(printf '%s' "$__hl" | cut -d= -f2-); __wv=$(printf '%s' "$__wl" | cut -d= -f2-); __pre=""; __st=off
  if [ -z "$__hl" ] && [ -z "$__wl" ]; then
    __pre=1
    echo "info neither container carries $__sw: started by a start.sh without the switch (r16z6sv or older) -> stock; its patched lines must be absent"
  elif [ -z "$__hl" ] || [ -z "$__wl" ]; then
    echo "MISS [$([ -z "$__hl" ] && echo head || echo worker)] container env has no $__sw line while the other rank has one (two different start.sh?)"; fail=1
  elif [ "$__hv" = "$__wv" ]; then echo "ok   [head=worker] container env $__sw $(ptshow "$__hv")"
  else echo "MISS [worker] container env $__sw $(ptshow "$__hv") (got: $(ptshow "$__wv")): ONE rank with the fix = silently degraded quality; restart both ranks with the same value"; fail=1; fi
  case "$__sw:$__hv" in
    *:|*:0) __st=off;;
    GLM53_KPOOL_TAIL_POSITIONS:2|GLM53_MAMBA_ALIGN_SEED:1) __st=on;;
    GLM53_KPOOL_TAIL_POSITIONS:1) __st=other; echo "MISS [head] container env GLM53_KPOOL_TAIL_POSITIONS=1: FULL CUDA graph decode replays never read it (the kit's start.sh refuses 1); use 2"; fail=1;;
    *) __st=other; echo "MISS [head] container env $__sw is not one of the kit's values: decide by hand"; fail=1;;
  esac
  if [ "$__sw" = GLM53_KPOOL_TAIL_POSITIONS ]; then SWPT=1; PT=$__st; PTV=$__hv; PREPT=$__pre
  else SWAS=1; AS=$__st; ASV=$__hv; PREAS=$__pre; fi
done
[ -z "$SWPT$SWAS" ] || echo "info kpooltail/mambaseed by the head container's env: GLM53_KPOOL_TAIL_POSITIONS=$PT GLM53_MAMBA_ALIGN_SEED=$AS"
# ---- r16z6ar operator knobs (decode6; env.r16 `#knob`, site py_module glm53_ar1shot.py; the kit's start.sh forwards
# them to both ranks): GLM53_DEC_AR1SHOT (unset/0 off | 1 | verify) + _MAX_KB / _LOG. Equality between the ranks
# (a one-rank install would hang the first served collective), classified by the HEAD container's value
# (docs/DEC_AR1SHOT.md).
SWAR=""; AR=off
grep -qE '^#knob GLM53_DEC_AR1SHOT ' "$ENVF" && SWAR=1
if [ -n "$SWAR" ]; then
  amask() { case "$1" in "") echo "<unset>";; 0|1|verify) echo "$1";; *[!0-9]*) echo "<not printed>";; *) echo "$1";; esac; }
  for n in GLM53_DEC_AR1SHOT GLM53_DEC_AR1SHOT_MAX_KB GLM53_DEC_AR1SHOT_LOG; do
    vh=$(hlv head "$n" || true); vw=$(hlv worker "$n" || true)
    if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env $n=$(amask "$vh" || true)"
    else echo "MISS [worker] container env $n=$(amask "$vh" || true) (got: $(amask "$vw" || true))"; fail=1; fi
  done
  ARV=$(hlv head GLM53_DEC_AR1SHOT || true)
  case "$ARV" in 1) AR=on;; verify) AR=verify;; ""|0) AR=off;; *) echo "MISS [head] container env GLM53_DEC_AR1SHOT is neither unset/0/1/verify: decide by hand"; fail=1;; esac
  echo "info ar1shot by the head container's env: GLM53_DEC_AR1SHOT=$AR"
fi
# ---- r16z6dl operator knobs (decode5; env.r16 `#knob`, site py_module glm53_dlmh.py + site/tf_dlmh_ext; the kit's
# start.sh forwards them to both ranks): GLM53_DEC_DLMH (unset/0 off | 1 | verify) + _C / _GROUP / _LOG. Equality
# between the ranks, classified by the HEAD container's value (docs/DEC_DLMH.md).
SWDL=""; DL=off
grep -qE '^#knob GLM53_DEC_DLMH ' "$ENVF" && SWDL=1
if [ -n "$SWDL" ]; then
  dmask() { case "$1" in "") echo "<unset>";; 0|1|verify) echo "$1";; *[!0-9]*) echo "<not printed>";; *) echo "$1";; esac; }
  for n in GLM53_DEC_DLMH GLM53_DEC_DLMH_C GLM53_DEC_DLMH_GROUP GLM53_DEC_DLMH_LOG; do
    vh=$(hlv head "$n" || true); vw=$(hlv worker "$n" || true)
    if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env $n=$(dmask "$vh" || true)"
    else echo "MISS [worker] container env $n=$(dmask "$vh" || true) (got: $(dmask "$vw" || true))"; fail=1; fi
  done
  DLV=$(hlv head GLM53_DEC_DLMH || true)
  case "$DLV" in 1) DL=on;; verify) DL=verify;; ""|0) DL=off;; *) echo "MISS [head] container env GLM53_DEC_DLMH is neither unset/0/1/verify: decide by hand"; fail=1;; esac
  echo "info dlmh by the head container's env: GLM53_DEC_DLMH=$DL"
fi
# ---- r16z8p operator knob (env.r16 `#knob`, site py_module glm53_mla_planpin.py; the kit's start.sh forwards it to both
# ranks): GLM53_MLA_PLAN_PIN (unset/0 off | 1). Equality between the ranks, classified by the HEAD container's value
# (docs/MLA_PLAN_PIN.md). No collective is involved (each rank's plan is local), but a one-rank install would make the
# two ranks' step times differ, and the A/B arm's PROOF needs it on both.
SWPP=""; PP=off
grep -qE '^#knob GLM53_MLA_PLAN_PIN ' "$ENVF" && SWPP=1
if [ -n "$SWPP" ]; then
  vh=$(hlv head GLM53_MLA_PLAN_PIN || true); vw=$(hlv worker GLM53_MLA_PLAN_PIN || true)
  pmask() { case "$1" in "") echo "<unset>";; 0|1) echo "$1";; *) echo "<not printed>";; esac; }
  if [ "$vh" = "$vw" ]; then echo "ok   [head=worker] container env GLM53_MLA_PLAN_PIN=$(pmask "$vh" || true)"
  else echo "MISS [worker] container env GLM53_MLA_PLAN_PIN=$(pmask "$vh" || true) (got: $(pmask "$vw" || true))"; fail=1; fi
  case "$vh" in 1) PP=on;; ""|0) PP=off;; *) echo "MISS [head] container env GLM53_MLA_PLAN_PIN is neither unset/0/1: decide by hand"; fail=1;; esac
  echo "info planpin by the head container's env: GLM53_MLA_PLAN_PIN=$PP"
fi
# ---- [r16z8p] ABLIT (the o_proj ablit transplant; owner decision 2026-10-05: production runs ABLIT=1). The kit's start.sh
# honours ABLIT from .env (a caller export still wins) and forwards the effective value to both ranks. Reported in every
# mode; head == worker required (one rank transplanted = the two TP halves of every edited o_proj disagree).
ABH=$(hlv head ABLIT || true); ABW=$(hlv worker ABLIT || true)
abmask() { case "$1" in "") echo "<unset>";; 0|1) echo "$1";; *) echo "<not printed>";; esac; }
if [ "$ABH" = "$ABW" ]; then echo "ok   [head=worker] container env ABLIT=$(abmask "$ABH" || true)"
else echo "MISS [worker] container env ABLIT=$(abmask "$ABH" || true) (got: $(abmask "$ABW" || true))"; fail=1; fi
case "$ABH" in 1) AB=on;; ""|0) AB=off;; *) AB=bad; echo "MISS [head] container env ABLIT is neither unset/0/1: decide by hand"; fail=1;; esac
echo "info ablit by the head container's env: ABLIT=$(abmask "$ABH" || true) (o_proj transplant $AB)"
need() {   # need <rank label> <log text> <min count> <fixed string>
  local n; n=$(printf '%s\n' "$2" | grep -cF -- "$4")
  if [ "$n" -ge "$3" ]; then echo "ok   [$1] ($n) $4"; else echo "MISS [$1] ($n < $3) $4"; fail=1; fi
}
never() {  # never <rank label> <log text> <fixed string>
  local n; n=$(printf '%s\n' "$2" | grep -cF -- "$3")
  if [ "$n" -eq 0 ]; then echo "ok   [$1] absent: $3"; else echo "BAD  [$1] ($n) $3"; printf '%s\n' "$2" | grep -F -- "$3" | head -3 | mask; fail=1; fi
}
# ---- mhcsp D-A (adversarial review 2, docs/MHC_SP.md): the SP prefill is a PAIRED collective at TP=2 (the
# attention/MLP all-reduce becomes a reduce-scatter + all-gather pair): a rank with GLM53_MHC_SP=1 whose peer does
# not carry the switch - or whose model.py the bundle did not patch - silently CORRUPTS the residual stream instead
# of failing. Hard requirement, checked BEFORE every traffic check: the values must be equal (checked above) and
# "model.py: patched;" must be in BOTH ranks' logs. ABORT = align GLM53_MHC_SP on both ranks (tools/env_r16.sh) and
# restart; do not serve traffic in this state.
if [ -n "$SWSP" ] && [ "$SP" = on ]; then
  gh=$(printf '%s\n' "$H" | grep -cF -- "[glm53-mhc-sp] vllm/models/glm5next/nvidia/model.py: patched;")
  gw=$(printf '%s\n' "$WK" | grep -cF -- "[glm53-mhc-sp] vllm/models/glm5next/nvidia/model.py: patched;")
  if [ "$gh" -ge 1 ] && [ "$gw" -ge 1 ]; then echo "ok   [mhcsp-pair] GLM53_MHC_SP=1: model.py patched on BOTH ranks (head + worker)"
  else echo "MISS [mhcsp-pair] GLM53_MHC_SP=1 needs 'model.py: patched;' on BOTH ranks (head $gh, worker $gw): the SP prefill is a paired RS+AG collective, one rank on = silent corruption; ABORT before traffic"; fail=1; fi
fi
# ---- mhcsp2 pair gate (r16z, docs/MHC_SP2.md): the pipelined SP extends the r16x SP patch of the SAME model.py
# and keeps the paired reduce-scatter/all-gather shape (sub-chunked, on a side stream). A rank with GLM53_MHC_SP2=1
# whose peer does not carry the switch (or whose model.py the sp2 patch did not run on) does NOT fail loudly: the
# two ranks' collective counts disagree = hang or silent corruption. Hard requirement, checked BEFORE every traffic
# check, next to [mhcsp-pair]: both values equal (checked above), GLM53_MHC_SP=1 on both (checked above), and the
# sp2 patched line in BOTH ranks' logs. ABORT = align GLM53_MHC_SP2 on both ranks (tools/env_r16.sh) and restart.
if [ -n "$SWSP2" ] && [ "$SP2" = on ]; then
  g2h=$(printf '%s\n' "$H" | grep -cF -- "[glm53-mhc-sp2] vllm/models/glm5next/nvidia/model.py: patched;")
  g2w=$(printf '%s\n' "$WK" | grep -cF -- "[glm53-mhc-sp2] vllm/models/glm5next/nvidia/model.py: patched;")
  if [ "$g2h" -ge 1 ] && [ "$g2w" -ge 1 ]; then echo "ok   [mhcsp2-pair] GLM53_MHC_SP2=1: the sp2 patch ran on BOTH ranks (head + worker)"
  else echo "MISS [mhcsp2-pair] GLM53_MHC_SP2=1 needs the sp2 patched line on BOTH ranks (head $g2h, worker $g2w): the pipelined SP is a paired RS+AG collective, one rank on = mismatched collective counts; ABORT before traffic"; fail=1; fi
fi
# ---- kpooltail pair gate (r16z7, docs/PREFIX_HIT_TAIL.md): with GLM53_KPOOL_TAIL_POSITIONS=2 BOTH ranks must have
# run BOTH edits of the overlay (indexer.py in place + mamba_hybrid.py positions) BEFORE any traffic: a rank without
# them keeps the shared stock tail ring, its pooled indexer keys differ from its peer's and the sparse top-k of the
# two TP halves silently disagree. ABORT = align the switch on both ranks (tools/env_r16.sh) and restart.
if [ -n "$SWPT" ] && [ "$PT" = on ]; then
  pih=$(printf '%s\n' "$H" | grep -cF -- "indexer.py: patched (in-place persistent tail slots)")
  piw=$(printf '%s\n' "$WK" | grep -cF -- "indexer.py: patched (in-place persistent tail slots)")
  pmh=$(printf '%s\n' "$H" | grep -cF -- "mamba_hybrid.py: patched (MambaHybridModelState.prepare_attn passes positions)")
  pmw=$(printf '%s\n' "$WK" | grep -cF -- "mamba_hybrid.py: patched (MambaHybridModelState.prepare_attn passes positions)")
  if [ "$pih" -ge 1 ] && [ "$piw" -ge 1 ] && [ "$pmh" -ge 1 ] && [ "$pmw" -ge 1 ]; then echo "ok   [kpooltail-pair] GLM53_KPOOL_TAIL_POSITIONS=2: indexer.py + mamba_hybrid.py patched on BOTH ranks (head + worker)"
  else echo "MISS [kpooltail-pair] GLM53_KPOOL_TAIL_POSITIONS=2 needs both patched lines on BOTH ranks (indexer head $pih worker $piw, mamba_hybrid head $pmh worker $pmw): a rank on the stock tail ring = silently different pooled keys; ABORT before traffic"; fail=1; fi
fi
for rank in head worker; do
  if [ $rank = head ]; then L="$H"; else L="$WK"; fi
  # ---- deploy-r15 features stay installed (R15 env unchanged)
  need $rank "$L" 1 "tf_exl3_moe installed: replaces"
  need $rank "$L" 1 "tf_fp8_gemv installed:"
  need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_tail_seed_stride.py: applied (GLM53_KPOOL_SEED_STRIDE=1)"
  # the R15 runbook's rollback triggers (docs/PRODUCTION_PLAN.md Phase 2) and integrate.plugin_register's
  # per-module import failures must stay absent with R16 on top
  for bad in "self-test mismatch" "not a version this fork was tested against" "K2 apply path NOT installed" \
             "production's build_exl3_fused_state raised" "no layer is registered" \
             "tf_exl3_moe plugin install failed" "glm53_moeglue not loaded" "glm53_prefill_cap not loaded" \
             "glm53_prefill_quickwins not loaded" "glm53_runtime not installed" "tf_fp8_gemv not installed" \
             "tf_fp8_roof not installed:" "tf_fp8_w8a8 not installed" "glm53_gemv_install not loaded" "glm53_mla_prefill not loaded" \
             "glm53_hostloop not loaded" "glm53_smallops_install not loaded" "glm53_dectrace not loaded"; do
    never $rank "$L" "$bad"
  done
  # ---- launcher APC overlays (run on both ranks)
  need $rank "$L" 1 "+ DFlash boundary lookup + Kpool replay floor)"
  # ---- R16 bundle, per feature: on -> its lines; off -> its install lines absent (the modules print nothing when unset)
  if [ "${ST[kpoolring]}" = on ]; then     # overlay chain, once per container start
    need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_tail_ring.py: applied (GLM53_KPOOL_RING=1)"
    need $rank "$L" 1 "[glm53-kpool-ring] indexer tail ring: 16 slots per request"
    never $rank "$L" "GLM53_KPOOL_RING unset -> skipped (stock)"
  else
    need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_tail_ring.py: GLM53_KPOOL_RING unset -> skipped (stock)"
    never $rank "$L" "[glm53-kpool-ring] indexer tail ring"
  fi
  if [ "${ST[quickwins]}" = on ]; then
    for it in mla_bmm mla_index kda_conv mhc_aux mhc_mean; do need $rank "$L" 1 "glm53 prefill quickwins: $it installed in"; done
    need $rank "$L" 1 "glm53 prefill quickwins: idx_gate installed"
  else
    never $rank "$L" "glm53 prefill quickwins: mla_bmm installed"
  fi
  if [ "${ST[mla]}" = on ]; then
    need $rank "$L" 1 "glm53_mla_prefill installed: FlashInferMLASparseSM90Impl.forward_mqa -> exact sparse-MLA prefill kernel"
    # ---- r16z4 mla-fused-index: the fused index pass is ON by default (the deliberate default change); =0 is the
    # revert lever. The calls line carries "fused index pass: N" only when the pass ran (first printed at call 1;
    # required with --after-traffic like the calls line itself); =0 - or an r16z3-or-older site module, when no
    # knob reached the container - must never print it.
    # ---- r16z5 kdamhc KV_ROWS: the module logs the write-back scope at install; the LIMITED default (the
    # deliberate default change) says "first N rows", 'all' says "every row" (the revert lever).
    echo "info [$rank] mla kv_indices write-back: $(printf '%s\n' "$L" | grep -oF 'FA2 kv_indices write-back every row' | head -1 || echo "first-row limited (the module's install line)")"
    if { [ -n "$MLAFIV" ] && [ "$MLAFIV" = 0 ]; } || [ -n "$PRE_MLAFI" ]; then
      never $rank "$L" "fused index pass:"
    elif [ "${1:-}" = --after-traffic ]; then
      need $rank "$L" 1 "fused index pass:"
    else
      echo "info [$rank] mla fused index line (required with --after-traffic): $(printf '%s\n' "$L" | grep -cF "fused index pass:")x"
    fi
  else
    never $rank "$L" "glm53_mla_prefill installed"
    never $rank "$L" "fused index pass:"
  fi
  if [ "${ST[fp8roof]}" = on ]; then
    need $rank "$L" 1 "tf_fp8_roof installed (GLM53_DEC_FP8ROOF): table additions ['4096x20480']; L2 prefetch"
    # sidestream: the module that keeps every side-stream fork inside one breakable CUDA-graph segment (the first R16
    # boot failed its PIECEWISE capture without it; the 0501c3e module does not print this)
    need $rank "$L" 1 "breakable CUDA graphs: segment guard hooked, not forked inside a segment:"
  else
    never $rank "$L" "tf_fp8_roof installed"
  fi
  if [ "${ST[moeglue]}" = on ]; then
    need $rank "$L" 1 "glm53_moeglue warm armed: set frgd"
    need $rank "$L" 1 "o_proj layers are wired after weight loading; breakable CUDA graphs: segment guard hooked"
    need $rank "$L" 1 "glm53_moeglue warm wired 42 MoE sublayers of"
    need $rank "$L" 1 "compile-cache tag additional_config[glm53_moeglue]=warm:v1:frgd"
  else
    never $rank "$L" "glm53_moeglue warm armed"
    never $rank "$L" "glm53_moeglue warm wired"
  fi
  if [ "${ST[hostloop]}" = on ]; then
    need $rank "$L" 1 "fast path ON (verify first 64, then 1/1024), meter "
    need $rank "$L" 1 "fast path ON (verify first 64, then 1/1024), meter $HL_METER, wake $HL_WAKE"
  else
    never $rank "$L" "fast path ON"
  fi
  # wake's refusal / self-disable lines (printed only while GLM53_DEC_HOSTLOOP_WAKE is set)
  never $rank "$L" " unusable ("
  never $rank "$L" "wake switched off ("
  if [ "${ST[smallops]}" = on ]; then
    need $rank "$L" 1 "glm53_dec_smallops: ext aot:"
    need $rank "$L" 1 "dconv (10 conv modules"
  else
    never $rank "$L" "glm53_dec_smallops: ext"
  fi
  never $rank "$L" "mhc (89 weights"           # KINDS=dconv: no mHC bf16 copies (66.75 MiB)
  # refusals / self-disable lines of every R16 feature (each string is printed by the shipped code, only when the
  # feature is enabled: none of them may appear in any state)
  for bad in "glm53 prefill quickwins NOT installed" "NOT installed (production code unchanged)" \
             "quickwins install failed" "idx_gate result differs" "idx_gate turned off" "glm53_mla_prefill not installed" \
             "has no FlashInferMLASparseSM90Impl.forward_mqa" "not understood (use 1/exact)" "call went to production FA2" \
             "tf_fp8_roof: NOT installed" "L2 prefetch NOT enabled" "L2 prefetch disabled for the rest" "of an earlier forward were never" \
             "tf_fp8_roof plugin install failed" "trigger t3 (MoE -> next layer) NOT enabled" \
             "glm53_moeglue warm NOT wired" "but NOT armed" "glm53_moeglue not installed" "warm: post-load wiring failed" \
             "forked again before its MoE" "could not tag the compile cache" \
             "was still pending at a breakable CUDA-graph break" "breakable CUDA graphs cannot be guarded" \
             "GLM53_DEC_HOSTLOOP NOT installed" "fast path switched OFF" "hostloop] install failed" \
             "hostloop] plugin install failed" "glm53_dec_smallops: NOT serving" "glm53_dec_smallops not installed" \
             "glm53_dec_smallops: post-load wiring failed"; do
    never $rank "$L" "$bad"
  done
  # ---- r16j operator switch GLM53_REJECTION_METHOD (blockverify), by the head container's value
  if [ -n "$SWREJ" ]; then
    never $rank "$L" "rejection_sample_method='block' is not exact for sampled"
    never $rank "$L" "[glm53-block-keys] preflight failed"
    never $rank "$L" "patch_spec_block_keys.py failed"
    never $rank "$L" "patch_spec_block_keys.py missing but"
    if [ "$REJ" = block ]; then
      need $rank "$L" 1 "[glm53-tf-bundle] patch_spec_block_keys.py: applied (GLM53_REJECTION_METHOD=block)"
      need $rank "$L" 1 "[glm53-block-keys] rejection_sampler_utils.py: patched; dflash2/speculator.py: patched"
      need $rank "$L" 1 "[glm53-block-keys] DFlash2 draft keys: (seed, pos, draft index) for block verification"
      [ "${1:-}" = --after-traffic ] || echo "info [$rank] verifier block-keys line (JIT warmup or first spec step; required with --after-traffic): $(printf '%s\n' "$L" | grep -cF "[glm53-block-keys] rejection_sample_method='block': acceptance, resample and bonus")x"
    elif [ "$REJ" = standard ]; then
      if [ -n "$PRESW" ]; then
        :   # a pre-r16j start.sh and bundle script: no line about the overlay; the never-lines below still apply
      elif [ -n "$REJV" ]; then
        need $rank "$L" 1 "[glm53-block-keys] GLM53_REJECTION_METHOD=standard: not needed, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_spec_block_keys.py: GLM53_REJECTION_METHOD unset -> skipped (stock)"
      fi
      never $rank "$L" "patch_spec_block_keys.py: applied (GLM53_REJECTION_METHOD=block)"
      never $rank "$L" "[glm53-block-keys] DFlash2 draft keys"
      never $rank "$L" "[glm53-block-keys] rejection_sample_method='block'"
    fi
  fi
  # ---- r16k operator switch GLM53_KDA_STRIDED_QKV (kdaqkv), by the head container's value
  if [ -n "$SWKDA" ]; then
    never $rank "$L" "[glm53-kda-strided-qkv] preflight failed"
    never $rank "$L" "patch_kda_strided_qkv.py failed"
    never $rank "$L" "patch_kda_strided_qkv.py missing but"
    never $rank "$L" "GLM53_KDA_STRIDED_QKV must be exactly 1 to install"
    if [ "$KDA" = on ]; then
      need $rank "$L" 1 "[glm53-kda-strided-qkv] ops/fused_recurrent.py: patched; ops/kda.py: patched"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_kda_strided_qkv.py: applied (GLM53_KDA_STRIDED_QKV=1)"
    elif [ "$KDA" = off ]; then
      if [ -n "$PREKDA" ]; then
        :   # a pre-r16k start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$KDAV" = 0 ]; then
        need $rank "$L" 1 "[glm53-kda-strided-qkv] GLM53_KDA_STRIDED_QKV=0: stock, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_kda_strided_qkv.py: GLM53_KDA_STRIDED_QKV unset -> skipped (stock)"
      fi
      never $rank "$L" "the KDA recurrent decode reads q/k/v/beta token-strided"
      never $rank "$L" "patch_kda_strided_qkv.py: applied (GLM53_KDA_STRIDED_QKV=1)"
    fi
  fi
  # ---- r16l operator switch GLM53_KPOOL_DROP_LOWEST (kpooldown), by the head container's value
  if [ -n "$SWDROP" ]; then
    never $rank "$L" "[glm53-kpool-drop-lowest] preflight failed"
    never $rank "$L" "patch_kpool_drop_lowest.py failed"
    never $rank "$L" "patch_kpool_drop_lowest.py missing but"
    never $rank "$L" "GLM53_KPOOL_DROP_LOWEST must be"
    if [ "$DROP" = on ]; then
      need $rank "$L" 1 "[glm53-kpool-drop-lowest] sparse_attn_indexer_kpool.py: patched (GLM53_KPOOL_DROP_LOWEST=1"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_drop_lowest.py: applied (GLM53_KPOOL_DROP_LOWEST=1)"
    elif [ "$DROP" = off ]; then
      if [ -n "$PREDROP" ]; then
        :   # a pre-r16l start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$DROPV" = 0 ]; then
        need $rank "$L" 1 "[glm53-kpool-drop-lowest] GLM53_KPOOL_DROP_LOWEST=0: stock, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_drop_lowest.py: GLM53_KPOOL_DROP_LOWEST unset -> skipped (stock)"
      fi
      never $rank "$L" "sparse_attn_indexer_kpool.py: patched (GLM53_KPOOL_DROP_LOWEST=1"
      never $rank "$L" "patch_kpool_drop_lowest.py: applied (GLM53_KPOOL_DROP_LOWEST=1)"
    fi
  fi
  # ---- r16z2x operator switch GLM53_MLA_EXACT_LENS (exactlens), by the head container's value. A rank whose module
  # refused (unverified source / install failed) or whose real-op self-test failed serves production's plan while
  # the other serves the exact one: decode then differs between the ranks' MLA outputs -> every such line is BAD.
  if [ -n "$SWXL" ]; then
    never $rank "$L" "patch_mla_exactlens.py failed"
    never $rank "$L" "patch_mla_exactlens.py missing but"
    never $rank "$L" "GLM53_MLA_EXACT_LENS must be"
    never $rank "$L" "[glm53-mla-exactlens] NOT installed"
    never $rank "$L" "[glm53-mla-exactlens] install failed"
    never $rank "$L" "[glm53-mla-exactlens] plugin install failed"
    never $rank "$L" "[glm53-mla-exactlens] self-test FAILED"
    never $rank "$L" "[glm53-mla-exactlens] exact_lens failed"
    never $rank "$L" "glm53_mla_exactlens not loaded"
    if [ "$XL" = on ]; then
      need $rank "$L" 1 "[glm53-tf-bundle] patch_mla_exactlens.py: applied (GLM53_MLA_EXACT_LENS=1)"
      need $rank "$L" 1 "[glm53-mla-exactlens] integrate.py: arm"
      need $rank "$L" 1 "[glm53-mla-exactlens] installed (pid"
      need $rank "$L" 1 "[glm53-mla-exactlens] self-test passed"
    elif [ "$XL" = off ]; then
      if [ -n "$PREXL" ]; then
        :
      elif [ "$XLV" = 0 ]; then
        need $rank "$L" 1 "[glm53-mla-exactlens] GLM53_MLA_EXACT_LENS=0: stock"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_mla_exactlens.py: GLM53_MLA_EXACT_LENS unset -> skipped (stock)"
      fi
      never $rank "$L" "[glm53-mla-exactlens] installed (pid"
      never $rank "$L" "patch_mla_exactlens.py: applied (GLM53_MLA_EXACT_LENS=1)"
    fi
  fi
  # ---- r16n operator switch GLM53_KDA_FLASHKDA (flashkda), by the head container's value
  if [ -n "$SWFK" ]; then
    never $rank "$L" "[glm53-kda-flashkda] preflight failed"
    never $rank "$L" "patch_flashkda.py failed"
    never $rank "$L" "patch_flashkda.py missing but"
    never $rank "$L" "GLM53_KDA_FLASHKDA must be exactly 1 to install"
    never $rank "$L" "FlashKDA ABI mismatch"
    if [ "$FK" = on ]; then
      need $rank "$L" 1 "[glm53-kda-flashkda] kda.py: patched; glm53_flashkda.py + _flashkda_fp32_C.abi3.so installed"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_flashkda.py: applied (GLM53_KDA_FLASHKDA=1)"
      need $rank "$L" 1 "[glm53-kda-flashkda] glm53_prefill_quickwins.py: kda_conv VERIFIED + "
      need $rank "$L" 1 "kda.py: chunked prefill -> FlashKDA 17a037d (_flashkda_fp32_C, fp32 recurrent state), GLM53_KDA_FLASHKDA=1; buffers RESERVED at boot"
      # r16z fkda-v: WHICH build the bundle staged (all three under the same site names). The shipped r16x build
      # prints no extension sha; fkda2/fkda3 print staging + boot lines that name theirs.
      # [r16z6, Run6 fix] patch_flashkda.py prints ONE "staging <file.name> sha256 <that file's sha>" line PER
      # staged file: the wrapper's OWN sha and the extension's OWN sha. An earlier revision paired the .py name
      # with the .so sha - a string no code path can emit - so production's fkda3 boot (Run6, 2026-10-03) MISSED
      # 'staging glm53_flashkda3.py sha256 d98adc4a...' while the suite stayed green: the mock had been written
      # from the same wrong assumption. The needs below name the two files the patch really prints, test_boot_checks.sh
      # cross-checks every 'staging' need against patch_flashkda.py's print format + the overlay files' shas, and
      # the patch's build-swap refusal ("exists with different bytes; refusing": V changed without the intermediate
      # FK=0 restart) is a BAD row, not a bare MISS.
      if [ -n "$SWFV" ] && [ "$FVV" = 2 ]; then
        need $rank "$L" 1 "[glm53-kda-flashkda] staging glm53_flashkda2.py sha256 c6b4ee63036fc813"
        need $rank "$L" 1 "[glm53-kda-flashkda] staging _flashkda_fp32_C2.abi3.so sha256 dd1788c24f10af2c"
        need $rank "$L" 1 "extension sha256 dd1788c24f10af2c"
        never $rank "$L" "staging glm53_flashkda3.py"
        never $rank "$L" "fkda3 precision build"
        never $rank "$L" "extension sha256 d98adc4a"
        never $rank "$L" "exists with different bytes; refusing"
      elif [ -n "$SWFV" ] && [ "$FVV" = 3 ]; then
        need $rank "$L" 1 "[glm53-kda-flashkda] staging glm53_flashkda3.py sha256 e0790d8eb8ffe071"
        need $rank "$L" 1 "[glm53-kda-flashkda] staging _flashkda_fp32_C3.abi3.so sha256 d98adc4aec0046a2"
        need $rank "$L" 1 "extension sha256 d98adc4aec0046a2"
        never $rank "$L" "staging glm53_flashkda2.py"
        never $rank "$L" "fkda2 precision build"
        never $rank "$L" "extension sha256 dd1788c2"
        never $rank "$L" "exists with different bytes; refusing"
      else
        never $rank "$L" "staging glm53_flashkda2.py"
        never $rank "$L" "staging glm53_flashkda3.py"
        never $rank "$L" "extension sha256 "
        never $rank "$L" "precision build ("
      fi
    elif [ "$FK" = off ]; then
      if [ -n "$PREFK" ]; then
        :   # a pre-r16n start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$FKV" = 0 ]; then
        need $rank "$L" 1 "[glm53-kda-flashkda] GLM53_KDA_FLASHKDA=0: stock, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_flashkda.py: GLM53_KDA_FLASHKDA unset -> skipped (stock)"
      fi
      never $rank "$L" "the KDA chunked prefill runs FlashKDA 17a037d"
      never $rank "$L" "patch_flashkda.py: applied (GLM53_KDA_FLASHKDA=1)"
      never $rank "$L" "staging glm53_flashkda2.py"
      never $rank "$L" "staging glm53_flashkda3.py"
      never $rank "$L" "extension sha256 "
    fi
  fi
  # ---- r16z operator switch GLM53_KDA_FLASHKDA_V (fkda-v), by the head container's value: with the flashkda
  # switch ON, the build rows above check the staged build; here only the impossible combinations.
  if [ -n "$SWFV" ]; then
    never $rank "$L" "GLM53_KDA_FLASHKDA_V must be unset, 1, 2 or 3"
    if [ "$FK" != on ] && { [ "$FVV" = 2 ] || [ "$FVV" = 3 ]; }; then
      never $rank "$L" "[glm53-kda-flashkda] kda.py: patched;"
    fi
  fi
  # ---- r16z6 operator knobs (opt-decode; kdalazy + vtrim), by the head container's values. The two site modules
  # log "plugin loaded ... -> off, production kernels unchanged" / "-> recording" in EVERY process (integrate.py
  # imports them), so the off state NEEDs that line too (the module is shipped by this kit's site/); the ON state
  # NEEDs the module's install lines. A 'self-check: ... differ' line = the fast commit disagreed with production's
  # kernel and the commit fell into repair mode: exact but ~5 ms/step SLOWER - a rollback trigger, treated here as
  # a refusal (docs/OPT_DECODE_REV.md 1: 'repair mode = no gain' understates it, ~-5 ms/step).
  if [ -n "$SWKL" ]; then
    # the '-> installing' / '-> off' arrows are on the module's own plugin line (not another module's wording)
    klp=$(printf '%s\n' "$L" | grep -cE "glm53_kda_lazy plugin loaded .*-> installing")
    klo=$(printf '%s\n' "$L" | grep -cE "glm53_kda_lazy plugin loaded .*-> off")
    if [ "$KL" = on ]; then
      need $rank "$L" 1 "glm53_kda_lazy on "
      need $rank "$L" 1 "fast commit == production's kernel byte for byte"
      if [ "$klp" -ge 1 ]; then echo "ok   [$rank] ($klp) glm53_kda_lazy plugin loaded ... -> installing"
      else echo "MISS [$rank] (0 < 1) glm53_kda_lazy plugin loaded ... -> installing"; fail=1; fi
      never $rank "$L" "glm53_kda_lazy: self-check:"
      # [r16z6rev] the loader hook runs post_load for EVERY model a worker loads: production's DFlash2 drafter
      # (SPEC_METHOD=dflash, DFLASH_DRAFT_TP=2 -> loaded on BOTH ranks) has no KDA layers and ALWAYS logs
      # "glm53_kda_lazy: not wired on DFlash2Qwen3ForCausalLM (no KDA modules); production's per-row stores stay"
      # (docs/OPT_DECODE_REV.md 2; every opt-decode engine log). The r16z6 `never "not wired on"` row therefore
      # refused EVERY production kdalazy boot on both ranks (the Run6 failure class again: the B.26 mock omitted the
      # drafter line). A model that has no KDA modules is not a refusal - the target's own install line is NEEDed
      # above; any OTHER not-wired reason (no speculative decoding, unsafe gate, config, a missing module global)
      # is the target staying on production's stores: BAD.
      klnw=$(printf '%s\n' "$L" | grep -F "glm53_kda_lazy: not wired on " | grep -cvF "(no KDA modules)")
      klnd=$(printf '%s\n' "$L" | grep -F "glm53_kda_lazy: not wired on " | grep -cF "(no KDA modules)")
      if [ "$klnw" -gt 0 ]; then echo "BAD  [$rank] ($klnw) glm53_kda_lazy: not wired on <a model WITH KDA layers> (production's per-row stores stay):"
        printf '%s\n' "$L" | grep -F "glm53_kda_lazy: not wired on " | grep -vF "(no KDA modules)" | head -3 | mask; fail=1
      else echo "ok   [$rank] absent: glm53_kda_lazy: not wired on <a model with KDA layers> (models without KDA layers, e.g. the DFlash2 drafter: $klnd)"; fi
      never $rank "$L" "glm53_kda_lazy: wiring failed"
      # the one FATAL fallback reason (a layer-to-layer gate-lower-bound disagreement) is carried in the module's
      # printf argument, so the fixed string can never appear before the reason words: check it with a regex here
      # instead of a `never` row (check_boot_strings.py would call the %-format-spanning needle unprintable)
      kllb=$(printf '%s\n' "$L" | grep -cE "production path for a KDA call .lower bound differs")
      if [ "$kllb" -gt 0 ]; then echo "BAD  [$rank] ($kllb) production path for a KDA call (lower bound differs between layers): the lazy commit needs ONE gate lower bound, this is a layer-mismatch refusal"; fail=1
      else echo "ok   [$rank] absent: production path for a KDA call (lower bound differs between layers)"; fi
    elif [ "$KL" = off ]; then
      if [ "$klo" -ge 1 ]; then echo "ok   [$rank] ($klo) glm53_kda_lazy plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_kda_lazy plugin loaded ... -> off, production kernels unchanged"; fail=1; fi
      never $rank "$L" "glm53_kda_lazy on "
      never $rank "$L" "fast commit == production's kernel byte for byte"
      never $rank "$L" "glm53_kda_lazy: not wired on"
    fi
  fi
  if [ -n "$SWVT" ]; then
    vtp=$(printf '%s\n' "$L" | grep -cE "glm53_vtrim_stats plugin loaded .*-> recording")
    vto=$(printf '%s\n' "$L" | grep -cE "glm53_vtrim_stats plugin loaded .*-> off")
    if [ "$VT" = on ]; then
      need $rank "$L" 1 "glm53_vtrim_stats on: one record per request and verify step"
      if [ "$vtp" -ge 1 ]; then echo "ok   [$rank] ($vtp) glm53_vtrim_stats plugin loaded ... -> recording"
      else echo "MISS [$rank] (0 < 1) glm53_vtrim_stats plugin loaded ... -> recording"; fail=1; fi
      never $rank "$L" "glm53_vtrim_stats not installed"
      never $rank "$L" "glm53_vtrim_stats: switched off"
    elif [ "$VT" = off ]; then
      if [ "$vto" -ge 1 ]; then echo "ok   [$rank] ($vto) glm53_vtrim_stats plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_vtrim_stats plugin loaded ... -> off"; fail=1; fi
      never $rank "$L" "glm53_vtrim_stats on: one record per request and verify step"
      never $rank "$L" "glm53_vtrim_stats not installed"
    fi
  fi
  # ---- r16z6sv operator knobs (specvtrim), by the head container's value. integrate.py imports the module in EVERY
  # process, so the plugin line is NEEDed in every state; on/shadow NEED the install line with every hook of the mode
  # (on: drafter + propose + rejection + combine + moe; shadow: drafter + propose + rejection).
  if [ -n "$SWSV" ]; then
    svo=$(printf '%s\n' "$L" | grep -cE "glm53_spec_vtrim plugin loaded .*-> off")
    svs=$(printf '%s\n' "$L" | grep -cE "glm53_spec_vtrim plugin loaded .*-> shadow tau")
    svn=$(printf '%s\n' "$L" | grep -cE "glm53_spec_vtrim plugin loaded .*-> on tau")
    never $rank "$L" "[glm53-spec-vtrim] not installed"
    never $rank "$L" "glm53_spec_vtrim not loaded"
    never $rank "$L" "[glm53-spec-vtrim] mode on but the hooks could not be installed"
    if [ "$SV" = on ]; then
      if [ "$svn" -ge 1 ]; then echo "ok   [$rank] ($svn) glm53_spec_vtrim plugin loaded ... -> on tau"
      else echo "MISS [$rank] (0 < 1) glm53_spec_vtrim plugin loaded ... -> on tau"; fail=1; fi
      need $rank "$L" 1 "(hooks: drafter,propose,rejection,combine,moe;"
    elif [ "$SV" = shadow ]; then
      if [ "$svs" -ge 1 ]; then echo "ok   [$rank] ($svs) glm53_spec_vtrim plugin loaded ... -> shadow tau"
      else echo "MISS [$rank] (0 < 1) glm53_spec_vtrim plugin loaded ... -> shadow tau"; fail=1; fi
      need $rank "$L" 1 "(hooks: drafter,propose,rejection;"
    else
      if [ "$svo" -ge 1 ]; then echo "ok   [$rank] ($svo) glm53_spec_vtrim plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_spec_vtrim plugin loaded ... -> off"; fail=1; fi
      never $rank "$L" "[glm53-spec-vtrim] installed rank"
    fi
  fi
  # ---- r16z7 operator switches (kpooltail / mambaseed), by the head container's value
  if [ -n "$SWPT" ]; then
    never $rank "$L" "patch_kpool_tail_positions.py failed"
    never $rank "$L" "patch_kpool_tail_positions.py missing but"
    if [ "$PT" = on ]; then
      need $rank "$L" 1 "[glm53-kpool-tail-positions] /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py: patched (in-place persistent tail slots)"
      need $rank "$L" 1 "[glm53-kpool-tail-positions] /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py: patched (MambaHybridModelState.prepare_attn passes positions)"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_tail_positions.py: applied (GLM53_KPOOL_TAIL_POSITIONS=2)"
    elif [ "$PT" = off ]; then
      if [ -n "$PREPT" ]; then
        :   # a pre-r16z7 start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$PTV" = 0 ]; then
        need $rank "$L" 1 "[glm53-kpool-tail-positions] GLM53_KPOOL_TAIL_POSITIONS=0 -> stock (nothing read or written)"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_kpool_tail_positions.py: GLM53_KPOOL_TAIL_POSITIONS unset -> skipped (stock)"
      fi
      never $rank "$L" "(in-place persistent tail slots)"
      never $rank "$L" "(MambaHybridModelState.prepare_attn passes positions)"
    fi
  fi
  if [ -n "$SWAS" ]; then
    never $rank "$L" "patch_mamba_align_seed.py failed"
    never $rank "$L" "patch_mamba_align_seed.py missing but"
    if [ "$AS" = on ]; then
      need $rank "$L" 1 "[glm53-mamba-align-seed] /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py: patched (add_request seeds in mamba blocks)"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_mamba_align_seed.py: applied (GLM53_MAMBA_ALIGN_SEED=1)"
    elif [ "$AS" = off ]; then
      if [ -n "$PREAS" ]; then
        :   # a pre-r16z7 start.sh
      elif [ "$ASV" = 0 ]; then
        need $rank "$L" 1 "[glm53-mamba-align-seed] GLM53_MAMBA_ALIGN_SEED=0 -> stock (nothing read or written)"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_mamba_align_seed.py: GLM53_MAMBA_ALIGN_SEED unset -> skipped (stock)"
      fi
      never $rank "$L" "(add_request seeds in mamba blocks)"
    fi
  fi
  # ---- r16z6dl operator knobs (dlmh), by the head container's value. The site module logs "glm53_dlmh plugin
  # loaded ... -> off, production's candidate head unchanged" in EVERY process (integrate.py imports it), so off NEEDs
  # that line; on / verify NEED the install, the drafter hook, the coarse-head build and the byte-equality self-test
  # (both run at the drafter's first eager call, during the boot's dummy runs) on every rank.
  if [ -n "$SWDL" ]; then
    dlp=$(printf '%s\n' "$L" | grep -cE "glm53_dlmh plugin loaded .*-> installing \(mode $DL")
    dlo=$(printf '%s\n' "$L" | grep -cE "glm53_dlmh plugin loaded .*-> off")
    if [ "$DL" = on ] || [ "$DL" = verify ]; then
      if [ "$dlp" -ge 1 ]; then echo "ok   [$rank] ($dlp) glm53_dlmh plugin loaded ... -> installing (mode $DL"
      else echo "MISS [$rank] (0 < 1) glm53_dlmh plugin loaded ... -> installing (mode $DL"; fail=1; fi
      need $rank "$L" 1 "glm53_dlmh: hooked DFlash2 compute_candidates"
      need $rank "$L" 1 "coarse candidate head built"
      need $rank "$L" 1 "self-test: candidates and unary logits byte-equal to production's"
      never $rank "$L" "glm53_dlmh: setup failed"
      never $rank "$L" "glm53_dlmh: hook failed"
      never $rank "$L" "self-test: two-stage candidates differ"
      never $rank "$L" "glm53_dlmh: not wired"
      never $rank "$L" "glm53_dlmh not installed"
      never $rank "$L" "glm53_dlmh: bad configuration"
      # [r16z8] the serving PROOF (the A/B's): once per rank after CUDA-graph replays of the two-stage head (first stats
      # line after 64 drafter steps) - required with --after-traffic (a few decode turns), else counted
      dls=$(printf '%s\n' "$L" | grep -cE "\[glm53-dlmh\] rank [0-9]+ serving confirmed \(mode $DL\): graph replays [1-9]")
      if [ "${1:-}" = --after-traffic ]; then
        if [ "$dls" -ge 1 ]; then echo "ok   [$rank] ($dls) [glm53-dlmh] rank R serving confirmed (mode $DL): graph replays N"
        else echo "MISS [$rank] (0 < 1) [glm53-dlmh] rank R serving confirmed (mode $DL): graph replays N (after >= 64 drafter steps)"; fail=1; fi
      else echo "info [$rank] dlmh serving-confirmed line (required with --after-traffic): ${dls}x"; fi
    elif [ "$DL" = off ]; then
      if [ "$dlo" -ge 1 ]; then echo "ok   [$rank] ($dlo) glm53_dlmh plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_dlmh plugin loaded ... -> off, production's candidate head unchanged"; fail=1; fi
      never $rank "$L" "glm53_dlmh: hooked DFlash2 compute_candidates"
      never $rank "$L" "coarse candidate head built"
    fi
  fi
  # ---- r16z8p operator knob (planpin), by the head container's value. The site module logs "glm53_mla_planpin plugin
  # loaded ... -> off, production's pageable plan staging unchanged" in EVERY process (integrate.py imports it), so off
  # NEEDs that line; on NEEDs the install, the class patch (at the sparse-MLA backend import) and the first-plan
  # self-test (the boot's profile/dummy runs plan) on every rank. The serving line '[glm53-mla-planpin] rank R serving
  # confirmed (mode on): N plans' comes at the 64th plan of the process (the boot's capture runs usually reach it;
  # required with --after-traffic, else counted).
  if [ -n "$SWPP" ]; then
    ppp=$(printf '%s\n' "$L" | grep -cE "glm53_mla_planpin plugin loaded .*-> installing \(mode on\)")
    ppo=$(printf '%s\n' "$L" | grep -cE "glm53_mla_planpin plugin loaded .*-> off")
    if [ "$PP" = on ]; then
      if [ "$ppp" -ge 1 ]; then echo "ok   [$rank] ($ppp) glm53_mla_planpin plugin loaded ... -> installing (mode on)"
      else echo "MISS [$rank] (0 < 1) glm53_mla_planpin plugin loaded ... -> installing (mode on)"; fail=1; fi
      need $rank "$L" 1 "glm53_mla_planpin: patched _SM90State.plan"
      need $rank "$L" 1 "self-test: 3/3 device plan buffers == pinned staging"
      never $rank "$L" "glm53_mla_planpin: NOT installed"
      never $rank "$L" "glm53_mla_planpin not loaded"
      never $rank "$L" "ring allocation failed"
      ppf=$(printf '%s\n' "$L" | grep -cE "glm53_mla_planpin: rank [0-9?]+ self-test FAILED")
      if [ "$ppf" -eq 0 ]; then echo "ok   [$rank] absent: glm53_mla_planpin: rank R self-test FAILED"
      else echo "BAD  [$rank] ($ppf) glm53_mla_planpin: rank R self-test FAILED"; fail=1; fi
      pps=$(printf '%s\n' "$L" | grep -cE "\[glm53-mla-planpin\] rank [0-9?]+ serving confirmed \(mode on\): [1-9][0-9]* plans through the pinned ring")
      if [ "${1:-}" = --after-traffic ]; then
        if [ "$pps" -ge 1 ]; then echo "ok   [$rank] ($pps) [glm53-mla-planpin] rank R serving confirmed (mode on): N plans through the pinned ring"
        else echo "MISS [$rank] (0 < 1) [glm53-mla-planpin] rank R serving confirmed (mode on): N plans through the pinned ring"; fail=1; fi
      else echo "info [$rank] planpin serving-confirmed line (required with --after-traffic): ${pps}x"; fi
    else
      if [ "$ppo" -ge 1 ]; then echo "ok   [$rank] ($ppo) glm53_mla_planpin plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_mla_planpin plugin loaded ... -> off, production's pageable plan staging unchanged"; fail=1; fi
      never $rank "$L" "glm53_mla_planpin: patched _SM90State.plan"
      never $rank "$L" "plans through the pinned ring"
    fi
  fi
  # ---- [r16z8p] ABLIT per rank, by the head container's value: 1 -> the container entry's "ablit: o_proj
  # orthogonalization ON" line on this rank; 0/unset -> that line absent
  if [ "$AB" = on ]; then
    need $rank "$L" 1 "ablit: o_proj orthogonalization ON"
  elif [ "$AB" = off ]; then
    never $rank "$L" "ablit: o_proj orthogonalization ON"
  fi
  # ---- r16msp operator switch GLM53_MHC_SP (mhcsp), by the head container's value
  if [ -n "$SWSP" ]; then
    never $rank "$L" "[glm53-mhc-sp] preflight failed"
    never $rank "$L" "patch_mhc_sp.py failed"
    never $rank "$L" "patch_mhc_sp.py missing but"
    never $rank "$L" "GLM53_MHC_SP must be"
    if [ "$SP" = on ]; then
      need $rank "$L" 1 "[glm53-mhc-sp] vllm/models/glm5next/nvidia/model.py: patched;"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_sp.py: applied (GLM53_MHC_SP=1)"
    elif [ "$SP" = off ]; then
      if [ -n "$PRESP" ]; then
        :   # a pre-r16msp start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$SPV" = 0 ]; then
        need $rank "$L" 1 "[glm53-mhc-sp] GLM53_MHC_SP=0: stock, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_sp.py: GLM53_MHC_SP unset -> skipped (stock)"
      fi
      never $rank "$L" "vllm/models/glm5next/nvidia/model.py: patched;"
      never $rank "$L" "patch_mhc_sp.py: applied (GLM53_MHC_SP=1)"
    fi
  fi
  # ---- r16z6ar operator knobs (ar1shot), by the head container's value. The site module logs "glm53_ar1shot plugin
  # loaded ... -> off, production all-reduce unchanged" in EVERY process (integrate.py imports it), so off NEEDs that
  # line; on / verify NEED the install, the CudaCommunicator hook and the TP-wide readiness agreement (taken inside
  # CudaCommunicator.__init__, i.e. at process-group construction, before the profile run and any capture) on every rank. The serving line
  # '[glm53-ar1shot] rank R serving confirmed' comes with the first eager all-reduce after capture (traffic: the A/B
  # PROOF, not a boot row).
  if [ -n "$SWAR" ]; then
    arp=$(printf '%s\n' "$L" | grep -cE "glm53_ar1shot plugin loaded .*-> installing \(mode $AR")
    aro=$(printf '%s\n' "$L" | grep -cE "glm53_ar1shot plugin loaded .*-> off")
    if [ "$AR" = on ] || [ "$AR" = verify ]; then
      if [ "$arp" -ge 1 ]; then echo "ok   [$rank] ($arp) glm53_ar1shot plugin loaded ... -> installing (mode $AR"
      else echo "MISS [$rank] (0 < 1) glm53_ar1shot plugin loaded ... -> installing (mode $AR"; fail=1; fi
      need $rank "$L" 1 "glm53_ar1shot: hooked CudaCommunicator.all_reduce (mode $AR"
      need $rank "$L" 1 "agreement: every rank ready -> one-shot all-reduce armed (mode $AR"
      never $rank "$L" "glm53_ar1shot: not installed"
      never $rank "$L" "glm53_ar1shot not loaded"
      never $rank "$L" "agreement: a rank is not ready"
      never $rank "$L" "one-shot all-reduce differs from production's"
      # [r16z8] the serving PROOF (the A/B's): at the first eager all-reduce after the decode graphs captured one-shot
      # collectives (N > 0) - required with --after-traffic, else counted
      ars=$(printf '%s\n' "$L" | grep -cE "\[glm53-ar1shot\] rank [0-9]+ serving confirmed \(mode $AR\): [1-9][0-9]* graph-captured one-shot all-reduces")
      if [ "${1:-}" = --after-traffic ]; then
        if [ "$ars" -ge 1 ]; then echo "ok   [$rank] ($ars) [glm53-ar1shot] rank R serving confirmed (mode $AR): N graph-captured one-shot all-reduces"
        else echo "MISS [$rank] (0 < 1) [glm53-ar1shot] rank R serving confirmed (mode $AR): N graph-captured one-shot all-reduces"; fail=1; fi
      else echo "info [$rank] ar1shot serving-confirmed line (required with --after-traffic): ${ars}x"; fi
    elif [ "$AR" = off ]; then
      if [ "$aro" -ge 1 ]; then echo "ok   [$rank] ($aro) glm53_ar1shot plugin loaded ... -> off"
      else echo "MISS [$rank] (0 < 1) glm53_ar1shot plugin loaded ... -> off, production all-reduce unchanged"; fail=1; fi
      never $rank "$L" "glm53_ar1shot: hooked CudaCommunicator.all_reduce"
      never $rank "$L" "one-shot all-reduce armed"
    fi
  fi
  # ---- r16z operator switch GLM53_MHC_SP2 (mhcsp2), by the head container's value
  if [ -n "$SWSP2" ]; then
    never $rank "$L" "[glm53-mhc-sp2] preflight failed"
    never $rank "$L" "patch_mhc_sp2.py failed"
    never $rank "$L" "patch_mhc_sp2.py missing but"
    never $rank "$L" "GLM53_MHC_SP2 must be exactly 1 to install"
    never $rank "$L" "GLM53_MHC_SP2=1 needs GLM53_MHC_SP=1"
    if [ "$SP2" = on ]; then
      need $rank "$L" 1 "[glm53-mhc-sp2] vllm/models/glm5next/nvidia/model.py: patched; fingerprints"
      need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_sp2.py: applied (GLM53_MHC_SP2=1)"
      need $rank "$L" 1 "pipelined SP prefill installed (GLM53_MHC_SP2=1: k=2 sub-chunks per rank shard"
    elif [ "$SP2" = off ]; then
      if [ -n "$PRESP2" ]; then
        :   # a pre-r16z start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$SP2V" = 0 ]; then
        need $rank "$L" 1 "[glm53-mhc-sp2] GLM53_MHC_SP2=0: stock, files untouched"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_sp2.py: GLM53_MHC_SP2 unset -> skipped (stock)"
      fi
      never $rank "$L" "[glm53-mhc-sp2] vllm/models/glm5next/nvidia/model.py: patched;"
      never $rank "$L" "patch_mhc_sp2.py: applied (GLM53_MHC_SP2=1)"
      never $rank "$L" "pipelined SP prefill installed"
    fi
  fi
  # ---- r16e4 operator switch GLM53_MOE_E4M3 (moee4m3), by the head container's value
  if [ -n "$SWE4" ]; then
    for bad in "patch_moe_e4m3.py failed" "patch_moe_e4m3.py missing but" "[glm53-moe-e4m3] GLM53_MOE_E4M3 must be" \
               "glm53_moe_e4m3 not installed" "glm53_moe_e4m3 not loaded" "glm53_moe_e4m3 layer self-test FAILED" \
               "glm53_moe_e4m3 layer self-test raised" "glm53_moe_e4m3: layer not served" \
               "glm53_moe_e4m3 DECODE BOUND EXCEEDED"; do
      never $rank "$L" "$bad"
    done
    if [ "$E4" = on ]; then
      need $rank "$L" 1 "[glm53-tf-bundle] patch_moe_e4m3.py: applied (GLM53_MOE_E4M3=1)"
      need $rank "$L" 1 "[glm53-moe-e4m3] glm53_moe_e4m3.py: installed"
      need $rank "$L" 1 "[glm53-moe-e4m3] glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so: installed"
      need $rank "$L" 1 "[glm53-moe-e4m3] integrate.py: "
      need $rank "$L" 1 "glm53_moe_e4m3 installed: GLM53_MOE_E4M3=1"
      need $rank "$L" 1 "per-layer self-test at load time"
      need $rank "$L" 1 "glm53_moe_e4m3 layer self-test passed"
      if [ -n "$SWLS" ] && [ -n "$LSV" ]; then
        # r16z3: a layer list is set -> the summary must serve EXACTLY the listed layers (K distinct indices,
        # expected count = the list; fewer = a layer fell back / a half-applied selection = MISS)
        KC=$(layer_count "$LSV"); UC=$((42 - KC))
        n=$(printf '%s\n' "$L" | grep -cE "glm53_moe_e4m3 summary: ${KC}/${KC} layers served, 0 fell back \(none\); self-test FAILED 0, raised 0.*GLM53_MOE_E4M3_LAYERS: ${KC} indices selected, ${UC} layers unselected \(production's path\)")
        if [ "$n" -ge 1 ]; then echo "ok   [$rank] ($n) glm53_moe_e4m3 summary: ${KC}/${KC} layers served, 0 fell back (none); self-test FAILED 0, raised 0; GLM53_MOE_E4M3_LAYERS: ${KC} indices selected, ${UC} layers unselected"
        else echo "MISS [$rank] ($n) glm53_moe_e4m3 summary must serve exactly the ${KC} listed layers, 0 fell back, with '${KC} indices selected, ${UC} layers unselected' (a half-applied selection is a MISS)"; fail=1; fi
      else
        need $rank "$L" 1 "glm53_moe_e4m3 summary: 42/42 layers served, 0 fell back (none); self-test FAILED 0, raised 0"
      fi
      need $rank "$L" 1 "glm53_moe_e4m3 active: first prefill routed-MoE call"
      # ---- r16z4 moe-opt: the accumulator / fold / token-gather dials, read back from the summary line.
      # The summary is one line; the suffixes are mutually consistent with the container env checked earlier.
      if { { [ -z "$E4ACCV" ] || [ "$E4ACCV" = f32 ]; } && [ -z "$PRE_ACC" ]; } || [ -n "$PRE_ACC" ]; then
        never $rank "$L" "GLM53_MOE_E4M3_ACC=bf16 (bf16 accumulator)"
      elif [ "$E4ACCV" = bf16 ]; then
        need $rank "$L" 1 "GLM53_MOE_E4M3_ACC=bf16 (bf16 accumulator)"
      fi
      if [ "$E4FOLDV" = 1 ]; then
        need $rank "$L" 1 "GLM53_MOE_E4M3_FOLD_SHARED=1 (routed sum accumulated into the shared experts' output)"
        # opt-moe-rev observability (boot-checkable by design: the profile run is a served call, folds): the FIRST
        # served call must log the fold INFO on EVERY rank BEFORE any traffic row - a rank whose first served call
        # did not fold serves a different MoE arithmetic than its peer (the dial is rank-local, the output feeds the
        # same all-reduce). The =1-not-folding WARNINGs are a MISS in every state.
        need $rank "$L" 1 "glm53_moe_e4m3 fold: first served call folded"
      elif [ -z "$PRE_FOLD" ]; then
        never $rank "$L" "(routed sum accumulated into the shared experts' output)"
        never $rank "$L" "fold: first served call folded"
      fi
      never $rank "$L" "GLM53_MOE_E4M3_FOLD_SHARED=1 NOT active"
      never $rank "$L" "was NOT folded"
      # ---- r16z5 moe2-rev: the lean mainloop. ON: the install line's "; mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)"
      # and the summary suffix on EVERY rank (a rank without it serves different MoE arithmetic than its peer).
      if [ "$MSV" = 1 ]; then
        need $rank "$L" 1 "mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)"
      elif [ -z "$PRE_MS" ]; then
        never $rank "$L" "mainloop: lean (GLM53_MOE_E4M3_MAINLOOP=1)"
      fi
      if [ -n "$E4TGV" ] && [ "$E4TGV" = 0 ]; then
        need $rank "$L" 1 "GLM53_MOE_E4M3_TOKGATHER=0 (per-pair gather)"
        never $rank "$L" "token gather "
      elif [ -z "$PRE_TG" ]; then
        # the token gather is the designed default: the summary must carry "token gather K/K served layers"
        # (K = the served layer count, 42 or the listed selection's size; a layer that does not qualify falls
        # back to the per-pair gather = a half-applied pair, MISS)
        KC=42; [ -n "$SWLS" ] && [ -n "$LSV" ] && KC=$(layer_count "$LSV")
        n=$(printf '%s\n' "$L" | grep -cF "token gather ${KC}/${KC} served layers")
        if [ "$n" -ge 1 ]; then echo "ok   [$rank] ($n) token gather ${KC}/${KC} served layers"
        else echo "MISS [$rank] ($n) the e4m3 summary must read 'token gather ${KC}/${KC} served layers' (a layer that does not qualify falls back to the per-pair gather = a half-applied pair)"; fail=1; fi
        never $rank "$L" "GLM53_MOE_E4M3_TOKGATHER=0 (per-pair gather)"
      else
        never $rank "$L" "token gather "
        never $rank "$L" "GLM53_MOE_E4M3_TOKGATHER=0 (per-pair gather)"
      fi
      # ---- r16z2 moe2: the down-projection width
      if [ -n "$SWED" ] && [ "$EDV" = f16 ]; then
        need $rank "$L" 1 "down projection: fp16"
      elif [ -n "$SWED" ]; then
        never $rank "$L" "down projection: fp16"    # unset/e4m3, or a pre-r16z2 start.sh (the V1 module prints no down line)
      fi
    elif [ "$E4" = off ]; then
      if [ -n "$PREE4" ]; then
        :   # a pre-r16e4 start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$E4V" = 0 ]; then
        need $rank "$L" 1 "[glm53-moe-e4m3] GLM53_MOE_E4M3=0 -> nothing installed"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_moe_e4m3.py: GLM53_MOE_E4M3 unset -> skipped (stock)"
      fi
      never $rank "$L" "[glm53-moe-e4m3] glm53_moe_e4m3.py: installed"
      never $rank "$L" "glm53_moe_e4m3 installed:"
      never $rank "$L" "glm53_moe_e4m3 active:"
    fi
  fi
  # ---- r16z3 operator switch GLM53_MOE_FUSED16 (fused16, moe3), by the head container's value
  if [ -n "$SWFF" ]; then
    for bad in "glm53_moe_fused16 not installed" "glm53_moe_fused16 not loaded" \
               "glm53_moe_fused16 layer self-test FAILED" "glm53_moe_fused16 layer self-test raised" \
               "glm53_moe_fused16: layer not served" "patch_moe_fused16.py failed" "patch_moe_fused16.py missing but" \
               "[glm53-moe-fused16] GLM53_MOE_FUSED16 must be"; do
      never $rank "$L" "$bad"
    done
    if [ "$FF" = on ]; then
      need $rank "$L" 1 "[glm53-tf-bundle] patch_moe_fused16.py: applied (GLM53_MOE_FUSED16=1)"
      need $rank "$L" 1 "[glm53-moe-fused16] glm53_moe_fused16.py: installed"
      need $rank "$L" 1 "[glm53-moe-fused16] glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so: installed"
      need $rank "$L" 1 "[glm53-moe-fused16] integrate.py: "
      need $rank "$L" 1 "glm53_moe_fused16 installed: GLM53_MOE_FUSED16=1"
      need $rank "$L" 1 "glm53_moe_fused16 layer self-test passed"
      need $rank "$L" 1 "glm53_moe_fused16 summary: 42/42 layers served, 0 fell back (none); self-test FAILED 0, raised 0"
      need $rank "$L" 1 "glm53_moe_fused16 active: first grouped prefill call"
    elif [ "$FF" = off ]; then
      if [ -n "$PREFF" ]; then
        :   # a pre-r16z3 start.sh: the variable never reached the container; the never-lines above still apply
      elif [ "$FFV" = 0 ]; then
        need $rank "$L" 1 "[glm53-moe-fused16] GLM53_MOE_FUSED16=0 -> nothing installed (stock prefill MoE kernels)"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_moe_fused16.py: GLM53_MOE_FUSED16 unset -> skipped (stock)"
      fi
      never $rank "$L" "[glm53-moe-fused16] glm53_moe_fused16.py: installed"
      never $rank "$L" "glm53_moe_fused16 installed:"
      never $rank "$L" "glm53_moe_fused16 active:"
    fi
  fi
  # ---- r16z5 kdamhc GLM53_MHC_FUSED (the fused mHC post+prenorm GEMM), by the head container's value. The
  # [mhcfused-pair] gate ran BEFORE traffic; here the per-rank patch/self-test rows.
  if [ -n "$PRE_MFC" ]; then
    :
  elif [ "$MFCV" = 1 ]; then
    need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_fused.py: applied (GLM53_MHC_FUSED=1)"
    need $rank "$L" 1 "[glm53-mhc-fused] glm53_mhc_fused.py: installed"
    need $rank "$L" 1 "[glm53-mhc-fused] glm53_mhc_fused_ext.cpython-312-aarch64-linux-gnu.so: installed"
    # the rank-consistent fixed-seed self-check (opt-kdamhc-rev): it must PASS on every rank; the UNINSTALLED line
    # (or any line naming a failed check) is a MISS - a rank whose self-check uninstalled serves production's mHC
    # op while its peer serves the fused kernel (the [mhcfused-pair] gate catches a full one-rank install; this
    # catches the self-check un-install the gate cannot see).
    need $rank "$L" 1 "self-check (1024 synthetic rows, fixed seed, this layer's fn): residual_cur =="
    never $rank "$L" "UNINSTALLED (production's op serves)"
    if [ -n "$MFRV" ] && [ "$MFRV" = 1 ]; then
      echo "info [$rank] mhc-fused ROUND_A=1 (the prefill-consistent rounding; the decode-consistent default is 0)"
    fi
  elif [ "$MFCV" = 0 ]; then
    need $rank "$L" 1 "[glm53-mhc-fused] GLM53_MHC_FUSED=0 -> nothing installed (production's mHC op)"
  else
    need $rank "$L" 1 "[glm53-tf-bundle] patch_mhc_fused.py: GLM53_MHC_FUSED unset -> skipped (stock)"
    never $rank "$L" "[glm53-mhc-fused] glm53_mhc_fused.py: installed"
  fi
  # ---- r16y operator switch GLM53_DENSE_W8A8 (w8a8), by the head container's value
  if [ -n "$SWW8A" ]; then
    never $rank "$L" "tf_fp8_w8a8 enabled (GLM53_DENSE_W8A8) but NOT installed"
    never $rank "$L" "tf_fp8_w8a8: a call is not eligible"
    # r16z4 w8a8-fp8ag: the opt-dense-rev refusal ERROR (a rank that will not join the per-layer vote while its
    # peer waits in it = boot hang) is a MISS in every state, before every traffic check (the [fp8ag-pair] gate)
    never $rank "$L" "will not join "
    never $rank "$L" "self-test FAILED, this layer stays on production's path"
    # r16z2rev: a self-test that RAISED (any non-CUDA exception, incl. torch's "CUDA out of memory", which the
    # module's CUDA-error regex does not match) also leaves the layer on production's path - W8A8 half-applied
    never $rank "$L" "self-test raised, layer stays on production's path"
    never $rank "$L" "tf_fp8_w8a8: self-test at load raised"
    never $rank "$L" "a different / moved arming block (drift)"
    if [ "$W8A" = on ]; then
      need $rank "$L" 1 "[glm53-tf-bundle] patch_dense_w8a8.py: applied (GLM53_DENSE_W8A8=1)"
      need $rank "$L" 1 "[glm53-w8a8] site-packages/fp8_w8a8.py: "
      need $rank "$L" 1 "tf_fp8_w8a8 installed:"
      need $rank "$L" 1 "[glm53-w8a8] integrate.py: armed"
      if [ "${1:-}" = --after-traffic ]; then
        need $rank "$L" 1 "tf_fp8_w8a8: first call served"
      else
        echo "info [$rank] tf_fp8_w8a8 first-served line (required with --after-traffic): $(printf '%s\n' "$L" | grep -cF "tf_fp8_w8a8: first call served")x"
      fi
      # ---- r16z2 w8a82: WHICH GEMM the module serves (only with the VERSION-2 extension) and the projection filter
      never $rank "$L" "custom GEMM output differs"
      if [ -n "$SWG" ] && [ "$WGV" = cutlass_mm ]; then
        never $rank "$L" "custom CUTLASS SM120 GEMM"
        never $rank "$L" "== cutlass_scaled_mm"
      elif [ -n "$SWG" ] && [ -n "$PREW82" ]; then
        :   # a pre-r16z2 start.sh: the variable never reached the container (the module there is w8a8's V1, whose
            # install/self-test lines name no custom GEMM) - the never-lines below still apply
        never $rank "$L" "custom CUTLASS SM120 GEMM"
        never $rank "$L" "== cutlass_scaled_mm"
      elif [ -n "$SWG" ]; then
        need $rank "$L" 1 "custom CUTLASS SM120 GEMM"
        need $rank "$L" 1 "== cutlass_scaled_mm"
      fi
      never $rank "$L" "GLM53_DENSE_W8A8=0: stock"
      never $rank "$L" "[glm53-w8a8] integrate.py: disarmed"
      # ---- r16z4 w8a8-fp8ag (the pair gate ran above, before traffic): the per-rank install line, and never the
      # refusal path. unset/0 = the bf16 gather, no FP8-all-gather line in any state.
      if [ "$FP8AGV" = 1 ]; then
        need $rank "$L" 1 "FP8 all-gather installed"
        never $rank "$L" "has no sp_all_gather"
        never $rank "$L" "FP8 all-gather off"
        if [ "${1:-}" = --after-traffic ]; then
          echo "info [$rank] fp8ag serve rows (the per-layer vote lines): $(printf '%s\n' "$L" | grep -cF 'local True, agreed True')x agreed"
        fi
      elif [ -z "$PRE_FP8AG" ]; then
        never $rank "$L" "FP8 all-gather installed"
      fi
      # ---- r16z5 w8a8layers: the layer filter. The install line names it; the load summary counts the excluded
      # projections. A rank whose filter differs is caught by the env equality above; the count is informational
      # (the module's own load-time accounting) unless the env is set and NO filter line exists.
      if [ -n "$SKV" ]; then
        need $rank "$L" 1 "layer filter GLM53_DENSE_W8A8_SKIP_LAYERS: excluded"
      else
        never $rank "$L" "layer filter GLM53_DENSE_W8A8_SKIP_LAYERS: excluded"
      fi
      # ---- r16z4 w8a8-hilo: the channel-set freeze line appears at each layer's first real served call
      if [ -n "$HILOV" ]; then
        if [ "${1:-}" = --after-traffic ]; then
          need $rank "$L" 1 "tf_fp8_w8a8: hi+lo ["
        else
          echo "info [$rank] hi+lo channel-set-frozen line (required with --after-traffic): $(printf '%s\n' "$L" | grep -cF "hi+lo [")x"
        fi
      else
        never $rank "$L" "tf_fp8_w8a8: hi+lo ["
      fi
    elif [ "$W8A" = off ]; then
      if [ -n "$PREW8A" ]; then
        :   # a pre-r16y start.sh: the variable never reached the container; the never-lines below still apply
      elif [ "$W8AV" = 0 ]; then
        need $rank "$L" 1 "[glm53-w8a8] GLM53_DENSE_W8A8=0: stock"
      else
        need $rank "$L" 1 "[glm53-tf-bundle] patch_dense_w8a8.py: GLM53_DENSE_W8A8 unset -> skipped (stock)"
      fi
      never $rank "$L" "site-packages/fp8_w8a8.py: installed"
      never $rank "$L" "tf_fp8_w8a8 installed:"
      never $rank "$L" "[glm53-w8a8] integrate.py: armed"
    fi
  fi
  if [ "${1:-}" = --after-traffic ]; then
    [ "$REJ" != block ] || need $rank "$L" 1 "[glm53-block-keys] rejection_sample_method='block': acceptance, resample and bonus"
    [ "${ST[mla]}" != on ] || need $rank "$L" 1 "glm53_mla_prefill: 1 calls on the exact kernel"
    [ "${ST[quickwins]}" != on ] || need $rank "$L" 1 "glm53 prefill quickwins active: mla_bmm"
    [ "${ST[hostloop]}" != on ] || need $rank "$L" 1 "first step planned from the post-sampling snapshot"
    [ "$HL_WAKE" = off ] || need $rank "$L" 1 ": wake: SCHED_IDLE "     # the tickers start at the first decode step
  fi
done
# ---- APC short-suffix fix (head: scheduler / KV manager); the launcher's own "short-suffix APC: ..." line is on
# start.sh's stdout (restart2.sh output), not in the container log
need head "$H" 1 "DFlash2 drafter KV: exact-fit block=1152"
# the two APC lines follow the flags the head container actually got (0/1 values, not secrets): env.r16 = compact 1 +
# LRU 0; after `env_r16.sh off apc` / `apc-lru` (or the all-off loop of ROLLOUT.md section 5) the launcher defaults
# compact 0 / low priority 1 print boundary_group_ids=[] and a non-empty low_priority list instead
apc_compact=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_DRAFT_KV_COMPACT=' | tail -n 1 | cut -d= -f2-)
apc_low=$(docker inspect "$HEAD_C" --format "$fmt" | grep -E '^GLM53_APC_DRAFTER_LOW_PRIORITY=' | tail -n 1 | cut -d= -f2-)
echo "info [head] container APC flags: GLM53_DRAFT_KV_COMPACT=${apc_compact:-<unset>} GLM53_APC_DRAFTER_LOW_PRIORITY=${apc_low:-<unset>}"
if [ "$apc_compact" = 1 ]; then
  need head "$H" 1 "[glm53-dflash-boundary-lookup-v1] boundary_group_ids=[6]"
else
  need head "$H" 1 "[glm53-dflash-boundary-lookup-v1] boundary_group_ids=[]"
fi
if [ "$apc_low" = 0 ]; then
  need head "$H" 1 "low_priority=[]"
else
  never head "$H" "low_priority=[]"
fi
need head "$H" 1 "GPU KV cache size: 2,003,436 tokens"
m=$(curl -s -m 5 "http://127.0.0.1:${PORT}/metrics" | grep -E '^vllm:cache_config_info' | head -1)
case "$m" in *'block_size="1152"'*'num_gpu_blocks="583"'*|*'num_gpu_blocks="583"'*'block_size="1152"'*) echo "ok   [head] /metrics cache_config_info block_size 1152, num_gpu_blocks 583";;
  *) echo "MISS [head] /metrics cache_config_info block_size 1152 / num_gpu_blocks 583: $(echo "$m" | grep -oE '(block_size|num_gpu_blocks)="[0-9]+"' | tr '\n' ' ')"; fail=1;; esac
# ---- non-KV memory gate: MemAvailable on both nodes (R6 incident: 1.4 / 0.9 GiB -> 3.5 GB swapped out on nodeA)
a2=$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo); a3=$(ssh -o BatchMode=yes "$W" "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo")
s2=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%.2f", (t-f)/1048576}' /proc/meminfo)
s3=$(ssh -o BatchMode=yes "$W" "awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{printf \"%.2f\", (t-f)/1048576}' /proc/meminfo")
echo "info MemAvailable nodeA ${a2} GiB, nodeB ${a3} GiB; swap used nodeA ${s2} GiB, nodeB ${s3} GiB (compare with the pre-restart numbers of ROLLOUT.md step 1)"
for v in "$a2" "$a3"; do awk -v v="$v" -v m="${MIN_AVAIL_GIB:-1.0}" 'BEGIN{exit !(v >= m)}' || { echo "BAD  MemAvailable $v GiB < ${MIN_AVAIL_GIB:-1.0} GiB"; fail=1; }; done
echo "boot_checks: $([ $fail = 0 ] && echo ALL OK || echo PROBLEMS)"; exit $fail
