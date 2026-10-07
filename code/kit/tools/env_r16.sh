#!/usr/bin/env bash
# deploy-r16 (nodeA, operator): env-only switch of ONE R16 feature in .env (takes effect at the next restart of BOTH
# ranks: tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh <label>). Keeps .env.bak-r16env-<ts>. Names only printed.
#   tools/env_r16.sh off <feature>    remove the feature's lines
#   tools/env_r16.sh on  <feature>    add them back with the env.r16 values
# features: quickwins mla fp8roof moeglue hostloop smallops kpoolring apc apc-lru (apc-lru: only
#           GLM53_APC_DRAFTER_LOW_PRIORITY -> off sets it to 1 = the previous low-priority drafter window, APC doc 5.1)
#           kdaqkv (r16k, the env.r16 `#switch` GLM53_KDA_STRIDED_QKV; docs/KDA_STRIDED_QKV.md): off removes the line
#           (unset = stock: the KDA recurrent decode keeps its four .contiguous() copies per layer); on writes
#           GLM53_KDA_STRIDED_QKV=1 (vLLM #55736's strided q/k/v/beta) after checking, with .env unchanged on refusal,
#           that the installed start.sh forwards the knob to BOTH ranks (head -e + worker serve_env_names: the r16k kit;
#           r16j's start.sh does not forward it, the line would be silently ignored), overlay/tf/overlay/
#           patch_kda_strided_qkv.py and an overlay/patch_tf_bundle.py that runs it
#           mhcsp (r16msp, the env.r16 `#switch` GLM53_MHC_SP; docs/MHC_SP.md): off removes the line (unset =
#           stock: the mHC bookkeeping runs on the full batch on both ranks); on writes GLM53_MHC_SP=1
#           (sequence-parallel mHC prefill over TP: a >= 1024-token forward shards the residual stream, the
#           mHC family runs on T/2 rows, the attention/MLP all-reduces become RS+AG pairs, decode and every
#           CUDA-graph capture stay plain TP) after checking, with .env unchanged on refusal, that the
#           installed start.sh forwards the knob to BOTH ranks (head -e + worker serve_env_names: the r16msp
#           kit; r16l's start.sh does not forward it, the line would be silently ignored), overlay/tf/
#           overlay/patch_mhc_sp.py and an overlay/patch_tf_bundle.py that runs it
#           kpooldown (r16l, the env.r16 `#switch` GLM53_KPOOL_DROP_LOWEST; docs/KPOOL_DROP_LOWEST.md): off removes
#           the line (unset = stock: the kpool indexer drops an arbitrary pool of the top-k, because the top-k ops
#           return a deterministic SET in a nondeterministic ORDER); on writes GLM53_KPOOL_DROP_LOWEST=1 (keep the
#           select_k-1 HIGHEST-scored pools, deterministic) after checking, with .env unchanged on refusal, that the
#           installed start.sh forwards the knob to BOTH ranks (head -e + worker serve_env_names: the r16l kit;
#           r16k's start.sh does not forward it, the line would be silently ignored), overlay/tf/overlay/
#           patch_kpool_drop_lowest.py and an overlay/patch_tf_bundle.py that runs it
#           w8a8 (the env.r16 `#switch` GLM53_DENSE_W8A8; docs/DENSE_W8A8.md): off removes the line (unset =
#           stock: production's W8A16/Marlin dense path, the W8A8 module not even in site-packages); on writes
#           GLM53_DENSE_W8A8=1 (per-token fp8 activations x the stored per-channel fp8 weights, cutlass_scaled_mm,
#           dense + shared-expert FP8 linears, PREFILL only; decode keeps Marlin) after checking, with .env unchanged
#           on refusal, that the installed start.sh forwards the knob to BOTH ranks (head -e + worker
#           serve_env_names), overlay/tf/overlay/patch_dense_w8a8.py + the fp8_w8a8.py / tf_fp8_w8a8_ext*.so it
#           installs, and an overlay/patch_tf_bundle.py that runs it
#           moee4m3 (r16e4, the env.r16 `#switch` GLM53_MOE_E4M3; docs/MOE_E4M3.md): off removes the line (unset =
#           stock: the bundle does not run the overlay, nothing of the feature reaches site-packages); on writes
#           GLM53_MOE_E4M3=1 (routed-MoE apply calls above the fused cap run on the e4m3 kernels) after checking,
#           with .env unchanged on refusal, that the installed start.sh forwards the knob to BOTH ranks (head -e +
#           worker serve_env_names: the r16e4 kit; r16l's start.sh does not forward it, the line would be silently
#           ignored), overlay/tf/overlay/{patch_moe_e4m3.py,glm53_moe_e4m3.py,glm53_moe_e4m3_ext.cpython-312-
#           aarch64-linux-gnu.so} and an overlay/patch_tf_bundle.py that runs the patch
#           trace (measurement-only, docs/DEC_HOSTGAP.md §3: GLM53_DEC_TRACE, env.r16 carries NO value line for it):
#           on writes GLM53_DEC_TRACE=/root/.cache/vllm/glm53-dectrace (the vLLM cache directory both containers
#           bind-mount from their host: the files land in the head's $CACHE_ROOT/glm53-dectrace and the worker's
#           $WORKER_VLLM_CACHE/glm53-dectrace and survive the container) and GLM53_DEC_TRACE_MAX_STEPS=20000, and
#           REFUSES with .env unchanged unless the installed start.sh forwards both names to BOTH ranks (one head
#           `-e NAME="${NAME:-}"` line + an entry in the worker serve_env_names loop, like every other GLM53_ knob;
#           GLM53_EXTRA_ENV rejects GLM53_* names): the r16j start.sh does NOT (a start.sh built with the names in
#           make_start_sh.py KNOBS_DEFAULT does), so writing the lines there would only cost a restart of both ranks
#           for no trace. off removes every GLM53_DEC_TRACE* line. boot_checks ignores it (not a rollout feature).
#           mhcsp2 (r16z, the env.r16 `#switch` GLM53_MHC_SP2; docs/MHC_SP2.md): off removes the line (unset =
#           the r16x SP result: GLM53_MHC_SP alone); on writes GLM53_MHC_SP2=1 (the pipelined SP: k=2 interleaved
#           sub-chunks per rank shard, the collectives of each sub-chunk on a side stream; odd-T SP) after
#           checking, with .env unchanged on refusal, that the installed start.sh forwards the knob to BOTH ranks
#           (head -e + worker serve_env_names: the r16z kit; an r16x one would ignore the line),
#           overlay/tf/overlay/patch_mhc_sp2.py, an overlay/patch_tf_bundle.py that runs it, and
#           GLM53_MHC_SP=1 already in .env (the pipelined SP extends the r16x SP patch of the same model.py)
#           exactlens (r16z2x, the env.r16 `#switch` GLM53_MLA_EXACT_LENS; docs/MLA_EXACT_LENS.md): off removes the
#           line (unset = production's FA2 sparse-MLA plan, 4 keys per row past index_topk more than the indexer
#           selected); on writes GLM53_MLA_EXACT_LENS=1 after checking, with .env unchanged on refusal, that the
#           installed start.sh forwards the knob to BOTH ranks (the r16z2x kit), overlay/tf/overlay/
#           {patch_mla_exactlens.py,glm53_mla_exactlens.py} and an overlay/patch_tf_bundle.py that runs the patch
#           fkda3 (r16z, the env.r16 `#switch` GLM53_KDA_FLASHKDA_V; docs/KDA_FLASHKDA3.md): off removes the line
#           (unset = the shipped r16x FlashKDA build, production's bytes); on writes GLM53_KDA_FLASHKDA_V=3 (the
#           fkda3 build: glm53_flashkda3.py + _flashkda_fp32_C3.abi3.so stage under the production site names,
#           the kda.py call passes out=) after checking, with .env unchanged on refusal, that the installed
#           start.sh forwards GLM53_KDA_FLASHKDA_V to BOTH ranks, overlay/tf/overlay/{glm53_flashkda3.py,
#           _flashkda_fp32_C3.abi3.so, patch_flashkda.py} exist and a bundle script runs patch_flashkda.py, and
#           GLM53_KDA_FLASHKDA=1 already in .env. Switching the build on an installed tree is refused by the
#           patch (byte mismatch): FK=0 restart, then FK=1 with the new value
#           blockverify (r16j, the env.r16 `#switch` GLM53_REJECTION_METHOD): on writes GLM53_REJECTION_METHOD=block,
#           off removes it (unset = standard). `on` first checks what the kit's start.sh would check only after
#           restart2.sh has stopped both ranks (+ that the installed start.sh knows the switch at all), and refuses with
#           .env unchanged: SPEC_METHOD=dflash, GLM53_SPEC_RESAMPLE_INDEPENDENT=1, overlay/tf/overlay/
#           patch_spec_block_keys.py, an overlay/patch_tf_bundle.py that runs it, a start.sh that forwards the switch
#           fused16 (r16z3, the env.r16 `#switch` GLM53_MOE_FUSED16; docs/MOE3.md): off removes the line (unset =
#           stock: production's own grouped kernels; the bundle does not run the overlay, nothing of the feature
#           reaches site-packages); on writes GLM53_MOE_FUSED16=1 (every E3 grouped prefill call >= 4,096 tokens on
#           the P16 schedules: production's arithmetic, h2 bit-identical) after checking, with .env unchanged on
#           refusal, that the installed start.sh forwards the knob to BOTH ranks (head -e + worker serve_env_names:
#           the r16z3 kit; an r16z2 one would ignore the line), overlay/tf/overlay/{patch_moe_fused16.py,
#           glm53_moe_fused16.py, glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so} and an overlay/
#           patch_tf_bundle.py that runs it
#           e4acc (r16z4, the env.r16 `#switch` GLM53_MOE_E4M3_ACC; docs/OPT_MOE.md): off removes the line (unset =
#           the fp32 accumulator, production's arithmetic); on writes GLM53_MOE_E4M3_ACC=bf16 (the bf16 accumulator,
#           -0.71 s per 32k request; DEFAULT OFF: the MoE refutation of opt-moe is still running) after checking,
#           with .env unchanged on refusal, that the installed start.sh forwards the knob to BOTH ranks, the r16z4
#           overlay glm53_moe_e4m3.py reads it, and GLM53_MOE_E4M3=1 is already in .env (the value is only read there)
#           e4fold (r16z4, the env.r16 `#switch` GLM53_MOE_E4M3_FOLD_SHARED; docs/OPT_MOE.md): off removes the line
#           (unset = the routed output is added by vLLM as before); on writes GLM53_MOE_E4M3_FOLD_SHARED=1 (the routed
#           sum accumulates into the shared experts' bf16 output, -0.19 s per 32k request) after the e4acc checks PLUS
#           GLM53_MOE_E4M3_ACC=bf16 already in .env (the start.sh refuses =1 without it, and without the bf16
#           accumulator the module refuses the whole e4m3 install). boot_checks MISSes without the
#           "GLM53_MOE_E4M3_FOLD_SHARED=1 (routed sum accumulated into the shared experts' output)" summary suffix
#           fp8ag (r16z4, the env.r16 `#switch` GLM53_DENSE_W8A8_FP8AG; docs/OPT_DENSE.md): off removes the line
#           (unset = the bf16 gather, production's bytes); on writes GLM53_DENSE_W8A8_FP8AG=1 (the sequence-parallel
#           attention gather of a served KDA in_proj carries per-token fp8 + scales, half the bytes; a PAIRED
#           collective - boot_checks pair-gates "FP8 all-gather installed" on BOTH ranks BEFORE traffic, and the
#           opt-dense-rev refusal ERROR is a MISS) after checking, with .env unchanged on refusal, that
#           GLM53_DENSE_W8A8=1 is already in .env (the value is only read there), the installed start.sh forwards the
#           knob to BOTH ranks, and the r16z4 overlay fp8_w8a8.py reads it
#           kdalazy (r16z6, the env.r16 `#knob` GLM53_DEC_KDA_LAZY + _VERIFY + _VERIFY_EVERY; docs/DEC_KDA_LAZY.md):
#           off removes the three lines (unset = stock: the KDA verify keeps production's per-row state stores); on
#           writes GLM53_DEC_KDA_LAZY=1 (the site module glm53_kda_lazy.py: ONE KDA recurrent-state store per verify
#           step; the dials stay at the module's defaults, the first 64 commits then 1/1024) after checking, with
#           .env unchanged on refusal, that the installed start.sh forwards the knobs to BOTH ranks and the bundle
#           site/ carries the module. boot_checks NEEDs the module's install line on both ranks and treats the
#           'self-check ... differ' line as a refusal (repair mode is exact but ~5 ms/step SLOWER: roll back)
#           vtrim (r16z6, the env.r16 `#knob` GLM53_DEC_VTRIM_STATS + _CAP + _FILE; docs/OPT_DECODE.md 4): off
#           removes the three lines (unset = off); on writes GLM53_DEC_VTRIM_STATS=2000 (the log-only collector:
#           16 probabilities/counts per request and verify step, no token ids, a .npy ring in the bind-mounted vLLM
#           cache dir; outputs untouched) after the same start.sh/site checks. A measurement-only experiment knob:
#           run a few hours, copy the .npy, evaluate with tests/optdec/vtrim_policy_from_stats.py
#           specvtrim / specvtrim-shadow (r16z6sv, the env.r16 `#knob` GLM53_SPEC_VTRIM + _TAU + _MIN + _LOG;
#           docs/SPEC_VTRIM.md): off removes the four lines (unset = production); on writes GLM53_SPEC_VTRIM=on
#           (per-step verify trimming from the DFlash2 drafter's confidence, TAU 0.25 unless hand-edited), shadow
#           writes GLM53_SPEC_VTRIM=shadow (log-only: what trimming would do), after the same start.sh/site checks
#           kpooltail (r16z7, the env.r16 `#switch` GLM53_KPOOL_TAIL_POSITIONS; docs/PREFIX_HIT_TAIL.md): off removes the
#           line (unset = stock: one kpool tail ring shared by all running requests + stale warm-up block ids that
#           overwrite pooled indexer keys); on writes GLM53_KPOOL_TAIL_POSITIONS=2 (per-request circular tail slots
#           written into the persistent slot buffer, FULL-graph safe; the overlay's value 1 is never written - start.sh
#           refuses it) after checking, with .env unchanged on refusal, overlay/tf/overlay/patch_kpool_tail_positions.py,
#           an overlay/patch_tf_bundle.py that runs it and a start.sh that forwards the switch to BOTH ranks (a one-rank
#           fix silently degrades quality: boot_checks requires the patched lines on both ranks)
#           mambaseed (r16z7, the env.r16 `#switch` GLM53_MAMBA_ALIGN_SEED; docs/PREFIX_HIT_TAIL.md): off removes the
#           line (unset = stock); on writes GLM53_MAMBA_ALIGN_SEED=1 (prefix-hit KDA seed column in mamba blocks;
#           byte-identical in today's multiprocess workers = hardening), with the same three checks
#           ar1shot / ar1shot-verify (r16z6ar, the env.r16 `#knob` GLM53_DEC_AR1SHOT + _MAX_KB + _LOG;
#           docs/DEC_AR1SHOT.md): off removes the three lines (unset = production); on writes GLM53_DEC_AR1SHOT=1 (a
#           2-rank decode all-reduce becomes all-gather + one bf16 add: one network hop, bit-identical sum),
#           ar1shot-verify writes GLM53_DEC_AR1SHOT=verify (serve production's all-reduce, count differing elements),
#           after the same start.sh/site checks. A PAIRED collective: both ranks must carry the same value
#           (the kit's start.sh forwards it to both; boot_checks compares head and worker)
#           dlmh / dlmh-verify (r16z6dl, the env.r16 `#knob` GLM53_DEC_DLMH + _C + _GROUP + _LOG; docs/DEC_DLMH.md):
#           off removes the four lines (unset = production); on writes GLM53_DEC_DLMH=1 (the drafter's two-stage
#           candidate head), dlmh-verify writes GLM53_DEC_DLMH=verify (serve production's candidates, log how often the
#           two-stage head would differ), after the same start.sh/site checks
#           planpin (r16z8p, the env.r16 `#knob` GLM53_MLA_PLAN_PIN; docs/MLA_PLAN_PIN.md): off removes the line (unset =
#           production); on writes GLM53_MLA_PLAN_PIN=1 (the sparse-MLA plan's CPU staging page-locked in a 2-slot ring:
#           no host stall on the drafter graph, same device bytes), after the same start.sh/site checks
#           (tokgather =0, GLM53_MLA_PREFILL_FUSED_INDEX=0, GLM53_DENSE_W8A8_HILO / _SEL are hand-edited like the
#           layer lists / GLM53_DENSE_W8A8_ONLY: the start.sh validates every value before any rank is stopped)
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="${LAUNCHER_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
op="${1:?on|off}"; feat="${2:?feature}"
case "$op" in on|off) ;; *) echo "env_r16: first argument must be on or off (got: $op); .env not changed"; exit 2;; esac
case "$feat" in
  quickwins) names="GLM53_PREFILL_QUICKWINS";;
  mla)       names="GLM53_MLA_PREFILL";;
  fp8roof)   names="GLM53_DEC_FP8ROOF";;
  moeglue)   names="GLM53_DEC_MOEGLUE_WARM";;
  ar1shot|ar1shot-verify) names="GLM53_DEC_AR1SHOT GLM53_DEC_AR1SHOT_MAX_KB GLM53_DEC_AR1SHOT_LOG";;   # operator knobs (env.r16 #knob, r16z6ar), not env.r16 lines
  hostloop)  names="GLM53_DEC_HOSTLOOP";;
  smallops)  names="GLM53_DEC_SMALLOPS GLM53_DEC_SMALLOPS_KINDS";;   # both: KINDS empty = every kind (+66.75 MiB mhc)
  kpoolring) names="GLM53_KPOOL_RING";;
  apc)       names="GLM53_DRAFT_KV_COMPACT GLM53_APC_DRAFTER_LOW_PRIORITY";;
  apc-lru)   names="GLM53_APC_DRAFTER_LOW_PRIORITY";;
  kdaqkv)    names="GLM53_KDA_STRIDED_QKV";;   # operator switch (env.r16 #switch, r16k), not an env.r16 line
  kpooldown) names="GLM53_KPOOL_DROP_LOWEST";;   # operator switch (env.r16 #switch, r16l), not an env.r16 line
  w8a8)      names="GLM53_DENSE_W8A8";;   # operator switch (env.r16 #switch, r16y), not an env.r16 line
  flashkda)  names="GLM53_KDA_FLASHKDA";;   # operator switch (env.r16 #switch, r16n), not an env.r16 line
  mhcsp)     names="GLM53_MHC_SP"            # operator switch (env.r16 #switch, r16msp), not an env.r16 line
             # [opt-decodekit] off also removes GLM53_MHC_SP2: SP2=1 without SP=1 is refused by start.sh's
             # validate_numeric_config, i.e. AFTER restart2.sh has stopped both ranks (an outage, not a rollback)
             [ "$op" = off ] && names="GLM53_MHC_SP GLM53_MHC_SP2";;
  moee4m3)   names="GLM53_MOE_E4M3";;   # operator switch (env.r16 #switch, r16e4), not an env.r16 line
  mhcsp2)    names="GLM53_MHC_SP2";;    # operator switch (env.r16 #switch, r16z), not an env.r16 line
  fkda3)     names="GLM53_KDA_FLASHKDA_V";;   # operator switch (env.r16 #switch, r16z), not an env.r16 line
  fused16)   names="GLM53_MOE_FUSED16";;   # operator switch (env.r16 #switch, r16z3), not an env.r16 line
  e4acc)     names="GLM53_MOE_E4M3_ACC";;   # operator switch (env.r16 #switch, r16z4 moe-opt), not an env.r16 line
  e4fold)    names="GLM53_MOE_E4M3_FOLD_SHARED";;   # operator switch (env.r16 #switch, r16z4 moe-opt)
  fp8ag)     names="GLM53_DENSE_W8A8_FP8AG";;   # operator switch (env.r16 #switch, r16z4 w8a8-fp8ag)
  exactlens) names="GLM53_MLA_EXACT_LENS";;   # operator switch (env.r16 #switch, opt-decodekit), not an env.r16 line
  blockverify) names="GLM53_REJECTION_METHOD";;   # operator switch (env.r16 #switch), not an env.r16 line
  trace)     names="GLM53_DEC_TRACE GLM53_DEC_TRACE_RING GLM53_DEC_TRACE_FLUSH GLM53_DEC_TRACE_MAX_STEPS GLM53_DEC_TRACE_EVENTS GLM53_DEC_TRACE_GPU";;
  kdalazy)   names="GLM53_DEC_KDA_LAZY GLM53_DEC_KDA_LAZY_VERIFY GLM53_DEC_KDA_LAZY_VERIFY_EVERY";;   # operator knobs (env.r16 #knob, r16z6), not env.r16 lines
  vtrim)     names="GLM53_DEC_VTRIM_STATS GLM53_DEC_VTRIM_STATS_CAP GLM53_DEC_VTRIM_STATS_FILE";;   # operator knobs (env.r16 #knob, r16z6), not env.r16 lines
  specvtrim|specvtrim-shadow) names="GLM53_SPEC_VTRIM GLM53_SPEC_VTRIM_TAU GLM53_SPEC_VTRIM_MIN GLM53_SPEC_VTRIM_LOG";;   # operator knobs (env.r16 #knob, r16z6sv), not env.r16 lines
  kpooltail) names="GLM53_KPOOL_TAIL_POSITIONS";;   # operator switch (env.r16 #switch, r16z7), not an env.r16 line
  mambaseed) names="GLM53_MAMBA_ALIGN_SEED";;       # operator switch (env.r16 #switch, r16z7), not an env.r16 line
  dlmh|dlmh-verify) names="GLM53_DEC_DLMH GLM53_DEC_DLMH_C GLM53_DEC_DLMH_GROUP GLM53_DEC_DLMH_LOG";;   # operator knobs (env.r16 #knob, r16z6dl), not env.r16 lines
  planpin)   names="GLM53_MLA_PLAN_PIN";;           # operator knob (env.r16 #knob, r16z8p), not an env.r16 line
  *) echo "unknown feature $feat"; exit 2;;
esac
if [ "$feat" = blockverify ] && [ "$op" = on ]; then
  envval() { grep -E "^$1=" "$L/.env" | tail -n 1 | cut -d= -f2- | sed -E "s/^[\"']//; s/[\"']$//; s/[[:space:]]+#.*$//"; }
  miss=""
  [ "$(envval SPEC_METHOD || true)" = dflash ] || miss+="SPEC_METHOD=dflash in .env; "
  [ "$(envval GLM53_SPEC_RESAMPLE_INDEPENDENT || true)" = 1 ] || miss+="GLM53_SPEC_RESAMPLE_INDEPENDENT=1 in .env; "
  [ -s "$L/overlay/tf/overlay/patch_spec_block_keys.py" ] || miss+="overlay/tf/overlay/patch_spec_block_keys.py (the r16j kit); "
  grep -qF '("patch_spec_block_keys.py", "GLM53_REJECTION_METHOD")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_spec_block_keys.py (the r16j kit); "
  [ "$(grep -cF 'os.environ.get("GLM53_REJECTION_METHOD") or "standard"' "$L/start.sh" 2>/dev/null || true)" = 2 ] \
    || miss+="a start.sh that forwards GLM53_REJECTION_METHOD (the r16j kit; an older one would run standard); "
  [ -z "$miss" ] || { echo "env_r16: blockverify on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = kdaqkv ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_kda_strided_qkv.py" ] || miss+="overlay/tf/overlay/patch_kda_strided_qkv.py (the r16k kit); "
  grep -qF '("patch_kda_strided_qkv.py", "GLM53_KDA_STRIDED_QKV")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_kda_strided_qkv.py (the r16k kit); "
  grep -qxF '        -e GLM53_KDA_STRIDED_QKV="${GLM53_KDA_STRIDED_QKV:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_KDA_STRIDED_QKV \
    || miss+="a start.sh that forwards GLM53_KDA_STRIDED_QKV to both ranks (the r16k kit; r16j's would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: kdaqkv on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = kpooldown ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_kpool_drop_lowest.py" ] || miss+="overlay/tf/overlay/patch_kpool_drop_lowest.py (the r16l kit); "
  grep -qF '("patch_kpool_drop_lowest.py", "GLM53_KPOOL_DROP_LOWEST")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_kpool_drop_lowest.py (the r16l kit); "
  grep -qxF '        -e GLM53_KPOOL_DROP_LOWEST="${GLM53_KPOOL_DROP_LOWEST:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_KPOOL_DROP_LOWEST \
    || miss+="a start.sh that forwards GLM53_KPOOL_DROP_LOWEST to both ranks (the r16l kit; r16k's would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: kpooldown on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = w8a8 ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_dense_w8a8.py" ] || miss+="overlay/tf/overlay/patch_dense_w8a8.py (the r16y kit); "
  [ -s "$L/overlay/tf/overlay/fp8_w8a8.py" ] || miss+="overlay/tf/overlay/fp8_w8a8.py (the module the patch installs; the r16y kit); "
  [ -s "$L/overlay/tf/overlay/tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so" ] || miss+="overlay/tf/overlay/tf_fp8_w8a8_ext.cpython-312-aarch64-linux-gnu.so (the AOT kernel; the r16y kit); "
  grep -qF '("patch_dense_w8a8.py", "GLM53_DENSE_W8A8")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_dense_w8a8.py (the r16y patch_tf_bundle.py); "
  grep -qxF '        -e GLM53_DENSE_W8A8="${GLM53_DENSE_W8A8:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_DENSE_W8A8 \
    || miss+="a start.sh that forwards GLM53_DENSE_W8A8 to both ranks (the r16y kit; an r16n one would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: w8a8 on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = flashkda ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_flashkda.py" ] || miss+="overlay/tf/overlay/patch_flashkda.py (the r16n kit); "
  grep -qF '("patch_flashkda.py", "GLM53_KDA_FLASHKDA")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_flashkda.py (the r16n kit); "
  grep -qxF '        -e GLM53_KDA_FLASHKDA="${GLM53_KDA_FLASHKDA:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_KDA_FLASHKDA \
    || miss+="a start.sh that forwards GLM53_KDA_FLASHKDA to both ranks (the r16n kit; r16l's would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: flashkda on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi

if [ "$feat" = mhcsp ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_mhc_sp.py" ] || miss+="overlay/tf/overlay/patch_mhc_sp.py (the r16msp kit); "
  grep -qF '("patch_mhc_sp.py", "GLM53_MHC_SP")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_mhc_sp.py (the r16msp kit); "
  grep -qxF '        -e GLM53_MHC_SP="${GLM53_MHC_SP:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MHC_SP \
    || miss+="a start.sh that forwards GLM53_MHC_SP to both ranks (the r16msp kit; r16l's would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: mhcsp on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi

if [ "$feat" = moee4m3 ] && [ "$op" = on ]; then
  miss=""
  for f in patch_moe_e4m3.py glm53_moe_e4m3.py glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so; do
    [ -s "$L/overlay/tf/overlay/$f" ] || miss+="overlay/tf/overlay/$f (the r16e4 kit); "
  done
  grep -qF '("patch_moe_e4m3.py", "GLM53_MOE_E4M3")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_moe_e4m3.py (the r16e4 kit); "
  grep -qxF '        -e GLM53_MOE_E4M3="${GLM53_MOE_E4M3:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MOE_E4M3 \
    || miss+="a start.sh that forwards GLM53_MOE_E4M3 to both ranks (the r16e4 kit; r16l's would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: moee4m3 on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = mhcsp2 ] && [ "$op" = on ]; then
  miss=""
  [ -s "$L/overlay/tf/overlay/patch_mhc_sp2.py" ] || miss+="overlay/tf/overlay/patch_mhc_sp2.py (the r16z kit); "
  grep -qF '("patch_mhc_sp2.py", "GLM53_MHC_SP2")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_mhc_sp2.py (the r16z kit); "
  grep -qxF '        -e GLM53_MHC_SP2="${GLM53_MHC_SP2:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MHC_SP2 \
    || miss+="a start.sh that forwards GLM53_MHC_SP2 to both ranks (the r16z kit; an r16x one would ignore the line); "
  grep -qE '^GLM53_MHC_SP=1$' "$L/.env" \
    || miss+="GLM53_MHC_SP=1 in .env first (env_r16.sh on mhcsp): GLM53_MHC_SP2=1 extends the r16x SP patch of the same model.py and is refused by the start.sh and the patch without it; "
  [ -z "$miss" ] || { echo "env_r16: mhcsp2 on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = fkda3 ] && [ "$op" = on ]; then
  miss=""
  for f in glm53_flashkda3.py _flashkda_fp32_C3.abi3.so patch_flashkda.py; do
    [ -s "$L/overlay/tf/overlay/$f" ] || miss+="overlay/tf/overlay/$f (the r16z kit); "
  done
  grep -qF '("patch_flashkda.py", "GLM53_KDA_FLASHKDA")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_flashkda.py (the r16n kit); "
  grep -qxF '        -e GLM53_KDA_FLASHKDA_V="${GLM53_KDA_FLASHKDA_V:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_KDA_FLASHKDA_V \
    || miss+="a start.sh that forwards GLM53_KDA_FLASHKDA_V to both ranks (the r16z kit; an r16x one would ignore the line); "
  grep -qE '^GLM53_KDA_FLASHKDA=1$' "$L/.env" \
    || miss+="GLM53_KDA_FLASHKDA=1 in .env: GLM53_KDA_FLASHKDA_V only selects the build the =1 patch installs; "
  [ -z "$miss" ] || { echo "env_r16: fkda3 on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = exactlens ] && [ "$op" = on ]; then
  miss=""
  for f in patch_mla_exactlens.py glm53_mla_exactlens.py; do
    [ -s "$L/overlay/tf/overlay/$f" ] || miss+="overlay/tf/overlay/$f (the r16z2x kit); "
  done
  grep -qF '("patch_mla_exactlens.py", "GLM53_MLA_EXACT_LENS")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_mla_exactlens.py (the r16z2x kit); "
  grep -qxF '        -e GLM53_MLA_EXACT_LENS="${GLM53_MLA_EXACT_LENS:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MLA_EXACT_LENS \
    || miss+="a start.sh that forwards GLM53_MLA_EXACT_LENS to both ranks (the r16z2x kit; an r16z2 one would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: exactlens on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
TRACE_ON="GLM53_DEC_TRACE=/root/.cache/vllm/glm53-dectrace GLM53_DEC_TRACE_MAX_STEPS=20000"
if [ "$feat" = trace ] && [ "$op" = on ]; then   # refuse BEFORE the backup/edit when the lines could not reach a rank
  miss=""
  wloop=$(awk '/^[[:space:]]*for v in SERVED_MODEL_NAME /{f=1} f{print} f&&/; do/{exit}' "$L/start.sh" 2>/dev/null || true)
  for kv in $TRACE_ON; do
    n="${kv%%=*}"
    [ "$(grep -cF -- "-e $n=\"\${$n:-}\"" "$L/start.sh" 2>/dev/null || true)" = 1 ] || miss+="$n (head -e line); "
    grep -qw -- "$n" <<< "$wloop" || miss+="$n (worker serve_env_names loop); "
  done
  [ -z "$miss" ] || { echo "env_r16: trace on refused, .env not changed: the installed start.sh does not forward ${miss%; } (the r16j kit's start.sh forwards no GLM53_DEC_TRACE* name; a kit whose make_start_sh.py KNOBS_DEFAULT lists them does)"; exit 2; }
fi
if [ "$feat" = fused16 ] && [ "$op" = on ]; then
  miss=""
  for f in patch_moe_fused16.py glm53_moe_fused16.py glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so; do
    [ -s "$L/overlay/tf/overlay/$f" ] || miss+="overlay/tf/overlay/$f (the r16z3 kit); "
  done
  grep -qF '("patch_moe_fused16.py", "GLM53_MOE_FUSED16")' "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs patch_moe_fused16.py (the r16z3 kit); "
  grep -qxF '        -e GLM53_MOE_FUSED16="${GLM53_MOE_FUSED16:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MOE_FUSED16 \
    || miss+="a start.sh that forwards GLM53_MOE_FUSED16 to both ranks (the r16z3 kit; an r16z2 one would ignore the line); "
  [ -z "$miss" ] || { echo "env_r16: fused16 on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = e4acc ] && [ "$op" = on ]; then
  miss=""
  grep -qF "GLM53_MOE_E4M3_ACC" "$L/overlay/tf/overlay/glm53_moe_e4m3.py" 2>/dev/null \
    || miss+="an r16z4 overlay/tf/overlay/glm53_moe_e4m3.py that reads GLM53_MOE_E4M3_ACC (an older one would silently serve the fp32 accumulator); "
  grep -qxF '        -e GLM53_MOE_E4M3_ACC="${GLM53_MOE_E4M3_ACC:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MOE_E4M3_ACC \
    || miss+="a start.sh that forwards GLM53_MOE_E4M3_ACC to both ranks (the r16z4 kit; an r16z3 one would ignore the line); "
  grep -qE '^GLM53_MOE_E4M3=1$' "$L/.env" \
    || miss+="GLM53_MOE_E4M3=1 in .env first (env_r16.sh on moee4m3): the accumulator dial is only read there; "
  [ -z "$miss" ] || { echo "env_r16: e4acc on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = e4fold ] && [ "$op" = on ]; then
  miss=""
  grep -qF "GLM53_MOE_E4M3_FOLD_SHARED" "$L/overlay/tf/overlay/glm53_moe_e4m3.py" 2>/dev/null \
    || miss+="an r16z4 overlay/tf/overlay/glm53_moe_e4m3.py that reads GLM53_MOE_E4M3_FOLD_SHARED (an older one would silently run the unfolded path); "
  grep -qxF '        -e GLM53_MOE_E4M3_FOLD_SHARED="${GLM53_MOE_E4M3_FOLD_SHARED:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_MOE_E4M3_FOLD_SHARED \
    || miss+="a start.sh that forwards GLM53_MOE_E4M3_FOLD_SHARED to both ranks (the r16z4 kit); "
  grep -qE '^GLM53_MOE_E4M3_ACC=bf16$' "$L/.env" \
    || miss+="GLM53_MOE_E4M3_ACC=bf16 in .env first (env_r16.sh on e4acc): the fold REQUIRES the bf16 accumulator - the start.sh refuses =1 without it (the module would refuse the whole e4m3 install); "
  [ -z "$miss" ] || { echo "env_r16: e4fold on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
if [ "$feat" = fp8ag ] && [ "$op" = on ]; then
  miss=""
  grep -qF "GLM53_DENSE_W8A8_FP8AG" "$L/overlay/tf/overlay/fp8_w8a8.py" 2>/dev/null \
    || miss+="an r16z4 overlay/tf/overlay/fp8_w8a8.py that reads GLM53_DENSE_W8A8_FP8AG (an older one would silently keep the bf16 gather); "
  grep -qxF '        -e GLM53_DENSE_W8A8_FP8AG="${GLM53_DENSE_W8A8_FP8AG:-}" \' "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw GLM53_DENSE_W8A8_FP8AG \
    || miss+="a start.sh that forwards GLM53_DENSE_W8A8_FP8AG to both ranks (the r16z4 kit); "
  grep -qE '^GLM53_DENSE_W8A8=1$' "$L/.env" \
    || miss+="GLM53_DENSE_W8A8=1 in .env first (env_r16.sh on w8a8): the fp8 all-gather is only read there - a PAIRED collective, both ranks must carry the same value and the same W8A8 install; "
  [ -z "$miss" ] || { echo "env_r16: fp8ag on refused, .env not changed; missing: ${miss%; }"; exit 2; }
fi
for feat_blk in kdalazy vtrim specvtrim specvtrim-shadow ar1shot ar1shot-verify dlmh dlmh-verify planpin; do   # [r16z6] the site-module knobs: refuse BEFORE the backup/edit when the value
  [ "$feat" = "$feat_blk" ] && [ "$op" = on ] || continue   # could not reach a rank (like trace/fp8ag above)
  miss=""
  case "$feat_blk" in
    kdalazy) mod="glm53_kda_lazy.py"; names_blk="GLM53_DEC_KDA_LAZY GLM53_DEC_KDA_LAZY_VERIFY GLM53_DEC_KDA_LAZY_VERIFY_EVERY";;
    vtrim)   mod="glm53_vtrim_stats.py"; names_blk="GLM53_DEC_VTRIM_STATS GLM53_DEC_VTRIM_STATS_CAP GLM53_DEC_VTRIM_STATS_FILE";;
    specvtrim|specvtrim-shadow) mod="glm53_spec_vtrim.py"; names_blk="GLM53_SPEC_VTRIM GLM53_SPEC_VTRIM_TAU GLM53_SPEC_VTRIM_MIN GLM53_SPEC_VTRIM_LOG";;
    ar1shot|ar1shot-verify) mod="glm53_ar1shot.py"; names_blk="GLM53_DEC_AR1SHOT GLM53_DEC_AR1SHOT_MAX_KB GLM53_DEC_AR1SHOT_LOG";;
    dlmh|dlmh-verify) mod="glm53_dlmh.py"; names_blk="GLM53_DEC_DLMH GLM53_DEC_DLMH_C GLM53_DEC_DLMH_GROUP GLM53_DEC_DLMH_LOG";;
    planpin) mod="glm53_mla_planpin.py"; names_blk="GLM53_MLA_PLAN_PIN";;
  esac
  [ -s "$L/overlay/tf/site/$mod" ] || miss+="overlay/tf/site/$mod (the bundle site/ this kit ships; the module is a site py_module, not an overlay install); "
  case "$feat_blk" in   # [r16z8] the dlmh module serves through its AOT kernel, shipped next to it in site/
    dlmh|dlmh-verify) [ -s "$L/overlay/tf/site/tf_dlmh_ext.cpython-312-aarch64-linux-gnu.so" ] \
      || miss+="overlay/tf/site/tf_dlmh_ext.cpython-312-aarch64-linux-gnu.so (the DLMH kernel the r16z6dl / r16z8 kit ships in site/); ";;
  esac
  for n in $names_blk; do
    grep -qxF "        -e $n=\"\${$n:-}\" \\" "$L/start.sh" 2>/dev/null \
      && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw "$n" \
      || miss+="$n (the installed start.sh does not forward it to both ranks; the r16z6 kit's does); "
  done
  [ -z "$miss" ] || { echo "env_r16: $feat on refused, .env not changed; missing: ${miss%; }"; exit 2; }
done
for feat_blk in kpooltail mambaseed; do   # [r16z7] the two bundle-overlay switches: refuse BEFORE the backup/edit when
  [ "$feat" = "$feat_blk" ] && [ "$op" = on ] || continue   # the value could not reach BOTH ranks' bundle
  case "$feat_blk" in
    kpooltail) pf="patch_kpool_tail_positions.py"; n="GLM53_KPOOL_TAIL_POSITIONS";;
    mambaseed) pf="patch_mamba_align_seed.py"; n="GLM53_MAMBA_ALIGN_SEED";;
  esac
  miss=""
  [ -s "$L/overlay/tf/overlay/$pf" ] || miss+="overlay/tf/overlay/$pf (the r16z7 kit); "
  grep -qF "(\"$pf\", \"$n\")" "$L/overlay/patch_tf_bundle.py" 2>/dev/null \
    || miss+="an overlay/patch_tf_bundle.py that runs $pf (the r16z7 kit); "
  grep -qxF "        -e $n=\"\${$n:-}\" \\" "$L/start.sh" 2>/dev/null \
    && sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' "$L/start.sh" | grep -qw "$n" \
    || miss+="a start.sh that forwards $n to both ranks (the r16z7 kit; an older one would ignore the line on BOTH ranks or, hand-edited, on one); "
  [ -z "$miss" ] || { echo "env_r16: $feat on refused, .env not changed; missing: ${miss%; }"; exit 2; }
done
ENVBAK="$L/.env.bak-r16env-$(date +%Y%m%d-%H%M%S)"
cp -p "$L/.env" "$ENVBAK"
# [opt-decodekit] every edit below is checked against the INSTALLED start.sh's own validate_numeric_config (pure reads:
# value checks, overlay files, the bundle registrations) with the new .env: a .env that start.sh would refuse is only
# refused after restart2.sh has stopped both ranks. Refused -> .env restored byte for byte from the backup, rc 2.
# A start.sh without validate_numeric_config (older than R16) or whose last line is not `main "$@"` -> info, no check.
launcher_check() {
  local t out rc=0
  if ! grep -q '^validate_numeric_config() {' "$L/start.sh" 2>/dev/null || [ "$(tail -n 1 "$L/start.sh")" != 'main "$@"' ]; then
    echo "env_r16: info: the installed start.sh has no validate_numeric_config/main tail; the edited .env was not pre-validated"
    return 0
  fi
  t=$(mktemp -d "$L/.env_r16-check.XXXXXX"); chmod 700 "$t"   # next to .env (same owner/fs), removed below
  sed '$d' "$L/start.sh" > "$t/check.sh"
  printf '%s\n' 'validate_numeric_config && echo "env_r16-launcher-check: OK"' >> "$t/check.sh"
  cp -p "$L/.env" "$t/.env"; chmod 600 "$t/.env"
  out=$(cd "$t" && env -i PATH="$PATH" HOME="${HOME:-/tmp}" PYTHONDONTWRITEBYTECODE=1 \
        TF_BUNDLE_PATCH_HOST="$L/overlay/patch_tf_bundle.py" \
        TF_BUNDLE_DIR_HOST="$L/overlay/tf" timeout 120 bash "$t/check.sh" 2>&1) || true
  rm -rf "$t"
  if ! printf '%s\n' "$out" | grep -qx 'env_r16-launcher-check: OK'; then
    cp -p "$ENVBAK" "$L/.env"
    echo "env_r16: $feat $op refused, .env restored: the installed start.sh's validate_numeric_config refuses the edited .env (it would refuse it only after restart2.sh has stopped both ranks):"
    printf '%s\n' "$out" | grep -vE '^$' | sed -E 's#(key|token|secret|password)([^ =:]*)[=:][^ ]*#\1\2=***#gi' | tail -n 5 | sed 's/^/  /'
    exit 2
  fi
}
for n in $names; do sed -i -E "/^$n=/d" "$L/.env"; done
if [ "$op" = on ] && [ "$feat" != blockverify ] && [ "$feat" != trace ]; then
  [ -z "$(tail -c1 "$L/.env")" ] || echo >> "$L/.env"
  if [ "$feat" = kdaqkv ]; then   # operator switch: env.r16 carries only `#switch GLM53_KDA_STRIDED_QKV 0|1`
    echo "GLM53_KDA_STRIDED_QKV=1" >> "$L/.env"
  elif [ "$feat" = kpooldown ]; then   # operator switch: env.r16 carries only `#switch GLM53_KPOOL_DROP_LOWEST 0|1`
    echo "GLM53_KPOOL_DROP_LOWEST=1" >> "$L/.env"
  elif [ "$feat" = w8a8 ]; then   # operator switch: env.r16 carries only `#switch GLM53_DENSE_W8A8 0|1`
    echo "GLM53_DENSE_W8A8=1" >> "$L/.env"
  elif [ "$feat" = flashkda ]; then   # operator switch: env.r16 carries only `#switch GLM53_KDA_FLASHKDA 0|1`
    echo "GLM53_KDA_FLASHKDA=1" >> "$L/.env"
  elif [ "$feat" = mhcsp ]; then       # operator switch: env.r16 carries only `#switch GLM53_MHC_SP 0|1`
    echo "GLM53_MHC_SP=1" >> "$L/.env"
  elif [ "$feat" = moee4m3 ]; then   # operator switch: env.r16 carries only `#switch GLM53_MOE_E4M3 0|1`
    echo "GLM53_MOE_E4M3=1" >> "$L/.env"
  elif [ "$feat" = mhcsp2 ]; then    # operator switch: env.r16 carries only `#switch GLM53_MHC_SP2 0|1`
    echo "GLM53_MHC_SP2=1" >> "$L/.env"
  elif [ "$feat" = fkda3 ]; then     # operator switch: env.r16 carries only `#switch GLM53_KDA_FLASHKDA_V 1|2|3`
    echo "GLM53_KDA_FLASHKDA_V=3" >> "$L/.env"
  elif [ "$feat" = fused16 ]; then   # operator switch: env.r16 carries only `#switch GLM53_MOE_FUSED16 0|1`
    echo "GLM53_MOE_FUSED16=1" >> "$L/.env"
  elif [ "$feat" = e4acc ]; then   # operator switch: env.r16 carries only `#switch GLM53_MOE_E4M3_ACC f32|bf16`
    echo "GLM53_MOE_E4M3_ACC=bf16" >> "$L/.env"
  elif [ "$feat" = e4fold ]; then   # operator switch: env.r16 carries only `#switch GLM53_MOE_E4M3_FOLD_SHARED 0|1`
    echo "GLM53_MOE_E4M3_FOLD_SHARED=1" >> "$L/.env"
  elif [ "$feat" = fp8ag ]; then   # operator switch: env.r16 carries only `#switch GLM53_DENSE_W8A8_FP8AG 0|1`
    echo "GLM53_DENSE_W8A8_FP8AG=1" >> "$L/.env"
  elif [ "$feat" = kdalazy ]; then   # operator knob: env.r16 carries only `#knob GLM53_DEC_KDA_LAZY*`; the dials stay at the module defaults
    echo "GLM53_DEC_KDA_LAZY=1" >> "$L/.env"
  elif [ "$feat" = vtrim ]; then   # operator knob: env.r16 carries only `#knob GLM53_DEC_VTRIM_STATS*`; 2000 = the collector doc's suggested cadence
    echo "GLM53_DEC_VTRIM_STATS=2000" >> "$L/.env"
  elif [ "$feat" = specvtrim ]; then   # operator knob (r16z6sv): trimming on at the module's default TAU (0.25); TAU/MIN/LOG are hand-edited
    echo "GLM53_SPEC_VTRIM=on" >> "$L/.env"
  elif [ "$feat" = specvtrim-shadow ]; then   # operator knob (r16z6sv): log-only, outputs untouched
    echo "GLM53_SPEC_VTRIM=shadow" >> "$L/.env"
  elif [ "$feat" = kpooltail ]; then # operator switch (r16z7): env.r16 carries only `#switch GLM53_KPOOL_TAIL_POSITIONS 0|2`
    echo "GLM53_KPOOL_TAIL_POSITIONS=2" >> "$L/.env"
  elif [ "$feat" = mambaseed ]; then # operator switch (r16z7): env.r16 carries only `#switch GLM53_MAMBA_ALIGN_SEED 0|1`
    echo "GLM53_MAMBA_ALIGN_SEED=1" >> "$L/.env"
  elif [ "$feat" = ar1shot ]; then   # operator knob (r16z6ar): the one-shot all-reduce at the module defaults (<= 512 KiB)
    echo "GLM53_DEC_AR1SHOT=1" >> "$L/.env"
  elif [ "$feat" = ar1shot-verify ]; then   # operator knob (r16z6ar): log-only, production's all-reduce served
    echo "GLM53_DEC_AR1SHOT=verify" >> "$L/.env"
  elif [ "$feat" = dlmh ]; then   # operator knob (r16z6dl): the two-stage candidate head at the module defaults (C 128, g 128)
    echo "GLM53_DEC_DLMH=1" >> "$L/.env"
  elif [ "$feat" = dlmh-verify ]; then   # operator knob (r16z6dl): log-only, production's candidates served
    echo "GLM53_DEC_DLMH=verify" >> "$L/.env"
  elif [ "$feat" = planpin ]; then   # operator knob (r16z8p): the page-locked plan staging ring
    echo "GLM53_MLA_PLAN_PIN=1" >> "$L/.env"
  elif [ "$feat" = exactlens ]; then # operator switch: env.r16 carries only `#switch GLM53_MLA_EXACT_LENS 0|1`
    echo "GLM53_MLA_EXACT_LENS=1" >> "$L/.env"
  else
    for n in $names; do grep -E "^$n=" "$KIT/env.r16" >> "$L/.env"; done
  fi
elif [ "$feat" = apc-lru ]; then
  echo "GLM53_APC_DRAFTER_LOW_PRIORITY=1" >> "$L/.env"
fi
if [ "$feat" = blockverify ]; then   # the switch's value is an enum (standard|block), not a secret
  if [ "$op" = on ]; then [ -z "$(tail -c1 "$L/.env")" ] || echo >> "$L/.env"; echo "GLM53_REJECTION_METHOD=block" >> "$L/.env"; fi
  launcher_check
  echo "env_r16: blockverify $op -> GLM53_REJECTION_METHOD $([ "$op" = on ] && echo "=block" || echo "unset (= standard)") (restart both ranks to apply)"
  exit 0
fi
if [ "$feat" = trace ] && [ "$op" = on ]; then   # measurement-only (docs/DEC_HOSTGAP.md §3): the values are the
  [ -z "$(tail -c1 "$L/.env")" ] || echo >> "$L/.env"   # tracer's own defaults, not env.r16 lines (never on by default)
  for kv in $TRACE_ON; do echo "$kv" >> "$L/.env"; done
fi
launcher_check
echo "env_r16: $feat $op -> .env now has: $(for n in $names; do grep -qE "^$n=" "$L/.env" && echo -n "$n " ; done)(restart both ranks to apply)"
