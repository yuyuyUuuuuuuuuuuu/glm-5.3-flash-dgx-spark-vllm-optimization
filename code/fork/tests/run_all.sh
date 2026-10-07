#!/usr/bin/env bash
# Build the AOT extensions, then run every test of the bundle in order, one GPU container per step. Every GPU step goes
# through tests/r16/gpu.sh = tests/gpu_run.sh (nodeC's host rules: >= 40 GB available, one tf-exl3 container at a time,
# --rm, --network none) under flock /tmp/tf-gpu-bench.lock, retried while another job's container runs.
# Exit status: non-zero if the build or any test failed. Logs: ${LOG_DIR:-tests/logs}/<step>.log
#
# deploy-r16: one suite for every merged branch (deploy-r15's suites + quickwins, mlaprefill/review-mla, dec-fp8roof,
# dec-moeglue, dec-hostloop, dec-smallops, kpoolring) plus the R16 interaction tests (tests/r16/) and the prefill ->
# decode state-handoff check (tests/handoff/, docs/HANDOFF.md). Knobs:
#   R16=off (default) | on    off: no R16 flag in any container (= deploy-r15 behaviour); on: tests/r16/flags.sh's
#                              R16_ON_ENV in EVERY container (GPU_RUN_ENV_EXTRA), on top of what each test sets itself
#   R16_DROP="K1 K2"          with R16=on: leave these knobs out of the forced flag set (a test whose own subject is one
#                              of them, e.g. the smallops `mhc` kind that KINDS=dconv excludes)
#   SKIP_BUILD=1              reuse the AOT .so files in the tree (only import-check them)
#   GPU_RUN_BIND              section A's production module: unset = the image's quantization/exl3.py; production runs
#                              the launcher overlay exl3.py (the sections that need it bind it themselves):
#   GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" \
#   LOG_DIR=docs/logs/run_all_live tests/run_all.sh
#   ONLY="name1 name2"        run only these steps (build/verify always runs)
#   R16_STRICT=1              with R16=on: also force the knob each of the 4 precondition-clash steps leaves out
#                              (r16drop below; those steps then fail by design, REPRODUCE.md section 13)
# Paths: tests/paths.sh (TF_EXL3_MODELS, TF_EXL3_ASSETS, TF_EXL3_KITS). A step whose inputs are not on this machine is
# reported as "SKIP (needs ...)", counted apart from PASS / FAIL, and listed at the end; tests/derive_assets.sh rebuilds
# the derivable ones.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
LOGS="${LOG_DIR:-tests/logs}"
mkdir -p "${LOGS}"
RUN=tests/r16/gpu.sh
SP=/usr/local/lib/python3.12/dist-packages
OV="${REPO}/docs/ref/prod_live/overlay_exl3.py=${SP}/vllm/model_executor/layers/quantization/exl3.py"
# shellcheck source=paths.sh
source tests/paths.sh
M="${TF_EXL3_MODELS}"; A="${TF_EXL3_ASSETS}"
NV="${M}/GLM-5.3-Flash-Uncensored-NVFP4/config.json"          # partial NVFP4 checkpoint (tests/derive_assets.sh models)
DF="${M}/GLM-5.3-Flash-DFlash2-dc77ff1c/model.safetensors"     # DFlash2 drafter (tests/derive_assets.sh models)
# production's SM90_KV mounts (section C, tests/mla_env.sh): flashinfer 0.6.18 (tests/derive_assets.sh fi618) + patched vLLM files
SM90_ASSETS=("${A}/fi618/flashinfer" "${A}/fi618/flashinfer_cubin" "${A}/vllm-patches/cuda.py.exl3.patched"
             "${A}/vllm-patches/flashinfer_mla_sparse_sm90.py.patched")
MODELS_RO=""
for _d in "${M}/GLM-5.3-Flash-Uncensored-NVFP4" "${M}/GLM-5.3-Flash-DFlash2-dc77ff1c" "${M}/GLM-5.3-Flash-EXL3-TR3-4bpw-partial"; do
  if [ -d "${_d}" ]; then MODELS_RO="${MODELS_RO:+${MODELS_RO}:}${_d}"; else echo "NOTE: ${_d} not present (not mounted)"; fi
done
[ -d "${M}/GLM-5.3-Flash-EXL3-TR3-4bpw-partial" ] || echo "NOTE: test_fp8_large_m runs without its real-sample checks (bf16_samples of the TR3 checkpoint are not published)"
# shellcheck source=r16/flags.sh
source tests/r16/flags.sh
case "${R16:-off}" in
  on)  _e="${R16_ON_ENV}"
       for _k in ${R16_DROP:-}; do _e=$(echo "$_e" | tr ';' '\n' | grep -v "^${_k}=" | paste -sd';'); done
       export GPU_RUN_ENV_EXTRA="${_e}";;
  off) export GPU_RUN_ENV_EXTRA="";;
  *)   echo "R16 must be on or off"; exit 2;;
esac
export GPU_RUN_RO="${GPU_RUN_RO:-${MODELS_RO}}"
declare -a failed=() passed=() skipped=()
echo "run_all: R16=${R16:-off} (extra env: ${GPU_RUN_ENV_EXTRA:-<none>}) GPU_RUN_BIND=${GPU_RUN_BIND:-<none: the image modules>} LOG_DIR=${LOGS} head=$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"

want() { [ -z "${ONLY:-}" ] || [[ " ${ONLY} " == *" $1 "* ]]; }
_record() {   # _record <name> <rc>
  if [ "$2" -eq 0 ]; then
    grep -v "exl3 e2 diag" "${LOGS}/$1.log" | tail -n "${TAIL:-6}"
    echo "--- $1: PASS"
    passed+=("$1")
  else
    grep -v "exl3 e2 diag" "${LOGS}/$1.log" | tail -n 40
    echo "--- $1: FAIL rc=$2 (log: ${LOGS}/$1.log)"
    failed+=("$1")
  fi
}
step() {   # step <name> <entrypoint> <args...>   (GPU container)
  local name="$1"; shift
  want "${name}" || return 0
  echo "=== ${name} ($(date +%H:%M:%S))${STEP_NOTE:+ [${STEP_NOTE}]}"
  "${RUN}" "$@" > "${LOGS}/${name}.log" 2>&1
  _record "${name}" $?
}
needs() {   # needs <name> <what> <path>...: true if every path exists, else reports and counts the step as SKIP
  local name="$1" what="$2" p; shift 2
  want "${name}" || return 1
  for p in "$@"; do
    [ -e "${p}" ] && continue
    echo "--- ${name}: SKIP (needs ${what}: ${p} missing; see REPRODUCE.md section 13)"
    skipped+=("${name}"); return 1
  done
}
# R16=on puts the 8 ship flags into EVERY container. Four tests assert a precondition that one of those flags overrides
# on purpose (REPRODUCE.md section 13); under R16=on each runs with just that knob left out (all other flags on) and says
# so in its header. R16_STRICT=1 keeps the knob (the step then fails by design).
r16drop() {   # r16drop <knob> step <name> <entrypoint> <args...>
  local k="$1"; shift
  if [ "${R16:-off}" = on ] && [ "${R16_STRICT:-0}" != 1 ]; then
    GPU_RUN_ENV_EXTRA="$(echo "${GPU_RUN_ENV_EXTRA}" | tr ';' '\n' | grep -v "^${k}=" | paste -sd';')" \
      STEP_NOTE="R16=on without ${k}: precondition clash" "$@"
  else
    "$@"
  fi
}
host() {   # host <name> <cmd...>   (host shell, CPU only)
  local name="$1"; shift
  want "${name}" || return 0
  echo "=== ${name} (host, $(date +%H:%M:%S))"
  "$@" > "${LOGS}/${name}.log" 2>&1
  _record "${name}" $?
}

IMPORTS='import sys, torch; sys.path.insert(0, "/w"); import tf_exl3_moe_ext as m; assert m.parity() == 1; print("AOT", m.__file__, "parity", m.parity(), "glue", hasattr(m, "moe_forward_glue"), "l2_warm", hasattr(m, "l2_warm")); assert hasattr(m, "moe_forward_glue") and hasattr(m, "l2_warm"); import tf_fp8_gemv_ext as f; assert f.VERSION == 1; print("AOT", f.__file__, "VERSION", f.VERSION); import tf_fp8_large_m_ext as fl; assert fl.VERSION == 1; print("AOT", fl.__file__, "VERSION", fl.VERSION); import tf_fp8_roof_ext as fr; assert fr.VERSION == 1; print("AOT", fr.__file__, "VERSION", fr.VERSION); import glm53_gemv_ext as g; print("AOT", g.__file__, "gemv version", g.version()); import glm53_mla_prefill_ext as a; print("AOT", a.__file__, "mla VERSION", a.VERSION); import glm53_smallops_ext as s; print("AOT", s.__file__, "smallops", [n for n in dir(s) if not n.startswith("_")])'
if [ "${SKIP_BUILD:-0}" = 1 ]; then
  ONLY= step verify python3 -c "${IMPORTS}"
else
  ONLY= step build bash -c "cd /w && python3 setup.py build_ext --inplace --force 2>&1 | grep -E 'error|Error|^copying build' ; python3 -c '${IMPORTS}'"
fi
if [ "${#failed[@]}" -ne 0 ]; then echo "build failed; not running tests"; exit 1; fi

# ---- A. deploy-r15 suites (the caller's GPU_RUN_BIND: image modules or the launcher overlay exl3.py) ----------------
# prod_module fails for a file that is not in integrate.KNOWN_PROD_MODULES (the check prints how to certify it)
TAIL=4 step prod_module python3 -u tests/prod_module_check.py
step test_layout_equiv python3 -u tests/test_layout_equiv.py
step test_units        python3 -u tests/test_units.py
step test_stage_vs_xl  python3 -u tests/test_stage_vs_xl.py
step test_e2e_vs_xl    python3 -u tests/test_e2e_vs_xl.py
step test_e2e_vs_f64   python3 -u tests/test_e2e_vs_f64.py
step test_graph        python3 -u tests/test_graph.py
step test_integrate    python3 -u tests/test_integrate.py
step test_apply_fused  python3 -u tests/test_apply_fused.py
step test_prefill_cap  python3 -u tests/test_prefill_cap.py   # GLM53_PREFILL_FUSED_CAP (docs/PREFILL_CAP.md)
step test_variants     python3 -u tests/test_variants.py
step test_fp8_gemv     python3 -u tests/test_fp8_gemv.py        # FP8 Marlin-layout GEMM (fp8_gemv.py, GLM53_FP8_GEMV)
step test_fp8_integrate python3 -u tests/test_fp8_integrate.py
step test_fp8_large_m  python3 -u tests/test_fp8_large_m.py    # GLM53_FP8_LARGE_M kernels + numerics (docs/FP8_LARGE_M.md)
step test_fp8_large_m_integrate python3 -u tests/test_fp8_large_m_integrate.py
step test_bf16_gemv    python3 -u tests/test_bf16_gemv.py      # GLM53_BF16_GEMV kernels (docs/BF16_GEMV.md)
needs test_gemv_install "NVFP4 + DFlash2 weights" "$NV" "$DF" && step test_gemv_install python3 -u tests/test_gemv_install.py   # its wiring (real weights: GPU_RUN_RO)
step test_glm53_runtime python3 -u tests/test_glm53_runtime.py # GLM53_MEM_HYGIENE / GLM53_TF_PROFILE (+ PROF_DIAG off)
TAIL=2 step native_kernel python3 -u tests/probe_native_kernel.py   # the baseline kernel E.5 is timed against
TAIL=16 step bench_ab_decode python3 -u tests/bench_ab_decode.py
TAIL=14 step bench_fp8_gemv python3 -u tests/bench_fp8_gemv.py --rounds 9
TAIL=22 step bench_fp8_large_m python3 -u tests/bench_fp8_large_m.py

# ---- B. the launcher overlay exl3.py (production's module) for the decode branches -------------------------------
_bind="${GPU_RUN_BIND:-}"; _env="${GPU_RUN_ENV:-}"
export GPU_RUN_BIND="${OV}" GPU_RUN_ENV=""
needs test_fp8_roof "the DFlash2 drafter" "$DF" && step test_fp8_roof     python3 -u tests/test_fp8_roof.py        # GLM53_DEC_FP8ROOF (docs/DEC_FP8ROOF.md), real DFlash2 fc
step test_moeglue      python3 -u tests/test_moeglue.py         # GLM53_DEC_MOEGLUE (docs/DEC_MOEGLUE.md)
step test_moeglue_warm python3 -u tests/test_moeglue_warm.py    # GLM53_DEC_MOEGLUE_WARM
step rv_moeglue_adversarial python3 -u tests/rv_moeglue_adversarial.py
needs test_smallops_kernels "NVFP4 + DFlash2 weights" "$NV" "$DF" && step test_smallops_kernels python3 -u tests/test_smallops_kernels.py   # GLM53_DEC_SMALLOPS (docs/DEC_SMALLOPS.md)
needs test_smallops_install "NVFP4 + DFlash2 weights" "$NV" "$DF" && \
  r16drop GLM53_DEC_SMALLOPS_KINDS step test_smallops_install python3 -u tests/test_smallops_install.py
needs test_smallops_tc "NVFP4 weights" "$NV" && step test_smallops_tc  python3 -u tests/test_smallops_tc.py
needs review_smallops_adversarial "NVFP4 + DFlash2 weights" "$NV" "$DF" && \
  r16drop GLM53_DEC_SMALLOPS_KINDS step review_smallops_adversarial python3 -u tests/review_smallops_adversarial.py
step test_kpool_ring_gpu python3 -u tests/kpoolring/test_kpool_ring_gpu.py  # GLM53_KPOOL_RING (docs/KPOOL_RING.md)
TAIL=30 step r16_decode_combo python3 -u tests/r16/test_r16_decode_combo.py  # fp8roof + moeglue warm + fp8_gemv + TF
TAIL=30 step r16_breakable python3 -u tests/r16/test_r16_breakable.py  # the same under vLLM's breakable (PIECEWISE) capture

# ---- C. production's SM90_KV mounts (flashinfer 0.6.18 + patched cuda.py / flashinfer_mla_sparse_sm90.py + overlay)
source tests/mla_env.sh
needs test_quickwins "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && \
  r16drop GLM53_MLA_PREFILL step test_quickwins    python3 -u tests/qw/test_quickwins.py   # GLM53_PREFILL_QUICKWINS (docs/PREFILL_QUICKWINS.md)
needs test_mla_prefill "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step test_mla_prefill  python3 -u tests/test_mla_prefill.py    # GLM53_MLA_PREFILL kernel: conversion, parity, edges
needs test_mla_integrate "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step test_mla_integrate python3 -u tests/test_mla_integrate.py # its forward_mqa hook on production's backend file
needs review_mla_adv "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step review_mla_adv    python3 -u tests/review_mla_adv.py a b c d e f
needs review_mla_adv2 "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step review_mla_adv2   python3 -u tests/review_mla_adv2.py
needs review_mla_plugin "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step review_mla_plugin python3 -u tests/review_mla_plugin.py
needs bench_mla_prefill "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && TAIL=8 step bench_mla_prefill python3 -u tests/bench_mla_prefill.py --T 13824 --variants v0,l2,hook --iters 10
needs test_hostloop "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step test_hostloop     python3 -u tests/test_hostloop.py        # GLM53_DEC_HOSTLOOP (docs/DEC_HOSTLOOP.md)
needs test_hostloop_plugin "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && \
  r16drop GLM53_DEC_MOEGLUE_WARM step test_hostloop_plugin python3 -u tests/test_hostloop_plugin.py
needs test_hostloop_wake "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step test_hostloop_wake python3 -u tests/test_hostloop_wake.py 12,0
needs test_prof_diag "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step test_prof_diag    python3 -u tests/test_prof_diag.py       # GLM53_DEC_PROF_DIAG
needs review_hostloop_async "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && step review_hostloop_async python3 -u tests/review_hostloop_async.py
needs r16_prefill_combo "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && TAIL=30 step r16_prefill_combo python3 -u tests/r16/test_r16_prefill_combo.py  # quickwins + MLA prefill on forward_mqa
R15_SITE_DIR="${R15_SITE:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher/overlay/tf/site}"   # deploy-r15 bundle (read-only copy)
needs r16_plugins "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && \
  needs r16_plugins "the deploy-r15 bundle (not published)" "${R15_SITE_DIR}/integrate.py" && \
  GPU_RUN_RO="${GPU_RUN_RO}:${R15_SITE_DIR}" TAIL=40 step r16_plugins python3 -u tests/r16/test_r16_plugins.py  # real plugin loader, r15 vs r16
needs r16_loo_census "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && TAIL=4 step r16_loo_census python3 -u tests/r16/test_r16_loo_census.py  # every env-only revert state of DEPLOY_R16.md §5
export GPU_RUN_BIND="${_bind}" GPU_RUN_ENV="${_env}"

# ---- C2. prefill -> decode state handoff (docs/HANDOFF.md): the real vLLM engine of the production image on the handoff
# mini model (tests/handoff/build_mini.py), containers composed like both production ranks (tests/handoff/run.sh, its own
# flock + host-memory rules): quickwins bit-identical incl. every KV byte, MLA prefill's kv_indices handoff, decode reads
needs handoff "the handoff mini model (tests/derive_assets.sh mini)" "${M}/GLM-5.3-Flash-handoff-mini/model.safetensors" && \
  needs handoff "flashinfer 0.6.18 + patched vLLM files" "${SM90_ASSETS[@]}" && \
  needs handoff "the launcher overlay + GLM-OCR tokenizer + DFlash2 (tests/derive_assets.sh launcher models)" \
    "${A}/prod-launcher/overlay" "${M}/GLM-OCR/tokenizer.json" "$DF" && \
  TAIL=14 host handoff tests/handoff/check.sh "${LOGS}/handoff"

# ---- D. CPU (host) ---------------------------------------------------------------------------------------------------
needs test_patch_kpool_tail_ring "the image's vllm sources (tests/derive_assets.sh vllm-src)" "${A}/vllm-src/vllm" && \
  host test_patch_kpool_tail_ring python3 tests/kpoolring/test_patch_kpool_tail_ring.py
host test_consistency_probe_mock python3 tests/kpoolring/test_consistency_probe_mock.py
host check_paths python3 tests/check_paths.py                   # no unexpanded placeholder / absolute home path in code

echo "SUMMARY: ${#passed[@]} PASS, ${#failed[@]} FAIL, ${#skipped[@]} SKIP"
[ "${#skipped[@]}" -eq 0 ] || echo "SKIPPED (inputs not on this machine, REPRODUCE.md section 13): ${skipped[*]}"
if [ "${#failed[@]}" -ne 0 ]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
if [ "${#skipped[@]}" -eq 0 ]; then echo "ALL PASSED"; else echo "ALL RUN STEPS PASSED (${#skipped[@]} skipped, see above)"; fi
