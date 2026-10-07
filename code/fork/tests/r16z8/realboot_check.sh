#!/usr/bin/env bash
# r16z8: run the kit's boot_checks.sh (host, PATH shims as tests/r16z7/realboot_check.sh) on PRODUCTION'S REAL boot logs
# and container env (read-only copies from nodeA/nodeB, credential-like env values masked on nodeA before transfer):
#   K   the logs as they are (production on r16z7, the new knobs absent): only the r16z8 rows (dlmh / ar1shot plugin
#       lines) may report, every other row as the kit production runs reports it
#   U   r16z8 installed, every new knob unset: the real logs + the two site modules' off lines (one per rank, rendered
#       from the shipped format with production's forwarded '' value) + the seven names empty in both containers
#   T   U with GLM53_DEC_DLMH=1 + GLM53_DEC_AR1SHOT=1 + GLM53_KPOOL_TAIL_POSITIONS=2 (the combined A/B arm on top of the
#       tail fix): the modules' boot lines exactly as the two-rank NCCL run printed them (tests/r16z8/test_combo_tp2.py,
#       rank 0 -> head, rank 1 -> worker) + the tail overlay lines; then --after-traffic with both serving lines
#   H1  T with GLM53_DEC_AR1SHOT missing in the worker container (a one-rank install = hang): MISS [worker]
#   H2  T with the worker's readiness-agreement line missing: MISS [worker]
#   H3  T --after-traffic with the worker's dlmh serving line missing: MISS [worker]
# Usage: tests/r16z8/realboot_check.sh <kit> <dir with head.log.raw worker.log.raw head.env.raw worker.env.raw>
set -uo pipefail
KIT="$(cd "$1" && pwd)"; D="$(cd "$2" && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT; mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'S'
#!/usr/bin/env bash
case "$1" in logs) cat "$FAKE_DIR/${FAKE_SIDE:-head}.log";; inspect) cat "$FAKE_DIR/${FAKE_SIDE:-head}.env";; *) exit 1;; esac
S
cat > "$tmp/bin/ssh" <<'S'
#!/usr/bin/env bash
while [ "${1:0:1}" = - ]; do shift; [ "$1" = BatchMode=yes ] && shift; done
shift; FAKE_SIDE=worker bash -c "$*"
S
cat > "$tmp/bin/curl" <<'S'
#!/usr/bin/env bash
echo 'vllm:cache_config_info{block_size="1152",cache_dtype="fp8",num_gpu_blocks="583",prefix_caching_hash_algo="sha256"} 1.0'
S
chmod +x "$tmp/bin"/*
SPV=/usr/local/lib/python3.12/dist-packages/vllm
NEW="GLM53_DEC_DLMH GLM53_DEC_DLMH_C GLM53_DEC_DLMH_GROUP GLM53_DEC_DLMH_LOG GLM53_DEC_AR1SHOT GLM53_DEC_AR1SHOT_MAX_KB GLM53_DEC_AR1SHOT_LOG"
mk() {   # mk <state dir> <head lines file> <worker lines file> <env lines: NAME=value ...> [worker-env sed expression]
  mkdir -p "$1"
  for s in head worker; do
    f="$2"; [ $s = worker ] && f="$3"
    { cat "$D/$s.log.raw"; cat "$f"; } > "$1/$s.log"
    { grep -vE "^($(echo $NEW GLM53_KPOOL_TAIL_POSITIONS | tr ' ' '|'))=" "$D/$s.env.raw"; printf '%s\n' $4; } > "$1/$s.env"
  done
  [ -z "${5:-}" ] || sed -i -E "$5" "$1/worker.env"
}
run() { PATH="$tmp/bin:$PATH" FAKE_DIR="$1" bash "$KIT/tools/boot_checks.sh" "${@:2}" > "$1.out" 2>&1; echo $?; }
others() { grep -E '^(MISS|BAD) ' "$1" | grep -vE '^BAD +\[(head|worker)\] \([0-9]+\) idx_gate result differs$'; }
fail=0; ck() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
# the module lines (real formats: the off lines are the shipped '%s=%r -> %s' with production's forwarded '' value;
# the on lines are copied from the two-rank NCCL run of test_combo_tp2.py, the dlmh serving line from decode5's engine run)
cat > "$tmp/off.ins" <<'L'
(EngineCore pid=2943) INFO 10-05 09:00:00 [glm53_ar1shot.py:263] glm53_ar1shot plugin loaded (pid 2943): GLM53_DEC_AR1SHOT='' -> off, production all-reduce unchanged
(EngineCore pid=2943) INFO 10-05 09:00:00 [glm53_dlmh.py:694] glm53_dlmh plugin loaded (pid 2943): GLM53_DEC_DLMH='' -> off, production's candidate head unchanged
L
for r in 0 1; do
  cat > "$tmp/on$r.ins" <<L
(EngineCore pid=2943) INFO 10-05 00:11:25 [glm53_ar1shot.py:263] glm53_ar1shot plugin loaded (pid 2943): GLM53_DEC_AR1SHOT='1' -> installing (mode on)
(EngineCore pid=2943) INFO 10-05 00:11:27 [glm53_ar1shot.py:245] glm53_ar1shot: hooked CudaCommunicator.all_reduce (mode on, 2-rank groups, <= 512 KiB, dtypes bf16/fp16/fp32) PROOF mode=on
(EngineCore pid=2943) INFO 10-05 00:11:27 [glm53_dlmh.py:694] glm53_dlmh plugin loaded (pid 2943): GLM53_DEC_DLMH='1' -> installing (mode on, C 128, g 128)
(EngineCore pid=2943) INFO 10-05 00:11:31 [glm53_ar1shot.py:133] glm53_ar1shot: rank $r/2 agreement: every rank ready -> one-shot all-reduce armed (mode on, <= 512 KiB)
(EngineCore pid=2943) INFO 10-05 00:11:32 [glm53_ar1shot.py:133] glm53_ar1shot: rank $r/2 agreement: every rank ready -> one-shot all-reduce armed (mode on, <= 512 KiB)
(EngineCore pid=2943) INFO 10-05 00:11:34 [glm53_dlmh.py:478] glm53_dlmh: rank $r/2 coarse candidate head built (int4 g128, C=128, vocab rows 77440, rows served [7, 14], +156.0 MiB)
(EngineCore pid=2943) INFO 10-05 00:11:36 [glm53_dlmh.py:507] glm53_dlmh: rank $r self-test: candidates and unary logits byte-equal to production's (T in [7, 14]) PROOF mode=on C=128 g=128
(EngineCore pid=2943) INFO 10-05 00:11:20 [glm53_dlmh.py:676] glm53_dlmh: hooked DFlash2 compute_candidates (mode on, C 128, g 128)
[glm53-kpool-tail-positions] $SPV/v1/attention/backends/mla/indexer.py: patched (in-place persistent tail slots)
[glm53-kpool-tail-positions] $SPV/v1/worker/gpu/model_states/mamba_hybrid.py: patched (MambaHybridModelState.prepare_attn passes positions)
[glm53-tf-bundle] patch_kpool_tail_positions.py: applied (GLM53_KPOOL_TAIL_POSITIONS=2)
L
  cat > "$tmp/srv$r.ins" <<L
(EngineCore pid=2943) INFO 10-05 00:11:36 [glm53_ar1shot.py:192] [glm53-ar1shot] rank $r serving confirmed (mode on): 2 graph-captured one-shot all-reduces
(EngineCore pid=2943) INFO 10-04 20:08:47 [glm53_dlmh.py:644] [glm53-dlmh] rank $r serving confirmed (mode on): graph replays 29 (captured graphs 6, eager calls 6) by step 32
L
done
# K
mkdir -p "$tmp/K"; for s in head worker; do cp "$D/$s.log.raw" "$tmp/K/$s.log"; cp "$D/$s.env.raw" "$tmp/K/$s.env"; done
r=$(run "$tmp/K"); cp "$tmp/K.out" "$D/K.out"
o=$(others "$tmp/K.out" | grep -vE 'glm53_(dlmh|ar1shot) plugin loaded \.\.\. -> off')
[ -z "$o" ] && grep -qF 'MISS [head] (0 < 1) glm53_ar1shot plugin loaded ... -> off' "$tmp/K.out"
ck $? "K production's real r16z7 boot (knobs absent): only the r16z8 plugin rows report (the modules are not installed yet), nothing else (rc $r) ${o:+-> $o}"
# U
mk "$tmp/U" "$tmp/off.ins" "$tmp/off.ins" "$(for n in $NEW GLM53_KPOOL_TAIL_POSITIONS; do printf '%s= ' "$n"; done)"
r=$(run "$tmp/U"); cp "$tmp/U.out" "$D/U.out"; o=$(others "$tmp/U.out")
[ -z "$o" ] && grep -qF 'ok   [worker] (1) glm53_ar1shot plugin loaded ... -> off' "$tmp/U.out" && grep -qF 'ok   [head] (1) glm53_dlmh plugin loaded ... -> off' "$tmp/U.out" \
  && grep -q 'info dlmh by the head container.s env: GLM53_DEC_DLMH=off' "$tmp/U.out" && grep -q 'info ar1shot by the head container.s env: GLM53_DEC_AR1SHOT=off' "$tmp/U.out"
ck $? "U r16z8 installed, every new knob unset (real logs + the modules' off lines): no MISS/BAD except production's benign idx_gate row (rc $r) ${o:+-> $o}"
# T
TENV="GLM53_DEC_DLMH=1 GLM53_DEC_DLMH_C= GLM53_DEC_DLMH_GROUP= GLM53_DEC_DLMH_LOG= GLM53_DEC_AR1SHOT=1 GLM53_DEC_AR1SHOT_MAX_KB= GLM53_DEC_AR1SHOT_LOG= GLM53_KPOOL_TAIL_POSITIONS=2"
mk "$tmp/T" "$tmp/on0.ins" "$tmp/on1.ins" "$TENV"
r=$(run "$tmp/T"); cp "$tmp/T.out" "$D/T.out"; o=$(others "$tmp/T.out")
[ -z "$o" ] && grep -qF 'ok   [kpooltail-pair]' "$tmp/T.out" && grep -qF 'ok   [worker] (2) agreement: every rank ready -> one-shot all-reduce armed (mode on' "$tmp/T.out" \
  && grep -qF "ok   [head] (1) self-test: candidates and unary logits byte-equal to production's" "$tmp/T.out" \
  && grep -qF 'info [head] dlmh serving-confirmed line (required with --after-traffic): 0x' "$tmp/T.out"
ck $? "T the combined arm (dlmh=1 + ar1shot=1 + tail=2) at boot: ALL OK (benign idx_gate aside), agreement x2 + self-test per rank, [kpooltail-pair] ok (rc $r) ${o:+-> $o}"
cat "$tmp/on0.ins" "$tmp/srv0.ins" > "$tmp/ts0.ins"; cat "$tmp/on1.ins" "$tmp/srv1.ins" > "$tmp/ts1.ins"
mk "$tmp/TS" "$tmp/ts0.ins" "$tmp/ts1.ins" "$TENV"
r=$(run "$tmp/TS" --after-traffic); cp "$tmp/TS.out" "$D/TS.out"
[ "$(grep -cE '^ok   \[(head|worker)\] \(1\) \[glm53-(dlmh|ar1shot)\] rank R serving confirmed \(mode on\)' "$tmp/TS.out")" = 4 ] \
  && ! others "$tmp/TS.out" | grep -qiE 'dlmh|ar1shot'
ck $? "T --after-traffic with both serving lines per rank: 4 serving rows ok, no dlmh/ar1shot MISS/BAD (other after-traffic rows as production's logs give them: $(others "$tmp/TS.out" | cut -c1-70 | tr '\n' ';'))"
# H1-H3
mk "$tmp/H1" "$tmp/on0.ins" "$tmp/on1.ins" "$TENV" '/^GLM53_DEC_AR1SHOT=/d'
r=$(run "$tmp/H1"); cp "$tmp/H1.out" "$D/H1.out"
[ "$r" = 1 ] && grep -qF 'MISS [worker] container env GLM53_DEC_AR1SHOT=1 (got: <unset>)' "$tmp/H1.out"
ck $? "H1 GLM53_DEC_AR1SHOT=1 on the head only (a one-rank install would hang): rc $r, MISS [worker]"
grep -v 'agreement: every rank ready' "$tmp/on1.ins" > "$tmp/h2.ins"
mk "$tmp/H2" "$tmp/on0.ins" "$tmp/h2.ins" "$TENV"
r=$(run "$tmp/H2"); cp "$tmp/H2.out" "$D/H2.out"
[ "$r" = 1 ] && grep -qF 'MISS [worker] (0 < 1) agreement: every rank ready -> one-shot all-reduce armed (mode on' "$tmp/H2.out"
ck $? "H2 the worker never reached the readiness agreement: rc $r, MISS [worker]"
grep -v 'glm53-dlmh' "$tmp/ts1.ins" > "$tmp/h3.ins"
mk "$tmp/H3" "$tmp/ts0.ins" "$tmp/h3.ins" "$TENV"
r=$(run "$tmp/H3" --after-traffic); cp "$tmp/H3.out" "$D/H3.out"
[ "$r" = 1 ] && grep -qF 'MISS [worker] (0 < 1) [glm53-dlmh] rank R serving confirmed (mode on)' "$tmp/H3.out"
ck $? "H3 --after-traffic, the worker's dlmh serving line missing: rc $r, MISS [worker]"
grep -qE 'masked|VLLM_API_KEY=' "$D"/{K,U,T,TS,H1,H2,H3}.out && { echo "FAIL a masked/env value reached the output"; fail=1; }
echo "realboot_check: $([ $fail = 0 ] && echo ALL OK || echo FAILURES)"; exit $fail
