#!/usr/bin/env bash
# r16z7: run the kit's boot_checks.sh (host, PATH shims like test_boot_checks.sh) on PRODUCTION'S REAL boot logs and
# container env (read-only copies from nodeA/nodeB, credential-like env values masked on nodeA before transfer), as
#   K   the logs as they are (production on r16z6rev-kl2): the r16z7 rows must read "neither container carries ..."
#   U   the r16z7 install with every new knob unset: the real logs + the lines the r16z7 bundle/site print (rendered from
#       THIS kit's compose/engine logs on nodeC, i.e. the exact strings) inserted where the bundle prints them
#   T   U + GLM53_KPOOL_TAIL_POSITIONS=2 on both ranks (the overlay's two patched lines + the bundle's applied line)
#   H   T with the worker's indexer line missing (must fail: [kpooltail-pair] MISS)
# Usage: tests/r16z7/realboot_check.sh <kit> <dir with head.log.raw worker.log.raw head.env.raw worker.env.raw>
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
ANCH='[glm53-tf-bundle] patch_mla_exactlens.py: GLM53_MLA_EXACT_LENS unset -> skipped (stock)'
mk() {   # mk <state dir> <head inserted lines file> <worker inserted lines file> <extra env lines>
  mkdir -p "$1"
  for s in head worker; do
    f="$2"; [ $s = worker ] && f="$3"
    python3 - "$D/$s.log.raw" "$f" "$ANCH" > "$1/$s.log" <<'PY'
import sys
src, ins, anch = open(sys.argv[1], errors="replace").read().split("\n"), open(sys.argv[2]).read().rstrip("\n").split("\n"), sys.argv[3]
assert src.count(anch) == 1, "anchor"
i = src.index(anch) + 1
out = src[:i] + [l for l in ins if l] + src[i:]
# the site module's plugin line, as the engine prints it (one per process; boot_checks needs >= 1)
out.append("(EngineCore pid=2942) INFO 10-05 03:00:00 [glm53_spec_vtrim.py:485] glm53_spec_vtrim plugin loaded (pid 2942): GLM53_SPEC_VTRIM='' -> off")
print("\n".join(out))
PY
    { cat "$D/$s.env.raw"; printf '%s\n' $4; } > "$1/$s.env"
  done
}
run() { PATH="$tmp/bin:$PATH" FAKE_DIR="$1" bash "$KIT/tools/boot_checks.sh" > "$1.out" 2>&1; echo $?; }
fail=0; ck() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
NEWENV="GLM53_SPEC_VTRIM= GLM53_SPEC_VTRIM_TAU= GLM53_SPEC_VTRIM_MIN= GLM53_SPEC_VTRIM_LOG="
# K: as they are
mkdir -p "$tmp/K"; for s in head worker; do cp "$D/$s.log.raw" "$tmp/K/$s.log"; cp "$D/$s.env.raw" "$tmp/K/$s.env"; done
r=$(run "$tmp/K"); cp "$tmp/K.out" "$D/K.out"
grep -qF 'info neither container carries GLM53_KPOOL_TAIL_POSITIONS' "$tmp/K.out" && grep -qF 'info neither container carries GLM53_MAMBA_ALIGN_SEED' "$tmp/K.out" \
  && ! grep -E '^(MISS|BAD) ' "$tmp/K.out" | grep -qE 'kpool-tail|kpooltail|mamba-align'
ck $? "K production's real kl2 boot: kpooltail/mambaseed read as 'neither container carries' (no r16z7 row fails); other rows: $(grep -E '^(MISS|BAD) ' "$tmp/K.out" | cut -c1-90 | tr '\n' ';')"
printf '%s\n' "[glm53-tf-bundle] patch_kpool_tail_positions.py: GLM53_KPOOL_TAIL_POSITIONS unset -> skipped (stock)" \
  "[glm53-tf-bundle] patch_mamba_align_seed.py: GLM53_MAMBA_ALIGN_SEED unset -> skipped (stock)" > "$tmp/unset.ins"
mk "$tmp/U" "$tmp/unset.ins" "$tmp/unset.ins" "$NEWENV GLM53_KPOOL_TAIL_POSITIONS= GLM53_MAMBA_ALIGN_SEED="
r=$(run "$tmp/U"); cp "$tmp/U.out" "$D/U.out"
bad=$(grep -E '^(MISS|BAD) ' "$tmp/U.out" | grep -vE '^BAD +\[(head|worker)\] \([0-9]+\) idx_gate result differs$')
[ -z "$bad" ] && grep -qF 'ok   [head] (1) [glm53-tf-bundle] patch_kpool_tail_positions.py: GLM53_KPOOL_TAIL_POSITIONS unset -> skipped (stock)' "$tmp/U.out" \
  && grep -qE '^ok +\[worker\] \(1\) glm53_spec_vtrim plugin loaded ... -> off' "$tmp/U.out" && grep -qE '^boot_checks: (ALL OK|PROBLEMS)$' "$tmp/U.out"
ck $? "U r16z7 installed, every new knob unset (real logs + the kit's lines): no MISS/BAD except production's known benign idx_gate row (rc $r) ${bad:+-> $bad}"
printf '%s\n' "[glm53-kpool-tail-positions] $SPV/v1/attention/backends/mla/indexer.py: patched (in-place persistent tail slots)" \
  "[glm53-kpool-tail-positions] $SPV/v1/worker/gpu/model_states/mamba_hybrid.py: patched (MambaHybridModelState.prepare_attn passes positions)" \
  "[glm53-tf-bundle] patch_kpool_tail_positions.py: applied (GLM53_KPOOL_TAIL_POSITIONS=2)" \
  "[glm53-tf-bundle] patch_mamba_align_seed.py: GLM53_MAMBA_ALIGN_SEED unset -> skipped (stock)" > "$tmp/t.ins"
mk "$tmp/T" "$tmp/t.ins" "$tmp/t.ins" "$NEWENV GLM53_KPOOL_TAIL_POSITIONS=2 GLM53_MAMBA_ALIGN_SEED="
r=$(run "$tmp/T"); cp "$tmp/T.out" "$D/T.out"
bad=$(grep -E '^(MISS|BAD) ' "$tmp/T.out" | grep -vE '^BAD +\[(head|worker)\] \([0-9]+\) idx_gate result differs$')
[ -z "$bad" ] && grep -qF 'ok   [kpooltail-pair]' "$tmp/T.out" && grep -qE '^ok +\[mhcsp-pair\]' "$tmp/T.out"
ck $? "T + GLM53_KPOOL_TAIL_POSITIONS=2 on both: [kpooltail-pair] ok, [mhcsp-pair] ok (arm_env's bootok needs it), no other MISS/BAD (rc $r) ${bad:+-> $bad}"
grep -v 'indexer.py: patched (in-place' "$tmp/t.ins" > "$tmp/h.ins"
mk "$tmp/H" "$tmp/t.ins" "$tmp/h.ins" "$NEWENV GLM53_KPOOL_TAIL_POSITIONS=2 GLM53_MAMBA_ALIGN_SEED="
r=$(run "$tmp/H"); cp "$tmp/H.out" "$D/H.out"
[ "$r" = 1 ] && grep -qF 'MISS [kpooltail-pair]' "$tmp/H.out" && grep -qF 'MISS [worker] (0 < 1) [glm53-kpool-tail-positions]' "$tmp/H.out"
ck $? "H =2 but the worker lacks the in-place indexer line: rc $r, MISS [kpooltail-pair] + MISS [worker]"
grep -qE 'masked|VLLM_API_KEY=' "$D"/[KUTH].out && { echo "FAIL a masked/env value reached the output"; fail=1; }
echo "realboot_check: $([ $fail = 0 ] && echo ALL OK || echo FAILURES)"; exit $fail
