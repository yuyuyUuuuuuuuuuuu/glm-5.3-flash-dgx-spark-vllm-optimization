#!/usr/bin/env bash
# [glm53-mla-exactlens] host test of the r16z2x start.sh stage (tools/deploy16/make_start_sh.py --stage r16z2x):
# S1 r16z2 still regenerates production's plain start.sh 0962bae7... byte for byte
# S2 r16z2x == the committed launcher/start.sh; r16z2x - r16z2 is exactly the XL1-XL4 hunk set (patch file)
# S3 GLM53_MLA_EXACT_LENS: one head -e line, one worker serve_env_names entry, one validation
# S4 the validation itself (sourced function body run in a subshell): empty/0/1 accepted, 2/yes refused (value not
#    printed), 1 refused without the overlay files / without a patch_tf_bundle.py that runs the patch
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROD=${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prod-launcher/start.sh; APC=${TF_EXL3_KITS:-$HOME}/tf-exl3-launcher-apc
T=$(mktemp -d "${TEST_TMP:-${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/opt-decodekit/tmp}/stage.XXXX"); trap 'rm -rf "$T"' EXIT
fail=0; ck() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
M="python3 $REPO/tools/deploy16/make_start_sh.py $PROD $APC"
$M $T/z2.sh --stage r16z2 >/dev/null
$M $T/x.sh --stage r16z2x >/dev/null
[ "$(sha256sum < $T/z2.sh | cut -c1-16)" = 0962bae7503d34a7 ]; ck $? "S1 r16z2 plain == 0962bae7503d34a7 (r16z2/r16z2rev kit start.sh)"
cmp -s $T/x.sh "$REPO/launcher/start.sh"; ck $? "S2 r16z2x == committed launcher/start.sh"
diff -u $T/z2.sh $T/x.sh | tail -n +3 > $T/d1; tail -n +3 "$REPO/launcher/start.sh.exactlens.patch" > $T/d2; cmp -s $T/d1 $T/d2
ck $? "S2 r16z2 -> r16z2x == launcher/start.sh.exactlens.patch"
[ "$(diff $T/z2.sh $T/x.sh | grep -c '^[<>]' )" -le 30 ] && ! diff $T/z2.sh $T/x.sh | grep '^<' | grep -qv 'GLM53_MOE_E4M3_DOWN; do'
ck $? "S2 the only r16z2 line r16z2x changes is the worker list's last line"
for f in x; do
  h=$(grep -cxF '        -e GLM53_MLA_EXACT_LENS="${GLM53_MLA_EXACT_LENS:-}" \' $T/$f.sh)
  w=$(sed -n '/local -a serve_env_names=()/,/; do  # \[tf-exl3-fork\]/p' $T/$f.sh | grep -cw GLM53_MLA_EXACT_LENS)
  v=$(grep -c '_glm53_validate_bool_flag GLM53_MLA_EXACT_LENS' $T/$f.sh)
  [ "$h$w$v" = 111 ] && bash -n $T/$f.sh; ck $? "S3 $f.sh: head -e x$h, worker list x$w, validation x$v, bash -n"
done
# S4: run the start.sh's own validation function on a fake bundle
fn=$(awk '/^_glm53_validate_bool_flag\(\) *\{/,/^\}/' $T/x.sh)
blk=$(awk '/# \[mla-exactlens\] the exact-length FA2 sparse-MLA plan/{f=1} f{print} f&&/^    fi$/{n++; if(n==2) exit}' $T/x.sh)
[ -n "$fn" ] && [ -n "$blk" ]; ck $? "S4 extracted _glm53_validate_bool_flag and the XL1 block"
B=$T/b; mkdir -p $B/overlay; echo 'PATCHES = (("patch_mla_exactlens.py", "GLM53_MLA_EXACT_LENS"),)' > $T/ptb.py
run() { ( eval "$fn"; TF_BUNDLE_DIR_HOST=$B; TF_BUNDLE_PATCH_HOST=$T/ptb.py; GLM53_MLA_EXACT_LENS="$1"; f() { eval "$blk"; return 0; }; f ) 2>$T/err; }
run "" ; ck $? "S4 unset accepted"
run 0 ; ck $? "S4 0 accepted"
run 1 ; [ $? != 0 ] && grep -q "requires" $T/err; ck $? "S4 1 refused without the overlay files"
echo x > $B/overlay/patch_mla_exactlens.py; echo x > $B/overlay/glm53_mla_exactlens.py
run 1 ; ck $? "S4 1 accepted with the overlay files and a patch_tf_bundle.py that runs the patch"
echo 'PATCHES = ()' > $T/ptb.py; run 1; [ $? != 0 ] && grep -q "patch_tf_bundle" $T/err; ck $? "S4 1 refused with a patch_tf_bundle.py that does not run the patch"
run 2 ; [ $? != 0 ] && ! grep -q "=2" $T/err; ck $? "S4 2 refused, value not printed"
run yes ; [ $? != 0 ]; ck $? "S4 yes refused"
[ $fail = 0 ] && echo "RESULT: ALL OK" || echo "RESULT: FAIL"; exit $fail
