#!/usr/bin/env bash
# r16_measure.sh hook for the KL harness (docs/KL_HARNESS.md). Operator-run on nodeA, idle production only.
#   klh_measure.sh <label> [klh.py run options, e.g. --quick]
# Runs `klh.py run <label>` (refuses / retries while other traffic runs; exit 3 = could not get a clean measurement)
# and compares the run with the fixed reference run ($KLH_REF, made once with `klh.py run ref_v1 --reps 3`) and, when
# KLH_BASE=<label of this session's base run> is set, judges it against that base (KLH-CMP ... d_kl=... lines).
# Layout: $KLH_HOME (default ~/tf-exl3-deploy/klh) holds klh.py, fixtures/klh_v1.json.gz, traj_v1.json, runs.
set -uo pipefail
L="${1:?label}"; shift
D="${KLH_HOME:-$HOME/tf-exl3-deploy/klh}"
K="$D/klh.py"
REF="${KLH_REF:-$D/ref_v1.klh.gz}"
R15="${KLH_R15:-$HOME/tf-exl3-deploy/QL_R15a.json}"
export KLH_OUT="$D"
[ -f "$K" ] || { echo "klh: $K missing (copy tools/klh/ from the repo)"; exit 2; }
[ -f "$D/traj_v1.json" ] || { echo "klh: no $D/traj_v1.json - run once on the base config: python3 $K make-traj"; exit 2; }
python3 "$K" run "$L" --traj "$D/traj_v1.json" --out "$D" "$@" || exit $?
if [ -f "$REF" ]; then
  args=("$REF")
  [ -n "${KLH_BASE:-}" ] && [ "$KLH_BASE" != "$L" ] && args+=("$D/$KLH_BASE.klh.gz")
  args+=("$D/$L.klh.gz")
  r15=(); [ -f "$R15" ] && r15=(--r15 "$R15")
  python3 "$K" compare "${args[@]}" "${r15[@]}" --json "$D/$L.cmp.json"
else
  echo "klh: no reference run $REF yet (make it once: python3 $K run ref_v1 --reps 3) - summary only"
fi
