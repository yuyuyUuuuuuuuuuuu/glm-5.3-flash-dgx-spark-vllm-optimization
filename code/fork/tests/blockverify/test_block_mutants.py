"""Power check of test_block_exactness.py (reviewer): the exactness harness must FAIL on a biased sampler.

Runs test_block_exactness.py (unchanged file, exec'd with a hook) after applying ONE mutation to the block-keys-patched
copies ("fix" site) right before they are imported, and asserts that the harness reports blk-fix as NOT exact (seq /
step2 chi-square or the acceptance-law z test) for every mutant, and passes for the control ("none").
  none            control, no mutation: must pass (also with a fresh seed base -> not a lucky seed)
  verifier_only   drafter walk keeps the stock key (P+k, 0); verifier keyed (P+j, j): next step's drafts reuse the
                  noise of the failed rows' drafts -> cross-step coupling through the drafts
  drafter_only    u keeps the stock key (P+j, 0); drafter keyed: next step's u_0 IS the failed row's u
  residual_lane0  keyed residual resample uses lane 0 (the rejected draft's Gumbel key): within-step bias
  h_bias          acceptance threshold h -> min(1.03 h, 1): a 3 % algorithmic bias of Sun et al.'s rule
  resmass_skip    the residual-mass kernel also skips the last draft row (its r stays uninitialized): wrong h_{n-2}
Env: BLOCKVERIFY_MUTANT (required), BLOCKVERIFY_NLIST (default "5"), BLOCKVERIFY_VARS (default "blk-fix"),
     BLOCKVERIFY_SEED_BASE (default 1000 = the harness's), plus the harness's own BLOCKVERIFY_* knobs.
Run: GPU_RUN_ENV="BLOCKVERIFY_ONLY=prod;BLOCKVERIFY_MUTANT=h_bias" tests/r16/gpu.sh python3 -u tests/blockverify/test_block_mutants.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE / "test_block_exactness.py"
MUTANTS = {
    "none": None,
    "verifier_only": ("spec", "            position = position + (tl.cast(step, tl.int64) << 32)\n",
                      "            position = position\n"),
    "drafter_only": ("rsu", "                pos = pos + (tl.cast(i, tl.int64) << 32)\n", "                pos = pos\n"),
    "residual_lane0": ("rsu", "    key = tl.where(residual_lane, k1, k0)\n", "    key = k0\n"),
    "h_bias": ("rsu", "                accepted_length = tl.where(u <= h, i + 1, accepted_length)\n",
               "                accepted_length = tl.where(u <= tl.minimum(h * 1.03, 1.0), i + 1, accepted_length)\n"),
    "resmass_skip": ("rsu", "    has_next_row = logit_idx + 1 < tl.num_programs(0)\n"
                            "    next_local_pos = tl.load(expanded_local_pos_ptr + logit_idx + 1, mask=has_next_row, other=0)\n",
                     "    has_next_row = logit_idx + 2 < tl.num_programs(0)\n"
                     "    next_local_pos = tl.load(expanded_local_pos_ptr + logit_idx + 2, mask=has_next_row, other=0)\n"),
}
name = os.environ.get("BLOCKVERIFY_MUTANT", "")
if name not in MUTANTS:
    sys.exit(f"BLOCKVERIFY_MUTANT must be one of {sorted(MUTANTS)} (got {name!r})")
nlist = [int(x) for x in os.environ.get("BLOCKVERIFY_NLIST", "5").split(",")]
vars_ = [x for x in os.environ.get("BLOCKVERIFY_VARS", "blk-fix").split(",") if x]
seed_base = int(os.environ.get("BLOCKVERIFY_SEED_BASE", "1000"))

src = TEST.read_text()
edits = [
    ('RSU_PROD = load_module("glm53_rsu_prod", SITES["prod"] / REL_RSU)\n',
     "if MUTANT is not None:\n"
     "    _p = SITES['fix'] / (REL_SPEC if MUTANT[0] == 'spec' else REL_RSU)\n"
     "    _t = _p.read_text()\n"
     "    assert _t.count(MUTANT[1]) == 1, ('mutant anchor not unique', MUTANT[1])\n"
     "    _p.write_text(_t.replace(MUTANT[1], MUTANT[2]))\n"
     "    print('== MUTANT applied to', _p.name, ':', MUTANT[2].strip(), flush=True)\n"
     'RSU_PROD = load_module("glm53_rsu_prod", SITES["prod"] / REL_RSU)\n'),
    ("[4, 5, 7], 1024, int(1000 * SCALE)", "NLIST, 1024, int(1000 * SCALE)"),
    ('vs = ALL_V if model.name == "prod" else ["std-prod", "blk-prod", "blk-fix", "std-fix"]', "vs = VARS"),
    ("manual_seed(1000 + n)", "manual_seed(SEED_BASE + n)"),
]
for old, new in edits:
    n_old = src.count(old)
    assert n_old >= 1, ("harness anchor missing", old)
    src = src.replace(old, new)
ns = {"__name__": "__main__", "__file__": str(TEST), "MUTANT": MUTANTS[name], "NLIST": nlist, "VARS": vars_,
      "SEED_BASE": seed_base}
rc = 0
try:
    exec(compile(src, str(TEST), "exec"), ns)
except SystemExit as e:
    rc = int(e.code or 0)
except AssertionError as e:
    # the harness's own invariant (every emitted token has target mass) tripped: a loud detection, e.g. resmass_skip
    # lets h pick a row whose residual is empty and the Gumbel argmax over all -inf emits token 0
    rc = 1
    ns.setdefault("FAIL", []).append(f"blk-fix: exact -- harness assertion: {e}")
    print(f"== harness AssertionError: {e}", flush=True)
fails = ns.get("FAIL", [])
setup = [f for f in fails if "blk-fix: exact" not in f and "follow the block rule" not in f]
biased = [f for f in fails if "blk-fix: exact" in f or "follow the block rule" in f]
print(f"\n== mutant {name}: harness rc={rc}, blk-fix failures {len(biased)}, other failures {len(setup)}")
if name == "none":
    ok = rc == 0 and not fails
    print(f"{'PASS' if ok else 'FAIL'} control: the unmutated block-keys modules pass the harness (seed base {seed_base})")
else:
    ok = rc != 0 and biased and not setup
    print(f"{'PASS' if ok else 'FAIL'} mutant {name}: the harness flags blk-fix as biased" + "".join(f"\n  - {f}" for f in biased))
sys.exit(0 if ok else 1)
