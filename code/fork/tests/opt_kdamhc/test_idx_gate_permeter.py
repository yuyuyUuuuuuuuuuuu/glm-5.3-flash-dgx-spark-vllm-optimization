"""opt-kdamhc: the idx_gate quick win's runtime check (glm53_prefill_quickwins._make_head_gate_impl), CPU only.

Production (nodeA, 2026-10-02 08:07:51 UTC boot) logged "idx_gate result differs from production's at M=16384 ...;
idx_gate turned off": the engine's dummy profile run (M = max_num_batched_tokens = 16384) tripped the bitwise check and
the GLOBAL off switch kept the quick win off for every later M (the 13,824-token chunks it was built for).
Checks:
  P1 a result that matches production except where BOTH are NaN is accepted (the dummy run's garbage)
  P2 a real mismatch at one M refuses only that M (production's op serves it); another M is still checked and served
  P3 a checked M is served without calling production's op again; below GATE_MIN_M production's op serves
Run (no GPU): docker run --rm --network none -v $PWD:/w -w /w --entrypoint python3 <image> tests/opt_kdamhc/test_idx_gate_permeter.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch  # noqa: E402
import glm53_prefill_quickwins as Q  # noqa: E402

FAIL = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


Q._STATE["items"] = frozenset({"idx_gate"})
Q._capturing = lambda: False
calls = {"orig": 0}
MODE = {"bad_m": set(), "nan": False}


def orig(x, weight, w32, handle, row0):
    calls["orig"] += 1
    r = x.float() @ w32
    if MODE["nan"]:
        r[0, 0] = float("nan")
    return r


def fake_fast(x, w32):
    r = x.float() @ w32
    if MODE["nan"]:
        r[0, 0] = float("nan")
    if x.shape[0] in MODE["bad_m"]:
        r[1, 1] += 1.0
    return r


Q.qw_head_gate = fake_fast
impl = Q._make_head_gate_impl(orig)
w32 = torch.randn(64, 32)


def run(M):
    x = torch.randn(M, 64).bfloat16()
    return impl(x, None, w32, 0, 0)


# P1 (opt-kdamhc-rev): a non-finite production result at the boot-time M proves nothing -> production's op serves,
# the M is neither checked nor refused; the next finite call at that M is checked and served
MODE["nan"] = True
f0, c0 = Q.STATS["idx_gate_fast"], calls["orig"]
run(16384)
check(Q.STATS["idx_gate_fast"] == f0 and calls["orig"] == c0 + 1 and 16384 not in Q._HG["checked"]
      and 16384 not in Q._HG["bad"] and not Q._HG["off"] and Q.STATS["idx_gate_deferred"] == 1,
      "P1 non-finite check at M=16384 deferred: production's op served, M neither checked nor refused")
MODE["nan"] = False
run(16384)
check(Q.STATS["idx_gate_fast"] == f0 + 1 and 16384 in Q._HG["checked"],
      "P1 the next finite call at M=16384 is checked and served")
# P4: only after HG_MAX_DEFER non-finite calls at one M does the NaN-aware comparison decide
MODE["nan"] = True
f0 = Q.STATS["idx_gate_fast"]
for _ in range(Q.HG_MAX_DEFER):
    run(15360)
check(Q.STATS["idx_gate_fast"] == f0 and 15360 not in Q._HG["checked"], "P4 HG_MAX_DEFER non-finite calls: deferred")
run(15360)
check(Q.STATS["idx_gate_fast"] == f0 + 1 and 15360 in Q._HG["checked"] and 15360 not in Q._HG["bad"],
      "P4 then both-NaN positions count as equal: served")
MODE["nan"] = False
# P2: a real mismatch at one M
MODE["bad_m"] = {12288}
f0, c0 = Q.STATS["idx_gate_fast"], calls["orig"]
run(12288)
check(12288 in Q._HG["bad"] and not Q._HG["off"] and Q.STATS["idx_gate_fast"] == f0,
      "P2 mismatch at M=12288 refuses that M only (no global off)")
c1 = calls["orig"]
run(12288)
check(calls["orig"] == c1 + 1 and Q.STATS["idx_gate_fast"] == f0, "P2 the refused M keeps production's op")
f1 = Q.STATS["idx_gate_fast"]
run(13824)
check(Q.STATS["idx_gate_fast"] == f1 + 1 and 13824 in Q._HG["checked"], "P2 another M (13,824) is checked and served")
# P3
c2 = calls["orig"]
run(13824)
check(calls["orig"] == c2 and Q.STATS["idx_gate_fast"] == f1 + 2, "P3 a checked M is served without production's op")
c3 = calls["orig"]
run(4289)
check(calls["orig"] == c3 + 1, "P3 M below GATE_MIN_M -> production's op")
# P5 (r16z5rev): an ALL-NaN production result (the dummy profile run can be all garbage) never decides, however many
# calls: production's op serves and the M stays unchecked; the first FINITE call at that M is compared bitwise (a real
# mismatch there is refused, not served unverified)
MODE["allnan"] = True
_o, _f = orig, fake_fast
def orig_allnan(x, weight, w32, handle, row0):
    calls["orig"] += 1
    return torch.full((x.shape[0], w32.shape[1]), float("nan")) if MODE["allnan"] else _o(x, weight, w32, handle, row0)
def fast_allnan(x, w32):
    return torch.full((x.shape[0], w32.shape[1]), float("nan")) if MODE["allnan"] else _f(x, w32)
Q.qw_head_gate = fast_allnan
impl2 = Q._make_head_gate_impl(orig_allnan)
f0 = Q.STATS["idx_gate_fast"]
for _ in range(3 * Q.HG_MAX_DEFER):
    impl2(torch.randn(14336, 64).bfloat16(), None, w32, 0, 0)
check(Q.STATS["idx_gate_fast"] == f0 and 14336 not in Q._HG["checked"] and 14336 not in Q._HG["bad"],
      "P5 all-NaN results never decide (M neither checked nor refused, production's op served)")
MODE["allnan"] = False
MODE["bad_m"] = {14336}
impl2(torch.randn(14336, 64).bfloat16(), None, w32, 0, 0)
check(14336 in Q._HG["bad"] and Q.STATS["idx_gate_fast"] == f0,
      "P5 the first finite call at that M is compared: a real mismatch is refused (not served unverified)")
print(f"RESULT: {'ALL OK' if not FAIL else f'{len(FAIL)} FAILED'}")
sys.exit(1 if FAIL else 0)
