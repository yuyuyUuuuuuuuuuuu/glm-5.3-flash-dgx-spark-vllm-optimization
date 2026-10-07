"""Offline evaluation of per-step verify-trimming rules on GLM53_DEC_VTRIM_STATS records (glm53_vtrim_stats.py).

Record = [n drafts, a accepted, qmax_1..7, qd_1..7]. A rule may verify draft i only from qmax_i and qd_1..qd_{i-1}
(never qd_i: that would bias the proposal at row i). Rule evaluated here (calibrated on one half of the records,
measured on the other half):
    r_i = P(draft i accepted | drafts < i accepted, bin(qmax_i), bin(qd_{i-1}))       (bins from the fit half)
    survival s_i = r_1 * ... * r_i ; verify drafts 1..n' with n' = the last i with s_i >= tau (n' <= n)
    tokens = min(a, n') + 1 ; verified rows = n' + 1
Step cost model: step_ms = base + c_row * rows, base set so that the measured mean rows cost STEP_MS (default 75 ms,
production's average); c_row in {2.5, 5, 8} ms (docs/OPT_DECODE.md: corr40 model ~2.5, 0922 production sweep ~8).
Usage: python3 vtrim_policy_from_stats.py records.npy [STEP_MS]
"""
import sys

import numpy as np

NPOS = 7
arr = np.load(sys.argv[1])
STEP_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
arr = arr[arr[:, 0] > 0]
rng = np.random.default_rng(0)
perm = rng.permutation(len(arr))
fit, ev = arr[perm[: len(arr) // 2]], arr[perm[len(arr) // 2:]]
NB = 8
EDGES = np.linspace(0, 1, NB + 1)


def b(x):
    return np.clip(np.digitize(x, EDGES[1:-1]), 0, NB - 1)


# calibration table r[i, bin(qmax_i), bin(qd_{i-1}) or NB for i = 0]
num = np.zeros((NPOS, NB, NB + 1))
den = np.zeros((NPOS, NB, NB + 1))
n, a = fit[:, 0].astype(int), fit[:, 1].astype(int)
for i in range(NPOS):
    reach = (n > i) & (a >= i)
    bq = b(fit[:, 2 + i])
    bp = b(fit[:, 2 + NPOS + i - 1]) if i > 0 else np.full(len(fit), NB)
    acc = a > i
    np.add.at(den, (i, bq[reach], bp[reach]), 1)
    np.add.at(num, (i, bq[reach], bp[reach]), acc[reach])
prior = (num.sum((1, 2)) + 1) / (den.sum((1, 2)) + 2)
r_tab = (num + 4 * prior[:, None, None]) / (den + 4)        # shrink sparse cells to the position's mean


def evaluate(recs, tau):
    n, a = recs[:, 0].astype(int), recs[:, 1].astype(int)
    s = np.ones(len(recs))
    nprime = np.zeros(len(recs), dtype=int)
    alive = np.ones(len(recs), dtype=bool)
    for i in range(NPOS):
        bq = b(recs[:, 2 + i])
        bp = b(recs[:, 2 + NPOS + i - 1]) if i > 0 else np.full(len(recs), NB)
        s = s * r_tab[i, bq, bp]
        alive &= (n > i) & (s >= tau)
        nprime += alive
    tokens = np.minimum(a, nprime) + 1
    return tokens, nprime + 1


n_ev, a_ev = ev[:, 0].astype(int), ev[:, 1].astype(int)
base_tokens, base_rows = a_ev + 1, n_ev + 1
print(f"records {len(arr)} (fit {len(fit)}, eval {len(ev)}); eval: tokens/step {base_tokens.mean():.3f}, verified rows "
      f"{base_rows.mean():.3f}; P(accepted >= i): {[round(float((a_ev >= i).mean()), 3) for i in range(1, 8)]}")
for c_row in (2.5, 5.0, 8.0):
    base_ms = STEP_MS - c_row * base_rows.mean()
    tps0 = base_tokens.sum() / (base_ms * len(ev) + c_row * base_rows.sum())
    best = (0.0, 0.0, base_tokens.mean(), base_rows.mean())
    for tau in np.linspace(0.02, 0.6, 30):
        tk, rw = evaluate(ev, tau)
        tps = tk.sum() / (base_ms * len(ev) + c_row * rw.sum())
        if tps / tps0 - 1 > best[0]:
            best = (tps / tps0 - 1, tau, tk.mean(), rw.mean())
    orac_rows = np.minimum(a_ev, n_ev) + 1
    tps_or = base_tokens.sum() / (base_ms * len(ev) + c_row * orac_rows.sum())
    print(f"c_row {c_row:.1f} ms (base {base_ms:.1f} ms): best rule tau {best[1]:.2f} -> tokens/step {best[2]:.3f}, rows "
          f"{best[3]:.3f}, tokens/s {100 * best[0]:+.1f} %; oracle (verify exactly the accepted prefix) "
          f"{100 * (tps_or / tps0 - 1):+.1f} %")
