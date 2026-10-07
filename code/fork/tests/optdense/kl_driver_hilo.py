"""opt-dense: how much of the W8A8 activation-rounding error would an 'outlier-channel hi+lo' correction remove?
(the residual r = x - dq(q(x)) of a few input channels S, quantized to e4m3 with the SAME per-token scale, added as
extra K columns: y = [q(x) | q(r_S)] @ [W ; W_S]^T - one GEMM with K + |S|, the existing epilogue.)

Runs tests/w8a82/kl_driver.py unchanged (same args; use --stats 1 with GLM53_DENSE_W8A8=1) but replaces its per-layer
stats function: at each layer's first served prefill call (real activations, first 4096 rows split into calibration
rows [0, h) and evaluation rows [h, 2h)) it reports, per channel budget c, e_hilo^2 / e_a^2 on the evaluation rows,
where e_a = ||Y_w8a8 - Y_prod|| (production = bf16 activations x the same fp8 weights) and S is chosen by
  - calib: top-c channels by the residual error energy sum_t r_tk^2 * ||W_k||^2 on the calibration rows
  - xen:   top-c by the activation energy sum_t x_tk^2 * ||W_k||^2 on the calibration rows (an offline-calibrated set)
  - oracle: top-c by the residual error energy on the evaluation rows themselves (upper bound)
and c = K ('full': every channel's residual, = hi+lo at 2x GEMM). No numerics of the engine change."""
from __future__ import annotations

import json
import os
import sys

sys.argv[0] = "/w/tests/w8a82/kl_driver.py"
sys.path.insert(0, "/w/tests/w8a82")
import kl_driver as KD  # noqa: E402  (parses the same argv)
import torch  # noqa: E402

BUDGETS = (32, 64, 128, 256, 512)


def one(self, layer, x, bias, pre, y_served, W):
    k = x.shape[-1]
    x2 = x.reshape(-1, k)[:4096].contiguous()
    h = x2.shape[0] // 2
    n, kk = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
    key = W._key(layer.weight, layer.weight_scale, kk)
    alpha = W.ALPHA[key][:n].float()
    w8 = torch.empty(n, kk, dtype=torch.float8_e4m3fn, device=x.device)
    W.repack(w8, layer.weight, n, kk)
    wdq = w8.float() * alpha[:, None]                                   # production's dequantized weight [n, k]
    del w8
    xf = x2.float()
    q, s = W.quant_per_token(x2)
    xq = q.float() * s
    r = xf - xq                                                          # activation rounding residual
    rq = (r / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s   # residual in e4m3, same per-token scale
    wn2 = wdq.pow(2).sum(0)                                              # ||W_k||^2 per input channel
    cal, ev = slice(0, h), slice(h, 2 * h)
    y_p = xf[ev] @ wdq.t()
    y_w = xq[ev] @ wdq.t()
    e_a2 = (y_w - y_p).pow(2).sum().item()
    crit = {"calib": r[cal].pow(2).sum(0) * wn2, "xen": xf[cal].pow(2).sum(0) * wn2,
            "oracle": r[ev].pow(2).sum(0) * wn2}
    rec = {"prefix": pre, "group": self.group, "n": n, "k": kk, "rows_eval": h,
           "rel_a": (e_a2 ** 0.5) / max(y_p.norm().item(), 1e-30)}
    tot_res = crit["oracle"].sum().item()
    for name, cv in crit.items():
        order = torch.argsort(cv, descending=True)
        for c in BUDGETS:
            if c >= kk:
                continue
            S = order[:c]
            y_c = y_w + rq[ev][:, S] @ wdq[:, S].t()
            rec[f"{name}{c}"] = (y_c - y_p).pow(2).sum().item() / max(e_a2, 1e-30)
            if name == "oracle":
                rec[f"share{c}"] = crit["oracle"][S].sum().item() / max(tot_res, 1e-30)
    y_full = y_w + rq[ev] @ wdq.t()
    rec["full"] = (y_full - y_p).pow(2).sum().item() / max(e_a2, 1e-30)
    KD.ST["rows"].append(rec)
    KD.log("HILO " + json.dumps(rec))
    del wdq, xf, xq, r, rq, y_p, y_w, y_c, y_full


KD.one = one
_orig_main = KD.main


def main():
    _orig_main_stats_off()


def _orig_main_stats_off():
    # kl_driver's STATSUM expects its own record keys; print ours instead after the run
    rows = KD.ST["rows"]
    try:
        _orig_main()
    except KeyError:
        pass
    if rows:
        by = {}
        for r_ in rows:
            by.setdefault(r_["prefix"].rsplit(".", 1)[-1] + "@" + r_["group"], []).append(r_)
        import statistics as S
        for g, rs in sorted(by.items()):
            cols = [f"{nm}{c}" for nm in ("calib", "xen", "oracle") for c in BUDGETS if f"{nm}{c}" in rs[0]] + ["full"]
            KD.log(f"HILOSUM {g} layers {len(rs)} k {rs[0]['k']}: " +
                   " ".join(f"{c}={S.median(x[c] for x in rs):.3f}" for c in cols))


if __name__ == "__main__":
    main()
