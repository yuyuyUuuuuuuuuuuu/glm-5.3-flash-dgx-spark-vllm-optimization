"""r16z5rev probe (CPU): does an ALL-NaN dummy profile run (M = 16384, HG_MAX_DEFER calls - one per MLA layer of the
single profile forward) mark that M 'checked' WITHOUT a single finite value compared? Then every later real 16384-row
call is served by the fast op unverified."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import glm53_prefill_quickwins as Q
Q._STATE["items"] = frozenset({"idx_gate"})
Q._capturing = lambda: False
calls = {"orig": 0, "fast": 0}
mode = {"nan": True, "fast_differs": True}
def orig(x, weight, w32, handle, row0):
    calls["orig"] += 1
    if mode["nan"]:
        return torch.full((x.shape[0], 4), float("nan"))
    return torch.ones(x.shape[0], 4)
def fast(x, w32):
    calls["fast"] += 1
    if mode["nan"]:
        return torch.full((x.shape[0], 4), float("nan"))
    return torch.ones(x.shape[0], 4) * (2.0 if mode["fast_differs"] else 1.0)   # a REAL mismatch on finite data
Q.qw_head_gate = fast
impl = Q._make_head_gate_impl(orig)
M = 16384
x = torch.zeros(M, 8)
for i in range(Q.HG_MAX_DEFER):
    impl(x, None, None, None, 0)
print("after", Q.HG_MAX_DEFER, "all-NaN calls: checked =", M in Q._HG["checked"], "bad =", M in Q._HG["bad"])
mode["nan"] = False
o0 = calls["orig"]
y = impl(x, None, None, None, 0)
print("first FINITE call at M=16384: production op called =", calls["orig"] > o0, "| served value", float(y[0, 0]),
      "(production's = 1.0, the differing fast op = 2.0)")
print("VERDICT:", "UNVERIFIED SERVE (defect)" if float(y[0, 0]) == 2.0 else "ok (the finite call was checked)")
