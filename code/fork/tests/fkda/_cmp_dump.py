"""FKDA check 1 comparison: tests/fkda/check_rename.sh's two op dumps.

Run INSIDE the production image (tests/fkda/docker_cpu.sh mounts this repo at
/w and FKDA_SCRATCH at /fkda) as a FILE argument, not on stdin -- docker_cpu.sh
does not pass -i, so a `python3 -` heredoc silently reads nothing and would make
the check pass without comparing anything (the r16o review finding).

  python3 /w/tests/fkda/_cmp_dump.py <dump_fp32.pt> <dump_orig.pt> <mode> [rel_l2_max rel_inf_max]

mode bitwise (a stock 17a037d build, the r16n expectation): every output
  bit-equal (same shapes, dtypes, bytes).
mode precision (the fkda2 build dd1788c2..., shipped in the r16o kit's overlay/,
  docs/KDA_FLASHKDA2.md): same shapes/dtypes, all finite, NOT all bit-equal, and
  every output within the bf16-rounding-class gates rel-L2 <= rel_l2_max and
  rel-Linf <= rel_inf_max of the stock build's output.
"""
import sys

import torch

a = torch.load(sys.argv[1], weights_only=False)
b = torch.load(sys.argv[2], weights_only=False)
mode = sys.argv[3] if len(sys.argv) > 3 else "bitwise"
rel_l2_max = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
rel_inf_max = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
assert sorted(a) == sorted(b), (sorted(a), sorted(b))
ok = True
any_diff = False
worst_l2 = worst_inf = 0.0
for name in sorted(a):
    x, y = a[name], b[name]
    structural = x.shape == y.shape and x.dtype == y.dtype and bool(torch.isfinite(x).all())
    same = structural and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
    if mode == "precision":
        d = x.float() - y.float()
        rel_l2 = (d.norm() / y.float().norm()).item() if y.numel() else 0.0
        scale = y.float().abs().max().item()
        rel_inf = (d.abs().max() / (scale if scale > 0 else 1.0)).item()
        worst_l2, worst_inf = max(worst_l2, rel_l2), max(worst_inf, rel_inf)
        any_diff |= not same
        good = structural and rel_l2 <= rel_l2_max and rel_inf <= rel_inf_max
        print(f"{'ok ' if good else 'FAIL'} {name}: {tuple(x.shape)} {x.dtype} finite={structural} "
              f"bit-equal={same} rel-L2={rel_l2:.3e} rel-Linf={rel_inf:.3e}")
        ok &= good
    else:
        mx = (x.float() - y.float()).abs().max().item() if x.numel() and x.shape == y.shape else float("nan")
        print(f"{'ok ' if same else 'FAIL'} {name}: {tuple(x.shape)} {x.dtype} bit-equal={same} max|diff|={mx:.4g}")
        ok &= same
if mode == "precision":
    if not any_diff:
        print("FAIL every output bit-equal: not a precision build's behaviour (wrong .so dumped?)")
        ok = False
    print(f"(precision build deviations: worst rel-L2 {worst_l2:.3e}, worst rel-Linf {worst_inf:.3e}; "
          f"gates rel-L2 <= {rel_l2_max}, rel-Linf <= {rel_inf_max})")
print("RENAME EQUIVALENCE:", "ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
