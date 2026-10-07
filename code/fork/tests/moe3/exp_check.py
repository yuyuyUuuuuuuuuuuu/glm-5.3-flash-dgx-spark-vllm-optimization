"""Exhaustive: exp_prod(a) == (float) exp((double) a) bit for bit for every float a in [-87, 88] (and the out-of-range
statement is the double one itself). Run: tests/gpu_run.sh python3 tests/moe3/exp_check.py"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "overlay"))
import torch  # noqa: E402
import glm53_moe_e4m3_ext as E  # noqa: E402


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


t0 = time.time()
cnt = torch.zeros(2, dtype=torch.int64, device="cuda")
# positive floats [0, 88): bit patterns 0 .. bits(88); negative (-87, -0]: 0x80000000 .. bits(-87)
ranges = [(0, bits(88.0) + 1), (0x80000000, bits(-87.0) + 1)]
total = 0
for b0, b1 in ranges:
    step = 1 << 26
    for s in range(b0, b1, step):
        E.exp_check(s, min(s + step, b1), cnt)
    total += b1 - b0
torch.cuda.synchronize()
bad, fb = cnt.tolist()
print(f"exp_check: {total} floats in [-87, 88] (incl. the range edges handled by the double statement): mismatches "
      f"{bad}, ambiguous-rounding fallbacks {fb} ({fb / total:.2e}); {time.time() - t0:.1f}s")
print("RESULT:", "PASS" if bad == 0 else "FAIL")
