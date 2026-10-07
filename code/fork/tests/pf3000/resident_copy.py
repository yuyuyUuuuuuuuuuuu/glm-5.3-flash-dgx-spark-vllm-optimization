"""PF3000 TEST C, layout option (i): actually allocate one resident standard-layout fp8 [N,K] copy per
production dense shape (all of them, as the real implementation would), and report the peak device
allocation above baseline (GiB/rank) and host MemAvailable before/after. GB10 is unified memory, so this
allocation competes with the KV pool exactly like a resident copy would.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/pf3000/resident_copy.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard  # noqa: E402
import torch  # noqa: E402

from bench_test_c import SHAPES  # noqa: E402


def main():
    gpu_guard(8.0)
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            print("host MemAvailable before:", line.split()[1], "kB")
    torch.cuda.init()
    base = torch.cuda.memory_allocated()
    peak = 0
    total = 0
    for name, (n, k, *_r) in SHAPES.items():
        calls = SHAPES[name][4]
        kept = []
        for _ in range(calls):   # every layer of this shape holds its own copy, as the real thing would
            kept.append((torch.randn(n, k, device="cuda") * 0.02).to(torch.float8_e4m3fn))
        total += sum(w.numel() for w in kept)
        peak = max(peak, torch.cuda.memory_allocated() - base)
        print(f"{name:15s} [{n}x{k}] x{calls} layers: resident copy {sum(w.numel() for w in kept) / 2**20:.1f} MiB (held)")
    print(f"TOTAL resident standard-layout fp8: {total / 2**30:.2f} GiB/rank "
          f"(peak simultaneously held above baseline {peak / 2**30:.2f} GiB) vs 0.5 GiB/rank budget")
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            print("host MemAvailable after:", line.split()[1], "kB")
    # and the per-rank KV-token price at production's 8,039 B/token (PREFILL3000 PLAN.md section 6)
    print(f"KV pool price at 8,039 B/token: {total / 8039:.0f} tokens of the 2.00M pool")
    print(f"VERDICT layout (i): resident copy {total / 2**30:.2f} GiB/rank -> "
          f"{'PASS' if total / 2**30 <= 0.5 else 'KILL'} against the 0.5 GiB/rank budget")


run_main(main)
