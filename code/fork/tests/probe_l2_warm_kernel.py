"""Probe: l2_warm (tf_exl3_moe_ext) throughput on 4 cold regions of the MoE sublayer's sizes (1.5 + 2.25 + 8 + 4 MiB)
vs the test touch kernel (tests/l2touch.py, one launch per region), blocks x unroll x load flavour; and the time of a
subsequent full read (L2 hits). Eager, CUDA events; 96 MiB flush before every trial."""
from __future__ import annotations

import statistics

import torch

import harness as H


def main():
    H.gpu_guard(4.0)
    tf = H.load_tf()
    E = tf.load_ext()
    import l2touch
    T = l2touch.ext()
    dev = torch.device("cuda", 0)
    sizes = [(24 * 16384 * 4), (288 * 4096 * 2), (8 << 20), (4 << 20)]
    regs = [torch.randint(0, 255, (s,), dtype=torch.uint8, device=dev) for s in sizes]
    flush = torch.randint(0, 2**31 - 1, (24 << 20,), dtype=torch.int32, device=dev)
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    tot = sum(sizes)
    variants = {}
    for b in (8, 16, 24, 48):
        variants[f"l2_warm b{b} u8"] = lambda b=b: E.l2_warm(regs, b, 8, sink)
        variants[f"l2_warm b{b} u8 no_alloc"] = lambda b=b: E.l2_warm(regs, b, -8, sink)
        variants[f"touch   b{b} u8 (4 launches)"] = lambda b=b: [T.touch(r, r.numel(), b, 8 << 8, sink) for r in regs]
    res = {k: [] for k in variants}
    after = {k: [] for k in variants}
    for it in range(15):
        for k, fn in (list(variants.items()) if it % 2 else list(variants.items())[::-1]):
            E.l2_warm([flush], 96, 8, sink)
            s, m, e = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            s.record()
            fn()
            m.record()
            E.l2_warm(regs, 96, 8, sink)
            e.record()
            torch.cuda.synchronize()
            res[k].append(s.elapsed_time(m) * 1000)
            after[k].append(m.elapsed_time(e) * 1000)
    for k in variants:
        w, a = statistics.median(res[k]), statistics.median(after[k])
        print(f"{k:32s}: warm {w:6.1f} us ({tot / w / 1e3:5.0f} GB/s); full read after it {a:5.1f} us", flush=True)
    H.report_peak(4.0)


if __name__ == "__main__":
    H.run_main(main)
