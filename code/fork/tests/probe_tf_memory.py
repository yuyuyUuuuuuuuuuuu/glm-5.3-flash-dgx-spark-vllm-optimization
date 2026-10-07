"""Non-KV device memory TF adds to a rank (docs/PRODUCTION_PLAN.md Phase 2 memory gate): production layers (n=288,
K=4096, N=1024, as on one TP=2 rank) built through production's process_weights_after_loading, first without TF, then
with TF installed (pre-flight, scratch, self-test and K2 self-test run in the build hook). The difference of the
PyTorch allocator's reserved bytes per layer is TF's cost; the AOT module's device code is loaded outside the allocator
(its .so size is printed as an upper bound)."""
from __future__ import annotations

import os

import torch

import harness as H

K, N, NEXP = 4096, 1024, 288


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    MiB = 2 ** 20

    def build(seed):
        W = H.Weights(NEXP, K, N, dev, seed=seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        layer = H.make_layer(prod, W)
        torch.cuda.synchronize()
        return (layer, W), (torch.cuda.memory_allocated() - a0, torch.cuda.max_memory_allocated() - a0,
                            torch.cuda.memory_reserved() - r0)

    def fmt(d):
        return f"allocated +{d[0] / MiB:.1f} MiB (peak +{d[1] / MiB:.1f}), reserved +{d[2] / MiB:.1f} MiB"

    keep = []
    for i in range(2):              # the first layer also allocates production's shared fused temps (cached after)
        lw, base = build(700 + i)
        keep.append(lw)
        print(f"without TF, layer {i}: production's process_weights_after_loading (weights excluded): {fmt(base)}")
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep["installed"], f"install: {rep}")
    rows = []
    for i in range(2):
        lw, d = build(710 + i)
        keep.append(lw)
        ck((0, lw[0]._exl3_ptrs["gate_trellis"].data_ptr()) in tf.REG, f"layer {i} not registered")
        rows.append(d)
        print(f"with TF, layer {i}: {fmt(d)}; TF's share: allocated +{(d[0] - base[0]) / MiB:.1f} MiB, transient peak "
              f"+{(d[1] - base[1]) / MiB:.1f} MiB")
    sc = next(iter(tf._SCRATCH.values()))
    so = tf.EXT_SOURCE.split(":", 1)[1]
    print(f"TF scratch {sc.nbytes() / MiB:.1f} MiB (shared by all layers of the device); AOT module {so} "
          f"{os.path.getsize(so) / MiB:.1f} MiB on disk (device code loaded outside the PyTorch allocator)")
    ck(rows[1][0] - base[0] <= 1 * MiB, "TF must add no persistent allocation per layer after the first one")
    integrate.uninstall(prodmod=prod, ext=xl)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
