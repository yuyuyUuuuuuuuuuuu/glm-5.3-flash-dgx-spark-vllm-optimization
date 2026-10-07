"""Real GLM-5.3-Flash layer-10 routed experts (TP=2 rank-0 shard, production's stacked layout) on the GPU, finished by
production's own process_weights_after_loading (tests/harness.make_layer). The .npy files are written on the host by
tests/moee4m3/extract_layer.py; run the GPU scripts with GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3."""
from __future__ import annotations

import os
import sys

import numpy as np

# glm53_moe_e4m3.py and its extension live in the bundle's overlay dir (installed into site-packages only by
# overlay/patch_moe_e4m3.py when GLM53_MOE_E4M3=1): make them importable for the tests
_OVL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "overlay")
if _OVL not in sys.path:
    sys.path.insert(0, _OVL)
import torch

import harness as H

DIR = os.environ.get("MOEE4M3_LAYER_DIR", os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "moee4m3/layer10_tp2r0"))


class RealWeights:
    def __init__(self, dev, path: str = DIR):
        def ld(name):
            return torch.from_numpy(np.load(os.path.join(path, name + ".npy"))).to(dev)

        self.w13_trellis = ld("w13_trellis")
        self.w13_suh = ld("w13_suh")
        self.w13_svh = ld("w13_svh")
        self.w2_trellis = ld("w2_trellis")
        self.w2_suh = ld("w2_suh")
        self.w2_svh = ld("w2_svh")
        self.n = int(self.w13_trellis.shape[0])
        self.K = int(self.w13_suh.shape[-1])
        self.N = int(self.w13_svh.shape[-1])
        self.w13_mcg = torch.full((self.n, 2, 1), H.MCG_MARKER, dtype=torch.int32, device=dev)
        self.w2_mcg = torch.full((self.n, 1), H.MCG_MARKER, dtype=torch.int32, device=dev)
        self.path = path

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13_trellis, self.w13_suh, self.w13_svh,
                                                          self.w2_trellis, self.w2_suh, self.w2_svh))


def make_real_layer(prod, dev):
    W = RealWeights(dev)
    L = H.make_layer(prod, W)
    L._test_weights = W
    print(f"real layer: {W.path} ({W.n} experts, K {W.K}, N_loc {W.N}, {W.nbytes() / 2**30:.2f} GiB), shared gate/up "
          f"suh {bool(getattr(L, '_exl3_shared_w13_suh', False))}, tier {getattr(L, '_exl3_fat_effective_tier', None)}",
          flush=True)
    return L
