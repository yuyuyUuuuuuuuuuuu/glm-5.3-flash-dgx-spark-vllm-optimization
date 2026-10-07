"""Which native kernel does production's exllamav3_ext.exl3_moe launch in this process? (stock / thin-decode)

Builds one production layer (n=16 experts, K=4096, N=1024; TF_EXL3_BENCH_SHARED_SUH=1 -> gate/up share suh) through
production's process_weights_after_loading and lists the CUDA kernels one exl3_moe call launches (harness.kernel_names).
With the thin-decode build bound over the image's exllamav3_ext and GLM53_EXL3_MOE_FAST=1, the native dispatcher must
pick glm53_exl3_moe_fast_kernel<4, 256, shared_input> (shared_input = the up_suh table aliases gate_suh); otherwise
exl3_moe_kernel<...>. Exit 1 if the kernel is not the expected one.
"""
from __future__ import annotations

import os

import torch

import harness as H


def main():
    H.gpu_guard(2.0)
    xl = H.load_xl()
    prod = H.load_prod()
    ck = H.Checks()
    dev = torch.device("cuda", 0)
    shared = os.environ.get("TF_EXL3_BENCH_SHARED_SUH", "0") == "1"
    W = H.Weights(16, 4096, 1024, dev, seed=7, shared_suh=shared)
    layer = H.make_layer(prod, W)
    aliased = layer._exl3_ptrs["up_suh"] is layer._exl3_ptrs["gate_suh"]
    g = torch.Generator().manual_seed(3)
    x = torch.randn(8, 4096, generator=g).to(torch.bfloat16).to(dev)
    args = H.capture_args(prod, xl, x, H.random_ids(8, 16, 8, g, dev), H.random_weights(8, 8, g, dev), layer, 10.0)
    names = H.kernel_names(xl.exl3_moe, *H.with_out(args, torch.zeros(8, 4096, dtype=torch.float32, device=dev)))
    fast = os.environ.get("GLM53_EXL3_MOE_FAST") == "1" and hasattr(xl, "glm53_fast_moe_version")
    want = f"glm53_exl3_moe_fast_kernel<4, 256, {'true' if aliased else 'false'}>" if fast else "exl3_moe_kernel<"
    moe = [n for n in names if "moe" in n]
    print(f"exllamav3_ext {xl.__file__}; glm53_fast_moe_version "
          f"{xl.glm53_fast_moe_version() if hasattr(xl, 'glm53_fast_moe_version') else 'absent'}; GLM53_EXL3_MOE_FAST "
          f"{os.environ.get('GLM53_EXL3_MOE_FAST')!r}; shared suh {shared}, up_suh aliased {aliased}")
    print(f"exl3_moe launched: {moe}")
    ck(any(want in n for n in moe) and not (not fast and any("glm53" in n for n in moe)),
       f"expected a kernel containing {want!r}, got {moe}")
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
