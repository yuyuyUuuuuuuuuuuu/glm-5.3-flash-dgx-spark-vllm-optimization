"""compute-sanitizer target: one small rejection_sample call per variant with exact-size allocations.

Run: PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python3 tests/blockverify/sanitize_block.py
Shapes: 2 requests, adaptive-K n = 4 (5 rows each, num_speculative_steps 7) and n = 7 (8 rows), V = 9000, T = 1.0.
Variants: standard (production overlays), block (production overlays), block + block-keys overlay.
With n < num_speculative_steps the bonus row's expanded_local_pos (n) is < 7, so stock
_compute_local_residual_mass_kernel does not skip it and reads draft_sampled[logit_idx + 1]; for the last request that
index is one past the end of draft_sampled. This script makes that visible to memcheck.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

import vllm
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as STOCK  # noqa: F401

dev = torch.device("cuda")
REPO = Path(__file__).resolve().parents[2]
REL = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REL_SPEC = "v1/worker/gpu/spec_decode/dflash2/speculator.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tmp = Path(tempfile.mkdtemp(prefix="bvsan_"))
for label in ("prod", "fix"):
    for rel in (REL, REL_SPEC):
        (tmp / label / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(vllm.__file__).parent / rel, tmp / label / rel)
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"
rn = load("rn", REPO / "overlay/patch_spec_resample_noise.py")
for label in ("prod", "fix"):
    rn.TARGET = tmp / label / REL
    rn.main(["x"])
bk = load("bk", REPO / "overlay/patch_spec_block_keys.py")
bk.RSU, bk.SPEC = tmp / "fix" / REL, tmp / "fix" / REL_SPEC
os.environ["GLM53_REJECTION_METHOD"] = "block"
bk.main(["x"])
MODS = {"prod": load("rsu_prod", tmp / "prod" / REL), "fix": load("rsu_fix", tmp / "fix" / REL)}
variants = [("standard", "prod", False), ("block", "prod", True), ("block+keys", "fix", True)]
only = os.environ.get("SAN_ONLY")
V, S, R = 9000, 7, 2
g = torch.Generator(device=dev).manual_seed(1)
for n in (4, 7):
    rows = n + 1
    for name, mod, blk in variants:
        if only and name != only:
            continue
        target = torch.randn(R * rows, V, device=dev, generator=g) * 2
        dl = torch.full((R, S, V), float("-inf"), device=dev)
        cand = torch.randint(0, V, (R, S, 16), device=dev, generator=g)
        dl.scatter_(2, cand, torch.randn(R, S, 16, device=dev, generator=g))
        ds = torch.empty(R * rows, dtype=torch.int32, device=dev)          # exact size: R * rows
        ds.view(R, rows)[:, 0] = 3
        ds.view(R, rows)[:, 1:] = cand[:, :n, 0].to(torch.int32)
        pos = (torch.arange(rows, device=dev) + 1000).repeat(R)
        cu = torch.arange(R + 1, dtype=torch.int32, device=dev) * rows
        im = torch.arange(R, dtype=torch.int32, device=dev)
        em = im.repeat_interleave(rows)
        el = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        temp = torch.ones(R, device=dev)
        seeds = torch.tensor([11, 22], dtype=torch.int64, device=dev)
        out, ns = MODS[mod].rejection_sample(target, dl, ds, cu, pos, im, em, el, temp, seeds, S, None,
                                            use_fp64=False, use_block_verification=blk)
        torch.cuda.synchronize()
        print(f"n={n} {name}: num_sampled {ns.tolist()}", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
print("done")
