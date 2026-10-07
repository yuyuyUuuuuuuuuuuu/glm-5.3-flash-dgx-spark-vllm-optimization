"""Reviewer probe: does a block-mode kernel recompile when the batch shape changes? (GPU, nodeC, < 1 GiB)

The first block-keys overlay (R8/R9) added an integer argument `num_logits` to _compute_local_residual_mass_kernel.
Triton specializes integer arguments (value == 1, divisibility by 16), so a batch-dependent integer argument compiles
extra kernel variants at run time (a decode-loop stall the first time each class appears; measured 3 variants,
review/probe_jit_specialization.log). The overlay now bounds the row by tl.num_programs(0) instead
(review/probe_jit_specialization_after.log: 1 variant). This probe counts the compiled variants of every block-mode
kernel after rejection_sample calls with num_logits in {8 (the boot warmup's shape), 5, 16, 1, 15, 32, 40} for:
  prod        production overlays (resample-noise), stock kernels, block mode
  fix         + patch_spec_block_keys.py as in the worktree
  fix-np      the same with a `num_logits` argument rewritten to tl.num_programs(0) (no-op for the current overlay)
and times each first call; then runs every (target, draft) dtype pair of the image's JIT warmup in block mode. Run: tests/r16/gpu.sh python3 -u tests/blockverify/probe_jit_specialization.py
"""
import importlib.util
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
import vllm

dev = torch.device("cuda")
REPO = Path(__file__).resolve().parents[2]
REL = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REL_SPEC = "v1/worker/gpu/spec_decode/dflash2/speculator.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tmp = Path(tempfile.mkdtemp(prefix="bvjit_"))
os.environ["TRITON_CACHE_DIR"] = str(tmp / "triton_cache")     # cold cache: every variant really compiles
labels = ("prod", "fix", "fix-np")
for label in labels:
    for rel in (REL, REL_SPEC):
        (tmp / label / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(vllm.__file__).parent / rel, tmp / label / rel)
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"
rn = load("rn", REPO / "overlay/patch_spec_resample_noise.py")
for label in labels:
    rn.TARGET = tmp / label / REL
    rn.main(["x"])
os.environ["GLM53_REJECTION_METHOD"] = "block"
for label in ("fix", "fix-np"):
    bk = load(f"bk_{label}", REPO / "overlay/patch_spec_block_keys.py")
    bk.RSU, bk.SPEC = tmp / label / REL, tmp / label / REL_SPEC
    bk.main(["x"])
p = tmp / "fix-np" / REL
t = p.read_text()
if "num_logits,  # [glm53-block-keys]" in t:        # the shipped overlay: rewrite to tl.num_programs(0)
    t = t.replace("    num_logits,  # [glm53-block-keys]\n    BLOCK_SIZE: tl.constexpr,\n    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n):\n",
                  "    BLOCK_SIZE: tl.constexpr,\n    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n):\n", 1)
    t = t.replace("                num_logits,  # [glm53-block-keys]\n", "", 1)
    t = t.replace("has_next_row = logit_idx + 1 < num_logits", "has_next_row = logit_idx + 1 < tl.num_programs(0)", 1)
    p.write_text(t)
MODS = {label: load(f"rsu_{label.replace('-', '_')}", tmp / label / REL) for label in labels}
KERNELS = ("_compute_local_logits_stats_kernel", "_compute_cumulative_log_p_kernel", "_compute_local_residual_mass_kernel",
           "_rejection_kernel", "_resample_kernel", "_insert_resampled_kernel")


def variants(mod):
    out = {}
    for k in KERNELS:
        fn = getattr(mod, k)
        n = 0
        for dc in fn.device_caches.values():
            n += len(dc[0])
        out[k] = n
    return out


V, S = 9000, 7
g = torch.Generator(device=dev).manual_seed(1)
shapes = [(1, 7), (1, 4), (2, 7), (1, 0), (3, 4), (4, 7), (8, 4)]     # (requests, verified drafts) -> num_logits
for label in labels:
    mod = MODS[label]
    print(f"== {label}", flush=True)
    for R, n in shapes:
        rows = n + 1
        target = torch.randn(R * rows, V, device=dev, generator=g) * 2
        dl = torch.full((R, S, V), float("-inf"), device=dev)
        cand = torch.randint(0, V, (R, S, 16), device=dev, generator=g)
        dl.scatter_(2, cand, torch.randn(R, S, 16, device=dev, generator=g))
        ds = torch.empty(R * rows, dtype=torch.int32, device=dev)
        ds.view(R, rows)[:, 0] = 3
        if n:
            ds.view(R, rows)[:, 1:] = cand[:, :n, 0].to(torch.int32)
        pos = (torch.arange(rows, device=dev) + 1000).repeat(R)
        cu = torch.arange(R + 1, dtype=torch.int32, device=dev) * rows
        im = torch.arange(R, dtype=torch.int32, device=dev)
        em = im.repeat_interleave(rows)
        el = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        temp = torch.ones(R, device=dev)
        seeds = torch.arange(R, dtype=torch.int64, device=dev) + 11
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, ns = mod.rejection_sample(target, dl, ds, cu, pos, im, em, el, temp, seeds, S, None,
                                       use_fp64=False, use_block_verification=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mod.rejection_sample(target, dl, ds, cu, pos, im, em, el, temp, seeds, S, None,
                             use_fp64=False, use_block_verification=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        v = variants(mod)
        print(f"  R={R} n={n} num_logits={R * rows:3d}: first call {1000 * (t1 - t0):8.1f} ms, second {1000 * (t2 - t1):6.2f} ms"
              f" | compiled variants {v}", flush=True)
# every (target, draft) logits dtype pair the image's JIT warmup compiles (spec_decode_rejection_warmup.py), sampled
# (T 1.0) and greedy rows, block mode: the keyed kernels compile and run, and the fix module matches prod on the greedy row
print("== dtype pairs (block mode, 2 requests: one sampled, one greedy; n=7)", flush=True)
R, n = 2, 7
rows = n + 1
for tdt in (torch.bfloat16, torch.float32):
    for ddt in (torch.bfloat16, torch.float32):
        target = (torch.randn(R * rows, V, device=dev, generator=g) * 2).to(tdt)
        dl = torch.full((R, S, V), float("-inf"), device=dev)
        cand = torch.randint(0, V, (R, S, 16), device=dev, generator=g)
        dl.scatter_(2, cand, torch.randn(R, S, 16, device=dev, generator=g))
        dl = dl.to(ddt)
        ds = torch.empty(R * rows, dtype=torch.int32, device=dev)
        ds.view(R, rows)[:, 0] = 3
        ds.view(R, rows)[:, 1:] = cand[:, :n, 0].to(torch.int32)
        pos = (torch.arange(rows, device=dev) + 1000).repeat(R)
        cu = torch.arange(R + 1, dtype=torch.int32, device=dev) * rows
        im = torch.arange(R, dtype=torch.int32, device=dev)
        em = im.repeat_interleave(rows)
        el = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        temp = torch.tensor([1.0, 0.0], device=dev)
        seeds = torch.tensor([11, 22], dtype=torch.int64, device=dev)
        res = {}
        for label in labels:
            out, ns = MODS[label].rejection_sample(target, dl, ds, cu, pos, im, em, el, temp, seeds, S, None,
                                                   use_fp64=False, use_block_verification=True)
            torch.cuda.synchronize()
            res[label] = (out, ns)
        g_same = all(torch.equal(res[l][0][1, :int(res[l][1][1])], res["prod"][0][1, :int(res["prod"][1][1])])
                     and int(res[l][1][1]) == int(res["prod"][1][1]) for l in labels)
        print(f"  target {str(tdt)[6:]:8s} draft {str(ddt)[6:]:8s}: num_sampled "
              f"{ {l: res[l][1].tolist() for l in labels} } greedy row identical across modules: {g_same}", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
print("done")
