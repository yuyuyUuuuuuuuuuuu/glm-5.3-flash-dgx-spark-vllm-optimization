"""Greedy requests are untouched by the method and by the block-keys overlay (real kernels, nodeC, production image).

A mixed batch (every other request temperature 0, the rest 1.0), random full-vocab target rows (V = 154880), DFlash2
walk drafts (7 drafted, n = 4 / 5 / 7 verified), verified by: standard (production overlays), block (production
overlays), block + block-keys overlay. For every greedy request the emitted tokens and num_sampled must be bit-identical
across the three (the greedy branch of _rejection_kernel ignores u and the method; the greedy bonus row is an argmax),
and the drafts of greedy requests (SAMPLE_PROBABILISTIC applies temperature 0 -> argmax) must not depend on
BLOCK_KEYS. Sampled requests must differ somewhere between standard and block (the switch acts).
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

import vllm
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as SPEC_STOCK

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
REPO = Path(__file__).resolve().parents[2]
REL, REL_SPEC = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py", "v1/worker/gpu/spec_decode/dflash2/speculator.py"
V, S, TOPK, R = 154880, 7, 16, 64
FAIL = []


def check(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c:
        FAIL.append(m)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tmp = Path(tempfile.mkdtemp(prefix="bvg_"))
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
RSU_PROD, RSU_FIX = load("rp", tmp / "prod" / REL), load("rf", tmp / "fix" / REL)
SPEC_FIX = load("sf", tmp / "fix" / REL_SPEC)
g = torch.Generator(device=dev).manual_seed(9)
temp = torch.where(torch.arange(R, device=dev) % 2 == 0, 0.0, 1.0).float()
greedy = temp == 0
for n in (4, 5, 7):
    rows = n + 1
    diff_g = diff_draft = acted = 0
    for it in range(10):
        seeds = torch.randint(-(2**62), 2**62, (R,), device=dev, generator=g)
        P = torch.randint(1000, 200000, (R,), device=dev, generator=g)
        cand = torch.randint(0, V, (R * S, TOPK), device=dev, generator=g)
        scores = 2 * torch.randn(R * S, TOPK, TOPK, device=dev, generator=g)
        sp = (P.repeat_interleave(S) + 1 + torch.arange(S, device=dev).repeat(R)).contiguous()
        mp = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(S)
        outs = {}
        for name, spec_mod, keys in (("stock", SPEC_STOCK, False), ("keys", SPEC_FIX, True)):
            toks = torch.zeros(R * S, dtype=torch.int64, device=dev)
            real = torch.empty(R * S * TOPK, device=dev)
            kw = {"BLOCK_KEYS": True} if keys else {}
            spec_mod._selector_walk_kernel[(R,)](scores, cand, sp, mp, temp, seeds, toks, real, num_steps=S, top_k=TOPK,
                                                 BLOCK_K=TOPK, SAMPLE_PROBABILISTIC=True, USE_FP64=False, num_warps=1, **kw)
            dl = torch.full((R, S, V), float("-inf"), device=dev)
            cached = torch.zeros(R, S, TOPK, dtype=torch.int64, device=dev)
            spec_mod._cache_draft_logits_kernel[(R * S,)](dl, cached, cand, real, mp, dl.stride(0), dl.stride(1),
                                                          num_steps=S, top_k=TOPK, BLOCK_K=TOPK, num_warps=1)
            outs[name] = (toks.view(R, S), dl)
        diff_draft += int((outs["stock"][0][greedy] != outs["keys"][0][greedy]).sum())
        target = 3 * torch.randn(R * rows, V, device=dev, generator=g)
        boost = torch.rand(R, n, device=dev, generator=g) < 0.7
        # make the greedy drafts often right so greedy requests run several rows deep
        ds = torch.empty(R, rows, dtype=torch.int32, device=dev)
        ds[:, 0] = 7
        pos = (P.repeat_interleave(rows) + torch.arange(rows, device=dev).repeat(R)).contiguous()
        cu = torch.arange(R + 1, dtype=torch.int32, device=dev) * rows
        im = torch.arange(R, dtype=torch.int32, device=dev)
        em, el = im.repeat_interleave(rows), torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        res = {}
        for name, mod, blk, spec_name in (("standard", RSU_PROD, False, "stock"), ("block", RSU_PROD, True, "stock"),
                                          ("block+keys", RSU_FIX, True, "keys")):
            toks, dl = outs[spec_name]
            ds[:, 1:] = toks[:, :n].to(torch.int32)
            t = target.clone()
            # the draft is the target argmax on ~70 % of the rows, so greedy requests run several rows deep and reject too
            t.view(R, rows, V)[:, :n].scatter_(2, toks[:, :n, None], torch.where(boost, 40.0, -40.0)[..., None])
            sampled, ns = mod.rejection_sample(t, dl, ds.view(-1), cu, pos, im, em, el, temp, seeds, S, None,
                                               use_fp64=False, use_block_verification=blk)
            valid = torch.arange(S + 1, device=dev)[None, :] < ns[:, None]
            res[name] = (torch.where(valid, sampled, -1), ns.clone())
        for name in ("block", "block+keys"):
            diff_g += int((res[name][0][greedy] != res["standard"][0][greedy]).sum()) + int((res[name][1][greedy] != res["standard"][1][greedy]).sum())
        acted += int((res["block+keys"][0][~greedy] != res["standard"][0][~greedy]).any(dim=1).sum())
    check(diff_draft == 0, f"n={n}: greedy drafts identical with and without BLOCK_KEYS ({diff_draft} diffs)")
    check(diff_g == 0, f"n={n}: greedy requests bit-identical under standard, block and block+keys ({diff_g} diffs over 10 x {R // 2})")
    check(acted > 0, f"n={n}: the method acts on sampled requests ({acted} of {10 * R // 2} differ standard vs block+keys)")
shutil.rmtree(tmp, ignore_errors=True)
print("ALL PASSED" if not FAIL else f"FAILED: {len(FAIL)}")
sys.exit(1 if FAIL else 0)
