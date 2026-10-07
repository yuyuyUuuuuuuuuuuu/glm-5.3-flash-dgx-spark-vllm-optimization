"""GPU cost of block vs standard verification on the real kernels at production shape (nodeC, production image).

Times RejectionSampler._verify (the real V2 Sampler.apply_sampling_params with temperature 1.0 / top_p 0.95 /
repetition_penalty 1.05, then rejection_sample) at the production vocab V = 154880, for R = 1, 2, 4 requests and
adaptive-K lengths n = 4, 5, 7 (n + 1 rows per request, num_speculative_steps 7), with LLM-like target logits
(Gaussian bulk + a head) and DFlash2-like draft logits (16 finite candidates). Variants: standard with production's
overlay (resample-noise =1) and block with the block-keys overlay. Also times the DFlash2 walk kernel with and
without BLOCK_KEYS. CUDA events, median of 7 x 50 calls after warmup. Informational (no pass/fail).
"""
import importlib.util
import os
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
import vllm  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as RSM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.sample.sampler import Sampler  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REL_RSU = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REL_SPEC = "v1/worker/gpu/spec_decode/dflash2/speculator.py"
V, S, TOPK = 154880, 7, 16


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tmp = Path(tempfile.mkdtemp(prefix="bvcost_"))
for rel in (REL_RSU, REL_SPEC):
    (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(vllm.__file__).parent / rel, tmp / rel)
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"
rn = load("rn", REPO / "overlay/patch_spec_resample_noise.py")
rn.TARGET = tmp / REL_RSU
rn.main(["x"])
RSU_PROD = load("rsu_prod", tmp / REL_RSU)
bk = load("bk", REPO / "overlay/patch_spec_block_keys.py")
bk.RSU, bk.SPEC = tmp / REL_RSU, tmp / REL_SPEC
os.environ["GLM53_REJECTION_METHOD"] = "block"
bk.main(["x"])
RSU_FIX = load("rsu_fix", tmp / REL_RSU)
SPEC_FIX = load("spec_fix", tmp / REL_SPEC)


def timeit(f, reps=50, rounds=7):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            f()
        b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / reps * 1000.0)
    return sorted(ts)[rounds // 2]


g = torch.Generator(device=dev).manual_seed(3)
print(f"V={V}; us per call (median of 7 x 50)")
for R in (1, 2, 4):
    rs_state = RequestState(R, 64, 4096, S, V, dev)
    smp = Sampler(max_num_reqs=R, vocab_size=V, device=dev, req_states=rs_state)
    for r in range(R):
        smp.add_request(r, 8, SamplingParams(temperature=1.0, top_p=0.95, repetition_penalty=1.05))
    smp.penalties_state._new_penalties_reqs.clear()
    smp.apply_staged_writes()
    smp.penalties_state.output_bin_counts[:, :64] = 1
    for n in (4, 5, 7):
        rows = n + 1
        logits = 2.5 * torch.randn(R * rows, V, device=dev, generator=g)
        head = torch.randint(0, V, (R * rows, 8), device=dev, generator=g)
        logits.scatter_(1, head, 8 + 4 * torch.rand(R * rows, 8, device=dev, generator=g))
        dl = torch.full((R, S, V), float("-inf"), device=dev)
        cand = torch.randint(0, V, (R, S, TOPK), device=dev, generator=g)
        dl.scatter_(2, cand, 3 * torch.randn(R, S, TOPK, device=dev, generator=g))
        ds = torch.empty(R, rows, dtype=torch.int32, device=dev)
        ds[:, 0] = 5
        ds[:, 1:] = cand[:, :n, 0].to(torch.int32)
        pos = (torch.randint(1000, 100000, (R, 1), device=dev, generator=g) + torch.arange(rows, device=dev)).view(-1)
        cu = (torch.arange(R + 1, dtype=torch.int32, device=dev) * rows)
        im = torch.arange(R, dtype=torch.int32, device=dev)
        em = im.repeat_interleave(rows)
        el = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        res = {}
        for name, mod, method in (("standard", RSU_PROD, "standard"), ("block", RSU_FIX, "block")):
            rs = RSM.RejectionSampler(smp, types.SimpleNamespace(num_speculative_tokens=S, rejection_sample_method=method,
                                                                 synthetic_acceptance_rates=None), dev)
            RSM.rejection_sample = mod.rejection_sample
            res[name] = timeit(lambda: rs._verify(logits, dl, ds.view(-1), pos, cu, im, np.arange(R), em, el))
        print(f"  R={R} n={n}: _verify standard {res['standard']:.1f} us, block {res['block']:.1f} us "
              f"(+{res['block'] - res['standard']:.1f} us)", flush=True)
        del logits, dl
        torch.cuda.empty_cache()
# the DFlash2 walk with / without row keys
R = 4
scores = torch.randn(R * S, TOPK, TOPK, device=dev)
cand = torch.randint(0, V, (R * S, TOPK), device=dev)
sp = torch.arange(R * S, device=dev, dtype=torch.int64) + 1000
mp = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(S)
temp = torch.ones(R, device=dev)
seeds = torch.randint(0, 2**62, (R,), device=dev)
toks = torch.zeros(R * S, dtype=torch.int64, device=dev)
real = torch.empty(R * S * TOPK, device=dev)
for label, kw in (("stock walk", {}), ("walk BLOCK_KEYS", {"BLOCK_KEYS": True})):
    t = timeit(lambda: SPEC_FIX._selector_walk_kernel[(R,)](scores, cand, sp, mp, temp, seeds, toks, real, num_steps=S, top_k=TOPK,
                                                            BLOCK_K=TOPK, SAMPLE_PROBABILISTIC=True, USE_FP64=False, num_warps=1, **kw))
    print(f"  DFlash2 {label} (R={R}): {t:.1f} us")
shutil.rmtree(tmp, ignore_errors=True)
