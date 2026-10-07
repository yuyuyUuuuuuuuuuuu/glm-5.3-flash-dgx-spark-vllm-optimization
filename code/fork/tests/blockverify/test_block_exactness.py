"""Exactness of speculative sampling, standard vs block verification, on production's real kernels, across steps.

Runs inside ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor (tests/gpu_run.sh, one GPU, <= 8 GiB).

What runs (per step, for R requests at once, exactly the production order):
  1. draft   : dflash2/speculator.py _selector_walk_kernel (probabilistic DFlash2 pairwise-selector walk, Gumbel key
               randint(seed, sample_pos - 1)) + _cache_draft_logits_kernel (fp32 draft-logits cache), 7 drafts per
               step (num_speculative_tokens 7); optionally replayed from a captured CUDA graph (production drafts
               inside the DFlash FULL graph: DFlashSpeculator.capture -> _generate_draft -> _sample_path).
  2. verify  : spec_decode/rejection_sampler.py RejectionSampler._verify with the real V2 Sampler
               (apply_sampling_params: logit bias, repetition penalty over prompt + output + draft prefix, temperature,
               min_p, top-k/top-p) and rejection_sampler_utils.rejection_sample; only the first n of the 7 drafts
               are verified (adaptive K: n in {4, 5, 7}, rows = n + 1, num_speculative_steps stays 7).
  3. commit  : emitted tokens appended to the output counts (penalties), the next step starts at P + num_sampled
               with the same per-request seed (production positions/keys).
The rejection sampler is not captured in production (GPUModelRunner.sample_tokens -> sample -> rejection_sampler runs
eagerly); the draft kernels are, so the CUDA-graph variant covers the captured part.

Target "model": A active tokens spread over a V-wide vocab (other logits -1e4); logits of the next token are a fixed
random function of the last two tokens (a 2nd-order Markov chain), then production's sampling params make it
history-dependent (repetition penalty) and truncated (top_p). Draft: q_k(. | previous token) from a noisy copy of
the target (context-dependent, worse with depth), on 16 DFlash2 candidates (A active + padding at -inf).
Ground truth: the processed next-token distribution of every state (last two tokens, set of seen tokens) computed
with the SAME Sampler.apply_sampling_params (float32 -> float64 softmax), then the exact probability of every
length-L token sequence. Every verified row's processed logits are checked against that table (the reference is what
the verifier used). Tests (chi-square, bins with expected < 5 merged):
  seq     : the first L emitted tokens (A**L cells) vs the exact sequence distribution
  step2   : the first token of step 2 given (state after step 1, step-1 num_sampled) vs the table -- the cross-step
            test (block verification's rows after the rejection point share (seed, pos) keys with the next step)
  law     : per verified step, the realized accepted length vs E[tau | drafts] of the variant's rule computed from the
            rows the verifier saw (standard: sum_k prod min(1,p/q); block: Sun et al. Alg. 2 h_i) -- the kernels
            implement the acceptance law that tests/blockverify/estimate_gain.py uses; the same drafts also give the
            Rao-Blackwellized block-vs-standard tokens/step gain on this model
Variants: std-prod   production today (resample-noise overlay =1, stock drafter, standard)
          blk-prod   rejection_sample_method=block with today's overlays
          blk-fix    block + overlay/patch_spec_block_keys.py (row-keyed randomness)
          std-fix    standard with the block-keys-patched modules: must be bit-identical to std-prod
          blk-fix-cg blk-fix with the draft kernels replayed from a CUDA graph: must be bit-identical to blk-fix
Configs: prod   (T 1.0, top_p 0.95, repetition_penalty 1.05; the production defaults), n = 4, 5, 7
         stress (T 0.8, top_p 0.9, repetition_penalty 1.5), n = 5
         textbook (context-independent p=(.1,.5,.4), q=(.5,.4,.1), no sampling params), n = 4, 7
         vocab  (prod params at the production vocab V=154880, fewer samples), n = 7
Exit non-zero on any failed check. BLOCKVERIFY_SCALE (default 1.0) scales the number of launches.
"""
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
assert avail > 40 * 2**30, f"host MemAvailable {avail / 2**30:.1f} GiB < 40"
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))

import vllm  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as RSM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402
from vllm.v1.worker.gpu.sample.sampler import Sampler  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as RSU_STOCK  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as SPEC_STOCK  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REL_RSU = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REL_SPEC = "v1/worker/gpu/spec_decode/dflash2/speculator.py"
S = 7            # num_speculative_tokens (drafts per step)
TOPK = 16        # DFlash2 selector_top_k
SCALE = float(os.environ.get("BLOCKVERIFY_SCALE", "1.0"))
ONLY = [x for x in os.environ.get("BLOCKVERIFY_ONLY", "").split(",") if x]
FAIL: list[str] = []
RESULTS: dict = {}


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


# ------------------------------------------------------------------ patched copies (never the image)
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TMP = Path(tempfile.mkdtemp(prefix="glm53_bv_"))
SITES = {}
for label in ("prod", "fix"):
    site = TMP / label
    for rel in (REL_RSU, REL_SPEC):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(vllm.__file__).parent / rel, site / rel)
    SITES[label] = site
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"   # production value
resample = load_module("glm53_patch_resample", REPO / "overlay/patch_spec_resample_noise.py")
blockkeys = load_module("glm53_patch_blockkeys", REPO / "overlay/patch_spec_block_keys.py")
print("== overlays on copies of the image files")
for label, site in SITES.items():
    resample.TARGET = site / REL_RSU
    check(resample.main(["x"]) == 0, f"{label}: patch_spec_resample_noise.py applied (GLM53_SPEC_RESAMPLE_INDEPENDENT=1)")
blockkeys.RSU, blockkeys.SPEC = SITES["fix"] / REL_RSU, SITES["fix"] / REL_SPEC
os.environ["GLM53_REJECTION_METHOD"] = "block"
check(blockkeys.main(["x"]) == 0, "fix: patch_spec_block_keys.py applied (GLM53_REJECTION_METHOD=block)")
RSU_PROD = load_module("glm53_rsu_prod", SITES["prod"] / REL_RSU)
RSU_FIX = load_module("glm53_rsu_fix", SITES["fix"] / REL_RSU)
SPEC_FIX = load_module("glm53_spec_fix", SITES["fix"] / REL_SPEC)
check(not hasattr(SPEC_STOCK, "_glm53_block_keys") and "[glm53-block-keys]" in (SITES["fix"] / REL_SPEC).read_text(),
      "fix drafter module carries the block-keys edit, the image module does not")


NB = int(os.environ.get("BLOCKVERIFY_BITWISE_LAUNCHES", "20"))
NRB = int(os.environ.get("BLOCKVERIFY_RB_LAUNCHES", "100"))   # launches with the acceptance-law check
BITWISE = {"std-fix": "std-prod", "blk-fix-cg": "blk-fix", "std-prod-cg": "std-prod"}
BITWISE_WHY = {"std-fix": "the block-keys modules are inert in standard mode",
               "blk-fix-cg": "draft kernels replayed from a captured CUDA graph == eager (block-keys modules)",
               "std-prod-cg": "draft kernels replayed from a captured CUDA graph == eager (production modules)"}


class Variant:
    def __init__(self, name, rsu, spec, method, graph=False):
        self.name, self.rsu, self.spec, self.method, self.graph = name, rsu, spec, method, graph
        self.block_keys = spec is SPEC_FIX and method == "block"   # DFlash2Speculator: config says block


VARIANTS = {
    "std-prod": Variant("std-prod", RSU_PROD, SPEC_STOCK, "standard"),
    "blk-prod": Variant("blk-prod", RSU_PROD, SPEC_STOCK, "block"),
    "blk-fix": Variant("blk-fix", RSU_FIX, SPEC_FIX, "block"),
    "std-fix": Variant("std-fix", RSU_FIX, SPEC_FIX, "standard"),
    "blk-fix-cg": Variant("blk-fix-cg", RSU_FIX, SPEC_FIX, "block", graph=True),
    "std-prod-cg": Variant("std-prod-cg", RSU_PROD, SPEC_STOCK, "standard", graph=True),
}


# ------------------------------------------------------------------ statistics
def chi2_sf(x: float, k: int) -> float:
    if k < 1:
        return float("nan")
    return float(torch.special.gammaincc(torch.tensor(k / 2, dtype=torch.float64), torch.tensor(x / 2, dtype=torch.float64)))


def chi2_merge(counts: np.ndarray, probs: np.ndarray):
    """Pearson chi2 with bins of expected < 5 merged; returns (x2, dof, p, TV, n)."""
    n = counts.sum()
    exp = probs * n
    order = np.argsort(exp)
    c, e = counts[order].astype(np.float64), exp[order]
    small = e < 5
    if small.any():
        c = np.append(c[~small], c[small].sum())
        e = np.append(e[~small], e[small].sum())
    keep = e > 0
    bad = (~keep) & (c > 0)
    x2 = float((((c - e) ** 2)[keep] / e[keep]).sum()) + (math.inf if bad.any() else 0.0)
    k = int(keep.sum()) - 1
    return x2, k, chi2_sf(x2, k) if math.isfinite(x2) else 0.0, 0.5 * float(np.abs(counts / max(n, 1) - probs).sum()), int(n)


# ------------------------------------------------------------------ the synthetic model
class Model:
    def __init__(self, name, V, act, tgt, drf, temp, top_p, rep, prompt, L, filler="lm"):
        self.name, self.V, self.A, self.L = name, V, len(act), L
        # logits of the non-active tokens (same for every row). "lm": a fixed N(-16, 1.5) tail (~0.1-1 % of the
        # mass) that a correct top_p 0.95 removes; its Gaussian bulk is what the image's triton top-p (Qrita pivot:
        # max_sample = mean + 10 std of the first tile) expects. "flat": -1e4 (no sampling params; zero mass).
        gf = torch.Generator().manual_seed(77)
        self.filler = (-16.0 + 1.5 * torch.randn(V, generator=gf)) if filler == "lm" else torch.full((V,), -1e4)
        self.filler[torch.tensor(act)] = 0.0
        self.act = torch.tensor(act, dtype=torch.int64)
        self.tgt = tgt.float()          # [A, A, A]: logits of the next token given (a2, a1)
        self.drf = drf.float()          # [S, A, A]: draft logits for draft index k given the previous token
        self.temp, self.top_p, self.rep = temp, top_p, rep
        self.prompt = prompt            # active indices; the last two are (a2, a1)
        self.lut = torch.full((V,), -1, dtype=torch.int64)
        self.lut[self.act] = torch.arange(self.A)
        self.pad_ids = torch.tensor([i for i in range(100, 100 + 64) if i not in act][: TOPK - self.A])


def make_sampler(model: Model, R: int):
    rs = RequestState(R, 64, 4096, S, model.V, dev)
    smp = Sampler(max_num_reqs=R, vocab_size=model.V, device=dev, req_states=rs)
    sp = SamplingParams(temperature=model.temp, top_p=model.top_p, repetition_penalty=model.rep)
    for r in range(R):
        smp.add_request(r, len(model.prompt), sp)
    smp.penalties_state._new_penalties_reqs.clear()   # prompt/output bins are set by the harness
    smp.apply_staged_writes()
    return smp


def processed_table(model: Model):
    """Processed next-token probabilities for every state (a2, a1, seen-mask), via Sampler.apply_sampling_params."""
    A = model.A
    states = [(a2, a1, m) for a2 in range(A) for a1 in range(A) for m in range(1 << A)
              if (m >> a2) & 1 and (m >> a1) & 1]
    idx = {s: i for i, s in enumerate(states)}
    NT = len(states)
    smp = make_sampler(model, NT)
    act = model.act.to(dev)
    logits = model.filler.to(dev).repeat(NT, 1)
    pb = smp.penalties_state
    pb.prompt_bin_mask.zero_()
    pb.output_bin_counts.zero_()
    for i, (a2, a1, m) in enumerate(states):
        logits[i, act] = model.tgt[a2, a1].to(dev)
        for t in range(A):
            if (m >> t) & 1:
                pb.output_bin_counts[i, int(model.act[t])] = 1
    ar = torch.arange(NT, dtype=torch.int32, device=dev)
    kw = dict(expanded_idx_mapping=ar, idx_mapping=ar, idx_mapping_np=np.arange(NT), pos=torch.zeros(NT, dtype=torch.int64, device=dev),
              input_ids=torch.zeros(NT, dtype=torch.int32, device=dev), expanded_local_pos=torch.zeros(NT, dtype=torch.int32, device=dev))
    proc = smp.apply_sampling_params(logits.clone(), **kw)                       # triton top-p (>= 8 rows, as in the runs)
    pre = smp.apply_sampling_params(logits.clone(), skip_top_k_top_p=True, **kw)
    torch_p = apply_top_k_top_p_pytorch(pre.clone(), None, smp.sampling_states.top_p.gpu[ar.long()]) if model.top_p < 1.0 else pre
    other = torch.ones(model.V, dtype=torch.bool, device=dev)
    other[act] = False
    other_max = float(proc[:, other].max()) if proc is not logits else float("-inf")
    same_mask = bool(torch.equal(torch.isinf(proc[:, act]), torch.isinf(torch_p[:, act])))
    if not same_mask and os.environ.get("BLOCKVERIFY_TOPP_DEBUG"):
        pre_p = torch.softmax(pre.double(), dim=-1)[:, act]
        for i in range(NT):
            a, b = torch.isinf(proc[i, act]), torch.isinf(torch_p[i, act])
            if not torch.equal(a, b):
                print(f"    state {states[i]}: pre-top-p probs {[round(x, 5) for x in pre_p[i].tolist()]} "
                      f"triton masked {a.tolist()} pytorch masked {b.tolist()}; triton non-active finite "
                      f"{int(torch.isfinite(proc[i, other]).sum())}, pytorch {int(torch.isfinite(torch_p[i, other]).sum())}")
    lp = proc[:, act].double()
    probs = torch.softmax(lp, dim=-1).cpu().numpy()
    masked = int(torch.isinf(proc[:, act]).any(dim=1).sum())
    return states, idx, probs, proc[:, act].clone(), dict(other_max=other_max, same_mask=same_mask, masked_states=masked, n_states=NT)


def exact_sequence_probs(model: Model, idx, probs):
    A, L = model.A, model.L
    seqs = np.array(np.meshgrid(*[np.arange(A)] * L, indexing="ij")).reshape(L, -1).T   # [A**L, L], row-major = cell id
    a2 = np.full(len(seqs), model.prompt[-2]); a1 = np.full(len(seqs), model.prompt[-1])
    m = np.full(len(seqs), sum(1 << t for t in set(model.prompt)))
    pr = np.ones(len(seqs))
    for t in range(L):
        st = np.array([idx[(x, y, z)] for x, y, z in zip(a2, a1, m)])
        pr *= probs[st, seqs[:, t]]
        a2, a1, m = a1, seqs[:, t], m | (1 << seqs[:, t])
    return pr


# ------------------------------------------------------------------ one batch of R requests, several steps
class Runner:
    def __init__(self, model: Model, R: int, n_verify: int):
        self.m, self.R, self.n = model, R, n_verify
        self.rows = n_verify + 1
        self.smp = make_sampler(model, R)
        V, A = model.V, model.A
        self.act = model.act.to(dev)
        self.lut = model.lut.to(dev)
        self.tgt = model.tgt.to(dev)
        # draft side (DFlash2 buffers)
        cand = torch.cat([model.act, model.pad_ids]).to(dev)
        self.cand = cand.repeat(R * S, 1).contiguous()                                  # [R*S, 16]
        sc = torch.full((S, TOPK, TOPK), float("-inf"))
        for k in range(1, S):
            sc[k, :A, :A] = model.drf[k]                                                 # prev index = active index
        self.scores = sc.to(dev).repeat(R, 1, 1).contiguous()                           # [R*S, 16, 16]
        self.drf0 = model.drf[0].to(dev)                                                 # [A(prev token), A]
        self.sample_pos = torch.zeros(R * S, dtype=torch.int64, device=dev)
        self.map_s = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(S)
        self.kstep = torch.arange(S, dtype=torch.int64, device=dev).repeat(R)
        self.temp = torch.full((R,), model.temp, dtype=torch.float32, device=dev)
        self.seeds = torch.zeros(R, dtype=torch.int64, device=dev)
        self.draft_tokens = torch.zeros(R * S, dtype=torch.int64, device=dev)
        self.realized = torch.empty(R * S * TOPK, dtype=torch.float32, device=dev)
        self.draft_logits = torch.full((R, S, V), float("-inf"), device=dev)
        self.cached_ids = torch.zeros(R, S, TOPK, dtype=torch.int64, device=dev)
        # verify side
        rows = self.rows
        self.raw = model.filler.to(dev).repeat(R * rows, 1)
        self.cu = (torch.arange(R + 1, dtype=torch.int32, device=dev) * rows).contiguous()
        self.idx_mapping = torch.arange(R, dtype=torch.int32, device=dev)
        self.idx_np = np.arange(R)
        self.exp_map = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(rows)
        self.exp_local = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        self.krow = torch.arange(rows, dtype=torch.int64, device=dev).repeat(R)
        self.graphs = {}

    # ---- draft kernels (eager or replayed from a captured CUDA graph)
    def _draft_launch(self, spec_mod, block_keys):
        kw = dict(num_steps=S, top_k=TOPK, BLOCK_K=TOPK, SAMPLE_PROBABILISTIC=True, USE_FP64=False, num_warps=1)
        if block_keys:
            kw["BLOCK_KEYS"] = True
        spec_mod._selector_walk_kernel[(self.R,)](
            self.scores, self.cand, self.sample_pos, self.map_s, self.temp, self.seeds, self.draft_tokens, self.realized, **kw)
        spec_mod._cache_draft_logits_kernel[(self.R * S,)](
            self.draft_logits, self.cached_ids, self.cand, self.realized, self.map_s,
            self.draft_logits.stride(0), self.draft_logits.stride(1), num_steps=S, top_k=TOPK, BLOCK_K=TOPK, num_warps=1)

    def draft(self, v: Variant):
        if not v.graph:
            self._draft_launch(v.spec, v.block_keys)
            return
        key = (id(v.spec), v.block_keys)
        if key not in self.graphs:
            self._draft_launch(v.spec, v.block_keys)       # compile outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._draft_launch(v.spec, v.block_keys)
            self.graphs[key] = g
        self.graphs[key].replay()

    def expected_tau(self, proc, ds, n):
        """E[tau | drafts] for both rules from what the verifier saw (processed target rows, cached draft logits):
        standard sum_k prod_{i<=k} min(1, p/q); block (Sun et al. Alg. 2) sum_k 1 - prod_{i>=k} (1 - h_i)."""
        R, rows, V = self.R, self.rows, self.m.V
        lp = torch.log_softmax(proc.float(), dim=-1).view(R, rows, V)
        lq = torch.log_softmax(self.draft_logits[:, :n].float() / self.m.temp, dim=-1)
        x = ds.view(R, rows)[:, 1:].long()
        ratio = torch.exp(lp[:, :n].gather(2, x[..., None])[..., 0] - lq.gather(2, x[..., None])[..., 0])
        e_std = torch.cumprod(torch.clamp(ratio, max=1.0), dim=1).sum(1)
        P = torch.ones(R, device=dev)
        hs = []
        for i in range(n):
            P = torch.clamp(P * ratio[:, i], max=1.0)
            if i < n - 1:
                r = torch.clamp(P[:, None] * torch.exp(lp[:, i + 1]) - torch.exp(lq[:, i + 1]), min=0.0).sum(1)
                den = r + 1.0 - P
                hs.append(torch.where(den > 0, r / torch.where(den > 0, den, torch.ones_like(den)), torch.ones_like(den)))
            else:
                hs.append(P)
        H = torch.stack(hs, 1)
        tail = torch.flip(torch.cumprod(torch.flip(1.0 - H, [1]), 1), [1])
        e_blk = (1.0 - tail).sum(1)
        return e_std.double(), e_blk.double()

    def launch(self, v: Variant, seeds: torch.Tensor, P0: torch.Tensor, record_steps=False, rb=None):
        m, R, n, rows, A, L = self.m, self.R, self.n, self.rows, self.m.A, self.m.L
        st = self.smp.sampling_states
        st.seeds.np[:R] = seeds.cpu().numpy()
        st.seeds.copy_to_uva()
        self.seeds.copy_(seeds)
        pb = self.smp.penalties_state
        pb.prompt_bin_mask.zero_()
        pb.output_bin_counts.zero_()
        for t in set(m.prompt):
            tok = int(m.act[t])
            bit = int(np.array([1 << (tok % 32)], dtype=np.uint32).view(np.int32)[0])
            pb.prompt_bin_mask[:, tok // 32] |= bit
        a2 = torch.full((R,), m.prompt[-2], dtype=torch.int64, device=dev)
        a1 = torch.full((R,), m.prompt[-1], dtype=torch.int64, device=dev)
        seen = torch.zeros(R, dtype=torch.int64, device=dev) + sum(1 << t for t in set(m.prompt))
        P = P0.clone()
        n_emit = torch.zeros(R, dtype=torch.int64, device=dev)
        out = torch.full((R, L + S + 1), -1, dtype=torch.int64, device=dev)
        steps = []
        rs = RSM.RejectionSampler(self.smp, types.SimpleNamespace(
            num_speculative_tokens=S, rejection_sample_method=v.method, synthetic_acceptance_rates=None), dev)
        RSM.rejection_sample = v.rsu.rejection_sample
        ar = torch.arange(R, device=dev)
        while int(n_emit.min()) < L:
            # 1. draft: step 0 conditioned on the last committed token, later steps on the previous draft
            self.sample_pos.copy_((P.repeat_interleave(S) + 1 + self.kstep))
            self.scores.view(R, S, TOPK, TOPK)[:, 0, 0, :A] = self.drf0[a1]
            self.draft(v)
            dt = self.draft_tokens.view(R, S)
            d_act = self.lut[dt[:, :n]]                                                  # [R, n] active indices
            # 2. verify rows: row j input = (last token, d_1..d_n)[j]; target logits from its last two tokens
            seq = torch.cat([a2[:, None], a1[:, None], d_act], dim=1)                    # [R, n+2]
            ctx2, ctx1 = seq[:, 0:rows], seq[:, 1:rows + 1]                               # [R, rows]
            self.raw[:, self.act] = self.tgt[ctx2.reshape(-1), ctx1.reshape(-1)]
            ds = torch.empty(R, rows, dtype=torch.int32, device=dev)
            ds[:, 0] = self.act[a1].to(torch.int32)
            ds[:, 1:] = dt[:, :n].to(torch.int32)
            pos = (P.repeat_interleave(rows) + self.krow).contiguous()
            proc, sampled, ns = rs._verify(self.raw, self.draft_logits, ds.view(-1), pos, self.cu, self.idx_mapping, self.idx_np,
                                           self.exp_map, self.exp_local)
            ns = ns.long()
            if rb is not None:     # acceptance law: realized tau vs E[tau | drafts] of the variant's rule
                e_std, e_blk = self.expected_tau(proc, ds, n)
                e = e_blk if v.method == "block" else e_std
                d = (ns - 1).double() - e
                rb["n"] += R; rb["d"] += float(d.sum()); rb["d2"] += float((d * d).sum())
                rb["e_std"] += float(e_std.sum()); rb["e_blk"] += float(e_blk.sum())
                g_ = e_blk - e_std
                rb["g"] += float(g_.sum()); rb["g2"] += float((g_ * g_).sum())
            if record_steps:
                # row states (seen incl. the draft prefix) for the reference check
                orm = seen[:, None].expand(R, rows).clone()
                for j in range(1, rows):
                    orm[:, j] = orm[:, j - 1] | (1 << d_act[:, j - 1])
                steps.append(dict(proc=proc[:, self.act].view(R, rows, A).clone(), c2=ctx2.clone(), c1=ctx1.clone(), seen=orm))
            # 3. commit
            cols = torch.arange(S + 1, device=dev)[None, :]
            valid = cols < ns[:, None]
            sampled = torch.where(valid, sampled, 0)          # entries past num_sampled are uninitialized
            tok_act = torch.where(valid, self.lut[sampled], -1)
            assert bool((tok_act[valid] >= 0).all()), "emitted a token outside the active set"
            dest = (n_emit[:, None] + cols).clamp(max=out.shape[1] - 1)
            write = valid & (n_emit[:, None] + cols < out.shape[1])
            out[ar[:, None].expand_as(dest)[write], dest[write]] = tok_act[write]
            rr = ar[:, None].expand(R, S + 1)[valid]
            pb.output_bin_counts.index_put_((rr, sampled[valid]), torch.ones_like(rr, dtype=torch.int32), accumulate=True)
            full = torch.cat([seq[:, :2], tok_act], dim=1)                               # last two of (a2, a1, emitted)
            last = ns + 1
            a1n = full[ar, last]
            a2n = full[ar, last - 1]
            for j in range(S + 1):
                seen = seen | torch.where(valid[:, j], 1 << tok_act[:, j].clamp(min=0), torch.zeros_like(seen))
            a2, a1 = a2n, a1n
            P = P + ns
            n_emit = n_emit + ns
            steps.append(dict(ns=ns.clone())) if not record_steps else steps[-1].update(ns=ns.clone())
        return out[:, :L], steps


def rand_seeds(g, R):
    return torch.randint(-(2**62), 2**62, (R,), generator=g, dtype=torch.int64).to(dev)


def rand_pos(g, R):
    return torch.randint(1000, 200000, (R,), generator=g, dtype=torch.int64).to(dev)


def run_config(model: Model, n_list, variants, R, launches, g):
    print(f"\n== config {model.name}: V={model.V} A={model.A} L={model.L} T={model.temp} top_p={model.top_p} "
          f"rep={model.rep}; R={R} x {launches} launches per (variant, n)")
    states, idx, probs, proc_tab, info = processed_table(model)
    print(f"  table: {info['n_states']} states, {info['masked_states']} with a top-p-masked active token; "
          f"max non-active processed logit {info['other_max']}; triton top-p mask == pytorch top-p mask: {info['same_mask']}")
    check(info["same_mask"], f"{model.name}: triton and pytorch top-p masks agree on every state (production uses both: "
          ">= 8 rows triton, < 8 rows pytorch)")
    if model.top_p < 1.0:
        check(info["other_max"] == float("-inf"), f"{model.name}: top_p removes every non-active token (max processed "
              f"non-active logit {info['other_max']})")
    else:
        check(info["other_max"] < -1e3, f"{model.name}: non-active tokens carry no mass (flat filler)")
    exact = exact_sequence_probs(model, idx, probs)
    A, L = model.A, model.L
    pow_ = torch.tensor([A ** (L - 1 - t) for t in range(L)], device=dev)
    st_index = torch.full((A, A, 1 << A), -1, dtype=torch.int64)
    for s, i in idx.items():
        st_index[s] = i
    st_index = st_index.to(dev)
    probs_t = torch.tensor(probs, device=dev)
    for n in n_list:
        runner = Runner(model, R, n)
        res_n = {}
        outs_cache = {}
        for vname in variants:
            v = VARIANTS[vname]
            if vname in BITWISE:
                ref_name = BITWISE[vname]
                nb = min(launches, NB)
                gv = torch.Generator().manual_seed(1000 + n)
                diff = 0
                for it in range(nb):
                    seeds, P0 = rand_seeds(gv, R), rand_pos(gv, R)
                    out, steps = runner.launch(v, seeds, P0)
                    ns_all = torch.stack([sd["ns"] for sd in steps])
                    ro, rns = outs_cache[(ref_name, it)]
                    diff += int(not (torch.equal(out, ro) and torch.equal(ns_all, rns)))
                check(diff == 0, f"{model.name} n={n}: {vname} bit-identical to {ref_name} over {nb} launches x {R} requests "
                      f"(emitted tokens and every step's num_sampled): {BITWISE_WHY[vname]} ({diff} launches differ)")
                continue
            counts = torch.zeros(A ** L, dtype=torch.int64, device=dev)
            c_step2 = torch.zeros(S + 2, len(states), A, dtype=torch.int64, device=dev)   # [ns1, state, token]
            acc = dict(steps=0, tokens=0)
            rb = dict(n=0, d=0.0, d2=0.0, e_std=0.0, e_blk=0.0, g=0.0, g2=0.0)
            ref_bad = 0
            t0 = time.time()
            gv = torch.Generator().manual_seed(1000 + n)       # same seeds/positions for every variant
            for it in range(launches):
                seeds, P0 = rand_seeds(gv, R), rand_pos(gv, R)
                out, steps = runner.launch(v, seeds, P0, record_steps=(it == 0), rb=rb if it < NRB else None)
                if it == 0:
                    for sd in steps:
                        if "proc" not in sd:
                            continue
                        si = st_index[sd["c2"], sd["c1"], sd["seen"]]
                        ref = proc_tab[si]                                                   # [R, rows, A]
                        diff = ~((sd["proc"] == ref) | (torch.isinf(sd["proc"]) & torch.isinf(ref)))
                        ref_bad += int(diff.sum())
                if it < NB:
                    outs_cache[(vname, it)] = (out.clone(), torch.stack([sd["ns"] for sd in steps]))
                for sd in steps:
                    acc["tokens"] += int(sd["ns"].sum())
                counts += torch.bincount((out * pow_).sum(1), minlength=A ** L)
                # step-2 first token given (state after step 1, ns1)
                ns1 = steps[0]["ns"]
                has2 = ns1 < L
                y = out
                # state after the first ns1 tokens: replay (a2, a1, seen)
                a2 = torch.full((R,), model.prompt[-2], dtype=torch.int64, device=dev)
                a1 = torch.full((R,), model.prompt[-1], dtype=torch.int64, device=dev)
                sm = torch.zeros(R, dtype=torch.int64, device=dev) + sum(1 << t for t in set(model.prompt))
                for t in range(L):
                    upd = t < ns1
                    tok = y[:, t]
                    a2 = torch.where(upd, a1, a2); a1 = torch.where(upd, tok, a1); sm = torch.where(upd, sm | (1 << tok), sm)
                si = st_index[a2, a1, sm]
                first2 = y[torch.arange(R, device=dev), ns1.clamp(max=L - 1)]
                c_step2.index_put_((ns1[has2], si[has2], first2[has2]), torch.ones(int(has2.sum()), dtype=torch.int64, device=dev), accumulate=True)
                acc["steps"] += len(steps) * R
            torch.cuda.synchronize()
            dt_ = time.time() - t0
            cnt = counts.cpu().numpy()
            x2, dof, pv, tv, N = chi2_merge(cnt, exact)
            # step-2 conditional test: sum of per-(ns1, state) Pearson chi2 with merged small bins
            cs = c_step2.cpu().numpy()
            X2 = 0.0; DOF = 0; n2 = 0; worst = (1.0, None)
            for j in range(S + 2):
                for si_ in range(len(states)):
                    c = cs[j, si_]
                    if c.sum() < 20:
                        continue
                    a, b, pp, _, nn = chi2_merge(c, probs[si_])
                    if b >= 1:
                        X2 += a; DOF += b; n2 += nn
                        if pp < worst[0]:
                            worst = (pp, (j, states[si_]))
            p2 = chi2_sf(X2, DOF) if DOF > 0 else float("nan")
            res_n[vname] = dict(seq_p=pv, seq_tv=tv, seq_chi2=x2, seq_dof=dof, N=N, step2_p=p2, step2_chi2=X2, step2_dof=DOF,
                                step2_n=n2, ref_mismatch=ref_bad, secs=round(dt_, 1),
                                tokens_per_step=acc["tokens"] / acc["steps"])
            nr = max(rb["n"], 1)
            md = rb["d"] / nr
            sd = math.sqrt(max(rb["d2"] / nr - md * md, 1e-30))
            mg = rb["g"] / nr
            sg = math.sqrt(max(rb["g2"] / nr - mg * mg, 1e-30))
            res_n[vname].update(rb_steps=rb["n"], rb_mean_tau_minus_expected=md, rb_z=md / (sd / math.sqrt(nr)),
                                rb_e_std=rb["e_std"] / nr, rb_e_blk=rb["e_blk"] / nr,
                                rb_gain_pct=100 * mg / (1 + rb["e_std"] / nr), rb_gain_pct_se=100 * sg / math.sqrt(nr) / (1 + rb["e_std"] / nr))
            print(f"    acceptance law ({v.method} rule): mean(tau - E[tau|drafts]) = {md:+.5f} (z={md / (sd / math.sqrt(nr)):+.2f}, "
                  f"{rb['n']} request-steps); same drafts: E[tau] standard {rb['e_std'] / nr:.4f} block {rb['e_blk'] / nr:.4f} "
                  f"-> tokens/step {100 * mg / (1 + rb['e_std'] / nr):+.2f} % (SE {100 * sg / math.sqrt(nr) / (1 + rb['e_std'] / nr):.2f})", flush=True)
            print(f"  n={n} {vname:12s} N={N} seq chi2={x2:.1f}/{dof} p={pv:.3g} TV={tv:.4f} | step2|(ns1,state) "
                  f"chi2={X2:.1f}/{DOF} p={p2:.3g} (N={n2}; worst cell p={worst[0]:.2g} at ns1,state={worst[1]}) | "
                  f"tokens/step {acc['tokens'] / acc['steps']:.4f} | processed-logit mismatches vs table: {ref_bad} | "
                  f"{dt_:.0f}s", flush=True)
        RESULTS[f"{model.name}/n{n}"] = res_n
        del runner
        torch.cuda.empty_cache()
    return RESULTS


# ------------------------------------------------------------------ configs
def lm_like(seed, A, sig_draft):
    gg = torch.Generator().manual_seed(seed)
    # mixed-sign logits with a dominant token per context (so top_p truncates the tail in many states)
    base = torch.randn(A, A, A, generator=gg) * 1.6
    base += torch.nn.functional.one_hot(torch.randint(0, A, (A, A), generator=gg), A) * 1.5
    drf = torch.empty(S, A, A)
    for k in range(S):
        drf[k] = base.mean(0) + sig_draft * (1 + 0.25 * k) * torch.randn(A, A, generator=gg)
    return base, drf


g = torch.Generator().manual_seed(20260929)
cfgs = []
V_SMALL = 9000                                  # 2 rejection vocab blocks of 8192, 9 resample blocks of 1024
ACT4 = [11, 4100, 8300, 8990]
tgt, drf = lm_like(1, 4, 0.8)
cfgs.append((Model("prod", V_SMALL, ACT4, tgt, drf, 1.0, 0.95, 1.05, [1, 0], 5), [4, 5, 7], 1024, int(1000 * SCALE)))
tgt2, drf2 = lm_like(2, 4, 1.0)
cfgs.append((Model("stress", V_SMALL, ACT4, tgt2, drf2, 0.8, 0.9, 1.5, [2, 3], 5), [5], 1024, int(600 * SCALE)))
p3 = torch.log(torch.tensor([0.1, 0.5, 0.4])); q3 = torch.log(torch.tensor([0.5, 0.4, 0.1]))
cfgs.append((Model("textbook", V_SMALL, [11, 4100, 8990], p3.expand(3, 3, 3).clone(), q3.expand(S, 3, 3).clone(),
                   1.0, 1.0, 1.0, [0, 1], 6, filler="flat"), [4, 7], 1024, int(500 * SCALE)))
ACT_BIG = [11, 40000, 100003, 154870]
cfgs.append((Model("vocab", 154880, ACT_BIG, tgt, drf, 1.0, 0.95, 1.05, [1, 0], 5), [7], 128, int(400 * SCALE)))

ALL_V = ["std-prod", "blk-prod", "blk-fix", "std-fix", "blk-fix-cg", "std-prod-cg"]   # references before bitwise variants
t_all = time.time()
for model, ns_, R, launches in cfgs:
    if ONLY and model.name not in ONLY:
        continue
    vs = ALL_V if model.name == "prod" else ["std-prod", "blk-prod", "blk-fix", "std-fix"]
    run_config(model, ns_, vs, R, max(2, launches), g)

# ------------------------------------------------------------------ verdicts
print("\n== verdicts (seq and step2 tests; a PASS needs p > 1e-4 for exact variants)")
for key, res in RESULTS.items():
    for vname in ("std-prod", "blk-fix"):
        if vname in res:
            r = res[vname]
            check(r["seq_p"] > 1e-4 and (r["step2_p"] != r["step2_p"] or r["step2_p"] > 1e-4) and r["ref_mismatch"] == 0,
                  f"{key} {vname}: exact (seq p={r['seq_p']:.3g}, step2 p={r['step2_p']:.3g}, reference mismatches {r['ref_mismatch']})")
            check(abs(r["rb_z"]) < 4.5, f"{key} {vname}: accepted lengths follow the {VARIANTS[vname].method} rule's law "
                  f"(mean tau - E[tau|drafts] = {r['rb_mean_tau_minus_expected']:+.5f}, z={r['rb_z']:+.2f})")
    if "blk-prod" in res:
        r = res["blk-prod"]
        print(f"  {key} blk-prod (today's overlays + block): seq p={r['seq_p']:.3g} TV={r['seq_tv']:.4f}, step2 p={r['step2_p']:.3g}")
    if "std-prod" in res and "blk-fix" in res:
        a_, b_ = res["std-prod"]["tokens_per_step"], res["blk-fix"]["tokens_per_step"]
        print(f"  {key} tokens/step: standard {a_:.4f}, block {b_:.4f} ({100 * (b_ / a_ - 1):+.2f} %)")
tb = [k for k in RESULTS if k.startswith("textbook")]
for k in tb:
    r = RESULTS[k]["blk-prod"]
    check(min(r["seq_p"], r["step2_p"]) < 1e-12, f"{k}: block with today's overlays is NOT exact across steps "
          f"(min p={min(r['seq_p'], r['step2_p']):.3g}) -- the bias the block-keys patch removes")
print(f"\ntotal {time.time() - t_all:.0f}s")
out_json = os.environ.get("BLOCKVERIFY_JSON")
if out_json:
    Path(out_json).write_text(json.dumps(RESULTS, indent=1, default=float))
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL PASSED' if not FAIL else 'FAILED: ' + str(len(FAIL))}")
for f in FAIL:
    print("  -", f)
sys.exit(1 if FAIL else 0)
