"""A: shared-noise bias of probabilistic-draft rejection sampling, on production's real Triton kernels.

Runs inside ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor (tests/gpu_run.sh), one GPU, <= 8 GiB.

Pipeline per launch (exactly the production order of DFlash2 + the V2 rejection sampler):
  1. draft: vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py::_selector_walk_kernel
     (Gumbel key = randint(seed, sample_pos - 1) = the verification row's pos) and
     ::_cache_draft_logits_kernel (fp32 draft-logits cache: realized scores on 16 candidates, -inf elsewhere)
  2. verify: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py::rejection_sample
     (standard Leviathan test u = tl.rand(seed, pos); residual resample; bonus token)
  3. non-speculative reference sample: vllm/v1/worker/gpu/sample/gumbel.py::gumbel_sample at the same (seed, pos)
Variants: STOCK (image file), PATCHED (overlay/patch_spec_resample_noise.py applied to a COPY of the image file,
GLM53_SPEC_RESAMPLE_INDEPENDENT=1), PATCHED_OFF (same copy, =0), and STOCK with the draft walk keyed by an
independent seed ("independent noise": proves the stock verifier is exact when the noise is not shared).
Real vocab: V = 154880 (GLM-5.3-Flash / DFlash2), temperature 1.0 unless stated.

Asserts (exit non-zero on any failure):
  patch script: idempotent, fails closed on drift / partial patch / invalid env, and on an UNSET
     GLM53_SPEC_RESAMPLE_INDEPENDENT (no default: a rank without the variable must not start), both in the script
     and at import of the patched module.
  E1 (p=(.1,.5,.4), q=(.5,.4,.1), 1 draft): STOCK output != p (chi2 p < 1e-12); independent-noise STOCK == p;
     PATCHED == p (chi2 p > 1e-4); acceptance decisions and bonus tokens bit-identical STOCK vs PATCHED;
     acceptance rate == sum min(p,q) = 0.6 within 4 SE; bonus distribution == p_bonus; PATCHED_OFF == STOCK bitwise;
     an ACCEPTED draft is not the same thing as the seeded non-speculative sample (agreement among accepted < 0.99:
     the draft shares the non-speculative noise vector, but it is argmax(log q + G), not argmax(log p + G)).
  E2 (production shape: 7 drafts, top-16 pairwise selector, LLM-like target over the full vocab; verified length
     n = 7 and n = 4 (adaptive-K), per verified row): STOCK biased on some row; PATCHED and independent-noise
     unbiased on every row incl. the bonus row; accepted lengths bit-identical STOCK vs PATCHED.
  E3 (alternative A, measured only): greedy drafts (one-hot q) on the STOCK verifier are exact on every row; prints
     mean accepted drafts per step for greedy vs probabilistic drafts on the same synthetic model.
  E4 (two consecutive steps, 3 drafts, context-independent target p=(.1,.5,.4), q=(.5,.4,.1); step 2 starts at
     P + num_sampled with the same seed, as in production): with rejection_sample_method="standard", PATCHED is
     exact across steps (step-2 first token == p given every step-1 outcome, step-2 length independent of step-1
     length). With "block", PATCHED is exact within a step but NOT across steps (min chi2 p < 1e-12): rows after
     the rejection point consume (seed, pos) keys that the next step reuses. Control: block + PATCHED with a fresh
     seed for step 2 is exact. The patched module logs its block-mode warning exactly once, never in standard mode.
  G  (mixed greedy/sampled batch): greedy requests bit-identical STOCK vs PATCHED; PATCHED_OFF == STOCK bitwise.
"""
import importlib.util
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
assert avail > 40 * 2**30, f"host MemAvailable {avail / 2**30:.1f} GiB < 40"
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))

import vllm  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as STOCK  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (  # noqa: E402
    _cache_draft_logits_kernel,
    _selector_walk_kernel,
)

V = 154880
TOPK = 16
REL = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
REPO = Path(__file__).resolve().parents[2]
FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


# ------------------------------------------------------------------ patched copies (never the image)
TMP = Path(tempfile.mkdtemp(prefix="glm53_rsu_"))
site = TMP / "site"
(site / REL).parent.mkdir(parents=True)
shutil.copy(Path(vllm.__file__).parent / REL, site / REL)
os.environ["GLM53_SITE"] = str(site)
spec = importlib.util.spec_from_file_location("glm53_patch", REPO / "overlay/patch_spec_resample_noise.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
print("== patch script on a copy of the image file")
stock_src = (site / REL).read_text()
os.environ.pop("GLM53_SPEC_RESAMPLE_INDEPENDENT", None)
try:
    patch.main(["x", "--preflight"])
    check(False, "unset env value must fail closed")
except SystemExit as e:
    check("must be set to 0 or 1" in str(e) and (site / REL).read_text() == stock_src, f"unset env value fails closed, file untouched ({e})")
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"
check(patch.main(["x", "--preflight"]) == 0 and (site / REL).read_text() == stock_src, "--preflight leaves the file untouched")
check(patch.main(["x"]) == 0 and (site / REL).read_text() != stock_src, "first run patches")
once = (site / REL).read_text()
check(patch.main(["x"]) == 0 and (site / REL).read_text() == once, "second run is a no-op (idempotent)")
for bad_label, mutate in (("drifted anchor", lambda s: s.replace("    # Resample the rejected/bonus token.\n", "    # changed\n")),
                          ("partially patched", lambda s: s.replace(patch.A4_OLD, patch.A4_NEW))):
    d = TMP / f"drift_{bad_label.replace(' ', '_')}"
    (d / REL).parent.mkdir(parents=True)
    (d / REL).write_text(mutate(stock_src))
    patch.TARGET = d / REL
    try:
        patch.main(["x"])
        check(False, f"{bad_label}: must fail closed")
    except SystemExit as e:
        check("preflight failed" in str(e) and (d / REL).read_text() == mutate(stock_src), f"{bad_label}: fails closed, file untouched ({e})")
patch.TARGET = site / REL
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "maybe"
try:
    patch.main(["x"])
    check(False, "invalid env value must fail")
except SystemExit as e:
    check("0 or 1" in str(e), f"invalid env value fails closed ({e})")
os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = "1"


def load_patched(name: str, env: str):
    os.environ["GLM53_SPEC_RESAMPLE_INDEPENDENT"] = env
    s = importlib.util.spec_from_file_location(name, site / REL)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


PATCHED = load_patched("glm53_rsu_on", "1")
PATCHED_OFF = load_patched("glm53_rsu_off", "0")
check(PATCHED._GLM53_RESAMPLE_INDEPENDENT is True and PATCHED_OFF._GLM53_RESAMPLE_INDEPENDENT is False, "runtime switch read at import")
os.environ.pop("GLM53_SPEC_RESAMPLE_INDEPENDENT")
try:
    s_ = importlib.util.spec_from_file_location("glm53_rsu_unset", site / REL)
    s_.loader.exec_module(importlib.util.module_from_spec(s_))
    check(False, "import with the variable unset must fail")
except ValueError as e:
    check("must be set to 0 or 1" in str(e), f"import of the patched module with the variable unset fails closed ({e})")


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


WARN = {}
for _name in ("glm53_rsu_on", "glm53_rsu_off"):
    WARN[_name] = _Records()
    logging.getLogger(_name).addHandler(WARN[_name])


# ------------------------------------------------------------------ one speculative step on the real kernels
class Step:
    """Fixed synthetic model: target rows [n+1, V] (log p), candidate ids [S, 16], pairwise selector scores
    [S, 16(prev), 16]. Every request gets the same model; seeds and positions differ per request."""

    def __init__(self, target_rows: torch.Tensor, cand: torch.Tensor, scores: torch.Tensor, n_verify: int, R: int):
        self.S = cand.shape[0]
        self.n = n_verify
        self.R = R
        rows = n_verify + 1
        self.rows = rows
        self.target = target_rows[:rows].to(dev).float().repeat(R, 1).contiguous()          # [R*rows, V]
        self.cand = cand.to(dev).repeat(R, 1).contiguous()                                    # [R*S, 16]
        self.scores = scores.to(dev).float().repeat(R, 1, 1).contiguous()                    # [R*S, 16, 16]
        self.draft_logits = torch.full((R, self.S, V), float("-inf"), device=dev)
        self.cached_ids = torch.zeros(R, self.S, TOPK, dtype=torch.int64, device=dev)
        self.realized = torch.empty(R * self.S * TOPK, dtype=torch.float32, device=dev)
        self.draft_tokens = torch.zeros(R, self.S, dtype=torch.int64, device=dev)
        self.map_s = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(self.S)
        self.idx_mapping = torch.arange(R, dtype=torch.int32, device=dev)
        self.exp_map = torch.arange(R, dtype=torch.int32, device=dev).repeat_interleave(rows)
        self.exp_local = torch.arange(rows, dtype=torch.int32, device=dev).repeat(R)
        self.cu = (torch.arange(R + 1, dtype=torch.int32, device=dev) * rows).contiguous()
        self.kstep = torch.arange(self.S, dtype=torch.int64, device=dev).repeat(R)
        self.krow = torch.arange(rows, dtype=torch.int64, device=dev).repeat(R)

    def run(self, mod, seeds, P, temp, draft_seeds=None, want_ref=False, greedy_draft=False, use_block=False):
        R, S, rows = self.R, self.S, self.rows
        sample_pos = (P.repeat_interleave(S) + 1 + self.kstep).contiguous()   # predicted token position Q
        _selector_walk_kernel[(R,)](
            self.scores, self.cand, sample_pos, self.map_s, temp,
            seeds if draft_seeds is None else draft_seeds, self.draft_tokens, self.realized,
            num_steps=S, top_k=TOPK, BLOCK_K=TOPK, SAMPLE_PROBABILISTIC=not greedy_draft, USE_FP64=False, num_warps=1)
        if not greedy_draft:
            _cache_draft_logits_kernel[(R * S,)](
                self.draft_logits, self.cached_ids, self.cand, self.realized, self.map_s,
                self.draft_logits.stride(0), self.draft_logits.stride(1),
                num_steps=S, top_k=TOPK, BLOCK_K=TOPK, num_warps=1)
        ds = torch.zeros(R, rows, dtype=torch.int32, device=dev)
        ds[:, 1:] = self.draft_tokens[:, : rows - 1].to(torch.int32)
        pos = (P.repeat_interleave(rows) + self.krow).contiguous()
        sampled, num_sampled = mod.rejection_sample(
            self.target, None if greedy_draft else self.draft_logits, ds.view(-1), self.cu, pos, self.idx_mapping, self.exp_map,
            self.exp_local, temp, seeds, S, None, use_fp64=False, use_block_verification=use_block)
        ref = None
        if want_ref:  # the non-speculative sampler's token for row 0 at the same (seed, pos)
            ref = gumbel_sample(self.target[0::rows], self.idx_mapping, temp, seeds, P.contiguous(), apply_temperature=True)
        return sampled, num_sampled, self.draft_tokens.clone(), ref


def chi2_sf(x: float, k: int) -> float:
    """Exact chi-square survival function Q(k/2, x/2) (scipy is not in the image). nan if k < 1."""
    if k < 1:
        return float("nan")
    return float(torch.special.gammaincc(torch.tensor(k / 2, dtype=torch.float64), torch.tensor(x / 2, dtype=torch.float64)))


def compare(counts: np.ndarray, probs: np.ndarray):
    """chi2 (bins with expected < 5 merged), p-value, TV, max |z|."""
    n = counts.sum()
    exp = probs * n
    small = exp < 5
    if small.any():
        counts = np.append(counts[~small], counts[small].sum())
        exp = np.append(exp[~small], exp[small].sum())
    keep = exp > 0
    x2 = float((((counts - exp) ** 2)[keep] / exp[keep]).sum())
    k = int(keep.sum()) - 1
    emp = counts / n
    pr = exp / n
    se = np.sqrt(np.maximum(pr * (1 - pr), 1e-300) / n)
    return x2, k, chi2_sf(x2, k), 0.5 * np.abs(emp - pr).sum(), float(np.max(np.abs(emp - pr) / se))


def fmt(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"


g = torch.Generator().manual_seed(20260927)


def rand_seeds(R):
    return torch.randint(-(2**62), 2**62, (R,), generator=g, dtype=torch.int64).to(dev)


def rand_pos(R):
    return torch.randint(1000, 200000, (R,), generator=g, dtype=torch.int64).to(dev)


# ================================================================== E1: the textbook case on real kernels
print("\n== E1: p=(.1,.5,.4) q=(.5,.4,.1), one draft token, V=154880, temperature 1.0")
toks = torch.tensor([17, 70001, 154000])
btoks = torch.tensor([5, 80000, 150000])
p = np.array([0.1, 0.5, 0.4]); q = np.array([0.5, 0.4, 0.1]); pb = np.array([0.2, 0.3, 0.5])
tr = torch.full((2, V), float("-inf"))
tr[0, toks] = torch.tensor(np.log(p), dtype=torch.float32)
tr[1, btoks] = torch.tensor(np.log(pb), dtype=torch.float32)
cand = torch.tensor([[17, 70001, 154000] + list(range(100, 113))])
sc = torch.full((1, TOPK, TOPK), float("-inf"))
sc[0, :, :3] = torch.tensor(np.log(q), dtype=torch.float32)
R1, L1 = 2048, int(os.environ.get("E1_LAUNCHES", "500"))
e1 = Step(tr, cand, sc, 1, R1)
tid = {int(t): i for i, t in enumerate(toks)}
bid = {int(t): i for i, t in enumerate(btoks)}
lut = torch.full((V,), -1, dtype=torch.int64, device=dev); lut[toks.to(dev)] = torch.arange(3, device=dev)
blut = torch.full((V,), -1, dtype=torch.int64, device=dev); blut[btoks.to(dev)] = torch.arange(3, device=dev)
temp1 = torch.ones(R1, device=dev)
acc = {k: dict(out=torch.zeros(3, dtype=torch.int64, device=dev), bonus=torch.zeros(3, dtype=torch.int64, device=dev),
               n=0, accepted=0, agree=0, agree_acc=0) for k in ("stock", "indep", "patched")}
mismatch_acc = mismatch_bonus = mismatch_off = n_diff_out = 0
t0 = time.time()
for it in range(L1):
    seeds, P = rand_seeds(R1), rand_pos(R1)
    dseeds = rand_seeds(R1)
    res = {}
    for k, mod, dsd in (("stock", STOCK, None), ("indep", STOCK, dseeds), ("patched", PATCHED, None), ("off", PATCHED_OFF, None)):
        sampled, ns, drafts, ref = e1.run(mod, seeds, P, temp1, draft_seeds=dsd, want_ref=(k != "off"))
        res[k] = (sampled.clone(), ns.clone(), ref)
        if k == "off":
            break
        a = acc[k]
        o = lut[sampled[:, 0]]
        assert (o >= 0).all(), "output outside the support of p"
        a["out"] += torch.bincount(o, minlength=3)
        accepted = ns == 2
        a["accepted"] += int(accepted.sum()); a["n"] += R1
        bo = blut[sampled[accepted, 1]]
        assert (bo >= 0).all()
        a["bonus"] += torch.bincount(bo, minlength=3)
        a["agree"] += int((sampled[:, 0] == ref).sum())
        a["agree_acc"] += int((sampled[accepted, 0] == ref[accepted]).sum())
    s_st, n_st, _ = res["stock"]; s_pa, n_pa, _ = res["patched"]; s_of, n_of, _ = res["off"]
    mismatch_acc += int((n_st != n_pa).sum())
    both = n_st == 2
    mismatch_bonus += int((s_st[both, 1] != s_pa[both, 1]).sum())
    mismatch_off += int((n_st != n_of).sum()) + int((s_st[:, 0] != s_of[:, 0]).sum()) + int((s_st[both, 1] != s_of[both, 1]).sum())
    n_diff_out += int((s_st[:, 0] != s_pa[:, 0]).sum())
torch.cuda.synchronize()
print(f"  {L1} launches x {R1} requests x 4 variants in {time.time() - t0:.1f} s")
N1 = acc["stock"]["n"]
for k, label in (("stock", "STOCK (production: draft + resample share noise)"),
                 ("indep", "STOCK, draft walk keyed by an independent seed"),
                 ("patched", "PATCHED (GLM53_SPEC_RESAMPLE_INDEPENDENT=1)")):
    a = acc[k]
    c = a["out"].cpu().numpy(); emp = c / c.sum()
    x2, dof, pv, tv, zmax = compare(c, p)
    ci = 1.96 * np.sqrt(emp * (1 - emp) / c.sum())
    ar = a["accepted"] / a["n"]; se_ar = math.sqrt(0.6 * 0.4 / a["n"])
    bc = a["bonus"].cpu().numpy(); bx2, bdof, bpv, btv, _ = compare(bc, pb)
    print(f"  {label}\n    output  {fmt(emp)} +/- {fmt(ci)} (95% CI, N={c.sum()})  vs p {fmt(p)}  chi2={x2:.1f} dof={dof} p={pv:.3g} TV={tv:.4f} max|z|={zmax:.1f}"
          f"\n    accept  {ar:.5f} (theory sum min(p,q) = 0.6, z={(ar - 0.6) / se_ar:+.2f})"
          f"\n    bonus   {fmt(bc / bc.sum())} vs {fmt(pb)} (N={bc.sum()}) chi2={bx2:.1f} p={bpv:.3g}"
          f"\n    agreement with the non-speculative sample at the same (seed,pos): {a['agree'] / a['n']:.4f}"
          f" (accepted requests {a['agree_acc'] / a['accepted']:.4f}, rejected {(a['agree'] - a['agree_acc']) / (a['n'] - a['accepted']):.4f})")
    acc[k]["pv"], acc[k]["bpv"], acc[k]["ar"] = pv, bpv, ar
    acc[k]["agree_acc_rate"] = a["agree_acc"] / a["accepted"]
check(acc["stock"]["pv"] < 1e-12, f"STOCK output differs from p (chi2 p={acc['stock']['pv']:.3g}) -- the bias, on the real kernels")
check(acc["indep"]["pv"] > 1e-4, f"independent draft noise: STOCK verifier output == p (chi2 p={acc['indep']['pv']:.3g})")
check(acc["patched"]["pv"] > 1e-4, f"PATCHED output == p (chi2 p={acc['patched']['pv']:.3g})")
for k in ("stock", "patched", "indep"):
    check(abs(acc[k]["ar"] - 0.6) < 4 * math.sqrt(0.24 / N1), f"{k}: acceptance rate {acc[k]['ar']:.5f} == 0.6 within 4 SE")
    check(acc[k]["bpv"] > 1e-4, f"{k}: bonus distribution == p_bonus (chi2 p={acc[k]['bpv']:.3g})")
check(mismatch_acc == 0, f"acceptance decisions bit-identical STOCK vs PATCHED ({mismatch_acc} mismatches / {N1})")
check(mismatch_bonus == 0, f"bonus tokens bit-identical STOCK vs PATCHED ({mismatch_bonus} mismatches)")
check(mismatch_off == 0, f"PATCHED_OFF (=0) bit-identical to STOCK ({mismatch_off} mismatches)")
check(acc["patched"]["agree_acc_rate"] < 0.99, f"an accepted draft is often NOT the seeded non-speculative sample (agreement among accepted "
      f"{acc['patched']['agree_acc_rate']:.4f}): the alignment shares the noise vector, it does not make the tokens equal")
print(f"  output token changed by the patch in {n_diff_out / N1:.4f} of requests (only rejected ones can change)")
del e1
torch.cuda.empty_cache()

# ================================================================== E2: production-shaped step
print("\n== E2: production shape: 7 drafts, top-16 pairwise selector, LLM-like target over V=154880, temperature 1.0")
S2 = 7
gen = torch.Generator().manual_seed(7)
cand2 = torch.stack([torch.randperm(V, generator=gen)[:TOPK] for _ in range(S2 + 1)])   # row S2 = bonus head
tr2 = -11.0 + 0.5 * torch.randn(S2 + 1, V, generator=gen)                                   # tail ~2-3 % of mass
head = torch.sort(1.5 * torch.randn(S2 + 1, TOPK, generator=gen), dim=1, descending=True).values
for k in range(S2 + 1):
    tr2[k, cand2[k]] = head[k]
sc2 = head[:S2, None, :] + 0.6 * torch.randn(S2, 1, TOPK, generator=gen) + 0.3 * torch.randn(S2, TOPK, TOPK, generator=gen)
p2 = torch.softmax(tr2.double(), dim=-1)
R2, L2 = 256, int(os.environ.get("E2_LAUNCHES", "600"))
e2_summary = {}
for n_verify in (7, 4):
    e2 = Step(tr2, cand2[:S2], sc2, n_verify, R2)
    rows = n_verify + 1
    # bin map per row: 16 head tokens of that row's distribution + tail
    heads = [cand2[k] for k in range(rows)]   # target row k's head tokens (row n_verify = the bonus row)
    luts = []
    probs = []
    for k in range(rows):
        lt = torch.full((V,), TOPK, dtype=torch.int64, device=dev)
        lt[heads[k].to(dev)] = torch.arange(TOPK, device=dev)
        luts.append(lt)
        pk = p2[k]
        probs.append(np.append(pk[heads[k]].numpy(), 1 - float(pk[heads[k]].sum())))
    counts = {v: [torch.zeros(TOPK + 1, dtype=torch.int64, device=dev) for _ in range(rows)] for v in ("stock", "indep", "patched")}
    len_hist = {v: torch.zeros(rows + 1, dtype=torch.int64, device=dev) for v in ("stock", "patched")}
    mism = mism_off = 0
    temp2 = torch.ones(R2, device=dev)
    t0 = time.time()
    for it in range(L2):
        seeds, P, dseeds = rand_seeds(R2), rand_pos(R2), rand_seeds(R2)
        out = {}
        for v, mod, dsd in (("stock", STOCK, None), ("indep", STOCK, dseeds), ("patched", PATCHED, None), ("off", PATCHED_OFF, None)):
            sampled, ns, _, _ = e2.run(mod, seeds, P, temp2, draft_seeds=dsd)
            out[v] = (sampled.clone(), ns.clone())
            if v == "off":
                continue
            for k in range(rows):
                reached = ns > k
                counts[v][k] += torch.bincount(luts[k][sampled[reached, k]], minlength=TOPK + 1)
            if v in len_hist:
                len_hist[v] += torch.bincount(ns.long(), minlength=rows + 1)
        mism += int((out["stock"][1] != out["patched"][1]).sum())
        ns_s = out["stock"][1]
        valid = torch.arange(out["stock"][0].shape[1], device=dev)[None, :] < ns_s[:, None]   # sampled is [R, S+1]
        mism_off += int((out["stock"][1] != out["off"][1]).sum()) + int((out["stock"][0][valid] != out["off"][0][valid]).sum())
    torch.cuda.synchronize()
    print(f"  n_verify={n_verify} ({rows} rows, {S2} drafted): {L2} launches x {R2} requests x 4 variants in {time.time() - t0:.1f} s")
    hs = len_hist["stock"].cpu().numpy()
    print(f"    emitted-length histogram (STOCK) 1..{rows}: {hs[1:].tolist()}  mean accepted drafts {((np.arange(rows + 1) - 1) * hs)[1:].sum() / hs.sum():.3f}")
    worst = {}
    for v in ("stock", "indep", "patched"):
        line = []
        wp = 1.0
        for k in range(rows):
            c = counts[v][k].cpu().numpy()
            x2, dof, pv, tv, zmax = compare(c, probs[k])
            line.append(f"r{k}{'(bonus)' if k == n_verify else ''}: N={c.sum()} TV={tv:.4f} chi2={x2:.0f}/{dof} p={pv:.2g}")
            wp = min(wp, pv)
            if k == n_verify:
                worst[(v, "bonus")] = pv
        worst[v] = wp
        print(f"    {v:8s} " + " | ".join(line))
    e2_summary[n_verify] = worst
    check(worst["stock"] < 1e-12, f"n={n_verify}: STOCK biased on at least one row (min chi2 p={worst['stock']:.3g})")
    # 3 * rows tests per shape: Bonferroni-style floor
    check(worst["patched"] > 1e-4, f"n={n_verify}: PATCHED unbiased on every row incl. bonus (min chi2 p={worst['patched']:.3g})")
    check(worst["indep"] > 1e-4, f"n={n_verify}: independent draft noise unbiased on every row (min chi2 p={worst['indep']:.3g})")
    check(worst[("stock", "bonus")] > 1e-4, f"n={n_verify}: STOCK bonus row unbiased (p={worst[('stock', 'bonus')]:.3g}) -- the bonus path needs no change")
    check(mism == 0, f"n={n_verify}: accepted lengths bit-identical STOCK vs PATCHED ({mism} mismatches / {L2 * R2})")
    check(mism_off == 0, f"n={n_verify}: PATCHED_OFF bit-identical to STOCK ({mism_off} mismatches)")
    del e2
    torch.cuda.empty_cache()

# ================================================================== E3: alternative A (draft_sample_method="greedy"), measured only
print("\n== E3: alternative A = greedy (argmax) drafts, one-hot q, STOCK verifier; same E2 model (not a production change)")
for n_verify in (7, 4):
    e3 = Step(tr2, cand2[:S2], sc2, n_verify, R2)
    rows = n_verify + 1
    luts = []
    probs = []
    for k in range(rows):
        lt = torch.full((V,), TOPK, dtype=torch.int64, device=dev)
        lt[cand2[k].to(dev)] = torch.arange(TOPK, device=dev)
        luts.append(lt)
        probs.append(np.append(p2[k][cand2[k]].numpy(), 1 - float(p2[k][cand2[k]].sum())))
    cnt = {v: [torch.zeros(TOPK + 1, dtype=torch.int64, device=dev) for _ in range(rows)] for v in ("greedy", "prob")}
    acc_len = {v: 0 for v in cnt}
    temp2 = torch.ones(R2, device=dev)
    L3 = L2 // 2
    for it in range(L3):
        seeds, P = rand_seeds(R2), rand_pos(R2)
        for v, gd, mod in (("greedy", True, STOCK), ("prob", False, PATCHED)):
            sampled, ns, _, _ = e3.run(mod, seeds, P, temp2, greedy_draft=gd)
            acc_len[v] += int((ns - 1).sum())
            for k in range(rows):
                reached = ns > k
                cnt[v][k] += torch.bincount(luts[k][sampled[reached, k]], minlength=TOPK + 1)
    wp = 1.0
    for v in ("greedy", "prob"):
        line = []
        for k in range(rows):
            c = cnt[v][k].cpu().numpy()
            x2, dof, pv, tv, _ = compare(c, probs[k])
            line.append(f"r{k}: N={c.sum()} p={pv:.2g}")
            if v == "greedy" and pv == pv:   # rows with too few samples for one dof are skipped (nan)
                wp = min(wp, pv)
        print(f"  n={n_verify} {v:6s} mean accepted drafts/step {acc_len[v] / (L3 * R2):.3f} | " + " | ".join(line))
    check(wp > 1e-4, f"n={n_verify}: greedy drafts are exact on the stock verifier (min chi2 p={wp:.3g})")
    del e3
    torch.cuda.empty_cache()

# ================================================================== E4: two consecutive steps, standard vs block verification
warn_before_e4 = len(WARN["glm53_rsu_on"].msgs) + len(WARN["glm53_rsu_off"].msgs)   # E1-E3 ran standard mode only
print("\n== E4: two consecutive steps (3 drafts; step 2 starts at P + num_sampled with the same seed), context-independent"
      " target p=(.1,.5,.4) on every row, q=(.5,.4,.1) for every draft")
N4 = 3
tr4 = torch.full((N4 + 1, V), float("-inf"))
tr4[:, toks] = torch.tensor(np.log(p), dtype=torch.float32)
cand4 = torch.tensor([[17, 70001, 154000] + list(range(100, 113))] * N4)
sc4 = torch.full((N4, TOPK, TOPK), float("-inf"))
sc4[:, :, :3] = torch.tensor(np.log(q), dtype=torch.float32)
R4, L4 = 512, int(os.environ.get("E4_LAUNCHES", "300"))
e4 = Step(tr4, cand4, sc4, N4, R4)
temp4 = torch.ones(R4, device=dev)
ones4 = torch.ones(R4, dtype=torch.int64, device=dev)
V4 = (("std-patched", PATCHED, False, False), ("blk-stock", STOCK, True, False), ("blk-patched", PATCHED, True, False),
      ("blk-patched-freshseed2", PATCHED, True, True))
st4 = {v[0]: dict(first2=torch.zeros(N4 + 2, 3, dtype=torch.int64, device=dev), rows1=torch.zeros(N4 + 1, 3, dtype=torch.int64, device=dev),
                  len12=torch.zeros(N4 + 2, N4 + 2, dtype=torch.int64, device=dev)) for v in V4}
t0 = time.time()
for it in range(L4):
    seeds, P, seeds2 = rand_seeds(R4), rand_pos(R4), rand_seeds(R4)
    for name, mod, blk, fresh in V4:
        s1, n1, _, _ = e4.run(mod, seeds, P, temp4, use_block=blk)
        s1, n1 = s1.clone(), n1.clone().long()
        s2, n2, _, _ = e4.run(mod, seeds2 if fresh else seeds, P + n1, temp4, use_block=blk)
        a = st4[name]
        f2 = lut[s2[:, 0]]
        assert (f2 >= 0).all(), "output outside the support of p"
        a["first2"].index_put_((n1, f2), ones4, accumulate=True)
        a["len12"].index_put_((n1, n2.long()), ones4, accumulate=True)
        for k in range(N4 + 1):
            reached = n1 > k
            a["rows1"][k] += torch.bincount(lut[s1[reached, k]], minlength=3)
torch.cuda.synchronize()
print(f"  {L4} launches x {R4} requests x 2 steps x {len(V4)} variants in {time.time() - t0:.1f} s")
e4_res = {}
for name, *_ in V4:
    a = st4[name]
    f2c, r1c, lc = a["first2"].cpu().numpy(), a["rows1"].cpu().numpy(), a["len12"].cpu().numpy()[1:, 1:]
    r1 = [compare(r1c[k], p) for k in range(N4 + 1)]
    f2 = [compare(f2c[j], p) for j in range(1, N4 + 2)]
    f2all = compare(f2c[1:].sum(0), p)
    rs, cs = lc.sum(1, keepdims=True), lc.sum(0, keepdims=True)
    ex = rs * cs / lc.sum()
    x2i = float(((lc - ex) ** 2 / ex).sum())
    dofi = (lc.shape[0] - 1) * (lc.shape[1] - 1)
    pind = chi2_sf(x2i, dofi)
    tot = f2c[1:].sum(0)
    print(f"  {name}\n    step 1, per row: " + " | ".join(f"r{k} N={r1c[k].sum()} TV={r1[k][3]:.4f} p={r1[k][2]:.2g}" for k in range(N4 + 1))
          + "\n    step 2 first token given step-1 num_sampled: "
          + " | ".join(f"{j}: N={f2c[j].sum()} {fmt(f2c[j] / max(1, f2c[j].sum()))} p={f2[j - 1][2]:.2g}" for j in range(1, N4 + 2))
          + f"\n    step 2 first token, all: {fmt(tot / tot.sum())} vs p {fmt(p)} TV={f2all[3]:.4f} p={f2all[2]:.2g}"
          + f"\n    step-2 num_sampled independent of step-1 num_sampled: chi2={x2i:.1f} dof={dofi} p={pind:.2g}; "
          + "rows " + str((lc / rs).round(3).tolist()))
    e4_res[name] = (min(x[2] for x in r1), min([x[2] for x in f2] + [f2all[2], pind]))
check(e4_res["std-patched"][0] > 1e-4 and e4_res["std-patched"][1] > 1e-4,
      f"standard + PATCHED: exact within a step and across steps (min chi2 p {e4_res['std-patched'][0]:.3g} / {e4_res['std-patched'][1]:.3g})")
check(e4_res["blk-stock"][0] < 1e-12, f"block + STOCK: biased within a step (min chi2 p {e4_res['blk-stock'][0]:.3g})")
check(e4_res["blk-patched"][0] > 1e-4, f"block + PATCHED: exact within one step (min chi2 p {e4_res['blk-patched'][0]:.3g})")
check(e4_res["blk-patched"][1] < 1e-12, f"block + PATCHED: NOT exact across steps (min chi2 p {e4_res['blk-patched'][1]:.3g}) -- "
      "the next step reuses (seed, pos) keys that block verification consumed; P0 does not make block mode exact")
check(e4_res["blk-patched-freshseed2"][1] > 1e-4, f"block + PATCHED with a fresh seed for step 2: exact across steps "
      f"(min chi2 p {e4_res['blk-patched-freshseed2'][1]:.3g}) -- the cross-step bias is the key reuse")
del e4
torch.cuda.empty_cache()

# ================================================================== G: greedy path bit-identical
print("\n== G: mixed batch (half temperature 0, half 1.0), random full-vocab logits, 7 drafts verified")
RG = 512
gg = torch.Generator().manual_seed(11)
trg = 3.0 * torch.randn(S2 + 1, V, generator=gg)
candg = torch.stack([torch.topk(trg[k], TOPK).indices for k in range(S2)])
scg = trg[torch.arange(S2)[:, None], candg][:, None, :] + 0.5 * torch.randn(S2, TOPK, TOPK, generator=gg)
eg = Step(trg, candg, scg, 7, RG)
tempg = torch.where(torch.arange(RG, device=dev) % 2 == 0, 0.0, 1.0).float()
greedy = tempg == 0
diff_greedy = diff_ns = diff_off = changed_sampled = 0
for it in range(50):
    seeds, P = rand_seeds(RG), rand_pos(RG)
    ss, ns_s, _, _ = eg.run(STOCK, seeds, P, tempg)
    ss, ns_s = ss.clone(), ns_s.clone()
    sp, ns_p, _, _ = eg.run(PATCHED, seeds, P, tempg)
    sp, ns_p = sp.clone(), ns_p.clone()
    so, ns_o, _, _ = eg.run(PATCHED_OFF, seeds, P, tempg)
    valid = torch.arange(8, device=dev)[None, :] < ns_s[:, None]
    diff_ns += int((ns_s != ns_p).sum())
    diff_greedy += int(((ss != sp) & valid)[greedy].sum())
    diff_off += int((ns_s != ns_o).sum()) + int(((ss != so) & valid).sum())
    changed_sampled += int(((ss != sp) & valid)[~greedy].any(dim=1).sum())
check(diff_greedy == 0, f"greedy requests: every emitted token bit-identical STOCK vs PATCHED ({diff_greedy} diffs over {50 * RG // 2} requests)")
check(diff_ns == 0, f"accepted lengths bit-identical for greedy and sampled requests ({diff_ns} diffs)")
check(diff_off == 0, f"PATCHED_OFF bit-identical to STOCK in the mixed batch ({diff_off} diffs)")
del eg
torch.cuda.empty_cache()
check(changed_sampled > 0, f"the patch does act on sampled requests ({changed_sampled} of {50 * RG // 2} sampled requests changed a rejected-position token)")
nw_on, nw_off = len(WARN["glm53_rsu_on"].msgs), len(WARN["glm53_rsu_off"].msgs)
for m_ in WARN["glm53_rsu_on"].msgs:
    print("  logged: " + m_)
check(warn_before_e4 == 0 and nw_on == 1 and "not exact" in WARN["glm53_rsu_on"].msgs[0] and nw_off == 0,
      f"block-mode warning: none during the standard-mode runs E1-E3 ({warn_before_e4}); exactly once for the many block-mode "
      f"calls of E4 ({nw_on}); none from the module that never ran block mode ({nw_off})")

# ================================================================== cost of the change (rejection_sample only)
print("\n== rejection_sample time, 8 rows/request (7 verified + bonus), temperature 1.0 (us, median of 7 x 50 calls)")
for B in (1, 8):
    st = Step(tr2, cand2[:S2], sc2, 7, B)
    seeds, P = rand_seeds(B), rand_pos(B)
    st.run(STOCK, seeds, P, torch.ones(B, device=dev))
    ds = torch.zeros(B, 8, dtype=torch.int32, device=dev); ds[:, 1:] = st.draft_tokens.to(torch.int32)
    pos = (P.repeat_interleave(8) + st.krow).contiguous()
    tt = torch.ones(B, device=dev)
    res = {}
    for name, mod in (("stock", STOCK), ("patched", PATCHED)):
        f = lambda: mod.rejection_sample(st.target, st.draft_logits, ds.view(-1), st.cu, pos, st.idx_mapping, st.exp_map, st.exp_local, tt, seeds, 7)
        f(); torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            e0, e1_ = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(50):
                f()
            e1_.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1_) / 50 * 1000)
        res[name] = sorted(ts)[3]
    print(f"  B={B}: stock {res['stock']:.1f} us, patched {res['patched']:.1f} us")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'ALL PASSED' if not FAIL else 'FAILED: ' + str(len(FAIL))}")
for f in FAIL:
    print("  -", f)
sys.exit(1 if FAIL else 0)
