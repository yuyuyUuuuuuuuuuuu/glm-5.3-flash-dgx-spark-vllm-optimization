"""GLM53_DEC_VTRIM_STATS: log-only data collector for per-step verify trimming (docs/OPT_DECODE.md section 4).

Production verifies 4.84 drafts per step for 2.61 accepted (24 h, 2026-10-03), and every verified row costs routed
expert bytes. Whether a per-step rule that verifies fewer rows when the drafter is unsure pays off depends on how well
DFlash2's own distributions predict production's acceptance at temperature 1.0 -- which only production traffic can
tell. This module records it without changing anything:

  after every rejection-sampling call (RejectionSampler.__call__, wrapped; the sampler's outputs are returned
  untouched), for every request with drafts and temperature > 0, one record of 16 floats goes to a GPU ring buffer:
      [n drafts, a accepted drafts, qmax_1..qmax_7, qd_1..qd_7]
  qmax_i = max_x q_i(x) and qd_i = q_i(d_i), q_i = the drafter distribution the verifier used for draft i
  (softmax of DFlash2's cached draft logits / temperature), d_i the draft token. Probabilities and counts only: no
  token id, no text. A trimming rule may use qmax_i and qd_<i for row i (not qd_i: that biases the proposal), so
  these two are exactly what an offline policy evaluation needs (tests/optdec/vtrim_policy_from_stats.py).
  Every GLM53_DEC_VTRIM_STATS verify calls (e.g. 2000) the ring (GLM53_DEC_VTRIM_STATS_CAP records, default 200000)
  is written with numpy to GLM53_DEC_VTRIM_STATS_FILE (default /root/.cache/vllm/glm53_vtrim_stats.npy: the
  launcher's bind-mounted cache dir), and one INFO line gives the running acceptance by position and qmax half.
Cost while on (nodeC, V = 154880, tests/optdec/bench_vtrim_stats.py): 0.35 / 0.35 / 0.46 ms per verify step at 1 / 4 / 8
requests (one streaming Triton pass over each verified row's draft logits + small ops, event-timed incl. launches), and
one D2H copy per write. Unset / 0 = nothing is wrapped.
"""
from __future__ import annotations

import logging
import os

import torch

_log = logging.getLogger("vllm.glm53_vtrim_stats")
ENV = "GLM53_DEC_VTRIM_STATS"
NPOS = 7
REC = 2 + 2 * NPOS


class _S:
    def __init__(self) -> None:
        self.every = 0
        self.cap = 200000
        self.path = "/root/.cache/vllm/glm53_vtrim_stats.npy"
        self.ring = None
        self.n = 0              # records written (host count; the ring index is n % cap)
        self.calls = 0
        self.installed = False
        self.disabled = None


ST = _S()


def env_every(environ=None) -> int:
    v = (os.environ if environ is None else environ).get(ENV)
    try:
        return max(0, int(v)) if v is not None and v.strip() else 0
    except ValueError:
        return 0


_KS: dict = {}


def _row_stats(draft_logits, rs, step, tok, temp):
    """Per logits row: log q(tok) and log max q, q = softmax(draft_logits[rs, step] / temp), one streaming pass over
    the vocab per row (Triton; online max / sum-exp). Falls back to torch when Triton is unavailable."""
    NL = rs.numel()
    try:
        if "k" not in _KS:
            from vllm.triton_utils import tl, triton

            @triton.jit
            def _k(dl, rs, step, tok, temp, out_d, out_m, s0, s1, V, BLOCK: tl.constexpr):
                j = tl.program_id(0)
                base = dl + tl.load(rs + j).to(tl.int64) * s0 + tl.load(step + j).to(tl.int64) * s1
                t = tl.load(temp + j)
                m = tl.full((), float("-inf"), tl.float32)
                acc = tl.zeros((), tl.float32)
                for v0 in range(0, V, BLOCK):
                    o = v0 + tl.arange(0, BLOCK)
                    x = tl.load(base + o, mask=o < V, other=float("-inf")).to(tl.float32) / t
                    bm = tl.max(x, axis=0)
                    nm = tl.maximum(m, bm)
                    acc = acc * tl.exp(m - nm) + tl.sum(tl.exp(x - nm), axis=0)
                    m = nm
                lse = m + tl.log(acc)
                xd = tl.load(base + tl.load(tok + j)).to(tl.float32) / t
                tl.store(out_d + j, xd - lse)
                tl.store(out_m + j, m - lse)

            _KS["k"] = _k
        out_d = torch.empty(NL, dtype=torch.float32, device=rs.device)
        out_m = torch.empty_like(out_d)
        if NL:
            _KS["k"][(NL,)](draft_logits, rs, step, tok, temp.to(torch.float32).contiguous(), out_d, out_m,
                            draft_logits.stride(0), draft_logits.stride(1), draft_logits.shape[2], BLOCK=4096,
                            num_warps=8)
        return out_d, out_m
    except Exception:  # noqa: BLE001 - torch reference path (same values up to fp32 rounding)
        dl = draft_logits[rs, step].to(torch.float32) / temp[:, None]
        lse = torch.logsumexp(dl, dim=-1)
        return dl.gather(1, tok[:, None]).squeeze(1) - lse, dl.max(dim=-1).values - lse


def records(draft_logits, draft_sampled, cu_num_logits, req_state_rows, local_pos, temperature, num_sampled,
            num_reqs: int) -> torch.Tensor:
    """[num_reqs, 16] float32 on the GPU (no host sync). Rows of requests without drafts or with temperature 0 have
    n = 0. draft_logits: [max_reqs, steps, V] pre-temperature (DFlash2's cache); draft_sampled: the verify batch's
    input ids at the logits rows (row j's draft is draft_sampled[j + 1]); cu_num_logits [R + 1]; req_state_rows /
    local_pos per logits row; temperature per request state; num_sampled per request (accepted + 1)."""
    dev = draft_sampled.device
    NL = draft_sampled.numel()
    R = int(num_reqs)
    out = torch.zeros(R, REC, dtype=torch.float32, device=dev)
    if NL == 0 or R == 0:
        return out
    cu = cu_num_logits[:R + 1].to(torch.int64)
    cnt = cu[1:] - cu[:-1]
    req = torch.repeat_interleave(torch.arange(R, device=dev), cnt, output_size=NL)
    n_r = (cnt - 1).clamp(min=0)
    p = local_pos[:NL].to(torch.int64)
    rs = req_state_rows[:NL].to(torch.int64)
    t = temperature[rs].to(torch.float32)
    j = torch.arange(NL, device=dev)
    nxt = draft_sampled[(j + 1).clamp(max=NL - 1)].to(torch.int64)
    steps = draft_logits.shape[1]
    valid = (p < n_r[req]) & (t > 0) & (nxt >= 0) & (p < min(steps, NPOS))
    tt = torch.where(t > 0, t, torch.ones_like(t))
    lqd, lqm = _row_stats(draft_logits, rs, p.clamp(max=steps - 1), nxt.clamp(min=0), tt)
    qd = torch.where(valid, lqd.exp(), torch.zeros_like(lqd))
    qm = torch.where(valid, lqm.exp(), torch.zeros_like(lqm))
    col = p.clamp(max=NPOS - 1)
    flat = req * REC
    out.view(-1).index_put_((torch.where(valid, flat + 2 + col, torch.zeros_like(flat)),), qm, accumulate=True)
    out.view(-1).index_put_((torch.where(valid, flat + 2 + NPOS + col, torch.zeros_like(flat)),), qd, accumulate=True)
    # the zero-index sink above only ever received zeros for invalid rows (qm = qd = 0 there); now n and a
    has = torch.zeros(R, dtype=torch.float32, device=dev).index_add_(0, req, valid.to(torch.float32))
    out[:, 0] = has
    a = (num_sampled[:R].to(torch.float32) - 1).clamp(min=0)
    out[:, 1] = torch.minimum(a, has)
    return out


def _append(rec: torch.Tensor) -> None:
    R = rec.shape[0]
    if ST.ring is None:
        ST.ring = torch.zeros(ST.cap, REC, dtype=torch.float32, device=rec.device)
    idx = (ST.n + torch.arange(R, device=rec.device)) % ST.cap
    ST.ring.index_copy_(0, idx, rec)
    ST.n += R


def _write() -> None:
    import numpy as np
    k = min(ST.n, ST.cap)
    arr = ST.ring[:k].cpu().numpy()
    arr = arr[arr[:, 0] > 0]
    tmp = ST.path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, ST.path)
    n = arr[:, 0]
    a = arr[:, 1]
    parts = []
    for i in range(NPOS):
        reach = (n > i) & (a >= i)
        if reach.sum() == 0:
            continue
        qm = arr[:, 2 + i]
        acc = a > i
        lo = reach & (qm < 0.5)
        hi = reach & (qm >= 0.5)
        parts.append(f"p{i + 1} {acc[reach].mean():.2f} (qmax<.5 {acc[lo].mean() if lo.any() else float('nan'):.2f}"
                     f" n{int(lo.sum())}, >=.5 {acc[hi].mean() if hi.any() else float('nan'):.2f} n{int(hi.sum())})")
    _log.info("glm53_vtrim_stats: %d records -> %s; P(accepted | reached) by position: %s", len(arr), ST.path,
              "; ".join(parts))


def _wrap(orig):
    def __call__(self, logits, input_batch, draft_logits=None):
        out = orig(self, logits, input_batch, draft_logits)
        if ST.disabled is not None or draft_logits is None:
            return out
        try:
            ib = input_batch
            rec = records(draft_logits, ib.input_ids[ib.logits_indices], ib.cu_num_logits, ib.expanded_idx_mapping,
                          ib.expanded_local_pos, self.sampler.sampling_states.temperature.gpu, out.num_sampled,
                          ib.num_reqs)
            _append(rec)
            ST.calls += 1
            if ST.every and ST.calls % ST.every == 0:
                _write()
        except Exception as exc:  # noqa: BLE001 - statistics never break sampling
            ST.disabled = repr(exc)
            _log.warning("glm53_vtrim_stats: switched off (%r); sampling unaffected", exc)
        return out

    __call__._glm53_vtrim_orig = orig
    return __call__


def install(every: int | None = None) -> bool:
    if ST.installed:
        return True
    ST.every = env_every() if every is None else int(every)
    if ST.every <= 0:
        return False
    try:
        ST.cap = max(1000, int(os.environ.get("GLM53_DEC_VTRIM_STATS_CAP", "200000")))
    except ValueError:
        ST.cap = 200000
    ST.path = os.environ.get("GLM53_DEC_VTRIM_STATS_FILE", ST.path) or ST.path
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as RS
    cls = RS.RejectionSampler
    if not hasattr(cls.__call__, "_glm53_vtrim_orig"):
        cls.__call__ = _wrap(cls.__call__)
    ST.installed = True
    _log.info("glm53_vtrim_stats on: one record per request and verify step (16 floats, no token ids); written every "
              "%d verify calls to %s (ring of %d records)", ST.every, ST.path, ST.cap)
    return True


def plugin_install() -> None:
    every = env_every()
    _log.info("glm53_vtrim_stats plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              f"recording, written every {every} verify calls" if every else "off")
    if not every:
        return
    try:
        import vllm.model_executor.model_loader.base_loader as BL
        orig = BL.process_weights_after_loading
        if getattr(orig, "_glm53_vtrim_hook", False):
            return

        def process_weights_after_loading(model, model_config, target_device):
            orig(model, model_config, target_device)
            try:
                install()
            except Exception as exc:  # noqa: BLE001
                _log.warning("glm53_vtrim_stats not installed (%r)", exc)

        process_weights_after_loading._glm53_vtrim_hook = True
        BL.process_weights_after_loading = process_weights_after_loading
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_vtrim_stats not installed: %r", exc)
