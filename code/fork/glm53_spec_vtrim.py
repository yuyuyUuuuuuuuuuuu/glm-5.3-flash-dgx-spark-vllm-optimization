"""GLM53_SPEC_VTRIM: per-step verify trimming from the DFlash2 drafter's own confidence (docs/SPEC_VTRIM.md).

Why: production verifies K in {4, 5, 7} drafts per step (adaptive-K, a CPU EMA that decides two steps late) and every
verified row costs routed-expert bytes: on nodeC at production shapes the decode MoE layer costs 452 / 585 / 693 / 784 /
850 / 906 / 956 us at 2..8 rows (tests/vtrim/bench_dead_rows.py, corr40 routing), i.e. 2.1-4.5 ms per row per step
over 42 MoE layers. Low-acceptance text (prose / Japanese) accepts 1.7-2.2 drafts of the 4.5 it verifies. The CPU
policy cannot know which step will fail; the drafter can: DFlash2's candidate selector gives, for every draft
position, a distribution over its 16 candidates. This module turns that into a per-request stop: drafts after the
first position whose survival estimate falls below a threshold are not verified.

Rule (exact for every temperature; the decision for draft i uses only d_<i and the shape of q_i, never d_i):
    q_i     = softmax(realized selector scores of step i / T)   (T = the request's temperature, 1.0 for greedy)
    S_i     = qd_1 * ... * qd_{i-1} * qmax_i                     (qd_j = q_j(d_j), qmax_i = max q_i)
    n*      = the largest i with S_1..S_i >= TAU (a prefix), at least GLM53_SPEC_VTRIM_MIN
Rows of drafts i > n* are DEAD for this verify step:
  * the rejection sampler sees draft -1 there (vLLM's placeholder: "a proposal of length n*", exact in standard AND
    block verification) - only in the rejection_sample call, never in input_ids or the penalty history;
  * the target forward still runs the graph at K+1 rows (no new CUDA graphs, no host sync), but the routed-expert ids
    of dead rows are set to -1 (the TF/production sentinel: no expert bytes read) before the decode MoE apply.
Dead rows are never accepted, so their (zero-routed) hidden states never reach an emitted token: attention and the
KDA recurrence only flow forward in the sequence and every other op is per token. At temperature 0 the emitted
tokens are the target's argmax at the live rows, exactly as before; only how many drafts a step can accept changes.
Batches with structured-output requests are never trimmed (their drafts go to the grammar on the host).

Modes (GLM53_SPEC_VTRIM): unset/off = nothing is installed (production byte for byte); shadow = compute n* and log
what trimming WOULD do (dead rows, accepted drafts that would have been cut), nothing else changes; on = trim.
Knobs: GLM53_SPEC_VTRIM_TAU (0..1, default 0.25), GLM53_SPEC_VTRIM_MIN (0..7, default 0),
GLM53_SPEC_VTRIM_LOG (stats line every N verify steps, default 2000; the first line comes after 64 steps).
Both TP ranks compute n* from the replicated drafter outputs (the same values the draft tokens come from), so the
ranks agree by construction; every stats line carries a checksum of n* (nsum) so head and worker logs can be compared.
Memory: max_num_seqs int32 + max_num_batched_tokens bytes + 8 int64 per rank. Cost when on: one tiny Triton kernel in
the drafter graph, one in the step preparation, one masked_fill per MoE layer (decode sizes only), a few small ops
around rejection sampling.
"""
from __future__ import annotations

import logging
import os

import torch

_log = logging.getLogger("vllm.glm53_spec_vtrim")
ENV = "GLM53_SPEC_VTRIM"
ENV_TAU = "GLM53_SPEC_VTRIM_TAU"
ENV_MIN = "GLM53_SPEC_VTRIM_MIN"
ENV_LOG = "GLM53_SPEC_VTRIM_LOG"
NO_TRIM = 127          # n* meaning "verify everything"
DECODE_MAX_TOKENS = 64  # the MoE mask only applies to decode-sized forwards (8 requests x 8 rows)
TAG = "[glm53-spec-vtrim]"


class _S:
    def __init__(self) -> None:
        self.mode = "off"
        self.tau = 0.25
        self.min_live = 0
        self.log_every = 2000
        self.installed = False
        self.nstar = None      # int32 [max_num_seqs], per request state
        self.live = None       # uint8 [max_num_batched_tokens], per token of the step being prepared
        self.stats = None      # int64 [8]: steps, reqsteps, drafts, dead, accepted, lost, nsum, trimmed_steps
        self.hist = None       # int64 [8*8*8]: request-steps by (K verified, min(n*, K), accepted)
        self.grid = None       # shadow: float32 [len(GRID)] thresholds, ngrid int32 [max_num_seqs, len(GRID)] n* each,
        self.ngrid = None      # hist_grid int64 [len(GRID)*512] the same histogram per threshold
        self.hist_grid = None
        self.hist_file = None
        self.steps = 0
        self.first_log = 64
        self.rank = "?"
        self.hooks = []
        self.disabled = None
        self.stats_off = None
        self.skip_structured = 0


ST = _S()


def parse_env(environ=None) -> tuple[str, float, int, int]:
    e = os.environ if environ is None else environ
    raw = (e.get(ENV) or "").strip().lower()
    mode = {"": "off", "0": "off", "off": "off", "shadow": "shadow", "1": "on", "on": "on"}.get(raw)
    if mode is None:
        raise ValueError(f"{ENV} must be unset/off/0, shadow, or on/1")
    tau = float((e.get(ENV_TAU) or "0.25").strip())
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"{ENV_TAU} must be in [0, 1]")
    mn = int((e.get(ENV_MIN) or "0").strip())
    if not (0 <= mn <= 7):
        raise ValueError(f"{ENV_MIN} must be 0..7")
    lg = int((e.get(ENV_LOG) or "2000").strip())
    if lg < 0:
        raise ValueError(f"{ENV_LOG} must be >= 0")
    return mode, tau, mn, lg


# ----------------------------------------------------------------------------------------------- kernels
_K: dict = {}


def _kernels():
    if _K:
        return _K
    from vllm.triton_utils import tl, triton

    @triton.jit
    def nstar_kernel(scores_ptr, cand_ptr, tok_ptr, sidx_ptr, temp_ptr, nstar_ptr, tau, min_live, grid_ptr, ngrid_ptr,
                     NUM_STEPS: tl.constexpr, TOP_K: tl.constexpr, BLOCK_K: tl.constexpr, G: tl.constexpr,
                     HAS_GRID: tl.constexpr):
        """One program per drafted request row: n* from the realized selector scores (see module doc)."""
        row = tl.program_id(0)
        rs = tl.load(sidx_ptr + row * NUM_STEPS)
        if rs >= 0:
            t = tl.load(temp_ptr + rs)
            t = tl.where(t > 0.0, t, 1.0)
            offs = tl.arange(0, BLOCK_K)
            m_k = offs < TOP_K
            surv = 1.0
            n = 0
            alive = 1
            goffs = tl.arange(0, G)
            if HAS_GRID:
                taus = tl.load(grid_ptr + goffs)
            else:
                taus = tl.full((G,), 2.0, tl.float32)
            ng = tl.zeros((G,), tl.int32)
            ag = tl.full((G,), 1, tl.int32)
            for step in range(NUM_STEPS):
                flat = row * NUM_STEPS + step
                s = tl.load(scores_ptr + flat * TOP_K + offs, mask=m_k, other=float("-inf")).to(tl.float32) / t
                mx = tl.max(s, axis=0)
                ex = tl.where(m_k, tl.exp(s - mx), 0.0)
                den = tl.sum(ex, axis=0)
                qmax = 1.0 / den
                tok = tl.load(tok_ptr + flat)
                cand = tl.load(cand_ptr + flat * TOP_K + offs, mask=m_k, other=-1)
                qd = tl.sum(tl.where((cand == tok) & m_k, ex, 0.0), axis=0) / den
                ok = (surv * qmax >= tau) & (alive == 1)
                n = tl.where(ok, step + 1, n)
                alive = tl.where(ok, 1, 0)
                okg = (surv * qmax >= taus) & (ag == 1)
                ng = tl.where(okg, step + 1, ng)
                ag = tl.where(okg, 1, 0)
                surv = surv * qd
            n = tl.maximum(n, min_live)
            tl.store(nstar_ptr + rs, n.to(tl.int32))
            if HAS_GRID:
                tl.store(ngrid_ptr + rs * G + goffs, tl.maximum(ng, min_live))

    @triton.jit
    def live_kernel(live_ptr, idx_ptr, qsl_ptr, cu_ptr, nstar_ptr, BLOCK: tl.constexpr):
        """One program per batch request: zero LIVE at the token rows of its dead drafts."""
        b = tl.program_id(0)
        rs = tl.load(idx_ptr + b)
        nl = tl.load(cu_ptr + b + 1) - tl.load(cu_ptr + b)
        end = tl.load(qsl_ptr + b + 1)
        n = tl.load(nstar_ptr + rs)
        j = tl.arange(0, BLOCK)
        dead = (j >= 1) & (j < nl) & (j > n) & (nl > 1)
        tl.store(live_ptr + (end - nl + j), tl.zeros((BLOCK,), tl.uint8), mask=dead)

    @triton.jit
    def stats_kernel(cu_ptr, idx_ptr, ns_ptr, nstar_ptr, ngrid_ptr, stats_ptr, hist_ptr, histg_ptr, R,
                     G: tl.constexpr, HAS_GRID: tl.constexpr, BLOCK_R: tl.constexpr):
        """One program per verify call: the per-step statistics (stats[8], hist[K][n][a], per-threshold hists) in
        one launch (the torch version was ~15 small launches on the host's critical path)."""
        offs = tl.arange(0, BLOCK_R)
        m = offs < R
        nl = tl.load(cu_ptr + offs + 1, mask=m, other=0) - tl.load(cu_ptr + offs, mask=m, other=0)
        k = tl.maximum(nl - 1, 0).to(tl.int64)
        has = (k > 0) & m
        rs = tl.load(idx_ptr + offs, mask=m, other=0).to(tl.int64)
        n = tl.minimum(tl.load(nstar_ptr + rs, mask=has, other=0).to(tl.int64), k)
        n = tl.maximum(n, 0)
        a = tl.maximum(tl.load(ns_ptr + offs, mask=m, other=1).to(tl.int64) - 1, 0)
        z = tl.zeros((BLOCK_R,), tl.int64)
        dead = tl.where(has, k - n, z)
        tl.atomic_add(stats_ptr + 0, 1)
        tl.atomic_add(stats_ptr + 1, tl.sum(has.to(tl.int64), axis=0))
        tl.atomic_add(stats_ptr + 2, tl.sum(tl.where(has, k, z), axis=0))
        tl.atomic_add(stats_ptr + 3, tl.sum(dead, axis=0))
        tl.atomic_add(stats_ptr + 4, tl.sum(tl.where(has, a, z), axis=0))
        tl.atomic_add(stats_ptr + 5, tl.sum(tl.where(has, tl.maximum(a - n, 0), z), axis=0))
        tl.atomic_add(stats_ptr + 6, tl.sum(tl.where(has, n * (rs + 1), z), axis=0))
        tl.atomic_add(stats_ptr + 7, (tl.sum(dead, axis=0) > 0).to(tl.int64))
        cell = tl.minimum(k, 7) * 64 + tl.minimum(n, 7) * 8 + tl.minimum(a, 7)
        tl.atomic_add(hist_ptr + cell, tl.full((BLOCK_R,), 1, tl.int64), mask=has)
        if HAS_GRID:
            for g in tl.static_range(G):
                ng = tl.minimum(tl.load(ngrid_ptr + rs * G + g, mask=has, other=0).to(tl.int64), k)
                ng = tl.maximum(ng, 0)
                cg = g * 512 + tl.minimum(k, 7) * 64 + tl.minimum(ng, 7) * 8 + tl.minimum(a, 7)
                tl.atomic_add(histg_ptr + cg, tl.full((BLOCK_R,), 1, tl.int64), mask=has)

    _K["nstar"] = nstar_kernel
    _K["live"] = live_kernel
    _K["stats"] = stats_kernel
    return _K


GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)   # shadow mode: n* for every one of these thresholds too


def nstar_launch(scores, cand, toks, sidx, temp, nstar, num_reqs, num_steps, top_k, tau, min_live, grid=None,
                 ngrid=None):
    from vllm.triton_utils import triton
    has = grid is not None and ngrid is not None
    _kernels()["nstar"][(num_reqs,)](scores, cand, toks, sidx, temp, nstar, float(tau), int(min_live),
                                      grid if has else nstar, ngrid if has else nstar,
                                      NUM_STEPS=num_steps, TOP_K=top_k, BLOCK_K=triton.next_power_of_2(top_k),
                                      G=len(GRID), HAS_GRID=has, num_warps=1)


def nstar_reference(scores, cand, toks, sidx, temp, num_reqs, num_steps, top_k, tau, min_live):
    """CPU/torch reference of nstar_kernel: dict req_state -> n*."""
    out = {}
    sc = scores.reshape(-1, num_steps, top_k)[:num_reqs].double().cpu()
    cd = cand.reshape(-1, num_steps, top_k)[:num_reqs].cpu()
    tk = toks.reshape(-1, num_steps)[:num_reqs].cpu()
    for r in range(num_reqs):
        rs = int(sidx[r * num_steps])
        if rs < 0:
            continue
        t = float(temp[rs])
        t = t if t > 0 else 1.0
        surv, n = 1.0, 0
        for i in range(num_steps):
            q = torch.softmax(sc[r, i] / t, dim=0)
            qmax = float(q.max())
            hit = (cd[r, i] == tk[r, i]).nonzero()
            qd = float(q[hit[0, 0]]) if hit.numel() else 0.0
            if surv * qmax >= tau:
                n = i + 1
            else:
                break
            surv *= qd
        out[rs] = max(n, min_live)
    return out


# ----------------------------------------------------------------------------------------------- hooks
def _rank_str() -> str:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return "?"


def _hook_drafter(cls) -> None:
    orig = cls._sample_path
    if getattr(orig, "_glm53_spec_vtrim", False):
        return

    def _sample_path(self, candidate_ids, scores, num_reqs):
        orig(self, candidate_ids, scores, num_reqs)
        if ST.nstar is None:
            return
        nstar_launch(self._selector_scores, candidate_ids.contiguous(), self.draft_tokens, self.sample_idx_mapping,
                     self.temperature, ST.nstar, num_reqs, self.num_speculative_steps, self.selector_top_k,
                     ST.tau, ST.min_live, ST.grid, ST.ngrid)

    _sample_path._glm53_spec_vtrim = True
    _sample_path._glm53_spec_vtrim_orig = orig
    cls._sample_path = _sample_path
    ST.hooks.append("drafter")

    orig_p = cls.propose
    if getattr(orig_p, "_glm53_spec_vtrim", False):
        return

    def propose(self, input_batch, *a, **k):
        out = orig_p(self, input_batch, *a, **k)
        if ST.nstar is not None and getattr(input_batch, "has_structured_output_reqs", False):
            ST.nstar[input_batch.idx_mapping] = NO_TRIM          # grammar requests: never trimmed
            if ST.ngrid is not None:
                ST.ngrid[input_batch.idx_mapping] = NO_TRIM
            ST.skip_structured += 1
        return out

    propose._glm53_spec_vtrim = True
    propose._glm53_spec_vtrim_orig = orig_p
    cls.propose = propose
    ST.hooks.append("propose")


def _hook_combine(mr_mod) -> None:
    orig = mr_mod.combine_sampled_and_draft_tokens
    if getattr(orig, "_glm53_spec_vtrim", False):
        return

    def combine_sampled_and_draft_tokens(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
                                         prefill_len, draft_tokens, cu_num_logits, num_logits, *a, **k):
        out = orig(input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens, prefill_len, draft_tokens,
                   cu_num_logits, num_logits, *a, **k)
        if ST.mode == "on" and ST.live is not None:
            ST.live.fill_(1)
            nreq = idx_mapping.shape[0]
            if nreq and num_logits > nreq and int(input_ids.shape[0]) <= ST.live.shape[0]:
                from vllm.triton_utils import triton
                _kernels()["live"][(nreq,)](ST.live, idx_mapping, query_start_loc, cu_num_logits, ST.nstar,
                                            BLOCK=triton.next_power_of_2(draft_tokens.shape[-1] + 1), num_warps=1)
        return out

    combine_sampled_and_draft_tokens._glm53_spec_vtrim = True
    combine_sampled_and_draft_tokens._glm53_spec_vtrim_orig = orig
    mr_mod.combine_sampled_and_draft_tokens = combine_sampled_and_draft_tokens
    ST.hooks.append("combine")


def stats_reference(cu_num_logits, idx_mapping, num_sampled, nstar, ngrid=None):
    """torch reference of stats_kernel (unit tests): (stats[8], hist[512], hist_grid or None)."""
    nl = (cu_num_logits[1:] - cu_num_logits[:-1]).to(torch.int64)
    k = (nl - 1).clamp(min=0)
    has = k > 0
    rs = idx_mapping.to(torch.int64)
    n = torch.minimum(nstar[rs].to(torch.int64), k).clamp(min=0)
    a = (num_sampled.to(torch.int64) - 1).clamp(min=0)
    z = torch.zeros_like(k)
    dead = torch.where(has, k - n, z)
    st = torch.tensor([1, int(has.sum()), int(torch.where(has, k, z).sum()), int(dead.sum()),
                       int(torch.where(has, a, z).sum()), int(torch.where(has, (a - n).clamp(min=0), z).sum()),
                       int(torch.where(has, n * (rs + 1), z).sum()), int(dead.sum() > 0)], dtype=torch.int64)
    hist = torch.zeros(512, dtype=torch.int64)
    cell = (k.clamp(max=7) * 64 + n.clamp(max=7) * 8 + a.clamp(max=7))[has].cpu()
    hist.index_add_(0, cell, torch.ones_like(cell))
    hg = None
    if ngrid is not None:
        G = ngrid.shape[1]
        hg = torch.zeros(G * 512, dtype=torch.int64)
        for g in range(G):
            ng = torch.minimum(ngrid[rs, g].to(torch.int64), k).clamp(min=0)
            c = (g * 512 + k.clamp(max=7) * 64 + ng.clamp(max=7) * 8 + a.clamp(max=7))[has].cpu()
            hg.index_add_(0, c, torch.ones_like(c))
    return st, hist, hg


def stats_launch(cu_num_logits, idx_mapping, num_sampled, nstar, stats, hist, ngrid=None, hist_grid=None):
    from vllm.triton_utils import triton
    R = int(idx_mapping.shape[0])
    has = ngrid is not None and hist_grid is not None
    _kernels()["stats"][(1,)](cu_num_logits, idx_mapping, num_sampled, nstar, ngrid if has else nstar, stats, hist,
                              hist_grid if has else hist, R, G=len(GRID), HAS_GRID=has,
                              BLOCK_R=max(16, triton.next_power_of_2(max(R, 1))), num_warps=1)


def _stats_and_log(cu_num_logits, idx_mapping, num_sampled, n_eff_dead, nstar_req) -> None:
    """GPU accumulation in one kernel launch (no sync) + a periodic host log."""
    st = ST.stats
    stats_launch(cu_num_logits, idx_mapping, num_sampled, ST.nstar, st, ST.hist, ST.ngrid, ST.hist_grid)
    ST.steps += 1
    if ST.steps == ST.first_log or (ST.log_every and ST.steps % ST.log_every == 0):
        v = [int(x) for x in st.cpu().tolist()]
        steps, reqsteps, drafts, dead, acc, lost, nsum, tsteps = v
        what = "dead" if ST.mode == "on" else "would-be dead"
        _log.info("%s rank %s mode %s tau %.3f min %d: verify steps %d, request-steps %d, drafts %d, %s %d (%.1f%%), "
                  "accepted %d (%.3f/request-step)%s, trimmed steps %d, structured-skips %d, nsum %d",
                  TAG, ST.rank, ST.mode, ST.tau, ST.min_live, steps, reqsteps, drafts, what, dead,
                  100.0 * dead / max(drafts, 1), acc, acc / max(reqsteps, 1),
                  (f", accepted drafts the trim would cut {lost} ({100.0 * lost / max(acc, 1):.1f}%)"
                   if ST.mode == "shadow" else ""), tsteps, ST.skip_structured, nsum)
        if ST.hist is not None and ST.hist_file and ST.rank in ("0", "?"):
            try:
                import json
                h = ST.hist.view(8, 8, 8).cpu().tolist()
                tmp = ST.hist_file + ".tmp"
                with open(tmp, "w") as fh:
                    out = {"mode": ST.mode, "tau": ST.tau, "min": ST.min_live, "verify_steps": steps,
                           "axes": "hist[K][min(n*,K)][accepted]", "hist": h}
                    if ST.hist_grid is not None:
                        out["grid"] = list(GRID)
                        out["hist_grid"] = ST.hist_grid.view(len(GRID), 8, 8, 8).cpu().tolist()
                    json.dump(out, fh)
                os.replace(tmp, ST.hist_file)
            except Exception as exc:  # noqa: BLE001
                _log.warning("%s histogram not written (%r)", TAG, exc)
                ST.hist_file = None


def _hook_rejection(rs_mod) -> None:
    orig = rs_mod.rejection_sample
    if getattr(orig, "_glm53_spec_vtrim", False):
        return

    def rejection_sample(target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
                         expanded_idx_mapping, expanded_local_pos, *a, **k):
        nst = ST.nstar
        if nst is None or ST.disabled is not None:
            return orig(target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
                        expanded_idx_mapping, expanded_local_pos, *a, **k)
        nreq_row = nst[expanded_idx_mapping.to(torch.int64)].to(expanded_local_pos.dtype)
        if ST.mode == "on":
            dead = (expanded_local_pos >= 1) & (expanded_local_pos > nreq_row)
            draft_sampled = torch.where(dead, torch.full_like(draft_sampled, -1), draft_sampled)
        sampled, num_sampled = orig(target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
                                    expanded_idx_mapping, expanded_local_pos, *a, **k)
        if ST.stats_off is not None:
            return sampled, num_sampled
        try:
            _stats_and_log(cu_num_logits, idx_mapping, num_sampled, None, None)
        except Exception as exc:  # noqa: BLE001 - statistics never break sampling (the masking above already ran)
            _log.warning("%s stats off (%r); sampling unaffected", TAG, exc)
            ST.stats_off = repr(exc)
        return sampled, num_sampled

    rejection_sample._glm53_spec_vtrim = True
    rejection_sample._glm53_spec_vtrim_orig = orig
    rs_mod.rejection_sample = rejection_sample
    ST.hooks.append("rejection")


def _hook_moe(prodmod) -> None:
    orig = getattr(prodmod, "apply_exl3_fused_moe", None)
    if orig is None or getattr(orig, "_glm53_spec_vtrim", False):
        return

    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit):
        live = ST.live
        b = x2d.shape[0]
        if live is not None and b <= DECODE_MAX_TOKENS and ids.dim() == 2 and ids.shape[0] == b:
            ids = ids.masked_fill(live[:b].unsqueeze(1) == 0, -1)
        return orig(x2d, ids, weights, layer, inners, expert_map, limit)

    apply_exl3_fused_moe.__doc__ = getattr(orig, "__doc__", None)
    apply_exl3_fused_moe._glm53_spec_vtrim = True
    apply_exl3_fused_moe._glm53_spec_vtrim_orig = orig
    for attr in ("_tf_exl3_apply_hook", "_tf_exl3_orig"):     # keep integrate's identity checks working
        if hasattr(orig, attr):
            setattr(apply_exl3_fused_moe, attr, getattr(orig, attr))
    prodmod.apply_exl3_fused_moe = apply_exl3_fused_moe
    ST.hooks.append("moe")


def install(mode=None, tau=None, min_live=None, log_every=None, max_num_seqs=None, max_tokens=None,
            device=None) -> bool:
    """Install the hooks (idempotent). Called after the target weights are loaded (before any CUDA graph capture)."""
    if ST.installed:
        return True
    if mode is None:
        mode, tau, min_live, log_every = parse_env()
    if mode == "off":
        return False
    ST.mode, ST.tau, ST.min_live, ST.log_every = mode, float(tau), int(min_live), int(log_every)
    if max_num_seqs is None or max_tokens is None:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        max_num_seqs = max_num_seqs or int(cfg.scheduler_config.max_num_seqs)
        max_tokens = max_tokens or int(cfg.scheduler_config.max_num_batched_tokens)
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    ST.nstar = torch.full((int(max_num_seqs),), NO_TRIM, dtype=torch.int32, device=dev)
    ST.live = torch.ones(int(max_tokens) + 1024, dtype=torch.uint8, device=dev) if mode == "on" else None
    ST.stats = torch.zeros(8, dtype=torch.int64, device=dev)
    ST.hist = torch.zeros(512, dtype=torch.int64, device=dev)
    if mode == "shadow":
        ST.grid = torch.tensor(GRID, dtype=torch.float32, device=dev)
        ST.ngrid = torch.full((int(max_num_seqs), len(GRID)), NO_TRIM, dtype=torch.int32, device=dev)
        ST.hist_grid = torch.zeros(len(GRID) * 512, dtype=torch.int64, device=dev)
    root = os.environ.get("VLLM_CACHE_ROOT") or ""
    ST.hist_file = os.path.join(root, "glm53_spec_vtrim_hist.json") if root and os.path.isdir(root) else None
    ST.rank = _rank_str()
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as D2
    import vllm.v1.worker.gpu.model_runner as MR
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as RS
    _hook_drafter(D2.DFlash2Speculator)
    _hook_rejection(RS)
    if mode == "on":
        _hook_combine(MR)
        import vllm.model_executor.layers.quantization.exl3 as Q
        _hook_moe(Q)
    ST.installed = True
    _log.info("%s installed rank %s: mode %s tau %.3f min %d log %d (hooks: %s; max_num_seqs %d, live rows %s)",
              TAG, ST.rank, mode, ST.tau, ST.min_live, ST.log_every, ",".join(ST.hooks), int(max_num_seqs),
              "n/a" if ST.live is None else ST.live.numel())
    return True


def plugin_install() -> None:
    try:
        mode, tau, mn, lg = parse_env()
    except ValueError as exc:
        _log.warning("%s not installed: %s", TAG, exc)
        return
    _log.info("glm53_spec_vtrim plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              "off" if mode == "off" else f"{mode} tau {tau} min {mn}")
    if mode == "off":
        return
    try:
        import vllm.model_executor.model_loader.base_loader as BL
        orig = BL.process_weights_after_loading
        if getattr(orig, "_glm53_spec_vtrim_hook", False):
            return

        def process_weights_after_loading(model, model_config, target_device):
            orig(model, model_config, target_device)
            try:
                install(mode, tau, mn, lg)
            except Exception as exc:  # noqa: BLE001
                if mode == "on":
                    # fail closed and loud: a rank that does not trim while the other does would accept different
                    # drafts (the TP ranks would diverge). Both ranks run the same code, so both stop here.
                    _log.error("%s mode on but the hooks could not be installed (%r): refusing to start", TAG, exc)
                    raise
                _log.warning("%s not installed (%r)", TAG, exc)

        process_weights_after_loading._glm53_spec_vtrim_hook = True
        BL.process_weights_after_loading = process_weights_after_loading
    except Exception as exc:  # noqa: BLE001
        _log.warning("%s not installed: %r", TAG, exc)
