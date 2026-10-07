#!/usr/bin/env python3
"""[glm53-mhc-sp2] Pipelined sequence-parallel mHC prefill (on top of GLM53_MHC_SP=1; default OFF = r16x byte-identical).

What r16x's GLM53_MHC_SP leaves on the table (docs/MHC_SP2.md)
---------------------------------------------------------------
Under SP every decoder layer runs, per phase (attention and MLP), strictly in series on ONE stream:

    reduce-scatter (rank partial -> my T/2 rows) -> mHC fused post/pre on T/2 rows -> all-gather (T/2 -> T)

The two collectives (56.6 MB each at a 13,824-token chunk, ~2.7 ms each on the RoCE pair) are fully exposed, as the
all-reduce was before (ANATOMY.md 4: 527 ms per chunk of exposed NCCL). The mHC kernels between them are DRAM-bound
(nodeC: 180-220 GB/s) and per-token, so they can be split into sub-chunks that each depend on one slice of the
collectives only.

What this does (GLM53_MHC_SP2=1, needs GLM53_MHC_SP=1)
------------------------------------------------------
1. Pipelined phases. The rank shard is laid out as k=2 interleaved blocks (rank r owns global token blocks r and
   r+2 of 4 equal blocks of B = ceil(T/4) rows), so an all-gather of sub-chunk j gives the global rows
   [2jB, 2(j+1)B) in natural order and a reduce-scatter of those rows gives rank r its sub-chunk j. Each phase
   then becomes, with the collectives on a side stream (one NCCL communicator, strictly serialized by events):

       RS0 | RS1 || mHC(sub0) | AG0 || mHC(sub1) | AG1          (|| = concurrent)

   so per phase one half-RS and one half-AG stay exposed instead of a full RS + AG. Only sub-chunks of
   >= MHC_SP2_MIN_SUB rows are pipelined (compute_num_split == 1 there: every mHC kernel is BITWISE the r16x SP
   result, and r16x SP is bitwise the TP result at these sizes); smaller steps fall back to r16x's single-shard SP.
   The mHC kernels are the production ones (mhc_post_tilelang / tf32_hc_prenorm_gemm / pre_big_fuse_with_norm),
   called through a mirror of mhc_fused_post_pre_tilelang that writes into row slices of preallocated outputs
   (no concatenation copies).
2. Odd token counts. r16x requires T % tp == 0 for SP; the stock sp_shard / sp_reduce_scatter already pad with zero
   rows and every gather is sliced back to T (attention, MLP, aux, final all see exactly T rows), so the parity
   condition is dropped: the request's tail chunk (e.g. 4,289 tokens) and odd mixed steps also get the halved mHC.

Collectives: vLLM's own PyNcclCommunicator of the TP group (all_gather / reduce_scatter with an explicit stream, the
same calls GroupCoordinator makes). Disabled pynccl / NCCL symmetric memory / non-CUDA -> no pipelining (r16x SP).

Edits (on top of patch_mhc_sp.py's model.py; all preflighted, idempotent, fail-closed):
  vllm/models/glm5next/nvidia/model.py   the helpers after _mhc_sp_begin, the parity gate, the decoder-layer mHC block
                                          (pipelined branch; the r16x branch kept verbatim), the shard / aux / final
                                          gathers of Glm5NextModel.forward
  glm53_prefill_quickwins.py             VERIFIED += the SP2 forwards' fingerprints (mhc_aux / mhc_mean anchors intact)
  glm53_moeglue.py                       WARM_VERIFIED += the SP2 layer forward's fingerprints (plain + quickwins)

Unset/empty -> patch_tf_bundle does not run it; 0 -> prints and touches nothing; 1 without GLM53_MHC_SP=1 or on a
model.py that patch_mhc_sp.py did not patch -> SystemExit (both ranks fail loudly; never a half-installed pair).
Env overrides for tests: GLM53_GLM5NEXT_MODEL_PY, GLM53_QUICKWINS_PY, GLM53_MOEGLUE_PY.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_mhc_sp as SP  # noqa: E402  (the r16x patch: MARK, fingerprint plumbing, install machinery)

ENV = "GLM53_MHC_SP2"
TAG = "[glm53-mhc-sp2]"
MARK = "# [glm53-mhc-sp2] pipelined sequence-parallel mHC prefill"

# ---------------------------------------------------------------------------------------------------------------------
# model.py hunks (anchors = the r16x SP-patched text)
# ---------------------------------------------------------------------------------------------------------------------
GATE_ANCHOR = """    return tp > 1 and full_num_tokens % tp == 0 and get_pp_group().world_size == 1
"""
GATE_PATCHED = """    @MARK@: odd T is fine (sp_shard /
    # sp_reduce_scatter pad with zero rows, every gather is sliced back to T rows)
    return tp > 1 and get_pp_group().world_size == 1
"""

HELPERS_ANCHOR = """        logger.info("glm53-mhc-sp: sequence-parallel mHC prefill %s (T=%d)", "ACTIVE" if sp else "off",
                    full_num_tokens)
"""
HELPERS_PATCHED = HELPERS_ANCHOR + '''

@MARK@ (docs/MHC_SP2.md; overlay/patch_mhc_sp2.py,
# GLM53_MHC_SP2=1). Under MHC_SP_ACTIVE the rank shard is k interleaved blocks (rank r owns global blocks r + N*j,
# j < k, of B rows each), so the reduce-scatter / all-gather of sub-chunk j touch only the global rows
# [j*N*B, (j+1)*N*B) and run on a side stream while the mHC of the neighbouring sub-chunk runs on the compute
# stream. Per-token kernels on sub-chunks of >= MHC_SP2_MIN_SUB rows are bitwise the r16x SP result.
MHC_SP2_K = 2
# opt-kdamhc: GLM53_MHC_SP2_K (unset = 2) - 1 keeps only the odd-T sharding (no side-stream pipelining), the A/B
# fallback arm if the pipelined overlap does not pay on RoCE. Must be EQUAL on both ranks (a k=2 rank issues 2+2
# collectives per phase, a k=1 rank 1+1: mismatch = hang); anything but 1/2/4 refuses at import (the engine fails).
_SP2_K_ENV = __import__("os").environ.get("GLM53_MHC_SP2_K", "").strip()
if _SP2_K_ENV:
    if _SP2_K_ENV not in ("1", "2", "4"):
        raise ValueError(f"GLM53_MHC_SP2_K must be 1, 2 or 4, got {_SP2_K_ENV!r}")
    MHC_SP2_K = int(_SP2_K_ENV)
MHC_SP2_MIN_SUB = 1537
_SP2 = {"k": 1, "B": 0, "N": 1, "r": 0, "T": 0, "pend": None, "comm": None, "stream": None, "logged": False}


def _sp2_comm():
    from vllm.distributed import get_tp_group
    dc = get_tp_group().device_communicator
    pc = getattr(dc, "pynccl_comm", None) if dc is not None else None
    if pc is None or getattr(pc, "disabled", True):
        return None
    try:
        from vllm.distributed.device_communicators.all_reduce_utils import should_nccl_symm_mem_ag_rs
        if should_nccl_symm_mem_ag_rs():
            return None
    except Exception:  # noqa: BLE001
        return None
    return pc


def _sp2_ag_into(out, inp, stream):
    _SP2["comm"].all_gather(out, inp, stream=stream)


def _sp2_rs_into(out, inp, stream):
    _SP2["comm"].reduce_scatter(out, inp, stream=stream)


def _sp2_stream():
    s = _SP2["stream"]
    if s is None or s.device.index != torch.cuda.current_device():
        s = _SP2["stream"] = torch.cuda.Stream(priority=-1)
    return s


def _sp2_sync() -> None:
    pend = _SP2["pend"]
    if pend is not None:
        _SP2["pend"] = None
        main = torch.cuda.current_stream()
        for ev in pend:
            main.wait_event(ev)


def _sp2_plan(full_num_tokens: int) -> None:
    # Per-forward: k sub-chunks when MHC_SP_ACTIVE and every sub-chunk keeps >= MHC_SP2_MIN_SUB rows; else 1
    # (= r16x's single-shard SP). Both ranks evaluate the same predicate on the same arguments.
    _sp2_sync()
    k, B = 1, 0
    if MHC_SP_ACTIVE and current_platform.is_cuda():
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        n = get_tensor_model_parallel_world_size()
        k = MHC_SP2_K
        while k > 1 and -(-full_num_tokens // (n * k)) < MHC_SP2_MIN_SUB:
            k //= 2
        if k > 1:
            comm = _sp2_comm()
            if comm is None:
                k = 1
            else:
                _SP2.update(comm=comm, N=n, r=get_tensor_model_parallel_rank())
                B = -(-full_num_tokens // (n * k))
    _SP2.update(k=k, B=B, T=full_num_tokens, pend=None)
    if MHC_SP_ACTIVE and not _SP2.get("klogged"):
        _SP2["klogged"] = True
        logger.info("glm53-mhc-sp2: GLM53_MHC_SP2_K=%d (1 = odd-T sharding only, no pipelining)", MHC_SP2_K)
    if k > 1 and not _SP2["logged"]:
        _SP2["logged"] = True
        logger.info("glm53-mhc-sp2: pipelined SP prefill ACTIVE (T=%d, k=%d sub-chunks of %d rows per rank)",
                    full_num_tokens, k, B)


def _sp2_shard(hidden_states, full_num_tokens: int):
    if _SP2["k"] == 1:
        return sp_shard(hidden_states)
    k, B, N, r = _SP2["k"], _SP2["B"], _SP2["N"], _SP2["r"]
    blocks = []
    for j in range(k):
        lo = (r + N * j) * B
        if lo + B <= full_num_tokens:
            blocks.append(hidden_states[lo:lo + B])
        else:
            blk = hidden_states.new_zeros((B,) + tuple(hidden_states.shape[1:]))
            if lo < full_num_tokens:
                blk[: full_num_tokens - lo] = hidden_states[lo:full_num_tokens]
            blocks.append(blk)
    return torch.cat(blocks, 0)


def _sp2_gather(x, full_num_tokens: int):
    # The layout-aware all-gather of a whole rank shard (aux / final / standalone-pre paths), on the compute stream.
    if _SP2["k"] == 1 or not MHC_SP_ACTIVE:
        return sp_all_gather(x)[:full_num_tokens]
    _sp2_sync()
    k, B, N = _SP2["k"], _SP2["B"], _SP2["N"]
    out = x.new_empty((N * k * B,) + tuple(x.shape[1:]))
    main = torch.cuda.current_stream()
    for j in range(k):
        _sp2_ag_into(out[j * N * B:(j + 1) * N * B], x[j * B:(j + 1) * B].contiguous(), main)
    return out[:full_num_tokens]


def _sp2_reduce_scatter_begin(x, wait: bool):
    # The rank partial [T, H] -> this rank's k sub-chunks [k*B, H], asynchronously on the side stream: the next
    # _sp2_fused_post_pre_gather waits per sub-chunk; anything else must _sp2_sync() first (wait=True does).
    _sp2_sync()
    k, B, N = _SP2["k"], _SP2["B"], _SP2["N"]
    main, cs = torch.cuda.current_stream(), _sp2_stream()
    out = x.new_empty((k * B,) + tuple(x.shape[1:]))
    inputs = []
    for j in range(k):
        lo, hi = j * N * B, (j + 1) * N * B
        if hi <= x.shape[0]:
            inputs.append(x[lo:hi])
        else:
            blk = x.new_zeros((N * B,) + tuple(x.shape[1:]))
            if lo < x.shape[0]:
                blk[: x.shape[0] - lo] = x[lo:]
            inputs.append(blk)
    cs.wait_stream(main)
    evs = []
    with torch.cuda.stream(cs):
        for j in range(k):
            _sp2_rs_into(out[j * B:(j + 1) * B], inputs[j], cs)
            evs.append(cs.record_event())
    x.record_stream(cs)
    for t in inputs:
        t.record_stream(cs)
    _SP2["pend"] = evs
    if wait:
        _sp2_sync()
    return out


def _sp2_post_pre_into(x, residual, post, comb, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                       hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
                       residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur):
    # vllm mhc_fused_post_pre_tilelang (num_tokens > 16 branch), writing into the given row slices instead of fresh
    # tensors: the same kernels with the same arguments, so the rows are bitwise the production op's.
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    from vllm.utils.math_utils import cdiv

    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    num_tokens = residual.shape[0]
    if norm_weight is not None:
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()
    use_deep_gemm = is_deep_gemm_supported()
    n_splits = compute_num_split(64, hc_mult * hidden_size, cdiv(num_tokens, 64)) if use_deep_gemm else 1
    gemm_out_mul = torch.empty(n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=residual.device)
    mhc_post_tilelang(comb.view(num_tokens, hc_mult, hc_mult), residual, post.view(num_tokens, hc_mult),
                      x.view(num_tokens, hidden_size), residual_cur, hc_mult, hidden_size)
    residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
        tf32_hc_prenorm_gemm(residual_cur_2d, fn, gemm_out_mul, gemm_out_sqrsum, n_splits)
    else:
        from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
        _tilelang_hc_prenorm_gemm(residual_cur_2d, fn, gemm_out_mul, gemm_out_sqrsum, hidden_size, hc_mult)
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual_cur, post_mix_cur,
                                  comb_mix_cur, layer_input_cur, hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                  hc_post_mult_value, sinkhorn_repeat, n_splits, hc_mult)
    else:
        mhc_pre_big_fuse_with_norm_tilelang(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual_cur,
                                            post_mix_cur, comb_mix_cur, layer_input_cur, norm_weight, hidden_size,
                                            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                                            sinkhorn_repeat, norm_eps, n_splits, hc_mult)


def _sp2_fused_post_pre_gather(layer, x, residual, post, comb, fn, hc_scale, hc_base, norm, full_num_tokens: int):
    # = layer.hc_fused_post_pre(...) on this rank's shard followed by the all-gather, pipelined over the k
    # sub-chunks: sub-chunk j waits only for its reduce-scatter slice (if one is pending) and its all-gather slice
    # runs on the side stream while sub-chunk j+1 computes. Returns (residual, post, comb, gathered x[:T]).
    pend, _SP2["pend"] = _SP2["pend"], None
    k, B, N = _SP2["k"], _SP2["B"], _SP2["N"]
    S = k * B
    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    dev = residual.device
    residual_cur = torch.empty((S, hc_mult, hidden_size), dtype=residual.dtype, device=dev)
    post_cur = torch.empty((S, hc_mult), dtype=torch.float32, device=dev)
    comb_cur = torch.empty((S, hc_mult * hc_mult), dtype=torch.float32, device=dev)
    layer_input = torch.empty((S, hidden_size), dtype=torch.bfloat16, device=dev)
    gathered = torch.empty((N * S, hidden_size), dtype=torch.bfloat16, device=dev)
    residual = residual.view(S, hc_mult, hidden_size)
    post = post.reshape(S, hc_mult)
    comb = comb.reshape(S, hc_mult, hc_mult)
    main, cs = torch.cuda.current_stream(), _sp2_stream()
    for j in range(k):
        lo, hi = j * B, (j + 1) * B
        if pend is not None:
            main.wait_event(pend[j])
        _sp2_post_pre_into(
            x[lo:hi], residual[lo:hi], post[lo:hi], comb[lo:hi], fn, hc_scale, hc_base, layer.rms_norm_eps,
            layer.hc_eps, layer.hc_eps, layer.mhc_post_mult_value, layer.mhc_sinkhorn_iterations,
            norm.weight.data, norm.variance_epsilon,
            residual_cur[lo:hi], post_cur[lo:hi], comb_cur[lo:hi], layer_input[lo:hi])
        cs.wait_stream(main)
        with torch.cuda.stream(cs):
            _sp2_ag_into(gathered[j * N * B:(j + 1) * N * B], layer_input[lo:hi], cs)
    layer_input.record_stream(cs)
    main.wait_stream(cs)
    return (residual_cur, post_cur.view(S, hc_mult, 1), comb_cur.view(S, hc_mult, hc_mult),
            gathered[:full_num_tokens])
'''

# the decoder-layer mHC block: the r16x SP-patched text (anchor) -> the pipelined branch + the r16x branch verbatim
LAYER_ANCHOR = """        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
"""
LAYER_PATCHED = """        x = hidden_states
        @MARK@: the residual stream is
        # k interleaved sub-chunks per rank; collectives on the side stream (helpers next to _mhc_sp_begin).
        _sp2 = MHC_SP_ACTIVE and _SP2["k"] > 1 and not self.is_sequence_parallel
        if post is None:
            if _sp2:
                _sp2_sync()
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
            if _sp2:
                x = _sp2_gather(x, positions.shape[0])
        elif _sp2:
            residual, post, comb, x = _sp2_fused_post_pre_gather(
                self, x, residual, post, comb, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                self.input_layernorm, positions.shape[0],
            )
        else:
"""

ATTN_AG_ANCHOR = """        if self.is_sequence_parallel or MHC_SP_ACTIVE:
            # [glm53-mhc-sp] sequence-parallel mHC prefill: the mHC above ran on
            # this rank's token shard; attention needs the full sequence (the code below is unchanged: it
            # also reads the full positions / metadata). Static is_sequence_parallel (EP) shares the branch.
            x = sp_all_gather(x)[: positions.shape[0]]
"""
ATTN_AG_PATCHED = ATTN_AG_ANCHOR.replace(
    "        if self.is_sequence_parallel or MHC_SP_ACTIVE:\n",
    "        if (self.is_sequence_parallel or MHC_SP_ACTIVE) and not _sp2:  @MARK@\n", 1)

ATTN_RS_ANCHOR = """        if self.is_sequence_parallel or MHC_SP_ACTIVE:
            x = sp_reduce_scatter(x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
"""
ATTN_RS_PATCHED = """        if _sp2:
            @MARK@
            x = _sp2_reduce_scatter_begin(x, False)
            residual, post, comb, x = _sp2_fused_post_pre_gather(
                self, x, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                self.post_attention_layernorm, positions.shape[0],
            )
        else:
            if self.is_sequence_parallel or MHC_SP_ACTIVE:
                x = sp_reduce_scatter(x)

            # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm_weight=self.post_attention_layernorm.weight.data,
                norm_eps=self.post_attention_layernorm.variance_epsilon,
            )
"""

MLP_AG_ANCHOR = """        if MHC_SP_ACTIVE and not self.is_sequence_parallel:
            # [glm53-mhc-sp] sequence-parallel mHC prefill: the mHC above ran on the
"""
MLP_AG_PATCHED = """        if MHC_SP_ACTIVE and not self.is_sequence_parallel and not _sp2:  @MARK@
            # [glm53-mhc-sp] sequence-parallel mHC prefill: the mHC above ran on the
"""

MLP_RS_ANCHOR = """            x = self.mlp(x)
        if MHC_SP_ACTIVE and not self.is_sequence_parallel:
            x = sp_reduce_scatter(x)
"""
MLP_RS_PATCHED = """            x = self.mlp(x)
        if _sp2:
            @MARK@: the last layer's
            # hc_post below reads x, so it waits; every other layer hands the pending slices to the next layer.
            x = _sp2_reduce_scatter_begin(x, self.layer_idx == self.num_hidden_layers - 1)
        elif MHC_SP_ACTIVE and not self.is_sequence_parallel:
            x = sp_reduce_scatter(x)
"""

SHARD_ANCHOR = """            _mhc_sp_begin(self, full_num_tokens)
            if MHC_SP_ACTIVE:
                # [glm53-mhc-sp] sequence-parallel mHC prefill: this rank keeps its
                # token shard of the residual stream for the whole stack (see the block comment above).
                hidden_states = sp_shard(hidden_states)
"""
SHARD_PATCHED = """            _mhc_sp_begin(self, full_num_tokens)
            _sp2_plan(full_num_tokens)  @MARK@
            if MHC_SP_ACTIVE:
                # [glm53-mhc-sp] sequence-parallel mHC prefill: this rank keeps its
                # token shard of the residual stream for the whole stack (see the block comment above).
                hidden_states = _sp2_shard(hidden_states, full_num_tokens)
"""

AUX_SYNC_ANCHOR = """            if idx + 1 not in self.aux_hidden_state_layers:
                continue
            # Mid-stack mHC defers hc_post; materialize then contract
"""
AUX_SYNC_PATCHED = """            if idx + 1 not in self.aux_hidden_state_layers:
                continue
            _sp2_sync()  @MARK@: the layer's reduce-scatter may be pending
            # Mid-stack mHC defers hc_post; materialize then contract
"""

AUX_AG_ANCHOR = """                value = sp_all_gather(value)[:full_num_tokens].clone()
"""
AUX_AG_PATCHED = """                value = _sp2_gather(value, full_num_tokens).clone()  @MARK@
"""

FINAL_AG_ANCHOR = """            # comes back together for the norm / lm_head.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
"""
FINAL_AG_PATCHED = """            # comes back together for the norm / lm_head.
            hidden_states = _sp2_gather(hidden_states, full_num_tokens)  @MARK@
"""

HUNKS = (GATE_ANCHOR, GATE_PATCHED, HELPERS_ANCHOR, HELPERS_PATCHED, LAYER_ANCHOR, LAYER_PATCHED,
         ATTN_AG_ANCHOR, ATTN_AG_PATCHED, ATTN_RS_ANCHOR, ATTN_RS_PATCHED, MLP_AG_ANCHOR, MLP_AG_PATCHED,
         MLP_RS_ANCHOR, MLP_RS_PATCHED, SHARD_ANCHOR, SHARD_PATCHED, AUX_SYNC_ANCHOR, AUX_SYNC_PATCHED,
         AUX_AG_ANCHOR, AUX_AG_PATCHED, FINAL_AG_ANCHOR, FINAL_AG_PATCHED)
MODEL_HUNKS = tuple(h.replace("@MARK@", MARK) for h in HUNKS)


def prepare(src: str) -> str:
    """Idempotent, fail-closed: r16x-SP-patched -> SP2-patched; SP2-patched -> no-op; anything else -> ValueError."""
    if SP.MARK not in src:
        raise ValueError("model.py does not carry the r16x GLM53_MHC_SP patch (patch_mhc_sp.py must run first)")
    if MARK in src and "[glm53-sp-fp8ag]" in src:
        import patch_sp_fp8ag   # opt-kdamhc: a second bundle pass over a tree GLM53_SP_FP8AG extended after us
        if not patch_sp_fp8ag.verify(src):
            raise ValueError("partial patch: the sp-fp8ag marker is present but its tree is not fully applied")
        return src
    pairs = list(zip(MODEL_HUNKS[::2], MODEL_HUNKS[1::2]))
    if MARK in src:
        for _, new in pairs:
            if src.count(new) != 1:
                raise ValueError("partial patch: the sp2 marker is present but a hunk is not fully applied")
        return src
    out = src
    for anchor, new in pairs:
        if out.count(anchor) != 1:
            raise ValueError(f"anchor count {out.count(anchor)} != 1: {anchor.strip()[:70]!r}")
        out = out.replace(anchor, new, 1)
    for _, new in pairs:
        if out.count(new) != 1:
            raise ValueError("post-patch verification failed (a patched hunk is missing/ambiguous)")
    if prepare(out) != out:
        raise ValueError("post-patch verification failed (not recognised as already-applied)")
    return out


def sp2_fingerprints(patched_model_src: str) -> dict:
    fps = SP.sp_fingerprints(patched_model_src)   # same keys, computed on the SP2 text
    return {"model_forward_sp2": fps["model_forward_sp"], "layer_forward_sp2": fps["layer_forward_sp"],
            "layer_forward_sp2_qw": fps["layer_forward_sp_qw"]}


# the tables as patch_mhc_sp.py left them: the r16x line (its marker at the end) gains the SP2 fingerprints
QW_KEYS = ((r'    \("mhc_aux", "Glm5NextModel\.forward"\): frozenset\(\{[^}]*\}\),  ' + re.escape(SP.MARK), ("model_forward_sp2",)),
           (r'    \("mhc_mean", "Glm5NextModel\.forward"\): frozenset\(\{[^}]*\}\),  ' + re.escape(SP.MARK), ("model_forward_sp2",)),
           (r'    \("mhc_mean", "Glm5NextDecoderLayer\.forward"\): frozenset\(\{[^}]*\}\),  ' + re.escape(SP.MARK),
            ("layer_forward_sp2",)))
MG_KEYS = ((r'    "Glm5NextDecoderLayer\.forward": frozenset\(\{[^}]*\}\),  ' + re.escape(SP.MARK),
            ("layer_forward_sp2", "layer_forward_sp2_qw")),)


def prepare_table(src: str, keys, fps: dict, label: str) -> tuple[str, list[str]]:
    out, notes = src, []
    for pat, names in keys:
        ms = list(re.finditer(pat, out))
        if len(ms) != 1:
            notes.append(f"{label}: r16x SP line not found ({len(ms)}x), sp2 fingerprints not added: {pat[:60]!r}")
            continue
        line = ms[0].group(0)
        add = ", ".join(f'"{fps[n]}"' for n in names)
        new = line.replace("}),  " + SP.MARK, ", " + add + "}),  " + SP.MARK + "  " + MARK, 1)
        out = out.replace(line, new, 1)
    return out, notes


def main() -> int:
    val = os.environ.get(ENV, "").strip()
    if val == "0":
        print(f"{TAG} {ENV}=0: stock, files untouched")
        return 0
    if val != "1":
        raise SystemExit(f"{TAG} {ENV} must be exactly 1 to install (unset/empty/0 = r16x), got {val!r}")
    if os.environ.get(SP.ENV, "").strip() != "1":
        raise SystemExit(f"{TAG} {ENV}=1 needs {SP.ENV}=1 on this rank (the pipelined SP extends the r16x SP patch)")
    kv = os.environ.get("GLM53_MHC_SP2_K", "").strip()
    if kv not in ("", "1", "2", "4"):
        raise SystemExit(f"{TAG} GLM53_MHC_SP2_K must be unset, 1, 2 or 4, got {kv!r}")
    model_path = SP.MODEL_PY
    if not model_path.is_file():
        raise SystemExit(f"{TAG} missing {model_path}")
    src = model_path.read_text()
    try:
        patched = prepare(src)
        compile(patched, str(model_path), "exec")
    except ValueError as exc:
        raise SystemExit(f"{TAG} preflight failed for vllm/models/glm5next/nvidia/model.py: {exc}") from exc
    fps = sp2_fingerprints(patched)
    notes, state = [], {}
    for path, keys, label in ((SP.QUICKWINS_PY, QW_KEYS, "glm53_prefill_quickwins.py"),
                              (SP.MOEGLUE_PY, MG_KEYS, "glm53_moeglue.py")):
        if not path.is_file():
            notes.append(f"{label}: not present, fingerprints not added")
            state[label] = "absent"
            continue
        tsrc = path.read_text()
        if MARK in tsrc:
            state[label] = "already present"
            continue
        tnew, tnotes = prepare_table(tsrc, keys, fps, label)
        notes.extend(tnotes)
        if tnew != tsrc:
            compile(tnew, str(path), "exec")
            SP.replace_file(path, tnew)
            SP.clear_pyc(path)
        state[label] = "updated" if tnew != tsrc else "unchanged"
    act = "already present" if patched == src else "patched"
    if patched != src:
        SP.replace_file(model_path, patched)
        SP.clear_pyc(model_path)
    print(f"{TAG} vllm/models/glm5next/nvidia/model.py: {act}; fingerprints "
          f"quickwins={state.get('glm53_prefill_quickwins.py')} moeglue={state.get('glm53_moeglue.py')}; "
          f"layer_forward_sp2={fps['layer_forward_sp2'][:6]} model_forward_sp2={fps['model_forward_sp2'][:6]}; "
          f"pipelined SP prefill installed (GLM53_MHC_SP2=1: k=2 sub-chunks per rank shard, collectives on a side "
          f"stream; odd T sharded; decode byte-identical)")
    for n in notes:
        print(f"{TAG} WARNING {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
