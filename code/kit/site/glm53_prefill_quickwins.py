"""GLM53_PREFILL_QUICKWINS: exact prefill-only quick wins (docs/PREFILL_QUICKWINS.md).

Each item changes how production computes a prefill step, never what it computes: every fast path is bit-identical to
production's code on the same inputs (tests/qw/test_quickwins.py checks torch.equal on production shapes), and runs only
for a step with at least GLM53_PREFILL_QUICKWINS_MIN_T tokens (default 256, above every CUDA-graph capture size) while
no CUDA graph is being captured. Anything else (decode, capture, an unexpected layout / dtype) runs production's own
statement, unchanged.

Items (GLM53_PREFILL_QUICKWINS = comma list, or "all"; unset / empty / 0 / off = nothing is installed):
  mla_bmm    MLA absorption matmuls (mla_attention.py MLAAttention.forward_impl W_UK bmm, _v_up_proj W_UV bmm).
             cuBLAS picks cutlass_80_wmma 32x32 kernels for these strided batched GEMMs on sm_121 (16-22 TFLOPS);
             a Triton strided batched GEMM (bf16 x bf16 -> fp32 mma, K in 64-steps, bf16 store) runs them at the
             memory roofline and gives the same bits (same k16 mma accumulation order; measured torch.equal).
  mla_index  SM90 sparse-MLA index conversion (flashinfer_mla_sparse_sm90.py forward_mqa): production runs
             full_like(-1) + convert/compact + clamp_(min=0) + int32 copy_ into the wrapper's kv_indices
             (4 kernels, 2 x 113 MB temporaries at 13824 tokens); one Triton kernel writes the same int32 values
             straight into kv_indices (compacted valid prefix, 0 tail).
  kda_conv   KDA short conv (glm5next kda.py Glm5NextLinearAttention._forward, prefill branch): production runs one
             merged conv over q|k|v whose output is channel-first, so q, k, v reach FLA as column slices and are
             copied by q.contiguous() / k.contiguous() / v.contiguous() (fla/ops/kda.py). The conv is per channel;
             three launches (q, k, v channels, same kernel, same weights / state slices) write q, k, v contiguous.
  mhc_aux    mHC aux layers (glm5next model.py Glm5NextModel.forward, EAGLE3/DFlash2 aux hidden states): production
             materializes hc_post for the aux value and the next layer's fused post->pre recomputes the same hc_post.
             The materialized streams are handed to the next layer as its residual with post=None, which runs the
             standalone pre (T > 16: the same tf32 prenorm GEMM + pre_big_fuse kernels, same n_splits).
  mhc_mean   the 4-stream mean (hc_contract = aten mean(dim=1)) computed inside the post kernel: for the aux layers
             (post output + mean in one pass instead of post, then a 453 MB re-read) and the last layer (mean only: the
             post output is never needed). A Triton post kernel with the tilelang kernel's exact arithmetic
             (fma(c, d, a0*b0) then the fma chain over the 4 streams, as nvcc contracts production's generated CUDA;
             bf16 RNE store) and aten's sequential fp32 sum * 0.25 (measured torch.equal against both).
  idx_gate   the sparse indexer's fp32 head gate at prefill (this fork's glm53_gemv::head_gate op, production
             fallback torch.mm(x.float(), w32)): cuBLAS runs it on a cutlass simt sgemm after a bf16->fp32 copy of x;
             a Triton IEEE-fp32 dot reads x as bf16 (exact conversion) and accumulates k sequentially like that sgemm.
             Only for M >= GATE_MIN_M, where cuBLAS does not split K; the first call at each M is checked against
             production's result. opt-dense: the verdict is PER M and NaN-aware. Production's first 16384-row call
             is vLLM's profile run on dummy activations; with non-finite values torch.equal is False even for
             identical bits, and the old global switch-off then disabled the item for the whole server life
             (production log 2026-10-02 08:07:51 "idx_gate result differs ... at M=16384": the -32 ms per 13,824-token
             chunk never ran). Now: a call whose production result is not all finite decides nothing (production's
             result is returned, the next call at that M is checked again); a real mismatch keeps only THAT M on
             production's op (WARNING once per M); the item turns off globally after GATE_MAX_BAD_M distinct
             mismatching M values (a systematically different cuBLAS).
             GLM53_PREFILL_QUICKWINS_GATE=fp32 (opt-in; default "exact"): also serve M < GATE_MIN_M (cuBLAS split-K
             there, a deterministic but different fp32 summation order: Triton result within 3e-7 * sum|x w| of
             float64 vs production's 5e-8, measured tests/optdense/probe_gate_det.py) and accept a checked M whose
             bits differ when every element is within GATE_FP32_TOL * sum|x w| of production's (the fp32-order class).

Enable: GLM53_PREFILL_QUICKWINS=<items> (read once when the vLLM plugin loads, integrate.plugin_register; independent
of TF_EXL3_MOE). Each item patches its production function when that module is imported (or at once if it already is),
only if the function's source is a fingerprinted version (sha256 of the AST); otherwise that item logs a WARNING and
stays off. Revert: unset and restart vLLM. TP=2: set it on both ranks (each rank computes the same thing either way).
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import linecache
import logging
import os
import sys
import textwrap

_log = logging.getLogger("vllm.glm53_prefill_quickwins")
ENV = "GLM53_PREFILL_QUICKWINS"
ENV_MIN_T = "GLM53_PREFILL_QUICKWINS_MIN_T"
ITEMS = ("mla_bmm", "mla_index", "kda_conv", "mhc_aux", "mhc_mean", "idx_gate")
_OFF = frozenset({"", "0", "off", "false", "no", "none"})
DEFAULT_MIN_T = 256
# idx_gate: cuBLAS runs production's fp32 head-gate GEMM (M x 4096 x 32) as one cutlass simt sgemm (sequential k) only
# from M ~ 10k on; below it splits K (memset + atomics or splitKreduce), a different summation order. Measured on nodeC:
# M <= 9216 split, M >= 10240 not (tests/qw/probe_head_gate.py sweep). Each new M is also checked once at run time.
GATE_MIN_M = 10240
GATE_MAX_BAD_M = 4               # distinct mismatching M before idx_gate turns itself off for good
GATE_FP32_TOL = 2e-6             # fp32 mode: |fast - production| <= tol * sum_k |x_k w_k| (measured worst 3e-7)
ENV_GATE = "GLM53_PREFILL_QUICKWINS_GATE"
GATE_SMALL_M = 256               # fp32 mode serves M >= this (decode / capture never reach it)
GATE_TRUST_AFTER = 8             # fp32 mode: after this many checked M (and none bad) new M are served unchecked
M_GEMV = "glm53_gemv_install"

M_MLA = "vllm.model_executor.layers.attention.mla_attention"
M_SM90 = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
M_KDA = "vllm.models.glm5next.nvidia.kda"
M_CONV = "vllm.model_executor.layers.mamba.ops.causal_conv1d"
M_MODEL = "vllm.models.glm5next.nvidia.model"

# sha256(ast.dump(ast.parse(dedent(getsource(fn)))))[:16] of the production functions (the image
# ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor, with the launcher's runtime patches applied: they
# do not touch these functions; tests/qw/test_quickwins.py prints and checks them).
VERIFIED = {
    ("mla_bmm", "MLAAttention.forward_impl"): frozenset({"7b7cbfcfa94c8b92"}),
    ("mla_bmm", "MLAAttention._v_up_proj"): frozenset({"558a22f54b818511"}),
    ("mla_index", "FlashInferMLASparseSM90Impl.forward_mqa"): frozenset({"e43adcd5a6e2acde"}),
    ("kda_conv", "Glm5NextLinearAttention._forward"): frozenset({"1e4f45149fceddf5"}),
    ("kda_conv", "causal_conv1d_fn"): frozenset({"718ef047cdc3f7ac"}),
    ("mhc_aux", "Glm5NextModel.forward"): frozenset({"224750fda049837c"}),
    ("mhc_mean", "Glm5NextModel.forward"): frozenset({"224750fda049837c"}),
    ("mhc_mean", "Glm5NextDecoderLayer.forward"): frozenset({"9c0fe21938cdc177"}),
    ("idx_gate", "_head_gate_impl"): frozenset({"f2406f3c25bb4297"}),     # this fork's glm53_gemv_install (deploy-r13)
}

STATS = {k: 0 for k in ("mla_bmm_fast", "mla_bmm_prod", "mla_index_fast", "mla_index_prod", "kda_conv_fast",
                        "kda_conv_prod", "mhc_aux_reuse", "mhc_aux_prod", "mhc_mean_aux_fast", "mhc_mean_final_fast",
                        "mhc_mean_prod", "idx_gate_fast", "idx_gate_prod", "idx_gate_checked",
                        "idx_gate_nonfinite", "idx_gate_bad_m", "idx_gate_fp32_ok", "idx_gate_deferred")}
_STATE = {"items": frozenset(), "min_t": DEFAULT_MIN_T, "installed": {}, "refused": {}}
_LOGGED: set = set()


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def parse_env(environ=None) -> frozenset:
    """Items to install. Raises ValueError for an unknown item."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV)
    if raw is None or raw.strip().lower() in _OFF:
        return frozenset()
    names = {v.strip().lower() for v in raw.split(",") if v.strip()}
    if names & {"all", "1", "on", "true", "yes"}:
        return frozenset(ITEMS)
    bad = names - set(ITEMS)
    if bad:
        raise ValueError(f"{ENV}={raw!r}: unknown item(s) {sorted(bad)} (known: {', '.join(ITEMS)}, all)")
    return frozenset(names)


def parse_min_t(environ=None) -> int:
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_MIN_T) or "").strip()
    if not raw:
        return DEFAULT_MIN_T
    n = int(raw)
    if n < 65:
        raise ValueError(f"{ENV_MIN_T}={raw!r} must be >= 65 (above every CUDA-graph capture size)")
    return n


def source_fingerprint(fn) -> str | None:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(ast.dump(ast.parse(src)).encode()).hexdigest()[:16]


def _capturing() -> bool:
    import torch
    return torch.cuda.is_current_stream_capturing()


def _big(t: int) -> bool:
    return t >= _STATE["min_t"] and not _capturing()


# ---------------------------------------------------------------------------------------------------------------------
# kernels (Triton; compiled on first use)

_K = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def qw_bmm_kernel(A, B, C, M, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                      K: tl.constexpr, NN: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr, ORDER: tl.constexpr):
        # C[b] = A[b] @ B[b]; fp32 accumulation over K in BK steps (tl.dot = k16 mma sequence), bf16 store.
        pid = tl.program_id(0)
        n_t: tl.constexpr = NN // BN
        m_t = tl.cdiv(M, BM)
        if ORDER == 0:
            pn = pid % n_t
            pm = (pid // n_t) % m_t
            b = pid // (n_t * m_t)
        else:
            pn = pid % n_t
            b = (pid // n_t) % NB
            pm = pid // (n_t * NB)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptr = A + b.to(tl.int64) * sab + rm[:, None].to(tl.int64) * sam + rk[None, :] * sak
        b_ptr = B + b.to(tl.int64) * sbb + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):
            a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.0)
            w = tl.load(b_ptr)
            acc = tl.dot(a, w, acc)
            a_ptr += BK * sak
            b_ptr += BK * sbk
        c_ptr = C + b.to(tl.int64) * scb + rm[:, None].to(tl.int64) * scm + rn[None, :] * scn
        tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rm[:, None] < M)

    @triton.jit
    def qw_sm90_index_kernel(req_id_ptr, block_table_ptr, token_indices_ptr, out_ptr, max_num_blocks_per_req,
                             bt_stride0, bt_stride1, ti_stride0, ti_stride1,
                             BLOCK_SIZE: tl.constexpr, W: tl.constexpr):
        # = sparse_utils._convert_req_index_to_global_index_kernel(SINGLE_TILE, COMPACT_TO_FRONT, DCP_SIZE=1,
        #   no prefill workspace) on a -1-filled row, then clamp_(min=0), int32, row-major into out (row stride W).
        token_id = tl.program_id(0)
        indice_id = tl.arange(0, W)
        req = tl.load(req_id_ptr + token_id)
        tok = tl.load(token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1)
        is_invalid_tok = tok < 0
        block_id = tok // BLOCK_SIZE
        inblock_off = tok % BLOCK_SIZE
        valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
        bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
        is_invalid_tok |= ~valid_block
        base = tl.load(bt_ptr, mask=valid_block, other=0)
        out_val = base * BLOCK_SIZE + inblock_off
        out_val = tl.where(is_invalid_tok, -1, out_val)
        out_val = tl.maximum(out_val, 0)                     # clamp_(min=0)
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        cnt = tl.sum(is_valid)
        row = out_ptr + token_id.to(tl.int64) * W
        tl.store(row + local_offset, out_val, mask=is_valid == 1)
        tl.store(row + indice_id, tl.zeros_like(out_val), mask=indice_id >= cnt)   # the -1 tail, clamped

    @triton.jit
    def _post_j(A, C, n, j: tl.constexpr, d, b0, b1, b2, b3):
        # tilelang mhc_post_tilelang_kernel, stream j: x = c_j * d; x += a[i, j] * b_i (i = 0..3), fp32, as nvcc
        # contracts it (first add -> fma(c_j, d, a0 * b0), then fma(a_i, b_i, x)); bf16 round-to-nearest-even store.
        cj = tl.load(C + n * 4 + j)
        a0 = tl.load(A + n * 16 + 0 * 4 + j)
        a1 = tl.load(A + n * 16 + 1 * 4 + j)
        a2 = tl.load(A + n * 16 + 2 * 4 + j)
        a3 = tl.load(A + n * 16 + 3 * 4 + j)
        x = tl.fma(cj, d, a0 * b0)
        x = tl.fma(a1, b1, x)
        x = tl.fma(a2, b2, x)
        x = tl.fma(a3, b3, x)
        return x.to(tl.bfloat16)

    @triton.jit
    def qw_post_mean_kernel(A, Bres, C, D, OUT, MEAN, H: tl.constexpr, BH: tl.constexpr, WRITE_FULL: tl.constexpr):
        # OUT = production's mhc_post (optional), MEAN = aten mean(dim=1) of it: ((o0 + o1) + o2) + o3 in fp32, * 0.25
        n = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BH + tl.arange(0, BH)
        d = tl.load(D + n * H + offs).to(tl.float32)
        b0 = tl.load(Bres + n * 4 * H + 0 * H + offs).to(tl.float32)
        b1 = tl.load(Bres + n * 4 * H + 1 * H + offs).to(tl.float32)
        b2 = tl.load(Bres + n * 4 * H + 2 * H + offs).to(tl.float32)
        b3 = tl.load(Bres + n * 4 * H + 3 * H + offs).to(tl.float32)
        o0 = _post_j(A, C, n, 0, d, b0, b1, b2, b3)
        o1 = _post_j(A, C, n, 1, d, b0, b1, b2, b3)
        o2 = _post_j(A, C, n, 2, d, b0, b1, b2, b3)
        o3 = _post_j(A, C, n, 3, d, b0, b1, b2, b3)
        if WRITE_FULL:
            tl.store(OUT + n * 4 * H + 0 * H + offs, o0)
            tl.store(OUT + n * 4 * H + 1 * H + offs, o1)
            tl.store(OUT + n * 4 * H + 2 * H + offs, o2)
            tl.store(OUT + n * 4 * H + 3 * H + offs, o3)
        s = ((o0.to(tl.float32) + o1.to(tl.float32)) + o2.to(tl.float32)) + o3.to(tl.float32)
        tl.store(MEAN + n * H + offs, (s * 0.25).to(tl.bfloat16))

    @triton.jit
    def qw_gate_kernel(X, W, O, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        # O = float(X) @ W, X bf16 [M, K], W fp32 [K, N]: IEEE fp32 FMA, k sequential (tl.dot ieee)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.arange(0, N)
        acc = tl.zeros((BM, N), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(X + rm[:, None].to(tl.int64) * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
            b = tl.load(W + rk[:, None] * N + rn[None, :])
            acc = tl.dot(a.to(tl.float32), b, acc, input_precision="ieee")
        tl.store(O + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=rm[:, None] < M)

    _K.update(bmm=qw_bmm_kernel, index=qw_sm90_index_kernel, post_mean=qw_post_mean_kernel, gate=qw_gate_kernel,
              triton=triton)
    return _K


def _bmm_cfg(M: int, NN: int):
    """(BM, BN, BK, warps, stages, order) measured on nodeC GB10 (tests/qw/bench_mla_triton.py). BK is 64 in every
    config, so the accumulation order (and the result bits) do not depend on the choice."""
    if NN % 256 == 0 and M >= 4096:
        return (128, 256, 64, 8, 3, 0) if NN >= 512 else (64, 256, 64, 4, 3, 1)
    if NN % 128 == 0 and NN >= 512:
        return (64, 128, 64, 8, 3, 1)
    return (64, 64, 64, 4, 3, 1)


def qw_bmm(a, b, out) -> bool:
    """out[i] = a[i] @ b[i] (bf16, any strides) with the Triton kernel. Returns False (nothing done) when the shapes
    are outside what it handles; the caller then runs production's torch.bmm."""
    import torch
    if not (a.dtype == b.dtype == out.dtype == torch.bfloat16 and a.is_cuda and a.dim() == b.dim() == out.dim() == 3):
        return False
    NB, M, K = a.shape
    if b.shape[0] != NB or b.shape[1] != K or out.shape[0] != NB or out.shape[1] != M or out.shape[2] != b.shape[2]:
        return False
    NN = int(b.shape[2])
    if K % 64 or NN % 64 or M == 0:
        return False
    BM, BN, BK, warps, stages, order = _bmm_cfg(M, NN)
    k = _kernels()
    grid = (k["triton"].cdiv(M, BM) * (NN // BN) * NB,)
    k["bmm"][grid](a, b, out, M, *a.stride(), *b.stride(), *out.stride(), K=K, NN=NN, NB=NB, BM=BM, BN=BN, BK=BK,
                   ORDER=order, num_warps=warps, num_stages=stages)
    return True


def _qw_mla_bmm(a, b, out):
    """Replaces `torch.bmm(a, b, out=out)` in production's MLA absorption (W_UK in forward_impl, W_UV in _v_up_proj)."""
    import torch
    if _big(int(a.shape[1])) and qw_bmm(a, b, out):
        STATS["mla_bmm_fast"] += 1
        if STATS["mla_bmm_fast"] == 1:
            _log.info("glm53 prefill quickwins active: mla_bmm (first call %s x %s, %d tokens)", tuple(a.shape),
                      tuple(b.shape), int(a.shape[1]))
        return out
    STATS["mla_bmm_prod"] += 1
    return torch.bmm(a, b, out=out)


def _qw_sm90_convert(attn_metadata, topk_indices, num_tokens, state, convert):
    """Replaces production's convert + clamp + copy_ into state.kv_indices. Returns None when it wrote kv_indices
    itself, else production's topk_slots (the caller then runs production's clamp + copy_ unchanged)."""
    import torch
    req_id = attn_metadata.req_id_per_token[:num_tokens]
    block_table = attn_metadata.block_table
    W = int(topk_indices.shape[1]) if topk_indices.dim() == 2 else 0
    if (state is not None and _big(int(num_tokens)) and W > 0 and (W & (W - 1)) == 0 and W <= 8192
            and topk_indices.dtype == req_id.dtype == block_table.dtype == state.kv_indices.dtype == torch.int32
            and block_table.dim() == 2 and int(topk_indices.shape[0]) == int(num_tokens) == int(req_id.shape[0])
            and state.kv_indices.is_contiguous() and state.kv_indices.numel() >= num_tokens * W):
        k = _kernels()
        bt = block_table.contiguous()
        rid = req_id.contiguous()
        k["index"][(int(num_tokens),)](rid, bt, topk_indices, state.kv_indices, bt.shape[1], bt.stride(0),
                                       bt.stride(1), topk_indices.stride(0), topk_indices.stride(1),
                                       BLOCK_SIZE=int(attn_metadata.block_size), W=W, num_warps=8)
        STATS["mla_index_fast"] += 1
        if STATS["mla_index_fast"] == 1:
            _log.info("glm53 prefill quickwins active: mla_index (first call %d tokens x %d)", int(num_tokens), W)
        return None
    STATS["mla_index_prod"] += 1
    topk_slots, _valid_counts = convert(
        attn_metadata.req_id_per_token[:num_tokens],
        attn_metadata.block_table,
        topk_indices,
        BLOCK_SIZE=attn_metadata.block_size,
        NUM_TOPK_TOKENS=topk_indices.shape[1],
        return_valid_counts=True,
    )
    return topk_slots


_CONV_OUT = {}


def _qw_kda_conv(layer, qkv_ns, conv_weights, conv_bias, conv_state, has_initial_state, cache_indices,
                 query_start_loc, metadata, conv_fn):
    """Replaces production's merged conv + split. Returns (q, k, v)."""
    import torch
    P = int(layer.local_projection_size)
    fn_out = _CONV_OUT.get("fn")
    if (fn_out is not None and conv_bias is None and qkv_ns.dim() == 2 and int(qkv_ns.shape[1]) == 3 * P
            and conv_state.dtype == qkv_ns.dtype and conv_state.dim() == 3 and int(conv_state.shape[1]) == 3 * P
            and conv_weights.dim() == 2 and int(conv_weights.shape[0]) == 3 * P and _big(int(qkv_ns.shape[0]))):
        T = int(qkv_ns.shape[0])
        buf = torch.empty((3, T, P), dtype=qkv_ns.dtype, device=qkv_ns.device)
        xt = qkv_ns.transpose(0, 1)
        for j in range(3):
            sl = slice(j * P, (j + 1) * P)
            fn_out(xt[sl], conv_weights[sl], None, activation="silu", conv_states=conv_state[:, sl],
                   has_initial_state=has_initial_state, cache_indices=cache_indices, query_start_loc=query_start_loc,
                   metadata=metadata, out=buf[j].transpose(0, 1))
        STATS["kda_conv_fast"] += 1
        if STATS["kda_conv_fast"] == 1:
            _log.info("glm53 prefill quickwins active: kda_conv (first call %d tokens, 3 x %d channels)", T, P)
        return buf[0], buf[1], buf[2]
    STATS["kda_conv_prod"] += 1
    qkv_ns = conv_fn(
        qkv_ns.transpose(0, 1),
        conv_weights,
        conv_bias,
        activation="silu",
        conv_states=conv_state,
        has_initial_state=has_initial_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        metadata=metadata,
    ).transpose(0, 1)
    return qkv_ns.split(P, dim=-1)


def qw_post_mean(x, residual, post, comb, want_full: bool):
    """(post output or None, its mean over the 4 streams) = production's (mhc_post_tilelang(x, residual, post, comb),
    .mean(dim=1)); None when the layout is not the one production's kernels assume (the caller then runs production)."""
    import torch
    if residual.dim() != 3 or int(residual.shape[1]) != 4:
        return None
    T, _hc, H = (int(v) for v in residual.shape)
    if not (x.dtype == residual.dtype == torch.bfloat16 and post.dtype == comb.dtype == torch.float32 and x.is_cuda
            and tuple(x.shape) == (T, H) and post.numel() == 4 * T and tuple(comb.shape) == (T, 4, 4)
            and x.is_contiguous() and residual.is_contiguous() and post.is_contiguous() and comb.is_contiguous()
            and H % 1024 == 0):
        return None
    BH = 4096 if H % 4096 == 0 else 1024
    full = torch.empty_like(residual) if want_full else None
    mean = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
    _kernels()["post_mean"][(T, H // BH)](comb, residual, post, x, full if want_full else mean, mean, H=H, BH=BH,
                                          WRITE_FULL=bool(want_full), num_warps=8)
    return full, mean


def _qw_final_post(layer, x, residual, post, comb, hc_contract):
    """Production's last-layer `x = self.hc_post(x, residual, post, comb); x = hc_contract(x, self.n)`."""
    if "mhc_mean" in _STATE["items"] and int(getattr(layer, "n", 0)) == 4 and _big(int(x.shape[0])):
        r = qw_post_mean(x, residual, post, comb, False)
        if r is not None:
            STATS["mhc_mean_final_fast"] += 1
            if STATS["mhc_mean_final_fast"] == 1:
                _log.info("glm53 prefill quickwins active: mhc_mean (last layer, %d tokens)", int(x.shape[0]))
            return r[1]
    STATS["mhc_mean_prod"] += 1
    x = layer.hc_post(x, residual, post, comb)
    return hc_contract(x, layer.n)


def _qw_aux_post(model, idx, layer, hidden_states, residual, post, comb, hc_contract):
    """Production: value = hc_contract(layer.hc_post(hidden_states, residual, post, comb), layer.n) and the next layer
    recomputes the same hc_post inside its fused post->pre. Returns (value, hidden_states, residual, post, comb) for the
    next layer: the materialized streams with post=None when the next layer's standalone pre is the same computation."""
    T = int(hidden_states.shape[0])
    r = None
    if "mhc_mean" in _STATE["items"] and int(getattr(layer, "n", 0)) == 4 and _big(T):
        r = qw_post_mean(hidden_states, residual, post, comb, True)
    if r is not None:
        full, value = r
        STATS["mhc_mean_aux_fast"] += 1
        if STATS["mhc_mean_aux_fast"] == 1:
            _log.info("glm53 prefill quickwins active: mhc_mean (aux layer %d, %d tokens)", idx, T)
    else:
        full = layer.hc_post(hidden_states, residual, post, comb)
        value = hc_contract(full, layer.n)
    if "mhc_aux" not in _STATE["items"]:
        STATS["mhc_aux_prod"] += 1
        return value, hidden_states, residual, post, comb
    nxt = None
    try:
        nxt = model.layers[idx + 1] if idx + 1 < int(model.end_layer) else None
    except Exception:  # noqa: BLE001
        nxt = None
    if (nxt is not None and T > 16 and _big(T) and not getattr(model, "is_sequence_parallel", False)
            and getattr(nxt, "mhc", False) and not getattr(nxt, "is_mtp_layer", False)
            and int(getattr(nxt, "layer_idx", 0)) != 0 and hasattr(nxt, "hc_pre")):
        STATS["mhc_aux_reuse"] += 1
        if STATS["mhc_aux_reuse"] == 1:
            _log.info("glm53 prefill quickwins active: mhc_aux (first reuse at layer %d, %d tokens)", idx, T)
        return value, full, full, None, None
    STATS["mhc_aux_prod"] += 1
    return value, hidden_states, residual, post, comb


def qw_head_gate(x, w32):
    """float(x) @ w32 (x bf16 [M, K] contiguous, w32 fp32 [K, N] contiguous, N in {16, 32, 64}); None if not handled."""
    import torch
    if not (x.dtype == torch.bfloat16 and w32.dtype == torch.float32 and x.is_cuda and x.dim() == 2 and w32.dim() == 2
            and x.is_contiguous() and w32.is_contiguous() and int(x.shape[1]) == int(w32.shape[0])
            and int(w32.shape[1]) in (16, 32, 64) and int(x.shape[1]) % 32 == 0):
        return None
    M, K = (int(v) for v in x.shape)
    N = int(w32.shape[1])
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    k = _kernels()
    k["gate"][(k["triton"].cdiv(M, 32),)](x, w32, out, M, K=K, N=N, BM=32, BK=32, num_warps=4)
    return out


_HG = {"checked": set(), "off": False, "bad": set(), "mode": "exact", "deferred": {}}


def parse_gate_mode(environ=None) -> str:
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_GATE) or "").strip().lower()
    if raw in ("", "exact"):
        return "exact"
    if raw == "fp32":
        return "fp32"
    raise ValueError(f"{ENV_GATE}={raw!r} is not exact|fp32")


def _gate_min_m() -> int:
    return GATE_SMALL_M if _HG["mode"] == "fp32" else GATE_MIN_M


# opt-kdamhc-rev (969dc13, the TIGHTER NaN-aware check this kit ships): a check whose production result carries
# NaN/inf proves nothing at the positions that are NaN in both (the engine's dummy profile run can be ALL garbage),
# so such a call is served by production's op and the verdict for that M is DEFERRED; only after HG_MAX_DEFER
# non-finite calls at ONE M does the NaN-aware comparison (_same_bits_or_both_nan) decide - an M must not be marked
# 'checked' without a single real value compared (a later real step at that M would run unverified).
HG_MAX_DEFER = 8


def _same_bits_or_both_nan(got, ref) -> bool:
    """torch.equal, except that positions where BOTH are NaN count as equal: the engine's boot-time dummy (profile)
    run feeds the indexer garbage that can carry NaN, and NaN != NaN made the first check fail (opt-kdamhc: the
    idx_gate quick win switched itself off at boot at M = max_num_batched_tokens and never ran in production)."""
    import torch
    if torch.equal(got, ref):
        return True
    if got.shape != ref.shape:
        return False
    return bool(((got == ref) | (torch.isnan(got) & torch.isnan(ref))).all())


def _gate_verdict(x, w32, got, ref, M: int) -> str:
    """'ok' (bitwise), 'fp32' (fp32 mode: within the fp32-order bound), 'nonfinite' (undecidable: the caller
    defers, see HG_MAX_DEFER), or 'bad'."""
    import torch
    if not bool(torch.isfinite(ref).all().item()):
        return "nonfinite"
    if torch.equal(got, ref):
        return "ok"
    if _HG["mode"] == "fp32" and bool(torch.isfinite(got).all().item()):
        bound = torch.mm(x.float().abs(), w32.abs())
        if bool(((got - ref).abs() <= GATE_FP32_TOL * bound).all().item()):
            return "fp32"
    return "bad"


def _make_head_gate_impl(orig):
    def _head_gate_impl(x, weight, w32, handle, row0):
        M = int(x.shape[0]) if x.dim() == 2 else -1
        if ("idx_gate" in _STATE["items"] and not _HG["off"] and M >= _gate_min_m() and M not in _HG["bad"]
                and not _capturing()):
            trusted = _HG["mode"] == "fp32" and not _HG["bad"] and len(_HG["checked"]) >= GATE_TRUST_AFTER
            if M not in _HG["checked"] and not trusted and len(_HG["checked"]) >= 256:
                STATS["idx_gate_prod"] += 1          # verdict table full: an unchecked M stays on production's op
                return orig(x, weight, w32, handle, row0)
            got = qw_head_gate(x, w32)
            if got is not None:
                if M not in _HG["checked"] and not trusted:
                    import torch
                    ref = orig(x, weight, w32, handle, row0)
                    if not bool(torch.isfinite(ref).all()):
                        # opt-kdamhc-rev: DEFER (the dummy profile run can be all garbage) - production's op
                        # serves this call; after HG_MAX_DEFER deferred calls at this M the NaN-aware comparison
                        # decides on the next non-finite call (969dc13's order; tests/opt_kdamhc P4).
                        # r16z5rev: an ALL-non-finite result never decides (nothing was compared): the merged code
                        # marked M=16384 'checked' after the profile run's 8 all-NaN layer calls, and every later real
                        # call at that M was served unverified (tests/r16z5rev/idx_gate_allnan_probe.py).
                        nd = _HG["deferred"].get(M, 0) + 1
                        _HG["deferred"][M] = nd
                        if nd == 1:
                            _log.info("glm53 prefill quickwins: idx_gate check at M=%d deferred: production's result "
                                      "is not finite (NaN %d, inf %d of %d; the dummy profile run?) - production's "
                                      "op serves, the next finite call at this M decides", M,
                                      int(torch.isnan(ref).sum()), int(torch.isinf(ref).sum()), ref.numel())
                        STATS["idx_gate_nonfinite"] += 1
                        if nd <= HG_MAX_DEFER or not bool(torch.isfinite(ref).any()):
                            STATS["idx_gate_deferred"] += 1
                            return ref
                        if not _same_bits_or_both_nan(got, ref):
                            _HG["bad"].add(M)
                            STATS["idx_gate_bad_m"] += 1
                            if len(_HG["bad"]) >= GATE_MAX_BAD_M:
                                _HG["off"] = True
                            _log.warning("glm53 prefill quickwins: idx_gate M=%d differs from production's op "
                                         "(cuBLAS chose another reduction?; the NaN-aware comparison after %d "
                                         "non-finite calls); this M stays on production's op%s", M, nd,
                                         "; idx_gate turned off (%d distinct M differ)" % len(_HG["bad"])
                                         if _HG["off"] else "")
                            _HG["deferred"].pop(M, None)
                            return ref
                        v = "ok"
                    else:
                        v = _gate_verdict(x, w32, got, ref, M)
                    if v == "bad":
                        _HG["bad"].add(M)
                        STATS["idx_gate_bad_m"] += 1
                        if len(_HG["bad"]) >= GATE_MAX_BAD_M:
                            _HG["off"] = True
                        _log.warning("glm53 prefill quickwins: idx_gate M=%d differs from production's op (cuBLAS "
                                     "chose another reduction?); this M stays on production's op%s", M,
                                     "; idx_gate turned off (%d distinct M differ)" % len(_HG["bad"])
                                     if _HG["off"] else "")
                        return ref
                    if v == "fp32":
                        STATS["idx_gate_fp32_ok"] += 1
                    _HG["checked"].add(M)
                    STATS["idx_gate_checked"] += 1
                    _HG["deferred"].pop(M, None)
                STATS["idx_gate_fast"] += 1
                if STATS["idx_gate_fast"] == 1:
                    _log.info("glm53 prefill quickwins active: idx_gate (first call M=%d, checked %s)", M,
                              "bit-identical" if STATS["idx_gate_fp32_ok"] == 0 else "within the fp32-order bound")
                return got
        STATS["idx_gate_prod"] += 1
        return orig(x, weight, w32, handle, row0)

    _head_gate_impl._glm53_qw = True
    _head_gate_impl._glm53_qw_orig = orig
    return _head_gate_impl


def install_idx_gate() -> None:
    G = importlib.import_module(M_GEMV)
    cur = getattr(G, "_head_gate_impl", None)
    if getattr(cur, "_glm53_qw", False):
        return
    fp = source_fingerprint(cur)
    if fp not in VERIFIED[("idx_gate", "_head_gate_impl")]:
        raise Refused(f"{M_GEMV}._head_gate_impl fingerprint {fp} not verified")
    G._head_gate_impl = _make_head_gate_impl(cur)
    _STATE["installed"]["idx_gate"] = ["_head_gate_impl"]


# ---------------------------------------------------------------------------------------------------------------------
# source transplant: production's function with statements replaced, compiled in production's module globals

class Refused(Exception):
    pass


def _get_attr(mod, qual):
    obj = mod
    parts = qual.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    owner = obj
    fn = owner.__dict__[parts[-1]] if isinstance(owner, type) else getattr(owner, parts[-1])
    return owner, parts[-1], fn


def transplant(mod, qual: str, edits, fingerprints, *, install: bool = True):
    """Compile production's `qual` (in module `mod`) with each (old, new) text edit applied exactly once, in `mod`'s
    globals. Refused (nothing changed) unless the original's AST fingerprint is in `fingerprints` and every `old`
    occurs exactly once. install=False returns the new function without setting it."""
    owner, name, cur = _get_attr(mod, qual)
    if getattr(cur, "_glm53_qw", False):
        return cur
    fp = source_fingerprint(cur)
    if fp not in fingerprints:
        raise Refused(f"{mod.__name__}.{qual} fingerprint {fp} not in {sorted(fingerprints)}")
    src = inspect.getsource(cur)                  # production's indentation: the anchors are verbatim file text
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise Refused(f"{mod.__name__}.{qual}: edit anchor found {n} times: {old[:60]!r}")
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    if "super()" in src or "__class__" in src:
        raise Refused(f"{mod.__name__}.{qual}: uses super()/__class__ (cannot be recompiled outside the class)")
    fname = f"<glm53-quickwins {mod.__name__}.{qual}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns: dict = {}
    exec(compile(src, fname, "exec"), mod.__dict__, ns)
    new = ns[name]
    try:
        new.__qualname__ = cur.__qualname__
    except Exception:  # noqa: BLE001
        pass
    new._glm53_qw = True
    new._glm53_qw_orig = cur
    if install:
        setattr(owner, name, new)
    return new


# edits (old, new) per item; the old text is production's source verbatim
MLA_BMM_UK = ("torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)",
              "_glm53_qw_mla_bmm(mqa_q_nope, W_UK_T, mqa_ql_nope)")
MLA_BMM_UV = ("torch.bmm(x, self.W_UV, out=out.transpose(0, 1))",
              "_glm53_qw_mla_bmm(x, self.W_UV, out.transpose(0, 1))")
SM90_CONVERT = (
    """        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        state = _SM90_STATE
        assert state is not None
""",
    """        state = _SM90_STATE
        assert state is not None
        topk_slots = _glm53_qw_sm90_convert(
            attn_metadata, topk_indices, num_tokens, state, triton_convert_req_index_to_global_index
        )
""")
SM90_COPY = (
    """        width = topk_slots.shape[1]
        state.kv_indices[: num_tokens * width].copy_(
            topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)
        )
""",
    """        if topk_slots is not None:
            width = topk_slots.shape[1]
            state.kv_indices[: num_tokens * width].copy_(
                topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)
            )
""")
KDA_CONV = (
    """            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
""",
    """            q_ns, k_ns, v_ns = _glm53_qw_kda_conv(
                self,
                qkv_ns,
                conv_weights,
                conv_bias,
                conv_state,
                has_initial_state,
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
                attn_metadata_narrowed,
                causal_conv1d_fn,
            )
""")
CONV_OUT_SIG = ("    validate_data=False,\n):", "    validate_data=False,\n    out=None,\n):")
CONV_OUT_ALLOC = ("    out = torch.empty_like(x)\n", "    out = torch.empty_like(x) if out is None else out\n")
MHC_FINAL = (
    """        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None
""",
    """        if self.layer_idx == self.num_hidden_layers - 1:
            x = _glm53_qw_final_post(self, x, residual, post, comb, hc_contract)
            return x, None, None, None
""")
MHC_AUX = (
    """            if post is not None and hasattr(layer, "hc_post"):
                value = hc_contract(
                    layer.hc_post(hidden_states, residual, post, comb),
                    layer.n,
                )
""",
    """            if post is not None and hasattr(layer, "hc_post"):
                value, hidden_states, residual, post, comb = _glm53_qw_aux_post(
                    self, idx, layer, hidden_states, residual, post, comb, hc_contract
                )
""")

# item -> [(module, qualname, edits)]; the helpers each item injects into its module's globals
PLAN = {
    "mla_bmm": [(M_MLA, "MLAAttention.forward_impl", [MLA_BMM_UK]), (M_MLA, "MLAAttention._v_up_proj", [MLA_BMM_UV])],
    "mla_index": [(M_SM90, "FlashInferMLASparseSM90Impl.forward_mqa", [SM90_CONVERT, SM90_COPY])],
    "kda_conv": [(M_KDA, "Glm5NextLinearAttention._forward", [KDA_CONV])],
    "mhc_aux": [(M_MODEL, "Glm5NextModel.forward", [MHC_AUX])],
    "mhc_mean": [(M_MODEL, "Glm5NextModel.forward", [MHC_AUX]), (M_MODEL, "Glm5NextDecoderLayer.forward", [MHC_FINAL])],
    "idx_gate": [],                                   # wraps this fork's glm53_gemv_install._head_gate_impl (install)
}
INJECT = {
    M_MLA: {"_glm53_qw_mla_bmm": _qw_mla_bmm},
    M_SM90: {"_glm53_qw_sm90_convert": _qw_sm90_convert},
    M_KDA: {"_glm53_qw_kda_conv": _qw_kda_conv},
    M_MODEL: {"_glm53_qw_aux_post": _qw_aux_post, "_glm53_qw_final_post": _qw_final_post},
}


def _fps(item, qual):
    return VERIFIED.get((item, qual), frozenset())


def install_item(item: str, mod) -> None:
    """Patch every function of `item` in module `mod` (all or nothing for this module). Raises Refused."""
    entries = [e for e in PLAN[item] if e[0] == mod.__name__]
    if item == "kda_conv" and "fn" not in _CONV_OUT:
        conv = importlib.import_module(M_CONV)
        _CONV_OUT["fn"] = transplant(conv, "causal_conv1d_fn", [CONV_OUT_SIG, CONV_OUT_ALLOC],
                                     _fps("kda_conv", "causal_conv1d_fn"), install=False)
    new = []
    for _m, qual, edits in entries:            # compile all first: nothing is set unless every function verifies
        new.append((qual, transplant(mod, qual, edits, _fps(item, qual), install=False)))
    for k, v in INJECT.get(mod.__name__, {}).items():
        mod.__dict__[k] = v
    for qual, fn in new:
        owner, name, _cur = _get_attr(mod, qual)
        setattr(owner, name, fn)
    _STATE["installed"].setdefault(item, []).extend(q for q, _ in new)


def uninstall_item(item: str) -> None:
    if item == "idx_gate":
        G = sys.modules.get(M_GEMV)
        cur = getattr(G, "_head_gate_impl", None) if G is not None else None
        if getattr(cur, "_glm53_qw", False):
            G._head_gate_impl = cur._glm53_qw_orig
        _STATE["installed"].pop(item, None)
        return
    for modname, qual, _e in PLAN[item]:
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        owner, name, cur = _get_attr(mod, qual)
        if getattr(cur, "_glm53_qw", False):
            setattr(owner, name, cur._glm53_qw_orig)
    _STATE["installed"].pop(item, None)


def _on_import(mod) -> None:
    for item in sorted(_STATE["items"]):
        if any(e[0] == mod.__name__ for e in PLAN[item]):
            try:
                install_item(item, mod)
                _log.info("glm53 prefill quickwins: %s installed in %s (steps >= %d tokens, outside CUDA-graph "
                          "capture)", item, mod.__name__, _STATE["min_t"])
            except Exception as exc:  # noqa: BLE001
                _STATE["refused"][item] = repr(exc)
                _log.warning("glm53 prefill quickwins: %s NOT installed (production code unchanged): %r", item, exc)


class _Finder(importlib.abc.MetaPathFinder):
    """Patches a target module right after it executes (the first import), without importing it early."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path, target=None):
        if name not in self.names:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        self.names.discard(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _on_import(module)
        spec.loader.exec_module = exec_module
        return spec


def install(items=None, min_t=None) -> dict:
    """Install `items` (default: from the env). Modules already imported are patched now, the others when imported."""
    items = parse_env() if items is None else frozenset(items)
    _STATE["items"] = frozenset(items)
    _STATE["min_t"] = parse_min_t() if min_t is None else int(min_t)
    pending = set()
    for item in sorted(items):
        for modname, _q, _e in PLAN[item]:
            if modname in sys.modules:
                continue
            pending.add(modname)
    for modname in sorted({e[0] for it in items for e in PLAN[it]} - pending):
        _on_import(sys.modules[modname])
    if "idx_gate" in items:
        try:
            _HG["mode"] = parse_gate_mode()
            install_idx_gate()
            _log.info("glm53 prefill quickwins: idx_gate installed (%s._head_gate_impl, M >= %d, %s; per-M check)",
                      M_GEMV, _gate_min_m(), "bit-identical to production" if _HG["mode"] == "exact" else
                      "fp32-order mode (%s=fp32)" % ENV_GATE)
        except Exception as exc:  # noqa: BLE001
            _STATE["refused"]["idx_gate"] = repr(exc)
            _log.warning("glm53 prefill quickwins: idx_gate NOT installed (production code unchanged): %r", exc)
    if pending and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(pending))
    return {"items": sorted(items), "min_t": _STATE["min_t"], "pending": sorted(pending)}


def plugin_install() -> None:
    """Called from integrate.plugin_register (every vLLM process); never raises."""
    try:
        items = parse_env()
        min_t = parse_min_t()
    except ValueError as exc:
        _log.warning("glm53 prefill quickwins NOT installed: %s", exc)
        return
    if not items:
        return
    try:
        r = install(items, min_t)
        _log.info("glm53 prefill quickwins: %s=%s -> items %s (min tokens %d); pending imports %s", ENV,
                  os.environ.get(ENV), r["items"], r["min_t"], r["pending"])
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53 prefill quickwins install failed (production unchanged): %r", exc)
