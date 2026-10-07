"""GLM53_DEC_KDA_LAZY: write ONE KDA recurrent state per verify step instead of one per verified row
(docs/DEC_KDA_LAZY.md).

Production's spec-verify step runs ``fused_recurrent_kda`` once per KDA layer (34 of 45 layers, every rank) over the
M = K + 1 rows of each request and stores the fp32 recurrent state after EVERY row into its own slot
(``ssm_state_indices[i_n, t]``; 32 heads x 128 x 128 x 4 B = 2 MiB per row per layer per rank), because the next step
starts from the state after the last ACCEPTED row (``num_accepted_tokens - 1``) and that is only known after
sampling. M = 5 / 6 / 8 -> 10 / 12 / 16 MiB of state written per layer per step, 340 - 544 MiB per step, of which only
one 2 MiB state per layer is ever read again.

With GLM53_DEC_KDA_LAZY=1:
  * verify (inside the target's decode CUDA graphs): the same kernel arithmetic (production's strided fused_recurrent
    kernel body, line for line), but no state store; instead the raw inputs of every row (k, v, g logits, beta logits:
    the values production's kernel loads, 24.6 KiB per row per layer) go to a per-layer scratch, together with the
    row's slot ids, its initial column and its row count (``meta``);
  * commit (once per step, eager, right after sampling, in MambaHybridModelState.postprocess_state BEFORE its own
    work): one launch over all lazy layers recomputes, from the untouched initial state and the saved inputs, the
    first A = max(num_sampled, 1) rows with the same arithmetic and stores the state after row A - 1 into column A - 1
    -- exactly the bytes production stored there -- and, when this step makes the mamba "align" prefix cache copy a
    block-aligned state (the same predicate as postprocess_mamba_fused_kernel), the state of that column too.
    Then every flag is cleared.
Everything the rest of vLLM reads afterwards (the next verify's initial column, the align pre-copy / post-copy
columns) holds production's bytes. Columns of rejected rows (and accepted rows that nobody reads) keep stale bytes
instead of production's never-read ones.

Bitwise: the verify outputs and every committed state are bit-identical to production's kernel
(tests/optdec/test_kda_lazy.py, real kernels, rtol = atol = 0, multi-step chains with random acceptance).
Only pure spec-verify batches (no prefill / plain decode rows: every FULL decode graph) take the lazy path; any other
call, any shape or dtype it does not take -> production's function unchanged (its per-row stores), and this layer's
pending flags are cleared so nothing stale is ever committed.
Memory per rank: scratch 3 x NL x max_num_seqs x (K+1) x 32 x 128 x 2 B + small tables (34 layers, 8 seqs, 8 rows:
51.7 MiB).
"""
from __future__ import annotations

import logging
import os
import threading

import torch

_log = logging.getLogger("vllm.glm53_kda_lazy")
_TRUE = frozenset({"1", "on", "true", "yes"})
ENV = "GLM53_DEC_KDA_LAZY"
VERSION = 2
# v2 (decode4-kdalazyfix): layers are keyed by (recurrent_state.data_ptr(), A_log.data_ptr()), not by the state tensor.
# GLM-5-Next's hybrid KV layout (vllm/v1/core/kv_cache_utils.py `_glm5_next_tensor_layout`) gives ONE recurrent-state
# view to one KDA layer of EACH mamba group (34 KDA layers / groups of 11 -> up to 4 layers per tensor, disjoint block
# ids). v1 keyed by the tensor: the second co-owner hit "a_log / g_bias tensors changed", took production's path and
# CLEARED the first co-owner's pending flags, so the first co-owner's verify was never committed and its recurrent state
# stopped advancing (production A/B 2026-10-04: dvptf 0.0115 -> 0.206, temp-0 outputs changed, doubled characters).
# v1's self-check could not see it: it compares the commit with production's kernel over the SAME saved rows, and a
# layer whose flags were cleared is simply not checked (checked_states stayed 0 while it logged "byte for byte").

_LOCK = threading.Lock()
COUNTERS = {"lazy_calls": 0, "prod_calls": 0, "commits": 0, "registered": 0, "verified": 0, "verify_mismatch": 0,
            "checked_states": 0, "checked_rows_gt1": 0, "checked_empty": 0, "shared_tensor_layers": 0}


def env_enabled(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


# ---------------------------------------------------------------------------------------------------------------
# kernels (built lazily: importing this module must not need triton / vllm)

_K: dict = {}


def _kernels():
    if _K:
        return _K
    from vllm.triton_utils import tl, triton
    from vllm.third_party.flash_linear_attention.ops.op import exp

    @triton.jit(do_not_specialize=["N", "T"])
    def _kda_lazy_verify_kernel(
        q, k, v, g, beta, o, h0, cu_seqlens, ssm_state_indices, num_accepted_tokens, a_log, g_bias, scale,
        sk, sv, sg, sb, meta,
        N: tl.int64, T: tl.int64,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
        stride_init_state_token: tl.constexpr, stride_indices_seq: tl.constexpr,
        stride_q_t, stride_k_t, stride_v_t, stride_beta_t,
        TMAX: tl.constexpr, META_W: tl.constexpr, LAZY: tl.constexpr, LOWER_BOUND: tl.constexpr,
    ):
        # Production's fused_recurrent_gated_delta_rule_fwd_kernel (strided, vLLM #55736 backport), specialised to
        # the spec-verify call (IS_VARLEN, IS_CONTINUOUS_BATCHING, IS_SPEC_DECODING, USE_INITIAL_STATE, IS_KDA,
        # COMPUTE_GATE / SAFE_GATE, SIGMOID_BETA, scalar beta, USE_QK_L2NORM_IN_KERNEL, INPLACE_FINAL_STATE). Every
        # arithmetic line is production's, in production's order. LAZY: rows' inputs -> scratch, no state store
        # (unless the row count exceeds the scratch, then production's per-row stores).
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
        p_meta = meta + i_n * META_W
        is_writer = (i_v == 0) & (i_hv == 0)
        if T == 0:
            if LAZY:
                if is_writer:
                    tl.store(p_meta, 0)
            return

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        p_q = q + bos * stride_q_t + i_h * K + o_k
        p_k = k + bos * stride_k_t + i_h * K + o_k
        p_v = v + bos * stride_v_t + i_hv * V + o_v
        p_beta = beta + bos * stride_beta_t + i_hv
        p_gk = g + (bos * HV + i_hv) * K + o_k

        b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))

        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t0).to(tl.int64)
        lazy = T <= TMAX
        if LAZY:
            if is_writer:
                ok = lazy & (state_idx > 0)
                tl.store(p_meta, ok.to(tl.int32))
                tl.store(p_meta + 1, T.to(tl.int32))
                tl.store(p_meta + 2, state_idx.to(tl.int32))     # the initial state's slot id itself
        if state_idx <= 0:
            return
        p_h0 = h0 + state_idx * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        # scratch row pointers (LAZY): [i_n, t, i_hv, :]
        p_sk = sk + (i_n * TMAX * HV + i_hv) * K + o_k
        p_sg = sg + (i_n * TMAX * HV + i_hv) * K + o_k
        p_sv = sv + (i_n * TMAX * HV + i_hv) * V + o_v
        p_sb = sb + i_n * TMAX * HV + i_hv
        save_kg = lazy & (i_v == 0)

        for i_t in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            if LAZY:
                # the raw values production's kernel just loaded (exact: same dtype in, same dtype out)
                tl.store(p_sk, tl.load(p_k, mask=mask_k, other=0), mask=mask_k & save_kg)
                tl.store(p_sv, tl.load(p_v, mask=mask_v, other=0), mask=mask_v & lazy)

            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale
            b_gk = tl.load(p_gk).to(tl.float32)
            if LAZY:
                tl.store(p_sg, tl.load(p_gk), mask=mask_k & save_kg)
            b_gk += tl.load(
                g_bias + i_h * K + o_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
            b_h *= exp(b_gk[None, :])
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_beta = tl.load(p_beta).to(tl.float32)
            if LAZY:
                if save_kg:
                    tl.store(p_sb, tl.load(p_beta))
            b_beta = tl.sigmoid(b_beta)
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            if LAZY:
                if is_writer & lazy:
                    tl.store(p_meta + 3 + i_t, final_state_idx.to(tl.int32))
                store_now = (final_state_idx > 0) & (T > TMAX)
            else:
                store_now = final_state_idx > 0
            if store_now:
                p_ht = h0 + final_state_idx * stride_init_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

            p_q += stride_q_t
            p_k += stride_k_t
            p_o += HV * V
            p_v += stride_v_t
            p_gk += HV * K
            p_beta += stride_beta_t
            p_sk += HV * K
            p_sg += HV * K
            p_sv += HV * V
            p_sb += HV

    @triton.jit
    def _kda_lazy_commit_kernel(
        anchor, state_off, alog_tab, gbias_tab, sk, sv, sg, sb, meta,
        num_sampled, idx_mapping, num_computed, num_reqs,
        STATE_STRIDE: tl.constexpr, NSEQ: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BK: tl.constexpr, BV: tl.constexpr, TMAX: tl.constexpr, META_W: tl.constexpr,
        SAMPLED_SCALAR: tl.constexpr, SAMPLED_VALUE: tl.constexpr,
        ALIGN: tl.constexpr, ALIGN_ALL: tl.constexpr, BLOCK_SIZE: tl.constexpr, LOWER_BOUND: tl.constexpr,
        dry_out, layer0, DRY: tl.constexpr,
    ):
        # DRY (self-check): one layer (layer0), the state after row A-1 goes to dry_out[i_n] instead of the cache
        i_v, i_nh, l = tl.program_id(0), tl.program_id(1), tl.program_id(2) + layer0
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)
        if i_n >= num_reqs:
            return
        p_meta = meta + (l * NSEQ + i_n) * META_W
        flag = tl.load(p_meta)
        if flag == 0:
            return
        T = tl.load(p_meta + 1)
        if SAMPLED_SCALAR:
            A = tl.full((), SAMPLED_VALUE, tl.int32)
        else:
            A = tl.load(num_sampled + i_n).to(tl.int32)
        A = tl.maximum(A, 1)                            # postprocess: num_accepted = max(num_sampled, 1)
        A = tl.minimum(A, T)
        bias = -1
        if ALIGN:
            req = tl.load(idx_mapping + i_n)
            if req >= 0:
                nc = tl.load(num_computed + req).to(tl.int32)      # post-step count (PRECOMPUTED_NEW_COMPUTED)
                running = nc - A + 1
                aligned = (nc // BLOCK_SIZE) * BLOCK_SIZE
                if aligned >= running:
                    bias = aligned - running
        state_idx = tl.load(p_meta + 2).to(tl.int64)
        if state_idx <= 0:
            return
        # every layer's state tensor addressed from one aligned anchor tensor + an element offset that is a multiple
        # of 16 (allocations are >= 512 B aligned), with production's constexpr slot stride: the same address
        # alignment facts production's kernel compiles with (they decide the register layout of the state tile and
        # with it the reduction order of every tl.sum).
        base = anchor + tl.multiple_of(tl.load(state_off + l), 16)
        sstride = STATE_STRIDE
        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        p_h0 = base + state_idx * sstride + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        b_a_log = tl.exp(tl.load(alog_tab + l * H + i_h).to(tl.float32))
        lb = l.to(tl.int64) * NSEQ * TMAX
        p_sk = sk + ((lb + i_n * TMAX) * HV + i_hv) * K + o_k
        p_sg = sg + ((lb + i_n * TMAX) * HV + i_hv) * K + o_k
        p_sv = sv + ((lb + i_n * TMAX) * HV + i_hv) * V + o_v
        p_sb = sb + (lb + i_n * TMAX) * HV + i_hv
        for i_t in range(0, A):
            b_k = tl.load(p_sk, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_sv, mask=mask_v, other=0).to(tl.float32)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_gk = tl.load(p_sg).to(tl.float32)
            b_gk += tl.load(
                gbias_tab + l * H * K + i_h * K + o_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
            b_h *= exp(b_gk[None, :])
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_beta = tl.load(p_sb).to(tl.float32)
            b_beta = tl.sigmoid(b_beta)
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]
            if DRY:
                if i_t == A - 1:
                    p_d = dry_out + i_n.to(tl.int64) * HV * V * K + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                    tl.store(p_d, b_h, mask=mask_h)
            else:
                keep = (i_t == A - 1) | (i_t == bias)
                if ALIGN_ALL:
                    keep = keep | (i_t >= 0)
                if keep:
                    slot = tl.load(p_meta + 3 + i_t).to(tl.int64)
                    if slot > 0:
                        p_ht = base + slot * sstride + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                        tl.store(p_ht, b_h, mask=mask_h)
            p_sk += HV * K
            p_sg += HV * K
            p_sv += HV * V
            p_sb += HV

    _K["verify"] = _kda_lazy_verify_kernel
    _K["commit"] = _kda_lazy_commit_kernel
    _K["triton"] = triton
    return _K


# ---------------------------------------------------------------------------------------------------------------
# state: scratch + per-layer tables (allocated once, before any graph capture)

class _Lazy:
    def __init__(self) -> None:
        self.enabled = False
        self.disabled_reason: str | None = None
        self.alloc = False
        self.nl = 0
        self.nseq = 0
        self.tmax = 0
        self.H = self.K = self.V = 0
        self.dtypes = None              # (k dtype, v dtype, g dtype, beta dtype)
        self.sk = self.sv = self.sg = self.sb = self.meta = None
        self.state_off = self.alog_tab = self.gbias_tab = None
        self.anchor = None                           # the first registered state tensor (kernel base pointer)
        self.slot_stride = None                      # elements between slots (the same for every layer)
        self.layers: dict[tuple, int] = {}          # (recurrent_state.data_ptr(), a_log.data_ptr()) -> layer slot
        self.layer_src: dict[int, tuple] = {}        # slot -> (a_log, g_bias) as registered
        self.logged: set = set()
        self.pending = False                         # a lazy verify ran since the last commit (host view)
        self.repair = False                          # self-check failed once: commits run production's kernel
        self.lower_bound = -5.0
        self.lb_seen = None                          # the KDA gate lower bound of the lazy calls


ST = _Lazy()


def _once(key, level, msg, *args) -> None:
    if key in ST.logged:
        return
    ST.logged.add(key)
    _log.log(level, msg, *args)


def allocate(nl: int, nseq: int, tmax: int, H: int, K: int, V: int, device, dtypes) -> None:
    """Scratch for nl KDA layers x nseq sequences x tmax rows (tests call it directly)."""
    kd, vd, gd, bd = dtypes
    ST.nl, ST.nseq, ST.tmax, ST.H, ST.K, ST.V, ST.dtypes = nl, nseq, tmax, H, K, V, dtypes
    ST.sk = torch.zeros(nl, nseq, tmax, H, K, dtype=kd, device=device)
    ST.sv = torch.zeros(nl, nseq, tmax, H, V, dtype=vd, device=device)
    ST.sg = torch.zeros(nl, nseq, tmax, H, K, dtype=gd, device=device)
    ST.sb = torch.zeros(nl, nseq, tmax, H, dtype=bd, device=device)
    ST.meta = torch.zeros(nl, nseq, 3 + tmax, dtype=torch.int32, device=device)
    ST.state_off = torch.zeros(nl, dtype=torch.int64, device=device)
    ST.anchor = None
    ST.slot_stride = None
    ST.alog_tab = torch.zeros(nl, H, dtype=torch.float32, device=device)
    ST.gbias_tab = torch.zeros(nl, H * K, dtype=torch.float32, device=device)
    ST.layers.clear()
    ST.layer_src.clear()
    ST.alloc = True


def reset_layers() -> None:
    """Forget every registered layer (tests)."""
    ST.lb_seen = None
    ST.layers.clear()
    ST.layer_src.clear()
    ST.anchor = None
    ST.slot_stride = None
    if ST.alloc:
        ST.meta.zero_()


def scratch_bytes() -> int:
    if not ST.alloc:
        return 0
    return sum(t.numel() * t.element_size() for t in (ST.sk, ST.sv, ST.sg, ST.sb, ST.meta, ST.state_off,
                                                        ST.alog_tab, ST.gbias_tab))


def layer_key(recurrent_state: torch.Tensor, a_log: torch.Tensor) -> tuple:
    """One key per KDA LAYER. The state tensor alone is not one: GLM-5-Next's hybrid layout shares one recurrent-state
    view between one KDA layer of each mamba group (disjoint block ids); A_log is a per-layer parameter."""
    return (recurrent_state.data_ptr(), a_log.data_ptr())


def register(recurrent_state: torch.Tensor, a_log: torch.Tensor, g_bias: torch.Tensor) -> int | None:
    """Scratch slot of this KDA layer (registering it if new). None: cannot (full, capturing, layout)."""
    key = layer_key(recurrent_state, a_log)
    slot = ST.layers.get(key)
    if slot is not None:
        return slot
    if len(ST.layers) >= ST.nl:
        _once("full", logging.WARNING, "glm53_kda_lazy: more KDA state tensors than the %d allocated layers; "
              "extra layers keep production's path", ST.nl)
        return None
    if torch.cuda.is_current_stream_capturing():
        _once("capreg", logging.WARNING, "glm53_kda_lazy: first call of a layer inside a CUDA-graph capture; "
              "that graph keeps production's path for it")
        return None
    if recurrent_state.dtype != torch.float32 or recurrent_state.dim() != 4 or not recurrent_state[0].is_contiguous():
        return None
    al = a_log.reshape(-1).to(torch.float32)
    gb = g_bias.reshape(-1).to(torch.float32)
    if al.numel() != ST.H or gb.numel() != ST.H * ST.K:
        return None
    # exactness of the fp32 copies: production's kernel loads these and converts .to(float32) - the same values
    if not (torch.equal(al.to(a_log.dtype).reshape(a_log.shape), a_log) and
            torch.equal(gb.to(g_bias.dtype).reshape(g_bias.shape), g_bias)):
        return None
    if ST.anchor is None:
        anchor, stride = recurrent_state, recurrent_state.stride(0)
    else:
        anchor, stride = ST.anchor, ST.slot_stride
    if recurrent_state.stride(0) != stride or recurrent_state.data_ptr() % 64 or anchor.data_ptr() % 64:
        _once("layout", logging.WARNING, "glm53_kda_lazy: a KDA state tensor with another slot stride or alignment "
              "(%d vs %s); that layer keeps production's path", recurrent_state.stride(0), stride)
        return None
    off = (recurrent_state.data_ptr() - anchor.data_ptr()) // 4
    slot = len(ST.layers)
    ST.anchor, ST.slot_stride = anchor, stride
    ST.state_off[slot] = off
    ST.alog_tab[slot].copy_(al)
    ST.gbias_tab[slot].copy_(gb)
    ST.layers[key] = slot
    ST.layer_src[slot] = (a_log, g_bias, recurrent_state)
    COUNTERS["registered"] += 1
    shared = sum(1 for k in ST.layers if k[0] == key[0])
    if shared > 1:
        COUNTERS["shared_tensor_layers"] += 1
    return slot


def _token_stride(x: torch.Tensor):
    st = x.stride()
    if x.dim() not in (3, 4) or st[-1] != 1:
        return None
    if x.dim() == 4 and st[2] != x.shape[3]:
        return None
    if st[1] < x.shape[2] * (x.shape[3] if x.dim() == 4 else 1):
        return None
    if x.shape[0] != 1:
        return None
    return st[1]


def lazy_verify(q, k, v, g, beta, scale, initial_state, cu_seqlens, ssm_state_indices, num_accepted_tokens,
                out, a_log, g_bias, lower_bound, slot: int):
    """Production's fused_recurrent_kda_fwd (strided) for the spec-verify call, LAZY state stores."""
    K_ = _kernels()
    B, T, H, Kd = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    N = len(cu_seqlens) - 1
    BK, BV = K_["triton"].next_power_of_2(Kd), min(K_["triton"].next_power_of_2(V), 8)
    g = g.contiguous()
    a_log = a_log.reshape(-1).contiguous()
    g_bias = g_bias.reshape(-1).contiguous()
    if out is None:
        o = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    else:
        assert out.shape == k.shape and out.dtype == k.dtype and out.is_contiguous()
        o = out
    if ssm_state_indices.ndim == 1:
        stride_indices_seq = ssm_state_indices.stride(0)
    else:
        stride_indices_seq = ssm_state_indices.stride(0)
    grid = (1, triton_cdiv(V, BV), N * HV)
    K_["verify"][grid](
        q, k, v, g, beta, o, initial_state, cu_seqlens, ssm_state_indices, num_accepted_tokens, a_log, g_bias, scale,
        ST.sk[slot], ST.sv[slot], ST.sg[slot], ST.sb[slot], ST.meta[slot],
        N, T, H=H, HV=HV, K=Kd, V=V, BK=BK, BV=BV,
        stride_init_state_token=initial_state.stride(0), stride_indices_seq=stride_indices_seq,
        stride_q_t=_token_stride(q), stride_k_t=_token_stride(k), stride_v_t=_token_stride(v),
        stride_beta_t=_token_stride(beta),
        TMAX=ST.tmax, META_W=3 + ST.tmax, LAZY=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        num_warps=1, num_stages=3,
    )
    return o, initial_state


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _accepted(num_sampled, nreq: int, T: torch.Tensor) -> torch.Tensor:
    """A per sequence on the GPU: clamp(max(num_sampled, 1), <= T) (int64)."""
    if isinstance(num_sampled, torch.Tensor):
        a = num_sampled[:nreq].to(torch.int64)
    else:
        a = torch.full((nreq,), int(num_sampled), dtype=torch.int64, device=T.device)
    return torch.minimum(torch.clamp(a, min=1), T)


def _align_bias(A, idx_mapping, num_computed, block_size, nreq):
    """The column postprocess_mamba_fused_kernel copies from (-1: none), the same arithmetic, on the GPU."""
    req = idx_mapping[:nreq].to(torch.int64)
    nc = num_computed[req.clamp(min=0)].to(torch.int64)
    running = nc - A + 1
    aligned = torch.div(nc, block_size, rounding_mode="floor") * block_size
    bias = torch.where((aligned >= running) & (req >= 0), aligned - running, torch.full_like(A, -1))
    return bias


def _reference(l: int, nreq: int, A, bias, align_all: bool, into: str):
    """Production's own kernel over the saved rows of layer l: the state after row A-1 (and the align column).
    into="temp": returns (states [nreq, H, V, K] after row A-1, mask of the sequences that were lazy) without touching
    the cache; into="slots": writes production's bytes into the cache slots (repair / fallback path)."""
    dev = ST.meta.device
    TM = ST.tmax
    m = ST.meta[l, :nreq].to(torch.int64)
    flag = m[:, 0] > 0
    T = torch.clamp(m[:, 1], min=1)
    A = torch.where(flag, torch.minimum(A, T), torch.ones_like(A))
    init = m[:, 2]
    slots = m[:, 3:3 + TM]
    state = ST.layer_src[l][2]
    ar = torch.arange(nreq, device=dev)
    cols = torch.arange(TM, device=dev)
    table = torch.zeros(2 * nreq, TM + 1, dtype=torch.int32, device=dev)
    if into == "temp":
        # two temp slots per sequence: 1+2n receives the state after row A-1 (and holds the initial state copy the
        # kernel starts from), 2+2n the align column's state when that column is another one
        tmp = torch.zeros(1 + 2 * nreq, *state.shape[1:], dtype=state.dtype, device=dev)
        tmp[1::2] = state[init.clamp(min=0)]
        last = (cols[None, :] == (A - 1)[:, None])
        bcol = (cols[None, :] == bias[:, None]) & (bias != A - 1)[:, None]
        row = torch.where(last, (1 + 2 * ar)[:, None], torch.where(bcol, (2 + 2 * ar)[:, None], torch.zeros_like(slots)))
        initcol = 1 + 2 * ar
        h0 = tmp
    else:
        keep = (cols[None, :] == (A - 1)[:, None]) | (cols[None, :] == bias[:, None])
        if align_all:
            keep = keep | (cols[None, :] < A[:, None])
        row = torch.where(keep, slots, torch.zeros_like(slots))
        initcol = init
        h0 = state
    row = torch.where(flag[:, None], row, torch.zeros_like(row))
    table[0::2, :TM] = row.to(torch.int32)
    table[0::2, TM] = torch.where(flag, initcol, torch.zeros_like(initcol)).to(torch.int32)
    nacc = torch.ones(2 * nreq, dtype=torch.int32, device=dev)
    nacc[0::2] = TM + 1
    starts = ar * TM
    cu = torch.empty(2 * nreq + 1, dtype=torch.int32, device=dev)
    cu[0:-1:2] = starts.to(torch.int32)
    cu[1::2] = (starts + A).to(torch.int32)
    cu[-1] = nreq * TM
    sh = (1, nreq * TM, ST.H, ST.K)
    k = ST.sk[l, :nreq].reshape(sh)
    v = ST.sv[l, :nreq].reshape(1, nreq * TM, ST.H, ST.V)
    g = ST.sg[l, :nreq].reshape(sh)
    b = ST.sb[l, :nreq].reshape(1, nreq * TM, ST.H)
    out = torch.empty(sh, dtype=k.dtype, device=dev)
    a_log, g_bias, _ = ST.layer_src[l]
    _ORIG["frk"](q=k, k=k, v=v, g=g, beta=b, initial_state=h0, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                 ssm_state_indices=table, num_accepted_tokens=nacc, out=out, sigmoid_beta=True, a_log=a_log,
                 g_bias=g_bias, compute_gate=True, lower_bound=ST.lower_bound)
    if into == "temp":
        return h0[1::2], h0[2::2], flag, A, slots
    return None


def _env_int(name, default):
    v = os.environ.get(name)
    try:
        return int(v) if v is not None and v.strip() else default
    except ValueError:
        return default


def commit(num_sampled, idx_mapping, num_reqs: int, num_computed=None, block_size: int | None = None,
           align: bool = False, align_all: bool = False, lower_bound: float = -5.0) -> None:
    """Materialize every pending lazy verify (all layers, rows < num_reqs), then clear every flag.

    Self-check (GLM53_DEC_KDA_LAZY_VERIFY first commits, default 64, then one in GLM53_DEC_KDA_LAZY_VERIFY_EVERY,
    default 1024; 0 = never): production's own kernel recomputes the committed state of every lazy sequence of every
    layer from the same saved rows (before the commit writes) and the two are compared byte for byte. Any difference
    -> WARNING, this step's states are rewritten with production's bytes, and every later commit runs production's
    kernel per layer (repair mode: exact, slower) for the life of the process (CUDA graphs already captured with the
    lazy verify keep needing a commit, so the commit itself is what falls back)."""
    if not ST.alloc or not ST.layers:
        return
    K_ = _kernels()
    nl = len(ST.layers)
    nreq = min(int(num_reqs), ST.nseq)
    if ST.lb_seen is not None:
        lower_bound = ST.lb_seen
    ST.lower_bound = lower_bound
    if nreq > 0:
        use_align = bool(align and num_computed is not None and idx_mapping is not None and block_size)
        n = COUNTERS["commits"]
        vfirst = _env_int("GLM53_DEC_KDA_LAZY_VERIFY", 64)
        vevery = _env_int("GLM53_DEC_KDA_LAZY_VERIFY_EVERY", 1024)
        check = (not ST.repair) and ("frk" in _ORIG) and (n < vfirst or (vevery > 0 and n % vevery == 0))
        all_cols = bool(align_all or (align and not use_align))
        scalar = not isinstance(num_sampled, torch.Tensor)
        ns = ST.meta if scalar else num_sampled
        im = idx_mapping if use_align else ST.meta
        nc = num_computed if use_align else ST.meta
        kargs = dict(STATE_STRIDE=ST.slot_stride, NSEQ=ST.nseq, H=ST.H, HV=ST.H, K=ST.K, V=ST.V,
                     BK=K_["triton"].next_power_of_2(ST.K), BV=8, TMAX=ST.tmax, META_W=3 + ST.tmax,
                     SAMPLED_SCALAR=scalar, SAMPLED_VALUE=int(num_sampled) if scalar else 0,
                     ALIGN=use_align, ALIGN_ALL=all_cols, BLOCK_SIZE=int(block_size or 1), LOWER_BOUND=lower_bound,
                     num_warps=1, num_stages=3)
        pos = (ST.anchor, ST.state_off, ST.alog_tab, ST.gbias_tab, ST.sk, ST.sv, ST.sg, ST.sb, ST.meta,
               ns, im, nc, nreq)
        if check or ST.repair:
            T_all = torch.clamp(ST.meta[:nl, :nreq, 1].to(torch.int64), min=1)
            A_all = [_accepted(num_sampled, nreq, T_all[l]) for l in range(nl)]
            bias = (_align_bias(A_all[0], idx_mapping, num_computed, int(block_size), nreq) if use_align
                    else torch.full((nreq,), -1, dtype=torch.int64, device=ST.meta.device))
        if check:
            # per layer: production's kernel into a temp buffer vs the fast kernel in DRY mode (nothing written to the
            # cache yet); then the cache gets production's bytes for this step whatever the comparison says
            bad = 0
            live_states = live_layers = 0
            dry = torch.empty(nreq, ST.H, ST.V, ST.K, dtype=torch.float32, device=ST.meta.device)
            for l in range(nl):
                ref, _refb, flag, A, slots = _reference(l, nreq, A_all[l], bias, False, "temp")
                dry.zero_()
                K_["commit"][(triton_cdiv(ST.V, 8), nreq * ST.H, 1)](*pos, **{**kargs, "ALIGN_ALL": False},
                                                                       dry_out=dry, layer0=l, DRY=True)
                tgt = slots.gather(1, (A - 1)[:, None]).squeeze(1)
                live = flag & (tgt > 0)
                same = (ref.view(torch.int32) == dry.view(torch.int32)).flatten(1).all(1) | ~live
                bad += int((~same).sum())                           # one sync per layer on a checked step
                nlive = int(live.sum())
                live_states += nlive
                live_layers += nlive > 0
                COUNTERS["checked_states"] += nlive
                COUNTERS["checked_rows_gt1"] += int((live & (A > 1)).sum())
                del ref, _refb
            for l in range(nl):
                _reference(l, nreq, A_all[l], bias, all_cols, "slots")
            if live_states == 0:
                COUNTERS["checked_empty"] += 1          # nothing was pending: not a proof of anything
            else:
                COUNTERS["verified"] += 1
                if live_layers < nl:
                    _once("partial", logging.WARNING, "glm53_kda_lazy: only %d of %d registered KDA layers had a "
                          "pending lazy verify at checked commit %d (a layer took production's path in this step)",
                          live_layers, nl, n)
            if bad:
                COUNTERS["verify_mismatch"] += 1
                ST.repair = True
                _log.warning("glm53_kda_lazy: self-check: %d state(s) of the fast commit differ from production's "
                             "kernel at commit %d (this step was committed with production's kernel); every later "
                             "commit runs production's kernel (repair mode)", bad, n)
            elif live_states and COUNTERS["verified"] in (1, vfirst):
                _log.info("glm53_kda_lazy: self-check %d: fast commit == production's kernel byte for byte "
                          "(v2 per-layer keys: %d of %d layers live x %d sequences, %d states)",
                          COUNTERS["verified"], live_layers, nl, nreq, live_states)
        elif ST.repair:
            for l in range(nl):
                _reference(l, nreq, A_all[l], bias, all_cols, "slots")
        else:
            K_["commit"][(triton_cdiv(ST.V, 8), nreq * ST.H, nl)](*pos, **kargs, dry_out=ST.meta, layer0=0, DRY=False)
    ST.meta[:nl, :, 0].zero_()
    COUNTERS["commits"] += 1
    if COUNTERS["commits"] in (1, 10, 100, 1000, 10000, 100000):
        _log.info("glm53_kda_lazy: %d commits; counters %s%s", COUNTERS["commits"], dict(COUNTERS),
                  " (repair mode)" if ST.repair else "")
    ST.pending = False


# ---------------------------------------------------------------------------------------------------------------
# wiring

_ORIG: dict = {}


def _pure_spec_batch() -> bool | None:
    """True iff the forward context's KDA metadata is a pure spec-verify batch (no prefill / plain decode rows)."""
    try:
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        md = get_forward_context().attn_metadata
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(md, dict):
        return None
    for m in md.values():
        if isinstance(m, GDNAttentionMetadata):
            return bool(m.num_spec_decodes > 0 and m.num_prefills == 0 and m.num_decodes == 0
                        and (m.non_spec_token_indx is None or m.non_spec_token_indx.numel() == 0))
    return None


def fused_recurrent_kda_lazy(q, k, v, g, beta=None, scale=None, initial_state=None, inplace_final_state=True,
                             use_qk_l2norm_in_kernel=True, cu_seqlens=None, ssm_state_indices=None,
                             num_accepted_tokens=None, out=None, sigmoid_beta=False, a_log=None, g_bias=None,
                             compute_gate=False, lower_bound=-5.0, **kwargs):
    """Drop-in for the KDA module's fused_recurrent_kda: the spec-verify call goes lazy, everything else is
    production's function with exactly the arguments received."""
    orig = _ORIG["frk"]
    why = None
    slot = None
    if not ST.enabled:
        why = "off"
    elif kwargs:
        why = "kwargs"
    elif num_accepted_tokens is None or ssm_state_indices is None or cu_seqlens is None:
        why = "not spec"
    elif not (inplace_final_state and use_qk_l2norm_in_kernel and sigmoid_beta and compute_gate):
        why = "flags"
    elif a_log is None or g_bias is None or lower_bound is None or initial_state is None:
        why = "args"
    elif ssm_state_indices.ndim != 2 or ssm_state_indices.size(1) != ST.tmax:
        why = "indices"
    elif beta is None or beta.dim() != 3 or q.dim() != 4 or q.shape[0] != 1:
        why = "shape"
    elif tuple(k.shape[2:]) != (ST.H, ST.K) or v.shape[2] != ST.H or v.shape[3] != ST.V or q.shape[2] != ST.H:
        why = "heads"
    elif (k.dtype, v.dtype, g.dtype, beta.dtype) != ST.dtypes:
        why = "dtype"
    elif len(cu_seqlens) - 1 > ST.nseq:
        why = "nseq"
    elif any(_token_stride(t) is None for t in (q, k, v, beta)):
        why = "stride"
    elif any(_token_stride(t) % 16 or t.data_ptr() % 16 for t in (q, k, v)):
        # the address-alignment class the commit kernel was validated against production's kernel in (production's
        # merged-conv q/k/v slices: token stride 3*H*K, 16-byte aligned); another class can change the compiled
        # register layout of production's kernel and with it the rounding of its reductions
        why = "alignment class"
    elif _pure_spec_batch() is not True:
        why = "mixed batch"
    if why is None:
        slot = register(initial_state, a_log, g_bias)
        if slot is None:
            why = "register"
        else:
            src = ST.layer_src[slot]
            # the commit uses fp32 copies of this layer's a_log / g_bias taken at registration and one lower bound:
            # the same tensors (no device sync here: this also runs inside graph captures) and the same bound
            if src[0].data_ptr() != a_log.data_ptr() or src[1].data_ptr() != g_bias.data_ptr():
                why = "a_log / g_bias tensors changed"
            elif ST.lb_seen is not None and float(lower_bound) != ST.lb_seen:
                why = "lower bound differs between layers"
            else:
                ST.lb_seen = float(lower_bound)
    if why is not None:
        COUNTERS["prod_calls"] += 1
        if why not in ("off", "not spec"):
            _once(("prod", why), logging.INFO, "glm53_kda_lazy: production path for a KDA call (%s)", why)
            s = (ST.layers.get(layer_key(initial_state, a_log))
                 if (ST.alloc and initial_state is not None and a_log is not None) else None)
            if s is not None:
                ST.meta[s, :, 0].zero_()            # nothing of this layer may be committed from an older call
        return orig(q, k, v, g, beta=beta, scale=scale, initial_state=initial_state,
                    inplace_final_state=inplace_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices,
                    num_accepted_tokens=num_accepted_tokens, out=out, sigmoid_beta=sigmoid_beta, a_log=a_log,
                    g_bias=g_bias, compute_gate=compute_gate, lower_bound=lower_bound, **kwargs)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    COUNTERS["lazy_calls"] += 1
    ST.pending = True
    return lazy_verify(q, k, v, g, beta, scale, initial_state, cu_seqlens, ssm_state_indices, num_accepted_tokens,
                       out, a_log, g_bias, lower_bound, slot)


fused_recurrent_kda_lazy._glm53_kda_lazy = True


def _postprocess_state_wrapper(orig):
    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        try:
            if ST.enabled and ST.alloc and ST.layers:
                align = bool(getattr(self, "_align_mode", False))
                bs = None
                align_all = False
                if align:
                    ctx = getattr(self, "_mamba_ctx", None)
                    bs = getattr(ctx, "block_size", None) if ctx is not None else None
                    if bs is None:
                        spec = getattr(self, "_mamba_spec", None)
                        bs = getattr(spec, "block_size", None)
                    if bs is None or num_computed_tokens is None:
                        align_all = True            # cannot tell which column the copy reads: commit them all
                commit(num_sampled, idx_mapping, idx_mapping.shape[0], num_computed=num_computed_tokens,
                       block_size=bs, align=align, align_all=align_all)
        except Exception as exc:  # noqa: BLE001
            # a commit that did not run leaves the states of this step unwritten: never continue lazily after it
            ST.enabled = False
            ST.disabled_reason = f"commit failed: {exc!r}"
            _log.error("glm53_kda_lazy: commit failed (%r); lazy path switched OFF for this process", exc)
            raise
        return orig(self, idx_mapping, num_sampled, num_computed_tokens)

    postprocess_state._glm53_kda_lazy_orig = orig
    return postprocess_state


def post_load(model) -> dict:
    """Wire after weight loading: the KDA module global + MambaHybridModelState.postprocess_state; allocate scratch."""
    import sys
    out = {"layers": 0, "why": None}
    kda_mods = [m for m in model.modules() if type(m).__name__ in ("KimiDeltaAttention", "Glm5NextKDA")
                or (hasattr(m, "kda_safe_gate") and hasattr(m, "A_log") and hasattr(m, "dt_bias"))]
    if not kda_mods:
        out["why"] = "no KDA modules"
        return out
    m0 = kda_mods[0]
    modname = type(m0).__module__
    mod = sys.modules.get(modname)
    cur = getattr(mod, "fused_recurrent_kda", None) if mod is not None else None
    if cur is None:
        out["why"] = f"{modname}.fused_recurrent_kda missing"
        return out
    if not getattr(m0, "kda_safe_gate", False):
        out["why"] = "KDA gate is not the bounded (safe) variant"
        return out
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        nseq = int(cfg.scheduler_config.max_num_seqs)
        nspec = int(getattr(m0, "num_spec", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"config: {exc!r}"
        return out
    if nspec <= 0:
        out["why"] = "no speculative decoding"
        return out
    tp = int(getattr(m0, "tp_size", 1) or 1)
    H = int(m0.num_heads) // tp
    Kd = int(m0.head_dim)
    dev = m0.A_log.device
    dt = next((p.dtype for p in m0.parameters() if p.dtype in (torch.bfloat16, torch.float16)), torch.bfloat16)
    allocate(len(kda_mods), nseq, nspec + 1, H, Kd, Kd, dev, (dt, dt, dt, dt))
    if not getattr(cur, "_glm53_kda_lazy", False):
        _ORIG["frk"] = cur
        mod.fused_recurrent_kda = fused_recurrent_kda_lazy
    import vllm.v1.worker.gpu.model_states.mamba_hybrid as MH
    cls = MH.MambaHybridModelState
    if not hasattr(cls.postprocess_state, "_glm53_kda_lazy_orig"):
        cls.postprocess_state = _postprocess_state_wrapper(cls.postprocess_state)
    ST.enabled = True
    out["layers"] = len(kda_mods)
    import atexit
    atexit.register(lambda: _log.info("glm53_kda_lazy: exit summary %s", summary()))
    _log.info("glm53_kda_lazy on %s: %d KDA layers, %d seqs x %d rows, %d heads x %d; scratch %.1f MiB; "
              "commit in MambaHybridModelState.postprocess_state; v2 per-layer keys", type(model).__name__, len(kda_mods), nseq,
              nspec + 1, H, Kd, scratch_bytes() / 2**20)
    return out


_HOOKED = {"loader": False}


def _hook_loader() -> None:
    if _HOOKED["loader"]:
        return
    import vllm.model_executor.model_loader.base_loader as BL
    orig = BL.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        try:
            r = post_load(model)
            if r["why"]:
                _log.info("glm53_kda_lazy: not wired on %s (%s); production's per-row stores stay",
                          type(model).__name__, r["why"])
        except Exception as exc:  # noqa: BLE001
            ST.enabled = False
            _log.warning("glm53_kda_lazy: wiring failed (%r); production's per-row stores stay", exc)

    process_weights_after_loading._glm53_kda_lazy_orig = orig
    BL.process_weights_after_loading = process_weights_after_loading
    _HOOKED["loader"] = True


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_DEC_KDA_LAZY is on."""
    on = env_enabled()
    _log.info("glm53_kda_lazy plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              "installing" if on else "off, production kernels unchanged")
    if not on:
        return
    try:
        _hook_loader()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_kda_lazy not installed (production unchanged): %r", exc)


def summary() -> dict:
    return {"counters": dict(COUNTERS), "enabled": ST.enabled, "layers": len(ST.layers),
            "scratch_mib": scratch_bytes() / 2**20, "disabled_reason": ST.disabled_reason}
