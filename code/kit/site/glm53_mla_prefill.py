"""GLM53_MLA_PREFILL: exact sparse-MLA prefill kernel for production's FLASHINFER_MLA_SPARSE_SM90 backend
(docs/MLA_PREFILL.md). Kernel: kernels/mla_prefill/mla_prefill.cu (AOT extension glm53_mla_prefill_ext).

Inert unless GLM53_MLA_PREFILL is set. The plugin hook (plugin_install, called from integrate.plugin_register in every
vLLM process) wraps FlashInferMLASparseSM90Impl.forward_mqa; the wrapper sends a call to the new kernel only when it
is an eager (not CUDA-graph-captured) forward with at least GLM53_MLA_PREFILL_MIN_TOKENS query rows (default 256),
32 heads, kv_lora_rank 512, no rope and an fp8 KV cache; everything else (in particular every decode step, which runs
inside captured CUDA graphs) goes to production's FA2 wrapper exactly as before. A mixed step (decode rows + a prefill
chunk, >= MIN tokens) runs all its rows on the new kernel; GLM53_MLA_PREFILL_MIXED=0 keeps such steps on FA2.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_log = logging.getLogger("vllm.glm53_mla_prefill")
HERE = Path(__file__).resolve().parent
ENV = "GLM53_MLA_PREFILL"
ENV_MIN = "GLM53_MLA_PREFILL_MIN_TOKENS"
_TRUE = frozenset({"1", "on", "true", "yes", "exact"})


class _State:
    ext = None
    calls = 0
    fallbacks: dict = {}
    min_tokens = 256
    mixed = True
    variant = 4
    installed = False
    # Write the FA2 wrapper's process-wide kv_indices exactly as production's forward_mqa does (see make_forward_mqa).
    # False only in tests/handoff (it reproduces the deploy-r16 decode defect for the regression check); never in
    # production.
    write_kv_indices = True
    kv_indices_writes = 0
    # opt-dense: one Triton pass writes production's kv_indices bytes AND the valid counts; the kernel reads its slots
    # from that kv_indices view (GLM53_MLA_PREFILL_FUSED_INDEX, default 1; 0 = the triton_convert + clamp + copy chain)
    fused_index = True
    fused_index_calls = 0
    # opt-kdamhc: rows of the FA2 wrapper's kv_indices this path writes back (None = every row, the r16 behaviour).
    # Only an FA2 call can read rows it did not write itself, and it reads at most row n (the first ctx % 4 < 4
    # entries) where n = its own row count. FA2 serves only calls with < min_tokens rows (decode steps, CUDA-graph
    # capture sizes) while this path serves every eligible call >= min_tokens, so rows [0, max(min_tokens, 1024) + 1)
    # hold every entry any FA2 call can read beyond its own rows; they stay byte-identical to production's. The rest of
    # a 13,824-row prefill (113 MB clamp + 113 MB copy, ~1.9 ms per MLA layer) is never read. GLM53_MLA_PREFILL_MIXED=0
    # (FA2 also serves mixed steps of any size) and GLM53_MLA_PREFILL_KV_ROWS=all keep the full write-back.
    kv_rows = None


STATE = _State()


def load_ext(jit: bool | None = None):
    if STATE.ext is not None:
        return STATE.ext
    try:
        import glm53_mla_prefill_ext as m
    except ImportError as err:
        if jit is None:
            jit = os.environ.get("TF_EXL3_JIT", "0").strip().lower() in _TRUE
        if not jit:
            raise ImportError(f"glm53_mla_prefill_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load
        from tf_exl3_moe import _cuda_include_shim
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = _cuda_include_shim()
        import hashlib
        kdir = HERE / "kernels" / "mla_prefill"
        tag = hashlib.sha256(b"".join((kdir / f).read_bytes() for f in ("mla_prefill.cu", "mla_prefill_kernel.cuh")))
        m = load(name=f"glm53_mla_prefill_jit_{tag.hexdigest()[:10]}", sources=[str(kdir / "mla_prefill.cu")],
                 extra_cuda_cflags=["-O3", "-use_fast_math", *inc], extra_cflags=["-O3", *inc], verbose=False)
    STATE.ext = m
    return m


_FK = {}


def _fused_index_kernel():
    """Triton: production's forward_mqa index chain in ONE pass.

    production (flashinfer_mla_sparse_sm90.py forward_mqa + this wrapper's old path):
      topk_slots, valid = triton_convert_req_index_to_global_index(..., return_valid_counts=True)
          # full_like(-1) [T, W] + _convert_req_index_to_global_index_kernel(SINGLE_TILE, COMPACT_TO_FRONT, DCP 1)
      kv_indices[:T*W].copy_(topk_slots.reshape(-1).clamp_(min=0).to(int32))
    i.e. ~4 passes over a [T, W] int32 tensor (113 MB each at T = 13824) plus the -1 fill. This kernel writes the
    clamped compacted row (valid prefix, 0 tail) straight into kv_indices and the valid count into valid: the same
    values (the quickwins mla_index kernel's arithmetic, + the count the SINGLE_TILE path stores)."""
    if _FK:
        return _FK["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def mla_fused_index_kernel(req_id_ptr, block_table_ptr, token_indices_ptr, out_ptr, valid_ptr,
                               max_num_blocks_per_req, bt_stride0, bt_stride1, ti_stride0, ti_stride1,
                               BLOCK_SIZE: tl.constexpr, W: tl.constexpr):
        token_id = tl.program_id(0)
        indice_id = tl.arange(0, W)
        req = tl.load(req_id_ptr + token_id)
        tok = tl.load(token_indices_ptr + token_id.to(tl.int64) * ti_stride0 + indice_id * ti_stride1)
        is_invalid_tok = tok < 0
        block_id = tok // BLOCK_SIZE
        inblock_off = tok % BLOCK_SIZE
        valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
        bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
        is_invalid_tok |= ~valid_block
        base = tl.load(bt_ptr, mask=valid_block, other=0)
        out_val = base * BLOCK_SIZE + inblock_off
        out_val = tl.where(is_invalid_tok, -1, out_val)
        out_val = tl.maximum(out_val, 0)                     # clamp_(min=0) of the valid prefix (a no-op there)
        is_valid = (~is_invalid_tok).to(tl.int32)
        local_offset = tl.cumsum(is_valid) - is_valid
        cnt = tl.sum(is_valid)
        row = out_ptr + token_id.to(tl.int64) * W
        tl.store(row + local_offset, out_val, mask=is_valid == 1)
        tl.store(row + indice_id, tl.zeros_like(out_val), mask=indice_id >= cnt)   # the -1 tail, clamped
        tl.store(valid_ptr + token_id, cnt)

    _FK["k"] = mla_fused_index_kernel
    _FK["triton"] = triton
    return mla_fused_index_kernel


def fused_index(req_id, block_table, topk_indices, block_size: int, kv_indices):
    """(slots [T, W] = a view of kv_indices, valid int32 [T]) or None when the inputs are outside what it handles."""
    T, W = int(topk_indices.shape[0]), int(topk_indices.shape[1])
    if not (req_id.dtype == block_table.dtype == topk_indices.dtype == kv_indices.dtype == torch.int32
            and W > 0 and (W & (W - 1)) == 0 and W <= 4096 and req_id.shape[0] >= T and kv_indices.is_contiguous()
            and kv_indices.numel() >= T * W and block_table.dim() == 2 and topk_indices.dim() == 2):
        return None
    k = _fused_index_kernel()
    slots = kv_indices[: T * W].view(T, W)
    valid = torch.empty(T, dtype=torch.int32, device=topk_indices.device)
    rid = req_id[:T]
    if not rid.is_contiguous():
        rid = rid.contiguous()
    k[(T,)](rid, block_table, topk_indices, slots, valid, int(block_table.shape[1]), *block_table.stride(),
            *topk_indices.stride(), BLOCK_SIZE=int(block_size), W=W, num_warps=8)
    return slots, valid


# Kernel variants (kernels/mla_prefill/): 4 = persistent v3 layout (default, fastest measured at 1.8k-100k context),
# 3 = v3 one CTA per token, 2 = the first exact kernel (2 head groups x 4 dim quarters). All give FA2-level error.
DEFAULT_VARIANT = 4


def run(ext, q, kv_u8, slots, valid, out, sm_scale: float, kv_scale: float, variant: int | None = None) -> None:
    """out[T, 32, 512] = sparse attention of q[T, 32, 512] over kv rows slots[t, :valid[t]] (fp8 e4m3 [N, 512])."""
    ext.run(q, kv_u8.reshape(-1, 512), slots, valid, out, float(sm_scale), float(kv_scale),
            int(STATE.variant if variant is None else variant))


# --------------------------------------------------------------------------------------------------- plugin hook
TARGET_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"


def env_mode() -> str:
    """'' (inert) or 'exact'. Unknown values are refused (logged, inert)."""
    v = os.environ.get(ENV, "").strip().lower()
    if not v or v in ("0", "off", "false", "no"):
        return ""
    if v in _TRUE:
        return "exact"
    _log.warning("%s=%r not understood (use 1/exact); sparse-MLA prefill stays on production FA2", ENV, v)
    return ""


def _ineligible(impl, q, kv_cache, num_tokens: int) -> str | None:
    """None if the new kernel can serve this forward_mqa call, else the reason."""
    if num_tokens < STATE.min_tokens:
        return "small"
    if torch.cuda.is_current_stream_capturing():
        return "capturing"
    if not isinstance(q, tuple):
        return "q not split"
    q_nope, q_pe = q
    if getattr(impl, "num_heads", None) != 32 or getattr(impl, "kv_lora_rank", None) != 512:
        return "geometry"
    if getattr(impl, "qk_rope_head_dim", None) != 0 or (q_pe is not None and q_pe.numel() != 0):
        return "rope"
    if not getattr(impl, "use_fp8_kv_cache", False):
        return "kv dtype"
    if q_nope.dtype != torch.bfloat16 or q_nope.dim() != 3 or q_nope.shape[1:] != (32, 512) or q_nope.stride(2) != 1:
        return "q layout"
    if q_nope.stride(0) % 8 or q_nope.stride(1) % 8 or q_nope.data_ptr() % 16:
        return "q alignment"
    if kv_cache.element_size() != 1 or kv_cache.shape[-1] != 512 or not kv_cache.is_contiguous():
        return "kv layout"
    return None


def make_forward_mqa(orig):
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        num_tokens = q[1].shape[0] if isinstance(q, tuple) else q.shape[0]
        why = _ineligible(self, q, kv_c_and_k_pe_cache, num_tokens)
        if why is None and not STATE.mixed and int(getattr(attn_metadata, "num_decode_tokens", 0) or 0) > 0:
            why = "mixed batch"
        if why is not None:
            n = STATE.fallbacks.get(why, 0)
            STATE.fallbacks[why] = n + 1
            if n == 0 and why not in ("small", "capturing", "mixed batch"):
                _log.warning("glm53_mla_prefill: a %d-token call went to production FA2 (%s); later ones are only "
                             "counted", num_tokens, why)
            return orig(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        q_nope = q[0]
        topk_indices = self.topk_indices_buffer[:num_tokens]
        out = torch.empty_like(q_nope)          # same (permuted) layout FA2 returns; _v_up_proj transposes it back
        k_scale = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
        sm = sys.modules.get(TARGET_MODULE) if STATE.write_kv_indices else None
        st = getattr(sm, "_SM90_STATE", None) if sm is not None else None
        fused = None
        fused_buf = None
        if STATE.fused_index and st is not None:
            if STATE.kv_rows is None:
                # kv_indices gets exactly production's bytes (see below) BEFORE the kernel, which reads its slots
                # from it (zero copy: the slots ARE the FA2 wrapper's kv_indices)
                fused = fused_index(attn_metadata.req_id_per_token, attn_metadata.block_table, topk_indices,
                                    int(attn_metadata.block_size), st.kv_indices)
            else:
                # opt-kdamhc KV_ROWS with the opt-dense fused pass: the fused kernel READS its slots from the
                # buffer it writes, so a limited write-back goes through a private [T, W] buffer (the same bytes
                # production writes) and only the rows an FA2 call can read are copied back after the launch
                # (stream order: the kernel has read them).
                fused_buf = torch.empty(topk_indices.numel(), dtype=torch.int32, device=topk_indices.device)
                fused = fused_index(attn_metadata.req_id_per_token, attn_metadata.block_table, topk_indices,
                                    int(attn_metadata.block_size), fused_buf)
        if fused is not None:
            slots, valid = fused
            STATE.ext.run(q_nope, kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1, 512), slots, valid, out,
                          float(self.scale), k_scale, STATE.variant)
            if fused_buf is not None:
                width = topk_indices.shape[1]
                rows = num_tokens if STATE.kv_rows is None else min(num_tokens, STATE.kv_rows)
                st.kv_indices[: rows * width].copy_(fused_buf[: rows * width])
            STATE.kv_indices_writes += 1
            STATE.fused_index_calls += 1
            STATE.calls += 1
            if STATE.calls in (1, 10, 1000, 100000):
                _log.info("glm53_mla_prefill: %d calls on the exact kernel (last %d tokens, fused index pass: %d); "
                          "production FA2 kept for %s", STATE.calls, num_tokens, STATE.fused_index_calls,
                          dict(STATE.fallbacks))
            return out, None
        from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
        # Same conversion as production's forward_mqa (compacted prefix + exact valid counts); the kernel reads the
        # valid counts on the device.
        topk_slots, valid = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        if valid.dtype != torch.int32:
            valid = valid.to(torch.int32)
        STATE.ext.run(q_nope, kv_c_and_k_pe_cache.view(torch.uint8).reshape(-1, 512), topk_slots, valid, out,
                      float(self.scale), k_scale, STATE.variant)
        # The kernel does not need the FA2 wrapper's kv_indices, but DECODE reads what prefill leaves there: production
        # plans FA2 with kv_len = 2048 + ctx % 4 on a 2048-wide row (docs/MLA_PREFILL.md section 6), so every decode row
        # with ctx >= 2048 reads ctx % 4 entries past its row, and the LAST row of a decode step reads them past the
        # rows that step wrote, i.e. whatever the previous forward_mqa with more rows left in this process-wide buffer.
        # With production FA2 that is this request's prefill; skipping the copy here left an older call's slot ids
        # there (another request's keys), which decode then attended (tests/handoff, docs/HANDOFF.md). So write the
        # same bytes production writes, after the kernel launch (stream order: the kernel has read topk_slots' prefix).
        if st is not None:
            width = topk_slots.shape[1]
            rows = num_tokens if STATE.kv_rows is None else min(num_tokens, STATE.kv_rows)
            st.kv_indices[: rows * width].copy_(topk_slots[:rows].reshape(-1).clamp(min=0).to(torch.int32))
            STATE.kv_indices_writes += 1
        STATE.calls += 1
        if STATE.calls in (1, 10, 1000, 100000):
            _log.info("glm53_mla_prefill: %d calls on the exact kernel (last %d tokens); production FA2 kept for %s",
                      STATE.calls, num_tokens, dict(STATE.fallbacks))
        return out, None

    forward_mqa.__glm53_mla_prefill__ = True
    forward_mqa.__doc__ = orig.__doc__
    return forward_mqa


def install(module=None) -> bool:
    """Wrap FlashInferMLASparseSM90Impl.forward_mqa. Returns True when installed."""
    if STATE.installed:
        return True
    if module is None:
        import importlib
        module = importlib.import_module(TARGET_MODULE)
    cls = getattr(module, "FlashInferMLASparseSM90Impl", None)
    if cls is None or not hasattr(cls, "forward_mqa"):
        _log.warning("glm53_mla_prefill: %s has no FlashInferMLASparseSM90Impl.forward_mqa; not installed",
                     TARGET_MODULE)
        return False
    orig = cls.forward_mqa
    if getattr(orig, "__glm53_mla_prefill__", False):
        STATE.installed = True
        return True
    load_ext()
    try:
        STATE.min_tokens = max(1, int(os.environ.get(ENV_MIN, "") or 256))
    except ValueError:
        STATE.min_tokens = 256
    STATE.mixed = os.environ.get("GLM53_MLA_PREFILL_MIXED", "1").strip().lower() not in ("0", "off", "false", "no")
    STATE.fused_index = os.environ.get("GLM53_MLA_PREFILL_FUSED_INDEX", "1").strip().lower() not in (
        "0", "off", "false", "no")
    try:
        v = int(os.environ.get("GLM53_MLA_PREFILL_VARIANT", "") or DEFAULT_VARIANT)
        STATE.variant = v if v in (2, 3, 4) else DEFAULT_VARIANT
    except ValueError:
        STATE.variant = DEFAULT_VARIANT
    kvr = os.environ.get("GLM53_MLA_PREFILL_KV_ROWS", "").strip().lower()
    if kvr == "all" or not STATE.mixed:
        STATE.kv_rows = None
    else:
        try:
            STATE.kv_rows = max(int(kvr), STATE.min_tokens + 1) if kvr else max(STATE.min_tokens, 1024) + 1
        except ValueError:
            STATE.kv_rows = None
    cls.forward_mqa = make_forward_mqa(orig)
    STATE.installed = True
    _log.info("glm53_mla_prefill installed: FlashInferMLASparseSM90Impl.forward_mqa -> exact sparse-MLA prefill "
              "kernel (ext v%s, variant %d) for eager calls with >= %d tokens%s; CUDA-graph decode stays on FA2",
              getattr(STATE.ext, "VERSION", "?"), STATE.variant, STATE.min_tokens,
              "" if STATE.mixed else " and no decode rows")
    _log.info("glm53_mla_prefill: FA2 kv_indices write-back %s", "every row" if STATE.kv_rows is None else
              f"first {STATE.kv_rows} rows (every row an FA2 call can read beyond its own)")
    return True


def plugin_install() -> None:
    """integrate.plugin_register hook: inert unless GLM53_MLA_PREFILL is set."""
    mode = env_mode()
    if not mode:
        return
    try:
        install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_mla_prefill not installed (production FA2 unchanged): %r", exc)
