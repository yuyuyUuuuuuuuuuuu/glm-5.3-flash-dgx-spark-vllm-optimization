"""[glm53-kda-flashkda] FlashKDA 17a037d chunked prefill for GLM-5.3-Flash's KDA layers.

Installed into site-packages by ``overlay/patch_flashkda.py`` (and only then:
the file does not exist in a stock or switch-unset tree). The patched
``vllm/models/glm5next/nvidia/kda.py`` dispatches the chunked prefill
(``chunk_kda_with_fused_gate``) here when ``GLM53_KDA_FLASHKDA=1``.

Contract (identical inputs/outputs to the Triton chain it replaces):

* inputs: the same tensors production passes -- ``q/k/v`` ``[1, T, H, D]`` bf16
  (column slices of the merged short-conv output), ``g`` the RAW gate logits
  ``g1`` ``[1, T, H, D]``, ``beta`` the RAW bf16 beta logits ``[1, T, H]``
  (the Triton chain is handed the pre-sigmoided fp32 beta and sigmoids nothing;
  FlashKDA applies ``sigmoid(beta)`` in-kernel, exactly like the decode path's
  ``fused_recurrent_kda(..., sigmoid_beta=True)``), ``initial_state``
  ``[N, H, D, D]`` fp32 (``gather_initial_states``), ``cu_seqlens``
  ``[N + 1]`` int32 (``non_spec_query_start_loc``).
* in-kernel: q/k l2-normalisation (``use_qk_l2norm_in_kernel=True``) and the
  bounded gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
  (``safe_gate=True``) -- the same arithmetic ``chunk_kda_with_fused_gate``
  performs, computed by a different tile algorithm (FlashKDA's fp32 recurrent
  state, vLLM #58846). Conv1d, gating layout, o_norm/o_proj and everything
  downstream stay in production's hands.
* outputs: ``(out, final_state)`` with ``out`` ``[1, T, H, D]`` model dtype and
  ``final_state`` ``[N, H, D, D]`` fp32 -- the tensor ``scatter_states`` copies
  into the recurrent-state cache, i.e. what decode later reads. Unlike the
  Triton chain this output does NOT alias ``v`` (the Triton chunk kernel writes
  its result in place into ``v``).

Buffers come from the rank's ``current_workspace_manager()`` (graph-safe, like
every other production kernel workspace) and are sized once for
``max_num_batched_tokens`` / ``max_num_seqs``.

Decode is untouched: the recurrent path keeps ``fused_recurrent_kda``.
"""

from __future__ import annotations

import os

ENV = "GLM53_KDA_FLASHKDA"
TAG = "[glm53-kda-flashkda]"
EXT_NAME = "_flashkda_fp32_C"
FLASHKDA_REF = "17a037d98da546deb4591e967cf961a43c034d8b"  # fp32 recurrent state (vllm#58846)

# fkda2: the extension this wrapper accepts. FlashKDA 17a037d + the precision
# edits of tests/fkda2/patch_flashkda_precision.py (FKDA2_FP32_DECAY: decayed
# q/k rounded to bf16 once; FKDA2_FP32_U: u = (v - kS)*beta in fp32;
# FKDA2_FP32_OUT: out = qS + Mqk U in one fp32 accumulator), built by
# tests/fkda2/build_variant.sh v_fix. Same kernels/tiling/speed; output error
# vs an fp64 reference 0.68x the stock build's (0.84x the Triton chain's) and
# final-state error 0.69x (1.02x Triton's) on real mini-model inputs. A
# different .so (e.g. the stock-precision fkda build c286213f...) is refused at
# boot, so the boot line's sha names exactly the kernel that runs.
# fkda3: + tests/fkda3/patch_flashkda_fkda3.py (FKDA3_DECAY_NOFTZ: the decay
# products keep subnormals, which the fkda2 fp32 path flushed in tiles whose
# gates sit at the lower bound; FKDA3_FP32_STATE_IO: the recurrent state is
# loaded from and stored to the fp32 cache WITHOUT a bf16 rounding at the call
# boundary, as the Triton chain does), built by tests/fkda3/build_variant.sh
# f3_prod _flashkda_fp32_C 1 (SASS identical to the f3_all build the fkda3 rigs
# measured). Same tiling/speed as fkda2.
EXT_SHA256 = "d98adc4aec0046a294caebac32df5f974c07e239a9d29bac64f37233024ce394"
EXT_BUILD = "fkda3 precision build (fp32 decay/u/out, no-ftz decay, fp32 state io)"

# Harness handles (tests/handoff/run_engine.py ProdMode): "off" True forces the
# stock Triton chain for the control run, like every other feature's off
# switch; "allow_any_ext" True lets an A/B rig load another FlashKDA build;
# "direct_out" False (fkda3) makes chunk_prefill ignore the caller's out= and
# use the workspace buffer (the fkda2 behaviour) for an A/B of the merge copy.
STATE = {"off": False, "calls": 0, "tokens": 0, "logged": False, "allow_any_ext": False,
         "direct_out": True, "direct_calls": 0}

# Read once at import: only a patched tree (overlay/patch_flashkda.py installs
# this file) ever gets here, and patch_tf_bundle.py runs it only for =1; a
# value that is not exactly "1" makes the kda.py dispatch stay on the Triton
# chain (the patch itself SystemExits on anything but 0/1).
ENV_ENABLED = os.environ.get(ENV, "").strip() == "1"

# fkda3 harness-only knob (tests/handoff A/B of the direct output; production's start.sh does not forward it, so it is
# always unset there): "0" makes chunk_prefill ignore the caller's out= (the fkda2 workspace + merge-copy behaviour).
if os.environ.get("GLM53_FKDA3_DIRECT_OUT", "").strip() == "0":
    STATE["direct_out"] = False

_cfg: dict = {}


def _log_once(msg: str) -> None:
    if STATE["logged"]:
        return
    STATE["logged"] = True
    print(f"{TAG} {msg}", flush=True)


def configure(layer, vllm_config) -> None:
    """Resolve the extension ABI and size the workspace buffers (once per rank,
    from the layer's __init__, where vllm_config is in scope)."""
    import torch

    import _flashkda_fp32_C  # noqa: F401  registers torch.ops._flashkda_fp32_C.*

    if _cfg:
        return
    import hashlib

    with open(_flashkda_fp32_C.__file__, "rb") as fh:
        ext_sha = hashlib.sha256(fh.read()).hexdigest()
    if ext_sha != EXT_SHA256 and not STATE["allow_any_ext"]:
        raise RuntimeError(
            f"{TAG} {_flashkda_fp32_C.__file__} sha256 {ext_sha[:16]} is not the {EXT_BUILD} "
            f"{EXT_SHA256[:16]} this wrapper was validated with"
        )
    schema = torch.ops._flashkda_fp32_C.fwd.default._schema
    expected = [
        "q", "k", "v", "g", "beta", "scale", "out", "workspace", "A_log", "dt_bias",
        "lower_bound", "initial_state", "final_state", "cu_seqlens",
        "checkpoint_state", "checkpoint_offsets",
    ]
    got = [a.name for a in schema.arguments]
    if got != expected:
        raise RuntimeError(
            f"{TAG} FlashKDA ABI mismatch (want the 16-arg fp32-state fwd, {FLASHKDA_REF[:7]}): {schema}"
        )
    if layer.kda_safe_gate is not True:
        raise RuntimeError(f"{TAG} requires the bounded (safe_gate) KDA gate; got {layer.kda_safe_gate}")
    if layer.get_state_dtype()[1] is not torch.float32:
        raise RuntimeError(f"{TAG} requires the fp32 recurrent-state cache; got {layer.get_state_dtype()}")
    sc = vllm_config.scheduler_config
    max_tokens = int(sc.max_num_batched_tokens)
    max_seqs = int(sc.max_num_seqs)
    H, D = layer.local_num_heads, layer.head_dim
    ws = int(torch.ops._flashkda_fp32_C.get_workspace_size(max_tokens, H, max_seqs))
    dtype = vllm_config.model_config.dtype
    _cfg.update(
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        specs=(
            ((max_seqs, H, D, D), torch.float32),        # final recurrent state
            ((ws,), torch.uint8),                        # FlashKDA scratch
            ((1, max_tokens, H, D), dtype),              # attention output
        ),
        heads=H,
        head_dim=D,
        lower_bound=float(layer.kda_lower_bound),
    )
    # Reserve the buffers HERE (GPUWorker.init_device runs init_workspace_manager
    # before the model is constructed, and lock_workspace() only happens after
    # profiling/capture): a shortfall fails at boot with this tag instead of at
    # the first prefill, and the memory is on the books before vLLM measures the
    # free memory for the KV pool. get_simultaneous after the lock with the same
    # specs cannot grow the workspace, so the locked path cannot assert.
    from vllm.v1.worker.workspace import current_workspace_manager

    try:
        reserved = current_workspace_manager().get_simultaneous(*_cfg["specs"])
    except AssertionError as exc:                    # should not happen pre-lock
        raise RuntimeError(f"{TAG} reserving the FlashKDA buffers at boot failed: {exc}") from exc
    total = sum(t.numel() * t.element_size() for t in reserved)
    _log_once(
        f"kda.py: chunked prefill -> FlashKDA {FLASHKDA_REF[:7]} ({EXT_NAME}, fp32 recurrent state), "
        f"{ENV}=1; buffers RESERVED at boot: {total / 2**30:.2f} GiB for max_num_batched_tokens={max_tokens} "
        f"max_num_seqs={max_seqs} H={H} D={D} (decode path untouched); extension sha256 {ext_sha[:16]}"
        f"{' = ' + EXT_BUILD if ext_sha == EXT_SHA256 else ' (NOT the validated build: harness override)'}"
    )


def enabled_for(layer) -> bool:
    """False = run production's Triton chain. configure() has run when this
    would otherwise be True (the patched kda.py calls configure() in __init__)."""
    if STATE["off"] or not _cfg:
        return False
    return True


def chunk_prefill(layer, q, k, v, g, beta, initial_state, cu_seqlens, out=None):
    """FlashKDA chunked prefill with production's inputs/outputs contract.

    fkda3: ``out`` (optional) is the layer output's first T rows (``core_attn_out[:, :num_actual_tokens]``, passed
    by the patched kda.py when the step has no spec tokens). When it is exactly a contiguous ``[1, T, H, D]`` bf16
    CUDA tensor FlashKDA writes there directly and kda.py's merge copy becomes a same-storage no-op (bit-identical
    result, one T x H x D read+write less per KDA layer); anything else (or STATE["direct_out"] False) uses the
    workspace buffer exactly as before."""
    import torch
    from vllm.v1.worker.workspace import current_workspace_manager

    if not _cfg:
        raise RuntimeError(f"{TAG} chunk_prefill before configure()")
    T = q.shape[1]
    N = cu_seqlens.numel() - 1
    if STATE["off"]:
        raise RuntimeError(f"{TAG} disabled mid-flight (control run); kda.py must take the Triton branch")
    if q.ndim != 4 or q.shape[0] != 1 or T > _cfg["max_tokens"]:
        raise RuntimeError(f"{TAG} token capacity exceeded: {q.shape} vs {_cfg['max_tokens']}")
    if N < 1 or initial_state.shape[0] != N:
        raise RuntimeError(f"{TAG} state/cu_seqlens mismatch: state {tuple(initial_state.shape)}, cu {N + 1} rows")
    if initial_state.dtype is not torch.float32:
        raise RuntimeError(f"{TAG} initial_state must be fp32 (kda recurrent state), got {initial_state.dtype}")
    if cu_seqlens.dtype is not torch.int32:
        raise RuntimeError(f"{TAG} cu_seqlens must be int32 (non_spec_query_start_loc), got {cu_seqlens.dtype}")

    final_state_ws, workspace, out_ws = current_workspace_manager().get_simultaneous(*_cfg["specs"])
    final_state = final_state_ws[:N]
    if (
        out is not None
        and STATE["direct_out"]
        and out.dtype is torch.bfloat16
        and tuple(out.shape) == (1, T, _cfg["heads"], _cfg["head_dim"])
        and out.is_contiguous()
        and out.device == q.device
        and out.data_ptr() % 16 == 0
    ):
        STATE["direct_calls"] += 1
    else:
        out = out_ws[:, :T]
    # FlashKDA reads q/k/v/g with dense strides (beta may be row-strided): the
    # Triton chain pays the same four .contiguous() copies before its kernels.
    torch.ops._flashkda_fp32_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        _cfg["head_dim"] ** -0.5,
        out,
        workspace,
        layer.A_log.reshape(-1).contiguous(),
        layer.dt_bias.reshape(-1, _cfg["head_dim"]).contiguous(),
        _cfg["lower_bound"],
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
        None,  # checkpoint_state
        None,  # checkpoint_offsets
    )
    STATE["calls"] += 1
    STATE["tokens"] += int(T)
    return out, final_state
