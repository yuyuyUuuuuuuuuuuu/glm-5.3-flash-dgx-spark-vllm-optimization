"""GLM53_BF16_GEMV: serve decode's small-M dense GEMMs with kernels/gemv_bf16.cu (docs/BF16_GEMV.md).

Enable: GLM53_BF16_GEMV in {1, on, true, yes} (read once, when the vLLM plugin loads; unset = nothing happens:
no op is registered, no class or module is touched). Revert: unset it and restart vLLM.
Sub-switches (only read when enabled):
  GLM53_BF16_GEMV_KINDS          comma list, default all: router, idx_wk, idx_head, idx_kpool, draft_conv
  GLM53_BF16_GEMV_DEDUP_ROUTER   1 (default) / 0: drop the MoE runner's second, identical router GEMM

What is served (each module self-tested on its real weight at load; a module that fails keeps production's path):
  router      GateLinear (mlp.gate, bf16 [288, 4096], fp32 logits) of every MoE layer. Production (SM 12.x has none
              of GateLinear's specialized tiers) runs F.linear -> bf16 -> .to(float32); the kernel writes the same
              bf16-rounded value as fp32 (out_mode 2), one kernel instead of GEMM + splitKreduce + cast.
  idx_wk      Indexer.wk_weights_proj (bf16 [160, 4096], bf16 out), through its UnquantizedLinearMethod.
  idx_head    Indexer head gate: production's torch.mm(x.float(), W32) (fp32 on purpose) -> gemm_f32 (IEEE fp32 FMA).
  idx_kpool   Indexer kpool gate F.linear(x, index_kpool_compress_gate) (bf16 [128, 4096]).
  draft_conv  DFlash2 attention_conv / mlp_conv kernel_projection (bf16 [1024, 4096]).
  The kernel is used only at the M buckets where it measured faster (glm53_bf16_gemv.PLANS, F32_MAX_M); every other
  call runs production's exact op (inside the same custom op), e.g. prefill, or M = 49..64 for the router.
Router dedup: Glm5NextMoE.forward computes router_logits = self.gate(x) and passes them to the MoE runner, which
  holds the same gate and computes gate(x) again (moe_runner.py _forward_impl). With runner.gate = None the runner
  uses the logits it was given: same input, same deterministic kernel -> bitwise the same routing, one GEMM less.

How: torch custom ops glm53_gemv::linear / glm53_gemv::head_gate (fake impls for torch.compile). They are opaque to
dynamo, so the M-dependent choice (kernel or production op) is made at run / CUDA-graph-capture time with the real
M, never baked into a traced branch. Wiring, after the weights are loaded (vLLM's base_loader
process_weights_after_loading, wrapped): GateLinear.forward (class, per-instance opt-in), Indexer.forward (class,
replaced by a copy that differs only in the two GEMM statements; installed only when the production function is the
fingerprinted version the copy was made from), the Unquantized quant_method of the linears (instance, subclass).
torch.compile cache: vllm_config.additional_config["glm53_bf16_gemv"] gets a tag (version, kinds, dedup, plan
digest), so the AOT / inductor cache keys differ from the stock run and no graph traced in one mode is reused in the
other.
"""
from __future__ import annotations

import logging
import os
import threading
import types
import zlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_log = logging.getLogger("vllm.glm53_bf16_gemv")
_TRUE = frozenset({"1", "on", "true", "yes"})
ENV = "GLM53_BF16_GEMV"
ENV_KINDS = "GLM53_BF16_GEMV_KINDS"
ENV_DEDUP = "GLM53_BF16_GEMV_DEDUP_ROUTER"
ALL_KINDS = ("router", "idx_wk", "idx_head", "idx_kpool", "draft_conv")
VERSION = 1

# Production functions the wiring relies on (integrate.source_fingerprint = sha256 of ast.dump of the source).
# Measured in the image and after the live launcher overlay chain (tests/gemv_chain_check.sh): identical.
FP_GATE_FORWARD = frozenset({"987f92393fb9ef4f"})        # GateLinear.forward (tier 6 = F.linear + .to(out_dtype))
FP_RUNNER_FORWARD_IMPL = frozenset({"5adb8c99ac36c175"})  # MoERunner._forward_impl (recomputes gate if held)
FP_MOE_FORWARD = frozenset({"5492b934207c3148"})         # Glm5NextMoE.forward (gate -> experts(router_logits=))
FP_INDEXER_FORWARD = frozenset({"4d4ce5ad2a9f7ba0"})     # Indexer.forward (the copy below is made from it)
FP_UNQUANT_APPLY = frozenset({"ec998046d0e4a535"})       # UnquantizedLinearMethod.apply (-> F.linear on CUDA)


def env_enabled(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


def env_kinds(environ=None) -> tuple[str, ...]:
    env = os.environ if environ is None else environ
    v = env.get(ENV_KINDS)
    if v is None or not v.strip():
        return ALL_KINDS
    ks = tuple(k.strip() for k in v.split(",") if k.strip())
    bad = [k for k in ks if k not in ALL_KINDS]
    if bad:
        raise ValueError(f"{ENV_KINDS}: unknown kind(s) {bad}; known {ALL_KINDS}")
    return ks


def env_dedup(environ=None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV_DEDUP)
    # unset or empty (the launcher passes every listed variable, empty when not set in .env) = default on
    return v is None or not v.strip() or v.strip().lower() in _TRUE


# ---------------------------------------------------------------------------------------------------------
# registry: handle (stable int from the module's qualified name) -> what the custom ops need

@dataclass
class Entry:
    kind: str
    name: str
    N: int
    K: int
    ctx: object
    row0: int = 0
    served: dict = field(default_factory=dict)     # M -> "gemv <plan>" | "production" (first call per M logged)
    calls_gemv: int = 0
    calls_prod: int = 0


_REG: dict[int, Entry] = {}
_LOCK = threading.Lock()
_STATE = {"ops": False, "loader": False, "tag": None, "gate_cls": False, "indexer_cls": False, "logged": set()}


def handle_for(name: str) -> int:
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def _register(kind: str, name: str, N: int, K: int, ctx, row0: int = 0) -> int:
    h = handle_for(name)
    with _LOCK:
        old = _REG.get(h)
        if old is not None and old.name != name:
            raise RuntimeError(f"handle collision {name} / {old.name}")
        _REG[h] = Entry(kind, name, N, K, ctx, row0)
    return h


def _note(e: Entry, M: int, how: str) -> None:
    key = (e.kind, M, how)
    if key not in _STATE["logged"]:
        _STATE["logged"].add(key)
        _log.info("glm53_bf16_gemv: %s M=%d -> %s", e.kind, M, how)


def _x_ok(x: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_cuda and x.stride(1) == 1 and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0)


def _linear_impl(x: torch.Tensor, weight: torch.Tensor, handle: int, out_mode: int) -> torch.Tensor:
    import glm53_bf16_gemv as G
    e = _REG.get(handle)
    M = x.shape[0] if x.dim() == 2 else -1
    plan = G.serve_plan(weight.shape[0], weight.shape[1], M) if (e is not None and _x_ok(x)) else None
    if plan is not None:
        e.calls_gemv += 1
        if M not in e.served:
            e.served[M] = f"gemv {plan}"
            _note(e, M, f"gemv plan {plan}")
        return G.gemm(x, weight, e.ctx, out_mode=out_mode, plan=plan)
    if e is not None:
        e.calls_prod += 1
        if M not in e.served:
            e.served[M] = "production"
            _note(e, M, "production op (outside the measured-faster range)")
    y = F.linear(x, weight)   # production's op, exactly
    if out_mode == 2:
        return y.to(torch.float32)
    return y.float() if out_mode == 1 else y


def _head_gate_impl(x: torch.Tensor, weight: torch.Tensor, w32: torch.Tensor, handle: int, row0: int) -> torch.Tensor:
    import glm53_bf16_gemv as G
    e = _REG.get(handle)
    M = x.shape[0] if x.dim() == 2 else -1
    if e is not None and _x_ok(x) and 1 <= M <= G.F32_MAX_M:
        e.calls_gemv += 1
        if M not in e.served:
            e.served[M] = "gemm_f32"
            _note(e, M, "gemm_f32 (fp32 CUDA cores)")
        return G.gemm_f32(x, weight[row0:], e.ctx)
    if e is not None:
        e.calls_prod += 1
        if M not in e.served:
            e.served[M] = "production"
            _note(e, M, "production op (outside the measured-faster range)")
    return torch.mm(x.float(), w32)   # production's op, exactly


def register_ops() -> None:
    """torch custom ops glm53_gemv::linear / ::head_gate (idempotent)."""
    if _STATE["ops"]:
        return
    lib_ns = "glm53_gemv"

    @torch.library.custom_op(f"{lib_ns}::linear", mutates_args=())
    def linear(x: torch.Tensor, weight: torch.Tensor, handle: int, out_mode: int) -> torch.Tensor:
        return _linear_impl(x, weight, handle, out_mode)

    @linear.register_fake
    def _(x, weight, handle, out_mode):
        return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16 if out_mode == 0 else torch.float32)

    @torch.library.custom_op(f"{lib_ns}::head_gate", mutates_args=())
    def head_gate(x: torch.Tensor, weight: torch.Tensor, w32: torch.Tensor, handle: int, row0: int) -> torch.Tensor:
        return _head_gate_impl(x, weight, w32, handle, row0)

    @head_gate.register_fake
    def _(x, weight, w32, handle, row0):
        return x.new_empty((*x.shape[:-1], w32.shape[1]), dtype=torch.float32)

    _STATE["ops"] = True


# ---------------------------------------------------------------------------------------------------------
# self-test of one module's weight (load time, eager, outside any graph capture)

def selftest(weight: torch.Tensor, ctx, out_mode: int, f32_rows: int | None = None) -> tuple[bool, str]:
    """The kernel against float64 on this weight, at every M it will serve (plus M = 1): each output within the
    fp32-accumulation bound 1e-5 * sum_k |x_k w_k| of the exact value, bf16 outputs additionally within 1 bf16 ulp
    of it; bitwise deterministic; ticket counters back at zero."""
    import glm53_bf16_gemv as G
    N, K = weight.shape
    w = weight[f32_rows:] if f32_rows is not None else weight
    if f32_rows is not None:
        Ms = [m for m in (1, 5, 8, 16, 32, 48, 64) if m <= G.F32_MAX_M]
    else:
        Ms = [m for m in (1, 5, 8, 16, 32, 48, 64) if G.serve_plan(N, K, m) is not None]
    if not Ms:
        return False, "no M bucket is served for this shape"
    gen = torch.Generator(device=weight.device).manual_seed(20260928)
    worst = 0.0
    for M in Ms:
        x = torch.randn(M, K, device=weight.device, generator=gen).to(torch.bfloat16)
        if f32_rows is not None:
            y = G.gemm_f32(x, w, ctx)
            y2 = G.gemm_f32(x, w, ctx)
        else:
            plan = G.serve_plan(N, K, M)
            y = G.gemm(x, w, ctx, out_mode=out_mode, plan=plan)
            y2 = G.gemm(x, w, ctx, out_mode=out_mode, plan=plan)
        torch.cuda.synchronize()
        ref = x.double() @ w.double().t()
        mag = x.double().abs() @ w.double().abs().t()
        err = (y.double() - ref).abs()
        tol = 1e-5 * mag + 1e-30
        if out_mode in (0, 2) and f32_rows is None:
            tol = tol + ref.abs() * 2.0 ** -8
        if not bool(torch.isfinite(y).all()):
            return False, f"M={M}: non-finite output"
        if not bool((err <= tol).all()):
            i = int((err - tol).argmax())
            return False, f"M={M}: error {float(err.flatten()[i]):.3g} > bound {float(tol.flatten()[i]):.3g}"
        if not torch.equal(y, y2):
            return False, f"M={M}: not deterministic"
        if int(ctx.counters.abs().sum()) != 0:
            return False, f"M={M}: ticket counters not reset"
        worst = max(worst, float((err / mag.clamp_min(1e-30)).max()))
    return True, f"M {Ms}: max err / sum|xw| {worst:.2e}"


# ---------------------------------------------------------------------------------------------------------
# wiring

def _fp(fn) -> str | None:
    from integrate import source_fingerprint
    return source_fingerprint(getattr(fn, "_glm53_gemv_orig", fn))


def _patch_gate_class(GateLinear) -> bool:
    if _STATE["gate_cls"]:
        return True
    orig = GateLinear.forward
    if _fp(orig) not in FP_GATE_FORWARD:
        _log.warning("glm53_bf16_gemv: GateLinear.forward fingerprint %s not in %s; router not served", _fp(orig),
                     sorted(FP_GATE_FORWARD))
        return False

    def forward(self, x):
        h = getattr(self, "_glm53_gemv_handle", None)
        if h is None:
            return orig(self, x)
        if x.dtype != self.weight.dtype:   # as tier 6 does before its F.linear
            x = x.to(self.weight.dtype)
        return torch.ops.glm53_gemv.linear(x, self.weight, h, 2), None

    forward._glm53_gemv_orig = orig
    forward.__doc__ = orig.__doc__
    GateLinear.forward = forward
    _STATE["gate_cls"] = True
    return True


_WRAPPED_QM: dict[type, type] = {}


def _wrap_unquantized(layer, handle: int) -> bool:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    qm = getattr(layer, "quant_method", None)
    if type(qm) is not UnquantizedLinearMethod:
        return False
    if _fp(UnquantizedLinearMethod.apply) not in FP_UNQUANT_APPLY:
        _log.warning("glm53_bf16_gemv: UnquantizedLinearMethod.apply fingerprint %s unknown",
                     _fp(UnquantizedLinearMethod.apply))
        return False
    base = type(qm)
    cls = _WRAPPED_QM.get(base)
    if cls is None:
        def apply(self, layer, x, bias=None):
            h = getattr(layer, "_glm53_gemv_handle", None)
            if h is None or bias is not None:
                return base.apply(self, layer, x, bias)
            return torch.ops.glm53_gemv.linear(x, layer.weight, h, 0)
        cls = type(f"Glm53Gemv{base.__name__}", (base,), {"apply": apply, "_glm53_gemv_base": base})
        _WRAPPED_QM[base] = cls
    qm.__class__ = cls
    layer._glm53_gemv_handle = handle
    return True


def _indexer_forward_gemv(self, hidden_states, qr, positions, rotary_emb):
    # Copy of vllm/models/glm5next/nvidia/attention.py Indexer.forward (fingerprint FP_INDEXER_FORWARD) with two
    # statements replaced (marked GLM53_BF16_GEMV); it runs with that module's globals (_install_indexer).
    # tests/test_gemv_install.py checks that the ASTs differ in exactly those two statements.
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)

    kw, _ = self.wk_weights_proj(hidden_states)
    k = kw[:, : self.head_dim]
    if getattr(self, "_wp_fp32", None) is None:
        self._wp_fp32 = (
            self.wk_weights_proj.weight.data[self.head_dim :, :]
            .t()
            .contiguous()
            .float()
        )
    # GLM53_BF16_GEMV: was  weights = torch.mm(hidden_states.float(), self._wp_fp32)
    h_head, h_kpool = self._glm53_gemv_idx
    if h_head >= 0:
        weights = torch.ops.glm53_gemv.head_gate(
            hidden_states, self.wk_weights_proj.weight, self._wp_fp32, h_head, self.head_dim
        )
    else:
        weights = torch.mm(hidden_states.float(), self._wp_fp32)

    k = _fused_indexer_k_norm(  # noqa: F821 (attention.py global)
        k, self.k_norm.weight, self.k_norm.bias, self.head_dim, self.k_norm.eps
    )

    if self.rope_dim > 0:
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

    assert self.head_dim == 128 and self.quant_block_size == 128
    assert self.scale_fmt == "ue8m0"
    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = fwht128_quant_fp8(q)  # noqa: F821
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)

    weights = _fused_indexer_weight_scale(  # noqa: F821
        weights, q_scale, self.softmax_scale * self.n_head**-0.5
    )

    # GLM53_BF16_GEMV: was  gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)
    if h_kpool >= 0:
        gate_score = torch.ops.glm53_gemv.linear(hidden_states, self.index_kpool_compress_gate, h_kpool, 0)
    else:
        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)

    if self.n_head < 32:
        pad = 32 - self.n_head
        q_fp8 = _pad_indexer_heads(q_fp8, pad)  # noqa: F821
        weights = _pad_indexer_heads(weights, pad)  # noqa: F821

    return self.indexer_op(
        hidden_states,
        q_fp8,
        k,
        weights,
        gate_score=gate_score,
        compress_ape=self.index_kpool_compress_ape,
        index_kpool=self.index_kpool,
        positions=positions,
    )


def _install_indexer_class(Indexer) -> bool:
    if _STATE["indexer_cls"]:
        return True
    import sys
    orig = Indexer.forward
    if _fp(orig) not in FP_INDEXER_FORWARD:
        _log.warning("glm53_bf16_gemv: Indexer.forward fingerprint %s not in %s; indexer head/kpool gates not served",
                     _fp(orig), sorted(FP_INDEXER_FORWARD))
        return False
    mod_globals = sys.modules[orig.__module__].__dict__
    patched = types.FunctionType(_indexer_forward_gemv.__code__, mod_globals, "forward",
                                 _indexer_forward_gemv.__defaults__, _indexer_forward_gemv.__closure__)

    def forward(self, hidden_states, qr, positions, rotary_emb):
        if getattr(self, "_glm53_gemv_idx", None) is None:
            return orig(self, hidden_states, qr, positions, rotary_emb)
        return patched(self, hidden_states, qr, positions, rotary_emb)

    forward._glm53_gemv_orig = orig
    Indexer.forward = forward
    _STATE["indexer_cls"] = True
    return True


def _find_runner(moe) -> object | None:
    experts = getattr(moe, "experts", None)
    if experts is None:
        return None
    for sub in [experts, *experts.modules()]:
        if getattr(sub, "gate", None) is moe.gate and hasattr(sub, "_forward_impl"):
            return sub
    for name in ("runner", "_runner", "moe_runner"):
        r = getattr(experts, name, None)
        if r is not None and getattr(r, "gate", None) is moe.gate and hasattr(r, "_forward_impl"):
            return r
    return None


def _tag_compile_cache(kinds, dedup) -> None:
    import glm53_bf16_gemv as G
    tag = f"v{VERSION}:{','.join(kinds)}:dedup={int(dedup)}:plans={G.plans_digest()}"
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        ac = cfg.additional_config
        if not isinstance(ac, dict):
            raise TypeError(f"additional_config is {type(ac).__name__}, not a dict")
        if ac.get("glm53_bf16_gemv") != tag:
            ac["glm53_bf16_gemv"] = tag
        _STATE["tag"] = tag
        _log.info("glm53_bf16_gemv: compile-cache tag additional_config[glm53_bf16_gemv]=%s (config hash %s)", tag,
                  cfg.compute_hash()[:12])
    except Exception as exc:  # noqa: BLE001
        _STATE["tag"] = None
        _log.warning("glm53_bf16_gemv: could not tag the compile cache (%r): a torch.compile artifact traced in the "
                     "other mode could be reused; the custom ops still fall back to production's op for any "
                     "unregistered module", exc)


def post_load(model, environ=None) -> dict:
    """Self-test and wire every served module of a freshly loaded model. Returns a summary (also logged)."""
    import glm53_bf16_gemv as G
    kinds = env_kinds(environ)
    dedup = env_dedup(environ)
    _tag_compile_cache(kinds, dedup)
    tag = type(model).__name__
    out = {k: 0 for k in ALL_KINDS}
    out.update(dedup=0, rejected=[])
    try:
        import vllm.envs as envs
        if getattr(envs, "VLLM_BATCH_INVARIANT", False):
            _log.warning("glm53_bf16_gemv: VLLM_BATCH_INVARIANT is set (production uses linear_batch_invariant); "
                         "nothing served")
            return out
    except Exception:  # noqa: BLE001
        pass
    dev = None
    for p in model.parameters():
        dev = p.device
        break
    if dev is None or dev.type != "cuda":
        return out

    def reject(name, why):
        out["rejected"].append(f"{name}: {why}")
        _log.warning("glm53_bf16_gemv: %s not served: %s", name, why)

    for name, m in model.named_modules():
        cls = type(m).__name__
        qual = f"{tag}:{name}"
        # ---- router gate
        if cls == "GateLinear" and "router" in kinds:
            w = getattr(m, "weight", None)
            flags = [f for f in ("allow_ll_bf16_gemm", "allow_dsv3_router_gemm", "allow_fp32_router_gemm",
                                 "allow_bf16x3_router_gemm", "allow_cublas_router_gemm") if getattr(m, f, False)]
            if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or (w.shape[0], w.shape[1]) not in G.PLANS:
                reject(qual, f"weight {None if w is None else (w.dtype, tuple(w.shape))} not a served bf16 shape")
            elif getattr(m, "bias", None) is not None or m.out_dtype is not torch.float32 or flags:
                reject(qual, f"bias / out_dtype {m.out_dtype} / specialized tiers {flags}: not production's tier 6")
            elif not _patch_gate_class(type(m)):
                reject(qual, "GateLinear.forward not the verified version")
            else:
                ctx = G.GemmCtx(w.shape[0], w.shape[1], w.device, served_only=True)
                ok, why = selftest(w, ctx, 2)
                if ok:
                    m._glm53_gemv_handle = _register("router", qual, w.shape[0], w.shape[1], ctx)
                    out["router"] += 1
                else:
                    reject(qual, "self-test: " + why)
        # ---- router dedup
        elif cls == "Glm5NextMoE" and dedup:
            r = _find_runner(m)
            if r is None:
                reject(qual, "dedup: no MoE runner holding this layer's gate")
            elif _fp(type(r)._forward_impl) not in FP_RUNNER_FORWARD_IMPL or _fp(type(m).forward) not in FP_MOE_FORWARD:
                reject(qual, "dedup: MoERunner._forward_impl / Glm5NextMoE.forward not the verified versions")
            elif getattr(r, "_fse_fuse_gate", False) or getattr(r, "routed_input_transform", None) is not None:
                reject(qual, "dedup: runner fuses a shared-expert gate or transforms the routed input")
            else:
                r.gate = None
                m._glm53_gemv_dedup = True
                out["dedup"] += 1
        # ---- indexer (wk_weights_proj, head gate, kpool gate)
        elif cls == "Indexer" and any(k in kinds for k in ("idx_wk", "idx_head", "idx_kpool")):
            wkp = getattr(m, "wk_weights_proj", None)
            hd = getattr(m, "head_dim", None)
            w = getattr(wkp, "weight", None)
            kg = getattr(m, "index_kpool_compress_gate", None)
            if w is None or w.dtype != torch.bfloat16 or hd != 128 or tuple(w.shape) != (160, 4096):
                reject(qual, f"wk_weights_proj weight {None if w is None else (w.dtype, tuple(w.shape))}")
                continue
            if "idx_wk" in kinds:
                ctx = G.GemmCtx(160, 4096, w.device, served_only=True)
                ok, why = selftest(w, ctx, 0)
                if ok and _wrap_unquantized(wkp, _register("idx_wk", qual + ".wk_weights_proj", 160, 4096, ctx)):
                    out["idx_wk"] += 1
                else:
                    reject(qual + ".wk_weights_proj", why if not ok else "quant_method is not UnquantizedLinearMethod")
            h_head = h_kpool = -1
            if "idx_head" in kinds or "idx_kpool" in kinds:
                if not _install_indexer_class(type(m)):
                    reject(qual, "Indexer.forward not the verified version")
                    continue
            if "idx_head" in kinds:
                ctx = G.GemmCtx(32, 4096, w.device, f32=True)
                ok, why = selftest(w, ctx, 1, f32_rows=hd)
                if ok:
                    if getattr(m, "_wp_fp32", None) is None:   # production builds it lazily on the first forward
                        m._wp_fp32 = w.data[hd:, :].t().contiguous().float()
                    h_head = _register("idx_head", qual + ".head_gate", 32, 4096, ctx, row0=hd)
                    out["idx_head"] += 1
                else:
                    reject(qual + ".head_gate", "self-test: " + why)
            if "idx_kpool" in kinds:
                if kg is None or kg.dtype != torch.bfloat16 or tuple(kg.shape) != (128, 4096):
                    reject(qual + ".kpool_gate", f"weight {None if kg is None else (kg.dtype, tuple(kg.shape))}")
                else:
                    ctx = G.GemmCtx(128, 4096, kg.device, served_only=True)
                    ok, why = selftest(kg, ctx, 0)
                    if ok:
                        h_kpool = _register("idx_kpool", qual + ".kpool_gate", 128, 4096, ctx)
                        out["idx_kpool"] += 1
                    else:
                        reject(qual + ".kpool_gate", "self-test: " + why)
            if h_head >= 0 or h_kpool >= 0:
                m._glm53_gemv_idx = (h_head, h_kpool)
        # ---- DFlash2 conv kernel projections
        elif cls == "DFlashGroupedConv" and "draft_conv" in kinds:
            kp = getattr(m, "kernel_projection", None)
            w = getattr(kp, "weight", None)
            if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or (w.shape[0], w.shape[1]) not in G.PLANS:
                reject(qual, f"kernel_projection weight {None if w is None else (w.dtype, tuple(w.shape))}")
                continue
            ctx = G.GemmCtx(w.shape[0], w.shape[1], w.device, served_only=True)
            ok, why = selftest(w, ctx, 0)
            if ok and _wrap_unquantized(kp, _register("draft_conv", qual + ".kernel_projection", w.shape[0],
                                                      w.shape[1], ctx)):
                out["draft_conv"] += 1
            else:
                reject(qual + ".kernel_projection", why if not ok else "quant_method is not UnquantizedLinearMethod")
    served = {k: out[k] for k in (*ALL_KINDS, "dedup") if out[k]}
    _log.info("glm53_bf16_gemv installed on %s: %s; %d rejected; kinds %s; compile-cache tag %s", tag,
              served or "nothing", len(out["rejected"]), ",".join(kinds), _STATE["tag"])
    return out


def _hook_loader() -> bool:
    if _STATE["loader"]:
        return True
    import vllm.model_executor.model_loader.base_loader as BL
    orig = BL.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        try:
            post_load(model)
        except Exception as exc:  # noqa: BLE001 - never break model loading; the model keeps production's path
            _log.warning("glm53_bf16_gemv: post-load wiring failed (%r); modules wired before the failure stay "
                         "wired and fall back per call if unregistered", exc)

    process_weights_after_loading._glm53_gemv_orig = orig
    BL.process_weights_after_loading = process_weights_after_loading
    _STATE["loader"] = True
    return True


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_BF16_GEMV is on."""
    on = env_enabled()
    _log.info("glm53_bf16_gemv plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              "installing" if on else "off, production GEMMs unchanged")
    if not on:
        return
    try:
        kinds, dedup = env_kinds(), env_dedup()
        import glm53_bf16_gemv as G
        G.load_ext()
        register_ops()
        _hook_loader()
        _log.info("glm53_bf16_gemv: ext %s, kinds %s, router dedup %s; modules are wired after weight loading",
                  G.EXT_SOURCE, ",".join(kinds), "on" if dedup else "off")
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_bf16_gemv not installed (production GEMMs unchanged): %r", exc)


def summary() -> dict:
    """Per kind: modules, calls served by the kernel / by production's op, and the M values seen per path."""
    s: dict = {}
    for e in _REG.values():
        d = s.setdefault(e.kind, {"modules": 0, "gemv": 0, "production": 0, "M": {}})
        d["modules"] += 1
        d["gemv"] += e.calls_gemv
        d["production"] += e.calls_prod
        for M, how in e.served.items():
            d["M"][M] = how
    return s
