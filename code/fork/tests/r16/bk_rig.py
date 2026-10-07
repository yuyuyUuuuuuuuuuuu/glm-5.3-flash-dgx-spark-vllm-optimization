"""[sidestream] A production-shaped GLM-5.3-Flash decoder stack driven through vLLM's BREAKABLE CUDA-graph capture
(vllm/compilation/breakable_cudagraph.py, auto-enabled for Glm5Next* by vllm/config/vllm.py: production's PIECEWISE
graphs), with the R16 decode hooks installed in plugin order. Used by tests/r16/test_r16_breakable.py (correctness,
reproduction of the first R16 boot's "capturing stream has unjoined work") and its timing section.

Every forward below is vLLM's own code, called on shells (no __init__, attributes set by hand):
  Glm5NextModel.forward, Glm5NextDecoderLayer.forward (both as recompiled by GLM53_PREFILL_QUICKWINS item mhc_mean, the
  production order hc_pre / self_attn / hc_fused_post_pre / mlp), Glm5NextLinearAttention.forward (KDA: in_proj ->
  f_b / g_b -> self._forward = the eager break -> o_norm -> o_proj), Glm5NextMLAAttention.forward ->
  MultiHeadLatentAttentionWrapper.forward (fused_qkv_a -> q_b -> self.indexer(...) -> self.mla_attn(...) -> o_proj),
  Glm5NextMoE.forward, Glm5NextMLP.forward, RowParallelLinear.forward (o_proj).
Every FP8 linear = production's Glm53DenseFp8Method (launcher overlay exl3.py, GPU_RUN_BIND) at its production shape,
group and prefix, served by fp8_gemv (GLM53_FP8_GEMV=1, MAX_M 16) with fp8_roof's hooks; each MoE layer's routed experts
= a real EXL3 layer (tests/harness.make_layer) through production's Exl3MoEMethod.apply (fp8_roof t3 hook) ->
apply_exl3_experts (moeglue's warm join) -> TF K2; shared experts on an aux stream (vLLM's runner order); o_proj wired
by moeglue's own warm_post_load.

The eager breaks: vLLM's real @eager_break_during_capture decorator (VLLM_USE_BREAKABLE_CUDAGRAPH=1 before vllm is
imported) around stand-ins at the three places production has one (KDA core = Glm5NextLinearAttention._forward;
sparse_attn_indexer_kpool; unified_mla_attention_with_output), writing into buffers allocated inside the graph segment
before them, as the real ops do. kda_core="real" keeps production's own (quickwins-recompiled, decorated) _forward:
with attn_metadata None it returns at once, so the core output is uninitialised (capture / replay checks only).
Capture exactly as vllm/v1/worker/gpu/cudagraph_utils.py does: PIECEWISE = BreakableCUDAGraphWrapper under a forward
context (runtime mode PIECEWISE, BatchDescriptor) after an eager warm-up (mode NONE); FULL = torch.cuda.graph(pool)
with the forward context in mode NONE (no BreakableCUDAGraphCapture active: every break is inert).

Rig(quickwins=False) leaves GLM53_PREFILL_QUICKWINS out (production's state while quickwins is under investigation):
the same stack on vLLM's stock Glm5NextModel / DecoderLayer / KDA forwards.

Layers (NL = 8, production's pattern): 0-2 KDA + dense MLP, 3 MLA + MoE, 4-6 KDA + MoE, 7 MLA + MoE (the last layer:
final hc_post + hc_contract). Per forward: t1 6 (KDA in_proj -> o_proj), t2 8 (o_proj -> shared / dense gate_up), t3 7
(MoE / dense down -> next layer's first linear), t4 2 (MLA q_b -> o_proj), warm 5 (MoE o_proj -> MoE apply).
"""
from __future__ import annotations

import logging
import os
import socket
import sys
import types
from pathlib import Path

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"   # decorators are applied at import: must precede any vllm import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import harness as H  # noqa: E402
import torch  # noqa: E402

dev = "cuda"
K = 4096
NE, NI = 32, 1024                    # routed experts per test layer, intermediate per rank
LIMIT = 10.0
KINDS = ["kda", "kda", "kda", "mla", "kda", "kda", "kda", "mla"]
WANT = {"t0": 0, "t1": 6, "t2": 8, "t3": 7, "t4": 2, "t5": 0, "warm": 5}
BAD_ROOF = ("joined_late_safety", "stale_dropped", "dropped_capture_mismatch", "double_fork")
BAD_MG = ("warm_unjoined", "warm_stale_dropped", "warm_error", "warm_no_stream")


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelno, r.name, r.getMessage()))


def single_rank_tp():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.distributed import parallel_state as ps
    if ps.model_parallel_is_initialized():
        return
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(1, 1)


class Rig:
    """Build once per process: installs (plugin order), the stack, capture / replay helpers."""

    def __init__(self, kda_core: str = "standin", nl: int = 8, quickwins: bool = True):
        for v in [k for k in os.environ if k.startswith(("GLM53_", "TF_EXL3_"))]:
            if v != "TF_EXL3_JIT":
                os.environ.pop(v)
        single_rank_tp()
        self.xl = H.load_xl()
        self.prod = H.load_prod()
        self.tf = H.load_tf()
        self.tf.load_ext()
        import fp8_gemv as F
        import fp8_roof as R
        import glm53_moeglue as MG
        import glm53_prefill_quickwins as Q
        import integrate
        import vllm.compilation.breakable_cudagraph as BK
        from vllm.config import VllmConfig
        self.F, self.R, self.MG, self.Q, self.BK = F, R, MG, Q, BK
        self.vcfg = VllmConfig()
        self.cap = Cap()
        for nm in ("vllm.tf_fp8_roof", "vllm.tf_fp8_gemv", "vllm.glm53_moeglue", "vllm.glm53_prefill_quickwins"):
            logging.getLogger(nm).addHandler(self.cap)
            logging.getLogger(nm).setLevel(logging.DEBUG)
        # ---- installs in integrate.plugin_register's order (production env: TF_EXL3_MOE, quickwins, warm, fp8_gemv,
        # fp8roof); every feature armed here, configurations switch them with setcfg()
        integrate.install(prodmod=self.prod, ext=self.xl, force=True)
        os.environ["GLM53_DEC_MOEGLUE_WARM"] = "1"
        self.mrep = MG.install(prodmod=self.prod, hook_loader=False)
        # quickwins=False: production's state since 2026-09-29 (GLM53_PREFILL_QUICKWINS taken out of .env): vLLM's own
        # Glm5NextModel / DecoderLayer / KDA forwards, not the recompiled ones (the break positions are the same)
        self.qrep = Q.install(["mhc_mean", "kda_conv"]) if quickwins else {"installed": [], "skipped": "quickwins off"}
        self.quickwins = quickwins
        os.environ.update(GLM53_FP8_GEMV="1", GLM53_FP8_GEMV_MAX_M="16", GLM53_DEC_FP8ROOF="1")
        self.frep = F.install(self.prod)
        self.rrep = R.install(self.prod)
        MG.CFG.strict = True
        from vllm.models.glm5next.nvidia import kda as kda_mod
        from vllm.models.glm5next.nvidia import model as gm
        from vllm.models.glm5next.nvidia.attention import Glm5NextMLAAttention
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
        import vllm.model_executor.layers.attention.mla_attention as mla_att
        import vllm.model_executor.layers.sparse_attn_indexer_kpool as idx_kpool
        self.gm = gm
        # production's break points really are decorated (the positions the shells below reproduce)
        self.decorated = {
            "Glm5NextLinearAttention._forward": hasattr(kda_mod.Glm5NextLinearAttention._forward, "__wrapped__"),
            "unified_mla_attention_with_output": hasattr(mla_att.unified_mla_attention_with_output, "__wrapped__"),
            "sparse_attn_indexer_kpool": hasattr(idx_kpool.sparse_attn_indexer_kpool, "__wrapped__"),
        }
        self.spin_on = False
        self.cyc = 0.0
        self.inject = None            # (layer index, where) -> raise RuntimeError inside a forward (adversarial)
        g = torch.Generator(device=dev).manual_seed(0)
        self.g = g
        cls = self.prod.Glm53DenseFp8Method
        rig = self

        class Lin(torch.nn.Module):
            """A production FP8 linear (Glm53DenseFp8Method at its shape / group / prefix); forward like LinearBase."""

            def __init__(self, n, k, group, prefix, scale=0.02):
                super().__init__()
                w = (torch.randn(n, k, device=dev, generator=g) * scale *
                     torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
                self.weight = torch.nn.Parameter(w, requires_grad=False)
                self.output_size_per_partition, self.input_size_per_partition = n, k
                self.bias = None
                self.quant_method = cls(group, prefix)
                self.quant_method.process_weights_after_loading(self)

            def forward(self, x):
                return self.quant_method.apply(self, x), None

        def oproj(kin, group, prefix):
            with torch.device(dev):
                op = RowParallelLinear(kin, K, bias=False, params_dtype=torch.bfloat16, prefix=prefix)
            op.weight.data.copy_((torch.randn(K, kin, device=dev, generator=g) * 0.02).to(torch.bfloat16))
            op.quant_method = cls(group, prefix)
            op.quant_method.process_weights_after_loading(op)
            return op

        def spin(us):
            if rig.spin_on and rig.cyc > 0:
                torch.cuda._sleep(int(us * rig.cyc))

        self.spin = spin

        def maybe_inject(Lx, where):
            if rig.inject == (Lx, where):
                raise RuntimeError(f"injected failure at layer {Lx} {where}")

        EB = BK.eager_break_during_capture

        # ---- eager-break stand-ins (in-place writes into buffers allocated in the graph segment before them)
        def kda_core(qkv_proj_states, g1, beta, core_attn_out):
            T = core_attn_out.shape[1]
            spin(7)                                                     # conv
            if rig.spin_on:
                rig.state.copy_(rig.state_src)                          # per-token recurrent-state writes (8 MiB)
            spin(24)                                                    # recurrent
            v = qkv_proj_states[:T, 8192:12288].float()
            gate = torch.sigmoid(g1.reshape(T, K).float()) * torch.sigmoid(beta.reshape(T, 32, 1).float()).repeat_interleave(128, 1).reshape(T, K)
            core_attn_out.copy_((v * gate).reshape(1, T, 32, 128).to(core_attn_out.dtype))

        def idx_core(q_idx, topk_out):
            T = q_idx.shape[0]
            spin(40)                                                    # sparse indexer
            topk_out[:T].copy_(torch.argsort(q_idx.float(), dim=-1)[:, :64].to(topk_out.dtype))

        def attn_core(q, kv_c_normed, k_pe, output):
            T = q.shape[0]
            if rig.spin_on:
                torch.sum(rig.attw, dim=0, out=rig.red)                 # attention's own reads (16 MiB)
            spin(110)                                                   # sparse-MLA attention
            w = torch.sigmoid(kv_c_normed.float().mean(-1, keepdim=True) + k_pe.float().reshape(T, -1).mean(-1, keepdim=True))
            output.copy_((q.reshape(T, 8192).float() * w).to(output.dtype))

        self.kda_core_break = EB(kda_core)
        idx_break = EB(idx_core)
        attn_break = EB(attn_core)
        self.topk_buf = torch.zeros(256, 64, dtype=torch.int32, device=dev)
        self.state_src = torch.randn(2 << 20, device=dev)
        self.state = torch.empty_like(self.state_src)
        self.attw = torch.randn(8 << 20, device=dev).to(torch.bfloat16)
        self.red = torch.empty((), dtype=torch.float32, device=dev)

        # ---- mHC ops (production: tilelang kernels reading hc_*_fn, 1.5 MiB; AR of the previous sublayer before)
        def _mix(resid, fn, norm_w, eps):
            T = resid.shape[0]
            m = resid.reshape(T, -1).float() @ fn.t()                   # [T, 24]: reads fn (1.5 MiB)
            post = torch.sigmoid(m[:, :4])
            comb = torch.softmax(m[:, 8:24].reshape(T, 4, 4), -1)
            pre = torch.softmax(m[:, 4:8], -1)
            li = (pre.unsqueeze(-1) * resid.float()).sum(1)
            li = li * torch.rsqrt(li.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()
            return post, comb, li.to(torch.bfloat16)

        def _post(x, residual, post, comb):
            return (torch.einsum("tij,tjh->tih", comb, residual.float()) +
                    post.unsqueeze(-1) * x.float().unsqueeze(1)).to(torch.bfloat16)

        class PreOp(torch.nn.Module):
            def forward(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                        hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
                return _mix(residual, fn, norm_weight, norm_eps or 1e-6)

        class FusedPostPreOp(torch.nn.Module):
            def forward(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                        norm_eps=0.0):
                spin(25)                                                # the previous sublayer's all-reduce
                maybe_inject(self.layer_idx, "mhc")
                r = _post(x, residual, post_layer_mix, comb_res_mix)
                post, comb, li = _mix(r, fn, norm_weight, norm_eps or 1e-6)
                spin(6)
                return r, post, comb, li

        class PostOp(torch.nn.Module):
            def forward(self, x, residual, post, comb):
                return _post(x, residual, post, comb)

        class Norm(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(n, device=dev, generator=g), requires_grad=False)
                self.variance_epsilon = 1e-6

            def forward(self, x, *a):
                y = x.float()
                return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight.float()).to(x.dtype)

        class ONorm(torch.nn.Module):                                   # FusedRMSNormGated(head_dim, sigmoid)
            def forward(self, x, gate):
                y = x.float()
                y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
                return (y * torch.sigmoid(gate.float().reshape(y.shape))).to(x.dtype)

        class Act(torch.nn.Module):
            def forward(self, x):
                n = x.shape[-1] // 2
                return torch.nn.functional.silu(x[..., :n]) * x[..., n:]

        def mlp_shell(prefix, group, n_inter):
            m = gm.Glm5NextMLP.__new__(gm.Glm5NextMLP)
            torch.nn.Module.__init__(m)
            m.gate_up_proj = Lin(2 * n_inter, K, group, prefix + ".gate_up_proj")
            m.down_proj = Lin(K, n_inter, group, prefix + ".down_proj")
            m.act_fn = Act()
            return m

        # ---- routed experts (real EXL3) + shared on the aux stream
        em = self.prod.Exl3MoEMethod.__new__(self.prod.Exl3MoEMethod)
        em.moe = types.SimpleNamespace(swiglu_limit=LIMIT)
        self.em = em
        self.aux = torch.cuda.Stream()
        self.routes = {}
        self.topk = 1

        class Experts(torch.nn.Module):
            def __init__(self, Lx, shared):
                super().__init__()
                self.Lx = Lx
                self.tl = H.make_layer(rig.prod, H.Weights(NE, K, NI, dev, seed=40 + Lx))
                self.tl.layer_name = f"model.layers.{Lx}.mlp.experts"
                inner = torch.nn.Module()
                inner.quant_method = em                                  # moeglue: served by Exl3MoEMethod
                self.routed = inner
                self.shared = shared

            def forward(self, hidden_states, router_logits):
                M = hidden_states.shape[0]
                ids, ws = rig.routes[(M, rig.topk)][self.Lx]
                cur = torch.cuda.current_stream()
                rig.aux.wait_stream(cur)                                 # maybe_sync_shared_experts_stream
                spin(3); spin(24)                                        # topk, rot_in
                maybe_inject(self.Lx, "moe")
                routed = em.apply(self.tl, hidden_states, ws, ids, None, None)
                with torch.cuda.stream(rig.aux):                         # MULTI_STREAM_OVERLAPPED
                    sh = self.shared(hidden_states)
                cur.wait_stream(rig.aux)
                spin(8)                                                  # epilogue
                return routed + sh

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter((torch.randn(288, K, device=dev, generator=g) * 0.02)
                                                 .to(torch.bfloat16), requires_grad=False)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight).float(), None

        # ---- layers
        dls = []
        for Lx, kind in enumerate(KINDS[:nl]):
            p = f"model.layers.{Lx}"
            dl = gm.Glm5NextDecoderLayer.__new__(gm.Glm5NextDecoderLayer)
            torch.nn.Module.__init__(dl)
            dl.layer_idx, dl.num_hidden_layers, dl.is_mtp_layer, dl.mhc, dl.n = Lx, nl, False, True, 4
            dl.is_sequence_parallel = False
            dl.rms_norm_eps, dl.hc_eps, dl.mhc_post_mult_value, dl.mhc_sinkhorn_iterations = 1e-6, 1e-6, 2.0, 20
            for nm in ("attn", "ffn"):
                setattr(dl, f"hc_{nm}_fn", torch.nn.Parameter(torch.randn(24, 4 * K, device=dev, generator=g) * 0.01,
                                                             requires_grad=False))
                setattr(dl, f"hc_{nm}_scale", torch.nn.Parameter(torch.ones(3, device=dev), requires_grad=False))
                setattr(dl, f"hc_{nm}_base", torch.nn.Parameter(torch.zeros(24, device=dev), requires_grad=False))
            dl.input_layernorm, dl.post_attention_layernorm = Norm(K), Norm(K)
            dl.mhc_pre_op, dl.mhc_post_op = PreOp(), PostOp()
            dl.mhc_fused_post_pre_op = FusedPostPreOp()
            dl.mhc_fused_post_pre_op.layer_idx = Lx
            if kind == "kda":
                a = kda_mod.Glm5NextLinearAttention.__new__(kda_mod.Glm5NextLinearAttention)
                torch.nn.Module.__init__(a)
                a.prefix = p + ".self_attn"
                a.local_projection_size, a.local_num_heads, a.head_dim = 4096, 32, 128
                a.in_proj_qkvbfg_a = Lin(12576, K, "kda", p + ".self_attn.in_proj_qkvbfg_a")
                a.f_b_proj = Lin(4096, 128, "kda", p + ".self_attn.f_b_proj")
                a.g_b_proj = Lin(4096, 128, "kda", p + ".self_attn.g_b_proj")
                a.o_norm = ONorm()
                a.o_proj = oproj(4096, "kda", p + ".self_attn.o_proj")
                if kda_core != "real":
                    a._forward = self.kda_core_break                    # instance attribute: the class's is production's
            else:
                a = Glm5NextMLAAttention.__new__(Glm5NextMLAAttention)
                torch.nn.Module.__init__(a)
                w = MultiHeadLatentAttentionWrapper.__new__(MultiHeadLatentAttentionWrapper)
                torch.nn.Module.__init__(w)
                w.q_lora_rank, w.kv_lora_rank, w.qk_rope_head_dim = 1536, 448, 64
                w.qk_nope_head_dim, w.qk_head_dim, w.v_head_dim, w.num_heads = 192, 256, 256, 32
                w.fuse_qkv_rmsnorm, w.dcp_q_replicate, w.rotary_emb, w.g_proj = False, False, None, None
                w.is_sparse, w.skip_topk, w.indexer_rope_emb = True, False, None
                w.fused_qkv_a_proj = Lin(2048, K, "mla", p + ".self_attn.fused_qkv_a_proj")
                w.q_b_proj = Lin(8192, 1536, "mla", p + ".self_attn.q_b_proj")
                w.q_a_layernorm, w.kv_a_layernorm = Norm(1536), Norm(448)
                wq = (torch.randn(64, 1536, device=dev, generator=g) * 0.02).to(torch.bfloat16)

                class Indexer(torch.nn.Module):                         # wq_b GEMV, then the indexer op (break)
                    def forward(self, hidden_states, q_c, positions, rope):
                        idx_break(torch.nn.functional.linear(q_c, wq), rig.topk_buf)

                class MLAAttn(torch.nn.Module):                          # MLAAttention.forward: output, then the op
                    def forward(self, q, kv_c_normed, k_pe, output_shape=None, q_dcp_replicated=None):
                        out = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                        attn_break(q, kv_c_normed, k_pe, out)
                        return out

                w.indexer, w.mla_attn = Indexer(), MLAAttn()
                w.o_proj = oproj(8192, "mla", p + ".self_attn.o_proj")
                a.mla_attn = w
                # Glm5NextMLAAttention.__init__ keeps the projections it hands the wrapper (MLAModules) as its own
                a.fused_qkv_a_proj, a.q_b_proj, a.o_proj = w.fused_qkv_a_proj, w.q_b_proj, w.o_proj
            dl.self_attn = a
            if Lx < 3:
                dl.mlp = mlp_shell(p + ".mlp", "dense", 6144)
                dl._mlp_is_moe = False
            else:
                moe = gm.Glm5NextMoE.__new__(gm.Glm5NextMoE)
                torch.nn.Module.__init__(moe)
                moe.is_sequence_parallel = False
                moe.gate = Gate()
                moe.shared_experts = mlp_shell(p + ".mlp.shared_experts", "shared", 1024)
                moe.experts = Experts(Lx, moe.shared_experts)
                dl.mlp = moe
                dl._mlp_is_moe = True
            dls.append(dl)
        mdl = gm.Glm5NextModel.__new__(gm.Glm5NextModel)
        torch.nn.Module.__init__(mdl)
        mdl.layers = torch.nn.ModuleList(dls)
        mdl.start_layer, mdl.end_layer = 0, nl
        mdl._active_layers = mdl.layers[0:nl]
        mdl.aux_hidden_state_layers = ()
        mdl.is_sequence_parallel = False
        mdl.norm = Norm(K)
        self.model = mdl
        self.nl = nl
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(self.vcfg):
            self.wout = MG.warm_post_load(mdl)
        self.keep = []                # every wrapper / FULL graph captured in this process (see new_wrapper)
        self.wrapper = self.new_wrapper()
        self.pool = None

    # ---- configuration switches (the installed modules' own state, as an env-only revert leaves it)
    def setcfg(self, roof: bool, warm: bool, triggers=None):
        self.R.STATE.pf = roof
        self.R.STATE.triggers = frozenset(self.R.TRIGGERS if triggers is None else triggers)
        self.MG.STATE.warm = warm

    def counters(self):
        c = {t: self.R.COUNTERS.get("fork_" + t, 0) for t in self.R.TRIGGERS}
        c["warm"] = self.MG.COUNTERS.get("warm_eager", 0) + self.MG.COUNTERS.get("warm_captured", 0)
        return c

    def clear(self):
        self.R.COUNTERS.clear()
        self.MG.COUNTERS.clear()

    def bad(self):
        return ([k for k in BAD_ROOF if k in self.R.COUNTERS] + [k for k in BAD_MG if k in self.MG.COUNTERS] +
                (["roof pending"] if self.R.STATE.pending else []) +
                (["warm pending"] if any(self.MG.WARM.pending.values()) else []))

    def routes_for(self, M):
        key = (M, self.topk)
        if key not in self.routes:
            gen = torch.Generator().manual_seed(1000 * M + self.topk)
            self.routes[key] = {Lx: (torch.stack([torch.randperm(NE, generator=gen)[:self.topk] for _ in range(M)])
                                     .to(dev).to(torch.int32),
                                     torch.rand(M, self.topk, generator=gen).float().to(dev))
                                for Lx in range(3, self.nl)}
        return self.routes[key]

    def inputs(self, M, seed=None):
        gg = self.g if seed is None else torch.Generator(device=dev).manual_seed(seed)
        x = (torch.randn(M, K, device=dev, generator=gg) * 0.5).to(torch.bfloat16)
        return {"input_ids": None, "positions": torch.arange(M, device=dev), "intermediate_tensors": None,
                "inputs_embeds": x}

    def _ctx(self, M, mode):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context
        bd = BatchDescriptor(num_tokens=M) if mode == CUDAGraphMode.PIECEWISE else None
        return set_forward_context(None, self.vcfg, num_tokens=M, cudagraph_runtime_mode=mode, batch_descriptor=bd)

    @torch.inference_mode()
    def eager(self, inp):
        from vllm.config import CUDAGraphMode
        self.routes_for(inp["inputs_embeds"].shape[0])
        with self._ctx(inp["inputs_embeds"].shape[0], CUDAGraphMode.NONE):
            return self.model(**inp)

    @torch.inference_mode()
    def capture_pw(self, inp, wrapper=None, after_warmup=None):
        """vLLM's CudaGraphManager.capture, PIECEWISE with use_breakable_cg: graph_capture stream, eager warm-up (NONE),
        then BreakableCUDAGraphWrapper under mode PIECEWISE. Returns the wrapper's output (weak refs into the pool)."""
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import graph_capture
        M = inp["inputs_embeds"].shape[0]
        self.routes_for(M)
        wrapper = self.wrapper if wrapper is None else wrapper
        with graph_capture(device=torch.device(dev)):
            with self._ctx(M, CUDAGraphMode.NONE):
                self.model(**inp)
            if after_warmup is not None:
                after_warmup()
            with self._ctx(M, CUDAGraphMode.PIECEWISE):
                return wrapper(**inp)

    @torch.inference_mode()
    def replay_pw(self, inp, wrapper=None):
        from vllm.config import CUDAGraphMode
        M = inp["inputs_embeds"].shape[0]
        with self._ctx(M, CUDAGraphMode.PIECEWISE):
            return (self.wrapper if wrapper is None else wrapper)(**inp)

    def new_wrapper(self):
        """A BreakableCUDAGraphWrapper kept alive for the whole process, like production's (every graph shares vLLM's
        global graph pool; destroying all graphs of a pool and capturing into it again trips PyTorch's allocator
        assert, the reason vLLM's profile_memory dry-captures into a throwaway pool)."""
        w = self.BK.BreakableCUDAGraphWrapper(self.model, self.vcfg)
        self.keep.append(w)
        return w

    @torch.inference_mode()
    def capture_full(self, inp, after_warmup=None):
        """vLLM's FULL capture: torch.cuda.graph(graph, pool) with the forward context in mode NONE."""
        from vllm.config import CUDAGraphMode
        from vllm.distributed.parallel_state import graph_capture
        from vllm.platforms import current_platform
        M = inp["inputs_embeds"].shape[0]
        self.routes_for(M)
        if self.pool is None:
            self.pool = current_platform.get_global_graph_pool()
        with graph_capture(device=torch.device(dev)):
            with self._ctx(M, CUDAGraphMode.NONE):
                self.model(**inp)
            if after_warmup is not None:
                after_warmup()
            gr = torch.cuda.CUDAGraph()
            self.keep.append(gr)
            with torch.cuda.graph(gr, self.pool):
                with self._ctx(M, CUDAGraphMode.NONE):
                    out = self.model(**inp)
        return gr, out

    def pw_entry(self, M):
        from vllm.forward_context import BatchDescriptor
        return self.wrapper.entries.get(BatchDescriptor(num_tokens=M))

    def drop_pw(self, M):
        from vllm.forward_context import BatchDescriptor
        self.wrapper.entries.pop(BatchDescriptor(num_tokens=M), None)

    def calibrate_spin(self):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1000)
        e0.record()
        torch.cuda._sleep(2_000_000)
        e1.record()
        torch.cuda.synchronize()
        self.cyc = 2_000_000 / (e0.elapsed_time(e1) * 1000)
        return self.cyc
