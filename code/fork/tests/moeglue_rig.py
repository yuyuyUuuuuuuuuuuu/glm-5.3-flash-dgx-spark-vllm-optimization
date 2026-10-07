"""A production-shaped decode MoE layer on one GB10 (docs/DEC_MOEGLUE.md), for the moeglue bench and tests.

What one call runs, in production's order (vllm/models/glm5next/nvidia/model.py Glm5NextMoE.forward ->
fused_moe/runner/moe_runner.py _forward_impl / _apply_quant_method, shared_experts.py; the launcher overlay exl3.py
Exl3MoEMethod.apply -> apply_exl3_experts):
  main  router GateLinear bf16 [288, 4096] -> fp32 logits (glm53_bf16_gemv kernel, as GLM53_BF16_GEMV=1 + dedup)
  fork  aux.wait_stream(main)                                   (SharedExperts.maybe_sync_shared_experts_stream)
  main  grouped_topk (vLLM _moe_C, sigmoid + bias, renormalize, x2.5) -> fp32 weights, int32 ids
  main  production apply_exl3_experts(x, ids, weights, layer)  (K2 hook, or the moeglue hook under test)
  aux   shared expert: FP8 gate_up [2048, 4096] -> silu_and_mul_with_clamp(10) -> FP8 down [4096, 1024]
        (fp8_gemv kernels with the production TABLE config, as GLM53_FP8_GEMV=1)
  join  main.wait_stream(aux); main: shared_out + routed_out  (bf16 add)
The all-reduce is not run (one GPU). The routing the EXL3 apply sees is taken from pre-drawn ids (corr40 / rand,
tests/harness.py) so the bytes read match production's distinct-expert counts; the topk kernel still runs in the
stream (its result is not used), so the timeline has production's kernels in production's order.
"""
from __future__ import annotations

import statistics

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0
SH_GU, SH_DN = (2048, 4096), (4096, 1024)


class Rig:
    def __init__(self, prod, layers, dev, seed=77):
        import fp8_gemv as F
        import glm53_bf16_gemv as G
        from fp8_bench_common import random_marlin_layers

        self.prod, self.layers, self.dev = prod, layers, dev
        self.F, self.G = F, G
        g = torch.Generator(device=dev).manual_seed(seed)
        self.router = [(torch.randn(NEXP, K, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in layers]
        self.bias = [(torch.randn(NEXP, device=dev, generator=g) * 0.01).float() for _ in layers]
        self.rctx = [G.GemmCtx(NEXP, K, dev, served_only=True) for _ in layers]
        self.sh_gu = random_marlin_layers(*SH_GU, len(layers), dev, seed=seed + 1)
        self.sh_dn = random_marlin_layers(*SH_DN, len(layers), dev, seed=seed + 2)
        self.aux = torch.cuda.Stream(device=dev)
        self.aux_hi = torch.cuda.Stream(device=dev, priority=-5)   # the highest priority CUDA grants (clamped)
        self.aux_prio = False                                       # True: the shared expert runs on aux_hi
        self.side = torch.cuda.Stream(device=dev)                   # L2-warming experiments (touch)
        self.window_us = 0.0          # >0: a spin of this length before each layer call (all-reduce + mhc stand-in)
        self.touch = ""               # subset of "rgd": router / shared gate-up / shared down weights read into the
        self.touch_blocks = 96        #   L2 on `side` during the window (forked before the spin, joined at the end)
        self.touch_el = 0
        self._tout = torch.zeros(4, dtype=torch.int32, device=dev)
        self.warm_prod = False        # True: production's l2_warm kernel (tf_exl3_moe_ext) instead of `touch`
        self.warm_blocks = 16
        self.warm_set = "frgd"
        self.mhc = False              # True: production's mhc_fused_post_pre (tilelang) runs after the spin
        self._mhc = None
        self.fe = F.ext()
        self.ge = G.load_ext()

    def _mhc_state(self, T):
        """Per-layer mHC tensors of production's shapes (hc_mult 4, hidden 4096): fn [24, 16384] fp32 (1.5 MiB,
        distinct per layer so they are cold like production's), residual / mixes per call size T."""
        if self._mhc is None or self._mhc[0] != T:
            dev, g = self.dev, torch.Generator(device=self.dev).manual_seed(991)
            n, H = 4, K
            fns = [torch.randn(24, n * H, device=dev, generator=g) * 0.01 for _ in self.layers]
            scale = torch.tensor([0.5, 0.5, 0.5], device=dev)
            base = torch.randn(24, device=dev, generator=g) * 0.1
            nw = (1 + 0.1 * torch.randn(H, device=dev, generator=g)).to(torch.bfloat16)
            res = (torch.randn(T, n, H, device=dev, generator=g)).to(torch.bfloat16)
            post = torch.rand(T, n, 1, device=dev, generator=g)
            comb = torch.softmax(torch.randn(T, n, n, device=dev, generator=g), -1)
            xa = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
            self._mhc = (T, fns, scale, base, nw, res, post, comb, xa)
        return self._mhc

    def mhc_call(self, li, T):
        from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
        _, fns, scale, base, nw, res, post, comb, xa = self._mhc_state(T)
        return mhc_fused_post_pre_tilelang(xa, res, post, comb, fns[li], scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                                           norm_weight=nw, norm_eps=1e-6)

    def fp8(self, x, lay, nk):
        n, k = nk
        cfg = self.F.select_config(lay.weight.shape[1] // 4, k, x.shape[0])
        assert cfg is not None, (nk, x.shape)
        return self.fe.fp8_gemv(x, lay.weight, lay.weight_scale.view(-1), None, n, k, *cfg)

    def shared(self, li, x, mid_event=None):
        gu = self.fp8(x, self.sh_gu[li], SH_GU)
        if mid_event is not None:                 # act + down only after the routed gate/up GEMV was enqueued
            torch.cuda.current_stream().wait_event(mid_event)
        a = torch.empty(x.shape[0], SH_GU[0] // 2, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul_with_clamp(a, gu, LIMIT, 1.0, 0.0)
        return self.fp8(a, self.sh_dn[li], SH_DN)

    def router_logits(self, li, x):
        M = x.shape[0]
        plan = self.G.serve_plan(NEXP, K, M)
        if plan is None:
            return torch.nn.functional.linear(x, self.router[li]).float()
        return self.G.gemm(x, self.router[li], self.rctx[li], out_mode=2, plan=plan)

    def call(self, li, x, ids32, w, apply=None, prefork=False, sd_late=False):
        """One MoE layer call. apply(x, ids32, w, layer[, mid_event]) -> bf16 routed output (default: production's
        apply_exl3_experts, whatever hooks are installed). prefork: the shared expert forks before the router.
        sd_late: the shared expert's act + down wait for an event the routed apply records after its gate/up GEMV."""
        layer = self.layers[li]
        main = torch.cuda.current_stream()
        aux = self.aux_hi if self.aux_prio else self.aux
        touched = False
        if self.window_us > 0:
            import l2touch
            E = l2touch.ext()
            if self.warm_prod:                  # production's l2_warm (GLM53_DEC_MOEGLUE_WARM defaults: f,r,g,d, 16 x 8)
                import tf_exl3_moe as tfm
                src = {"f": self._mhc_state(x.shape[0])[1][li], "r": self.router[li], "g": self.sh_gu[li].weight,
                       "d": self.sh_dn[li].weight}
                regs = [src[c] for c in self.warm_set]
                self.side.wait_stream(main)
                with torch.cuda.stream(self.side):
                    tfm.load_ext().l2_warm([r.reshape(-1) for r in regs], self.warm_blocks, 8, self._tout)
                touched = True
            elif self.touch:
                self.side.wait_stream(main)
                with torch.cuda.stream(self.side):
                    fn = self._mhc_state(x.shape[0])[1][li] if "f" in self.touch else None
                    for c, t in (("f", fn), ("r", self.router[li]), ("g", self.sh_gu[li].weight),
                                 ("d", self.sh_dn[li].weight)):
                        if c in self.touch:
                            E.touch(t, t.numel() * t.element_size(), self.touch_blocks, self.touch_el, self._tout)
                touched = True
            E.spin(int(self.window_us * 1000))
            if self.mhc:
                self.mhc_call(li, x.shape[0])
        if prefork:
            aux.wait_stream(main)
        logits = self.router_logits(li, x)
        if not prefork:
            aux.wait_stream(main)
        torch.ops._moe_C.grouped_topk(logits, 1, 1, TOPK, True, 2.5, self.bias[li], 1)
        ev = torch.cuda.Event() if sd_late else None
        if apply is None:
            routed = self.prod.apply_exl3_experts(x, ids32, w, layer, limit=LIMIT)
        elif sd_late:
            routed = apply(x, ids32, w, layer, ev)
        else:
            routed = apply(x, ids32, w, layer)
        with torch.cuda.stream(aux):
            so = self.shared(li, x, ev)
        main.wait_stream(aux)
        if touched:
            main.wait_stream(self.side)
        return so + routed


class _SharedTrellis:
    """A layer's EXL3 weights that reuse another layer's trellis (the 6 MiB/expert streamed by the grouped GEMV)
    but have their own suh / svh vectors: production has 42 distinct layers whose per-expert vectors (8.6 MiB per
    layer) are never L2-resident when the next call of that layer comes; with only 3 synthetic layers they would be."""

    def __init__(self, base, seed, dev):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.n, self.K, self.N = base.n, base.K, base.N
        self.w13_trellis, self.w2_trellis = base.w13_trellis, base.w2_trellis
        self.w13_mcg, self.w2_mcg = base.w13_mcg, base.w2_mcg
        perm = lambda t: t[torch.randperm(t.shape[0], generator=torch.Generator().manual_seed(seed)).to(t.device)]
        self.w13_suh, self.w13_svh = perm(base.w13_suh).clone(), perm(base.w13_svh).clone()
        self.w2_suh, self.w2_svh = perm(base.w2_suh).clone(), perm(base.w2_svh).clone()


def make_layers(prod, dev, n_trellis=3, n_layers=42):
    """n_layers production layers over n_trellis distinct trellis sets (layer i uses set i % n_trellis)."""
    bases = [H.Weights(NEXP, K, N, dev, seed=500 + i) for i in range(n_trellis)]
    layers = []
    for i in range(n_layers):
        W = bases[i] if i < n_trellis else _SharedTrellis(bases[i % n_trellis], 9100 + i, dev)
        layers.append(H.make_layer(prod, W))
    return layers


def make_sets(kind, T, nlayers, routes, dev, seed):
    g = torch.Generator().manual_seed(seed)
    sets = []
    for li in range(nlayers):
        for _ in range(routes):
            x = torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)
            ids = H.routing_ids(kind, T, NEXP, TOPK, g, dev).to(torch.int32).contiguous()
            w = H.random_weights(T, TOPK, g, dev).float().contiguous()
            sets.append((li, x, ids, w))
    return sets


def capture(fns, dev, stream=None, before=None):
    """stream: the capture stream (e.g. a high-priority one); before(): called before each warm-up / capture pass."""
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    if before is not None:
        before()
    with torch.cuda.stream(side):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=stream):
        outs = [f() for f in fns]
    torch.cuda.synchronize()
    return gr, outs


def ab_rounds(graphs: dict, calls: int, rounds: int, reps: int):
    """Replay every graph `reps` times per round, rounds interleaved with the order rotated every round.
    Returns {name: [per-call us per round]}."""
    names = list(graphs)
    res = {n: [] for n in names}
    for n in names:
        graphs[n].replay()
    torch.cuda.synchronize()
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:
            order = order[::-1]
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                graphs[n].replay()
            e.record()
            torch.cuda.synchronize()
            res[n].append(s.elapsed_time(e) * 1000.0 / (reps * calls))
    return res


def pct(v, q):
    s_ = sorted(v)
    return s_[min(len(s_) - 1, max(0, int(round(q * (len(s_) - 1)))))]


def med(v):
    return statistics.median(v)


def spread(v):
    m = med(v)
    return (max(v) - min(v)) / m if m else 0.0
