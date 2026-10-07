"""moeglue warm (GLM53_DEC_MOEGLUE_WARM, glm53_moeglue.py, docs/DEC_MOEGLUE.md): after each MoE layer's o_proj GEMV a
side stream reads the MoE sublayer's small weights (hc_ffn_fn, router, shared expert) into the L2; the join is at the
end of that layer's apply_exl3_experts.

  W1  env unset: install inert (no wrapper, no loader hook); the custom op is registered (import time)
  W2  env parsing: on values; invalid WARM_SET / MIB / BLOCKS / MAX_M refused
  W3  l2_warm kernel: writes nothing (every region bitwise unchanged, sink 0); bad inputs raise before launch
  W4  l2_warm brings the bytes into the L2: a second read of a 12 MiB region (after a 96 MiB flush + warm) is faster
      than a cold read
  W5  wiring on real vLLM classes (RowParallelLinear o_proj with production's Glm53DenseFp8Method and with the
      unquantized method; Glm5NextDecoderLayer / Glm5NextMoE instances, Exl3MoEMethod experts): exactly the MoE
      decoder layers are wired (dense-MLP layer, MTP layer and a layer whose experts are not Exl3MoEMethod are not),
      regions f, r, g, d in that order with their byte sizes, the MiB budget cuts the last region, the compile-cache
      tag is set and changes vllm_config.compute_hash()
  W6  o_proj output with warm == production apply bit for bit at M in 1..200 (forks only at M <= MAX_M), for both
      quant methods; the fork leaves pending, the MoE apply joins it
  W7  CUDA graph: 3 x (o_proj -> AR stand-in -> the hooked apply_exl3_experts on a real EXL3 layer) captured (no
      unjoined-stream error), 30 replays == eager bit for bit (o_proj and MoE outputs); the replay runs the warm
      kernel 3 times on a side stream; an o_proj forking twice before its MoE is joined first (counted)
  W8  torch.compile(fullgraph=True) of RowParallelLinear.forward: no graph break, the custom op runs the fork,
      output == eager; the compiled call captured in a CUDA graph with the join replays == eager
  W9  a vLLM function fingerprint mismatch wires nothing (warm disarmed, WARNING); glue + warm installed together
Run: tests/gpu_run.sh python3 tests/test_moeglue_warm.py
"""
from __future__ import annotations

import logging
import os
import socket

import torch

import harness as H

K, N, TOPK = 4096, 1024, 8
LIMIT = 10.0
ENVS = ("GLM53_DEC_MOEGLUE", "GLM53_DEC_MOEGLUE_WARM", "GLM53_DEC_MOEGLUE_WARM_SET", "GLM53_DEC_MOEGLUE_WARM_MIB",
        "GLM53_DEC_MOEGLUE_WARM_BLOCKS", "GLM53_DEC_MOEGLUE_WARM_MAX_M")


class Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, r):
        self.records.append((r.levelno, r.getMessage()))

    def has(self, level, text):
        return any(lv == level and text in m for lv, m in self.records)


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


def main():
    H.gpu_guard(8.0)
    for v in ENVS:
        os.environ.pop(v, None)
    single_rank_tp()
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    E = tf.load_ext()
    import glm53_moeglue as MG
    import integrate
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.models.glm5next.nvidia.model import Glm5NextDecoderLayer, Glm5NextMoE

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    cap = Cap()
    logging.getLogger("vllm.glm53_moeglue").addHandler(cap)
    g = torch.Generator(device=dev).manual_seed(7)

    # ---- W1 inert ---------------------------------------------------------------------------------------------------
    orig_apply = prod.apply_exl3_experts
    rep = MG.install(prodmod=prod)
    ck(not rep["installed"] and not rep["warm"] and prod.apply_exl3_experts is orig_apply and not MG.STATE.loader_hooked,
       f"W1 env unset not inert: {rep}")
    ck(hasattr(torch.ops.glm53_moeglue, "linear_warm"), "W1 custom op not registered")
    print(f"W1 env unset: install inert ({rep['reason']}; {rep['warm_reason']}); op registered")

    # ---- W2 env parsing ---------------------------------------------------------------------------------------------
    ck(all(MG.env_warm({"GLM53_DEC_MOEGLUE_WARM": v}) for v in ("1", "on", "true", "yes", " ON ")), "W2 on values")
    ck(not any(MG.env_warm({"GLM53_DEC_MOEGLUE_WARM": v}) for v in ("", "0", "off", "2")), "W2 off values")
    bads = [{"GLM53_DEC_MOEGLUE_WARM_SET": "frgx"}, {"GLM53_DEC_MOEGLUE_WARM_SET": "ff"},
            {"GLM53_DEC_MOEGLUE_WARM_MIB": "0"}, {"GLM53_DEC_MOEGLUE_WARM_MIB": "x"},
            {"GLM53_DEC_MOEGLUE_WARM_BLOCKS": "0"}, {"GLM53_DEC_MOEGLUE_WARM_MAX_M": "1000"}]
    for b in bads:
        ck(MG.warm_cfg_from_env(b) is not None, f"W2 {b} accepted")
    ck(MG.warm_cfg_from_env({}) is None and (MG.CFG.warm_set, MG.CFG.warm_mib, MG.CFG.warm_blocks, MG.CFG.warm_max_m)
       == ("frgd", 16.0, 16, 64), "W2 defaults")
    print(f"W2 env parsing: {len(bads)} invalid settings refused; defaults frgd / 16 MiB / 16 blocks / M <= 64")

    # ---- W3 kernel writes nothing, input checks -----------------------------------------------------------------
    regs = [torch.randn(24, 16384, device=dev, generator=g),
            torch.randn(288, 4096, device=dev, generator=g).to(torch.bfloat16),
            torch.randint(-2**31, 2**31 - 1, (256, 8192), device=dev, generator=g, dtype=torch.int32),
            torch.randint(0, 255, (4 << 20,), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)]
    snap = [r.clone() for r in regs]
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    for unroll in (1, 4, 8):
        for blocks in (1, 16, 96):
            E.l2_warm(regs, blocks, unroll, sink)
    torch.cuda.synchronize()
    ck(all(torch.equal(a, b) for a, b in zip(regs, snap)) and int(sink.abs().sum()) == 0, "W3 kernel wrote")
    bad_calls = {"7 regions": lambda: E.l2_warm([regs[1]] * 7, 16, 8, sink),
                 "no regions": lambda: E.l2_warm([], 16, 8, sink),
                 "unaligned": lambda: E.l2_warm([regs[3][1:]], 16, 8, sink),
                 "non-contiguous": lambda: E.l2_warm([regs[1].t()], 16, 8, sink),
                 "cpu region": lambda: E.l2_warm([regs[1].cpu()], 16, 8, sink),
                 "unroll 3": lambda: E.l2_warm(regs, 16, 3, sink),
                 "blocks 0": lambda: E.l2_warm(regs, 0, 8, sink),
                 "sink int64": lambda: E.l2_warm(regs, 16, 8, sink.long())}
    for nm, fn in bad_calls.items():
        try:
            fn()
            ck(False, f"W3 {nm} accepted")
        except RuntimeError:
            pass
    torch.cuda.synchronize()
    print(f"W3 l2_warm: 4 regions ({sum(r.numel() * r.element_size() for r in regs) / 2**20:.2f} MiB) x 9 launch "
          f"configs wrote nothing; {len(bad_calls)} bad inputs rejected")

    # ---- W4 the bytes land in the L2 ----------------------------------------------------------------------------
    reg = torch.randint(0, 2**31 - 1, (3 << 20,), dtype=torch.int32, device=dev)     # 12 MiB
    flush = torch.randint(0, 2**31 - 1, (24 << 20,), dtype=torch.int32, device=dev)  # 96 MiB
    cold, warm = [], []
    for it in range(21):
        for mode in (("cold", "warm") if it % 2 else ("warm", "cold")):
            E.l2_warm([flush], 96, 8, sink)
            if mode == "warm":
                E.l2_warm([reg], 16, 8, sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            E.l2_warm([reg], 96, 8, sink)
            e.record()
            torch.cuda.synchronize()
            (cold if mode == "cold" else warm).append(s.elapsed_time(e) * 1000)
    mc, mw = MR_med(cold), MR_med(warm)
    ck(mw < 0.7 * mc, f"W4 warmed read {mw:.1f} us not faster than cold {mc:.1f} us")
    print(f"W4 12 MiB read: cold {mc:.1f} us ({12 * 2**20 / mc / 1e3:.0f} GB/s), after l2_warm {mw:.1f} us "
          f"({12 * 2**20 / mw / 1e3:.0f} GB/s)")
    del reg, flush

    # ---- W5 wiring ------------------------------------------------------------------------------------------------
    integrate.install(prodmod=prod, ext=xl, force=True)
    import fp8_gemv as F
    F.install(prod, force=True)                            # production's decode FP8 path (GLM53_FP8_GEMV=1)
    os.environ["GLM53_DEC_MOEGLUE_WARM"] = "1"
    rep = MG.install(prodmod=prod, hook_loader=False)
    ck(rep["warm"] and not rep["installed"] and MG.STATE.warm and not MG.STATE.enabled and
       hasattr(prod.apply_exl3_experts, "_glm53_moeglue_orig"), f"W5 warm-only install: {rep}")
    MG.CFG.strict = True
    fp8_cls = prod.Glm53DenseFp8Method
    exl3_moe_cls = prod.Exl3MoEMethod

    def lin(nm, w):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(w, requires_grad=False)
        return m

    def make_layer(i, kind="fp8", moe=True, mtp=False, exl3=True):
        dl = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
        torch.nn.Module.__init__(dl)
        dl._mlp_is_moe, dl.is_mtp_layer, dl.mhc, dl.layer_idx = moe, mtp, True, i
        dl.hc_ffn_fn = torch.nn.Parameter(torch.randn(24, 4 * K, device=dev, generator=g) * 0.01, requires_grad=False)
        attn = torch.nn.Module()
        with torch.device(dev):
            op = RowParallelLinear(K, K, bias=False, params_dtype=torch.bfloat16,
                                   prefix=f"model.layers.{i}.self_attn.o_proj")
        op.weight.data.copy_((torch.randn(K, K, device=dev, generator=g) * 0.02).to(torch.bfloat16))
        if kind == "fp8":
            op.quant_method = fp8_cls("kda", f"model.layers.{i}.self_attn.o_proj")
            op.quant_method.process_weights_after_loading(op)
        attn.o_proj = op
        dl.self_attn = attn
        m = Glm5NextMoE.__new__(Glm5NextMoE)
        torch.nn.Module.__init__(m)
        m.gate = lin("gate", (torch.randn(288, K, device=dev, generator=g) * 0.02).to(torch.bfloat16))
        sh = torch.nn.Module()
        sh.gate_up_proj = lin("gu", torch.randint(-2**31, 2**31 - 1, (K // 16, 2048 * 16 // 4 * 4 // 4), device=dev,
                                                  generator=g, dtype=torch.int32))
        sh.down_proj = lin("dn", torch.randint(-2**31, 2**31 - 1, (1024 // 16, 4096 * 16 // 4 * 4 // 4), device=dev,
                                               generator=g, dtype=torch.int32))
        m.shared_experts = sh
        ex = torch.nn.Module()
        inner = torch.nn.Module()
        inner.quant_method = exl3_moe_cls.__new__(exl3_moe_cls) if exl3 else object()
        ex.routed = inner
        m.experts = ex
        dl.mlp = m
        return dl

    class Model(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

    layers = [make_layer(0, "fp8"), make_layer(1, "unq"), make_layer(2, "fp8"), make_layer(3, "fp8", moe=False),
              make_layer(4, "fp8", mtp=True), make_layer(5, "fp8", exl3=False)]
    base_cls = {i: type(l.self_attn.o_proj.quant_method) for i, l in enumerate(layers)}
    model = Model(layers)
    vc = VllmConfig()
    h0 = vc.compute_hash()
    with set_current_vllm_config(vc):
        out = MG.warm_post_load(model)
    ck(out["wired"] == 3 and len(out["rejected"]) == 1 and "layers.5" in out["rejected"][0],
       f"W5 wired {out}")
    ck(isinstance(vc.additional_config, dict) and str(vc.additional_config.get("glm53_moeglue", "")).startswith(
        "warm:v1:frgd") and vc.compute_hash() != h0, f"W5 compile-cache tag {vc.additional_config}")
    wired = [0, 1, 2]
    for i in wired:
        op = layers[i].self_attn.o_proj
        h = MG.WARM.handles[op._glm53_warm_handle]
        want = [layers[i].hc_ffn_fn, layers[i].mlp.gate.weight, layers[i].mlp.shared_experts.gate_up_proj.weight,
                layers[i].mlp.shared_experts.down_proj.weight]
        ok = len(h.regions) == 4 and all(r.data_ptr() == w.data_ptr() and r.numel() == w.numel()
                                         for r, w in zip(h.regions, want))
        ck(ok and type(op.quant_method).__name__.startswith("Glm53Warm") and
           type(op.quant_method)._glm53_warm_base is base_cls[i], f"W5 layer {i} regions / class")
    for i in (3, 4, 5):
        ck(type(layers[i].self_attn.o_proj.quant_method) is base_cls[i], f"W5 layer {i} wired")
    nb = MG.WARM.handles[0].nbytes
    base_buf = torch.randn(24 * 4 * K + 3, device=dev, generator=g)          # hc_ffn_fn as an unaligned view
    fn_view = base_buf[3:].view(24, 4 * K)
    save_fn = layers[0].hc_ffn_fn
    layers[0].hc_ffn_fn = torch.nn.Parameter(fn_view, requires_grad=False)
    ua = MG._warm_regions(layers[0], layers[0].mlp, "f", int(16 * 2**20))
    ck(len(ua) == 1 and ua[0].data_ptr() % 16 == 0 and ua[0].data_ptr() - fn_view.data_ptr() == 4 and
       ua[0].numel() == fn_view.numel() - 1, "W5 unaligned region not reduced to its aligned interior")
    layers[0].hc_ffn_fn = save_fn
    MG.CFG.warm_mib = 10.0
    cut = MG._warm_regions(layers[0], layers[0].mlp, "frgd", int(10 * 2**20))
    ck(sum(t.numel() * t.element_size() for t in cut) == 10 * 2**20 and len(cut) == 3, "W5 budget cut")
    MG.CFG.warm_mib = 16.0
    print(f"W5 wired layers 0-2 (fp8 / unquantized / fp8 o_proj), {nb / 2**20:.2f} MiB f,r,g,d each; not wired: dense "
          f"MLP, MTP, {out['rejected'][0].split(': ', 1)[1]}; 10 MiB budget -> 3 regions; tag "
          f"{vc.additional_config['glm53_moeglue']}")

    # ---- W6 o_proj output bitwise, fork only at M <= MAX_M, join --------------------------------------------------
    worst = 0
    for i in wired:
        op = layers[i].self_attn.o_proj
        base = base_cls[i]
        for M in (1, 5, 8, 16, 64, 65, 200):
            x = (torch.randn(M, K, device=dev, generator=g)).to(torch.bfloat16)
            c0 = dict(MG.COUNTERS)
            y = op(x)[0]
            yo = base.apply(op.quant_method, op, x, None)
            forked = MG.COUNTERS.get("warm_eager", 0) - c0.get("warm_eager", 0)
            ck(torch.equal(y, yo), f"W6 layer {i} M={M}: o_proj output differs")
            ck(forked == (1 if M <= 64 else 0) and bool(MG.WARM.pending.get(0)) == (M <= 64),
               f"W6 layer {i} M={M}: forked {forked}, pending {MG.WARM.pending}")
            MG.warm_join()
            ck(not any(MG.WARM.pending.values()), "W6 join left pending")
            worst += 1
    torch.cuda.synchronize()
    print(f"W6 {worst} o_proj calls bitwise == production apply; forks exactly at M <= 64; joins clear pending")

    # ---- W7 CUDA graph -----------------------------------------------------------------------------------------------
    Wt = H.Weights(32, K, N, dev, seed=43)
    tlayer = H.make_layer(prod, Wt)
    gen = torch.Generator().manual_seed(5)
    T = 5
    xs = [(torch.randn(T, K, generator=gen) * 0.5).to(torch.bfloat16).to(dev) for _ in wired]
    ids = [torch.stack([torch.randperm(32, generator=gen)[:1] for _ in range(T)]).to(torch.int64).to(dev)
           for _ in wired]
    ws = [torch.rand(T, 1, generator=gen).float().to(dev) for _ in wired]
    mm_b = (torch.randn(K, 256, device=dev, generator=g) * 0.02).to(torch.bfloat16)

    def step(double_fork=False):
        outs = []
        for j, i in enumerate(wired):
            op = layers[i].self_attn.o_proj
            y = op(xs[j])[0]
            if double_fork and j == 1:
                y = op(xs[j])[0]
            z = (y[:, :256] @ mm_b[:256]).float().sum()          # AR / mHC stand-in on the main stream
            r = prod.apply_exl3_experts(y, ids[j], ws[j], tlayer, limit=LIMIT)
            outs.append((y, z, r))
        return outs

    ref = step()
    torch.cuda.synchronize()
    ck(not any(MG.WARM.pending.values()), "W7 eager step left a fork pending")
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    c0 = dict(MG.COUNTERS)
    with torch.cuda.graph(gr):
        gout = step()
    ck(MG.COUNTERS.get("warm_captured", 0) - c0.get("warm_captured", 0) == 3, "W7 capture did not fork 3 times")
    same = True
    for _ in range(30):
        gr.replay()
        torch.cuda.synchronize()
        for (a, b, c), (a0, b0, c0_) in zip(gout, ref):
            same &= torch.equal(a, a0) and torch.equal(b, b0) and torch.equal(c, c0_)
    ck(same, "W7 graph replays != eager")
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gr.replay()
        torch.cuda.synchronize()
    wk = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "l2_warm_kernel" in e.name]
    ck(len(wk) == 3, f"W7 warm kernels in one replay: {len(wk)}")
    c0 = dict(MG.COUNTERS)
    gr2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr2):
        gout2 = step(double_fork=True)
    gr2.replay()
    torch.cuda.synchronize()
    ck(MG.COUNTERS.get("warm_unjoined", 0) - c0.get("warm_unjoined", 0) == 1 and
       all(torch.equal(a[2], b[2]) for a, b in zip(gout2, ref)), "W7 double fork")
    del gr, gr2
    print(f"W7 graph: 3 x (o_proj fork -> stand-in -> MoE join) captured, 30 replays bitwise == eager, "
          f"{len(wk)} warm kernels per replay; a double fork is joined first (counted) and still captures")

    # ---- W8 torch.compile ------------------------------------------------------------------------------------------
    op = layers[0].self_attn.o_proj
    x = (torch.randn(T, K, device=dev, generator=g)).to(torch.bfloat16)
    yo = base_cls[0].apply(op.quant_method, op, x, None)
    torch._dynamo.reset()
    f = torch.compile(lambda t: op(t)[0], fullgraph=True, dynamic=False)
    c0 = dict(MG.COUNTERS)
    y = f(x)
    MG.warm_join()
    torch.cuda.synchronize()
    ck(torch.equal(y, yo) and MG.COUNTERS.get("warm_eager", 0) - c0.get("warm_eager", 0) == 1,
       "W8 compiled o_proj: output / fork")

    def cstep():
        yy = f(x)
        MG.warm_join()
        return yy
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        cstep()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gr3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr3):
        yg = cstep()
    gr3.replay()
    torch.cuda.synchronize()
    ck(torch.equal(yg, yo), "W8 compiled + captured != eager")
    del gr3
    print("W8 torch.compile(fullgraph=True): no graph break, fork inside the op, output bitwise; captured replay ==")

    # ---- W9 fingerprint mismatch; glue + warm together ------------------------------------------------------------
    saved = dict(MG.WARM_VERIFIED)
    MG.WARM_VERIFIED["Glm5NextDecoderLayer.forward"] = frozenset({"0000000000000000"})
    model2 = Model([make_layer(10, "fp8")])
    b2 = type(model2.layers[0].self_attn.o_proj.quant_method)
    with set_current_vllm_config(VllmConfig()):
        out2 = MG.warm_post_load(model2)
    ck(out2["wired"] == 0 and not MG.STATE.warm and type(model2.layers[0].self_attn.o_proj.quant_method) is b2 and
       cap.has(logging.WARNING, "warm NOT wired"), f"W9 mismatch wired {out2}")
    MG.WARM_VERIFIED.clear()
    MG.WARM_VERIFIED.update(saved)
    y = layers[0].self_attn.o_proj(xs[0])[0]               # disarmed: wired layers run production's apply, no fork
    ck(not any(MG.WARM.pending.values()) and torch.equal(y, ref[0][0]), "W9 disarmed wired layer forked / differs")
    os.environ["GLM53_DEC_MOEGLUE"] = "1"
    rep = MG.install(prodmod=prod, hook_loader=False)
    ck(rep["installed"] and rep["warm"] and MG.STATE.enabled and MG.STATE.warm, f"W9 glue + warm: {rep}")
    for v in ENVS:
        os.environ.pop(v, None)
    print("W9 fingerprint mismatch: nothing wired, warm disarmed (wired layers fall back to production, no fork); "
          "glue + warm install together")
    MG.uninstall(prod)
    print(f"counters: {MG.summary()}")
    H.report_peak(8.0)
    ck.summary()


def MR_med(v):
    s = sorted(v)
    return s[len(s) // 2]


if __name__ == "__main__":
    H.run_main(main)
