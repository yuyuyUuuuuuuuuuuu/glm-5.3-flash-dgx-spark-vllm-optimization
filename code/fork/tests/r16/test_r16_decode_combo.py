"""deploy-r16 decode interaction: GLM53_DEC_FP8ROOF (L2 prefetch t1..t4 + table) and GLM53_DEC_MOEGLUE_WARM (L2 warm
of the MoE sublayer) on the same o_proj GEMVs and the same MoE apply, with production's other decode hooks installed
in plugin order: TF (integrate.install, K2 apply hook), moeglue warm, fp8_gemv (GLM53_FP8_GEMV=1, MAX_M 16), fp8_roof,
and GLM53_PREFILL_QUICKWINS item mhc_mean (it recompiles Glm5NextDecoderLayer.forward, which warm verifies).
Run with the launcher overlay exl3.py bound (GPU_RUN_BIND) through tests/r16/gpu.sh.

Mini model = test_fp8_roof R.8's production-structured stack (layer 2 KDA + dense MLP, 3 MLA + MoE, 4/5 KDA + MoE; every
FP8 linear a production Glm53DenseFp8Method at its real shape and prefix, shared expert on an aux stream), with
  - each MoE layer's routed experts = a real EXL3 layer (tests/harness.make_layer, 32 experts, topk 8) served through
    production's Exl3MoEMethod.apply (fp8roof's t3 hook) -> apply_exl3_experts (moeglue's warm join) -> TF K2
  - each MoE layer's o_proj = a RowParallelLinear with production's Glm53DenseFp8Method, wired by moeglue's own
    warm_post_load (Glm5NextDecoderLayer / Glm5NextMoE shells like tests/test_moeglue_warm.py W5)

  Y.1 installs: all armed; warm_post_load wires the 3 MoE o_proj layers WITH quickwins' recompiled DecoderLayer.forward
  Y.2 eager M = 5, configs off / roof / warm / both: outputs bitwise equal; forks per forward: roof t1 3 t2 4 t3 3 t4 1,
      warm 3; every fork joined (no safety join, no stale drop, no unjoined)
  Y.3 CUDA graphs M = 1, 5, 8, 16, 64 with both on: capture ok, forks as in Y.2 (x2: warm-up + capture), replay ==
      eager (both off) bitwise with fresh inputs; counters frozen during replay; M = 65: no fork
      (Y.2 / Y.3 route 1 expert per token: production's K2 sums a token's routes with fp32 atomics in arrival order,
      which is not run-to-run deterministic with 8 routes (docs/DEC_MOEGLUE.md 10), so bitwise checks need 1 route)
  Y.4 timing on nodeC (indicative only: no RoCE all-reduce; the latency-bound windows are clock spins at the R15
      trace's medians, as tests/roof/sim_step.py's 'spin' mode: KDA conv + recurrent 31 us, AR 23 us + mHC reading
      hc_*_fn, MLA attention 150 us), topk 8: per-forward CUDA-graph replay time of off / off (null repeat) / roof /
      warm / both / both without t2 (the trigger that prefetches the shared-expert bytes warm also reads) / both with
      warm set "fr" (warm reads only hc_ffn_fn + router, t2 the shared expert), interleaved rounds, medians and
      paired ratios vs off
"""
from __future__ import annotations

import logging
import os
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import harness as H  # noqa: E402
import torch  # noqa: E402

dev = "cuda"
K, N, NE, TOPK = 4096, 1024, 32, 8
LIMIT = 10.0


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelno, r.name, r.getMessage()))


class L(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.bias = None


def single_rank_tp():
    import socket

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
    ck = H.Checks()
    for v in [k for k in os.environ if k.startswith(("GLM53_", "TF_EXL3_"))]:
        if v != "TF_EXL3_JIT":
            os.environ.pop(v)
    single_rank_tp()
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    tf.load_ext()
    import fp8_gemv as F
    import fp8_roof as R
    import glm53_moeglue as MG
    import glm53_prefill_quickwins as Q
    import integrate
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.models.glm5next.nvidia.model import Glm5NextDecoderLayer, Glm5NextMoE
    cap = Cap()
    for nm in ("vllm.tf_fp8_roof", "vllm.tf_fp8_gemv", "vllm.glm53_moeglue", "vllm.glm53_prefill_quickwins"):
        logging.getLogger(nm).addHandler(cap)

    # ---- Y.1 installs, in integrate.plugin_register's order
    integrate.install(prodmod=prod, ext=xl, force=True)                          # TF (TF_EXL3_MOE=1)
    os.environ["GLM53_DEC_MOEGLUE_WARM"] = "1"
    mrep = MG.install(prodmod=prod, hook_loader=False)
    Q.install(["mhc_mean"])                                                       # recompiles DecoderLayer.forward
    os.environ.update(GLM53_FP8_GEMV="1", GLM53_FP8_GEMV_MAX_M="16", GLM53_DEC_FP8ROOF="1")
    frep = F.install(prod)
    rrep = R.install(prod)
    ck(mrep["warm"] and frep["installed"] and rrep["installed"] and rrep["pf"] and rrep["table"],
       f"Y.1 installs: moeglue {mrep} fp8_gemv {frep.get('installed')} roof {rrep}")
    ck(getattr(Glm5NextDecoderLayer.forward, "_glm53_qw", False), "Y.1 quickwins recompiled DecoderLayer.forward")
    ck(getattr(prod.Exl3MoEMethod.apply, "_tf_fp8_roof_hook", False) and
       hasattr(prod.apply_exl3_experts, "_glm53_moeglue_orig"), "Y.1 roof MoE hook + moeglue join hook")
    MG.CFG.strict = True

    g = torch.Generator(device=dev).manual_seed(0)
    cls = prod.Glm53DenseFp8Method

    def build(n, k, group, prefix):
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
             torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        lay, m = L(w), cls(group, prefix)
        m.process_weights_after_loading(lay)
        return lay, m

    lin_ = {}
    for Lx in (2, 4, 5):
        p = f"model.layers.{Lx}.self_attn."
        lin_[(Lx, "in")] = build(12576, 4096, "kda", p + "in_proj_qkvbfg_a")
        lin_[(Lx, "fb")] = build(4096, 128, "kda", p + "f_b_proj")
        lin_[(Lx, "gb")] = build(4096, 128, "kda", p + "g_b_proj")
    lin_[(2, "o")] = build(4096, 4096, "kda", "model.layers.2.self_attn.o_proj")
    p = "model.layers.3.self_attn."
    lin_[(3, "qkv_a")] = build(2048, 4096, "mla", p + "fused_qkv_a_proj")
    lin_[(3, "q_b")] = build(8192, 1536, "mla", p + "q_b_proj")
    lin_[(2, "gu")] = build(12288, 4096, "dense", "model.layers.2.mlp.gate_up_proj")
    lin_[(2, "dn")] = build(4096, 6144, "dense", "model.layers.2.mlp.down_proj")
    for Lx in (3, 4, 5):
        lin_[(Lx, "sgu")] = build(2048, 4096, "shared", f"model.layers.{Lx}.mlp.shared_experts.gate_up_proj")
        lin_[(Lx, "sdn")] = build(4096, 1024, "shared", f"model.layers.{Lx}.mlp.shared_experts.down_proj")

    # MoE layers 3, 4, 5: o_proj RowParallelLinear (moeglue wires it), real EXL3 experts, shells for warm_post_load
    dls, oprojs, tlayers, gates = {}, {}, {}, {}
    em = prod.Exl3MoEMethod.__new__(prod.Exl3MoEMethod)
    em.moe = types.SimpleNamespace(swiglu_limit=LIMIT)
    for Lx, (group, kin) in ((3, ("mla", 8192)), (4, ("kda", 4096)), (5, ("kda", 4096))):
        with torch.device(dev):
            op = RowParallelLinear(kin, K, bias=False, params_dtype=torch.bfloat16,
                                   prefix=f"model.layers.{Lx}.self_attn.o_proj")
        op.weight.data.copy_((torch.randn(K, kin, device=dev, generator=g) * 0.02).to(torch.bfloat16))
        op.quant_method = cls(group, f"model.layers.{Lx}.self_attn.o_proj")
        op.quant_method.process_weights_after_loading(op)
        oprojs[Lx] = op
        tl = H.make_layer(prod, H.Weights(NE, K, N, dev, seed=40 + Lx))
        tl.layer_name = f"model.layers.{Lx}.mlp.experts"
        tlayers[Lx] = tl
        gates[Lx] = (torch.randn(288, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        dl = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
        torch.nn.Module.__init__(dl)
        dl._mlp_is_moe, dl.is_mtp_layer, dl.mhc, dl.layer_idx = True, False, True, Lx
        dl.hc_ffn_fn = torch.nn.Parameter(torch.randn(24, 4 * K, device=dev, generator=g) * 0.01, requires_grad=False)
        attn = torch.nn.Module()
        attn.o_proj = op
        dl.self_attn = attn
        moe = Glm5NextMoE.__new__(Glm5NextMoE)
        torch.nn.Module.__init__(moe)
        moe.gate = L(gates[Lx])
        sh = torch.nn.Module()
        sh.gate_up_proj = lin_[(Lx, "sgu")][0]
        sh.down_proj = lin_[(Lx, "sdn")][0]
        moe.shared_experts = sh
        ex = torch.nn.Module()
        inner = torch.nn.Module()
        inner.quant_method = em
        ex.routed = inner
        moe.experts = ex
        dl.mlp = moe
        dls[Lx] = dl

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([dls[3], dls[4], dls[5]])

    with set_current_vllm_config(VllmConfig()):
        wout = MG.warm_post_load(Model())
    ck(wout["wired"] == 3 and not wout["rejected"] and MG.STATE.warm, f"Y.1 warm wired with quickwins on: {wout} "
       f"({MG.STATE.warm_reason})")
    reg = sorted(R.STATE.reg)
    ck(("kda_o", 4) in reg and ("mla_o", 3) in reg and ("sh_gu", 5) in reg and ("kda_in", 5) in reg,
       f"Y.1 roof registry {reg}")
    print(f"Y.1 installed: roof {rrep['pf']}/{rrep['table']}, warm wired {wout['wired']} ({wout['bytes'] / 2**20:.2f} "
          f"MiB each), roof registry {len(reg)} weights", flush=True)

    aux = torch.cuda.Stream()
    gen = torch.Generator().manual_seed(5)
    route = {}
    topk = [1]
    cyc = [0.0]                                        # clock cycles per us (torch.cuda._sleep), calibrated below

    def routing(M):
        key = (M, topk[0])
        if key not in route:
            route[key] = {Lx: (torch.stack([torch.randperm(NE, generator=gen)[:topk[0]] for _ in range(M)]).to(dev),
                               torch.rand(M, topk[0], generator=gen).float().to(dev)) for Lx in (3, 4, 5)}
        return route[key]

    def spin(us):
        if cyc[0] > 0:
            torch.cuda._sleep(int(us * cyc[0]))

    def lin(key, h):
        lay, m = lin_[key]
        return m.apply(lay, h)

    def window(Lx, y):                                 # all-reduce (spin) + mHC stand-in reading hc_ffn_fn (1.5 MiB)
        spin(23)
        return (y.float() @ dls[Lx].hc_ffn_fn.view(-1, K)[:, :K].t()).sum(-1, keepdim=True).to(y.dtype) * 0

    def moe_block(Lx, h):
        ids, ws = routing(h.shape[0])[Lx]
        _ = torch.nn.functional.linear(h, gates[Lx]).float().softmax(-1)          # router GEMV (2.25 MiB)
        e = torch.cuda.Event()
        e.record()
        aux.wait_event(e)
        routed = em.apply(tlayers[Lx], h, ws, ids, None, None)                    # roof t3 / moeglue join / TF K2
        with torch.cuda.stream(aux):
            su = lin((Lx, "sgu"), h)
            sh = lin((Lx, "sdn"), torch.nn.functional.silu(su[:, :1024]) * su[:, 1024:])
        torch.cuda.current_stream().wait_stream(aux)
        spin(23)                                                                  # the MoE's all-reduce
        return routed + sh

    def kda(Lx, h):
        y = lin((Lx, "in"), h)
        fb = lin((Lx, "fb"), y[:, 12320:12448])
        gb = lin((Lx, "gb"), y[:, 12448:12576])
        spin(31)                                                                  # conv + recurrent
        core = (y[:, :4096].float() * torch.sigmoid(fb.float()) + gb.float() * 0.01).to(torch.bfloat16)
        if Lx == 2:
            return lin((2, "o"), core)
        o = oprojs[Lx](core)[0]
        return o + window(Lx, o)

    def mla(h):
        a = lin((3, "qkv_a"), h)
        q = lin((3, "q_b"), a[:, :1536])
        spin(150)                                                                 # sparse-MLA attention
        o = oprojs[3](q * 0.05)[0]
        return o + window(3, o)

    def forward(h):
        h = h + kda(2, h)
        u = lin((2, "gu"), h)
        h = h + lin((2, "dn"), torch.nn.functional.silu(u[:, :6144]) * u[:, 6144:])
        h = h + mla(h)
        h = h + moe_block(3, h)
        for Lx in (4, 5):
            h = h + kda(Lx, h)
            h = h + moe_block(Lx, h)
        return h

    def setcfg(roof, warm, triggers=None):
        R.STATE.pf = roof
        R.STATE.triggers = frozenset(R.TRIGGERS if triggers is None else triggers)
        MG.STATE.warm = warm

    def counters():
        c = {t: R.COUNTERS.get("fork_" + t, 0) for t in R.TRIGGERS}
        c["warm"] = MG.COUNTERS.get("warm_eager", 0) + MG.COUNTERS.get("warm_captured", 0)
        return c

    bad_keys = ("joined_late_safety", "stale_dropped", "dropped_capture_mismatch", "double_fork")
    bad_mg = ("warm_unjoined", "warm_stale_dropped", "warm_error", "warm_no_stream")
    want = {"t0": 0, "t1": 3, "t2": 4, "t3": 3, "t4": 1, "t5": 0, "warm": 3}
    CFGS = {"off": (False, False), "roof": (True, False), "warm": (False, True), "both": (True, True)}

    # ---- Y.2 eager
    M = 5
    xin = (torch.randn(M, K, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    outs, cnts = {}, {}
    for name, (ro, wa) in CFGS.items():
        setcfg(ro, wa)
        R.COUNTERS.clear()
        MG.COUNTERS.clear()
        outs[name] = forward(xin)
        torch.cuda.synchronize()
        cnts[name] = counters()
        exp = {k: (v if (k == "warm" and wa) or (k != "warm" and ro) else 0) for k, v in want.items()}
        ok = cnts[name] == exp and not R.STATE.pending and not any(MG.WARM.pending.values()) and \
            not any(k in R.COUNTERS for k in bad_keys) and not any(k in MG.COUNTERS for k in bad_mg)
        ck(ok, f"Y.2 {name}: forks {cnts[name]} (want {exp}); roof {dict(R.COUNTERS)} warm {dict(MG.COUNTERS)}")
    same = all(torch.equal(outs[n], outs["off"]) for n in CFGS)
    ck(same, "Y.2 eager outputs differ between off / roof / warm / both")
    print(f"Y.2 eager M=5: off/roof/warm/both bitwise equal {same}; forks both {cnts['both']}", flush=True)

    # ---- Y.3 graphs, both on
    def capture(fn, h):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(h)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn(h)
        return gr, out

    graphs = {}
    for M in (1, 5, 8, 16, 64):
        setcfg(True, True)
        R.COUNTERS.clear()
        MG.COUNTERS.clear()
        xin = (torch.randn(M, K, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        try:
            gr, yg = capture(forward, xin)
            ok_cap = True
        except Exception as exc:  # noqa: BLE001
            ok_cap = False
            print(f"   M={M}: capture FAILED: {exc!r}")
        ck(ok_cap, f"Y.3 M={M}: capture failed")
        if not ok_cap:
            continue
        c = counters()
        ok_c = c == {k: 2 * v for k, v in want.items()} and not R.STATE.pending and \
            not any(MG.WARM.pending.values()) and not any(k in R.COUNTERS for k in bad_keys) and \
            not any(k in MG.COUNTERS for k in bad_mg)
        ck(ok_c, f"Y.3 M={M}: forks {c}; roof {dict(R.COUNTERS)} warm {dict(MG.COUNTERS)}")
        c_cap = (dict(R.COUNTERS), dict(MG.COUNTERS))
        same = True
        for _ in range(3):
            xin.copy_((torch.randn(M, K, device=dev, generator=g) * 0.5).to(torch.bfloat16))
            gr.replay()
            setcfg(False, False)
            ye = forward(xin)
            setcfg(True, True)
            torch.cuda.synchronize()
            same &= torch.equal(yg, ye)
        ck(same, f"Y.3 M={M}: replay != eager (both off)")
        ck((dict(R.COUNTERS), dict(MG.COUNTERS)) == c_cap, f"Y.3 M={M}: counters moved during replay")
        print(f"Y.3 M={M:2d}: capture ok, forks per forward {({k: v // 2 for k, v in c.items()})}, replay == eager "
              f"(off) bitwise x3 {same}", flush=True)
        if M == 5:
            graphs["both"] = (gr, xin, yg)
        else:
            del gr
    setcfg(True, True)
    R.COUNTERS.clear()
    MG.COUNTERS.clear()
    forward((torch.randn(65, K, device=dev, generator=g) * 0.5).to(torch.bfloat16))
    torch.cuda.synchronize()
    c = counters()
    ck(sum(c.values()) == 0, f"Y.3 M=65 forked: {c}")

    # ---- Y.4 timing (nodeC, indicative)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(1000)
    e0.record()
    torch.cuda._sleep(2_000_000)
    e1.record()
    torch.cuda.synchronize()
    cyc[0] = 2_000_000 / (e0.elapsed_time(e1) * 1000)
    print(f"Y.4 spin calibration: {cyc[0]:.0f} cycles/us", flush=True)
    topk[0] = TOPK
    # (roof on, warm on, roof triggers, warm regions): "fr" = warm only hc_ffn_fn + router (GLM53_DEC_MOEGLUE_WARM_SET=fr)
    tcfg = {"off": (False, False, None, None), "off2": (False, False, None, None), "roof": (True, False, None, None),
            "warm": (False, True, None, None), "both": (True, True, None, None),
            "both_no_t2": (True, True, ("t0", "t1", "t3", "t4", "t5"), None),
            "both_warm_fr": (True, True, None, "fr")}
    full_regions = [list(h.regions) for h in MG.WARM.handles]
    names = list(tcfg)
    reps, rounds = 20, int(os.environ.get("R16_TIMING_ROUNDS", "31"))
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for M in (5, 8):
        xin = (torch.randn(M, K, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        gs = {}
        for name, (ro, wa, trig, wset) in tcfg.items():
            setcfg(ro, wa, trig)
            for h, full in zip(MG.WARM.handles, full_regions):
                h.regions = full[:2] if wset == "fr" else list(full)   # f, r are the first two regions
            gs[name] = capture(forward, xin)
        for h, full in zip(MG.WARM.handles, full_regions):
            h.regions = list(full)
        setcfg(True, True)
        ts = {n: [] for n in tcfg}
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            if r % 2:
                order = order[::-1]
            for n in order:
                gr = gs[n][0]
                gr.replay()
                torch.cuda.synchronize()
                ev[0].record()
                for _ in range(reps):
                    gr.replay()
                ev[1].record()
                torch.cuda.synchronize()
                ts[n].append(ev[0].elapsed_time(ev[1]) * 1000 / reps)
        base = ts["off"]
        for n in names:
            rat = sorted(a / b for a, b in zip(ts[n], base))
            print(f"Y.4 M={M} {n:11s} {statistics.median(ts[n]):8.1f} us/forward  ratio vs off median "
                  f"{statistics.median(rat):.4f} [p10 {rat[len(rat) // 10]:.4f}, p90 {rat[9 * len(rat) // 10]:.4f}]",
                  flush=True)
        diffs = {n: (gs[n][1].float() - gs["off"][1].float()).norm().item() / gs["off"][1].float().norm().item()
                 for n in names}
        print(f"Y.4 M={M} rel_l2 vs off of each graph's output (topk 8; off2 = null repeat: K2's fp32 atomics order "
              f"is not run-to-run deterministic): " + ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()), flush=True)
        ck(max(diffs.values()) < 0.05, f"Y.4 M={M} outputs far beyond the null repeat's noise: {diffs}")
        del gs
        torch.cuda.empty_cache()
    warns = [m for lv, nm, m in cap.records if lv >= logging.WARNING]
    ck(not warns, f"no WARNING from the hooks: {warns[:5]}")
    H.report_peak(8.0)
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
