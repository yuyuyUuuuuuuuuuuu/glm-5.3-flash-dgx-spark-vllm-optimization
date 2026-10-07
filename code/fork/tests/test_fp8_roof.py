"""GLM53_DEC_FP8ROOF (fp8_roof.py, kernels/fp8_roof.cu, docs/DEC_FP8ROOF.md) against the real production module.
Run with the launcher overlay exl3.py bound over the image's (its Glm53DenseFp8Method takes (group, prefix), which
the prefetch registry needs), and the real drafter fc weight mounted:
  GPU_RUN_BIND="$PWD/docs/ref/prod_live/overlay_exl3.py=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py" \
  GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c tests/gpu_run.sh python3 -u tests/test_fp8_roof.py
Exit 1 on any failure.

R.1  inert: GLM53_DEC_FP8ROOF unset / 0 -> install() and integrate.plugin_register() change nothing: fp8_gemv.TABLE
     identical, fp8_gemv.ROOF_HOOK None, Exl3MoEMethod.apply untouched, no roof extension imported.
R.2  GLM53_DEC_FP8ROOF=1 without the fp8_gemv small-M path -> WARNING, nothing installed.
R.3  env parsing: bad _PF / _PF_MIB / _PF_CTAS / _PF_POL -> WARNING, prefetch off, table still installed; _TABLE=0.
R.4  install: fp8_gemv.TABLE gains (4096, 20480) and no existing entry changes; idempotent; ROOF_HOOK set.
R.5  drafter fc (real GLM-5.3-Flash-DFlash2 fc.weight [4096, 20480], production's Glm53DenseFp8Method("draft",
     "model.fc")): load-time self-test passes; registered as the t5 prefetch target ("fc", -1); M = 1..16 served by the new kernel (counters); vs float64 every
     element within 0.5 bf16 ulp + the fp32-accumulation bound, correctly rounded at least as often as Marlin; rel_l2 / max-abs vs
     Marlin and vs float64 printed for both; M = 17 and 64 -> Marlin bitwise.
R.6  l2_prefetch: bad arguments raise; the prefetched weight is bitwise unchanged; a GEMV whose whole weight was
     prefetched right before is faster than the same GEMV on cold weights (CUDA graph, medians).
R.7  roles: realistic target prefixes map to the plan's roles (the dense down_proj is t3's anchor, the lm_head and the
     drafter fc are ("lm_head", -1) / ("fc", -1)); drafter layers / mtp / f_b / g_b / kv_b prefixes do not.
R.8  a production-structured mini model (layer 2 = KDA + dense MLP, 3 = MLA + MoE, 4 and 5 = KDA + MoE; every FP8
     linear a production Glm53DenseFp8Method at its real shape and prefix; the shared expert on an aux stream as
     vLLM's runner does; the routed MoE a stand-in class wrapped like production's Exl3MoEMethod.apply):
     a. eager, M = 5: outputs with the prefetch == without, bitwise; forks t1 3 / t2 4 / t3 3 / t4 1 per forward,
        every fork joined by its successor, no safety joins, no drops;
     b. CUDA graph capture at M = 1, 5, 8, 16, 64 succeeds (every forked stream rejoined), replay == eager
        (prefetch off), bitwise, several replays with fresh inputs; capture-time forks as in (a);
     c. M = 65 -> no fork;
     d. stale pending: an eager prefetch left pending, then a new eager forward (layer index goes back) -> dropped;
        an eager pending, then a capture -> dropped (never joined across a capture boundary) and the capture
        succeeds;
     e. every trigger subset (GLM53_DEC_FP8ROOF_PF=t1 / t2 / t3 / t4 / off) captures and replays bitwise.
R.9  torch.compile of a module calling a registered layer's apply: no fork, no join (the hook is eager-only).
R.10 eager anchors: an lm_head built like glm53_runtime (no process_weights_after_loading) is classified at its first
     call; per step  fc, lm_head, layer 0, lm_head  -> t0 (12 MiB of layer 0's in_proj) joined by layer 0, t5 (16 MiB of
     the fc) joined by the next fc; outputs bitwise equal with / without; no fc registered -> every lm_head call is t0,
     joined by the next lm_head call; an eager pending t0, then a capture containing layer 0 and an lm_head call:
     nothing forked or counted inside the capture, the eager pending dropped (not joined), capture ok, replay == eager.
R.11 uninstall() restores Exl3MoEMethod.apply, fp8_gemv.TABLE and ROOF_HOOK.
R.12 a non-CUDA error raised by a prefetch launch during a CUDA-graph capture (first / middle / last launch):
     the capture still succeeds (every forked stream rejoined, prefetch off for the process), replay == eager.
"""
import copy
import inspect
import logging
import os
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard, load_prod, report_peak  # noqa: E402
import torch  # noqa: E402

dev = "cuda"
FC_PATH = Path(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c/model.safetensors"))


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelno, r.getMessage()))

    def has(self, level, text):
        return any(lv == level and text in m for lv, m in self.records)

    def clear(self):
        self.records.clear()


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


class L(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.bias = None


def bf16_ulp(a):
    e = torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -126)))
    return torch.pow(2.0, e - 7)


def check_vs_f64(y, x, fp8, sc, rows=2048):
    """(max err / allowed, fraction correctly rounded, max |err|, rel_l2) of bf16 y vs the float64 reference."""
    n = fp8.shape[0]
    worst, exact, tot, mx, num, den = 0.0, 0, 0, 0.0, 0.0, 0.0
    xd = x.double()
    for a in range(0, n, rows):
        deq = fp8[a:a + rows].double() * sc[a:a + rows].double()[:, None]
        ref = xd @ deq.t()
        acc = (xd.abs() @ deq.abs().t()) * 2.0 ** -20
        err = (y[:, a:a + rows].double() - ref).abs()
        allowed = 0.5 * bf16_ulp(ref.abs() + acc) + acc + 1e-300
        worst = max(worst, (err / allowed).max().item())
        exact += (err <= 0.5 * bf16_ulp(ref) * (1 + 1e-9)).sum().item()
        tot += err.numel()
        mx = max(mx, err.max().item())
        num += (err ** 2).sum().item()
        den += (ref ** 2).sum().item()
    return worst, exact / tot, mx, (num / max(den, 1e-300)) ** 0.5


def main():
    gpu_guard(6.0)
    ck = Checks()
    for v in list(os.environ):
        if v.startswith("GLM53_DEC_FP8ROOF") or v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE",
                                                      "GLM53_KDA_BF16_LARGE_M", "GLM53_FP8_LARGE_M"):
            os.environ.pop(v, None)
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_roof as R
    import integrate
    cap = Cap()
    for name in ("vllm.tf_fp8_roof", "vllm.tf_fp8_gemv"):
        logging.getLogger(name).addHandler(cap)
        logging.getLogger(name).setLevel(logging.DEBUG)
    cls = prod.Glm53DenseFp8Method
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    print(f"Glm53DenseFp8Method{inspect.signature(cls.__init__)}; prefix-aware: {takes_prefix}")
    ck(takes_prefix, "R: the production module under test must be the launcher overlay (GPU_RUN_BIND)")
    table0 = copy.deepcopy(F.TABLE)

    class FakeMoE:
        """Stand-in for production's Exl3MoEMethod (routed experts): same apply signature."""
        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            return layer.moe_fn(x)

    fake_prod = types.SimpleNamespace(Exl3MoEMethod=FakeMoE, __file__="<fake exl3 for the MoE hook>")
    moe_apply0 = FakeMoE.apply

    # ---------------------------------------------------------------- R.1 inert
    print("== R.1 inert")
    for val in (None, "", "0", "off"):
        if val is None:
            os.environ.pop("GLM53_DEC_FP8ROOF", None)
        else:
            os.environ["GLM53_DEC_FP8ROOF"] = val
        rep = R.install(fake_prod)
        R.plugin_register()
        integrate.plugin_register()
        ck(not rep["installed"] and F.ROOF_HOOK is None and F.TABLE == table0 and FakeMoE.apply is moe_apply0,
           f"R.1 {val!r}: something changed")
    ck(R.STATE.ext is None, "R.1 roof extension imported while disabled")
    print(f"   unset/''/0/off: nothing installed, TABLE unchanged, ROOF_HOOK None, extension not imported")

    # ---------------------------------------------------------------- R.2 without fp8_gemv
    print("== R.2 GLM53_DEC_FP8ROOF=1 without GLM53_FP8_GEMV")
    os.environ["GLM53_DEC_FP8ROOF"] = "1"
    cap.clear()
    rep = R.install(fake_prod)
    ck(not rep["installed"] and cap.has(logging.WARNING, "small-M path is not installed") and F.ROOF_HOOK is None,
       f"R.2 {rep}")
    print(f"   {rep['reason'][:110]}")

    os.environ["GLM53_FP8_GEMV"] = "1"
    frep = F.install(prod)
    ck(frep["installed"], f"fp8_gemv install {frep}")

    # ---------------------------------------------------------------- R.3 env parsing
    print("== R.3 env parsing")
    for var, val, text in (("GLM53_DEC_FP8ROOF_PF", "t9", "unknown trigger"),
                           ("GLM53_DEC_FP8ROOF_PF_MIB", "t1=abc", "t1=abc"),
                           ("GLM53_DEC_FP8ROOF_PF_MIB", "t1=99", "t1=99"),
                           ("GLM53_DEC_FP8ROOF_PF_CTAS", "0", "_PF_CTAS"),
                           ("GLM53_DEC_FP8ROOF_PF_POL", "3", "_PF_POL")):
        os.environ[var] = val
        cap.clear()
        rep = R.install(fake_prod)
        ok = rep["installed"] and rep["table"] and not rep["pf"] and cap.has(logging.WARNING, text)
        ck(ok, f"R.3 {var}={val}: {rep}")
        print(f"   {var}={val!r}: table {rep['table']}, prefetch {rep['pf']}, WARNING {cap.has(logging.WARNING, text)}")
        R.uninstall()
        os.environ.pop(var)
    ck(F.TABLE == table0 and F.ROOF_HOOK is None and FakeMoE.apply is moe_apply0, "R.3 uninstall did not restore")
    os.environ["GLM53_DEC_FP8ROOF_TABLE"] = "0"
    rep = R.install(fake_prod)
    ck(rep["installed"] and not rep["table"] and rep["pf"] and F.TABLE == table0, f"R.3 _TABLE=0: {rep}")
    print(f"   GLM53_DEC_FP8ROOF_TABLE=0: table {rep['table']} (TABLE unchanged {F.TABLE == table0}), prefetch {rep['pf']}")
    R.uninstall()
    os.environ.pop("GLM53_DEC_FP8ROOF_TABLE")

    # ---------------------------------------------------------------- R.4 install
    print("== R.4 install")
    cap.clear()
    rep = R.install(fake_prod)
    rep2 = R.install(fake_prod)
    changed = [s for s in table0 if F.TABLE.get(s) != table0[s]]
    ck(rep["installed"] and rep["pf"] and rep["table"] and rep2["reason"] == "already installed", f"R.4 {rep} {rep2}")
    ck((4096, 20480) in F.TABLE and not changed and len(F.TABLE) == len(table0) + 1, f"R.4 table: changed {changed}")
    ck(F.ROOF_HOOK is R and getattr(FakeMoE.apply, "_tf_fp8_roof_hook", False), "R.4 hooks not installed")
    ck(cap.has(logging.INFO, "tf_fp8_roof installed"), "R.4 INFO line")
    print(f"   {rep}")

    g = torch.Generator(device=dev).manual_seed(0)

    def method(group, prefix):
        return cls(group, prefix)

    def build(n, k, group, prefix, w=None):
        if w is None:
            w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
                 torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        lay, m = L(w), method(group, prefix)
        m.process_weights_after_loading(lay)
        return lay, m

    # ---------------------------------------------------------------- R.5 drafter fc
    print("== R.5 drafter fc [4096 x 20480] (real DFlash2 fc.weight)")
    if FC_PATH.is_file():
        from safetensors import safe_open
        with safe_open(str(FC_PATH), framework="pt", device="cpu") as f:
            wfc = f.get_tensor("fc.weight").to(dev)
        src = "real"
    else:
        wfc = (torch.randn(4096, 20480, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        src = "SYNTHETIC (real weight not mounted)"
    ck(src == "real", "R.5 real drafter fc weight not mounted (GPU_RUN_RO)")
    wf = wfc.float()
    sc_ref = (wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0).to(torch.bfloat16)   # the stored scale
    fp8_ref = (wf / (wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0)[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    del wf
    F.COUNTERS.clear()
    fc, mfc = build(4096, 20480, "draft", "model.fc", w=wfc)
    del wfc
    key = F._key(fc.weight, fc.weight_scale, 20480)
    v = F.VERDICT.get(key)
    ck(v is not None and v[0], f"R.5 self-test {v}")
    print(f"   {src}; self-test: {v}")
    ck(getattr(fc, "_glm53_roof_key", None) == ("fc", -1) and R.STATE.reg.get(("fc", -1)) is fc.weight,
       "R.5 drafter fc must be registered as the t5 target")
    # hidden states of the target have O(1)..O(100) channels: a heavy-tailed mixture
    for M in (1, 2, 4, 5, 6, 7, 8, 12, 16, 17, 64):
        x = (torch.randn(M, 20480, device=dev, generator=g) *
             torch.exp(1.5 * torch.randn(1, 20480, device=dev, generator=g))).to(torch.bfloat16)
        F.COUNTERS.clear()
        y = mfc.apply(fc, x)
        served = F.COUNTERS.get("new_eager", 0) == 1
        ym = F.STATE.orig_apply(mfc, fc, x, None)
        if M <= 16:
            ck(served, f"R.5 M={M}: not served by the new kernel ({F.summary()})")
            r, fr, mx, rl = check_vs_f64(y, x, fp8_ref, sc_ref)
            rm, frm, mxm, rlm = check_vs_f64(ym, x, fp8_ref, sc_ref)
            d = (y.float() - ym.float())
            rel_m = d.norm().item() / max(ym.float().norm().item(), 1e-30)
            # K = 20480 with heavy-tailed activations: neither kernel reaches test_fp8_gemv's 99.9 % correctly rounded
            # (K <= 8192 there); the contract here: inside the bound, and not less often correctly rounded than Marlin
            ck(r <= 1.0 and fr >= 0.997 and fr >= frm - 5e-4, f"R.5 M={M}: err/allowed {r:.3f}, rounded {fr:.5f} "
               f"(Marlin {frm:.5f})")
            ck(rel_m <= F.SELFTEST_TOL, f"R.5 M={M}: rel_l2 vs Marlin {rel_m:.2e}")
            print(f"   M={M:2d} new: err/allowed {r:.3f} rounded {fr * 100:.3f}% max|err| {mx:.3g} rel_l2(f64) {rl:.2e} | "
                  f"Marlin: {rm:.3f} {frm * 100:.3f}% {mxm:.3g} {rlm:.2e} | new vs Marlin rel_l2 {rel_m:.2e} max-abs "
                  f"{d.abs().max().item():.3g} bitwise {(y == ym).float().mean().item() * 100:.2f}%")
        else:
            ck(not served and torch.equal(y, ym), f"R.5 M={M}: expected Marlin bitwise ({F.summary()})")
            print(f"   M={M:2d}: Marlin (bitwise == production's apply)")
    del fc, mfc, fp8_ref
    R.STATE.reg.pop(("fc", -1), None)                  # R.10 registers its own
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- R.6 l2_prefetch kernel
    print("== R.6 l2_prefetch")
    E = R.ext()
    t = torch.randint(-2 ** 31, 2 ** 31 - 1, (4 << 20,), dtype=torch.int32, device=dev)
    before = t.clone()
    for bad, fn in (("offset+nbytes > size", lambda: E.l2_prefetch(t, 16, t.numel() * 4, 4, 0)),
                    ("misaligned offset", lambda: E.l2_prefetch(t, 8, 1024, 4, 0)),
                    ("nbytes % 16", lambda: E.l2_prefetch(t, 0, 1000, 4, 0)),
                    ("ctas 0", lambda: E.l2_prefetch(t, 0, 1024, 0, 0)),
                    ("pol 3", lambda: E.l2_prefetch(t, 0, 1024, 4, 3)),
                    ("non-contiguous", lambda: E.l2_prefetch(t.view(-1, 2)[:, 0], 0, 1024, 4, 0))):
        try:
            fn()
            torch.cuda.synchronize()
            ck(False, f"R.6 {bad}: did not raise")
        except RuntimeError:
            print(f"   {bad}: raised")
    for pol in (0, 1, 2):
        E.l2_prefetch(t, 0, t.numel() * 4, 16, pol)
        E.l2_prefetch(t, 4096, 1 << 20, 3, pol)
    E.l2_prefetch(t, 0, 0, 4, 0)
    torch.cuda.synchronize()
    ck(torch.equal(t, before), "R.6 prefetch modified memory")
    print("   16 MiB prefetched with every policy: memory bitwise unchanged")
    del t, before
    # effect: o_proj-sized GEMV (16.8 MB, fits L2) with / without its weight prefetched first
    lays = [build(4096, 4096, "draft", f"bench.o_proj{i}")[0] for i in range(4)]   # not in the prefetch plan
    x = torch.randn(5, 4096, device=dev, generator=g).to(torch.bfloat16)
    cfg = F.select_config(4096, 4096, 5)
    out = torch.empty(5, 4096, dtype=torch.bfloat16, device=dev)
    Eg = F.ext()

    def gemv(l):
        Eg.fp8_gemv_out(out, x, l.weight, l.weight_scale.view(-1), None, 4096, 4096, *cfg, True)

    def graph(fn):
        fn(); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        return gr

    def seq_cold():
        for l in lays:
            gemv(l)

    ev = [torch.cuda.Event(True) for _ in range(4)]
    gp = [graph(lambda l=l: E.l2_prefetch(l.weight, 0, l.weight.numel() * 4, 16, 0)) for l in lays]
    gg = [graph(lambda l=l: gemv(l)) for l in lays]
    cold, warm = [], []
    for r in range(15):
        for i in range(4):                       # cold: the other 3 weights (50 MB) were read since this one
            ev[0].record(); gg[i].replay(); ev[1].record()
            torch.cuda.synchronize()
            cold.append(ev[0].elapsed_time(ev[1]))
        for i in range(4):
            gp[i].replay()
            ev[2].record(); gg[i].replay(); ev[3].record()
            torch.cuda.synchronize()
            warm.append(ev[2].elapsed_time(ev[3]))
    mc, mw = statistics.median(cold) * 1000, statistics.median(warm) * 1000
    print(f"   4096x4096 GEMV (M=5, cfg {cfg}): cold {mc:.1f} us, right after a full prefetch of its weight {mw:.1f} us")
    ck(mw < 0.8 * mc, f"R.6 prefetched GEMV {mw:.1f} us not faster than cold {mc:.1f} us")
    del lays, gp, gg
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- R.7 roles
    print("== R.7 roles")
    cases = {
        ("kda", "model.layers.4.self_attn.in_proj_qkvbfg_a"): ("kda_in", 4),
        ("kda", "language_model.model.layers.10.self_attn.o_proj"): ("kda_o", 10),
        ("mla", "model.layers.3.self_attn.fused_qkv_a_proj"): ("mla_qkv_a", 3),
        ("mla", "model.layers.3.self_attn.q_b_proj"): ("mla_q_b", 3),
        ("mla", "model.layers.43.self_attn.o_proj"): ("mla_o", 43),
        ("shared", "model.layers.7.mlp.shared_experts.gate_up_proj"): ("sh_gu", 7),
        ("shared", "model.layers.7.mlp.shared_experts.down_proj"): ("sh_dn", 7),
        ("dense", "model.layers.0.mlp.gate_up_proj"): ("dense_gu", 0),
        ("dense", "model.layers.0.mlp.down_proj"): ("dense_dn", 0),
        ("shared", "model.layers.0.mlp.down_proj"): None,
        ("kda", "model.layers.4.self_attn.f_b_proj"): None,
        ("kda", "model.layers.4.self_attn.g_b_proj"): None,
        ("mla", "model.layers.3.self_attn.kv_b_proj"): None,
        ("draft", "model.layers.45.self_attn.qkv_proj"): None,
        ("draft", "model.fc"): ("fc", -1),
        ("lm_head", "lm_head"): ("lm_head", -1),
        ("draft", "model.layers.45.mlp.fc"): None,
        ("draft", "model.fc2"): None,
        ("kda", "model.fc"): None,
        ("kda", "model.mtp.layers.45.self_attn.o_proj"): None,
        ("kda", "draft_model.model.layers.1.self_attn.o_proj"): None,
    }
    bad = {k: (R.role_of(*k), want) for k, want in cases.items() if R.role_of(*k) != want}
    ck(not bad, f"R.7 {bad}")
    print(f"   {len(cases)} prefixes: {'all as expected' if not bad else bad}")

    # ---------------------------------------------------------------- R.8 mini model
    print("== R.8 production-structured mini model (layers 2: KDA+dense, 3: MLA+MoE, 4/5: KDA+MoE)")
    layers = {}
    for Lx in (2, 4, 5):
        p = f"model.layers.{Lx}.self_attn."
        layers[(Lx, "in")] = build(12576, 4096, "kda", p + "in_proj_qkvbfg_a")
        layers[(Lx, "fb")] = build(4096, 128, "kda", p + "f_b_proj")
        layers[(Lx, "gb")] = build(4096, 128, "kda", p + "g_b_proj")
        layers[(Lx, "o")] = build(4096, 4096, "kda", p + "o_proj")
    p = "model.layers.3.self_attn."
    layers[(3, "qkv_a")] = build(2048, 4096, "mla", p + "fused_qkv_a_proj")
    layers[(3, "q_b")] = build(8192, 1536, "mla", p + "q_b_proj")
    layers[(3, "o")] = build(4096, 8192, "mla", p + "o_proj")
    layers[(2, "gu")] = build(12288, 4096, "dense", "model.layers.2.mlp.gate_up_proj")
    layers[(2, "dn")] = build(4096, 6144, "dense", "model.layers.2.mlp.down_proj")
    for Lx in (3, 4, 5):
        layers[(Lx, "sgu")] = build(2048, 4096, "shared", f"model.layers.{Lx}.mlp.shared_experts.gate_up_proj")
        layers[(Lx, "sdn")] = build(4096, 1024, "shared", f"model.layers.{Lx}.mlp.shared_experts.down_proj")
    reg = sorted(R.STATE.reg)
    print(f"   registered: {reg}")
    ck(len(reg) == 3 * 2 + 3 + 2 + 3 * 2, f"R.8 registry {reg}")
    moe_w = torch.randn(4096, 4096, device=dev, generator=g).to(torch.bfloat16) * 0.01
    moe_layers = {}
    for Lx in (3, 4, 5):
        ml = torch.nn.Module()
        ml.layer_name = f"model.layers.{Lx}.mlp.experts"
        ml.moe_fn = lambda h, w=moe_w: torch.nn.functional.linear(h, w)
        moe_layers[Lx] = ml
    moe = FakeMoE()
    aux = torch.cuda.Stream()

    def lin(key, h):
        lay, m = layers[key]
        return m.apply(lay, h)

    def moe_block(Lx, h):
        e = torch.cuda.Event()
        e.record()
        aux.wait_event(e)                                 # vLLM: shared experts forked before the router
        router = h[:, :288].float().softmax(-1)            # router stand-in
        routed = moe.apply(moe_layers[Lx], h, router, None, None, None)
        with torch.cuda.stream(aux):                       # vLLM: shared experts launched after the routed call
            su = lin((Lx, "sgu"), h)
            sh = lin((Lx, "sdn"), torch.nn.functional.silu(su[:, :1024]) * su[:, 1024:])
        torch.cuda.current_stream().wait_stream(aux)
        return routed + sh

    def kda(Lx, h):
        y = lin((Lx, "in"), h)
        fb = lin((Lx, "fb"), y[:, 12320:12448])
        gb = lin((Lx, "gb"), y[:, 12448:12576])
        core = (y[:, :4096].float() * torch.sigmoid(fb.float()) + gb.float() * 0.01).to(torch.bfloat16)
        return lin((Lx, "o"), core)

    def mla(Lx, h):
        a = lin((Lx, "qkv_a"), h)
        q = lin((Lx, "q_b"), a[:, :1536])
        att = q * 0.05                                     # attention stand-in [M, 8192]
        return lin((Lx, "o"), att)

    def forward(h):
        h = h + kda(2, h)
        u = lin((2, "gu"), h)
        h = h + lin((2, "dn"), torch.nn.functional.silu(u[:, :6144]) * u[:, 6144:])
        h = h + mla(3, h)
        h = h + moe_block(3, h)
        for Lx in (4, 5):
            h = h + kda(Lx, h)
            h = h + moe_block(Lx, h)
        return h

    def forks():
        return {t: R.COUNTERS.get("fork_" + t, 0) for t in R.TRIGGERS}

    # t3: dense down(2) -> MLA(3), MoE(3) -> KDA(4), MoE(4) -> KDA(5); t0 / t5 are eager lm_head anchors (R.10)
    want = {"t0": 0, "t1": 3, "t2": 4, "t3": 3, "t4": 1, "t5": 0}

    def run_off(h):
        R.STATE.pf = False
        try:
            return forward(h)
        finally:
            R.STATE.pf = True

    # a. eager
    M = 5
    xin = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    R.COUNTERS.clear()
    y_on = forward(xin)
    torch.cuda.synchronize()
    c_on = dict(R.COUNTERS)
    y_off = run_off(xin)
    torch.cuda.synchronize()
    fk = forks()
    ck(torch.equal(y_on, y_off), "R.8a eager: prefetch changed the output")
    ck(fk == want and c_on.get("joined", 0) == sum(want.values()) and not R.STATE.pending, f"R.8a counters {c_on}")
    ck(not any(k in c_on for k in ("joined_late_safety", "stale_dropped", "dropped_capture_mismatch", "double_fork")),
       f"R.8a unexpected {c_on}")
    # t3 budget 12 MiB each: MLA(3) gets all of fused_qkv_a (8 MiB) + the first 4 MiB of q_b; KDA(4), KDA(5) 12 MiB
    ck(c_on.get("bytes_t3", 0) == 3 * 12 * 2 ** 20, f"R.8a t3 bytes {c_on.get('bytes_t3', 0)} != 36 MiB")
    print(f"   a. eager M=5: output bitwise equal with/without prefetch {torch.equal(y_on, y_off)}; forks {fk}; "
          f"joined {c_on.get('joined')}; bytes " + " ".join(f"{t}={c_on.get('bytes_' + t, 0) / 2**20:.1f}MiB" for t in R.TRIGGERS))

    # b. graphs
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

    for M in (1, 5, 8, 16, 64):
        xin = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        R.COUNTERS.clear()
        try:
            gr, yg = capture(forward, xin)
            ok_cap = True
        except Exception as exc:  # noqa: BLE001
            ok_cap = False
            print(f"   M={M}: capture FAILED: {exc!r}")
        ck(ok_cap, f"R.8b M={M}: capture failed")
        if not ok_cap:
            continue
        c_cap = dict(R.COUNTERS)
        fk = {t: c_cap.get("fork_" + t, 0) for t in R.TRIGGERS}
        # warmup forward + captured forward: 2x the per-forward forks
        ck(fk == {t: 2 * n for t, n in want.items()} and not R.STATE.pending, f"R.8b M={M}: {c_cap}")
        same = True
        for _ in range(3):
            xin.copy_((torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16))
            gr.replay()
            ye = run_off(xin)
            torch.cuda.synchronize()
            same &= torch.equal(yg, ye)
        ck(same, f"R.8b M={M}: replay != eager (prefetch off)")
        ck(dict(R.COUNTERS) == c_cap, f"R.8b M={M}: counters moved during replay")
        print(f"   b. M={M:2d}: capture ok, forks per forward {({t: n // 2 for t, n in fk.items()})}, replay == eager "
              f"(prefetch off) bitwise x3 {same}")
        del gr

    # c. M = 65
    R.COUNTERS.clear()
    xin = (torch.randn(65, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    forward(xin)
    torch.cuda.synchronize()
    ck(sum(forks().values()) == 0 and not R.STATE.pending, f"R.8c M=65 forked: {R.COUNTERS}")
    print(f"   c. M=65: forks {forks()}")

    # d. stale pending
    xin = (torch.randn(5, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    R.COUNTERS.clear()
    kda(4, xin)                                        # forks t1 (joined by o_proj(4)) and t2 -> pending sh_gu(4)
    ck(("sh_gu", 4) in R.STATE.pending, f"R.8d pending {list(R.STATE.pending)}")
    forward(xin)                                       # new forward starts at layer 2 < 4: the stale one is dropped
    torch.cuda.synchronize()
    ck(R.COUNTERS.get("stale_dropped", 0) == 1 and not R.STATE.pending, f"R.8d eager stale: {R.COUNTERS}")
    print(f"   d. eager pending + new forward: dropped {R.COUNTERS.get('stale_dropped', 0)}, pending {len(R.STATE.pending)}")
    R.COUNTERS.clear()
    kda(4, xin)                                        # eager pending again (t2 -> sh_gu(4))
    R.STATE.last_L = -1                                # hide the forward boundary: only the capture-state check remains
    def tail(h):                                       # MoE(4) -> KDA(5) -> MoE(5): every fork's successor inside
        h = h + moe_block(4, h)
        h = h + kda(5, h)
        return h + moe_block(5, h)

    try:
        gr, _ = capture(tail, xin)                     # warmup (eager) joins it; then the capture itself
        ok_cap = True
    except Exception as exc:  # noqa: BLE001
        ok_cap = False
        print(f"   capture failed: {exc!r}")
    ck(ok_cap, "R.8d capture after an eager pending failed")
    R.COUNTERS.clear()
    kda(4, xin)                                        # eager pending (sh_gu(4))
    R.STATE.last_L = -1
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            tail(xin)                                  # capture: the eager pending must be dropped, not joined
        ok_cap = True
    except Exception as exc:  # noqa: BLE001
        ok_cap = False
        print(f"   capture failed: {exc!r}")
    ck(ok_cap and R.COUNTERS.get("dropped_capture_mismatch", 0) == 1 and not R.STATE.pending,
       f"R.8d capture-state mismatch: {R.COUNTERS}")
    print(f"   d. eager pending, then a capture: capture ok {ok_cap}, dropped {R.COUNTERS.get('dropped_capture_mismatch', 0)}")
    torch.cuda.synchronize()

    # e. trigger subsets
    for sub in ("t1", "t2", "t3", "t4", "off"):
        R.STATE.triggers = frozenset() if sub == "off" else frozenset({sub})
        R.COUNTERS.clear()
        xin = (torch.randn(8, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        try:
            gr, yg = capture(forward, xin)
            gr.replay()
            ye = run_off(xin)
            torch.cuda.synchronize()
            ok = torch.equal(yg, ye)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"   {sub}: {exc!r}")
        fk = {t: n // 2 for t, n in forks().items() if n}
        exp = {} if sub == "off" else {sub: want[sub]}
        ck(ok and fk == exp, f"R.8e {sub}: ok {ok} forks {fk}")
        print(f"   e. triggers {sub}: capture + replay bitwise {ok}, forks per forward {fk}")
    R.STATE.triggers = frozenset(R.TRIGGERS)

    # ---------------------------------------------------------------- R.9 torch.compile
    print("== R.9 torch.compile")
    lay, m = layers[(4, "in")]

    class Mod(torch.nn.Module):
        def forward(self, h):
            return m.apply(lay, h)

    torch._dynamo.reset()
    cm = torch.compile(Mod(), fullgraph=True, dynamic=True)
    R.COUNTERS.clear()
    xin = (torch.randn(5, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    yc = cm(xin)
    yc2 = cm(xin)
    torch.cuda.synchronize()
    ye = m.apply(lay, xin)                              # eager call (forks t1 -> pending kda_o(4))
    R.STATE.pending.clear()
    ck(torch.equal(yc, yc2), "R.9 compiled not deterministic")
    ck(R.COUNTERS.get("fork_t1", 0) == 1, f"R.9 compiled calls forked: {R.COUNTERS}")   # only the eager call
    print(f"   compiled calls: no fork (fork_t1 = {R.COUNTERS.get('fork_t1', 0)} from the one eager call); "
          f"compiled == eager {torch.equal(yc, ye)}")

    # ---------------------------------------------------------------- R.10 eager anchors t0 / t5
    print("== R.10 eager anchors: lm_head -> target layer 0 (t0), 2nd lm_head since the fc -> drafter fc (t5)")
    R.STATE.pending.clear()
    R.STATE.lm_count = 0
    in0, min0 = build(12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a")
    fc2, mfc2 = build(4096, 20480, "draft", "model.fc")
    ck(R.STATE.reg.get(("fc", -1)) is fc2.weight and R.STATE.reg.get(("kda_in", 0)) is in0.weight, "R.10 registry")

    def lm_head_like(n, k):
        """glm53_runtime.convert_lm_head_fp8: FP8 Marlin holder + Glm53DenseFp8Method("lm_head", "lm_head"), ready,
        WITHOUT process_weights_after_loading (so never register()ed: the key is classified at the first call)."""
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        sc = w.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
        h = torch.nn.Module()
        h.output_size_per_partition, h.input_size_per_partition = n, k
        h.orig_dtype = torch.bfloat16
        h.weight = torch.nn.Parameter((w.float() / sc[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn),
                                      requires_grad=False)
        h.weight_scale = torch.nn.Parameter(sc.to(torch.bfloat16), requires_grad=False)
        h.weight_block_size = None
        prepare_fp8_layer_for_marlin(h, size_k_first=False)
        h.glm53_fp8_n, h.glm53_fp8_k = n, k
        mm = cls("lm_head", "lm_head")
        mm.ready = True
        return h, mm

    lm, mlm = lm_head_like(8192, 4096)
    ck(not hasattr(lm, "_glm53_roof_key"), "R.10 lm_head must not be registered at build time")
    xs = {M: ((torch.randn(M, 20480, device=dev, generator=g) * 0.5).to(torch.bfloat16),
              (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)) for M in (5,)}

    def step(M):                                       # one decode step's eager FP8 calls, production order
        xf, xh = xs[M]
        return [mfc2.apply(fc2, xf),                   # drafter fc          (joins a pending t5, resets the count)
                mlm.apply(lm, xh),                     # drafter lm_head     -> t0: layer 0's in_proj
                min0.apply(in0, xh),                   # target layer 0      (eager here; production: a graph replay)
                mlm.apply(lm, xh)]                     # target lm_head      -> t5: the fc

    R.COUNTERS.clear()
    outs_on = step(5) + step(5) + [mfc2.apply(fc2, xs[5][0])]
    torch.cuda.synchronize()
    c10 = dict(R.COUNTERS)
    R.STATE.pf = False
    outs_off = step(5) + step(5) + [mfc2.apply(fc2, xs[5][0])]
    R.STATE.pf = True
    torch.cuda.synchronize()
    same = all(torch.equal(a_, b_) for a_, b_ in zip(outs_on, outs_off))
    ck(same, "R.10 eager anchors changed an output")
    ck(getattr(lm, "_glm53_roof_key", None) == ("lm_head", -1), "R.10 lm_head key not classified")
    ck(c10.get("fork_t0", 0) == 2 and c10.get("fork_t5", 0) == 2 and c10.get("joined", 0) == 4
       and not R.STATE.pending and R.STATE.lm_count == 0, f"R.10 counters {c10}")
    ck(c10.get("bytes_t0", 0) == 2 * 12 * 2 ** 20 and c10.get("bytes_t5", 0) == 2 * 16 * 2 ** 20, f"R.10 bytes {c10}")
    print(f"   2 steps + fc: outputs bitwise equal with/without {same}; forks t0 {c10.get('fork_t0', 0)} "
          f"t5 {c10.get('fork_t5', 0)}, joined {c10.get('joined', 0)}, bytes t0 "
          f"{c10.get('bytes_t0', 0) / 2**20:.0f} MiB t5 {c10.get('bytes_t5', 0) / 2**20:.0f} MiB")
    # no FP8 fc (drafter off / GLM53_DRAFT_FP8 without fc): every lm_head call -> t0, joined by the next lm_head call
    wfc_saved = R.STATE.reg.pop(("fc", -1))
    R.COUNTERS.clear()
    for _ in range(3):
        mlm.apply(lm, xs[5][1])
    torch.cuda.synchronize()
    c10b = dict(R.COUNTERS)
    ck(c10b.get("fork_t0", 0) == 3 and c10b.get("fork_t5", 0) == 0 and c10b.get("joined_eager", 0) == 2
       and list(R.STATE.pending) == [("kda_in", 0)], f"R.10 no-fc counters {c10b}")
    print(f"   no fc registered: 3 lm_head calls -> t0 x{c10b.get('fork_t0', 0)}, joined by the next lm_head "
          f"x{c10b.get('joined_eager', 0)}, one pending")
    R.STATE.reg[("fc", -1)] = wfc_saved
    # an eager t0 pending, then a capture of the target's layer 0 and of an lm_head call: nothing forked or counted
    # inside the capture, the eager pending is dropped (never joined across the capture boundary), capture succeeds
    R.COUNTERS.clear()
    n0 = R.STATE.lm_count

    def target_graph(h):
        y = min0.apply(in0, h)
        return y, mlm.apply(lm, h)

    try:
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg0, yg1 = target_graph(xs[5][1])
        gr.replay()
        torch.cuda.synchronize()
        ok_cap = True
    except Exception as exc:  # noqa: BLE001
        ok_cap = False
        print(f"   capture failed: {exc!r}")
    c10c = dict(R.COUNTERS)
    ck(ok_cap and c10c.get("dropped_capture_mismatch", 0) == 1 and not R.STATE.pending and R.STATE.lm_count == n0
       and c10c.get("fork_t0", 0) == 0 and c10c.get("fork_t5", 0) == 0, f"R.10 capture: ok {ok_cap} {c10c}")
    ck(torch.equal(yg0, min0.apply(in0, xs[5][1])), "R.10 captured layer 0 != eager")
    R.STATE.pending.clear()
    print(f"   eager t0 pending + capture (layer 0 + lm_head inside): capture ok {ok_cap}, dropped "
          f"{c10c.get('dropped_capture_mismatch', 0)}, forks in capture {c10c.get('fork_t0', 0) + c10c.get('fork_t5', 0)}, "
          f"lm_head count unchanged {R.STATE.lm_count == n0}")
    del gr, in0, fc2, lm

    # ---------------------------------------------------------------- R.12 a launch error inside a capture
    print("== R.12 a non-CUDA error raised by a prefetch launch in the middle of a CUDA-graph capture")
    real_E = R.STATE.ext

    class Flaky:
        """l2_prefetch that raises a TORCH_CHECK-like RuntimeError at the n-th launch of a capture."""
        VERSION = 1

        def __init__(self, fail_at):
            self.fail_at, self.nc = fail_at, 0

        def l2_prefetch(self, *a):
            if torch.cuda.is_current_stream_capturing():
                self.nc += 1
                if self.nc == self.fail_at:
                    raise RuntimeError("l2_prefetch: synthetic launch failure (R.12)")
            return real_E.l2_prefetch(*a)

    # one forward = 11 forks / 15 launches (t2 of a MoE layer and t3 into MLA launch two): first, a second-of-two, last
    for fail_at in (1, 7, 15):
        R.STATE.pending.clear()
        R.STATE.triggers, R.STATE.pf, R.STATE.ext = frozenset(R.TRIGGERS), True, Flaky(fail_at)
        R.COUNTERS.clear()
        xin = (torch.randn(5, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        try:
            gr, yg = capture(forward, xin)
            ok_cap = True
        except Exception as exc:  # noqa: BLE001
            ok_cap = False
            print(f"   fail_at={fail_at}: capture FAILED: {exc!r}"[:300])
        c12 = dict(R.COUNTERS)
        ck(ok_cap, f"R.12 fail_at={fail_at}: the capture must survive a hook error (prefetch off, streams rejoined)")
        ck(not R.STATE.pf and not R.STATE.pending, f"R.12 fail_at={fail_at}: pf {R.STATE.pf} pending {list(R.STATE.pending)}")
        if ok_cap:
            same = True
            for _ in range(2):
                xin.copy_((torch.randn(5, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16))
                gr.replay()
                ye = forward(xin)                          # prefetch is off now: production's path
                torch.cuda.synchronize()
                same &= torch.equal(yg, ye)
            ck(same, f"R.12 fail_at={fail_at}: replay != eager")
            del gr
            print(f"   fail_at={fail_at}: capture ok, prefetch now off, rejoined {c12.get('joined_on_disable', 0)}, "
                  f"replay == eager bitwise x2 {same}")
        torch.cuda.synchronize()
    R.STATE.ext, R.STATE.pf, R.STATE.disabled_reason = real_E, True, None
    R.STATE.pending.clear()

    # ---------------------------------------------------------------- R.11 uninstall
    print("== R.11 uninstall")
    rep = R.uninstall()
    ck(FakeMoE.apply is moe_apply0 and F.TABLE == table0 and F.ROOF_HOOK is None, "R.11 not restored")
    print(f"   {rep}; Exl3MoEMethod.apply restored, TABLE restored, ROOF_HOOK None")
    F.uninstall()
    report_peak(6.0)
    ck.summary()


run_main(main)
