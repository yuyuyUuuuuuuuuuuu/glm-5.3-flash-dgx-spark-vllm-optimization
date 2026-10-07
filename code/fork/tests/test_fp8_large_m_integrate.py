"""GLM53_FP8_LARGE_M (fp8_gemv.py large-M path) against the real production module (image exl3.py, or the launcher
overlay's via GPU_RUN_BIND). Exit 1 on any failure.

L.1  inert: GLM53_FP8_LARGE_M unset / '' / 0 / off -> nothing patched by it; with only GLM53_FP8_GEMV=1 every
     large-M call (M 65..13824, every LARGE_TABLE shape) is bitwise production's apply and no large counter moves.
L.2  env parsing: on values; invalid GLM53_FP8_LARGE_M_TEMP_MIB refused with a WARNING (nothing installed when it
     is the only feature), never raises.
L.3  install with only GLM53_FP8_LARGE_M=1 (via integrate.plugin_register): wrappers installed once, INFO line;
     small-M path off -> M 1..64 bitwise production's apply.
L.4  per-layer large-M self-test at load (process_weights_after_loading) for every production shape whose (Npad, K)
     is in LARGE_TABLE (INFO once per shape), none for the other shapes; persistent memory = the fp32 scale (4 N B).
L.5  apply(): M < LARGE_TABLE[(Npad, K)] -> bitwise production's apply; M >= it -> served (counter, one INFO line
     "first large-M call served" per shape: the rollout's engagement evidence) and bitwise
     equal to fp8_gemv.large_forward, within fp32-order of Marlin (rel_l2 <= 1e-3, >= 99 % bitwise); 3-D input;
     a layer with a bias (Marlin-permuted layer.bias) == Marlin's bias epilogue class; strided x (column slice).
L.6  fallbacks, each bitwise == production's apply: failed large-M self-test (WARNING), pre-launch error (WARNING
     once; strict raises; CUDA-error text re-raised), CUDA-graph capture of a large-M call (Marlin captured, no
     sync), KDA large-M BF16 copy present, method not ready, fp16 activations.
L.7  both GLM53_FP8_GEMV=1 and GLM53_FP8_LARGE_M=1 (in_proj): M <= 16 -> fp8_gemv kernel, 17..639 -> Marlin,
     >= min_m -> large path (GLM53_FP8_GEMV_MAX_M=16 as in production).
L.8  torch.compile(fullgraph=True, dynamic=True): the op tf_fp8.linear chooses at run time (M 8 / 100 / 1024).
L.9  memory: peak extra allocation of one call at M = 13824 (and 2202 / fc 8192), large path (both backends) vs
     Marlin; the transient never exceeds the output + GLM53_FP8_LARGE_M_TEMP_MIB (+ 5 %).
L.10 uninstall restores both methods and clears the caches.
L.11 GLM53_FP8_LARGE_M_GEMM=cublas selects the cuBLAS backend: results bitwise equal to the default TileLang backend.
L.12 TileLang failing at the first self-test (synthetic compile error): WARNING, the cuBLAS backend takes over, the layer
     passes its self-test and is served.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard, load_prod  # noqa: E402
import inspect  # noqa: E402
import torch  # noqa: E402
from test_fp8_integrate import Cap, single_rank_tp, L  # noqa: E402

dev = "cuda"


def main():
    gpu_guard(8.0)
    ck = Checks()
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "GLM53_FP8_LARGE_M", "GLM53_FP8_LARGE_M_TEMP_MIB",
              "GLM53_FP8_LARGE_M_GEMM", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M"):
        os.environ.pop(v, None)
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import integrate
    cap = Cap()
    logging.getLogger("vllm.tf_fp8_gemv").addHandler(cap)
    logging.getLogger("vllm.tf_fp8_gemv").setLevel(logging.DEBUG)
    cls = prod.Glm53DenseFp8Method
    orig_apply, orig_pwal = cls.apply, cls.process_weights_after_loading
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters

    def method(group, prefix):
        return cls(group, prefix) if takes_prefix else cls(group)

    g = torch.Generator(device=dev).manual_seed(0)

    def build(n, k, group="kda", prefix="model.layers.0.self_attn.in_proj_qkvbfg_a", dtype=torch.bfloat16, bias=False):
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(dtype)
        b = (torch.randn(n, device=dev, generator=g) * 0.1).to(dtype) if bias else None
        lay, m = L(w, b), method(group, prefix)
        m.process_weights_after_loading(lay)
        return lay, m

    def orig(m, lay, x, bias=None):
        return orig_apply(m, lay, x, bias)

    def xr(M, k, dtype=torch.bfloat16):
        return torch.randn(M, k, device=dev, generator=g).to(dtype)

    def agree(y, yo):
        rl2 = (y.float() - yo.float()).norm().item() / max(yo.float().norm().item(), 1e-30)
        return rl2, (y == yo).float().mean().item()

    shapes = {"kda.in_proj": (12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a"),
              "dense.gate_up": (12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj"),
              "draft.fc": (4096, 20480, "draft", "model.fc")}
    Ms = (65, 200, 639, 640, 8191, 8192, 13824)

    # ---------------------------------------------------------------- L.1 inert
    print("== L.1 inert")
    for val in (None, "", "0", "off"):
        if val is None:
            os.environ.pop("GLM53_FP8_LARGE_M", None)
        else:
            os.environ["GLM53_FP8_LARGE_M"] = val
        rep = F.install(prod)
        F.plugin_register()
        integrate.plugin_register()
        ck(not rep["installed"] and cls.apply is orig_apply and cls.process_weights_after_loading is orig_pwal,
           f"L.1 {val!r}: patched")
    ck(F.STATE.large_ext is None, "L.1 large extension imported while disabled")
    os.environ.pop("GLM53_FP8_LARGE_M", None)
    os.environ["GLM53_FP8_GEMV"] = "1"
    os.environ["GLM53_FP8_GEMV_MAX_M"] = "16"
    rep = F.install(prod)
    ck(rep["installed"] and not F.STATE.large, "L.1 GEMV-only install")
    built = {}
    for name, (n, k, grp, pre) in shapes.items():
        built[name] = build(n, k, grp, pre)
    before = {k: v for k, v in F.COUNTERS.items() if k.startswith("large")}
    for name, (lay, m) in built.items():
        k = lay.weight.shape[0] * 16
        for M in Ms:
            x = xr(M, k)
            ck(torch.equal(m.apply(lay, x), orig(m, lay, x)), f"L.1 GEMV-only {name} M={M}: != production")
    after = {k: v for k, v in F.COUNTERS.items() if k.startswith("large")}
    ck(before == after and not F.LVERDICT, f"L.1 large counters moved {before} -> {after}")
    ck(F.STATE.large_ext is None, "L.1 large extension imported with only GLM53_FP8_GEMV")
    F.uninstall()
    print(f"   unset/''/0/off: nothing installed; GEMV-only: {len(Ms) * len(built)} large-M calls bitwise production, "
          "no large counter, large extension not imported")

    # ---------------------------------------------------------------- L.2 env parsing
    print("== L.2 env parsing")
    os.environ.pop("GLM53_FP8_GEMV", None)
    ck(all(F.env_large_enabled({"GLM53_FP8_LARGE_M": v}) for v in ("1", "on", "true", "yes", " ON ")), "L.2 on")
    ck(not any(F.env_large_enabled({"GLM53_FP8_LARGE_M": v}) for v in ("", "0", "off", "2")), "L.2 off")
    for bad in ("0", "8", "x", "5000"):
        os.environ["GLM53_FP8_LARGE_M_TEMP_MIB"] = bad
        cap.clear()
        rep = F.install(prod, force_large=True)
        ck(not rep["installed"] and cap.has(logging.WARNING, "large-M path is NOT enabled") and cls.apply is orig_apply,
           f"L.2 TEMP_MIB={bad!r} not refused: {rep}")
    os.environ.pop("GLM53_FP8_LARGE_M_TEMP_MIB")
    os.environ["GLM53_FP8_LARGE_M_GEMM"] = "cutlass"
    cap.clear()
    rep = F.install(prod, force_large=True)
    ck(not rep["installed"] and cap.has(logging.WARNING, "large-M path is NOT enabled"), f"L.2 GEMM=cutlass: {rep}")
    os.environ.pop("GLM53_FP8_LARGE_M_GEMM")
    print("   GLM53_FP8_LARGE_M on: 1/on/true/yes; invalid GLM53_FP8_LARGE_M_TEMP_MIB (0, 8, x, 5000) and "
          "GLM53_FP8_LARGE_M_GEMM (cutlass) refused (WARNING)")

    # ---------------------------------------------------------------- L.3 install large only
    print("== L.3 install (GLM53_FP8_LARGE_M=1 only)")
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    os.environ.pop("GLM53_FP8_GEMV_MAX_M", None)
    F.CFG.max_m = F.MAX_M
    cap.clear()
    integrate.plugin_register()
    ck(getattr(cls.apply, "_tf_fp8_hook", False) and F.STATE.large and not F.STATE.enabled, "L.3 not installed")
    ck(F.STATE.large_gemm == "tilelang", f"L.3 backend {F.STATE.large_gemm} (tilelang expected: it is in the image)")
    ck(cap.has(logging.INFO, "large-M path installed"), "L.3 no INFO")
    rep = F.install(prod)
    ck(rep["reason"] == "already installed", "L.3 not idempotent")
    lay, m = built["kda.in_proj"]
    for M in (1, 8, 16, 64):
        x = xr(M, 4096)
        ck(torch.equal(m.apply(lay, x), orig(m, lay, x)), f"L.3 M={M}: small M != production with GEMV off")
    print("   installed via integrate.plugin_register; small-M path off -> M 1..64 bitwise production")

    # ---------------------------------------------------------------- L.4 self-test at load
    print("== L.4 per-layer large-M self-test at load")
    from fp8_bench_common import PROD_SHAPES
    cap.clear()
    built = {}
    groups = {"kda": ("kda", "model.layers.0.self_attn.in_proj_qkvbfg_a"), "mla": ("mla", "model.layers.3.self_attn.q_b_proj"),
              "dense": ("dense", "model.layers.1.mlp.gate_up_proj"), "shared": ("shared", "model.layers.5.mlp.shared_experts.down_proj"),
              "draft": ("draft", "model.layers.45.mlp.gate_up_proj")}
    all_shapes = {nm: (n, k) for nm, (n, k, _) in PROD_SHAPES.items() if nm != "draft.lmhead"}
    all_shapes["draft.fc"] = (4096, 20480)
    alpha_bytes = 0
    for name, (n, k) in all_shapes.items():
        if (n, k) in {v[:2] for v in built.values()}:
            continue
        grp, pre = ("draft", "model.fc") if name == "draft.fc" else groups[name.split(".")[0]]
        lay, m = build(n, k, grp, pre)
        key = F._key(lay.weight, lay.weight_scale, k)
        npad = lay.weight.shape[1] // 4
        want = (npad, k) in F.LARGE_TABLE
        v = F.LVERDICT.get(key)
        # (the self-test's M = 320 is run in 128-row calls: for K = 20480 cuBLAS then picks a split-K kernel, a
        # different fp32 summation order; at the served M both backends are bitwise equal, test_fp8_large_m.py N)
        ck((v is not None and v[0] and "(tilelang)" in v[1]) if want else v is None,
           f"L.4 {name}: verdict {v} (want {want})")
        if want:
            alpha_bytes += F.LALPHA[key].numel() * 4
            built[name] = (n, k, lay, m)
        print(f"   {name:16s} {n:6d}x{k:5d} Npad {npad:6d}: {'large-M self-test ' + v[1] if v else 'not a large-M shape'}")
    nst = sum(1 for lv, msg in cap.records if lv == logging.INFO and "large-M self-test passed" in msg)
    ck(nst == len(F.LARGE_TABLE), f"L.4 INFO lines {nst} != {len(F.LARGE_TABLE)}")
    print(f"   persistent memory for these layers: {alpha_bytes} B of fp32 scale (4 N per layer)")

    # ---------------------------------------------------------------- L.5 apply
    print("== L.5 apply: threshold, equivalence")
    cap.clear()
    for name, (n, k, lay, m) in built.items():
        min_m = F.LARGE_TABLE[(lay.weight.shape[1] // 4, k)]
        line = f"   {name} (min M {min_m}):"
        for M in sorted({65, min_m - 1, min_m, min_m + 1, 2202, 13824}):
            x = xr(M, k)
            n0 = F.COUNTERS.get("large_eager", 0)
            y = m.apply(lay, x)
            served = F.COUNTERS.get("large_eager", 0) - n0
            yo = orig(m, lay, x)
            ck(y.shape == (M, n) and y.dtype == torch.bfloat16, f"L.5 {name} M={M}: shape {y.shape}")
            if M >= min_m:
                yd = F.large_forward(x, lay.weight, F.LALPHA[F._key(lay.weight, lay.weight_scale, k)], None, n, k)
                rl2, same = agree(y, yo)
                ck(served == 1 and torch.equal(y, yd), f"L.5 {name} M={M}: not served / != large_forward")
                ck(rl2 <= 1e-3 and same >= 0.99, f"L.5 {name} M={M}: vs Marlin rel_l2 {rl2:.1e} same {same:.4f}")
                line += f" {M}:large(rl2 {rl2:.1e}, ={same*100:.2f}%)"
            else:
                ck(served == 0 and torch.equal(y, yo), f"L.5 {name} M={M}: below threshold != production")
                line += f" {M}:marlin"
            del x, y, yo
        print(line, flush=True)
    nfirst = sum(1 for lv, msg in cap.records if lv == logging.INFO and "first large-M call served" in msg)
    ck(nfirst == len(built), f"L.5 first-call INFO lines {nfirst} != {len(built)}")
    n, k, lay, m = built["kda.in_proj"]
    x3 = torch.randn(2, 1024, 4096, device=dev, generator=g).to(torch.bfloat16)
    y3 = m.apply(lay, x3)
    ck(y3.shape == (2, 1024, n) and torch.equal(y3.reshape(2048, n), m.apply(lay, x3.reshape(2048, 4096))), "L.5 3-D")
    wide = torch.randn(1024, 4096 + 256, device=dev, generator=g).to(torch.bfloat16)
    xs = wide[:, 128:128 + 4096]
    ck(torch.equal(m.apply(lay, xs), m.apply(lay, xs.contiguous())), "L.5 strided x")
    layb, mb = build(12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj", bias=True)
    for M in (640, 2202):
        x = xr(M, 4096)
        n0 = F.COUNTERS.get("large_eager", 0)
        yb, yob = mb.apply(layb, x, layb.bias), orig(mb, layb, x, layb.bias)
        rl2, same = agree(yb, yob)
        ck(F.COUNTERS.get("large_eager", 0) == n0 + 1 and rl2 <= 1e-3 and same >= 0.99,
           f"L.5 bias M={M}: rl2 {rl2:.1e} same {same:.4f}")
    print("   3-D input and a strided x (column slice) ok; bias (Marlin-permuted) agrees with Marlin's epilogue")

    # ---------------------------------------------------------------- L.6 fallbacks
    print("== L.6 fallbacks (each bitwise == production's apply)")

    def fallback(label, m, lay, x, counter, bias=None):
        before = F.COUNTERS.get(counter, 0) if counter else 0
        y = m.apply(lay, x, bias)
        ok = torch.equal(y, orig(m, lay, x, bias)) and (counter is None or F.COUNTERS.get(counter, 0) == before + 1)
        ck(ok, f"L.6 {label}")
        print(f"   {label}: {'ok' if ok else 'FAILED'}")

    tol = F.SELFTEST_TOL
    F.SELFTEST_TOL = -1.0
    cap.clear()
    layf, mf = build(12288, 4096, "dense", "model.layers.2.mlp.gate_up_proj")
    F.SELFTEST_TOL = tol
    ck(cap.has(logging.WARNING, "large-M self-test FAILED"), "L.6 no WARNING for a failed self-test")
    fallback("failed large-M self-test", mf, layf, xr(1024, 4096), "large_marlin_selftest_failed")
    n, k, lay, m = built["kda.in_proj"]

    class Boom:
        def __init__(self, msg):
            self.msg = msg

        def dequant(self, *a):
            raise RuntimeError(self.msg)

        scale_cast = dequant

    real = F.STATE.large_ext
    F.STATE.large_ext = Boom("fp8_large_m.dequant: synthetic pre-check failure")
    cap.clear()
    fallback("pre-launch error", m, lay, xr(700, 4096), "large_marlin_error")
    fallback("pre-launch error (again)", m, lay, xr(700, 4096), "large_marlin_error")
    ck(sum(1 for lv, msg in cap.records if lv == logging.WARNING and "large-M path failed" in msg) == 1,
       "L.6 WARNING not exactly once")
    F.CFG.strict = True
    try:
        m.apply(lay, xr(700, 4096))
        ck(False, "L.6 strict did not raise")
    except RuntimeError:
        pass
    F.CFG.strict = False
    F.STATE.large_ext = Boom("CUDA error: an illegal memory access was encountered")
    try:
        m.apply(lay, xr(700, 4096))
        ck(False, "L.6 CUDA error text not re-raised")
    except RuntimeError:
        pass
    F.STATE.large_ext = real
    print("   strict mode raises; CUDA-error text re-raised")
    x = xr(1024, 4096)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    n0 = F.COUNTERS.get("large_marlin_capture", 0)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(gr, stream=s):
            yg = m.apply(lay, x)
    torch.cuda.synchronize()
    gr.replay()
    torch.cuda.synchronize()
    ok = F.COUNTERS.get("large_marlin_capture", 0) == n0 + 1 and torch.equal(yg, orig(m, lay, x))
    ck(ok, "L.6 capture")
    print(f"   CUDA-graph capture of an M=1024 call: Marlin captured, replay == production {ok}")
    del gr
    lay.glm53_bf16_lm_w = torch.zeros(1, device=dev)
    lay.glm53_bf16_lm_min_m = 1 << 30
    fallback("KDA large-M BF16 copy present (M=1024)", m, lay, xr(1024, 4096), None)
    del lay.glm53_bf16_lm_w, lay.glm53_bf16_lm_min_m
    mnr = method("kda", "model.layers.0.self_attn.in_proj_qkvbfg_a")
    laynr = L((torch.randn(12576, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16))
    fallback("method not ready (BF16 weight)", mnr, laynr, xr(1024, 4096), None)
    lay16, m16 = build(12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj", dtype=torch.float16)
    fallback("fp16 layer / activations", m16, lay16, xr(1024, 4096, torch.float16), None)
    del lay16, m16, laynr

    # ---------------------------------------------------------------- L.7 both on
    print("== L.7 GLM53_FP8_GEMV=1 + GLM53_FP8_LARGE_M=1 (MAX_M 16)")
    F.uninstall()
    os.environ["GLM53_FP8_GEMV"] = "1"
    os.environ["GLM53_FP8_GEMV_MAX_M"] = "16"
    rep = F.install(prod)
    ck(rep["installed"] and F.STATE.enabled and F.STATE.large, f"L.7 install {rep}")
    lay, m = build(12576, 4096)
    line = "   in_proj:"
    for M in (1, 8, 16, 17, 64, 65, 639, 640, 2202):
        c0 = (F.COUNTERS.get("new_eager", 0), F.COUNTERS.get("large_eager", 0))
        x = xr(M, 4096)
        y = m.apply(lay, x)
        c1 = (F.COUNTERS.get("new_eager", 0), F.COUNTERS.get("large_eager", 0))
        which = "gemv" if c1[0] > c0[0] else "large" if c1[1] > c0[1] else "marlin"
        want = "gemv" if M <= 16 else "large" if M >= 640 else "marlin"
        ck(which == want, f"L.7 M={M}: {which} != {want}")
        if which == "marlin":
            ck(torch.equal(y, orig(m, lay, x)), f"L.7 M={M}: != production")
        line += f" {M}:{which}"
    print(line)

    # ---------------------------------------------------------------- L.8 torch.compile
    print("== L.8 torch.compile(fullgraph=True, dynamic=True)")
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(nd.target) for nd in gm.graph.nodes if nd.op == "call_function")
        return gm.forward

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lay = lay

        def forward(self, x):
            return m.apply(self.lay, x) * 1.0

    torch._dynamo.reset()
    cmod = torch.compile(Mod(), backend=backend, fullgraph=True, dynamic=True)
    for M, want in ((8, "gemv"), (100, "marlin"), (1024, "large")):
        x = xr(M, 4096)
        c0 = (F.COUNTERS.get("new_eager", 0), F.COUNTERS.get("large_eager", 0))
        yc = cmod(x)
        c1 = (F.COUNTERS.get("new_eager", 0), F.COUNTERS.get("large_eager", 0))
        which = "gemv" if c1[0] > c0[0] else "large" if c1[1] > c0[1] else "marlin"
        ck(which == want and torch.equal(yc, m.apply(lay, x)), f"L.8 M={M}: {which} (want {want})")
    ck(any("tf_fp8.linear" in t for t in targets), f"L.8 targets {targets}")
    print(f"   targets {sorted(set(targets))}; M=8 gemv, 100 Marlin, 1024 large; == eager")

    # ---------------------------------------------------------------- L.9 memory
    print("== L.9 transient memory of one call (peak allocated above the inputs)")
    del built
    torch.cuda.empty_cache()
    for name, (n, k, grp, pre) in shapes.items():
        lay, m = build(n, k, grp, pre)
        for M in ((2202, 13824) if k == 4096 else (8192, 13824)):
            x = xr(M, k)
            res = {}
            def with_backend(be):
                def f():
                    keep = F.STATE.large_gemm
                    F.STATE.large_gemm = be
                    try:
                        return m.apply(lay, x)
                    finally:
                        F.STATE.large_gemm = keep
                return f
            for label, fn in (("marlin", lambda: orig(m, lay, x)), ("large", with_backend("tilelang")),
                              ("large_cublas", with_backend("cublas"))):
                fn()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                y = fn()
                torch.cuda.synchronize()
                res[label] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                del y
            outp = M * n * 2 / 2 ** 20
            ck(max(res["large"], res["large_cublas"]) <= outp + F.CFG.large_temp_mib * 1.05, f"L.9 {name} M={M}: {res}")
            print(f"   {name} M={M}: output {outp:.0f} MiB; peak extra: Marlin {res['marlin']:.0f} MiB, large "
                  f"{res['large']:.0f} MiB (tilelang) / {res['large_cublas']:.0f} MiB (cublas) (budget "
                  f"{F.CFG.large_temp_mib} MiB + output)")
            del x
        del lay, m
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- L.11 cublas backend, L.12 TileLang failure
    print("== L.11 GLM53_FP8_LARGE_M_GEMM=cublas: same results")
    lay, m = build(12576, 4096)
    xs = [xr(M, 4096) for M in (640, 2202, 13824)]
    y_tl = [m.apply(lay, x) for x in xs]
    F.uninstall()
    os.environ["GLM53_FP8_LARGE_M_GEMM"] = "cublas"
    rep = F.install(prod)
    ck(rep["installed"] and F.STATE.large_gemm == "cublas", f"L.11 install {rep} {F.STATE.large_gemm}")
    lay2, m2 = build(12576, 4096)
    lay2.weight.data.copy_(lay.weight.data)
    lay2.weight_scale.data.copy_(lay.weight_scale.data)
    F.LVERDICT.clear()
    F.LALPHA.clear()
    for x, yt in zip(xs, y_tl):
        yc = m2.apply(lay2, x)
        ck(torch.equal(yc, yt), f"L.11 M={x.shape[0]}: cublas backend != tilelang backend")
    print(f"   backend {F.STATE.large_gemm}; M 640 / 2202 / 13824 bitwise equal to the tilelang backend")
    del xs, y_tl, lay2, m2
    F.uninstall()
    os.environ.pop("GLM53_FP8_LARGE_M_GEMM")
    print("== L.12 TileLang unusable at the first self-test -> cuBLAS backend (WARNING), layer still served")
    rep = F.install(prod)
    real_build = F._tl_build

    def boom(k):
        raise RuntimeError("synthetic TileLang compile failure")

    F._tl_build = boom
    F.STATE.tl_kernels.clear()
    cap.clear()
    lay3, m3 = build(12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj")
    v = F.LVERDICT.get(F._key(lay3.weight, lay3.weight_scale, 4096))
    ck(F.STATE.large_gemm == "cublas" and cap.has(logging.WARNING, "TileLang GEMM of the large-M path failed") and
       v is not None and v[0] and "(cublas)" in v[1], f"L.12 fallback: {F.STATE.large_gemm} {v}")
    x = xr(2202, 4096)
    n0 = F.COUNTERS.get("large_eager", 0)
    y = m3.apply(lay3, x)
    rl2, same = agree(y, orig(m3, lay3, x))
    ck(F.COUNTERS.get("large_eager", 0) == n0 + 1 and rl2 <= 1e-3 and same >= 0.99, "L.12 served by cublas")
    F._tl_build = real_build
    print(f"   backend now {F.STATE.large_gemm}; self-test {v[1]}; M=2202 served (rel_l2 vs Marlin {rl2:.1e})")
    del lay3, m3, lay, m

    # ---------------------------------------------------------------- L.10 uninstall
    print("== L.10 uninstall")
    rep = F.uninstall()
    ck(rep["restored"] and cls.apply is orig_apply and cls.process_weights_after_loading is orig_pwal and
       not F.LVERDICT and not F.LALPHA and not F.STATE.large, "L.10")
    print(f"   restored {rep['restored']}, caches cleared")
    print("counters:", F.summary())
    ck.summary()


run_main(main)
