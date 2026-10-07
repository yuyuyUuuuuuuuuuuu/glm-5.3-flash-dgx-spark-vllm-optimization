"""Reviewer's adversarial checks of GLM53_FP8_LARGE_M (branch fp8largem), against the real production module.

A  decode untouched: GLM53_FP8_GEMV=1 (MAX_M 16 and 64) with and without the large path -> bitwise identical outputs
   for every production FP8 shape at M 1..639 (eager) and in CUDA-graph replays (M 1, 8, 16, 64);
   LARGE only (GEMV off) -> bitwise production's apply for M 1..639.
B  production T sweep (in_proj, dense gate_up, drafter fc): M 640 .. 16384 incl. mixed-batch odd sizes; served?,
   finite, vs Marlin (bitwise %, > 1 ulp %, rel_l2), with normal / heavy-outlier / tiny-magnitude activations; rows vs
   float64 for new and Marlin (error / (0.5 ulp + fp32 bound)).
C  row invariance through apply: rows of an M=13824 call == the same rows of an M=1791 / 700 call.
D  transient memory at M = 16384 (in_proj, gate_up, fc) vs Marlin.
E  exception type the TileLang kernel raises on bad input (does try_large's fallback catch it?).
F  kernel source: are the global stores / loads predicated for partial tiles?
G  8 distinct in_proj layers interleaved (M 13824 / 1791): per-call ms Marlin vs large (A/B rounds).
"""
import inspect
import logging
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard, load_prod  # noqa: E402
import torch  # noqa: E402
from test_fp8_integrate import single_rank_tp, L, Cap  # noqa: E402
from fp8_bench_common import PROD_SHAPES, quantize_like_prod  # noqa: E402

dev = "cuda"
SECTIONS = set((os.environ.get("RV_SECTIONS") or "ABCDEFG"))


def bf16_ulp(a):
    e = torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -126)))
    return torch.pow(2.0, e - 7)


def cmp(y, ym):
    d = (y.float() - ym.float()).abs() / bf16_ulp(ym.float())
    same = (y == ym).float().mean().item()
    gt1 = (d > 1).float().mean().item()
    mx = d.max().item()
    rl2 = ((y.float() - ym.float()).norm() / ym.float().norm().clamp(min=1e-30)).item()
    return same, gt1, mx, rl2


def f64_ratio(y, x, fp8, sc, rows):
    n, k = fp8.shape
    gam = (k / 16 + 2) * 2.0 ** -23
    xd = x[rows].double()
    worst, exact, tot = 0.0, 0, 0
    for a in range(0, n, 4096):
        deq = fp8[a:a + 4096].double() * sc[a:a + 4096].double()[:, None]
        ref = xd @ deq.t()
        S = xd.abs() @ deq.abs().t()
        err = (y[rows, a:a + 4096].double() - ref).abs()
        fb = S * gam + ref.abs() * 2.0 ** -24
        allowed = 0.5 * bf16_ulp(ref.abs() + fb) + fb + 1e-300
        worst = max(worst, (err / allowed).max().item())
        exact += (err <= 0.5 * bf16_ulp(ref) * (1 + 1e-9)).sum().item()
        tot += err.numel()
    return worst, exact / tot


def main():
    gpu_guard(8.0)
    ck = Checks()
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "GLM53_FP8_LARGE_M", "GLM53_FP8_LARGE_M_TEMP_MIB",
              "GLM53_FP8_LARGE_M_GEMM", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M"):
        os.environ.pop(v, None)
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    cap = Cap()
    logging.getLogger("vllm.tf_fp8_gemv").addHandler(cap)
    cls = prod.Glm53DenseFp8Method
    orig_apply = cls.apply
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    g = torch.Generator(device=dev).manual_seed(7)

    def method(group, prefix):
        return cls(group, prefix) if takes_prefix else cls(group)

    groups = {"kda": "model.layers.0.self_attn.in_proj_qkvbfg_a", "mla": "model.layers.3.self_attn.q_b_proj",
              "dense": "model.layers.1.mlp.gate_up_proj", "shared": "model.layers.5.mlp.shared_experts.down_proj",
              "draft": "model.layers.45.mlp.gate_up_proj"}

    def build(n, k, grp, pre):
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
             torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        lay, m = L(w), method(grp, pre)
        m.process_weights_after_loading(lay)
        return lay, m

    def xr(M, k, kind="normal"):
        x = torch.randn(M, k, device=dev, generator=g)
        if kind == "outlier":
            idx = torch.randperm(k, device=dev, generator=g)[:8]
            x[:, idx] *= 300.0
        elif kind == "tiny":
            x = x * 1e-4
        elif kind == "rows":        # per-row magnitudes over 6 decades
            x = x * torch.pow(10.0, torch.empty(M, 1, device=dev).uniform_(-4, 2, generator=g))
        return x.to(torch.bfloat16)

    # ---------------------------------------------------------------- install both
    os.environ["GLM53_FP8_GEMV"] = "1"
    os.environ["GLM53_FP8_GEMV_MAX_M"] = "16"
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    rep = F.install(prod)
    print("install:", rep, "backend", F.STATE.large_gemm, flush=True)
    ck(rep["installed"] and F.STATE.enabled and F.STATE.large and F.STATE.large_gemm == "tilelang", "install")

    if "A" in SECTIONS:
        print("== A decode untouched", flush=True)
        shapes = dict(PROD_SHAPES)
        shapes["draft.fc"] = (4096, 20480, 1)
        built = {}
        seen = set()
        for name, (n, k, _) in shapes.items():
            if (n, k) in seen:
                continue
            seen.add((n, k))
            grp = "draft" if name.startswith("draft") else name.split(".")[0]
            pre = "model.fc" if name == "draft.fc" else ("lm_head" if name == "draft.lmhead" else groups[grp])
            built[name] = build(n, k, grp, pre)
        Ms = (1, 2, 3, 5, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 100, 128, 200, 255, 256, 511, 639)
        for max_m in (16, 64):
            F.CFG.max_m = max_m
            bad = 0
            tot = 0
            for name, (lay, m) in built.items():
                k = lay.weight.shape[0] * 16
                for M in Ms:
                    x = xr(M, k)
                    F.STATE.large = True
                    y1 = m.apply(lay, x)
                    F.STATE.large = False
                    y0 = m.apply(lay, x)
                    F.STATE.large = True
                    tot += 1
                    if not torch.equal(y0, y1):
                        bad += 1
                        print(f"   MISMATCH max_m={max_m} {name} M={M}")
            ck(bad == 0, f"A eager GEMV+LARGE vs GEMV-only max_m={max_m}: {bad}/{tot} differ")
            print(f"   GEMV_MAX_M={max_m}: {tot} (shape, M) pairs, GEMV+LARGE == GEMV only bitwise: {bad == 0}",
                  flush=True)
        F.CFG.max_m = 16
        # CUDA graphs
        badg = 0
        totg = 0
        for name, (lay, m) in built.items():
            k = lay.weight.shape[0] * 16
            for M in (1, 8, 16, 64):
                x = xr(M, k)
                outs = []
                for large in (True, False):
                    F.STATE.large = large
                    m.apply(lay, x)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(s):
                        with torch.cuda.graph(gr, stream=s):
                            yg = m.apply(lay, x)
                    torch.cuda.synchronize()
                    yg.zero_()
                    gr.replay()
                    torch.cuda.synchronize()
                    outs.append(yg.clone())
                    del gr
                F.STATE.large = True
                totg += 1
                if not torch.equal(outs[0], outs[1]):
                    badg += 1
                    print(f"   GRAPH MISMATCH {name} M={M}")
        ck(badg == 0, f"A graph replays differ {badg}/{totg}")
        print(f"   CUDA-graph replays (M 1/8/16/64, every shape): GEMV+LARGE == GEMV only: {badg == 0} ({totg})",
              flush=True)
        # LARGE only
        F.STATE.enabled = False
        bad = 0
        tot = 0
        for name, (lay, m) in built.items():
            k = lay.weight.shape[0] * 16
            for M in Ms:
                x = xr(M, k)
                tot += 1
                if not torch.equal(m.apply(lay, x), orig_apply(m, lay, x, None)):
                    bad += 1
                    print(f"   LARGE-only MISMATCH {name} M={M}")
        F.STATE.enabled = True
        ck(bad == 0, f"A LARGE only vs production {bad}/{tot}")
        print(f"   LARGE only (GEMV off): M 1..639 == production apply: {bad == 0} ({tot})", flush=True)
        del built
        torch.cuda.empty_cache()

    if "B" in SECTIONS or "C" in SECTIONS or "D" in SECTIONS:
        cases = {"kda.in_proj": (12576, 4096, "kda", groups["kda"]),
                 "dense.gate_up": (12288, 4096, "dense", groups["dense"]),
                 "draft.fc": (4096, 20480, "draft", "model.fc")}
        for name, (n, k, grp, pre) in cases.items():
            w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
                 torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
            lay, fp8, sc = quantize_like_prod(w)
            del w
            m = method(grp, pre)
            m.ready = True
            min_m = F.LARGE_TABLE[(lay.weight.shape[1] // 4, k)]
            if "B" in SECTIONS:
                print(f"== B {name} (min M {min_m})", flush=True)
                Ms = [640, 641, 767, 1024, 1791, 2202, 4097, 8191, 8192, 13824, 13831, 16384]
                for M in Ms:
                    for kind in (("normal", "outlier", "tiny", "rows") if M in (min_m, 13824) else ("outlier",)):
                        x = xr(M, k, kind)
                        c0 = F.COUNTERS.get("large_eager", 0)
                        y = m.apply(lay, x)
                        served = F.COUNTERS.get("large_eager", 0) - c0
                        ym = orig_apply(m, lay, x, None)
                        ck(served == (1 if M >= min_m else 0), f"B {name} M={M}: served {served}")
                        ck(y.shape == ym.shape and y.dtype == ym.dtype and y.is_contiguous() == ym.is_contiguous(),
                           f"B {name} M={M}: shape/dtype/contig {y.shape} {ym.shape} {y.is_contiguous()} "
                           f"{ym.is_contiguous()}")
                        fin = bool(torch.isfinite(y).all().item()) == bool(torch.isfinite(ym).all().item())
                        same, gt1, mx, rl2 = cmp(y, ym)
                        line = (f"   M={M:5d} {kind:7s} served {served}: =Marlin {same * 100:.3f}% >1ulp "
                                f"{gt1 * 100:.4f}% max {mx:.1f} ulp rl2 {rl2:.1e} finite-match {fin}")
                        if M in (min_m, 13824) or kind != "outlier":
                            rows = torch.cat([torch.arange(0, 8, device=dev), torch.arange(M - 8, M, device=dev)])
                            r1, e1 = f64_ratio(y, x, fp8, sc, rows)
                            r2, e2 = f64_ratio(ym, x, fp8, sc, rows)
                            line += f" | f64 bound ratio new {r1:.3f} marlin {r2:.3f}; exact new {e1*100:.2f}% marlin {e2*100:.2f}%"
                            ck(r1 <= 1.0, f"B {name} M={M} {kind}: new outside bound {r1}")
                        if served:
                            ck(fin and rl2 <= 1e-3 and gt1 <= 1e-3, f"B {name} M={M} {kind}: {line}")
                        print(line, flush=True)
                        del x, y, ym
            if "C" in SECTIONS:
                x = xr(13824, k, "outlier")
                y = m.apply(lay, x)
                for Mp in (1791, 700):
                    if Mp < min_m:
                        continue
                    yp = m.apply(lay, x[:Mp].contiguous())
                    ok = torch.equal(yp, y[:Mp])
                    ck(ok, f"C {name} rows invariance M={Mp}")
                    print(f"== C {name}: rows 0..{Mp} of M=13824 == M={Mp} call: {ok}", flush=True)
                del x, y
            if "D" in SECTIONS:
                for M in (13824, 16384):
                    x = xr(M, k)
                    res = {}
                    for label, fn in (("marlin", lambda: orig_apply(m, lay, x, None)), ("large", lambda: m.apply(lay, x))):
                        fn()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        yy = fn()
                        torch.cuda.synchronize()
                        res[label] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                        del yy
                    print(f"== D {name} M={M}: peak extra Marlin {res['marlin']:.0f} MiB, large {res['large']:.0f} MiB "
                          f"(output {M * n * 2 / 2**20:.0f} MiB)", flush=True)
                    del x
            del lay, fp8, sc, m
            torch.cuda.empty_cache()

    if "E" in SECTIONS:
        print("== E TileLang kernel exception types", flush=True)
        kern = F._tl_kernel(4096)
        A = torch.randn(256, 4096, device=dev).to(torch.bfloat16)
        B = torch.randn(256, 4096, device=dev).to(torch.bfloat16)
        S = torch.ones(256 + 256, device=dev)
        C = torch.empty(256, 256, dtype=torch.bfloat16, device=dev)
        for label, args in (("fp16 A", (A.half(), B, S[:256], C)), ("fp32 S wrong dtype bf16", (A, B, S[:256].bfloat16(), C)),
                            ("K mismatch", (A[:, :2048], B[:, :2048], S[:256], C)),
                            ("non-contig A", (torch.randn(256, 8192, device=dev).to(torch.bfloat16)[:, ::2], B, S[:256], C)),
                            ("cpu C", (A, B, S[:256], C.cpu()))):
            try:
                kern(*args)
                torch.cuda.synchronize()
                print(f"   {label}: no exception")
            except BaseException as exc:  # noqa: BLE001
                caught = isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError))
                cuda_txt = bool(F._CUDA_ERR.search(str(exc)))
                print(f"   {label}: {type(exc).__module__}.{type(exc).__name__} mro-caught-by-try_large {caught}, "
                      f"CUDA-regex {cuda_txt}: {str(exc).splitlines()[0][:120] if str(exc) else ''}")

    if "F" in SECTIONS:
        print("== F kernel source", flush=True)
        kern = F._tl_kernel(4096)
        src = None
        for attr in ("get_kernel_source", "get_host_source"):
            try:
                src = getattr(kern, attr)()
                break
            except Exception as exc:  # noqa: BLE001
                print("   ", attr, "failed", exc)
        if src:
            p = Path("/w/tests/logs/rv_kernel_src.cu")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(src)
            print(f"   source {len(src)} chars -> {p}; 'if (' count {src.count('if (')}; 'cp.async' "
                  f"{src.count('cp_async') + src.count('cp.async')}; tma {src.count('tma')}")

    if "G" in SECTIONS:
        print("== G 8 distinct in_proj layers, interleaved", flush=True)
        n, k = 12576, 4096
        layers = [build(n, k, "kda", groups["kda"]) for _ in range(8)]
        for M in (13824, 1791, 700):
            x = xr(M, k, "outlier")
            fns = {"marlin": lambda: [orig_apply(m, lay, x, None) for lay, m in layers],
                   "large": lambda: [m.apply(lay, x) for lay, m in layers]}
            for f in fns.values():
                f()
            torch.cuda.synchronize()
            ts = {a: [] for a in fns}
            for r in range(5):
                for a in (("marlin", "large") if r % 2 == 0 else ("large", "marlin")):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    fns[a]()
                    e1.record()
                    torch.cuda.synchronize()
                    ts[a].append(e0.elapsed_time(e1) / len(layers))
            med = {a: statistics.median(v) for a, v in ts.items()}
            spr = {a: (max(v) - min(v)) / statistics.median(v) * 100 for a, v in ts.items()}
            print(f"   M={M}: Marlin {med['marlin']:.2f} ms/call (spread {spr['marlin']:.0f}%), large "
                  f"{med['large']:.2f} ms/call (spread {spr['large']:.0f}%) -> x{med['marlin'] / med['large']:.2f}",
                  flush=True)
            del x
        del layers
    print("counters:", F.summary())
    ck.summary()


run_main(main)
