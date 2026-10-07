"""Reviewer's adversarial check of moeglue (GLM53_DEC_MOEGLUE) and warm (GLM53_DEC_MOEGLUE_WARM), independent of
tests/test_moeglue*.py:

  (1) glue vs production's apply (K2, unhooked) vs production's python loop (a third implementation) vs a float64
      reference of the EXL3 layer definition (kernels/exl3_format_ref.py, dequantized trellises), n = 288 experts
      (routing restricted to 16 experts spread over 0..287 so the fp64 reference stays cheap), bf16 x, int32 router
      ids, fp32 / bf16 weights, T in {1, 2, 3, 5, 7, 8, 13, 32, 64}, sentinel -1 routes, an all-sentinel token,
      3-D x, row-strided x, misaligned x, non-contiguous ids / weights (served or delegated: the result must match
      production either way)
  (2) production's own run-to-run determinism (is K2's atomics order really not deterministic?)
  (3) warm + glue in one CUDA graph: M in {1, 5, 8, 32} o_proj (production Unquantized apply) -> stand-in -> hooked
      MoE, 20 replays == eager bitwise; and the M > WARM_MAX_M path forks nothing
"""
from __future__ import annotations

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0
SUBSET = [0, 1, 17, 40, 63, 64, 100, 127, 128, 160, 199, 200, 255, 256, 280, 287]


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import exl3_format_ref as R
    import glm53_moeglue as MG
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep.get("apply_hook") is True, f"K2 apply hook: {rep}")
    r2 = MG.install(prodmod=prod, force=True, warm=True, hook_loader=False)
    ck(r2["installed"] and r2["warm"], f"moeglue install: {r2}")
    W = H.Weights(NEXP, K, N, dev, seed=7311)
    layer = H.make_layer(prod, W)
    v = MG.VERDICT.get((0, layer._exl3_ptrs["gate_trellis"].data_ptr()))
    ck(v is not None and v[0], f"self-test verdict {v}")
    R_rows = int(layer._exl3_fused_temps[0].shape[1])
    # scale the down projection so the layer output is O(1) (as test_e2e_vs_f64 does)
    g = torch.Generator().manual_seed(99)
    x0 = torch.randn(16, K, generator=g).to(torch.bfloat16).to(dev)
    ids0 = torch.tensor([[SUBSET[(t + k) % 16] for k in range(TOPK)] for t in range(16)], dtype=torch.int32, device=dev)
    MG.STATE.enabled = False
    o0 = prod.apply_exl3_experts(x0, ids0, H.random_weights(16, TOPK, g, dev), layer, limit=LIMIT)
    MG.STATE.enabled = True
    W.rescale_svh_d(1.0 / float(o0.float().std()))

    print("unpacking 16 x 3 trellises (float64 reference) ...", flush=True)
    Wq, sc = {}, {}
    for e in SUBSET:
        Wq[e] = (R.unpack(W.w13_trellis[e, 0].cpu(), 4).double(), R.unpack(W.w13_trellis[e, 1].cpu(), 4).double(),
                 R.unpack(W.w2_trellis[e].cpu(), 4).double())
        sc[e] = tuple(t.double().cpu() for t in (W.w13_suh[e, 0], W.w13_svh[e, 0], W.w13_suh[e, 1], W.w13_svh[e, 1],
                                                 W.w2_suh[e], W.w2_svh[e]))

    def lin(xv, wq, suh, svh):
        return R.rotate(R.rotate(xv * suh, -1) @ wq, -1) * svh

    def reference(x, ids, w):
        xs = x.float().half().double().cpu()             # the EXL3 kernels take x in fp16
        w16 = w.float().to(torch.float16).double().cpu()  # production: weights.to(fp16)
        ids = ids.long().cpu()
        out = torch.zeros(xs.shape[0], K, dtype=torch.float64)
        for t in range(xs.shape[0]):
            for k in range(ids.shape[1]):
                e = int(ids[t, k])
                if e < 0:
                    continue
                wg, wu, wd = Wq[e]
                suh_g, svh_g, suh_u, svh_u, suh_d, svh_d = sc[e]
                gg = lin(xs[t:t + 1], wg, suh_g, svh_g)
                uu = lin(xs[t:t + 1], wu, suh_u, svh_u)
                s = torch.clamp(gg * torch.sigmoid(gg), max=LIMIT)
                uu = torch.clamp(uu, min=-LIMIT, max=LIMIT)
                d = lin(s * uu, wd, suh_d, svh_d)
                out[t] += w16[t, k] * d[0]
        return out

    def served_count():
        return sum(v_ for k_, v_ in MG.COUNTERS.items() if k_.startswith("served"))

    def run(x, ids, w, glue: bool):
        MG.STATE.enabled = glue
        c0 = served_count()
        y = prod.apply_exl3_experts(x, ids, w, layer, limit=LIMIT)
        s = served_count() != c0
        MG.STATE.enabled = True
        return y, s

    def loop(x, ids, w):
        MG.STATE.enabled = False
        y = prod.apply_exl3_experts(x, ids, w, layer, limit=LIMIT, fused=False)
        MG.STATE.enabled = True
        return y

    gg_ = torch.Generator().manual_seed(2718)
    worst = dict(glue=0.0, prod=0.0, loop=0.0)
    worst_gp_ulp, worst_gp_rel, worst_gp_abs = 0.0, 0.0, 0.0
    nondet_prod, nondet_glue, n_cases, served_cases = 0, 0, 0, 0
    rows = []
    for T in (1, 2, 3, 5, 7, 8, 13, 32, 64):
        for variant in ("plain", "sentinel", "wbf16", "x3d", "xstrided", "xmisaligned", "ids_noncontig", "w_noncontig"):
            if variant == "sentinel" and T < 2:
                continue
            ids = torch.stack([torch.tensor(SUBSET)[torch.randperm(16, generator=gg_)[:TOPK]] for _ in range(T)])
            ids = ids.to(torch.int32)
            if variant == "sentinel":
                ids[0, 3] = -1
                ids[T - 1, :] = -1                       # a token with no route at all
            w = H.random_weights(T, TOPK, gg_, "cpu").float()
            if variant == "wbf16":
                w = w.to(torch.bfloat16)
            xb = torch.randn(T, K, generator=gg_).to(torch.bfloat16)
            ref = reference(xb, ids, w)
            ids_d, w_d = ids.to(dev), w.to(dev)
            if variant == "x3d":
                x_d = xb.to(dev).reshape(1, T, K)
            elif variant == "xstrided":
                big = torch.zeros(T, K + 64, dtype=torch.bfloat16, device=dev)
                big[:, :K] = xb.to(dev)
                x_d = big[:, :K]
            elif variant == "xmisaligned":
                flat = torch.zeros(T * K + 1, dtype=torch.bfloat16, device=dev)
                flat[1:].copy_(xb.reshape(-1).to(dev))
                x_d = flat[1:].view(T, K)
            else:
                x_d = xb.to(dev)
            if variant == "ids_noncontig":
                bi = torch.full((T, 2 * TOPK), -7, dtype=torch.int32, device=dev)
                bi[:, ::2] = ids_d
                ids_d = bi[:, ::2]
            if variant == "w_noncontig":
                bw = torch.zeros(T, 2 * TOPK, dtype=torch.float32, device=dev)
                bw[:, ::2] = w_d
                w_d = bw[:, ::2]
            try:
                yg, served = run(x_d, ids_d, w_d, True)
                yg2, _ = run(x_d, ids_d, w_d, True)
                yp, _ = run(x_d, ids_d, w_d, False)
                yp2, _ = run(x_d, ids_d, w_d, False)
                yl = loop(x_d, ids_d, w_d)
            except Exception as exc:  # noqa: BLE001
                print(f"  T={T} {variant}: raised {type(exc).__name__}: {str(exc)[:160]}")
                ck(False, f"T={T} {variant} raised {exc!r}")
                continue
            torch.cuda.synchronize()
            n_cases += 1
            served_cases += served
            nondet_prod += not torch.equal(yp, yp2)
            if served:                                   # a delegated call runs production's (non-deterministic) K2
                nondet_glue += not torch.equal(yg, yg2)
                if not torch.equal(yg, yg2):
                    print(f"  glue not deterministic at T={T} {variant}")
            rn = max(float(ref.norm()), 1e-30)
            e = {k_: float((y.reshape(T, K).double().cpu() - ref).norm()) / rn for k_, y in
                 (("glue", yg), ("prod", yp), ("loop", yl))}
            for k_ in worst:
                worst[k_] = max(worst[k_], e[k_])
            d = (yg.float() - yp.float()).reshape(T, K)
            rel = float(d.norm()) / max(float(yp.float().norm()), 1e-30)
            rowmax = yp.float().reshape(T, K).abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            ulp = float((d.abs() / rowmax).max()) * 256.0
            worst_gp_rel, worst_gp_ulp = max(worst_gp_rel, rel), max(worst_gp_ulp, ulp)
            worst_gp_abs = max(worst_gp_abs, float(d.abs().max()))
            ok = (yg.shape == yp.shape and yg.dtype == yp.dtype and e["glue"] <= 1.05 * e["prod"] + 1e-5
                  and rel <= 1e-4 and ulp <= 2.0)
            rows.append((T, variant, served, e["glue"], e["prod"], e["loop"], rel, ulp))
            print(f"  T={T:3d} {variant:13s} served {int(served)}  e_glue {e['glue']:.3e}  e_prod {e['prod']:.3e}  "
                  f"e_loop {e['loop']:.3e}  glue-prod rel {rel:.2e} max|d| {float(d.abs().max()):.2e} "
                  f"({ulp:.2f} ulp@rowmax) {'ok' if ok else 'FAIL'}", flush=True)
            ck(ok, f"T={T} {variant}")
    print(f"(1) {n_cases} cases ({served_cases} served by glue): worst rel_l2 vs fp64: glue {worst['glue']:.3e}, "
          f"production K2 {worst['prod']:.3e}, production loop {worst['loop']:.3e}; glue vs production: rel_l2 "
          f"{worst_gp_rel:.2e}, max|d| {worst_gp_abs:.2e}, {worst_gp_ulp:.2f} bf16 ulp at the row max")
    print(f"(2) run-to-run: production K2 differed in {nondet_prod}/{n_cases} cases, glue in {nondet_glue}/"
          f"{served_cases} served cases")
    ck(nondet_glue == 0, "glue must be deterministic")

    # ---- (3) warm + glue in one CUDA graph over a production Unquantized o_proj --------------------------------
    import types as _t

    lin_o = torch.nn.Module()
    lin_o.weight = torch.nn.Parameter((torch.randn(K, 2048, device=dev) * 0.02).to(torch.bfloat16), requires_grad=False)
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    qm = UnquantizedLinearMethod()
    lin_o.quant_method = qm
    fn = torch.randn(24, 4 * K, device=dev)
    rg = (torch.randn(NEXP, K, device=dev) * 0.02).to(torch.bfloat16)
    regs = MG._warm_regions(_t.SimpleNamespace(hc_ffn_fn=fn), _t.SimpleNamespace(gate=_t.SimpleNamespace(weight=rg),
                            shared_experts=None), "frgd", 16 * 2 ** 20)
    MG.WARM.side[0] = MG.WARM.side.get(0) or torch.cuda.Stream(device=dev)
    MG.WARM.sink[0] = MG.WARM.sink.get(0) if MG.WARM.sink.get(0) is not None else torch.zeros(4, dtype=torch.int32,
                                                                                               device=dev)
    MG.WARM.handles.append(MG._WarmHandle(qm, lin_o, UnquantizedLinearMethod, regs, K, "rv"))
    ck(MG._wrap_o_proj(lin_o, len(MG.WARM.handles) - 1, K), "wrap o_proj")
    bad_graph = 0
    for T in (1, 5, 8, 32):
        xa = torch.randn(T, 2048, device=dev).to(torch.bfloat16)
        ids_s = torch.stack([torch.tensor(SUBSET)[torch.randperm(16, generator=gg_)[:TOPK]] for _ in range(T)])
        ids_s = ids_s.to(torch.int32).to(dev)
        ws = H.random_weights(T, TOPK, gg_, dev).float()

        def step():
            h = lin_o.quant_method.apply(lin_o, xa, None)
            h = (h.float() * 0.5).to(torch.bfloat16)          # stand-in for all-reduce + mHC
            return prod.apply_exl3_experts(h, ids_s, ws, layer, limit=LIMIT)

        c0 = MG.COUNTERS.get("warm_captured", 0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yo = step()
        forked = MG.COUNTERS.get("warm_captured", 0) - c0
        ck(forked == 1 and not any(MG.WARM.pending.values()), f"T={T}: {forked} forks captured, pending "
                                                                f"{MG.WARM.pending}")
        for _ in range(20):
            xa.copy_(torch.randn(T, 2048, device=dev).to(torch.bfloat16))
            gr.replay()
            ye = step()
            torch.cuda.synchronize()
            bad_graph += not torch.equal(yo, ye)
        del gr
    print(f"(3) warm + glue captured (1 fork + 1 join per graph, M = 1, 5, 8, 32): {bad_graph} of 80 replays differ "
          f"from eager")
    ck(bad_graph == 0, "(3) graph replays differ")
    c0 = MG.COUNTERS.get("warm_skip_m", 0)
    lin_o.quant_method.apply(lin_o, torch.randn(65, 2048, device=dev).to(torch.bfloat16), None)
    ck(MG.COUNTERS.get("warm_skip_m", 0) == c0 + 1 and not any(MG.WARM.pending.values()), "M = 65 must not fork")
    print("(3) M = 65 o_proj: no fork")

    # ---- (4) a stale EAGER fork (an o_proj call whose MoE never ran, e.g. an exception mid-forward) followed by a
    # CUDA-graph capture of o_proj -> MoE: the capture must still succeed (the stale fork must not poison it)
    xa = torch.randn(5, 2048, device=dev).to(torch.bfloat16)
    ids_s = torch.stack([torch.tensor(SUBSET)[torch.randperm(16, generator=gg_)[:TOPK]] for _ in range(5)])
    ids_s = ids_s.to(torch.int32).to(dev)
    ws = H.random_weights(5, TOPK, gg_, dev).float()

    def step5():
        h = lin_o.quant_method.apply(lin_o, xa, None)
        return prod.apply_exl3_experts(h, ids_s, ws, layer, limit=LIMIT)

    step5()
    torch.cuda.synchronize()
    lin_o.quant_method.apply(lin_o, xa, None)            # eager fork, no MoE after it
    stale = bool(MG.WARM.pending.get(0))
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            y4 = step5()
        gr.replay()
        torch.cuda.synchronize()
        ok4 = torch.equal(y4, step5())
        err4 = None
    except Exception as exc:  # noqa: BLE001
        ok4, err4 = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        MG.WARM.pending.clear()
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
    print(f"(4) stale eager fork (pending {stale}) then capture: {'ok' if ok4 else 'FAILED: ' + str(err4)}")
    ck(ok4, f"(4) capture after a stale eager fork: {err4}")
    H.report_peak(8.0)
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
