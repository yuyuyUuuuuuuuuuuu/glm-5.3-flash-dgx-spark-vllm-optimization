"""E.4: CUDA graph capture/replay of the REAL production apply_exl3_fused_moe with TF installed
(docs/DESIGN.md §E.4), on BOTH production serving paths:
  K2          the default install: production's apply is hooked and served from the router ids (moe_forward_ids);
  dispatcher  production's own apply -> its routing prelude -> exl3_moe = the TF dispatcher (moe_forward). Production
              runs this path when TF_EXL3_APPLY=0 (the apply hook is not installed) and after a K2 self-test mismatch
              (disable_apply: the hook is installed but hands every call to production's apply); both are tested.
Per path:
- layers built through production process_weights_after_loading -> hooked build_exl3_fused_state -> preflight
- warm up eagerly on a side stream, capture one graph per static T in {1, 8, 64, 128} (shared TF scratch;
  T = 128 = R is P = P_cap = 1024)
- replay the four graphs interleaved 100x with fresh x / ids / weights copied into the static inputs; every
  replay is compared with an eager production-orig result (E.1 tolerances)
- one graph holding TWO layers in sequence (n = 288 with sentinel ids, then n = 32 through an EP expert_map
  with half the experts non-local), both through the shared scratch, replayed 20x with fresh routing; each
  layer's output compared with eager orig (E.1)
- the path taken at every capture is identified by its own counter (tf_apply_calls counts K2 only; tf_calls counts
  both), so a capture served by the other path fails the check.
Then (dispatcher calls): no host sync in the TF path (sync debug mode "error", with a positive control), zero
allocations across 100 direct calls (allocator stats, positive control), a prefill-sized call (T = R + 1)
delegates, both eagerly and inside a captured graph.
"""
from __future__ import annotations

import os

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0
NB = 32


def graph_suite(mode, tf, prod, xl, orig, W, WB, ck, g, gr_rng):
    """Capture / replay the production apply on the path `mode` ("K2", "dispatcher (TF_EXL3_APPLY=0)",
    "dispatcher (K2 disabled by disable_apply)"); returns the n = 288 layer."""
    k2 = mode == "K2"
    layer = H.make_layer(prod, W)
    ck(tf.REG.get((0, layer._exl3_ptrs["gate_trellis"].data_ptr())) is not None, f"[{mode}] layer not registered")
    inners = layer._exl3_inners
    hooked = bool(getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False))
    k2_live = hooked and tf.STATE.apply_enabled and tf.CFG.apply
    print(f"[{mode}] production apply hooked: {hooked}; K2 serving: {k2_live}")
    ck(k2_live == k2, f"[{mode}] K2 serving state {k2_live} does not match the mode")

    def took(c_tf, c_ap, n):
        """n captures served by the expected path: tf_calls +n, tf_apply_calls +n on K2 and +0 on the dispatcher."""
        return tf.COUNTERS["tf_calls"] == c_tf + n and tf.COUNTERS["tf_apply_calls"] == c_ap + (n if k2 else 0)

    def fresh(T, gen):
        return (torch.randn(T, K, generator=gen).to(torch.bfloat16).to(dev),
                H.random_ids(T, NEXP, TOPK, gen, dev), H.random_weights(T, TOPK, gen, dev))

    dev = torch.device("cuda", 0)
    Ts = (1, 8, 64, 128)
    static = {T: fresh(T, g) for T in Ts}
    graphs, outs = {}, {}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for T in Ts:
            for _ in range(3):
                prod.apply_exl3_fused_moe(*static[T], layer, inners, None, LIMIT)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    pool = None
    for T in Ts:
        c_tf, c_ap = tf.COUNTERS["tf_calls"], tf.COUNTERS["tf_apply_calls"]
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, pool=pool):
            outs[T] = prod.apply_exl3_fused_moe(*static[T], layer, inners, None, LIMIT)
        pool = gr.pool()
        graphs[T] = gr
        chosen = took(c_tf, c_ap, 1)
        print(f"[{mode}] captured T={T}: TF served the capture on this path: {chosen}")
        ck(chosen, f"[{mode}] T={T}: TF not chosen at capture on this path")
    ck(tf.STATE.captured, "STATE.captured not set")

    worst = 0.0
    fails = 0
    for i in range(100):
        T = Ts[i % len(Ts)]
        x, ids, w = fresh(T, gr_rng)
        for dst, src in zip(static[T], (x, ids, w)):
            dst.copy_(src)
        graphs[T].replay()
        got = outs[T].clone()
        args = H.capture_args(prod, xl, x, ids, w, layer, LIMIT)
        ref = torch.zeros(T, K, dtype=torch.float32, device=dev)
        orig(*H.with_out(args, ref))
        torch.cuda.synchronize()
        m = tf.compare(got, ref)
        worst = max(worst, m["rel_l2"])
        if not tf.passes_e1(m):
            fails += 1
            print(f"  [{mode}] replay {i} T={T}: {m}")
    print(f"[{mode}] 100 interleaved replays (T in {Ts}): {100 - fails}/100 within E.1 tolerances, "
          f"worst rel_l2 {worst:.2e}")
    ck(fails == 0, f"[{mode}] {fails} replays out of tolerance")
    del graphs, outs

    # ---- two layers in ONE graph (shared scratch), sentinel ids and an EP expert_map, fresh routing ----
    layerB = H.make_layer(prod, WB)
    ck(tf.REG.get((0, layerB._exl3_ptrs["gate_trellis"].data_ptr())) is not None, f"[{mode}] layer B not registered")
    layerB.expert_map = torch.where(torch.arange(2 * NB) >= NB, torch.arange(2 * NB) - NB,
                                    torch.full((2 * NB,), -1))            # global 0..NB-1 are non-local
    emapB = prod.pin_exl3_expert_map(layerB, dev)
    TA, TB = 16, 8

    def fresh2(gen):
        xa = torch.randn(TA, K, generator=gen).to(torch.bfloat16).to(dev)
        ia = H.random_ids(TA, NEXP, TOPK, gen, dev)
        ia[torch.randperm(TA, generator=gen)[:5].to(dev), torch.randint(0, TOPK, (5,), generator=gen).to(dev)] = -1
        wa = H.random_weights(TA, TOPK, gen, dev)
        xb = torch.randn(TB, K, generator=gen).to(torch.bfloat16).to(dev)
        ib = torch.stack([torch.randperm(2 * NB, generator=gen)[:TOPK] for _ in range(TB)]).to(dev)
        wb = H.random_weights(TB, TOPK, gen, dev)
        return xa, ia, wa, xb, ib, wb

    st2 = fresh2(g)

    def two_layers():
        ya = prod.apply_exl3_fused_moe(st2[0], st2[1], st2[2], layer, inners, None, LIMIT)
        yb = prod.apply_exl3_fused_moe(st2[3], st2[4], st2[5], layerB, layerB._exl3_inners, emapB, LIMIT)
        return ya, yb

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            two_layers()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    c_tf, c_ap = tf.COUNTERS["tf_calls"], tf.COUNTERS["tf_apply_calls"]
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        y2 = two_layers()
    ck(took(c_tf, c_ap, 2), f"[{mode}] two-layer graph: TF not chosen for both layers at capture on this path")
    worst2, fails2 = 0.0, 0
    for i in range(20):
        new = fresh2(gr_rng)
        for dst, src in zip(st2, new):
            dst.copy_(src)
        g2.replay()
        got = [y.clone() for y in y2]
        for (x_, i_, w_, L_, em), y in zip(((new[0], new[1], new[2], layer, None), (new[3], new[4], new[5], layerB, emapB)),
                                          got):
            a2 = H.capture_args(prod, xl, x_, i_, w_, L_, LIMIT, expert_map=em)
            ref = torch.zeros(x_.shape[0], K, dtype=torch.float32, device=dev)
            orig(*H.with_out(a2, ref))
            torch.cuda.synchronize()
            m = tf.compare(y, ref)
            worst2 = max(worst2, m["rel_l2"])
            if not tf.passes_e1(m):
                fails2 += 1
                print(f"  [{mode}] two-layer replay {i}: {m}")
        layerB.expert_map = emapB
    print(f"[{mode}] two-layer graph (n=288 T={TA} with sentinel ids, then n=32 T={TB} through an EP map): 20 replays "
          f"x 2 layers, {40 - fails2}/40 within E.1 tolerances, worst rel_l2 {worst2:.2e}")
    ck(fails2 == 0, f"[{mode}] {fails2} two-layer replays out of tolerance")
    del g2, y2
    return layer


def main():
    H.gpu_guard(8.0)
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    W = H.Weights(NEXP, K, N, dev, seed=77)
    WB = H.Weights(NB, K, N, dev, seed=78)
    g = torch.Generator().manual_seed(3)
    gr_rng = torch.Generator().manual_seed(11)

    # (1) TF_EXL3_APPLY=0: install() reads it and leaves production's apply unhooked -> dispatcher path
    os.environ["TF_EXL3_APPLY"] = "0"
    try:
        rep = integrate.install(prodmod=prod, ext=xl, force=True)
    finally:
        os.environ.pop("TF_EXL3_APPLY", None)
    ck(rep["installed"] and not rep["apply_hook"], f"TF_EXL3_APPLY=0 install: {rep}")
    orig = rep["orig"]
    tf.CFG.strict = True
    cal = H.make_layer(prod, W)                         # calibrate svh_d so std(out) ~ 1 (in place)
    a = H.capture_args(prod, xl, torch.randn(16, K, generator=g).half().to(dev), H.random_ids(16, NEXP, TOPK, g, dev),
                       H.random_weights(16, TOPK, g, dev), cal, LIMIT)
    o = torch.zeros(16, K, dtype=torch.float32, device=dev)
    orig(*H.with_out(a, o))
    W.rescale_svh_d(1.0 / float(o.std()))
    del cal
    graph_suite("dispatcher (TF_EXL3_APPLY=0)", tf, prod, xl, orig, W, WB, ck, g, gr_rng)
    integrate.uninstall(prodmod=prod, ext=xl)

    # (2) the default install: K2
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep["installed"] and rep["apply_hook"], f"default install: {rep}")
    orig = rep["orig"]
    disp = xl.exl3_moe
    tf.CFG.strict = True
    graph_suite("K2", tf, prod, xl, orig, W, WB, ck, g, gr_rng)

    # (3) K2 disabled at runtime (what a K2 self-test mismatch does): the hook stays, every call goes to production
    tf.disable_apply("test_graph: simulated K2 self-test mismatch")
    layer = graph_suite("dispatcher (K2 disabled by disable_apply)", tf, prod, xl, orig, W, WB, ck, g, gr_rng)
    inners = layer._exl3_inners

    def fresh(T, gen):
        return (torch.randn(T, K, generator=gen).to(torch.bfloat16).to(dev),
                H.random_ids(T, NEXP, TOPK, gen, dev), H.random_weights(T, TOPK, gen, dev))

    # ---- no host sync in the TF path ----
    T = 8
    x, ids, w = fresh(T, g)
    args = H.capture_args(prod, xl, x.half(), ids, w, layer, LIMIT)
    out = torch.zeros(T, K, dtype=torch.float32, device=dev)
    a = H.with_out(args, out)
    disp(*a)                                                     # warm
    torch.cuda.synchronize()
    c0 = tf.COUNTERS["tf_calls"]
    torch.cuda.set_sync_debug_mode("error")
    err = None
    try:
        disp(*a)
    except RuntimeError as e:  # a synchronizing CUDA call inside the TF path
        err = e
    finally:
        torch.cuda.set_sync_debug_mode(0)
    control = False
    torch.cuda.set_sync_debug_mode("error")
    try:
        _ = out[0, 0].item()
    except RuntimeError:
        control = True
    finally:
        torch.cuda.set_sync_debug_mode(0)
    print(f"sync-debug 'error' around an eager TF call: raised={err is not None}; TF ran={tf.COUNTERS['tf_calls'] == c0 + 1};"
          f" positive control (.item()) raised={control}")
    ck(err is None and tf.COUNTERS["tf_calls"] == c0 + 1 and control, f"host sync in TF path: {err}")

    # ---- zero allocations across 100 direct calls ----
    torch.cuda.synchronize()
    s0 = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(100):
        disp(*a)
    torch.cuda.synchronize()
    s1 = torch.cuda.memory_stats()["allocation.all.allocated"]
    _tmp = torch.zeros(1, device=dev)
    s2 = torch.cuda.memory_stats()["allocation.all.allocated"]
    print(f"allocations across 100 dispatcher->TF calls: {s1 - s0} (positive control torch.zeros: {s2 - s1})")
    ck(s1 - s0 == 0 and s2 - s1 >= 1, "TF path allocated memory")

    # ---- prefill-sized call delegates (eager apply, and a captured bare call) ----
    R = int(layer._exl3_fused_temps[0].shape[1])
    xb, idsb, wb = fresh(R + 1, g)
    c0, d0 = tf.COUNTERS["tf_calls"], tf.COUNTERS["delegated"]
    prod.apply_exl3_fused_moe(xb, idsb, wb, layer, inners, None, LIMIT)
    torch.cuda.synchronize()
    eager_ok = tf.COUNTERS["tf_calls"] == c0 and tf.COUNTERS["delegated"] > d0
    ab = H.capture_args(prod, xl, xb, idsb, wb, layer, LIMIT)
    ob = torch.zeros(R + 1, K, dtype=torch.float32, device=dev)
    orig(*H.with_out(ab, ob))                                    # warm orig for capture
    ob.zero_()
    c0, d0 = tf.COUNTERS["tf_calls"], tf.COUNTERS["delegated"]
    gb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb):
        disp(*H.with_out(ab, ob))
    cap_ok = tf.COUNTERS["tf_calls"] == c0 and tf.COUNTERS["delegated"] == d0 + 1
    ob.zero_()
    gb.replay()
    ref = torch.zeros_like(ob)
    orig(*H.with_out(ab, ref))
    torch.cuda.synchronize()
    rel = float((ob - ref).norm() / ref.norm())
    print(f"T = R + 1 = {R + 1}: eager apply delegated={eager_ok}; captured dispatcher call delegated={cap_ok}; "
          f"replay vs eager orig rel {rel:.1e}")
    ck(eager_ok and cap_ok and rel <= 1e-6, "prefill-sized call did not delegate cleanly")

    integrate.uninstall(prodmod=prod, ext=xl)
    ck(len(tf._SCRATCH) >= 1, "scratch freed although a graph captured it")
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
