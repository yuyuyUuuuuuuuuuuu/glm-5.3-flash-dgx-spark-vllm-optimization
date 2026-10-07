"""GLM53_PREFILL_FUSED_CAP (glm53_prefill_cap.py, docs/PREFILL_CAP.md): wiring, guards, parity, float64 accuracy.

P.1 wiring: env parsing (unset / empty / 0 / off -> inert; invalid -> WARNING, not installed), the fingerprints of the
    production functions it relies on, refusal on an unverified module, install order with the TF K2 apply hook
    (both orders; K2 still installs; uninstall restores production's function object).
P.2 hook guards (fake production apply, no kernels): tokens <= cap untouched (same temps object), tokens > cap on a
    grouped layer swapped to (concurrency, n, .) temps and restored, restored on exception, pass-through for another
    tier / EXL3_FAT_GROUPED=0 / EXL3_MOE_ROW_TILE=1 / n >= cap / during CUDA graph capture.
P.3 decode unchanged (real kernels): T = 1, 8, 64, 256 through production's apply with the wrapper at n = 16 -> no swap;
    a CUDA graph captured at T = 64 through the full stack (wrapper -> K2) is served by TF and replays = eager.
P.4 parity vs production's cap-256 path, production shapes (288 experts, 4096 / 1024, top-8), T in {1791, 4608, 13824},
    routing real / collapsed, n in PARITY_N. Noise floor: the cap-256 path run twice (E3's fp32 atomics).
    E3 scratch: same capacity before / after every n, never grown.
P.5 float64 reference (kernels/exl3_format_ref.py definition, dequantized with exllamav3_ext.reconstruct which equals
    the reference unpack bit for bit, checked here), 96 sampled tokens:
    (a) balanced: T = 2048, 64 experts x exactly 256 rows -> cap 256 = every expert on the thin kernel, n = 1 / 16 =
        every expert on E3: the pure thin-vs-E3 difference class;
    (b) real routing T = 1791, n = 1 / 16 vs cap 256.
"""
from __future__ import annotations

import logging
import os
import types

import torch

import prefill_cap_common as C
import harness as H

PARITY_N = tuple(int(v) for v in os.environ.get("PARITY_N", "1,8,16,32,48,64,96,128").split(","))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def row_rel_max(a, b):
    d = (a.double() - b.double()).norm(dim=1)
    r = b.double().norm(dim=1).clamp_min(1e-30)
    return float((d / r).max())


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, rec):
        self.msgs.append((rec.levelno, rec.getMessage()))


def p1_wiring(ck, prod, xl, PC, integrate):
    print("P.1 wiring", flush=True)
    for raw, want in ((None, 0), ("", 0), ("0", 0), ("off", 0), (" 16 ", 16), ("1", 1)):
        env = {} if raw is None else {PC.ENV: raw}
        ck(PC.parse_env(env) == want, f"parse_env({raw!r}) != {want}")
    for raw in ("abc", "-3", "1.5"):
        try:
            PC.parse_env({PC.ENV: raw})
            ck(False, f"parse_env({raw!r}) did not raise")
        except ValueError:
            pass
    fps = {n: PC.source_fingerprint(PC._unwrap(getattr(prod, n))) for n in PC.VERIFIED_FINGERPRINTS}
    print(f"  production fingerprints: {fps}", flush=True)
    ok, why = PC.compatibility(prod)
    ck(ok, f"compatibility: {why}")

    cap = _Cap()
    lg = logging.getLogger("vllm.glm53_prefill_cap")
    lg.addHandler(cap)
    base_fn = PC._unwrap(prod.apply_exl3_fused_moe)
    top0 = prod.apply_exl3_fused_moe
    try:
        # inert when unset / empty; invalid -> WARNING and nothing installed
        for raw in (None, "", "0"):
            if raw is None:
                os.environ.pop(PC.ENV, None)
            else:
                os.environ[PC.ENV] = raw
            PC.plugin_install()
            ck(prod.apply_exl3_fused_moe is top0, f"plugin_install with {PC.ENV}={raw!r} changed apply")
        os.environ[PC.ENV] = "abc"
        cap.msgs.clear()
        PC.plugin_install()
        ck(prod.apply_exl3_fused_moe is top0, "invalid env installed something")
        ck(any(l == logging.WARNING and "NOT installed" in m for l, m in cap.msgs), "invalid env: no WARNING")
        os.environ.pop(PC.ENV, None)
        # unverified production module -> refused with WARNING
        fake = types.SimpleNamespace(**{k: getattr(prod, k) for k in dir(prod) if not k.startswith("__")})

        def apply_exl3_grouped_fat(*a):   # a different function body
            return None

        fake.apply_exl3_grouped_fat = apply_exl3_grouped_fat
        cap.msgs.clear()
        rep = PC.install(prodmod=fake, n=16)
        ck(not rep["installed"] and "apply_exl3_grouped_fat" in rep["reason"], f"unverified module installed: {rep}")
        ck(any(l == logging.WARNING for l, _ in cap.msgs), "unverified module: no WARNING")
        ck(fake.apply_exl3_fused_moe is prod.apply_exl3_fused_moe, "refused install still wrapped")
    finally:
        lg.removeHandler(cap)

    # order A (production's plugin_register): TF installed first (by prefill_cap_common.build), then the cap
    ck(getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False), "K2 hook not on top before the cap")
    rep = PC.install(prodmod=prod, n=16)
    ck(rep["installed"], f"order A install: {rep}")
    top = prod.apply_exl3_fused_moe
    ck(getattr(top, "_glm53_prefill_cap_hook", False) and getattr(top._tf_exl3_orig, "_tf_exl3_apply_hook", False),
       "order A chain is not cap -> K2 -> production")
    ck(PC.install(prodmod=prod, n=16)["reason"] == "already installed", "second install not idempotent")
    ck(PC.uninstall(prodmod=prod)["restored"], "order A cap uninstall")
    ck(getattr(prod.apply_exl3_fused_moe, "_tf_exl3_apply_hook", False), "order A: K2 hook not restored on top")
    # order B: cap first, then TF: K2 must still install (its fingerprint check follows _tf_exl3_orig)
    integrate.uninstall(prodmod=prod, ext=xl)
    ck(prod.apply_exl3_fused_moe is base_fn, "TF uninstall did not restore production's apply")
    PC.install(prodmod=prod, n=16)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep.get("apply_hook") is True, f"order B: K2 not installed on top of the cap: {rep.get('apply_off_reason')}")
    top = prod.apply_exl3_fused_moe
    ck(getattr(top, "_tf_exl3_apply_hook", False) and getattr(top._tf_exl3_orig, "_glm53_prefill_cap_hook", False),
       "order B chain is not K2 -> cap -> production")
    integrate.uninstall(prodmod=prod, ext=xl)
    ck(getattr(prod.apply_exl3_fused_moe, "_glm53_prefill_cap_hook", False), "order B: TF uninstall lost the cap")
    PC.uninstall(prodmod=prod)
    ck(prod.apply_exl3_fused_moe is base_fn, "order B: full uninstall did not restore production's function")
    # back to production's configuration (order A)
    rep = integrate.install(prodmod=prod, ext=xl, force=True)
    ck(rep.get("apply_hook") is True, "re-install TF")
    PC.install(prodmod=prod, n=16)


def p2_guards(ck, prod, PC):
    print("P.2 hook guards (fake production apply)", flush=True)
    seen = []

    def fake_orig(x2d, ids, weights, layer, inners, expert_map, limit):
        seen.append(layer._exl3_fused_temps)
        if getattr(layer, "_boom", False):
            raise RuntimeError("boom")
        x2d.add_(0)
        return "ok"

    hook = PC._make_hook(prod, fake_orig)
    dev = torch.device("cuda", 0)
    conc = 6
    temps = (torch.empty(conc, 256, 64, dtype=torch.float16, device=dev),
             torch.empty(conc, 256, 64, dtype=torch.float16, device=dev),
             torch.empty(conc, 256, 32, dtype=torch.float16, device=dev),
             torch.empty(conc, 256, 32, dtype=torch.float16, device=dev))
    layer = types.SimpleNamespace(_exl3_fused_temps=temps, _exl3_fat_effective_tier="grouped")

    def call(T):
        seen.clear()
        x = torch.zeros(T, 4, device=dev)
        r = hook(x, None, None, layer, None, None, 10.0)
        return r, seen[0]

    PC.set_cap(16)
    for T in (1, 64, 256):
        _, t = call(T)
        ck(t is temps, f"T={T} <= cap: temps not the production object")
    _, t = call(257)
    ck(t is not temps and tuple(t[0].shape) == (conc, 16, 64) and tuple(t[2].shape) == (conc, 16, 32)
       and all(a.dtype == b.dtype for a, b in zip(t, temps)), f"T=257: temps not swapped to n=16 ({t[0].shape})")
    ck(layer._exl3_fused_temps is temps, "temps not restored after the call")
    _, t2 = call(4000)
    ck(t2[0].data_ptr() == t[0].data_ptr(), "small temps re-allocated for the same key")
    layer._boom = True
    try:
        call(300)
        ck(False, "exception swallowed")
    except RuntimeError:
        pass
    layer._boom = False
    ck(layer._exl3_fused_temps is temps, "temps not restored after an exception")
    layer._exl3_fat_effective_tier = "kernel"
    ck(call(1000)[1] is temps, "tier 'kernel' swapped")
    layer._exl3_fat_effective_tier = "grouped"
    for var, val in (("EXL3_FAT_GROUPED", "0"), ("EXL3_MOE_ROW_TILE", "1")):
        old = os.environ.get(var)
        os.environ[var] = val
        try:
            ck(call(1000)[1] is temps, f"{var}={val} swapped")
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
    PC.set_cap(256)
    ck(call(1000)[1] is temps, "n >= cap swapped")
    PC.set_cap(16)
    # CUDA graph capture: pass through (no allocation while capturing)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    x = torch.zeros(1000, 4, device=dev)
    g = torch.cuda.CUDAGraph()
    before = PC.STATS["passthrough_capture"]
    seen.clear()
    with torch.cuda.graph(g, stream=s):
        hook(x, None, None, layer, None, None, 10.0)
    ck(seen and seen[0] is temps and PC.STATS["passthrough_capture"] == before + 1, "capture: swapped")
    PC.set_cap(16)


def p3_decode(ck, prod, tf, layers, PC):
    print("P.3 decode unchanged (real kernels, full stack)", flush=True)
    dev = torch.device("cuda", 0)
    L = layers[0]
    PC.set_cap(16)
    for T in (1, 8, 64, 256):
        g = torch.Generator().manual_seed(40 + T)
        x = H.xin(T, C.K, g, dev)
        ids = H.random_ids(T, C.NEXP, C.TOPK, g, dev)
        w = H.random_weights(T, C.TOPK, g, dev)
        sw, pt = PC.STATS["swapped"], PC.STATS["passthrough_tokens"]
        t0 = L._exl3_fused_temps
        C.apply(prod, x, ids, w, L)
        ck(PC.STATS["swapped"] == sw and PC.STATS["passthrough_tokens"] == pt + 1 and L._exl3_fused_temps is t0,
           f"decode T={T} was not passed through untouched")
    # CUDA graph at T=64 through wrapper -> K2
    T = 64
    g = torch.Generator().manual_seed(77)
    x = H.xin(T, C.K, g, dev)
    ids = H.random_ids(T, C.NEXP, C.TOPK, g, dev)
    w = H.random_weights(T, C.TOPK, g, dev)
    eager = C.apply(prod, x, ids, w, L).clone()
    gtf = tf.COUNTERS["graph_tf_calls"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        C.apply(prod, x, ids, w, L)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = C.apply(prod, x, ids, w, L)
    gr.replay()
    torch.cuda.synchronize()
    ck(tf.COUNTERS["graph_tf_calls"] == gtf + 1, "T=64 capture was not served by TF")
    e = rel(out, eager)
    print(f"  T=64 graph replay vs eager rel-L2 {e:.2e}", flush=True)
    ck(e < 1e-5, f"T=64 graph replay != eager ({e:.2e})")
    del gr


def p4_parity(ck, prod, layers, PC):
    print("P.4 parity vs production's cap-256 path", flush=True)
    dev = torch.device("cuda", 0)
    L = layers[0]
    worst = {}
    for kind in ("real", "collapsed"):
        for T in (1791, 4608, 13824):
            seed = T + (0 if kind == "real" else 50000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev)
            g = torch.Generator().manual_seed(seed + 3)
            x = H.xin(T, C.K, g, dev)
            h = C.histogram(ids)
            PC.set_cap(0)
            base = C.apply(prod, x, ids, w, L).clone()
            base2 = C.apply(prod, x, ids, w, L)
            scale = float(base.abs().max())
            noise = (rel(base2, base), float((base2 - base).abs().max()) / scale)
            del base2
            cache_before = {k: int(v["h13"].shape[0]) for k, v in prod._FAT_GROUPED_CACHE.items()}
            parts = [f"noise {noise[0]:.1e}/{noise[1]:.1e}"]
            for n in PARITY_N:
                PC.set_cap(n)
                o = C.apply(prod, x, ids, w, L)
                r = rel(o, base)
                ma = float((o - base).abs().max()) / scale
                rr = row_rel_max(o, base)
                moved = int(((h["counts"] > n) & (h["counts"] <= 256)).sum())
                parts.append(f"n={n}: {r:.1e}/{ma:.1e}/row {rr:.1e} ({moved} exp moved)")
                worst[n] = max(worst.get(n, 0.0), r)
                ck(bool(torch.isfinite(o).all()), f"{kind} T={T} n={n}: non-finite output")
                ck(r < 5e-3, f"{kind} T={T} n={n}: rel-L2 {r:.2e} vs cap-256 path")
                del o
            cache_after = {k: int(v["h13"].shape[0]) for k, v in prod._FAT_GROUPED_CACHE.items()}
            ck(cache_after == cache_before and all(v >= T * C.TOPK for v in cache_after.values()),
               f"E3 scratch changed / too small: {cache_before} -> {cache_after}")
            print(f"  [{kind} T={T}] {C.fmt_hist(h)}", flush=True)
            print(f"  [{kind} T={T}] rel-L2 / max-abs (of max|out| {scale:.2f}) vs cap 256: " + " | ".join(parts),
                  flush=True)
            PC.set_cap(16)
            del base
    print("  worst rel-L2 per n: " + ", ".join(f"n={n}: {v:.2e}" for n, v in sorted(worst.items())), flush=True)
    print(f"  E3 scratch rows {[int(v['h13'].shape[0]) for v in prod._FAT_GROUPED_CACHE.values()]}, "
          f"grouped_scratch_bytes {prod.exl3_fat_diag()['grouped_scratch_bytes']}; prefill-cap temps "
          f"{PC.STATS['temps_bytes']} B over {PC.STATS['temps_allocs']} sets "
          f"({PC.STATS['temps_bytes'] // max(1, PC.STATS['temps_allocs'])} B avg)", flush=True)


def _had(dev):
    import exl3_format_ref as R

    return (torch.from_numpy(R.hadamard()) / (128 ** 0.5)).to(dev)


def ref_f64(xl, W, x16, ids, w16, sample, dev, limit):
    """out[t] = sum_k w16[t,k] * down(min(silu(g), L) * clamp(u, -L, L)), float64, for the sampled tokens."""
    Hm = _had(dev)

    def rot(v):
        s = v.shape
        return (v.reshape(*s[:-1], s[-1] // 128, 128) @ Hm).reshape(s)

    def lin(v, wq, suh, svh):
        return rot(rot(v * suh.double()) @ wq) * svh.double()

    xs = x16[sample].double()
    idx = ids[sample]
    ws = w16[sample].double()
    out = torch.zeros(len(sample), C.K, dtype=torch.float64, device=dev)
    wg = torch.empty(C.K, C.N, dtype=torch.float16, device=dev)
    wu = torch.empty_like(wg)
    wd = torch.empty(C.N, C.K, dtype=torch.float16, device=dev)
    for e in torch.unique(idx).tolist():
        rows, ks = (idx == e).nonzero(as_tuple=True)
        xl.reconstruct(wg, W.w13_trellis[e, 0], 4, True, False)
        xl.reconstruct(wu, W.w13_trellis[e, 1], 4, True, False)
        xl.reconstruct(wd, W.w2_trellis[e], 4, True, False)
        v = xs[rows]
        gg = lin(v, wg.double(), W.w13_suh[e, 0], W.w13_svh[e, 0])
        uu = lin(v, wu.double(), W.w13_suh[e, 1], W.w13_svh[e, 1])
        a = gg * torch.sigmoid(gg)
        if limit != 0.0:
            a = a.clamp(max=limit)
            uu = uu.clamp(-limit, limit)
        d = lin(a * uu, wd.double(), W.w2_suh[e], W.w2_svh[e])
        out.index_add_(0, rows, ws[rows, ks, None] * d)
    return out


def p5_f64(ck, prod, xl, layers, PC):
    print("P.5 float64 reference", flush=True)
    import exl3_format_ref as R

    dev = torch.device("cuda", 0)
    L = layers[0]
    W = L._test_weights
    w = torch.empty(C.K, C.N, dtype=torch.float16, device=dev)
    xl.reconstruct(w, W.w13_trellis[5, 1], 4, True, False)
    ck(torch.equal(w.cpu(), R.unpack(W.w13_trellis[5, 1].cpu(), 4)), "reconstruct != format reference unpack")
    # output scale ~1 (as tests/test_e2e_vs_f64.py): rescale down svh once
    T = 2048
    j = torch.arange(C.TOPK)
    t = torch.arange(T)
    balanced = ((j[None, :] * 8 + t[:, None]) % 64).to(torch.long).to(dev).contiguous()
    cnt = torch.bincount(balanced.reshape(-1), minlength=C.NEXP)
    ck(int(cnt[:64].min()) == 256 and int(cnt[:64].max()) == 256 and int(cnt[64:].sum()) == 0, "balanced routing")
    g = torch.Generator().manual_seed(5)
    x = H.xin(T, C.K, g, dev)
    wts = H.random_weights(T, C.TOPK, g, dev)
    PC.set_cap(0)
    o = C.apply(prod, x, balanced, wts, L)
    W.rescale_svh_d(1.0 / float(o.std()))
    cases = [("balanced 64x256 rows", balanced, wts, x, T)]
    T2 = 1791
    ids2 = C.routing("real", T2, 1791, dev)
    w2 = C.weights_for(T2, 1791, dev)
    x2 = H.xin(T2, C.K, torch.Generator().manual_seed(6), dev)
    cases.append(("real T=1791", ids2, w2, x2, T2))
    gs = torch.Generator().manual_seed(9)
    for name, ids, wts_, xx, TT in cases:
        sample = torch.randperm(TT, generator=gs)[:96].sort().values.to(dev)
        ref = ref_f64(xl, W, xx.half(), ids, wts_.to(torch.float16), sample, dev, C.LIMIT)
        res = {}
        for n in (0, 1, 16):
            PC.set_cap(n)
            o = C.apply(prod, xx, ids, wts_, L)[sample].double()
            res[n] = (rel(o, ref), float((o - ref).abs().max()), row_rel_max(o, ref))
        PC.set_cap(16)
        e256 = res[0][0]
        print(f"  [{name}] vs float64 (96 tokens, |ref| max {float(ref.abs().max()):.2f}): "
              f"cap 256 (thin) rel-L2 {e256:.3e} max-abs {res[0][1]:.3e} row-max {res[0][2]:.3e}", flush=True)
        for n in (1, 16):
            e = res[n][0]
            print(f"  [{name}]   n={n} (E3 above n rows) rel-L2 {e:.3e} max-abs {res[n][1]:.3e} row-max "
                  f"{res[n][2]:.3e} | ratio to cap 256 {e / e256:.3f}", flush=True)
            ck(e <= 1.25 * e256 + 2.5e-4, f"{name}: n={n} less accurate than cap 256 ({e:.3e} vs {e256:.3e})")
            ck(e <= 5e-3, f"{name}: n={n} rel-L2 vs float64 {e:.3e} > 5e-3")


def main():
    H.gpu_guard(8.0)
    prod, xl, tf, _ = C.build(0)
    import glm53_prefill_cap as PC
    import integrate

    ck = H.Checks()
    p1_wiring(ck, prod, xl, PC, integrate)       # leaves TF (K2) + the cap installed, in production's order
    layers = C.make_layers(prod, 1)              # built after TF's install: registered by its build hook
    p2_guards(ck, prod, PC)
    p3_decode(ck, prod, tf, layers, PC)
    p4_parity(ck, prod, layers, PC)
    p5_f64(ck, prod, xl, layers, PC)
    print(f"prefill-cap STATS {PC.STATS}", flush=True)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
