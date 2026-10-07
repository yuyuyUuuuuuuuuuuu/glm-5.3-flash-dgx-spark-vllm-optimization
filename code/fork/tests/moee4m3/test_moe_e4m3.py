"""Correctness of GLM53_MOE_E4M3 (kernels/moe_e4m3.cu, glm53_moe_e4m3.py) on nodeC, real layer-10 experts (TP=2 rank 0).

  A. stage by stage against the spec on the SAME inputs:
     A1 gather: e4m3 bytes (k-permuted) + per-row scales vs torch's quantization of H(x * suh)
     A2 gate/up: the fp16 SwiGLU output vs the spec computed from the kernel's own e4m3 rows (isolates the GEMM +
        epilogue) and fp64 GEMM of the same e4m3 operands (accumulation error)
     A3 actq: down e4m3 bytes + scales vs torch's quantization of H(a16 * suh_d)
     A4 down: output vs fp64 of the same e4m3 operands
  B. end to end vs the spec (tests/moee4m3/emu_spec_copy.py = the moefq emulation's own torch arithmetic, weight
     cache off, fp32 matmul precision) and vs a float64 implementation of the spec; vs production's E3 (the expected
     e4m3-vs-fp16 difference, sanity)
  C. expert_map with non-local experts (sentinel rows), tiny T, one expert with all rows, an expert with 0 rows
  D. hook: knob off -> nothing installed (function identity), decode / capture / fused=False pass through, prefill
     served, self-test, uninstall
  E. the emulation's weight cache keys on the trellis STORAGE pointer: on production's stacked layout every expert
     of a layer shares one storage (reported, not asserted)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moee4m3/test_moe_e4m3.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402  (sets production's prefill env before the module import)
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
CHK = H.Checks()
HID, INT = 4096, 1024


def perm_cols(n: int) -> torch.Tensor:
    """byte position p of a row holds column perm[p] (kernels/moe_e4m3.cu header)."""
    p = torch.arange(n)
    g, pp = p // 32, p % 32
    half, pq = pp // 16, pp % 16
    q, i = pq // 4, pq % 4
    return g * 32 + half * 16 + 2 * q + (i & 1) + 8 * (i >> 1)


def unperm(a8: torch.Tensor) -> torch.Tensor:
    """kernel byte layout -> natural column order (uint8 [R, n])."""
    n = a8.shape[1]
    out = torch.empty_like(a8)
    out[:, perm_cols(n).to(a8.device)] = a8
    return out


def f8(u8: torch.Tensor) -> torch.Tensor:
    return u8.view(torch.float8_e4m3fn).to(torch.float32)


def rot64(x):
    import glm53_moe_e4m3 as M

    h = M._spec_tables(x.device)[2].double()
    s = x.shape
    return (x.double().reshape(-1, s[-1] // 128, 128) @ h.t()).reshape(s)


def quant_ref(v: torch.Tensor):
    sc = v.abs().amax(-1, keepdim=True) / 448.0
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    q = (v / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), sc.squeeze(-1)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def maxrel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def wq8_cache(layer):
    import glm53_moe_e4m3 as M

    cache = {}

    def get(e, proj):        # one expert at a time (288 x 3 fp32 matrices would be 13.8 GB)
        k = (e, proj)
        if k not in cache:
            if len(cache) >= 3:
                cache.clear()
            inner = layer._exl3_inners[e][proj]
            cache[k] = M.spec_decode(inner.trellis).to(torch.float8_e4m3fn).to(torch.float32)
        return cache[k]
    return get


def stage_checks(prod, L, T, kind, seed, xscale=1.0):
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, HID, generator=g) * xscale).to(torch.bfloat16).to(dev)
    ids = C.routing(kind, T, seed, dev)
    w = C.weights_for(T, seed, dev).float()
    keep = {}
    out = M.run(prod, x, ids, w, L, C.LIMIT, keep=keep)
    torch.cuda.synchronize()
    nr = int(keep["num_rows"].item())
    P = keep["P"]
    CHK(nr == P, f"[{kind} T={T}] all pairs local: num_rows {nr} == {P}")
    rt, re = keep["row_token"][:nr], keep["row_expert"][:nr].long()
    ptr_suh = torch.stack([L._exl3_inners[e]["gate"].suh for e in range(len(L._exl3_inners))])     # [E, 4096]
    # A1 gather
    xs = x.half()[rt].float() * ptr_suh[re].float()
    import glm53_moe_e4m3 as M2
    v = M2._spec_rot(xs)
    qref, scref = quant_ref(v)
    a8n = unperm(keep["a8"][:nr])
    eq = (a8n == qref).float().mean().item()
    neq = int((a8n != qref).sum())
    # differing bytes: at most one e4m3 step apart (fp32 Hadamard order -> rare ties)
    dq = (f8(a8n) - f8(qref)).abs()
    step = torch.maximum(f8(qref).abs() * 2 ** -2, torch.full_like(dq, 2 ** -9))
    CHK(eq > 0.9999, f"[{kind} T={T}] A1 gather bytes equal to torch's quantization: {eq:.7f} ({neq} differ)")
    CHK(bool((dq <= step + 1e-12).all()), f"[{kind} T={T}] A1 differing bytes are adjacent e4m3 values")
    srel = float(((keep["asc"][:nr] - scref).abs() / scref).max())
    CHK(srel < 1e-6, f"[{kind} T={T}] A1 row scales vs torch: max rel {srel:.2e}")
    print(f"  [{kind} T={T}] A1 gather: {eq * 100:.5f}% bytes identical ({neq} of {a8n.numel()} differ, all adjacent), "
          f"scale max rel {srel:.1e}", flush=True)
    # A2 gate/up from the kernel's own e4m3 rows
    get = wq8_cache(L)
    xa = f8(a8n) * keep["asc"][:nr, None]
    a16_spec = torch.empty(nr, INT, dtype=torch.float16, device=dev)
    gmax_rel64 = 0.0
    segs = []
    starts = torch.ones(nr, dtype=torch.bool, device=dev)
    starts[1:] = re[1:] != re[:-1]
    sidx = starts.nonzero().flatten().tolist() + [nr]
    for a, b in zip(sidx[:-1], sidx[1:]):
        segs.append((int(re[a]), a, b))
    for e, a, b in segs:
        inn = L._exl3_inners[e]
        gq = xa[a:b] @ get(e, "gate")
        uq = xa[a:b] @ get(e, "up")
        if (e % 37) == 0:   # fp64 GEMM of the same operands (accumulation error of the e4m3 mma)
            g64 = f8(a8n[a:b]).double() @ get(e, "gate").double() * keep["asc"][a:b, None].double()
            gmax_rel64 = max(gmax_rel64, maxrel(f8(a8n[a:b]) @ get(e, "gate") * keep["asc"][a:b, None], g64))
        gg = M2._spec_rot(gq) * inn["gate"].svh.float()
        uu = M2._spec_rot(uq) * inn["up"].svh.float()
        a16_spec[a:b] = (torch.nn.functional.silu(gg.clamp(max=C.LIMIT)) * uu.clamp(-C.LIMIT, C.LIMIT)).half()
    a16k = keep["a16"][:nr]
    beq = (a16k == a16_spec).float().mean().item()
    r16 = rel(a16k.float(), a16_spec.float())
    CHK(r16 < 2e-4, f"[{kind} T={T}] A2 gate/up fp16 output vs spec(from kernel rows): rel-L2 {r16:.2e}")
    print(f"  [{kind} T={T}] A2 gate/up: a16 rel-L2 {r16:.2e} vs spec on the kernel's e4m3 rows, {beq * 100:.3f}% "
          f"fp16 values bit-identical; torch fp32 GEMM vs fp64 (same operands) max rel {gmax_rel64:.1e}", flush=True)
    # A3 actq from the kernel's own a16
    dsuh = torch.stack([L._exl3_inners[e]["down"].suh for e in range(len(L._exl3_inners))])
    vd = M2._spec_rot(a16k.float() * dsuh[re].float())
    qd, scd = quant_ref(vd)
    a8dn = unperm(keep["a8d"][:nr])
    eqd = (a8dn == qd).float().mean().item()
    sreld = float(((keep["dsc"][:nr] - scd).abs() / scd).max())
    CHK(eqd > 0.9999, f"[{kind} T={T}] A3 actq bytes equal: {eqd:.7f}")
    CHK(sreld < 1e-6, f"[{kind} T={T}] A3 scales max rel {sreld:.2e}")
    print(f"  [{kind} T={T}] A3 actq: {eqd * 100:.5f}% bytes identical, scale max rel {sreld:.1e}", flush=True)
    # A4 down: fp64 of the kernel's own e4m3 operands
    ref64 = torch.zeros(T, HID, dtype=torch.float64, device=dev)
    xd = f8(a8dn).double() * keep["dsc"][:nr, None].double()
    for e, a, b in segs:
        inn = L._exl3_inners[e]
        y = xd[a:b] @ get(e, "down").double()
        d = rot64(y) * inn["down"].svh.double()
        ref64.index_add_(0, rt[a:b], d * keep["row_weight"][a:b, None].double())
    r4 = rel(out, ref64)
    m4 = maxrel(out, ref64)
    CHK(r4 < 1e-5, f"[{kind} T={T}] A4 down vs fp64 of the same operands: rel-L2 {r4:.2e}")
    print(f"  [{kind} T={T}] A4 down+scatter vs fp64 (same e4m3 operands): rel-L2 {r4:.2e}, max {m4:.2e}", flush=True)
    return x, ids, w, out


def spec_emu(L, x, ids, w, mode="fp32"):
    """End-to-end spec: the vendored emulation's own _ffn (fp32), or a float64 version of the same arithmetic."""
    import emu_spec_copy as EMU

    dev = x.device
    T = x.shape[0]
    out = torch.zeros(T, HID, dtype=torch.float32 if mode == "fp32" else torch.float64, device=dev)
    x16 = x.half()
    for e in torch.unique(ids).tolist():
        m = ids == e
        tok, kk = m.nonzero(as_tuple=True)
        inner = L._exl3_inners[e]
        packs = {p: (EMU.decode_wq(inner[p].trellis).to(torch.float8_e4m3fn), inner[p].suh, inner[p].svh)
                 for p in ("gate", "up", "down")}
        if mode == "fp32":
            d = EMU._ffn(x16.index_select(0, tok).contiguous(), packs, C.LIMIT, "e4m3", 4096)
            out.index_add_(0, tok, d * w[tok, kk].unsqueeze(-1).float())
        else:
            def proj(xx16, p):
                wq, suh, svh = packs[p]
                xs = xx16.double() * suh.double()
                v = rot64(xs)
                sc = v.abs().amax(-1, keepdim=True) / 448.0
                sc = torch.where(sc > 0, sc, torch.ones_like(sc))
                q = (v / sc).clamp(-448, 448).float().to(torch.float8_e4m3fn).double() * sc
                return rot64(q @ wq.double()) * svh.double()
            xx = x16.index_select(0, tok)
            gg, uu = proj(xx, "gate"), proj(xx, "up")
            act = (torch.nn.functional.silu(gg.clamp(max=C.LIMIT)) * uu.clamp(-C.LIMIT, C.LIMIT)).half()
            d = proj(act, "down")
            out.index_add_(0, tok, d * w[tok, kk].unsqueeze(-1).double())
    return out


def e2e(prod, L, x, ids, w, out, tag):
    ref32 = spec_emu(L, x, ids, w, "fp32")
    ref64 = spec_emu(L, x, ids, w, "fp64")
    r32, m32 = rel(out, ref32), maxrel(out, ref32)
    r64, m64 = rel(out, ref64), maxrel(out, ref64)
    e64 = rel(ref32, ref64)
    # the spec's own fp32 implementation is ~1e-3 from its fp64 one (an fp32 rounding difference flips an e4m3 or fp16
    # rounding now and then, which is amplified by the next quantization): the kernel must be at least as close to the
    # exact (fp64) spec as the fp32 emulation is, and within that noise of the emulation
    CHK(r64 <= 1.25 * e64, f"[{tag}] B kernel vs fp64 spec {r64:.2e} <= 1.25 x (fp32 emulation vs fp64 spec {e64:.2e})")
    CHK(r32 < 3e-3, f"[{tag}] B end-to-end vs the fp32 emulation: rel-L2 {r32:.2e} < 3e-3")
    # production (E3 + thin, fp16 operands) for scale: the e4m3 class difference
    emap = prod.pin_exl3_expert_map(L, x.device)
    prod_out = prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, C.LIMIT)
    rp = rel(out, prod_out)
    rs = rel(ref32, prod_out)
    print(f"  [{tag}] B kernel vs spec(fp32 emu): rel-L2 {r32:.2e} max {m32:.2e} | kernel vs spec(fp64): rel-L2 {r64:.2e} "
          f"max {m64:.2e} | spec fp32 vs fp64: {e64:.2e} | e4m3 vs production(E3, fp16): kernel {rp:.4f}, spec {rs:.4f}",
          flush=True)
    CHK(abs(rp - rs) < 0.05 * rs, f"[{tag}] kernel and spec differ from production by the same amount ({rp:.4f} vs {rs:.4f})")
    return {"r32": r32, "m32": m32, "r64": r64, "m64": m64, "rp": rp}


def edge_cases(prod, L):
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    g = torch.Generator().manual_seed(77)
    # C1 expert_map: experts 0..287 global ids 0..575, every other global id non-local
    T = 300
    x = torch.randn(T, HID, generator=g).to(torch.bfloat16).to(dev)
    ids = torch.randint(0, 576, (T, 8), generator=g).to(dev)
    w = torch.rand(T, 8, generator=g).to(dev)
    emap = torch.full((576,), -1, dtype=torch.long, device=dev)
    emap[0::2] = torch.arange(288, device=dev)
    out = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap)
    loc = emap[ids]
    ids_l = torch.where(loc >= 0, loc, torch.zeros_like(loc))
    w_l = torch.where(loc >= 0, w, torch.zeros_like(w))
    ref = spec_emu(L, x, ids_l, w_l, "fp32")
    r = rel(out, ref)
    CHK(r < 3e-3, f"C1 expert_map with non-local experts: rel-L2 {r:.2e}")
    print(f"  C1 expert_map (half the global ids non-local): rel-L2 {r:.2e}", flush=True)
    # C2 tiny T, duplicate expert across tokens; one expert takes everything
    for T, desc in ((1, "T=1"), (17, "T=17"), (520, "one hot expert")):
        x = torch.randn(T, HID, generator=g).to(torch.bfloat16).to(dev)
        if desc == "one hot expert":
            ids = torch.stack([torch.cat([torch.tensor([5]), torch.randperm(282, generator=g)[:7] + 6]) for _ in range(T)]).to(dev)
        else:
            ids = torch.stack([torch.randperm(288, generator=g)[:8] for _ in range(T)]).to(dev)
        w = torch.rand(T, 8, generator=g).to(dev)
        out = M.run(prod, x, ids, w, L, C.LIMIT)
        ref = spec_emu(L, x, ids, w, "fp32")
        r = rel(out, ref)
        CHK(r < 3e-3, f"C2 {desc}: rel-L2 {r:.2e}")
        print(f"  C2 {desc}: rel-L2 {r:.2e}", flush=True)
    # C3 huge activations (scale 300) and zero rows
    T = 64
    x = (torch.randn(T, HID, generator=g) * 300).to(torch.bfloat16).to(dev)
    x[3] = 0
    ids = torch.stack([torch.randperm(288, generator=g)[:8] for _ in range(T)]).to(dev)
    w = torch.rand(T, 8, generator=g).to(dev)
    out = M.run(prod, x, ids, w, L, C.LIMIT)
    ref = spec_emu(L, x, ids, w, "fp32")
    r = rel(out, ref)
    CHK(r < 3e-3 and bool(torch.isfinite(out).all()), f"C3 large / zero activations: rel-L2 {r:.2e}")
    print(f"  C3 large / zero activation rows: rel-L2 {r:.2e}", flush=True)


def hook_checks(prod, L):
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    orig = prod.apply_exl3_experts
    for v in (None, "", "0", "off"):
        env = {} if v is None else {M.ENV: v}
        rep = M.install(prod, environ=env)
        CHK(not rep["installed"] and prod.apply_exl3_experts is orig, f"D knob {v!r}: nothing installed")
    rep = M.install(prod, environ={M.ENV: "2"})
    CHK(not rep["installed"] and prod.apply_exl3_experts is orig, "D invalid knob: refused, nothing installed")
    rep = M.install(prod, environ={M.ENV: "1"})
    CHK(rep["installed"] and prod.apply_exl3_experts is not orig, f"D knob 1: installed ({rep})")
    g = torch.Generator().manual_seed(5)
    cap = M._cap(L)
    # decode-sized call: passes through to the original (bitwise equal: same function, same inputs)
    T = 64
    x = torch.randn(T, HID, generator=g).to(torch.bfloat16).to(dev)
    ids = torch.stack([torch.randperm(288, generator=g)[:8] for _ in range(T)]).to(dev)
    w = torch.rand(T, 8, generator=g).to(dev)
    s0 = dict(M.STATS)
    a = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
    b = orig(x, ids, w, L, limit=C.LIMIT)
    CHK(M.STATS["passed"] == s0["passed"] + 1 and M.STATS["served"] == s0["served"], "D decode call passed through")
    # production's decode kernel is not bitwise reproducible itself (fp32 atomics, spread ~1e-5 measured in
    # test_wiring.py): the pass-through is proven by the counters above; the output must be production's class
    rd = rel(a.float(), b.float())
    CHK(rd < 1e-4, f"D decode call output == production's within its own run-to-run spread ({rd:.1e})")
    # fused=False passes through
    s0 = dict(M.STATS)
    T = cap + 64
    x = torch.randn(T, HID, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", T, 99, dev)
    w = C.weights_for(T, 99, dev)
    try:
        prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT, fused=False)
    except Exception:  # noqa: BLE001  (the python loop may be slow / unsupported here; only the routing matters)
        pass
    CHK(M.STATS["served"] == s0["served"], "D fused=False passed through")
    # prefill call: self-test then served; equals run()
    L.__dict__.pop("_glm53_moe_e4m3_ok", None)
    s0 = dict(M.STATS)
    y = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
    CHK(M.STATS["selftests"] == s0["selftests"] + 1 and L._glm53_moe_e4m3_ok is True, "D self-test ran and passed")
    CHK(M.STATS["served"] == s0["served"] + 1, "D prefill call served")
    r = M.run(prod, x, ids, w, L, C.LIMIT).to(x.dtype)
    d = rel(y.float(), r.float())
    CHK(d < 1e-5, f"D served output == run() (fp32 atomics order + bf16 rounding only): {d:.1e}")
    CHK(y.dtype == x.dtype and y.shape == x.shape, "D output dtype/shape contract (x.dtype)")
    # capture: a call made while capturing passes through (no allocation, production's graph-safe path)
    st = M.selftest(prod, L, C.LIMIT)
    print(f"  D self-test on the real layer: rel-L2 {st['rel_l2']:.2e} max {st['max_rel']:.2e} ok {st['ok']}", flush=True)
    CHK(st["ok"], "D self-test ok")
    M.uninstall(prod)
    CHK(prod.apply_exl3_experts is orig, "D uninstall restores production's function object")


def emu_cache_probe(L):
    import emu_spec_copy as EMU

    keys = {int(L._exl3_inners[e]["gate"].trellis.untyped_storage().data_ptr()) for e in range(4)}
    ptrs = {int(L._exl3_inners[e]["gate"].trellis.data_ptr()) for e in range(4)}
    print(f"  E emulation weight-cache key (untyped_storage().data_ptr()) for experts 0..3: {len(keys)} distinct; "
          f"data_ptr(): {len(ptrs)} distinct -> {'BUG: every expert of a layer hits expert-first entry' if len(keys) == 1 else 'ok'}",
          flush=True)
    EMU.CFG.cache_bytes = 1 << 30
    c = EMU._WeightCache(1 << 30)
    w0 = c.get(L._exl3_inners[0])
    w1 = c.get(L._exl3_inners[1])
    same = w0 is w1
    print(f"  E emulation _WeightCache.get(expert 0) is get(expert 1): {same}", flush=True)
    return same


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    import glm53_moe_e4m3_ext as E
    print(f"extension version {E.version()}, tile rows {E.tile_rows()}", flush=True)
    res = {}
    for kind, T, seed in (("real", 2048, 11), ("collapsed", 1536, 12)):
        x, ids, w, out = stage_checks(prod, L, T, kind, seed)
        res[(kind, T)] = e2e(prod, L, x, ids, w, out, f"{kind} T={T}")
        del x, out
        torch.cuda.empty_cache()
    x, ids, w, out = stage_checks(prod, L, 1024, "real", 13, xscale=8.0)
    e2e(prod, L, x, ids, w, out, "real T=1024 x8 activations")
    # the production chunk shape, end to end only (stage checks would hold [110592, 4096] fp32 copies)
    import glm53_moe_e4m3 as M
    dev = torch.device("cuda", 0)
    g = torch.Generator().manual_seed(14)
    x = torch.randn(13824, HID, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", 13824, 14, dev)
    w = C.weights_for(13824, 14, dev).float()
    out = M.run(prod, x, ids, w, L, C.LIMIT)
    res[("real", 13824)] = e2e(prod, L, x, ids, w, out, "real T=13824")
    del x, out
    torch.cuda.empty_cache()
    edge_cases(prod, L)
    hook_checks(prod, L)
    bug = emu_cache_probe(L)
    print(f"SUMMARY e2e {res}", flush=True)
    print(f"SUMMARY emulation weight-cache collision on stacked layers: {bug}", flush=True)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
