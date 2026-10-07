"""GLM53_MOE_E4M3_DOWN=f16 (fused kernel variant 16): gate/up on e4m3 as before, the down projection on fp16 operands
(gate/up epilogue applies H(fp16(act) * suh_d) * r -> fp16; down = mma m16n8k16 on the fp16 trellis decode).
Real layer-10 experts (TP=2 rank 0 shard), synthetic activations, calibrated synthetic routing.

  1. end to end vs the module's own torch spec (spec_ffn(down16=True)) and the default path vs its spec (unchanged)
  2. distance to production (E3, all-fp16 operands): f16-down must be strictly closer than the e4m3 default
  3. hook: GLM53_MOE_E4M3_DOWN=f16 installs, the load-time self-test passes against the f16 spec, the served call ==
     run(variant 16); invalid values refused; uninstall resets the mode; unset/e4m3 = the default variant
  4. edge cases: T=17, one expert with every row, x8 activations, an all-zero row (finite)
  5. DEFAULT-PATH IDENTITY: with MOE2_SAVE=path the default-variant outputs of fixed inputs are saved; with
     MOE2_CMP=path they are compared (run once with the previous build's .so, once with this one)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe2/test_down16.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
CHK = H.Checks()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def spec(M, L, x, ids, w, down16):
    out = torch.zeros(x.shape[0], 4096, dtype=torch.float32, device=x.device)
    x16 = x.half()
    for e in torch.unique(ids).tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        d = M.spec_ffn(x16.index_select(0, tok), L._exl3_inners[e], C.LIMIT, down16=down16)
        out.index_add_(0, tok, d * w[tok, kk].unsqueeze(-1).float())
    return out


def inputs(T, kind, seed, dev, xscale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, 4096, generator=g) * xscale).to(torch.bfloat16).to(dev)
    ids = C.routing(kind, T, seed, dev)
    w = C.weights_for(T, seed, dev).float()
    return x, ids, w


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap = prod.pin_exl3_expert_map(L, dev)
    save, cmp = os.environ.get("MOE2_SAVE"), os.environ.get("MOE2_CMP")
    saved = {}
    if cmp:
        ref = torch.load(cmp)
    for T, kind, seed, xs in ((2048, "real", 11, 1.0), (1536, "collapsed", 12, 1.0), (13824, "real", 13, 1.0),
                              (4289, "real", 14, 1.0), (300, "real", 15, 1.0), (1024, "real", 16, 8.0)):
        x, ids, w = inputs(T, kind, seed, dev, xs)
        tag = f"T={T} {kind} x{xs:g}"
        o8 = M.run(prod, x, ids, w, L, C.LIMIT).clone()
        if save:                 # identity reference only (may run against the previous build's .so: no variant 16)
            saved[tag] = o8.cpu()
            continue
        o16 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": 16}).clone()
        s8, s16 = spec(M, L, x, ids, w, False), spec(M, L, x, ids, w, True)
        po = prod.apply_exl3_fused_moe(x, ids, w, L, L._exl3_inners, emap, C.LIMIT).float()
        r8, r16 = rel(o8, s8), rel(o16, s16)
        p8, p16, ps16 = rel(o8, po), rel(o16, po), rel(s16, po)
        print(f"  [{tag}] default vs its spec {r8:.2e} | f16-down vs its spec {r16:.2e} | vs production: default "
              f"{p8:.4f}, f16-down {p16:.4f} (spec {ps16:.4f}) -> error x{p16 / p8:.3f}, variance x{(p16 / p8) ** 2:.3f}",
              flush=True)
        CHK(r8 < 3e-3, f"[{tag}] default path vs its spec {r8:.2e} < 3e-3")
        CHK(r16 < 3e-3, f"[{tag}] f16-down vs its spec {r16:.2e} < 3e-3")
        CHK(abs(p16 - ps16) < 0.05 * ps16, f"[{tag}] f16-down kernel and its spec differ from production alike")
        CHK(p16 < 0.9 * p8, f"[{tag}] f16-down closer to production than the e4m3 down ({p16:.4f} < 0.9 x {p8:.4f})")
        CHK(bool(torch.isfinite(o16).all()), f"[{tag}] finite")
        if cmp:
            d = rel(o8.cpu(), ref[tag])
            print(f"  [{tag}] default output vs the previous build's: rel-L2 {d:.2e}", flush=True)
            CHK(d < 1e-5, f"[{tag}] default path == previous build within the fp32-atomics order spread ({d:.2e})")
        del x, ids, w, o8, o16, s8, s16, po
        torch.cuda.empty_cache()
    if save:
        torch.save(saved, save)
        print(f"saved {len(saved)} default-path outputs to {save}")
        return
    # bf16 input read directly by gather2 == the fp16 copy's path, byte for byte (a8 rows, scales), incl. values
    # that round to fp16 subnormals / overflow to inf in x.half()
    for T, kind, seed in ((2048, "real", 31), (4289, "collapsed", 32)):
        x, ids, w = inputs(T, kind, seed, dev)
        x[3, :64] = torch.tensor(3e-6, dtype=torch.bfloat16)      # fp16 subnormal range
        x[5, 100] = torch.tensor(1e5, dtype=torch.bfloat16)       # > 65504: inf in fp16 (the row's output is non-finite in both)
        kb, kh = {}, {}
        ob = M.run(prod, x, ids, w, L, C.LIMIT, keep=kb)
        bb = (kb["a8"].clone(), kb["asc"].clone())
        oh = M.run(prod, x.half(), ids, w, L, C.LIMIT, keep=kh)
        same = torch.equal(bb[0], kh["a8"]) and torch.equal(bb[1].view(torch.int32), kh["asc"].view(torch.int32))
        fin = torch.isfinite(ob).all(-1) == torch.isfinite(oh).all(-1)
        CHK(same and bool(fin.all()), f"[bf16 input T={T} {kind}] gather from bf16 == gather from x.half(): a8 + scales "
            f"bit-identical ({same}), same non-finite rows")
    # edge cases
    g = torch.Generator().manual_seed(5)
    for tag, T, ids_fn in (("T=17", 17, lambda T: C.routing("real", T, 3, dev)),
                           ("one expert holds all", 512, lambda T: torch.cat(
                               [torch.full((T, 1), 7, device=dev), C.routing("real", T, 4, dev)[:, 1:]], 1))):
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        x[0] = 0
        ids = ids_fn(T)
        w = C.weights_for(T, 9, dev).float()
        o16 = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": 16})
        r = rel(o16, spec(M, L, x, ids, w, True))
        CHK(r < 3e-3 and bool(torch.isfinite(o16).all()), f"[edge {tag}] f16-down vs spec {r:.2e}, finite")
    # hook
    for bad in ("fp16", "1", "F16x"):
        rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_DOWN": bad})
        CHK(not rep["installed"] and not M.DOWN16["on"], f"[hook] GLM53_MOE_E4M3_DOWN={bad!r} refused")
    base = prod.apply_exl3_experts
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_DOWN": "f16"}, load_selftest=False)
    CHK(rep["installed"] and rep.get("down16") is True and M.DOWN16["on"], f"[hook] f16 installed: {rep}")
    if hasattr(L, "_glm53_moe_e4m3_ok"):
        del L._glm53_moe_e4m3_ok
    st = M.selftest(prod, L, C.LIMIT)
    print(f"  [hook] self-test against the f16 spec: {st}", flush=True)
    CHK(st["ok"] and st["rel_l2"] < 2e-3, f"[hook] self-test passes (rel-L2 {st['rel_l2']:.2e})")
    x, ids, w = inputs(2048, "real", 21, dev)
    hooked = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT).float()
    direct = M.run(prod, x, ids, w, L, C.LIMIT, sched={"variant": 16}).to(x.dtype).float()
    d = rel(hooked, direct)
    CHK(d < 1e-2, f"[hook] served call == run(variant 16) to bf16 output precision ({d:.2e})")
    M.uninstall(prod)
    CHK(prod.apply_exl3_experts is base and not M.DOWN16["on"], "[hook] uninstall restores the function and the mode")
    for val in (None, "", "e4m3"):
        env = {"GLM53_MOE_E4M3": "1"} if val is None else {"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_DOWN": val}
        rep = M.install(prod, environ=env, load_selftest=False)
        CHK(rep["installed"] and rep.get("down16") is False and not M.DOWN16["on"],
            f"[hook] GLM53_MOE_E4M3_DOWN={val!r} -> e4m3 down (default)")
        M.uninstall(prod)
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "0", "GLM53_MOE_E4M3_DOWN": "f16"}, load_selftest=False)
    CHK(not rep["installed"] and prod.apply_exl3_experts is base and not M.DOWN16["on"],
        "[hook] GLM53_MOE_E4M3=0 with DOWN=f16: nothing installed")
    CHK.summary()
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
