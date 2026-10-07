"""opt-moe2 GLM53_MOE_E4M3_MAINLOOP=1 (the lean fused mainloop: lean trellis decode + A copy address table, variants
+ 8192): real layer-10 experts (TP=2 rank-0 shard), production shapes.

  1. every intermediate is BIT-IDENTICAL to the shipped mainloop: a16 (gate/up epilogue output, fp16), a8d + dsc (the
     in-kernel actq), for every computed row - variants 0 / 16 (f16 down) x TG on / off, fp32 and bf16 accumulators,
     T 13,824 / 4,289 / 2,049 / 300 / 17 / 1, real + collapsed routing (one expert holding every row), half the
     experts non-local
  2. the output is in the atomics-order class of the shipped mainloop: fp32 accumulator rel-L2 vs the shipped one <=
     max(4 x the shipped run-to-run spread, 1e-7); bf16 accumulator <= 2 x the shipped bf16 A/A spread (+1e-4 floor);
     non-local-only token rows exactly 0; finite everywhere (out NaN-prefilled and zeroed by the gather)
  3. the hook: MAINLOOP unset / "" / "0" = off, "1" = on, others refused (install refuses an invalid value)
  4. timing at 13,824 / 4,289 (run(), both accumulators, TG on): shipped vs MS
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe2/test_ms.py
Env: MS_QUICK=1 (13,824 + 300 only, no timing), MS_TIMING=0 (skip timing)
"""
from __future__ import annotations

import os
import statistics
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def inputs(T, kind, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    if kind == "collapsed":
        ids = torch.zeros(T, 8, dtype=torch.long)
        ids[:, 1:] = torch.stack([torch.randperm(287, generator=g)[:7] + 1 for _ in range(T)])
        ids = ids.to(dev)
    elif T > 8:
        ids = C.routing(kind, T, seed, dev)
    else:
        ids = torch.stack([torch.randperm(288, generator=g)[:8] for _ in range(T)]).to(dev)
    w = C.weights_for(T, seed, dev).float()
    return x, ids, w


def sample(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap_full = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    quick = os.environ.get("MS_QUICK") == "1"

    # 3. hook + qualification
    CHK(M.ms_mode({}) is False and M.ms_mode({"GLM53_MOE_E4M3_MAINLOOP": ""}) is False
        and M.ms_mode({"GLM53_MOE_E4M3_MAINLOOP": "0"}) is False
        and M.ms_mode({"GLM53_MOE_E4M3_MAINLOOP": "1"}) is True, "[hook] unset/''/0 off, 1 on")
    for bad in ("2", "on", "true", "yes"):
        try:
            M.ms_mode({"GLM53_MOE_E4M3_MAINLOOP": bad})
            CHK(False, f"[hook] {bad!r} refused")
        except ValueError:
            CHK(True, f"[hook] {bad!r} refused")
    r = M.install(prodmod=types.SimpleNamespace(), environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_MAINLOOP": "x"})
    CHK(r["installed"] is False and "MAINLOOP" in r["reason"], f"[hook] install refuses an invalid value ({r})")
    # 1 + 2. numerics
    n_exp = len(L._exl3_inners)
    half_map = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    half_map[: n_exp // 2] = torch.arange(n_exp // 2, dtype=torch.long, device=dev)

    Ts = [13824, 300] if quick else [13824, 4289, 2049, 300, 17, 1]
    cases = []
    for T in Ts:
        cases.append((T, "real", "full"))
    if not quick:
        cases += [(4289, "collapsed", "full"), (2049, "real", "half"), (300, "collapsed", "half")]
    for T, kind, mp in cases:
        emap = emap_full if mp == "full" else half_map
        x, ids, w = inputs(T, kind, 7000 + T, dev)
        for v0 in (0, 16):
            for tg in (True, False):
                for acc in ("f32", "bf16"):
                    def call(ms):
                        keep = {}
                        out = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, keep=keep,
                                    sched={"acc": acc, "tg": tg, "ms": ms, "variant": v0})
                        nr = int(keep["num_rows"].item()) if torch.is_tensor(keep["num_rows"]) else int(keep["num_rows"])
                        return (out.clone(), keep["a16"][:nr].clone(), keep["a8d"][:nr].clone(),
                                keep["dsc"][:nr].clone(), nr)
                    b0 = call(False)
                    b1 = call(False)
                    m0 = call(True)
                    tag = f"[T={T} {kind} {mp} v{v0} tg={int(tg)} {acc}]"
                    nr = b0[4]
                    same = (m0[4] == nr and torch.equal(m0[1].view(torch.int16), b0[1].view(torch.int16))
                            and torch.equal(m0[2], b0[2]) and torch.equal(m0[3].view(torch.int32), b0[3].view(torch.int32)))
                    if v0 == 16:     # DN16: no actq (a8d / dsc unused), a16 holds the rotated rows
                        same = m0[4] == nr and torch.equal(m0[1].view(torch.int16), b0[1].view(torch.int16))
                    CHK(same, f"{tag} intermediates bit-identical ({nr} rows)")
                    fin = bool(torch.isfinite(m0[0]).all())
                    e_ms, e_aa = rel(m0[0], b0[0]), rel(b1[0], b0[0])
                    if acc == "f32":
                        ok = e_ms <= max(4 * e_aa, 1e-7)
                    else:
                        ok = e_ms <= 2 * e_aa + 1e-4
                    CHK(fin and ok, f"{tag} out vs shipped {e_ms:.2e} (shipped A/A {e_aa:.2e}) finite={fin}")
                    print(f"  {tag} intermediates identical={same} ({nr} rows) | out vs shipped {e_ms:.2e}, "
                          f"shipped A/A {e_aa:.2e}", flush=True)
                    if mp == "half":
                        loc = (ids < n_exp // 2).any(dim=1)
                        z = m0[0][~loc]
                        CHK(z.numel() == 0 or bool((z == 0).all()), f"{tag} tokens with no local expert exactly 0")
        del x, ids, w
        torch.cuda.empty_cache()

    # 4. timing
    if not quick and os.environ.get("MS_TIMING", "1") != "0":
        for T in (13824, 4289):
            x, ids, w = inputs(T, "real", 9000 + T, dev)
            fns = {}
            for acc in ("f32", "bf16"):
                for ms in (False, True):
                    fns[f"run_{acc}{'_ms' if ms else ''}"] = (
                        lambda acc=acc, ms=ms: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap_full,
                                                     sched={"acc": acc, "ms": ms}))
            for f in fns.values():
                f()
            tt = {k: [] for k in fns}
            for rr in range(7):
                for k in (list(fns) if rr % 2 == 0 else list(fns)[::-1]):
                    tt[k].append(sample(fns[k], 5))
            med = {k: statistics.median(v) for k, v in tt.items()}
            print(f"[timing T={T}] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms", flush=True)
            del x, ids, w
            torch.cuda.empty_cache()
    PC.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
