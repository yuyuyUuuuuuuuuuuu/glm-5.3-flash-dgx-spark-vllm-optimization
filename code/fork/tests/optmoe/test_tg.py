"""opt-moe GLM53_MOE_E4M3_TOKGATHER (one gathered e4m3 gate/up row per TOKEN, read through row_token): real layer-10
experts (TP=2 rank-0 shard), production shapes.

  1. the layer qualifies: every expert's gate suh == expert 0's, gate == up (tok_gather_ok); a layer with one expert's
     suh changed does not (and its result is cached)
  2. the gathered bytes are the same (and gather_tok zeroes a NaN-prefilled fp32 / bf16 out completely): for every computed row r, gather_tok's a8[row_token[r]] == gather2's a8[r]
     BYTE FOR BYTE and asc equal bit for bit (T 13,824 / 4,289 / 300 / 17 / 1)
  3. the fused output with TG == without TG within the atomics-order class (fp32: rel-L2 vs the run-to-run spread of
     the per-pair path; bf16: same bound as two bf16 runs), variants 0 and 16 (f16 down), real + collapsed routing,
     half the experts non-local (all-non-local rows exactly 0), the fold (S + routed) with TG
  4. the hook: TOKGATHER unset = on, "0" = off (served == the per-pair path), invalid values refused
  5. timing at 13,824 / 4,289: gather2 vs gather_tok, run() with and without TG (fp32 and bf16 accumulators)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/test_tg.py
"""
from __future__ import annotations

import os
import statistics
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, HERE)
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
    ids = C.routing(kind, T, seed, dev) if T > 8 else torch.stack(
        [torch.randperm(288, generator=g)[:8] for _ in range(T)]).to(dev)
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
    emap = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    ext = M._ext()
    P = L._exl3_ptrs
    n_exp = len(L._exl3_inners)
    # 1. qualification
    CHK(M.tok_gather_ok(L) is True, "[layer] real layer 10: shared w13 suh across experts and gate == up")
    s2 = L.w13_suh.clone()
    s2[5, 0, 7] = -s2[5, 0, 7]
    fake = types.SimpleNamespace(w13_suh=s2, _exl3_shared_w13_suh=True)
    CHK(M.tok_gather_ok(fake) is False and fake._glm53_tok_gather is False,
        "[layer] one expert's suh changed: not qualified (cached False)")
    fake2 = types.SimpleNamespace(w13_suh=L.w13_suh, _exl3_shared_w13_suh=False)
    CHK(M.tok_gather_ok(fake2) is False, "[layer] gate suh != up suh: not qualified")
    # 2. gathered bytes
    for T, kind, seed in ((13824, "real", 61), (4289, "real", 62), (300, "collapsed", 63), (17, "real", 64),
                          (1, "real", 65)):
        x, ids, w = inputs(T, kind, seed, dev)
        t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
        a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
        ext.gather2(x, t["local"], t["pos"], P["gate_suh"], a8, asc, None, t["topk"], n_exp)
        nr = int(t["num_rows"].item()) if "num_rows" in t else t["P"]
        pa, ps = a8[:nr].clone(), asc[:nr].clone()
        ext.gather_tok(x, P["gate_suh"], a8, asc, None)
        rt = t["row_token"][:nr]
        ta, ts = a8.index_select(0, rt), asc.index_select(0, rt)
        CHK(torch.equal(pa, ta) and torch.equal(ps.view(torch.int32), ts.view(torch.int32)),
            f"[bytes T={T} {kind}] gather_tok row of the token == gather2 row of every pair ({nr} rows), scales bitwise")
        for dt in (torch.bfloat16, torch.float32):
            on = torch.full((T, 4096), float("nan"), dtype=dt, device=dev)
            ext.gather_tok(x, P["gate_suh"], a8, asc, on)
            CHK(bool((on == 0).all()), f"[zero T={T} {str(dt)[6:]}] gather_tok zeroes a NaN-prefilled out completely")
    # 3. outputs
    for T, kind, seed in ((13824, "real", 71), (13856, "collapsed", 72), (4289, "real", 73), (300, "real", 74),
                          (17, "real", 75)):
        x, ids, w = inputs(T, kind, seed, dev)
        tag = f"T={T} {kind}"
        for var in (0, 16):
            f0 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "variant": var, "tg": False}).clone()
            f1 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "variant": var, "tg": False}).clone()
            ft = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "variant": var, "tg": True}).clone()
            spread = rel(f1, f0)
            CHK(rel(ft, f0) <= max(4 * spread, 1e-7), f"[{tag} v{var}] fp32: TG vs per-pair {rel(ft, f0):.2e} "
                f"(per-pair run-to-run {spread:.2e})")
            b0 = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "variant": var, "tg": False}).clone()
            bt = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "variant": var, "tg": True}).clone()
            CHK(rel(bt, f0) < 6e-3 and rel(bt, f0) <= 1.1 * rel(b0, f0) + 2e-4,
                f"[{tag} v{var}] bf16: TG {rel(bt, f0):.2e} vs per-pair bf16 {rel(b0, f0):.2e} (both vs fp32)")
        del f0, f1, ft, b0, bt
    half_map = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    half_map[: n_exp // 2] = torch.arange(n_exp // 2, dtype=torch.long, device=dev)
    x, ids, w = inputs(2048, "real", 76, dev)
    ids[:5] = torch.arange(n_exp // 2, n_exp // 2 + 8, device=dev)
    fo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=half_map, sched={"acc": "f32", "tg": False})
    to = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=half_map, sched={"acc": "f32", "tg": True})
    CHK(bool((to[:5] == 0).all()) and rel(to, fo) < 1e-6,
        f"[half non-local] TG: all-non-local rows exactly 0, == per-pair ({rel(to, fo):.2e})")
    x, ids, w = inputs(4289, "real", 77, dev)
    g = torch.Generator().manual_seed(9)
    S = (torch.randn(4289, 4096, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    rf = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False})
    buf = S.clone()
    fo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": True}, fold_into=buf)
    CHK(fo is buf and rel(fo, S.double() + rf.double()) < 6e-3,
        f"[fold + TG] result is S's buffer, == S + routed ({rel(fo, S.double() + rf.double()):.2e})")
    # 4. hook
    for bad in ("2", "on", "true"):
        rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_TOKGATHER": bad})
        CHK(not rep["installed"], f"[hook] GLM53_MOE_E4M3_TOKGATHER={bad!r} refused")
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1"}, load_selftest=False)
    CHK(rep["installed"] and rep.get("tok_gather") is True and M.TG["on"], "[hook] unset: TG on")
    M.uninstall(prod)
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_TOKGATHER": "0"}, load_selftest=False)
    CHK(rep["installed"] and rep.get("tok_gather") is False and not M.TG["on"], "[hook] TOKGATHER=0: TG off")
    x, ids, w = inputs(2048, "real", 78, dev)
    served = prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
    ref = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False}).to(torch.bfloat16)
    CHK(rel(served, ref) < 1e-3, f"[hook] TOKGATHER=0 served == per-pair path ({rel(served, ref):.2e})")
    M.uninstall(prod)
    CHK(M.TG["on"] is True, "[hook] uninstall resets TG to on")
    # 5. timing
    rounds, n = 7, 5
    for T in (13824, 4289):
        x, ids, w = inputs(T, "real", 80 + T, dev)
        t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
        a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
        ob = torch.empty(T, 4096, dtype=torch.bfloat16, device=dev)
        fns = {
            "gather2": lambda: ext.gather2(x, t["local"], t["pos"], P["gate_suh"], a8, asc, ob, t["topk"], n_exp),
            "gather_tok": lambda: ext.gather_tok(x, P["gate_suh"], a8, asc, ob),
            "run_f32": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False}),
            "run_f32_tg": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": True}),
            "run_bf16": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": False}),
            "run_bf16_tg": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16", "tg": True}),
        }
        for f in fns.values():
            f()
        tt = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                tt[k].append(sample(fns[k], n))
        med = {k: statistics.median(v) for k, v in tt.items()}
        print(f"  [T={T} timing] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms", flush=True)
        CHK(med["run_bf16_tg"] < med["run_bf16"], f"[T={T} timing] TG faster (bf16 accumulator): "
            f"{med['run_bf16_tg'] - med['run_bf16']:+.2f} ms")
        del x, ob
        torch.cuda.empty_cache()
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
