"""opt-moe2-rev: adversarial edge cases for GLM53_MOE_E4M3_MAINLOOP=1 (fused variants + 8192), real layer-10 experts.

Differences from tests/optmoe2/test_ms.py (which compares a16 / a8d / dsc against the previous call's buffers):
  * the shared intermediate buffers (a16, a8d, dsc) are POISONED (0xFF bytes = fp16 NaN / e4m3 NaN / fp32 NaN) before
    every MAINLOOP call, so a tile the new kernel skipped cannot pass by inheriting the shipped call's bytes
  * grid = 1 / 2 / 3 / 0 (full) and lag = 1 / 2 / 12 / 64: with grid = 1 one CTA runs every job in ticket order, so
    the fp32 output is BITWISE deterministic and must be bit-identical to the shipped kernel's (the strongest check
    the atomics allow); small grids force long job chains per CTA (prologue vs prefetch transitions, both table
    buffers reused many times)
  * routings built to hit tile edges: experts with exactly 1 / 15 / 16 / 17 / 127 / 128 / 129 / 255 / 256 / 257 rows,
    many 1-row experts (T = 1 / 2 / 3), one local expert only, every expert non-local but one, invalid ids (-1)
  * every variant: v0 / v16 x TG on / off x fp32 / bf16 accumulator
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe2rev/adv_ms.py
Env: ADV_QUICK=1 (fewer grid/lag combos)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()
NEXP = 288


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def ids_with_counts(counts: dict, fill_T: int, g, dev):
    """Token ids [T, 8] where expert e gets exactly counts[e] rows (each token's 8 experts distinct)."""
    T = max(max(counts.values()), fill_T)
    ids = torch.full((T, 8), -1, dtype=torch.long)
    slot = torch.zeros(T, dtype=torch.long)
    for e, n in counts.items():
        toks = torch.argsort(slot + torch.rand(T, generator=g) * 0.5)[:n]     # least-filled tokens first
        assert int(slot[toks].max()) < 8
        ids[toks, slot[toks]] = e
        slot[toks] += 1
    # remaining slots: invalid (-1) -> never computed (non-local class)
    return ids.to(dev), T


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
    n_exp = len(L._exl3_inners)
    g = torch.Generator().manual_seed(4242)
    quick = os.environ.get("ADV_QUICK") == "1"

    one_local = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    one_local[5] = 5
    half_map = torch.full((n_exp,), -1, dtype=torch.long, device=dev)
    half_map[n_exp // 2:] = torch.arange(n_exp // 2, n_exp, dtype=torch.long, device=dev)   # upper half local

    cases = []
    # tile-edge row counts on distinct experts (incl. the first and the last expert)
    edge = {0: 1, 1: 15, 2: 16, 3: 17, 50: 127, 51: 128, 52: 129, 100: 255, 101: 256, 287: 257}
    ids, T = ids_with_counts(edge, 260, g, dev)
    cases.append(("edges", ids, T, emap_full))
    cases.append(("edges-half", ids, T, half_map))
    for T in (1, 2, 3):
        ids = torch.stack([torch.randperm(NEXP, generator=g)[:8] for _ in range(T)]).to(dev)
        cases.append((f"T{T}", ids, T, emap_full))
    ids = C.routing("real", 300, 31, dev)
    cases.append(("real300-onelocal", ids, 300, one_local))
    ids = C.routing("collapsed", 2049, 32, dev)
    cases.append(("collapsed2049", ids, 2049, emap_full))
    ids = C.routing("real", 4289, 33, dev)
    cases.append(("real4289-half", ids, 4289, half_map))

    grids = [(1, 12), (2, 1), (3, 64), (0, 12), (0, 1), (0, 2)] if not quick else [(1, 12), (0, 12)]
    nbit_fail = 0
    for name, ids, T, emap in cases:
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        w = torch.rand(T, 8, generator=g)
        w = (w / w.sum(-1, keepdim=True)).to(dev)
        for grid, lag in grids:
            if grid == 1 and T > 2100:
                continue        # one CTA for a whole 4,289-token layer: slow, covered by the smaller cases
            for v0 in (0, 16):
                for tg in (True, False):
                    for acc in ("f32", "bf16"):
                        def call(ms, poison):
                            keep = {}
                            if poison:
                                t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
                                a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
                                a16.view(torch.int16).fill_(-1)
                                a8d.fill_(255)
                                dsc.view(torch.int32).fill_(-1)
                            out = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, keep=keep,
                                        sched={"acc": acc, "tg": tg, "ms": ms, "variant": v0, "grid": grid,
                                               "lag": lag})
                            torch.cuda.synchronize()
                            nr = int(keep["num_rows"].item()) if torch.is_tensor(keep["num_rows"]) else int(keep["num_rows"])
                            return (out.clone(), keep["a16"][:nr].clone(), keep["a8d"][:nr].clone(),
                                    keep["dsc"][:nr].clone(), nr)
                        b0 = call(False, True)
                        m0 = call(True, True)
                        b1 = call(False, True) if grid != 1 else b0
                        tag = f"[{name} T={T} grid={grid} lag={lag} v{v0} tg={int(tg)} {acc}]"
                        nr = b0[4]
                        same = m0[4] == nr and torch.equal(m0[1].view(torch.int16), b0[1].view(torch.int16))
                        if v0 == 0:
                            same = same and torch.equal(m0[2], b0[2]) and torch.equal(m0[3].view(torch.int32),
                                                                                         b0[3].view(torch.int32))
                        fin = bool(torch.isfinite(m0[0]).all()) and bool(torch.isfinite(b0[0]).all())
                        bit_out = torch.equal(m0[0].view(torch.int16 if acc == "bf16" else torch.int32),
                                              b0[0].view(torch.int16 if acc == "bf16" else torch.int32))
                        e = rel(m0[0], b0[0]) if nr > 0 else 0.0
                        e_aa = rel(b1[0], b0[0]) if nr > 0 else 0.0
                        if grid == 1:
                            ok = same and fin and bit_out
                        else:
                            # atomics-order class: fp32 <= max(4 x A/A, 1e-6); bf16 <= max(2 x A/A, 8e-3) (ACC_SELFTEST_TOL)
                            ok = same and fin and (e <= (max(4 * e_aa, 1e-6) if acc == "f32" else max(2 * e_aa, 8e-3)))
                        if not ok:
                            nbit_fail += 1
                        CHK(ok, f"{tag} rows {nr}: intermediates identical={same} out bitwise={bit_out} "
                                f"rel {e:.2e} (shipped A/A {e_aa:.2e}) finite={fin}")
                        if grid != 1 and acc == "bf16":
                            print(f"  {tag} bf16 out vs shipped {e:.2e}, shipped A/A {e_aa:.2e}", flush=True)
        del x
        torch.cuda.empty_cache()
    print(f"failures: {nbit_fail}", flush=True)
    PC.uninstall(prod)
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
