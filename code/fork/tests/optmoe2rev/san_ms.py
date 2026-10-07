"""opt-moe2-rev: a small fused call per MAINLOOP variant for compute-sanitizer (racecheck / synccheck / memcheck /
initcheck), real layer-10 experts. Tile edges in one call: an expert with 130 rows (a full tile + a 2-row tile), 1, 17
and 64 rows, the rest of each token's slots invalid (-1). SAN_GRID (default 1: one CTA runs every job in ticket order,
no inter-CTA waits) / SAN_VARIANTS ("ms" = the 4 MAINLOOP variants, "ship" = the 4 shipped ones).
Run: GPU_RUN_BIND_DIR=/usr/local/cuda/compute-sanitizer=/opt/cs GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 \
     tests/gpu_run.sh /opt/cs/compute-sanitizer --tool racecheck --kernel-name kns=me_fused_kernel \
     python3 tests/optmoe2rev/san_ms.py
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
    n_exp = len(L._exl3_inners)
    g = torch.Generator().manual_seed(77)
    T = 130
    ids = torch.full((T, 8), -1, dtype=torch.long)
    ids[:, 0] = 0                                   # 130 rows: tiles of 128 + 2
    ids[torch.randperm(T, generator=g)[:1], 1] = 1
    ids[torch.randperm(T, generator=g)[:17], 2] = 2
    ids[torch.randperm(T, generator=g)[:64], 3] = 287
    ids = ids.to(dev)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    w = torch.rand(T, 8, generator=g).to(dev)
    grid = int(os.environ.get("SAN_GRID", "1"))
    ms = os.environ.get("SAN_VARIANTS", "ms") == "ms"
    combos = ((0, True, "bf16"), (16, True, "bf16"), (0, False, "f32"), (16, False, "f32"))
    only = os.environ.get("SAN_ONLY")            # e.g. "0" = just the first combo (initcheck is slow)
    if only:
        combos = tuple(combos[int(i)] for i in only.split(","))
    for v0, tg, acc in combos:
        out = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap,
                    sched={"acc": acc, "tg": tg, "ms": ms, "variant": v0, "grid": grid})
        torch.cuda.synchronize()
        print(f"  variant v{v0} tg={int(tg)} {acc} ms={int(ms)} grid={grid}: finite={bool(torch.isfinite(out).all())} "
              f"|out|={float(out.float().abs().sum()):.6e}", flush=True)
    PC.uninstall(prod)


if __name__ == "__main__":
    H.run_main(main)
