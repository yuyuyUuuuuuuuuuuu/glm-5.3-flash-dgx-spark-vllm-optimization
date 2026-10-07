"""Per-allocation variability of FP8 GEMV time: NCOPY independently allocated copies of one Marlin weight, each call
timed alone with CUDA events after an L2 flush, configs interleaved per round. Shows whether some allocations are
slow (physical placement) and whether a config is more robust on the slow ones.
Usage: probe_copies.py N K M ncopy rounds cfg1;cfg2;... [dirty_MiB]   (cfg = W,KW,U,pol)
L2 before each call: a clean read of 96 MiB (flush), then dirty_MiB (default 0) of freshly written lines."""
import sys, statistics
from pathlib import Path
R = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "tests"))
import torch
import fp8_gemv as G
from fp8_bench_common import random_marlin_layers

n, k, M, ncopy, rounds = (int(v) for v in sys.argv[1:6])
cfgs = [tuple(int(x) for x in c.split(",")) for c in sys.argv[6].split(";")]
cfgs = [(w, kw, u, bool(p)) for w, kw, u, p in cfgs]
E = G.ext()
base = random_marlin_layers(n, k, 1)[0]
ws, sc = base.weight, base.weight_scale.view(-1)
copies = [ws.clone() for _ in range(ncopy)]
x = torch.randn(M, k, device="cuda").to(torch.bfloat16)
out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
flush = torch.ones(96 << 18, dtype=torch.float32, device="cuda")
fsum = torch.empty((), dtype=torch.float32, device="cuda")
dirty_mib = int(sys.argv[7]) if len(sys.argv) > 7 else 0
dirty = torch.empty(max(dirty_mib, 1) << 20, dtype=torch.uint8, device="cuda")
nbytes = n * k
res = {(c, i): [] for c in cfgs for i in range(ncopy)}
for r in range(rounds):
    order = [(c, i) for i in range(ncopy) for c in (cfgs if r % 2 == 0 else cfgs[::-1])]
    evs = []
    for c, i in order:
        torch.sum(flush, dim=0, out=fsum)
        if dirty_mib:
            dirty.fill_(r & 255)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        E.fp8_gemv_out(out, x, copies[i], sc, None, n, k, *c, True)
        b.record()
        evs.append((c, i, a, b))
    torch.cuda.synchronize()
    for c, i, a, b in evs:
        res[(c, i)].append(a.elapsed_time(b) * 1000)
print(f"dirty {dirty_mib} MiB; N {n} K {k} M {M}: {ncopy} copies x {len(cfgs)} configs, {rounds} rounds; us (GB/s)")
print("copy  " + "  ".join(f"{str(c):>22s}" for c in cfgs))
tot = {c: [] for c in cfgs}
for i in range(ncopy):
    row = []
    for c in cfgs:
        m = statistics.median(res[(c, i)])
        tot[c].append(m)
        row.append(f"{m:9.1f} ({nbytes / m / 1e3:5.1f}) {100 * (max(res[(c, i)]) - min(res[(c, i)])) / m:4.0f}%")
    print(f"{i:4d}  " + "  ".join(f"{s:>22s}" for s in row), flush=True)
print("mean  " + "  ".join(f"{statistics.mean(tot[c]):>22.1f}" for c in cfgs))
print("max   " + "  ".join(f"{max(tot[c]):>22.1f}" for c in cfgs))
