"""Time build-flag variants of the exact MLA prefill kernel at production shapes (correct results, parity checked).
Usage: probe_mla_variants.py "<flags A>" "<flags B>" ... (e.g. "-DGLM53_MLA_L2HINT=0")"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mla_prefill_common as C
cases = [("13824 sticky", C.Case(13824, 0, "sticky", seed=1)), ("13824@86000 indep", C.Case(13824, 86000, "indep", seed=2))]
for flags in sys.argv[1:]:
    fl = [f for f in flags.split() if f]
    ext = C.build_ext("mla_var", "kernels/mla_prefill/mla_prefill.cu", fl)
    for name, case in cases:
        T = case.T
        out = torch.empty(T, 32, 512, dtype=torch.bfloat16, device="cuda")
        f = lambda: ext.run(case.q, case.cache.view(-1, 512), case.slots, case.valid, out, C.SM_SCALE, 1.0)
        med, best = C.cuda_time(f, 2, 8)
        rows = torch.arange(0, T, max(1, T // 128))
        st = C.err_stats(out[rows], C.reference(case, rows))
        flop = 4.0 * case.pairs() * 32 * 512
        print(f"[{flags or 'default'}] {name}: {med:7.3f} ms (best {best:7.3f}) {flop / med / 1e9:5.1f} TFLOPS "
              f"rel_l2_max {st['rel_l2_max']:.2e}", flush=True)
