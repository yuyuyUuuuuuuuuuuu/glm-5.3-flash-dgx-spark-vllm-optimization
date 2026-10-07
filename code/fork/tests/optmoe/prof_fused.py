"""opt-moe: per-phase clock64 accounting of the fused e4m3 kernel (debug build with -DME_DEBUG_VARIANTS, variant 164 =
DBG 64; OPTMOE_DBG_EXT = the debug .so). Thread 0 of every CTA; summed over CTAs and converted to ms of a 48-SM
machine (sum / CTAs / cycles-per-ms, cycles-per-ms from the total vs the event-timed wall).
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/prof_fused.py"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

TGV = (3164, 4460544)     # TG variants (opt-moe2: 4460544 = GLM53_MOE_E4M3_MAINLOOP=1, profiled)
NAMES = ["mainloop_gu", "mainloop_dn", "epi_gu", "actq", "epi_dn", "wait+prologue", "ticket", "total"]


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    so = os.environ.get("OPTMOE_DBG_EXT", os.path.join(HERE, "prev", "dbg",
                                                       "glm53_moe_e4m3_ext.cpython-312-aarch64-linux-gnu.so"))
    spec = importlib.util.spec_from_file_location("glm53_moe_e4m3_ext", so)
    ext = importlib.util.module_from_spec(spec)
    sys.modules["glm53_moe_e4m3_ext"] = ext
    spec.loader.exec_module(ext)
    import glm53_moe_e4m3 as M

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    P = L._exl3_ptrs
    n_exp = len(L._exl3_inners)
    emap = prod.pin_exl3_expert_map(L, dev)
    variants = [int(v) for v in os.environ.get("PROF_VARIANTS", "164").split(",")]
    for T in [int(v) for v in os.environ.get("BENCH_T", "13824,4289").split(",")]:
        g = torch.Generator().manual_seed(T)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        ids = C.routing("real", T, T, dev)
        w = C.weights_for(T, T, dev).float()
        t = M.plan(prod, ids.to(torch.long), w, n_exp, emap)
        a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
        ms = int(t["seg_expert"].numel())
        sync = torch.zeros(1 + 2 * ms + 64, dtype=torch.int32, device=dev)
        base = (sync.data_ptr() + 4 * (2 + 2 * ms + 1)) & ~7
        off = (base - sync.data_ptr()) // 4
        for var in variants:
            for dt in ((torch.bfloat16,) if var in TGV else (torch.bfloat16, torch.float32)):
                out = torch.empty(T, 4096, dtype=dt, device=dev)
                res = []
                for rep in range(4):
                    if var in TGV:     # TG: one row per token
                        ext.gather_tok(x, P["gate_suh"], a8, asc, out)
                    else:
                        ext.gather2(x, t["local"], t["pos"], P["gate_suh"], a8, asc, out, t["topk"], n_exp)
                    sync.zero_()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    ext.fused(a8, asc, a8d, dsc, a16, out, P["gate_trellis"], P["up_trellis"], P["gate_svh"],
                              P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"], t["row_token"],
                              t["row_weight"], t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], sync,
                              float(C.LIMIT), 12, 0, var)
                    e.record()
                    torch.cuda.synchronize()
                    cnt = sync[off:off + 22].clone().view(torch.int64).cpu().tolist()
                    res.append((s.elapsed_time(e), cnt))
                wall, cnt = sorted(res)[len(res) // 2]
                ctas = 48
                cyc_ms = cnt[7] / ctas / wall
                parts = " | ".join(f"{NAMES[i]} {cnt[i] / ctas / cyc_ms:.2f}" for i in range(8))
                print(f"[T={T} v{var} {str(dt)[6:]}] wall {wall:.2f} ms ({cyc_ms / 1e3:.0f} MHz) | {parts} | items gu "
                      f"{cnt[8]} dn {cnt[9]} waited {cnt[10]}", flush=True)
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    H.run_main(main)
