"""Development probe (debug build: python3 tools/moee4m3/build.py --nvcc=-DME_DEBUG_VARIANTS): time the gate/up and
down kernels with parts removed (variants 100 + DBG, see mainloop()) on real layer-10 weights, real and perfectly
balanced routing (every expert 384 rows at T=13824). Timing only; outputs are garbage for DBG variants."""
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "tests", "moee4m3")]
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402


def ev(fn, reps=5):
    for _ in range(2):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_e4m3 as M
    E = M._ext()
    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    p = L._exl3_ptrs
    T = int(os.environ.get("ANA_T", "13824"))
    gvars = [int(v) for v in os.environ.get("ANA_GU", "0").split(",")]
    dvars = [v for v in os.environ.get("ANA_DN", "0,101,108,116,132").split(",")]
    for kind in os.environ.get("ANA_KINDS", "real,balanced").split(","):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
        if kind == "balanced":
            base = torch.arange(T * 8, device=dev) % 288
            ids = base.reshape(T, 8)
        else:
            ids = C.routing(kind, T, T, dev)
        w = C.weights_for(T, T, dev).float()
        t = M.plan(prod, ids, w, 288, None)
        a8, a8d, asc, dsc, a16 = M._buffers(prod, dev, t["P"])
        out = torch.zeros(T, 4096, dtype=torch.float32, device=dev)
        E.gather(x.half(), t["local"], t["pos"], p["gate_suh"], a8, asc, t["topk"], 288, 0)
        E.gateup(a8, asc, p["gate_trellis"], p["up_trellis"], p["gate_svh"], p["up_svh"], a16, t["seg_expert"],
                 t["seg_row0"], t["seg_rows"], t["num_segs"], C.LIMIT, 0, 0, 1, 0)
        E.actq(a16, t["row_expert"], p["down_suh"], a8d, dsc, t["seg_row0"], t["seg_rows"], t["num_segs"], 0, 1)
        line = []
        for v in gvars:
            ms = ev(lambda: E.gateup(a8, asc, p["gate_trellis"], p["up_trellis"], p["gate_svh"], p["up_svh"], a16,
                                     t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], C.LIMIT, v, 0, 1, 0))
            line.append(f"{v}:{ms:.2f}")
        print(f"[{kind} T={T}] gate/up " + " ".join(line), flush=True)
        line = []
        for v in dvars:
            var, grp = (0, int(v[1:])) if v.startswith("g") else (int(v), 0)
            ms = ev(lambda: E.down(a8d, dsc, p["down_trellis"], p["down_svh"], out, t["row_token"], t["row_weight"],
                                   t["seg_expert"], t["seg_row0"], t["seg_rows"], t["num_segs"], var, 0, 1, 0))
            line.append(f"{v}:{ms:.2f}")
        print(f"[{kind} T={T}] down    " + " ".join(line), flush=True)
        line = []
        for name, f in (("gather", lambda: E.gather(x.half(), t["local"], t["pos"], p["gate_suh"], a8, asc, t["topk"], 288, 0)),
                        ("gather-nopass1", lambda: E.gather(x.half(), t["local"], t["pos"], p["gate_suh"], a8, asc, t["topk"], 288, 101)),
                        ("gather-nostore", lambda: E.gather(x.half(), t["local"], t["pos"], p["gate_suh"], a8, asc, t["topk"], 288, 102)),
                        ("gather-neither", lambda: E.gather(x.half(), t["local"], t["pos"], p["gate_suh"], a8, asc, t["topk"], 288, 103)),
                        ("actq", lambda: E.actq(a16, t["row_expert"], p["down_suh"], a8d, dsc, t["seg_row0"], t["seg_rows"], t["num_segs"], 0, 1))):
            line.append(f"{name} {ev(f):.2f}")
        print(f"[{kind} T={T}] " + " ".join(line), flush=True)


if __name__ == "__main__":
    H.run_main(main)
