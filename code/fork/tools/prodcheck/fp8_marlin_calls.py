"""Label every FP8-Marlin call of a production decode-step trace (torch profiler chrome trace, gzip) with its linear.

One decode step (between every second NCCL AllGather, as tools/prodcheck/trace_seg.py) holds 280 Marlin calls in a
fixed order: the DFlash2 drafter (5 layers x qkv, o, gate_up, down), the drafter's FP8 lm_head copy, then the 45
target layers (KDA: in_proj_qkvbfg_a [the only 128-thread Marlin instance], f_b, g_b, o_proj; MLA every 4th layer:
fused_qkv_a, q_b, o_proj; then dense MLP gate_up, down for layers 0-2, shared experts gate_up, down after). Steps
whose Marlin count differs are skipped. Per-rank (N, K) at TP=2 come from the model code + config (docs: report).
Usage: python3 tools/prodcheck/fp8_marlin_calls.py <trace.json.gz>
"""
import collections
import gzip
import json
import statistics
import sys

SHAPES = {"draft.qkv": (3072, 4096), "draft.o": (4096, 2048), "draft.gate_up": (12288, 4096),
          "draft.down": (4096, 6144), "draft.lmhead": (77440, 4096), "kda.in_proj": (12576, 4096),
          "kda.f_b": (4096, 128), "kda.g_b": (4096, 128), "kda.o_proj": (4096, 4096), "dense.gate_up": (12288, 4096),
          "dense.down": (4096, 6144), "mla.fused_qkv_a": (2048, 4096), "mla.q_b": (8192, 1536),
          "mla.o_proj": (4096, 8192), "shared.gate_up": (2048, 4096), "shared.down": (4096, 1024)}


def labels():
    lab = [f"draft.{p}" for _ in range(5) for p in ("qkv", "o", "gate_up", "down")] + ["draft.lmhead"]
    kinds = ["K", "K", "K", "M"] * 11 + ["K"]          # config layer_types: MLA (DSA) at 3, 7, ..., 43
    for i, t in enumerate(kinds):
        lab += (["kda.in_proj", "kda.f_b", "kda.g_b", "kda.o_proj"] if t == "K"
                else ["mla.fused_qkv_a", "mla.q_b", "mla.o_proj"])
        lab += ["dense.gate_up", "dense.down"] if i < 3 else ["shared.gate_up", "shared.down"]
    return lab


def main(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"],
               key=lambda e: e["ts"])
    ag = [i for i, e in enumerate(k) if "AllGather" in e["name"]]
    lab = labels()
    agg, used, tot = collections.defaultdict(list), 0, []
    for s in range(0, len(ag) - 2, 2):
        seq = [e for e in k[ag[s]:ag[s + 2]] if "Marlin" in e["name"]]
        if len(seq) != len(lab):
            continue
        thr = [e["args"]["block"][0] for e in seq]
        assert all((t == 128) == (lb == "kda.in_proj") for t, lb in zip(thr, lab)), "labelling out of step"
        used += 1
        tot.append(sum(e["dur"] for e in seq))
        for e, lb in zip(seq, lab):
            agg[lb].append(e["dur"])
    print(f"{path}: {used} steps labelled; Marlin per step median {statistics.median(tot) / 1000:.3f} ms")
    print(f"{'linear':16s} {'N':>6s} {'K':>5s} {'calls/step':>10s} {'median us':>9s} {'GB/s':>6s} {'ms/step':>8s}")
    s_all = 0.0
    for lb, v in agg.items():
        n, kk = SHAPES[lb]
        c, med = len(v) / used, statistics.median(v)
        s_all += sum(v) / used
        print(f"{lb:16s} {n:6d} {kk:5d} {c:10.0f} {med:9.1f} {(n * kk + 2 * n) / med / 1e3:6.0f} {sum(v) / used / 1000:8.3f}")
    print(f"total {s_all / 1000:.3f} ms/step")


if __name__ == "__main__":
    main(sys.argv[1])
