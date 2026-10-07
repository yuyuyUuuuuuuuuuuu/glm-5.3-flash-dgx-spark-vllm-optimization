"""moe3: P16 schedules vs production's apply_exl3_grouped_fat, per layer call, real layer-10 experts (TP=2 rank-0
shard), production's own arguments (captured from production's apply with prefill cap 1 installed, as deployed).
Interleaved rounds, median. Also checks every arm's h2 bit for bit against production's and its out against
production's within production's own atomics spread x 100.
Env (GPU_RUN_ENV): BENCH_T (13824,8192,4289,1791,512,300), BENCH_KINDS (real,collapsed), BENCH_ARMS (see ARMS),
BENCH_ROUNDS (7), BENCH_N (3)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/moe3/bench_p16.py
"""
from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

ARMS = {
    "p16b": {"mode": "p16b"},
    "auto": {"mode": "auto"},
    "sep1": {"mode": "sep", "nchunks": 1},
    "sep2": {"mode": "sep", "nchunks": 2},
    "sep4": {"mode": "sep", "nchunks": 4},
}


def sample(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def ndiff16(u, v):
    n = 0
    for i in range(0, u.shape[0], 8192):
        n += int((u[i:i + 8192].view(torch.int16) != v[i:i + 8192].view(torch.int16)).sum())
    return n


def rel(u, v):
    return float((u.double() - v.double()).norm() / v.double().norm().clamp_min(1e-300))


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import glm53_moe_fused16 as F
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    orig_gf = prod.apply_exl3_grouped_fat
    Ts = [int(v) for v in os.environ.get("BENCH_T", "13824,8192,4289,1791,512,300").split(",")]
    kinds = os.environ.get("BENCH_KINDS", "real,collapsed").split(",")
    arms = os.environ.get("BENCH_ARMS", "p16b,sep1,sep2").split(",")
    rounds = int(os.environ.get("BENCH_ROUNDS", "7"))
    n = int(os.environ.get("BENCH_N", "3"))
    table = []
    for kind in kinds:
        for T in Ts:
            g = torch.Generator().manual_seed(T)
            x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
            seed = T + (0 if kind == "real" else 50000)
            ids = C.routing(kind, T, seed, dev)
            w = C.weights_for(T, seed, dev).float()
            a = {}

            def capture(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit):
                a.update(xh=xh.clone(), counts=counts.clone(), token_sorted=token_sorted.clone(),
                         weight_sorted=weight_sorted.clone(), cap=cap, limit=limit, out_in=out.clone())
                return orig_gf(xh, out, counts, token_sorted, weight_sorted, layer, cap, limit)
            prod.apply_exl3_grouped_fat = capture
            prod.apply_exl3_experts(x, ids, w, L, limit=C.LIMIT)
            prod.apply_exl3_grouped_fat = orig_gf
            if not a:
                print(f"[{kind} T={T}] no grouped call (every expert at or below the cap)", flush=True)
                continue
            rows_cap = a["token_sorted"].numel()
            sc = prod._grouped_scratch(dev, rows_cap, 4096, 1024)
            ref = []
            for _ in range(2):
                o = a["out_in"].clone()
                orig_gf(a["xh"], o, a["counts"], a["token_sorted"], a["weight_sorted"], L, a["cap"], a["limit"])
                ref.append(o)
            h2p = sc["h2"][:rows_cap].clone()
            spread = rel(ref[0], ref[1])
            fns = {"prod": lambda: orig_gf(a["xh"], a["out_in"].clone(), a["counts"], a["token_sorted"],
                                           a["weight_sorted"], L, a["cap"], a["limit"])}
            ok = {}
            for arm in arms:
                keep = {}
                o = a["out_in"].clone()
                F.run(prod, a["xh"], o, a["counts"], a["token_sorted"], a["weight_sorted"], L, a["cap"], a["limit"],
                      keep=keep, sched=ARMS[arm])
                nr = int(keep["num_rows"].item())
                nd = ndiff16(keep["h2"][:nr], h2p[:nr])
                d = rel(o, ref[0])
                ok[arm] = (nd, d)
                fns[arm] = (lambda s=ARMS[arm]: F.run(prod, a["xh"], a["out_in"].clone(), a["counts"], a["token_sorted"],
                                                      a["weight_sorted"], L, a["cap"], a["limit"], sched=s))
            for f in fns.values():
                f()
            t = {k: [] for k in fns}
            names = list(fns)
            for r in range(rounds):
                for k in (names if r % 2 == 0 else names[::-1]):
                    t[k].append(sample(fns[k], n))
            med = {k: statistics.median(v) for k, v in t.items()}
            table.append((kind, T, med))
            print(f"[{kind} T={T}] rows {int((a['counts'] > a['cap']).sum())} fat experts; prod spread {spread:.1e}; "
                  + "; ".join(f"{k}: h2 diff {v[0]}, out rel {v[1]:.1e}" for k, v in ok.items()), flush=True)
            print(f"[{kind} T={T}] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items()) + " ms", flush=True)
            for k, (nd, d) in ok.items():
                assert nd == 0, f"{k}: h2 differs from production in {nd} values"
                assert d <= max(100 * spread, 2e-7), f"{k}: out rel {d} vs production spread {spread}"
            del x
            torch.cuda.empty_cache()
    print("TABLE kind T prod " + " ".join(arms) + " | best delta")
    for kind, T, m in table:
        best = min(arms, key=lambda k: m[k])
        print(f"TABLE {kind} {T} {m['prod']:.2f} " + " ".join(f"{m[k]:.2f}" for k in arms)
              + f" | {best} {m[best] - m['prod']:+.2f}")
    H.report_peak(8.0)


if __name__ == "__main__":
    H.run_main(main)
