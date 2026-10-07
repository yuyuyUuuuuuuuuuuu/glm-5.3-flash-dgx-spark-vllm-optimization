#!/usr/bin/env python3
"""FKDA3 Part A: adversarial re-check of the fkda2 accuracy claim with an INDEPENDENT fp64 reference and new inputs.

Three backends on identical inputs:
  stock  = FlashKDA 17a037d stock arithmetic (built as _fk3s_C; SASS identical to the r16n-shipped c286213f build)
  fix    = the fkda2 precision build        (built as _fk3f_C; SASS identical to the r16o-shipped dd1788c2 build)
  triton = production's chunk_kda_with_fused_gate (what R15 / FLASHKDA=0 runs)
Reference: my own fp64 recurrence, state kept as h[K, V] (FLA convention, o = h^T q), written independently of
tests/fkda2/short_accuracy.py (whose reference keeps S[V, K]); the state-layout convention every backend uses is
DETECTED on a random non-symmetric initial state instead of assumed (the fkda2 rig takes min() over transposes for
Triton's final state).
Cases (all different seeds/inputs from the fkda2 rig): edge lengths 1..4097 incl. T<16, T<64, T not a multiple of
16/64, zero vs random initial state; varlen batches with length-1 rows and mixed initial states; gate regimes (real
A_log/dt_bias, fast decay at the lower bound, long memory, mixed); large-v and saturated-beta inputs; chained
calls (state carried through each backend's own fp32 final state) vs one long call; real mini-model dumps (all
calls, not only the >=120 subset).
"""
import argparse
import glob
import json
import math
import os
import sys

import torch

D = 128
LB = -5.0


def ref64(c, kv_layout):
    """exact KDA recurrence in fp64. kv_layout: how c['initial_state'] is laid out ('VK' = [.., V, K] or 'KV').
    Returns out [T,H,D] and final state in the SAME layout as the input convention."""
    q, k, v, g1, beta = (c[x][0].double() for x in ("q", "k", "v", "g", "beta"))
    A = torch.exp(c["A_log"].double().reshape(-1))            # [H]
    bias = c["dt_bias"].double().reshape(-1, D)                 # [H, D]
    cu = c["cu_seqlens"].tolist()
    H = q.shape[1]
    out = torch.zeros(q.shape[0], H, D, dtype=torch.float64, device=q.device)
    fins = []
    for n in range(len(cu) - 1):
        a, b = cu[n], cu[n + 1]
        s0 = c["initial_state"][n].double()
        h = s0.transpose(-1, -2).clone() if kv_layout == "VK" else s0.clone()   # h[H, K, V]
        qq = q[a:b] / torch.sqrt((q[a:b] ** 2).sum(-1, keepdim=True) + 1e-6) / math.sqrt(D)
        kk = k[a:b] / torch.sqrt((k[a:b] ** 2).sum(-1, keepdim=True) + 1e-6)
        gate = LB * torch.sigmoid(A.view(1, H, 1) * (g1[a:b] + bias.view(1, H, D)))   # log-decay over K, [n,H,K]
        bt = 1.0 / (1.0 + torch.exp(-beta[a:b]))                                     # [n,H]
        dec = torch.exp(gate)
        for t in range(b - a):
            h = h * dec[t].unsqueeze(-1)                                   # decay rows (K)
            pred = torch.einsum("hk,hkv->hv", kk[t], h)
            delta = (v[a + t] - pred) * bt[t].unsqueeze(-1)
            h = h + kk[t].unsqueeze(-1) * delta.unsqueeze(-2)
            out[a + t] = torch.einsum("hk,hkv->hv", qq[t], h)
        fins.append(h.transpose(-1, -2) if kv_layout == "VK" else h)
    return out, torch.stack(fins)


def run_fk(ns, c):
    T, H = c["q"].shape[1], c["q"].shape[2]
    N = c["cu_seqlens"].numel() - 1
    op = getattr(torch.ops, ns)
    ws = torch.empty(int(op.get_workspace_size(T, H, N)), dtype=torch.uint8, device="cuda")
    out = torch.full((1, T, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    fin = torch.full((N, H, D, D), float("nan"), dtype=torch.float32, device="cuda")
    op.fwd(c["q"].contiguous(), c["k"].contiguous(), c["v"].contiguous(), c["g"].contiguous(), c["beta"], D ** -0.5,
           out, ws, c["A_log"].reshape(-1).contiguous(), c["dt_bias"].reshape(-1, D).contiguous(), LB,
           c["initial_state"].contiguous(), fin, c["cu_seqlens"].contiguous(), None, None)
    return out[0], fin


def run_tr(c):
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate
    o, h = chunk_kda_with_fused_gate(
        q=c["q"].clone(), k=c["k"].clone(), v=c["v"].contiguous().clone(), raw_g=c["g"].contiguous(),
        beta=c["beta"].float().sigmoid(), A_log=c["A_log"], g_bias=c["dt_bias"],
        initial_state=c["initial_state"].clone(), output_final_state=True, use_qk_l2norm_in_kernel=True,
        cu_seqlens=c["cu_seqlens"], safe_gate=True, lower_bound=LB)
    return o[0], h


def rel(a, b):
    a, b = a.double(), b.double()
    return float(((a - b) ** 2).sum().sqrt() / (b ** 2).sum().sqrt().clamp_min(1e-300))


def tokmax(o, ref):
    e = ((o.double() - ref) ** 2).sum((1, 2)).sqrt()
    r = (ref ** 2).sum((1, 2)).sqrt().clamp_min(1e-30)
    return float((e / r).max())


class Gen:
    def __init__(self, seed, real):
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.real = real

    def rn(self, *s, sc=1.0, mu=0.0):
        return (torch.randn(*s, generator=self.g) * sc + mu).cuda().to(torch.bfloat16)

    def case(self, lens, H=32, init="rand", regime="real", vsc=1.0, beta_sc=1.0, beta_mu=0.0, state_sc=None):
        T = sum(lens)
        c = dict(q=self.rn(1, T, H, D), k=self.rn(1, T, H, D), v=self.rn(1, T, H, D, sc=vsc),
                 beta=self.rn(1, T, H, sc=beta_sc, mu=beta_mu))
        if regime == "real" and self.real is not None:
            c["A_log"], c["dt_bias"] = self.real[0][:H].clone(), self.real[1][:H].clone()
            c["g"] = self.rn(1, T, H, D, sc=0.5)
        elif regime == "fast":           # gate pinned near the lower bound: strong decay every token
            c["A_log"] = torch.full((H,), 1.0, device="cuda")
            c["dt_bias"] = torch.full((H, D), 4.0, device="cuda")
            c["g"] = self.rn(1, T, H, D, sc=1.0, mu=3.0)
        elif regime == "long":           # almost no decay: long memory, state grows
            c["A_log"] = (torch.randn(H, generator=self.g) * 0.2).cuda()
            c["dt_bias"] = torch.full((H, D), -12.0, device="cuda")
            c["g"] = self.rn(1, T, H, D, sc=0.5, mu=-2.0)
        else:                            # mixed per channel
            c["A_log"] = (torch.randn(H, generator=self.g) * 0.5).cuda()
            c["dt_bias"] = (torch.rand(H, D, generator=self.g) * 16 - 12).cuda()
            c["g"] = self.rn(1, T, H, D, sc=2.0)
        N = len(lens)
        c["cu_seqlens"] = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32,
                                       device="cuda")
        s = torch.zeros(N, H, D, D, device="cuda")
        for n in range(N):
            want = init if init in ("rand", "zero") else ("rand" if n % 2 == 0 else "zero")
            if want == "rand":
                sc = state_sc if state_sc is not None else 0.05
                s[n] = (torch.randn(H, D, D, generator=self.g) * sc).cuda()
        c["initial_state"] = s
        return c


def evaluate(name, c, layout, backends, res):
    ref, fin = ref64(c, layout)
    ent = {"lens": (torch.diff(c["cu_seqlens"]).tolist()), "floor": rel(ref.to(torch.bfloat16), ref)}
    for bname, fn in backends.items():
        o, h = fn(c)
        torch.cuda.synchronize()
        ent[bname] = {"out": rel(o, ref), "tokmax": tokmax(o, ref), "state": rel(h, fin),
                      "finite": bool(torch.isfinite(o).all() and torch.isfinite(h).all())}
    res[name] = ent
    s = " ".join(f"{b}: out {ent[b]['out']:.2e} tok {ent[b]['tokmax']:.2e} st {ent[b]['state']:.2e}"
                 + ("" if ent[b]["finite"] else " NONFINITE") for b in backends)
    print(f"{name:42s} floor {ent['floor']:.2e} | {s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="layout,edge,varlen,regime,scale,chain,real")
    ap.add_argument("--real-dir", default="/fkda/in/kda_calls")
    ap.add_argument("--real-src", default="/fkda/in/src")
    ap.add_argument("--max-real", type=int, default=200)
    ap.add_argument("--first-real", type=int, default=0)
    ap.add_argument("--builds", default="stock=/fkda/builds/stock_n:_fk3s_C,fix=/fkda/builds/fix_n:_fk3f_C",
                    help="name=dir:module,... (each build registered under its own op namespace)")
    ap.add_argument("--no-triton", action="store_true")
    a = ap.parse_args()
    builds = {}
    for item in a.builds.split(","):
        name, rest = item.split("=")
        bdir, mod = rest.split(":")
        sys.path.insert(0, bdir)
        __import__(mod)
        builds[name] = mod
    import numpy as np
    real = None
    if os.path.exists(a.real_src + "/real_A_log.npy"):
        real = (torch.from_numpy(np.load(a.real_src + "/real_A_log.npy")).float().cuda().reshape(-1),
                torch.from_numpy(np.load(a.real_src + "/real_dt_bias.npy")).float().cuda().reshape(-1, D))
    backends = {name: (lambda c, m=mod: run_fk(m, c)) for name, mod in builds.items()}
    if not a.no_triton:
        backends["triton"] = run_tr
    groups = a.groups.split(",")
    res = {"cases": {}}
    R = res["cases"]

    # ---- 0. layout detection: which convention does each backend use for initial/final state?
    g = Gen(101, real)
    c = g.case([48], init="rand", state_sc=0.5)
    lay = {}
    for L in ("VK", "KV"):
        ref, _ = ref64(c, L)
        for b, fn in backends.items():
            o, _ = fn(c)
            lay.setdefault(b, {})[L] = rel(o, ref)
    res["layout"] = lay
    print("layout check (out rel err if initial_state is read as [V,K] / [K,V]):",
          {b: {L: f"{x:.2e}" for L, x in d.items()} for b, d in lay.items()}, flush=True)
    first = next(iter(backends))
    layout = "VK" if lay[first]["VK"] < lay[first]["KV"] else "KV"
    for b in backends:
        if (lay[b]["VK"] < lay[b]["KV"]) != (layout == "VK"):
            print(f"!! LAYOUT MISMATCH: {b} reads the state as the other convention", flush=True)
    res["layout_used"] = layout

    if "edge" in groups:
        g = Gen(202, real)
        for T in (1, 2, 3, 7, 15, 16, 17, 31, 33, 48, 63, 64, 65, 100, 127, 128, 129, 255, 257, 1000, 1023, 1025, 4097):
            for init in ("zero", "rand"):
                evaluate(f"edge T={T} init={init}", g.case([T], init=init), layout, backends, R)
    if "varlen" in groups:
        g = Gen(303, real)
        for lens in ([1, 1, 1, 1], [1, 17, 1, 64], [3, 129, 1000], [4096, 1, 1, 1], [16] * 5, [63, 65, 2, 300],
                     [1, 2047, 1], [7, 7, 7, 7, 7, 7, 7, 7]):
            for init in ("rand", "mixed"):
                evaluate(f"varlen {lens} init={init}", g.case(lens, init=init), layout, backends, R)
    if "regime" in groups:
        g = Gen(404, real)
        for reg in ("real", "fast", "long", "mixed"):
            for T in (64, 2048):
                evaluate(f"regime {reg} T={T}", g.case([T], init="rand", regime=reg), layout, backends, R)
    if "scale" in groups:
        g = Gen(505, real)
        evaluate("scale v x16 T=1024", g.case([1024], vsc=16.0), layout, backends, R)
        evaluate("scale beta sat+ (mu 8) T=1024", g.case([1024], beta_mu=8.0), layout, backends, R)
        evaluate("scale beta sat- (mu -8) T=1024", g.case([1024], beta_mu=-8.0), layout, backends, R)
        evaluate("scale state x20 T=256", g.case([256], state_sc=1.0), layout, backends, R)
        evaluate("scale H=16 (V-split path) T=2048", g.case([2048], H=16), layout, backends, R)
    if "chain" in groups:
        # one long call vs the same tokens as 6 chained calls (each backend carries its own fp32 final state)
        g = Gen(606, real)
        for reg in ("real", "long"):
            c = g.case([13824], init="rand", regime=reg)
            ref, fin = ref64(c, layout)
            ent = {"floor": rel(ref.to(torch.bfloat16), ref)}
            for b, fn in backends.items():
                o1, h1 = fn(c)
                outs, st = [], c["initial_state"]
                bounds = list(range(0, 13824 + 1, 2304))
                states = []
                for i in range(len(bounds) - 1):
                    s_, e_ = bounds[i], bounds[i + 1]
                    cc = dict(c)
                    for x in ("q", "k", "v", "g", "beta"):
                        cc[x] = c[x][:, s_:e_].contiguous()
                    cc["cu_seqlens"] = torch.tensor([0, e_ - s_], dtype=torch.int32, device="cuda")
                    cc["initial_state"] = st.float().contiguous()
                    o, st = fn(cc)
                    outs.append(o)
                    states.append(st)
                oc = torch.cat(outs, 0)
                per_seg = [rel(oc[bounds[i]:bounds[i + 1]], ref[bounds[i]:bounds[i + 1]]) for i in range(6)]
                ent[b] = {"single_out": rel(o1, ref), "single_state": rel(h1, fin), "chain_out": rel(oc, ref),
                          "chain_state": rel(st, fin), "chain_out_per_segment": per_seg,
                          "chain_vs_single_out": rel(oc, o1.double())}
            R[f"chain 6x2304 vs 1x13824 regime={reg}"] = ent
            print(f"chain regime={reg}: " + " | ".join(
                f"{b}: single out {ent[b]['single_out']:.2e} st {ent[b]['single_state']:.2e}; chained out "
                f"{ent[b]['chain_out']:.2e} st {ent[b]['chain_state']:.2e} seg {[f'{x:.2e}' for x in ent[b]['chain_out_per_segment']]}"
                for b in backends), flush=True)
    if "real" in groups:
        files = sorted(glob.glob(a.real_dir + "/*.pt"))[a.first_real: a.max_real]
        for f in files:
            c = torch.load(f)
            if c["q"].shape[1] > 4096:
                continue
            c = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in c.items()}
            c["lower_bound"] = c.get("lower_bound", LB)
            assert abs(c["lower_bound"] - LB) < 1e-9, c["lower_bound"]
            evaluate(f"real {os.path.basename(f)} call {c.get('call', '?')}", c, layout, backends, R)

    # ---- aggregates (geometric mean of ratios over cases, and how often fix beats triton / stock)
    agg = {}
    for key in ("out", "state", "tokmax"):
        names = list(backends)
        for pair in [(x, y) for i, x in enumerate(names) for y in names[i + 1:]] + [(y, x) for i, x in enumerate(names) for y in names[i + 1:]]:
            rs = [e[pair[0]][key] / e[pair[1]][key] for n, e in R.items() if isinstance(e.get(pair[0]), dict)
                  and "out" in e[pair[0]] and isinstance(e.get(pair[1]), dict) and "out" in e[pair[1]]
                  and e[pair[1]][key] > 0 and e[pair[0]][key] > 0 and math.isfinite(e[pair[0]][key] / e[pair[1]][key])]
            if rs:
                gm = math.exp(sum(math.log(x) for x in rs) / len(rs))
                agg[f"{key} {pair[0]}/{pair[1]}"] = {"geomean": gm, "n": len(rs), "worst": max(rs),
                                                     "frac_better": sum(r < 1 for r in rs) / len(rs)}
    res["agg"] = agg
    for k, v in agg.items():
        print(f"AGG {k:22s} geomean {v['geomean']:.3f} worst {v['worst']:.3f} better-in {v['frac_better']*100:.0f}% "
              f"of {v['n']}", flush=True)
    bad = [n for n, e in R.items() for b in backends if isinstance(e.get(b), dict) and e[b].get("finite") is False]
    print("non-finite:", bad or "none")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
