#!/usr/bin/env python3
"""FKDA2: per-POSITION output error (and final-state error) of the KDA chunked prefill vs an fp64 reference, for the
production Triton chain (chunk_kda_with_fused_gate, what FlashKDA=0 runs) and for a FlashKDA build
(torch.ops._flashkda_fp32_C.fwd, the op the glm53_flashkda wrapper calls), on
  * REAL inputs captured from the handoff mini model (tests/fkda2/dump_kda_inputs.py: every chunk_prefill call's
    q/k/v/g1/raw beta/initial state/cu_seqlens + the layer's real A_log/dt_bias), and/or
  * synthetic "long" regime inputs (pf3000: g1~N(0,.5), dt_bias~U[-10,-2], real A_log/dt_bias when staged).
fp64 reference = the exact KDA recurrence (pf3000 kda_state_error.py), token by token, with the output o_t = S_t q_t:
    qn = l2norm(q) * D**-0.5 ; kn = l2norm(k) (eps 1e-6 inside the sqrt)
    gate = lower_bound * sigmoid(exp(A_log) * (g1 + dt_bias)) ; S *= exp(gate) (over K)
    u = (v - S kn) * sigmoid(beta) ; S += u kn^T ; o = S qn
Reported: rel-RMS of the output per position bin (relative to the reference's RMS in the bin), worst row, final state
rel-RMS; the bf16 floor = rel-RMS of the reference itself rounded to bf16 (both backends emit bf16).
Run in the production image with the FlashKDA build on PYTHONPATH (tests/fkda/gpu_run.sh; --ext-dir selects a build).
"""
import argparse
import glob
import json
import os
import sys

import torch

D = 128
BINS = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 1 << 30)]


def ref_fp64(q, k, v, g1, beta, A_log, dt_bias, lb, s0, cu):
    """per-row exact recurrence; returns out [T,H,D] fp64, final [N,H,V,K] fp64"""
    T, H = q.shape[1], q.shape[2]
    out = torch.empty(T, H, D, dtype=torch.float64, device=q.device)
    fin = torch.empty(len(cu) - 1, H, D, D, dtype=torch.float64, device=q.device)
    A = torch.exp(A_log.double()).view(H, 1)
    bias = dt_bias.double().view(H, D)
    for n in range(len(cu) - 1):
        a, b = int(cu[n]), int(cu[n + 1])
        S = s0[n].double().clone()                                   # [H,V,K]
        qc, kc, vc = q[0, a:b].double(), k[0, a:b].double(), v[0, a:b].double()
        qn = qc / (qc.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * D ** -0.5
        kn = kc / (kc.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = torch.exp(lb * torch.sigmoid(A * (g1[0, a:b].double() + bias)))   # [n,H,K]
        bt = torch.sigmoid(beta[0, a:b].double())                     # [n,H]
        for i in range(b - a):
            S = S * decay[i].view(H, 1, D)
            u = (vc[i] - torch.einsum("hvk,hk->hv", S, kn[i])) * bt[i].view(H, 1)
            S = S + u.view(H, D, 1) * kn[i].view(H, 1, D)
            out[a + i] = torch.einsum("hvk,hk->hv", S, qn[i])
        fin[n] = S
    return out, fin


def run_triton(c):
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate
    v = c["v"].contiguous().clone()   # the chain writes its output into v
    o, h = chunk_kda_with_fused_gate(
        q=c["q"].clone(), k=c["k"].clone(), v=v, raw_g=c["g"].contiguous(), beta=c["beta"].float().sigmoid(),
        A_log=c["A_log"], g_bias=c["dt_bias"], initial_state=c["initial_state"].clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=c["cu_seqlens"], safe_gate=True, lower_bound=c["lower_bound"])
    return o[0], h


def run_flashkda(c):
    T, H = c["q"].shape[1], c["q"].shape[2]
    N = c["cu_seqlens"].numel() - 1
    ws = torch.empty(int(torch.ops._flashkda_fp32_C.get_workspace_size(T, H, N)), dtype=torch.uint8, device="cuda")
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.empty(N, H, D, D, dtype=torch.float32, device="cuda")
    torch.ops._flashkda_fp32_C.fwd(c["q"].contiguous(), c["k"].contiguous(), c["v"].contiguous(), c["g"].contiguous(),
                                   c["beta"], D ** -0.5, out, ws, c["A_log"].reshape(-1).contiguous(),
                                   c["dt_bias"].reshape(-1, D).contiguous(), c["lower_bound"],
                                   c["initial_state"].contiguous(), fin, c["cu_seqlens"].contiguous(), None, None)
    return out[0], fin


def stats(o, ref, cu):
    """o, ref [T,H,D]; per-position-bin rel-RMS (bin = position inside its row)"""
    e = (o.double() - ref).pow(2).sum((1, 2))
    r = ref.pow(2).sum((1, 2))
    pos = torch.cat([torch.arange(int(cu[n + 1]) - int(cu[n])) for n in range(len(cu) - 1)]).to(o.device)
    res = {}
    for lo, hi in BINS:
        m = (pos >= lo) & (pos < hi)
        if m.any():
            res[f"{lo}-{hi if hi < 1 << 30 else 'inf'}"] = [float((e[m].sum() / r[m].sum()).sqrt()), int(m.sum())]
    res["all"] = [float((e.sum() / r.sum()).sqrt()), int(e.numel())]
    tok = (e / r.clamp_min(1e-30)).sqrt()
    res["tok_p95"] = float(tok.quantile(0.95)) if tok.numel() > 1 else float(tok[0])
    res["tok_max"] = float(tok.max())
    return res


def relrms(a, b):
    return float(((a.double() - b.double()).pow(2).sum() / b.double().pow(2).sum().clamp_min(1e-300)).sqrt())


def synthetic(T, seed, real_gate):
    g = torch.Generator(device="cpu").manual_seed(seed)
    H = 32

    def rn(*s, scale=1.0):
        return (torch.randn(*s, generator=g) * scale).cuda().to(torch.bfloat16)
    c = dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, scale=0.5), beta=rn(1, T, H),
             initial_state=torch.zeros(1, H, D, D, device="cuda"),
             cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device="cuda"), lower_bound=-5.0,
             A_log=(torch.randn(H, generator=g) * 0.2).cuda(), dt_bias=(torch.rand(H, D, generator=g) * 8 - 10).cuda())
    if real_gate:
        import numpy as np
        c["A_log"] = torch.from_numpy(np.load(real_gate + "/real_A_log.npy")).float().cuda().reshape(-1)
        c["dt_bias"] = torch.from_numpy(np.load(real_gate + "/real_dt_bias.npy")).float().cuda().reshape(H, D)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="", help="glob of dump_kda_inputs .pt files")
    ap.add_argument("--max-tokens", type=int, default=4096, help="skip dumped calls longer than this")
    ap.add_argument("--synthetic", default="", help="comma list of T for synthetic long-regime inputs")
    ap.add_argument("--real-gate", default="/pf3000")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ext-dir", default="", help="directory holding the _flashkda_fp32_C build to test")
    ap.add_argument("--first-call", type=int, default=0, help="skip dumped calls with a lower call index (boot)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.ext_dir:
        sys.path.insert(0, args.ext_dir)
    import _flashkda_fp32_C  # noqa: F401
    print("flashkda build:", _flashkda_fp32_C.__file__, flush=True)
    cases = []
    for f in sorted(glob.glob(args.calls)) if args.calls else []:
        c = torch.load(f)
        if c["q"].shape[1] > args.max_tokens or c.get("call", 0) < args.first_call:
            continue
        c = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in c.items()}
        cases.append((os.path.basename(f) + f":L{c.get('prefix', '')}", c))
    for T in [int(x) for x in args.synthetic.split(",") if x]:
        cases.append((f"synthetic_T{T}", synthetic(T, T, args.real_gate if os.path.exists(args.real_gate) else "")))
    rep = {"tag": args.tag, "cases": {}}
    agg = {"fk": {}, "tr": {}, "floor": {}}
    for name, c in cases:
        cu = c["cu_seqlens"].cpu().tolist()
        ref, fin = ref_fp64(c["q"], c["k"], c["v"], c["g"], c["beta"], c["A_log"], c["dt_bias"], c["lower_bound"],
                            c["initial_state"], cu)
        ent = {"T": int(c["q"].shape[1]), "rows": [b - a for a, b in zip(cu, cu[1:])],
               "has_state": [bool(c["initial_state"][n].abs().max() > 0) for n in range(len(cu) - 1)],
               "floor": stats(ref.to(torch.bfloat16), ref, cu)}
        o, h = run_flashkda(c)
        torch.cuda.synchronize()
        ent["fk"] = stats(o, ref, cu)
        ent["fk_state"] = relrms(h, fin)
        if not args.no_triton:
            ot, ht = run_triton(c)
            torch.cuda.synchronize()
            ent["tr"] = stats(ot, ref, cu)
            ent["tr_state"] = min(relrms(ht, fin), relrms(ht.transpose(-1, -2), fin))
            # how far FlashKDA is from the Triton chain (what a KL-vs-a-Triton-reference run sees), relative to |ref|
            ent["fk_vs_tr"] = float(((o.double() - ot.double()).pow(2).sum() / ref.pow(2).sum()).sqrt())
            ent["ref_vs_tr"] = float(((ref.to(torch.bfloat16).double() - ot.double()).pow(2).sum()
                                      / ref.pow(2).sum()).sqrt())
        rep["cases"][name] = ent
        line = f"{name:34s} T={ent['T']:5d} rows={len(cu) - 1} | out rel-RMS all: floor {ent['floor']['all'][0]:.2e}"
        line += f" FK {ent['fk']['all'][0]:.2e}" + (f" TR {ent['tr']['all'][0]:.2e}" if "tr" in ent else "")
        line += f" | state FK {ent['fk_state']:.2e}" + (f" TR {ent['tr_state']:.2e}" if "tr" in ent else "")
        if "fk_vs_tr" in ent:
            line += f" | FK-vs-TR {ent['fk_vs_tr']:.2e} exact(bf16)-vs-TR {ent['ref_vs_tr']:.2e}"
        print(line, flush=True)
        for b in ent["fk"]:
            if b in ("tok_p95", "tok_max", "all"):
                continue
            for kk in ("fk", "tr", "floor"):
                if kk in ent and b in ent[kk]:
                    s = agg[kk].setdefault(b, [0.0, 0])
                    s[0] += ent[kk][b][0] ** 2 * ent[kk][b][1]
                    s[1] += ent[kk][b][1]
    real = [e for k, e in rep["cases"].items() if not k.startswith("synthetic")]
    if real and "fk_vs_tr" in real[0]:
        m = lambda key: sum(e[key] for e in real) / len(real)  # noqa: E731
        rep["real_mean"] = {"fk_out": sum(e["fk"]["all"][0] for e in real) / len(real),
                            "tr_out": sum(e["tr"]["all"][0] for e in real) / len(real),
                            "fk_state": m("fk_state"), "tr_state": m("tr_state"),
                            "fk_vs_tr": m("fk_vs_tr"), "exact_vs_tr": m("ref_vs_tr")}
        print("real-input means:", json.dumps({k: float(f"{v:.3e}") for k, v in rep["real_mean"].items()}))
    rep["agg_by_bin"] = {kk: {b: (v[0] / v[1]) ** 0.5 for b, v in d.items()} for kk, d in agg.items()}
    print("aggregate per-position-bin output rel-RMS (token-weighted over all cases):")
    for b in rep["agg_by_bin"]["fk"]:
        print(f"   pos {b:>10s}: floor {rep['agg_by_bin']['floor'][b]:.3e}  FK {rep['agg_by_bin']['fk'][b]:.3e}"
              + (f"  TR {rep['agg_by_bin']['tr'][b]:.3e}" if b in rep['agg_by_bin']['tr'] else ""))
    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
