"""r16z7: pooled indexer keys each request WROTE (prefill + concurrent decode) vs a solo re-prefill (run_engine.py
HANDOFF_CONC_TEST=1 output conc_req<i>.pt), per MLA layer: relative error per pool, prompt / generated region,
pools with rel err > 0.25 or NaN (prefixhit-adv's criterion, tests/prefixhit/concan.py, + NaN + any request count).
Usage (production image, CPU): python3 tests/r16z7/concan.py <run dir> [...]   last line per run: CONC <run> bad ..."""
import os, sys, torch
for run in sys.argv[1:]:
    tot_bad = tot_gen = tot_pbad = tot_p = 0
    i = 0
    while os.path.exists(f"{run}/conc_req{i}.pt"):
        d = torch.load(f"{run}/conc_req{i}.pt", weights_only=False)
        P = d["prompt_len"] // 4
        print(f"{run} req {i}: prompt pools {P}, total pools {d['n'] // 4}")
        for name in sorted(d["ref"]):
            def deq(x):
                return x[:, :128].contiguous().view(torch.float8_e4m3fn).float() * x[:, 128:132].contiguous().view(torch.float32)
            a, b = deq(d["got"][name]), deq(d["ref"][name])
            rel = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
            bad = (rel > 0.25) | rel.isnan()
            def st(r, m):
                if r.numel() == 0:
                    return "-"
                rr = torch.nan_to_num(r, nan=float("inf"))
                return f"p50 {rr.median():.3g} p99 {rr.quantile(0.99):.3g} max {rr.max():.3g} bad(>0.25|NaN) {int(m.sum())}/{r.numel()}"
            print(f"   {name.split('.')[2]:>3}: prompt {st(rel[:P], bad[:P])} | gen {st(rel[P:], bad[P:])}")
            tot_bad += int(bad[P:].sum()); tot_gen += int(bad[P:].numel())
            tot_pbad += int(bad[:P].sum()); tot_p += int(bad[:P].numel())
        i += 1
    print(f"CONC {os.path.basename(run)} requests {i}: generated pools bad {tot_bad}/{tot_gen}, prompt pools bad {tot_pbad}/{tot_p}")
