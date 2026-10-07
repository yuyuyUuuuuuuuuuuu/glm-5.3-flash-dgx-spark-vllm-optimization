"""Compare two state-handoff runs (tests/handoff/run_engine.py outputs): REF (feature off) vs RUN (feature on).

Bitwise, step by step (the engine steps are aligned while both runs schedule the same tokens):
  - every per-layer KV view row (KDA conv_state / recurrent state per slot, MLA KV pages, indexer K cache + scales,
    kpool pooled keys, kpool tail ring, drafter KV): 64-bit row hashes after each step
  - every raw KV allocation block (writes outside the views)
  - side buffers: SM90 kv_indices / top-k indices rows of the step
  - the step schedule itself (tokens, computed tokens, drafts)
At dump steps the differing rows are compared element by element (count, max |diff|, first index).
Outputs: generated tokens and their top-5 logprobs (exact float equality).
If both runs carry B (fresh-prefill consistency), prints KL(A_j || B_j) per position bucket for each.
Exit 0 iff everything is bit-identical (the consistency numbers are informational).
Usage: compare.py <ref dir> <run dir> [--max-report N]
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch


def load(d):
    """result.json + the step records, request ids canonicalized (vLLM appends a random suffix per run)."""
    r = json.load(open(os.path.join(d, "result.json")))
    rec = torch.load(os.path.join(d, "records.pt"), weights_only=False)["records"]
    canon = {}
    for x in rec:
        for key in ("sched", "computed", "spec"):
            x[key] = {canon.setdefault(k, f"req{len(canon)}"): v for k, v in x[key].items()}
    return r, rec


def kl_top5(pa, pb):
    keys = set(pa) | set(pb)
    if not keys:
        return float("nan")
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}
    lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values()))
    zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)


def consistency(res):
    if "B" not in res:
        return None
    ks, agree = [], 0
    for j, pb in enumerate(res["B"]):
        pa = res["logprobs"][j]
        ks.append(kl_top5(pa, pb))
        agree += max(pa, key=pa.get) == max(pb, key=pb.get)
    return ks, agree


def elem_diff(a, b):
    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or not a.is_floating_point():
        au, bu = a.view(torch.uint8) if a.element_size() == 1 else a, b.view(torch.uint8) if b.element_size() == 1 else b
        ne = au != bu
        mad = float("nan")
        if a.is_floating_point():
            mad = (a.float() - b.float()).abs().max().item()
    else:
        ne = a.view(torch.int16 if a.element_size() == 2 else torch.int32 if a.element_size() == 4 else torch.int64) != \
            b.view(torch.int16 if b.element_size() == 2 else torch.int32 if b.element_size() == 4 else torch.int64)
        mad = (a.float() - b.float()).abs().max().item()
    n = int(ne.sum().item())
    first = tuple(int(x) for x in torch.nonzero(ne)[0].tolist()) if n else None
    return n, ne.numel(), mad, first


def main():
    ref_d, run_d = sys.argv[1], sys.argv[2]
    maxr = int(sys.argv[sys.argv.index("--max-report") + 1]) if "--max-report" in sys.argv else 12
    ra, reca = load(ref_d)
    rb, recb = load(run_d)
    print(f"REF {ref_d} ({ra['label']}): {len(reca)} steps; RUN {run_d} ({rb['label']}): {len(recb)} steps")
    same = True
    if ra["prompt_ids"] != rb["prompt_ids"]:
        print("DIFFERENT PROMPTS: not comparable")
        return 2
    first_div = None
    reported = 0
    n = min(len(reca), len(recb))
    for i in range(n):
        a, b = reca[i], recb[i]
        sa = (a["sched"], a["computed"], a["spec"])
        sb = (b["sched"], b["computed"], b["spec"])
        kind = "PREFILL" if a["prefill"] else "decode "
        if sa != sb:
            print(f"step {i}: SCHEDULE differs: ref {sa} run {sb} (later steps not aligned; stop)")
            same = False
            first_div = first_div or (i, "schedule")
            break
        bad_views = []
        for name, ha in a["view_hash"].items():
            hb = b["view_hash"].get(name)
            if hb is None or hb.shape != ha.shape:
                bad_views.append((name, "shape"))
                continue
            rows = torch.nonzero(ha != hb).flatten().tolist()
            if rows:
                bad_views.append((name, rows))
        bad_raw = []
        for name, ha in a["raw_hash"].items():
            hb = b["raw_hash"].get(name)
            rows = torch.nonzero(ha != hb).flatten().tolist() if hb is not None and hb.shape == ha.shape else ["shape"]
            if rows:
                bad_raw.append((name, rows))
        bad_side = {k: (v, b["side"].get(k)) for k, v in a["side"].items() if b["side"].get(k) != v}
        fa = {k: v for k, v in b["fast"].items() if a["fast"].get(k) != v}
        status = "identical" if not (bad_views or bad_raw or bad_side) else "DIFFERS"
        print(f"step {i:3d} {kind} sched {a['sched']} computed {a['computed']} drafts {a['spec']}: {status}"
              f"{'  (fast paths in RUN: ' + str(fa) + ')' if fa else ''}")
        if status == "identical":
            continue
        same = False
        if first_div is None:
            first_div = (i, "state")
        for name, rows in bad_views[:maxr]:
            if isinstance(rows, str):
                print(f"    view {name}: {rows}")
                continue
            tot = a["view_hash"][name].numel()
            more = " ..." if len(rows) > 16 else ""
            print(f"    view {name}: rows {rows[:16]}{more} ({len(rows)} of {tot} rows)")
        if len(bad_views) > maxr:
            print(f"    ... {len(bad_views) - maxr} more views")
        for name, rows in bad_raw[:4]:
            print(f"    raw allocation (first layer {name}): blocks {rows[:16]}")
        for k, (va, vb) in bad_side.items():
            print(f"    side buffer {k}: ref {va} run {vb}")
        if a["dump"] and b["dump"] and reported < maxr:
            pa = os.path.join(ref_d, f"dump_step{i:03d}.pt")
            pb = os.path.join(run_d, f"dump_step{i:03d}.pt")
            if os.path.exists(pa) and os.path.exists(pb):
                da, db = torch.load(pa, weights_only=False), torch.load(pb, weights_only=False)
                for name, rows in bad_views:
                    if isinstance(rows, str) or name not in da or name not in db:
                        continue
                    ia = {r: k for k, r in enumerate(da[name]["rows"])}
                    ib = {r: k for k, r in enumerate(db[name]["rows"])}
                    for r in rows[:4]:
                        if r in ia and r in ib:
                            n_, tot, mad, first = elem_diff(da[name]["data"][ia[r]], db[name]["data"][ib[r]])
                            print(f"      {name} row {r}: {n_}/{tot} elements differ, max|diff| {mad:.4g}, first at "
                                  f"{first} (row shape {tuple(da[name]['data'][ia[r]].shape)} {da[name]['dtype']})")
                        else:
                            print(f"      {name} row {r}: present in ref {r in ia}, run {r in ib} (zero in the other)")
                    reported += 1
    if len(reca) != len(recb):
        print(f"step counts differ: ref {len(reca)} run {len(recb)}")
        same = False
    ga, gb = ra["gen"], rb["gen"]
    tok_div = next((j for j, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
    lp_div = next((j for j, (x, y) in enumerate(zip(ra["logprobs"], rb["logprobs"])) if x != y), None)
    print(f"outputs: tokens {'identical' if ga == gb else f'DIFFER from position {tok_div}'} ({len(ga)} vs {len(gb)}); "
          f"top-5 logprobs {'identical' if ra['logprobs'] == rb['logprobs'] else f'DIFFER from position {lp_div}'}")
    if ga != gb or ra["logprobs"] != rb["logprobs"]:
        same = False
        if lp_div is not None:
            j = lp_div
            print(f"    position {j}: ref {ra['logprobs'][j]}\n                 run {rb['logprobs'][j]}")
    for lab, r in (("REF", ra), ("RUN", rb)):
        c = consistency(r)
        if c:
            ks, agree = c
            bk = [(0, 4), (4, 16), (16, 64), (64, 10 ** 9)]
            parts = []
            for lo, hi in bk:
                v = [k for j, k in enumerate(ks) if lo <= j < hi]
                if v:
                    parts.append(f"[{lo},{min(hi, len(ks))}) {sum(v) / len(v):.4f}")
            print(f"consistency {lab} ({r['label']}): KL(decode A || fresh prefill B) mean {sum(ks) / len(ks):.4f} "
                  f"max {max(ks):.3f} | top-1 agree {agree}/{len(ks)} | by position: {'; '.join(parts)}")
    print(f"RESULT {'BIT-IDENTICAL' if same else 'DIFFERENT'}"
          f"{'' if first_div is None else f' (first divergence: step {first_div[0]}, {first_div[1]})'}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
