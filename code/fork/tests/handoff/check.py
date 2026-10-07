#!/usr/bin/env python3
"""Assertions over one tests/handoff/run_engine.py output directory (host side, stdlib only; shadow.json + harness.log).
Usage: check.py <mode> <run dir> [--expect-fast k1,k2,...]
  exact       bit-identical items (GLM53_PREFILL_QUICKWINS=<items>, no MLA prefill): for every shadowed prefill forward the
              control (production twice) AND the feature pass equal production bit for bit (outputs, every KV byte, the
              side buffers); the expected fast paths ran (idx_gate may instead have turned itself off after its runtime
              check, logged); no fast path ran in any decode step
  mla         GLM53_MLA_PREFILL=1 (+ any quickwins): control identical; on every sparse-MLA prefill call the kernel wrote
              the FA2 wrapper's kv_indices and they equal production's for the same call bit for bit; the kernel's error
              vs the fp32 reference over the valid keys stays at FA2's short-context level (max rel L2 <= 0.5 % on every
              row); decode: every entry the last row of a step reads past the step (production's FA2 plan) is a slot of
              the request; no fast path in decode
  mla-defect  run with --mla-kv-indices 0 (the deploy-r16 defect): at least one decode step's last row reads slots that
              are NOT the request's (proves the check sees the defect)
Exit 0 iff every assertion holds.
"""
import json
import os
import sys

FAIL = []


def ck(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' ' + msg) if msg else ''}", flush=True)
    if not ok:
        FAIL.append(name)


def same(r):
    return r["state"]["equal"] and all(o["equal"] for o in r["out"])


def main():
    mode, d = sys.argv[1], sys.argv[2]
    expect = []
    if "--expect-fast" in sys.argv:
        expect = [x for x in sys.argv[sys.argv.index("--expect-fast") + 1].split(",") if x]
    j = json.load(open(os.path.join(d, "shadow.json")))
    log = open(os.path.join(d, "harness.log")).read()
    steps = j["steps"]
    ck("shadowed prefill forwards", len(steps) >= 2, f"{len(steps)}")
    ck("control: production twice is bit-identical on every forward (the harness is exact)",
       all(s.get("control") is not None and same(s["control"]) for s in steps))
    ck("no fast path in any decode step", not j["summary"]["decode_fast_paths"], str(j["summary"]["decode_fast_paths"]))
    ran = {}
    for s in steps:
        for k, v in s["fast_F"].items():
            ran[k] = ran.get(k, 0) + v
    print(f"INFO fast paths in the feature passes: {ran}")
    for k in expect:
        if k == "idx_gate_fast" and "idx_gate turned off" in log:
            ck("idx_gate: turned itself off after its runtime check (production's op served)", True)
            continue
        ck(f"fast path {k} ran", ran.get(k, 0) > 0)
    if mode == "exact":
        for s in steps:
            ck(f"step {s['step']} ({s['tokens']} tokens): feature pass bit-identical (outputs + every KV byte + side "
               f"buffers)", same(s["feature"]),
               "" if same(s["feature"]) else json.dumps(s["feature"]["first_module"]))
    elif mode == "mla":
        probe = j["mla_probe"]
        ck("sparse-MLA prefill calls probed", len(probe) > 0, str(len(probe)))
        ck("every call: kv_indices written by the kernel path == production's for the same call (bitwise)",
           all(p["kv_indices_written_by_installed"] and p["kv_indices_equal_prod"] for p in probe))
        worst = max(v[1] for p in probe for k, v in p.items() if k.startswith("installed_vs_ref_valid"))
        ck("kernel vs fp32 reference over the valid keys: max row rel L2 <= 0.5 %", worst <= 5e-3, f"{worst:.4%}")
        fa2 = max(v[1] for p in probe for k, v in p.items() if k.startswith("fa2_vs_ref_valid"))
        over = max(v[1] for p in probe for k, v in p.items() if k.startswith("fa2_vs_ref_fa2_addressing"))
        print(f"INFO production FA2 vs the same reference: max row rel L2 {fa2:.4%}; FA2 vs a reference that reads what "
              f"its plan addresses (kv_len = 2048 + ctx % 4 on a 2048-wide row): {over:.4%}")
        ds = j["decode_stale"]
        reads = [x for x in ds if x["last_row_over"]]
        foreign = [x for x in reads if not all(x["stale_own"])]
        ck("decode steps with a last-row read past the step exist (the check is live)", len(reads) > 0,
           f"{len(reads)}/{len(ds)}")
        ck("decode: every slot the last row reads past the step is the request's own", not foreign,
           f"{len(foreign)} steps read other slots, e.g. {foreign[:2]}")
    elif mode == "mla-defect":
        ds = j["decode_stale"]
        foreign = [x for x in ds if x["last_row_over"] and not all(x["stale_own"])]
        ck("defect reproduced: decode's last row reads slots that are not the request's", len(foreign) > 0,
           f"{len(foreign)} steps, e.g. {foreign[:1]}")
    else:
        raise SystemExit(__doc__)
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
