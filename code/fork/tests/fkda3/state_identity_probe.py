#!/usr/bin/env python3
"""FKDA3 probe: with no write (beta logits -40 -> sigmoid 4e-18) and no decay (dt_bias -40 -> gate ~ 0) the KDA
recurrence leaves the state untouched, so final_state must equal initial_state. Measures how much each backend
perturbs a carried fp32 state per call (the chunk-boundary carry production does every 13,824 tokens)."""
import sys, torch
sys.path.insert(0, "/w/tests/fkda3"); sys.path.insert(0, "/fkda/builds/stock_n"); sys.path.insert(0, "/fkda/builds/fix_n"); sys.path.insert(0, "/fkda/builds/f3_all")
import _fk3s_C, _fk3f_C, _fk3n_C  # noqa
from adv_accuracy import Gen, run_fk, run_tr, rel
for T in (1, 16, 64, 256, 4096):
    g = Gen(9, None)
    c = g.case([T], init="rand", regime="mixed", state_sc=0.3)
    c["beta"] = torch.full_like(c["beta"], -40.0)
    c["dt_bias"] = torch.full_like(c["dt_bias"], -40.0)
    c["A_log"] = torch.zeros_like(c["A_log"])
    s0 = c["initial_state"].clone()
    b16 = rel(s0.to(torch.bfloat16), s0)
    line = f"T={T:5d} bf16 round-trip floor {b16:.2e} |"
    for name, fn in (("stock", lambda c: run_fk("_fk3s_C", c)), ("fix", lambda c: run_fk("_fk3f_C", c)), ("fkda3", lambda c: run_fk("_fk3n_C", c)), ("triton", run_tr)):
        o, h = fn(c)
        line += f" {name}: final-vs-initial {rel(h, s0):.2e} eq-to-bf16(s0) {rel(h, s0.to(torch.bfloat16).float()):.2e}"
    # repeated carry: 20 calls in a row through each backend's own final state
    for name, fn in (("fix", lambda c: run_fk("_fk3f_C", c)), ("fkda3", lambda c: run_fk("_fk3n_C", c)), ("triton", run_tr)):
        cc = dict(c); st = s0
        for i in range(20):
            cc["initial_state"] = st.contiguous(); _, st = fn(cc)
        line += f" | {name} after 20 carries {rel(st, s0):.2e}"
    print(line, flush=True)
