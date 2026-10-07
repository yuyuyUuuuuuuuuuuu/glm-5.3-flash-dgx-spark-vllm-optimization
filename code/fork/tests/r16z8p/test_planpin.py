"""r16z8p: GLM53_MLA_PLAN_PIN (glm53_mla_planpin.py) on nodeC against production's REAL sparse-MLA plan path:
the image + the launcher-mounted flashinfer_mla_sparse_sm90.py.patched + flashinfer 0.6.18 (tests/hostloop_gpu.sh).

  P0  source fingerprints of _SM90State.plan / BatchMLAPagedAttentionWrapper.plan == the module's VERIFIED values
  P1  env parsing: unset/''/0 off, 1 on, anything else refused (-> off, plugin line says so)
  P2  plugin off: nothing installed (_SM90State.plan untouched), the off line is logged
  P3  plugin on: installed through the module already imported; production's plan kept as _glm53_orig
  P4  sequential exactness at max_tokens 16384 (production) and 7168: a sequence of plans (decode K+1 rows, prefill
      chunks, full 16384, edge sizes) through production's plan (state A) and the patched plan (state B): after every
      plan the device qo/kv indptr, kv_len_arr, the WHOLE 8 MiB device int workspace and _plan_info are identical
  P5  async race stress (the GPU kept behind the host: a spin kernel queued after every plan, no host sync): stream-
      ordered snapshots of the device plan buffers + int workspace taken right after each plan == the reference of
      that plan, for (a) the patched ring (0 mismatches, ring waits counted) and (c) production pageable; (b) a NAIVE
      pinning (production's plan with the three staging tensors merely pinned, no ring, the shared page-locked int
      workspace) must show mismatches - proving the stress detects the hazard the ring removes
  P6  flashinfer's plan copies are ordered on torch's CURRENT stream (default and a side stream): the int workspace
      and indptr copies queue behind a running spin kernel (the per-slot event recorded on that stream covers them)
  P7  inside a CUDA-graph capture the patched plan raises production's RuntimeError; with the module disabled
      (ST.enabled False) the call goes to production's plan
  P8  logs: self-test line at the first plan, serving-confirmed at 64 plans; production attribute names updated
Usage: tests/hostloop_gpu.sh python3 tests/r16z8p/test_planpin.py
"""
from __future__ import annotations

import importlib
import logging
import os
import random
import sys

sys.path.insert(0, "/w")
import torch  # noqa: E402

FAILS: list[str] = []


def ck(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, r):
        self.lines.append(r.getMessage())


SM90 = importlib.import_module("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90")
FIM = importlib.import_module("flashinfer.mla")
import glm53_mla_planpin as PP  # noqa: E402

cap = Cap()   # (after the vllm import: its logging config would disable a logger configured before it)
lg = logging.getLogger("vllm.glm53_mla_planpin")
lg.disabled = False
lg.addHandler(cap)
lg.setLevel(logging.INFO)
NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count

dev = torch.device("cuda")
ORIG = SM90._SM90State.plan
TOPK = 2048

# ---- P0
fp_sm90 = PP.source_fingerprint(SM90._SM90State.plan)
fp_fi = PP.source_fingerprint(FIM.BatchMLAPagedAttentionWrapper.plan)
print(f"fingerprints: _SM90State.plan {fp_sm90}, BatchMLAPagedAttentionWrapper.plan {fp_fi}")
ck(fp_sm90 in PP.VERIFIED["_SM90State.plan"] and fp_fi in PP.VERIFIED["BatchMLAPagedAttentionWrapper.plan"],
   "P0 production fingerprints == VERIFIED")
if os.environ.get("PLANPIN_FP_ONLY"):
    sys.exit(0)

# ---- P1
ok = PP.env_mode({}) == "off" and PP.env_mode({PP.ENV: ""}) == "off" and PP.env_mode({PP.ENV: "0"}) == "off" \
    and PP.env_mode({PP.ENV: "1"}) == "on"
bad = 0
for v in ("2", "on", "true", "yes", "verify", "-1"):
    try:
        PP.env_mode({PP.ENV: v})
    except ValueError:
        bad += 1
ck(ok and bad == 6, f"P1 env: unset/''/0 off, 1 on, 6/6 other values refused ({bad})")

# ---- P2
os.environ.pop(PP.ENV, None)
PP.plugin_install()
ck(SM90._SM90State.plan is ORIG and not PP.ST.installed
   and any("-> off, production's pageable plan staging unchanged" in m for m in cap.lines),
   "P2 plugin off (unset): _SM90State.plan untouched, off line logged")
os.environ[PP.ENV] = "2"
PP.plugin_install()
ck(SM90._SM90State.plan is ORIG and any("must be unset, empty, 0 or 1" in m for m in cap.lines),
   "P2 plugin with a refused value: nothing installed, '-> off (...)' logged")

# ---- P3
os.environ[PP.ENV] = "1"
PP.plugin_install()
NEW = SM90._SM90State.plan
ck(getattr(NEW, "_glm53_planpin", False) and NEW._glm53_orig is ORIG and PP.ST.installed
   and any(m.startswith("glm53_mla_planpin: patched _SM90State.plan") for m in cap.lines),
   "P3 plugin on: patched, production's plan kept, install line logged")
PP.plugin_install()   # idempotent
ck(SM90._SM90State.plan is NEW, "P3 a second plugin_install keeps the single wrapper")


def mk(mt):
    s = SM90._SM90State(dev, 32, torch.float8_e4m3fn, mt, TOPK, kv_lora_rank=512, qk_rope_head_dim=0,
                        sm_scale=1.0 / 16.0)
    s.wrapper._int_workspace_buffer.zero_()
    s.wrapper._pin_memory_int_workspace_buffer.zero_()
    return s


def bufs(s):
    w = s.wrapper
    return [w._qo_indptr_buf, w._kv_indptr_buf, w._kv_len_arr_buf, w._int_workspace_buffer]


def ws_view(ws_bytes, info):
    """The int-workspace bytes the MLA kernels read (flashinfer MLAPlan, scheduler.cuh): every work array's first
    total_num_works entries, the five merge arrays (num_sm entries each, fully written), work_indptr's num_clusters + 1.
    Each work array is ALLOCATED for 16384 works and only its head is written: the tail keeps whatever an earlier
    plan left in the page-locked buffer (never read), so it is excluded from the equality."""
    info = [int(x) for x in info]
    w = ws_bytes.view(torch.int32)
    ncl = info[1]
    wi = w[info[15] // 4: info[15] // 4 + ncl + 1]
    tot = int(wi[-1])
    parts = [wi]
    for k in (2, 3, 4, 10, 11, 12, 13, 14):
        parts.append(w[info[k] // 4: info[k] // 4 + tot])
    for k in (5, 6, 7, 8, 9):
        parts.append(w[info[k] // 4: info[k] // 4 + NUM_SM])
    return torch.cat(parts)


def view(bl, info):
    return bl[:3] + [ws_view(bl[3], info)]


def same_view(b1, i1, b2, i2):
    return list(i1) == list(i2) and all(torch.equal(x, y) for x, y in zip(view(b1, i1), view(b2, i2)))


def seq_sizes(mt, rng, n):
    base = [1, 5, 6, 8, 40, 7, 4096, mt, mt - 1, 2, 16, 64, 1000, 3, mt // 2, 8, 8, 6]
    out = [x for x in base if 1 <= x <= mt]
    while len(out) < n:
        out.append(rng.choice([5, 6, 8, rng.randint(1, mt)]))
    return out[:n]


def lens_for(n, rng):
    return torch.tensor([rng.randint(1, TOPK) for _ in range(n)], dtype=torch.int32)


# ---- P4 sequential exactness
for mt in (16384, 7168):
    rng = random.Random(mt)
    A, B = mk(mt), mk(mt)
    diff_rows = whole_diff = 0
    plans = 0
    for n in seq_sizes(mt, rng, 40):
        lens = lens_for(n, rng)
        ORIG(A, n, lens)
        NEW(B, n, lens)
        torch.cuda.synchronize()
        plans += 1
        diff_rows += int(not same_view(bufs(A), A.wrapper._plan_info, bufs(B), B.wrapper._plan_info))
        whole_diff += int(not torch.equal(bufs(A)[3], bufs(B)[3]))
    ck(diff_rows == 0, f"P4 max_tokens {mt}: {plans} plans, device qo/kv indptr + kv_len_arr + plan_info + every "
       f"int-workspace byte the kernels read identical after every plan ({plans - diff_rows}/{plans}; the unread "
       f"stale tails of the 16384-entry work arrays differ in {whole_diff})")
    ck(B._qo_cpu.is_pinned() and B._kv_cpu.is_pinned() and B._lens_cpu.is_pinned() and not A._qo_cpu.is_pinned(),
       f"P8 max_tokens {mt}: production attribute names point at the pinned staging (B) / pageable (A)")
ck(any("self-test: 3/3 device plan buffers == pinned staging" in m for m in cap.lines),
   "P8 first-plan self-test line logged (3/3)")
ck(any("serving confirmed (mode on): 64 plans through the pinned ring" in m for m in cap.lines),
   "P8 serving-confirmed line at 64 plans")

# ---- P5 async race stress
cyc_per_ms = None


def calib():
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(10_000_000)
    e.record()
    e.synchronize()
    return 10_000_000 / s.elapsed_time(e)


cyc_per_ms = calib()


def stress(variant, mt, nplans, seed, spin_ms=3.0):
    """Back-to-back plans with the GPU behind the host. Returns (#plans whose snapshot != reference, #in ws only)."""
    rng = random.Random(seed)
    sizes = seq_sizes(mt, rng, nplans)
    lens_l = [lens_for(n, rng) for n in sizes]
    # reference: production's plan, fully synchronous, snapshot after each plan
    R = mk(mt)
    ref = []
    infos = []
    for n, lens in zip(sizes, lens_l):
        ORIG(R, n, lens)
        torch.cuda.synchronize()
        ref.append([b.clone() for b in bufs(R)])
        infos.append(list(R.wrapper._plan_info))
    S = mk(mt)
    if variant == "naive":
        S._arange_cpu = torch.arange(mt + 1, dtype=torch.int32).pin_memory()
        S._qo_cpu = torch.empty(mt + 1, dtype=torch.int32, pin_memory=True)
        S._kv_cpu = torch.empty(mt + 1, dtype=torch.int32, pin_memory=True)
        S._lens_cpu = torch.full((mt,), TOPK, dtype=torch.int32).pin_memory()
        fn = ORIG
    elif variant == "ring":
        fn = NEW
    else:
        fn = ORIG
    snaps = [[torch.empty_like(b) for b in bufs(S)] for _ in sizes]
    torch.cuda.synchronize()
    w0 = PP.ST.waits
    torch.cuda._sleep(int(20 * cyc_per_ms))          # the GPU starts behind
    for i, (n, lens) in enumerate(zip(sizes, lens_l)):
        fn(S, n, lens)
        for d, b in zip(snaps[i], bufs(S)):          # what the forward after plan i would read (stream order)
            d.copy_(b, non_blocking=True)
        torch.cuda._sleep(int(spin_ms * cyc_per_ms))  # the forward / drafter of that step
    torch.cuda.synchronize()
    bad = ws_only = 0
    for i in range(len(sizes)):
        eq = [torch.equal(a, b) for a, b in zip(view(snaps[i], infos[i]), view(ref[i], infos[i]))]
        if not all(eq):
            bad += 1
            ws_only += int(all(eq[:3]) and not eq[3])
    return bad, ws_only, PP.ST.waits - w0


for mt in (16384, 7168):
    b_ring, _, waits = stress("ring", mt, 24, 7)
    ck(b_ring == 0 and waits > 0, f"P5 max_tokens {mt} ring: 0/24 snapshots differ from the reference ({b_ring}); "
       f"ring waits {waits} (back-to-back plans: slot reuse waited for its copies)")
    b_prod, ws_prod, _ = stress("prod", mt, 24, 7)
    if mt == 16384:
        ck(b_prod == 0, f"P5 max_tokens {mt} production pageable: {b_prod}/24 snapshots differ (the 65540 B indptr "
           f"copy syncs the host with the stream before every plan)")
    else:   # informational: production's own latent hazard below 64 KiB (no implicit sync), which the ring removes
        print(f"     info: production pageable at max_tokens {mt} (copies <= 64 KiB are async, no implicit sync): "
              f"{b_prod}/24 snapshots differ, {ws_prod} only in the int workspace (flashinfer's shared page-locked "
              f"schedule buffer rewritten before the previous plan's cudaMemcpyAsync ran)")
    b_naive, ws_only, _ = stress("naive", mt, 24, 7)
    print(f"     naive pinning (no ring): {b_naive}/24 snapshots differ ({ws_only} only in the int workspace)")
    if mt == 16384:
        ck(b_naive > 0, f"P5 the stress detects the hazard: naive pinning at max_tokens {mt} gives {b_naive}/24 wrong "
           f"snapshots (so the ring's 0 is meaningful)")

# ---- P6 stream ordering of flashinfer's plan copies (int workspace via MLAPlan's cudaMemcpyAsync + torch copy_)
for label, use_side in (("default stream", False), ("side stream", True)):
    side = torch.cuda.Stream()
    reader = torch.cuda.Stream()
    S = mk(16384)
    n = 8
    lens = lens_for(n, random.Random(3))
    ctx = torch.cuda.stream(side) if use_side else torch.cuda.stream(torch.cuda.current_stream())
    with ctx:
        ORIG(S, n, lens)   # make the device buffers non-zero first
        torch.cuda.synchronize()
        before_ws = S.wrapper._int_workspace_buffer.clone()
        before_qo = S.wrapper._qo_indptr_buf.clone()
        torch.cuda._sleep(int(200 * cyc_per_ms))
        NEW(S, 16384, lens_for(16384, random.Random(4)))   # different shape -> different schedule + indptr
        with torch.cuda.stream(reader):                     # read device memory while the spin still runs
            peek_ws = S.wrapper._int_workspace_buffer.clone()
            peek_qo = S.wrapper._qo_indptr_buf.clone()
        reader.synchronize()
        still = torch.equal(peek_ws, before_ws) and torch.equal(peek_qo, before_qo)
        torch.cuda.synchronize()
        changed = not torch.equal(S.wrapper._int_workspace_buffer, before_ws) and \
            not torch.equal(S.wrapper._qo_indptr_buf, before_qo)
    ck(still and changed, f"P6 {label}: the plan's int-workspace and indptr copies queue behind the running kernel "
       f"on torch's current stream (unchanged while it ran: {still}; updated after: {changed})")

# ---- P7
S = mk(16384)
NEW(S, 8, lens_for(8, random.Random(5)))
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
raised = ""
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    try:
        with torch.cuda.graph(g):
            try:
                NEW(S, 8, lens_for(8, random.Random(6)))
            except RuntimeError as exc:
                raised = str(exc)
            torch.cuda._sleep(10)
    except Exception as exc:  # noqa: BLE001
        raised = raised or f"capture failed: {exc!r}"
ck("called inside CUDA graph capture" in raised, f"P7 plan inside a capture raises production's RuntimeError")
PP.ST.enabled = False
A = mk(16384)
NEW(A, 8, lens_for(8, random.Random(7)))
ck(getattr(A, "_glm53_planpin_ring", None) is None and not A._qo_cpu.is_pinned(),
   "P7 module disabled: the call went to production's plan (pageable staging, no ring)")
PP.ST.enabled = True

print("summary:", PP.summary())
print(f"test_planpin: {'ALL OK' if not FAILS else 'FAILURES: ' + '; '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
