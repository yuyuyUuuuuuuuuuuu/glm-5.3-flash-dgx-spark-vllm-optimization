"""r16z8p adversarial review checks for glm53_mla_planpin.py (nodeC, production image + production mounts):

  V1  self-test-failure fallback AFTER the ring ran: production's plan must get production's PAGEABLE staging and the
      wrapper's own int workspace back (else it runs the unprotected naive-pinning race). Back-to-back stress at
      max_tokens 16384 after an injected self-test failure: the fixed module 0/24 wrong snapshots, staging pageable,
      int workspace restored; the reviewed module be04148 (tests/r16z8p/_old_planpin_be04148.py) shown for contrast.
  V2  CUDA-graph replay as the reader: a captured graph copies the fixed device plan buffers (what captured attention
      kernels read) into snapshot buffers; plan -> replay -> spin, back to back with the GPU behind the host, ring vs
      the synchronous reference, max_tokens 16384 and 7168, mixed decode / prefill / full sizes, 2 plans per "step"
  V3  no per-step leak: 60000 plans through the ring (decode sizes, no host sync): host RSS, CUDA allocated bytes and
      the pinned host allocator stats at 5000 and at 60000 plans
Usage: tests/hostloop_gpu.sh python3 tests/r16z8p/test_planpin_adv.py
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import random
import sys

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/r16z8p")
import torch  # noqa: E402

FAILS: list[str] = []


def ck(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


SM90 = importlib.import_module("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90")
import glm53_mla_planpin as PP  # noqa: E402
from test_planpin_view import same_view, bufs_of  # noqa: E402

spec = importlib.util.spec_from_file_location("old_planpin", "/w/tests/r16z8p/_old_planpin_be04148.py")
OLDPP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLDPP)

dev = torch.device("cuda")
ORIG = SM90._SM90State.plan
TOPK = 2048
fp = PP.source_fingerprint(ORIG)
ck(fp in PP.VERIFIED["_SM90State.plan"], f"V0 production _SM90State.plan fingerprint {fp} verified")


def mk(mt):
    s = SM90._SM90State(dev, 32, torch.float8_e4m3fn, mt, TOPK, kv_lora_rank=512, qk_rope_head_dim=0,
                        sm_scale=1.0 / 16.0)
    s.wrapper._int_workspace_buffer.zero_()
    s.wrapper._pin_memory_int_workspace_buffer.zero_()
    return s


def calib():
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(10_000_000)
    e.record()
    e.synchronize()
    return 10_000_000 / s.elapsed_time(e)


CYC = calib()


def sizes_for(mt, rng, n):
    base = [8, 1, 6, 4096, 8, mt, 5, mt - 1, 8, 1000, 6, 8, 40, mt // 2, 8, 2]
    out = [x for x in base if 1 <= x <= mt]
    while len(out) < n:
        out.append(rng.choice([5, 6, 8, rng.randint(1, mt)]))
    return out[:n]


def reference(mt, sizes, lens_l):
    R = mk(mt)
    ref, infos = [], []
    for n, lens in zip(sizes, lens_l):
        ORIG(R, n, lens)
        torch.cuda.synchronize()
        ref.append([b.clone() for b in bufs_of(R)])
        infos.append(list(R.wrapper._plan_info))
    return ref, infos


def stress_on(S, fn, mt, seed, n=24, spin_ms=3.0):
    rng = random.Random(seed)
    sizes = sizes_for(mt, rng, n)
    lens_l = [torch.tensor([rng.randint(1, TOPK) for _ in range(k)], dtype=torch.int32) for k in sizes]
    ref, infos = reference(mt, sizes, lens_l)
    snaps = [[torch.empty_like(b) for b in bufs_of(S)] for _ in sizes]
    plinfo = []
    torch.cuda.synchronize()
    torch.cuda._sleep(int(20 * CYC))
    for i, (k, lens) in enumerate(zip(sizes, lens_l)):
        fn(S, k, lens)
        plinfo.append(list(S.wrapper._plan_info))
        for d, b in zip(snaps[i], bufs_of(S)):
            d.copy_(b, non_blocking=True)
        torch.cuda._sleep(int(spin_ms * CYC))
    torch.cuda.synchronize()
    return sum(int(not same_view(snaps[i], plinfo[i], ref[i], infos[i])) for i in range(n))


def inject_selftest_failure(mod, fn, S, mt):
    """One plan through fn with the module's self-test forced to fail (torch.equal -> False for that call)."""
    mod.ST.selftest = None
    real = torch.equal
    torch.equal = lambda a, b: False
    try:
        fn(S, 8, torch.full((8,), 100, dtype=torch.int32))
    finally:
        torch.equal = real
    torch.cuda.synchronize()


# ---- V1
for label, mod in (("fixed module", PP), ("reviewed be04148", OLDPP)):
    mod.ST.enabled = True
    mod.ST.selftest = True          # the boot self-test passed
    fn = mod.make_plan(ORIG)
    S = mk(16384)
    ws_own = S.wrapper._pin_memory_int_workspace_buffer
    for k in (8, 6, 8, 4096, 8):    # the ring serves for a while (both slots used)
        fn(S, k, torch.full((k,), 300, dtype=torch.int32))
    torch.cuda.synchronize()
    inject_selftest_failure(mod, fn, S, 16384)
    disabled = not mod.ST.enabled
    bad = stress_on(S, fn, 16384, seed=11)
    staged_pinned = bool(S._qo_cpu.is_pinned())
    ws_back = S.wrapper._pin_memory_int_workspace_buffer.data_ptr() == ws_own.data_ptr()
    msg = (f"V1 {label}: self-test failure after the ring served -> module disabled {disabled}; back-to-back stress "
           f"at 16384 through the fallback: {bad}/24 wrong snapshots; staging pinned {staged_pinned}; wrapper's own int "
           f"workspace back {ws_back}")
    if mod is PP:
        ck(disabled and bad == 0 and not staged_pinned and ws_back, msg)
    else:
        print("     info (defect being fixed): " + msg, flush=True)
    mod.ST.enabled = True
    mod.ST.selftest = True

# ---- V2: CUDA-graph replay reader, 2 plans per step, mixed sizes
NEW = PP.make_plan(ORIG)
PP.ST.enabled = True
PP.ST.selftest = True
for mt in (16384, 7168):
    rng = random.Random(mt + 1)
    nsteps = 16
    sizes = sizes_for(mt, rng, 2 * nsteps)
    lens_l = [torch.tensor([rng.randint(1, TOPK) for _ in range(k)], dtype=torch.int32) for k in sizes]
    ref, infos = reference(mt, sizes, lens_l)
    for variant, fn in (("ring", NEW), ("production", ORIG)):
        S = mk(mt)
        fn(S, 8, torch.full((8,), 50, dtype=torch.int32))
        torch.cuda.synchronize()
        fixed = [torch.empty_like(b) for b in bufs_of(S)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            with torch.cuda.graph(g, stream=side):
                for d, b in zip(fixed, bufs_of(S)):
                    d.copy_(b)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        out = [[torch.empty_like(b) for b in bufs_of(S)] for _ in sizes]
        plinfo = []
        w0 = PP.ST.waits
        torch.cuda._sleep(int(20 * CYC))
        for i, (k, lens) in enumerate(zip(sizes, lens_l)):
            fn(S, k, lens)                         # draft build / verify build: 2 plans per step
            plinfo.append(list(S.wrapper._plan_info))
            g.replay()                             # the captured kernels read the fixed device plan buffers
            for d, b in zip(out[i], fixed):
                d.copy_(b, non_blocking=True)
            if i % 2 == 1:
                torch.cuda._sleep(int(2.5 * CYC))  # the step's target + drafter
        torch.cuda.synchronize()
        bad = sum(int(not same_view(out[i], plinfo[i], ref[i], infos[i])) for i in range(len(sizes)))
        msg = (f"V2 max_tokens {mt} {variant}: CUDA-graph replay after each of {len(sizes)} back-to-back plans "
               f"(2 per step, mixed decode/prefill/full sizes) read the reference plan state: {bad} wrong"
               + (f"; ring waits {PP.ST.waits - w0}" if variant == "ring" else ""))
        if variant == "ring" or mt == 16384:
            ck(bad == 0, msg)
        else:
            print("     info (production's latent race below 64 KiB): " + msg, flush=True)

# ---- V3: leak check
def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


def host_stats():
    try:
        st = torch.cuda.host_memory_stats()
        return int(st.get("allocated_bytes.current", st.get("allocated_bytes", {}).get("current", -1))
                   if isinstance(st, dict) else -1), int(st.get("num_host_alloc", -1)) if isinstance(st, dict) else -1
    except Exception:  # noqa: BLE001
        return -1, -1


S = mk(16384)
lens_d = [torch.tensor([random.randint(1, TOPK) for _ in range(k)], dtype=torch.int32) for k in (5, 6, 8)]
marks = {}
w0 = PP.ST.waits
for i in range(60000):
    NEW(S, (5, 6, 8)[i % 3], lens_d[i % 3])
    if i % 512 == 511:
        torch.cuda._sleep(int(0.2 * CYC))
    if i + 1 in (5000, 60000):
        torch.cuda.synchronize()
        marks[i + 1] = (rss_kb(), torch.cuda.memory_allocated(), host_stats())
r5, r60 = marks[5000], marks[60000]
print(f"     V3 at 5000 plans: rss {r5[0]} kB, cuda allocated {r5[1]}, host stats {r5[2]}; at 60000: rss {r60[0]} kB, "
      f"cuda allocated {r60[1]}, host stats {r60[2]}; ring waits {PP.ST.waits - w0}", flush=True)
ck(r60[1] == r5[1] and r60[2] == r5[2] and r60[0] - r5[0] < 8192,
   f"V3 no per-step growth over 55000 plans: rss {r60[0] - r5[0]:+d} kB, cuda allocated {r60[1] - r5[1]:+d} B, pinned "
   f"host allocator unchanged {r60[2] == r5[2]}")

print(f"test_planpin_adv: {'ALL OK' if not FAILS else 'FAILURES: ' + '; '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
