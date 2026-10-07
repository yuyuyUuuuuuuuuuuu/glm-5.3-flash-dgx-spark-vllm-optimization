"""r16z8p nodeC bench: production's _SM90State.plan (pageable staging) vs GLM53_MLA_PLAN_PIN (glm53_mla_planpin.py,
the shipped module installed through plugin_install) at production's max_tokens 16384, on the REAL MRv2 prologue +
REAL _kv_lens_host (GLM53_DEC_HOSTLOOP fast path ON, as production) + REAL flashinfer plan(); target / drafter =
CUDA-graph spin kernels (20.0 / 5.1 ms), other builders = host busy time. Port of the profile analysis' bench
($TF_EXL3_ASSETS/prof-1005/ana/nodeC/bench_plan_pin.py) with arm B = the shipped module instead of hand-
pinned tensors. Arms (ABC-interleaved rounds):
  A: max_tokens 16384, production plan (pageable)           = production today
  B: max_tokens 16384, GLM53_MLA_PLAN_PIN=1 (pinned 2-slot ring)
  C: max_tokens 7168, production plan (pageable)            (indptr 28676 B < 64 KiB: the no-stall reference)
Metrics: GPU gap drafter end E(n-1) -> forward start F(n), step F(n-1) -> F(n), host time of plan(); exactness: at
step 5 of every B round production's plan on the same rows -> device qo/kv indptr, kv_len_arr, plan_info and every
int-workspace byte the kernels read identical. Plus the host call time of the 65540 B indptr copy_ itself (pageable
vs the ring's pinned slot) while a drafter-size graph is queued.
Usage: tests/hostloop_gpu.sh python3 tests/r16z8p/bench_planpin.py
"""
import sys, time, random, types, statistics, importlib, argparse
sys.path.insert(0, "/w"); sys.path.insert(0, "/w/tests"); sys.path.insert(0, "/w/tests/r16z8p")
import torch
import os
import glm53_hostloop as H
from test_hostloop import make_runner, make_cam, sleep_calibrate
from bench_hostloop import graph_of_sleep, busy_cpu, q
ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--target-ms", type=float, default=20.0); ap.add_argument("--drafter-ms", type=float, default=5.1)
ap.add_argument("--gdn-ms", type=float, default=1.16); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
MR = importlib.import_module(H.MR_MODULE); SM90 = importlib.import_module(H.SM90_MODULE)
import glm53_mla_planpin as PP
from test_planpin_view import same_view, bufs_of
os.environ[PP.ENV] = "1"; PP.plugin_install()
NEW = SM90._SM90State.plan; assert getattr(NEW, "_glm53_planpin", False), PP.summary()
ORIG = NEW._glm53_orig
rep = H.install_now(MR, SM90, fast=True, meter_every=0); assert rep["fast"], rep
H.ST.verify_first = 0; H.ST.verify_every = 0; H.ST.enabled = True
dev = torch.device("cuda"); rng = random.Random(args.seed); K = 7
R = make_runner(MR, dev, 4, K, 7168, 16384)
B = SM90.FlashInferMLASparseSM90Builder
bself = types.SimpleNamespace(_async_scheduling=True, _index_topk=2048, _index_kpool=4)
def mk(mt):
    return SM90._SM90State(dev, 32, torch.float8_e4m3fn, mt, 2048, kv_lora_rank=512, qk_rope_head_dim=0, sm_scale=1.0 / 16.0)
states = {"A": mk(16384), "B": mk(16384), "C": mk(7168)}
planfn = {"A": ORIG, "B": NEW, "C": ORIG}
cyc = sleep_calibrate()
g_target = graph_of_sleep(int(args.target_ms * cyc)); g_drafter = graph_of_sleep(int(args.drafter_ms * cyc))
reqs = {}; new = []
prompt = rng.randint(3000, 9000); toks = [rng.randint(1, 1000) for _ in range(prompt)]
reqs["r0"] = dict(true_nc=prompt, sched_nc=prompt)
new.append(types.SimpleNamespace(req_id="r0", prompt_token_ids=toks, prefill_token_ids=toks, num_computed_tokens=prompt,
                                 sampling_params=None, block_ids=([],), mm_features=[], lora_request=None, pooling_params=None))
fix_queue = []; first = [True]
def sched():
    while len(fix_queue) > 1:
        for r, rej in fix_queue.pop(0).items(): reqs[r]["sched_nc"] -= rej
    k = rng.choice((4, 5, 7)); ids = list(reqs); cached = [] if first[0] else ids
    so = types.SimpleNamespace(scheduled_new_reqs=new if first[0] else [],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=cached, num_computed_tokens=[reqs[r]["sched_nc"] for r in cached], new_block_ids=[None] * len(cached)),
        num_scheduled_tokens={r: k + 1 for r in ids}, total_num_scheduled_tokens=(k + 1) * len(ids),
        scheduled_spec_decode_tokens={r: [-1] * k for r in ids}, finished_req_ids=set(), preempted_req_ids=set(),
        new_block_ids_to_zero=None, kv_cache_block_copies=None, has_structured_output_requests=False)
    first[0] = False
    for r in ids: reqs[r]["sched_nc"] += k + 1
    return so, k
res = {a: dict(gap=[], plan=[], step=[]) for a in states}
order = ["A", "B", "C"]; prev_end = prev_fwd = None; pending = []; mism = 0; checked = 0
for rnd in range(3 * args.rounds):
    arm = order[(rnd + rnd // 3) % 3]; state = states[arm]
    for st in range(args.steps):
        so, k = sched(); R.finish_requests(so); R.add_requests(so); R.update_requests(so)
        brs, _ = R.gather_batch_req_state(so, False); nreq = len(brs.req_ids)
        ib = R.prepare_inputs(so, brs, types.SimpleNamespace(num_tokens=nreq * (k + 1), num_reqs=nreq))
        cam = make_cam(ib, True)
        rows, lens = B._kv_lens_host(bself, cam)
        t1 = time.perf_counter(); planfn[arm](state, rows, lens); t2 = time.perf_counter()
        if arm == "B" and st == 5:   # exactness: production's plan on the same rows -> identical kernel-visible buffers
            ORIG(states["A"], rows, lens); torch.cuda.synchronize()
            checked += 1
            mism += int(not same_view(bufs_of(states["A"]), states["A"].wrapper._plan_info, bufs_of(states["B"]), states["B"].wrapper._plan_info))
        busy_cpu(args.gdn_ms)
        H.ST.ctx = None
        fwd = torch.cuda.Event(enable_timing=True); fwd.record(); g_target.replay()
        ns, nrj, fix = [], [], {}
        for i, rid in enumerate(ib.req_ids):
            a = rng.randint(1, k + 1); ns.append(a); nrj.append(k + 1 - a); reqs[rid]["true_nc"] += a
            if k + 1 - a: fix[rid] = k + 1 - a
        fix_queue.append(fix)
        sampled = torch.randint(1, 1000, (nreq, K + 1), device=dev, dtype=torch.int64)
        R.postprocess_sampled(ib.idx_mapping, sampled, torch.tensor(ns, dtype=torch.int32, device=dev), torch.tensor(nrj, dtype=torch.int32, device=dev), ib.query_start_loc)
        g_drafter.replay()
        end = torch.cuda.Event(enable_timing=True); end.record()
        if prev_end is not None and st >= 3: pending.append((arm, prev_end, fwd, prev_fwd, (t2 - t1) * 1e3))
        prev_end, prev_fwd = end, fwd
    torch.cuda.synchronize()
    for a, e, f, pf, pl in pending:
        res[a]["gap"].append(e.elapsed_time(f)); res[a]["step"].append(pf.elapsed_time(f)); res[a]["plan"].append(pl)
    pending.clear(); prev_end = prev_fwd = None
print(f"config: target {args.target_ms} ms, drafter {args.drafter_ms} ms, other builders {args.gdn_ms} ms host, HOSTLOOP fast path ON, {args.rounds} rounds x {args.steps} steps per arm")
lab = {"A": "max_tokens 16384, production (pageable)", "B": "max_tokens 16384, GLM53_MLA_PLAN_PIN=1", "C": "max_tokens 7168, production (pageable)"}
for a in order:
    r = res[a]
    print(f"  {a} {lab[a]:42s}: gap drafter->forward p50 {q(r['gap'], .5):.3f} p90 {q(r['gap'], .9):.3f} ms | step p50 {q(r['step'], .5):.3f} ms | plan() host p50 {q(r['plan'], .5):.3f} p90 {q(r['plan'], .9):.3f} ms  (n {len(r['gap'])})")
print(f"device plan buffers (qo/kv indptr, kv_len_arr, plan_info, kernel-read int workspace) A vs B: {checked - mism}/{checked} identical")
ga, gb = statistics.median(res["A"]["gap"]), statistics.median(res["B"]["gap"])
sa, sb = statistics.median(res["A"]["step"]), statistics.median(res["B"]["step"])
print(f"B - A: gap p50 {gb - ga:+.3f} ms, step p50 {sb - sa:+.3f} ms; plan() host p50 {statistics.median(res['B']['plan']):.3f} vs {statistics.median(res['A']['plan']):.3f} ms; ring waits {PP.ST.waits}")
# host call time of the indptr copy itself, a drafter-size graph queued (pageable production staging vs the ring's pinned slot)
ring = states["B"]._glm53_planpin_ring
src_pg = states["A"]._qo_cpu; src_pin = ring.slots[0].qo; dst = states["A"].wrapper._qo_indptr_buf
hc = {"pageable": [], "pinned": []}
for r in range(14):
    for nm, s_ in (("pageable", src_pg), ("pinned", src_pin)):
        torch.cuda.synchronize(); g_drafter.replay()
        t = time.perf_counter(); dst.copy_(s_, non_blocking=True); hc[nm].append((time.perf_counter() - t) * 1e3)
        torch.cuda.synchronize()
print(f"indptr copy_ host call ({src_pg.numel() * 4} B, drafter graph queued): pageable p50 {q(hc['pageable'], .5):.3f} ms, pinned p50 {q(hc['pinned'], .5):.4f} ms")
print(f"bench_planpin: {'OK' if mism == 0 and checked > 0 else 'FAILED'}")
