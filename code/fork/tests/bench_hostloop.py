"""[dec-hostloop] paired A/B of the decode host loop on nodeC: production path vs GLM53_DEC_HOSTLOOP.

One emulated decode step = production's order of operations on one GPU stream:
  host: execute_model(n)  = REAL vLLM MRv2 prologue (finish/add/update requests, gather_batch_req_state,
                            prepare_inputs: its H2D copies and triton kernels) -> REAL production
                            FlashInferMLASparseSM90Builder._kv_lens_host (patched or not) -> REAL flashinfer
                            BatchMLAPagedAttentionWrapper.plan through production's _SM90State (32 heads/rank, fp8 KV,
                            max_tokens 7168, top-k 2048) -> the other metadata builders, emulated as host busy time
                            (production: 4 GDN/KDA builds ~0.29 ms each = --gdn-ms) -> event F(n) -> target graph
        sample_tokens(n) = REAL post_update (postprocess_sampled, + the hostloop snapshot when on) ->
                            drafter graph -> event E(n)
  target / drafter = captured CUDA graphs of a spin kernel of --target-ms / --drafter-ms.
Metrics per step: gap = E(n-1) -> F(n) on the GPU (GPU idle between the drafter and the next forward, plus the
prologue's small kernels), step = F(n-1) -> F(n), and host time of _kv_lens_host and of plan().
Rounds alternate A (production: fast path off) and B (GLM53_DEC_HOSTLOOP fast path on, verification off), paired
and interleaved; the first 3 steps of every round are dropped. Contexts are > index_topk (steady long-context decode,
one request, adaptive K in {4,5,7}, random acceptance) like the prose profile.
Run under the shared lock: tests/hostloop_gpu.sh python3 tests/bench_hostloop.py
"""
from __future__ import annotations

import argparse
import importlib
import random
import statistics
import sys
import time
import types

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests")

import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402
from test_hostloop import make_runner, make_cam, sleep_calibrate  # noqa: E402


def graph_of_sleep(cycles: int):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cuda._sleep(cycles)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        torch.cuda._sleep(cycles)
    return g


def busy_cpu(ms: float) -> None:
    end = time.perf_counter() + ms / 1e3
    while time.perf_counter() < end:
        pass


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--drafter-ms", type=float, default=6.0)
    ap.add_argument("--gdn-ms", type=float, default=1.16)
    ap.add_argument("--reqs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meter", type=int, default=0, help="also drive glm53_hostloop's meter, report every N steps")
    args = ap.parse_args()

    MR = importlib.import_module(H.MR_MODULE)
    SM90 = importlib.import_module(H.SM90_MODULE)
    rep = H.install_now(MR, SM90, fast=True, meter_every=args.meter)
    assert rep["fast"], rep
    if args.meter:
        H._METER.every = 10**9                    # read per round below instead of the periodic log line
    meter = {"A": dict(gap=[], step=[], c0=[]), "B": dict(gap=[], step=[], c0=[])}
    H.ST.verify_first = 0
    H.ST.verify_every = 0
    dev = torch.device("cuda")
    rng = random.Random(args.seed)
    K = 7
    R = make_runner(MR, dev, 4, K, 7168, 16384)
    B = SM90.FlashInferMLASparseSM90Builder
    bself = types.SimpleNamespace(_async_scheduling=True, _index_topk=2048, _index_kpool=4)
    state = SM90._SM90State(dev, 32, torch.float8_e4m3fn, 7168, 2048, kv_lora_rank=512, qk_rope_head_dim=0,
                            sm_scale=1.0 / 16.0)
    cyc = sleep_calibrate()
    g_target = graph_of_sleep(int(args.target_ms * cyc))
    g_drafter = graph_of_sleep(int(args.drafter_ms * cyc))

    # requests: long prompts, already prefilled (decode only)
    reqs = {}
    new = []
    for i in range(args.reqs):
        rid = f"r{i}"
        prompt = rng.randint(3000, 9000)
        toks = [rng.randint(1, 1000) for _ in range(prompt)]
        reqs[rid] = dict(true_nc=prompt, sched_nc=prompt)
        new.append(types.SimpleNamespace(req_id=rid, prompt_token_ids=toks, prefill_token_ids=toks,
                                         num_computed_tokens=prompt, sampling_params=None, block_ids=([],),
                                         mm_features=[], lora_request=None, pooling_params=None))
    fix_queue = []
    first = [True]

    def sched():
        while len(fix_queue) > 1:
            for r, rej in fix_queue.pop(0).items():
                reqs[r]["sched_nc"] -= rej
        k = rng.choice((4, 5, 7))
        ids = list(reqs)
        cached = [] if first[0] else ids
        so = types.SimpleNamespace(
            scheduled_new_reqs=new if first[0] else [],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=cached,
                                                        num_computed_tokens=[reqs[r]["sched_nc"] for r in cached],
                                                        new_block_ids=[None] * len(cached)),
            num_scheduled_tokens={r: k + 1 for r in ids}, total_num_scheduled_tokens=(k + 1) * len(ids),
            scheduled_spec_decode_tokens={r: [-1] * k for r in ids}, finished_req_ids=set(), preempted_req_ids=set(),
            new_block_ids_to_zero=None, kv_cache_block_copies=None, has_structured_output_requests=False)
        first[0] = False
        for r in ids:
            reqs[r]["sched_nc"] += k + 1
        return so, k

    res = {"A": dict(gap=[], step=[], klh=[], plan=[]), "B": dict(gap=[], step=[], klh=[], plan=[])}
    prev_end = prev_fwd = None
    pending = []
    for rnd in range(2 * args.rounds):
        arm = "A" if rnd % 2 == 0 else "B"
        if rnd % 4 >= 2:                          # ABBA ordering
            arm = "B" if arm == "A" else "A"
        H.ST.enabled = arm == "B"
        for st in range(args.steps):
            so, k = sched()
            R.finish_requests(so)
            R.add_requests(so)
            R.update_requests(so)
            brs, _ = R.gather_batch_req_state(so, False)
            nreq = len(brs.req_ids)
            desc = types.SimpleNamespace(num_tokens=nreq * (k + 1), num_reqs=nreq)
            ib = R.prepare_inputs(so, brs, desc)
            cam = make_cam(ib, True)
            t0 = time.perf_counter()
            rows, lens = B._kv_lens_host(bself, cam)
            t1 = time.perf_counter()
            state.plan(rows, lens)
            t2 = time.perf_counter()
            busy_cpu(args.gdn_ms)
            H.ST.ctx = None
            if args.meter:
                H._METER.forward_start()          # what the StepTimingCollector.forward_start hook does
            fwd = torch.cuda.Event(enable_timing=True)
            fwd.record()
            if args.meter:                        # an eager "collective" of 50 us, timed by the meter as c0
                H._METER.coll(torch.cuda._sleep, int(0.05 * cyc))
            g_target.replay()
            # sample_tokens: rejection sampling result + post_update (+ snapshot) then the drafter
            ns, nrj, fix = [], [], {}
            for i, rid in enumerate(ib.req_ids):
                a = rng.randint(1, k + 1)
                ns.append(a)
                nrj.append(k + 1 - a)
                reqs[rid]["true_nc"] += a
                if k + 1 - a:
                    fix[rid] = k + 1 - a
            fix_queue.append(fix)
            sampled = torch.randint(1, 1000, (nreq, K + 1), device=dev, dtype=torch.int64)
            R.postprocess_sampled(ib.idx_mapping, sampled, torch.tensor(ns, dtype=torch.int32, device=dev),
                                  torch.tensor(nrj, dtype=torch.int32, device=dev), ib.query_start_loc)
            g_drafter.replay()
            if args.meter:
                H._METER.step_end()               # what the sample_tokens hook does
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if prev_end is not None and st >= 3:
                pending.append((arm, prev_end, fwd, prev_fwd, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
            prev_end, prev_fwd = end, fwd
        torch.cuda.synchronize()
        for arm_, e, f, pf, klh, pl in pending:
            res[arm_]["gap"].append(e.elapsed_time(f))
            res[arm_]["step"].append(pf.elapsed_time(f))
            res[arm_]["klh"].append(klh)
            res[arm_]["plan"].append(pl)
        pending.clear()
        prev_end = prev_fwd = None                 # do not pair across the round boundary sync
        if args.meter:                             # the meter's own samples of this round (same drop of 3)
            H._METER._drain()
            for gap, stp, _hw, colls in H._METER.samples[2:]:
                meter[arm]["gap"].append(gap)
                meter[arm]["step"].append(stp)
                meter[arm]["c0"].append(colls[0] if colls else float("nan"))
            H._METER.samples = []
            H._METER.pending = []
            H._METER.cur = None

    print(f"config: target {args.target_ms} ms, drafter {args.drafter_ms} ms, other builders {args.gdn_ms} ms host, "
          f"{args.reqs} request(s), {args.rounds} rounds x {args.steps} steps per arm (ABBA), stats {H.STATS}")
    for arm, name in (("A", "production (sync)"), ("B", "GLM53_DEC_HOSTLOOP")):
        r = res[arm]
        print("%-20s n=%d  gap ms med %.3f p10 %.3f p90 %.3f | step ms med %.3f p10 %.3f p90 %.3f | host "
              "_kv_lens_host med %.3f p90 %.3f | plan() med %.3f p90 %.3f" % (
                  name, len(r["gap"]), q(r["gap"], .5), q(r["gap"], .1), q(r["gap"], .9), q(r["step"], .5),
                  q(r["step"], .1), q(r["step"], .9), q(r["klh"], .5), q(r["klh"], .9), q(r["plan"], .5),
                  q(r["plan"], .9)))
    if args.meter:
        for arm in ("A", "B"):
            m = meter[arm]
            print("meter %s  n=%d  gap ms med %.3f p90 %.3f | step ms med %.3f | c0 (50 us eager op) med %.3f" % (
                arm, len(m["gap"]), q(m["gap"], .5), q(m["gap"], .9), q(m["step"], .5), q(m["c0"], .5)))
    d_gap = statistics.median(res["A"]["gap"]) - statistics.median(res["B"]["gap"])
    d_step = statistics.median(res["A"]["step"]) - statistics.median(res["B"]["step"])
    print(f"saving: gap {d_gap:.3f} ms/step, step {d_step:.3f} ms/step (median A - median B)")


if __name__ == "__main__":
    main()
