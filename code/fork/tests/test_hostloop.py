"""[dec-hostloop] GLM53_DEC_HOSTLOOP correctness on nodeC (production image + production-mounted sm90/flashinfer).

Drives the REAL vLLM model-runner-V2 methods that decide seq_lens (GPUModelRunner.finish_requests / add_requests /
update_requests / gather_batch_req_state / prepare_inputs / postprocess_sampled, RequestState, InputBuffers, the
triton kernels prepare_pos_seq_lens / post_update / combine_sampled_and_draft_tokens) on a runner object whose
unrelated collaborators are stubs, through a randomized async-spec-decode schedule: adaptive K in {4,5,7}, random
acceptance, chunked prefills mixed with decodes, request finish / preemption / re-add with slot reuse, FULL-graph
request padding, contexts across index_topk=2048 with index_kpool=4. The scheduler's num_computed_tokens is
optimistic exactly like async scheduling (it learns step n's rejections only before scheduling step n+2).

For every step it checks, with glm53_hostloop installed (fast path, verification OFF so nothing syncs):
  A. the host seq_lens the fast path builds == input_batch.seq_lens on the device (the value production copies);
  B. production's _kv_lens_host on the fast path == production's _kv_lens_host on the original sync path
     (num_rows and every per-row length, torch.equal);
  C. the fast path does not wait for GPU work queued AFTER the snapshot (a 30 ms "drafter" is queued behind the
     snapshot event; host time of the fast _kv_lens_host << 30 ms, the original's >= the remaining drafter time).
Negative controls: (N1) dropping the fresh-slot bookkeeping makes the built seq_lens wrong and verification switches
the fast path off; (N2) metadata that is not this step's (other buffer) takes production's path; (N3) no snapshot ->
production's path. Plus the source fingerprints (T0).
Run: tests/hostloop_gpu.sh python3 tests/test_hostloop.py [--steps 400] [--seed 0]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import types

sys.path.insert(0, "/w")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg, flush=True)


class Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        return lambda *a, **k: None


def make_runner(MR, dev, max_reqs, K, max_tokens, max_len):
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.states import RequestState
    R = object.__new__(MR.GPUModelRunner)
    R.device = dev
    R.max_num_reqs = max_reqs
    R.req_states = RequestState(max_reqs, max_len, max_tokens, K, 1024, dev)
    R.adaptive_verification = None
    R.pooling_runner = None
    R.encoder_cache = None
    R.pp_handler = None
    R.prompt_logprobs_worker = None
    R.lora_state = Stub()
    R.model_state = Stub(num_new_sampled_tokens_per_step=1)
    R.block_tables = Stub()
    R.is_last_pp_rank = False
    R.sampler = None
    R.kv_block_zeroer = None
    R.input_buffers = InputBuffers(max_reqs, max_tokens, dev)
    R.decode_query_len = K + 1
    R.use_dcp = False
    R.use_pp = False
    R.model_config = types.SimpleNamespace(rswa_window=None)
    R.pcp_manager = None
    return R


class Sim:
    """Scheduler + sampler emulation around the real runner methods."""

    def __init__(self, MR, dev, rng, max_reqs=6, max_tokens=384, max_len=8192, K=7):
        self.MR, self.dev, self.rng = MR, dev, rng
        self.K, self.max_reqs, self.max_tokens = K, max_reqs, max_tokens
        self.R = make_runner(MR, dev, max_reqs, K, max_tokens, max_len)
        self.reqs = {}       # req_id -> dict(prompt, true_nc, sched_nc, running)
        self.next_id = 0
        self.fix_queue = []    # per step {req_id: rejected}; async scheduling learns step s's rejections before s+2

    def _new_req(self, resume=None):
        rid = resume or f"r{self.next_id}"
        if resume is None:
            self.next_id += 1
            prompt = self.rng.randint(8, 3000)
            nc0 = 0 if self.rng.random() < 0.7 else self.rng.randint(0, prompt - 1)   # prefix-cache hit
        else:
            prompt = self.reqs[rid]["prompt"]
            nc0 = self.rng.randint(0, min(self.reqs[rid]["true_nc"], prompt - 1))    # resume after preemption
        toks = [self.rng.randint(1, 1000) for _ in range(prompt)]
        for fx in self.fix_queue:
            fx.pop(rid, None)
        self.reqs[rid] = dict(prompt=prompt, true_nc=nc0, sched_nc=nc0, running=True)
        nr = types.SimpleNamespace(req_id=rid, prompt_token_ids=toks, prefill_token_ids=toks, num_computed_tokens=nc0,
                                   sampling_params=None, block_ids=([],), mm_features=[], lora_request=None,
                                   pooling_params=None)
        return nr

    def schedule(self):
        rng = self.rng
        finished, preempted, new = set(), set(), []
        running = [r for r, s in self.reqs.items() if s["running"]]
        for r in running:
            u = rng.random()
            if u < 0.03:
                finished.add(r)
                self.reqs[r]["running"] = False
                self.reqs[r]["gone"] = True
            elif u < 0.05:
                preempted.add(r)
                self.reqs[r]["running"] = False
        running = [r for r in running if self.reqs[r]["running"]]
        paused = [r for r, s in self.reqs.items() if not s["running"] and not s.get("gone") and r not in preempted]
        slots = self.max_reqs - len(running)
        if paused and slots > 0 and rng.random() < 0.3:
            new.append(self._new_req(resume=rng.choice(paused)))
            slots -= 1
        while slots > 0 and (len(running) + len(new) == 0 or rng.random() < 0.15):
            new.append(self._new_req())
            slots -= 1
        # scheduler's (optimistic) num_computed: exact after step n-1, all of step n assumed accepted
        while len(self.fix_queue) > 1:            # output(s-1) processed before scheduling s+1
            for r, rej in self.fix_queue.pop(0).items():
                if r in self.reqs and self.reqs[r]["running"]:
                    self.reqs[r]["sched_nc"] -= rej
        K = rng.choice((4, 5, 7))
        num_sched, spec, budget = {}, {}, self.max_tokens
        new_ids = {nr.req_id for nr in new}
        order = running + [nr.req_id for nr in new]
        for r in order:
            s = self.reqs[r]
            if s["sched_nc"] < s["prompt"]:                 # prefill (chunked)
                q = min(s["prompt"] - s["sched_nc"], max(1, budget // 2), 256)
                if q <= 0:
                    continue
                num_sched[r] = q
            else:
                if budget < K + 1:
                    continue
                num_sched[r] = K + 1
                spec[r] = [-1] * K
            budget -= num_sched[r]
        cached = [r for r in running if r in num_sched]
        so = types.SimpleNamespace(
            scheduled_new_reqs=new,
            scheduled_cached_reqs=types.SimpleNamespace(
                req_ids=cached, num_computed_tokens=[self.reqs[r]["sched_nc"] for r in cached],
                new_block_ids=[None] * len(cached)),
            num_scheduled_tokens=num_sched, total_num_scheduled_tokens=sum(num_sched.values()),
            scheduled_spec_decode_tokens=spec, finished_req_ids=finished, preempted_req_ids=preempted,
            new_block_ids_to_zero=None, kv_cache_block_copies=None, has_structured_output_requests=False)
        for r in num_sched:                              # scheduler advances optimistically
            self.reqs[r]["sched_nc"] += num_sched[r]
        return so, K, new_ids

    def batch_desc(self, so, brs):
        n = len(brs.req_ids)
        q = brs.num_scheduled_tokens
        uniform = (not brs.has_prefill) and bool(np.all(q == q[0])) and bool(so.scheduled_spec_decode_tokens)
        if uniform and self.rng.random() < 0.8:
            nr = min(self.max_reqs, n + self.rng.randint(0, 2))     # FULL graph: padded requests
            return types.SimpleNamespace(num_tokens=nr * int(q[0]), num_reqs=nr), True
        return types.SimpleNamespace(num_tokens=min(self.max_tokens, brs.num_tokens + self.rng.randint(0, 3)),
                                     num_reqs=None), False

    def sample(self, ib):
        n = ib.num_reqs
        qsl = ib.query_start_loc_np
        ns, nrj, fix = [], [], {}
        for i, rid in enumerate(ib.req_ids):
            q = int(qsl[i + 1] - qsl[i])
            s = self.reqs[rid]
            if ib.num_draft_tokens_per_req is not None and ib.num_draft_tokens_per_req[i] > 0:
                a = self.rng.randint(1, q)                  # sampled = accepted drafts + bonus
                ns.append(a)
                nrj.append(q - a)
            else:
                done = s["true_nc"] + q >= s["prompt"]
                ns.append(1 if done else 0)
                nrj.append(0)
            s["true_nc"] += q - nrj[-1]
            if nrj[-1]:
                fix[rid] = nrj[-1]
        self.fix_queue.append(fix)
        dev = self.dev
        sampled = torch.randint(1, 1000, (n, self.K + 1), device=dev, dtype=torch.int64)
        return (sampled, torch.tensor(ns, dtype=torch.int32, device=dev),
                torch.tensor(nrj, dtype=torch.int32, device=dev))


def make_cam(ib, full):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    nr = ib.num_reqs_after_padding if full else ib.num_reqs
    qsl_cpu = torch.from_numpy(ib.query_start_loc_np)
    return CommonAttentionMetadata(
        query_start_loc=ib.query_start_loc, query_start_loc_cpu=qsl_cpu, seq_lens=ib.seq_lens[:nr], num_reqs=nr,
        num_actual_tokens=ib.num_tokens, max_query_len=int(ib.num_scheduled_tokens.max()), max_seq_len=8192,
        block_table_tensor=torch.zeros(1, device=ib.seq_lens.device), slot_mapping=torch.zeros(1),
        seq_lens_cpu_upper_bound=ib.seq_lens_cpu_upper_bound[:nr])


def run_sim(MR, SM90, steps, seed, *, drop_fresh=False, verify=0, label="sim"):
    dev = torch.device("cuda")
    rng = random.Random(seed)
    H.ST.__init__()
    H.ST.enabled = True
    H.ST.verify_first = verify
    H.ST.verify_every = 0
    for k in H.STATS:
        H.STATS[k] = 0
    sim = Sim(MR, dev, rng)
    R = sim.R
    B = SM90.FlashInferMLASparseSM90Builder
    fast_fn = B._kv_lens_host
    orig_fn = fast_fn._glm53_orig
    bself = types.SimpleNamespace(_async_scheduling=True, _index_topk=2048, _index_kpool=4)
    if drop_fresh:
        H.ST.fresh = type("NoAdd", (set,), {"add": lambda self, x: None})()
    n_fast = n_checked = 0
    t_fast, t_orig = [], []
    drafter_ms = 30.0
    cyc_per_ms = sleep_calibrate()
    for step in range(steps):
        so, K, _ = sim.schedule()
        if so.total_num_scheduled_tokens == 0:
            continue
        # --- execute_model(n+1) prologue, as GPUModelRunner.execute_model does it
        R.finish_requests(so)
        R.add_requests(so)
        R.update_requests(so)
        brs, _ = R.gather_batch_req_state(so, False)
        desc, full = sim.batch_desc(so, brs)
        ib = R.prepare_inputs(so, brs, desc)
        cam = make_cam(ib, full)
        # --- the builder call (fast path; the drafter of the previous step may still be running)
        before = H.STATS["fast"]
        t0 = time.perf_counter()
        rows_f, lens_f = fast_fn(bself, cam)
        t1 = time.perf_counter()
        used_fast = H.STATS["fast"] > before
        n_fast += used_fast
        if used_fast and step > 0:
            t_fast.append((t1 - t0) * 1e3)
        # --- reference: production's own sync path on the same metadata (timed first: it drains the stream)
        t2 = time.perf_counter()
        rows_o, lens_o = orig_fn(bself, cam)
        if step > 0:
            t_orig.append((time.perf_counter() - t2) * 1e3)
        dev_seq = cam.seq_lens.cpu()
        if not drop_fresh and H.ST.ctx is not None and H.ST.enabled:
            host_seq = H.exact_seq_lens(H.ST.ctx, cam)
            check(host_seq is not None and torch.equal(host_seq, dev_seq),
                  f"{label} step {step}: host seq_lens {None if host_seq is None else host_seq.tolist()} != device "
                  f"{dev_seq.tolist()}")
        n_checked += 1
        if not drop_fresh:
            check(rows_f == rows_o and torch.equal(lens_f, lens_o),
                  f"{label} step {step}: lens differ fast={rows_f},{lens_f.tolist()[:8]} orig={rows_o},"
                  f"{lens_o.tolist()[:8]}")
        H.ST.ctx = None                                      # execute_model returned
        # --- sample_tokens(n+1): sampling + post_update + snapshot, then the drafter (queued after the snapshot)
        sampled, ns, nrj = sim.sample(ib)
        R.postprocess_sampled(ib.idx_mapping, sampled, ns, nrj, ib.query_start_loc)
        torch.cuda._sleep(int(drafter_ms * cyc_per_ms))     # "drafter": queued behind the snapshot event
    torch.cuda.synchronize()
    return dict(n_fast=n_fast, n_checked=n_checked, t_fast=t_fast, t_orig=t_orig, stats=dict(H.STATS),
                enabled=H.ST.enabled, reason=H.ST.disabled_reason)


def sleep_calibrate():
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(10_000_000)
    e.record()
    e.synchronize()
    return 10_000_000 / s.elapsed_time(e)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import importlib
    MR = importlib.import_module(H.MR_MODULE)
    SM90 = importlib.import_module(H.SM90_MODULE)

    # T0: fingerprints of the production functions
    ok, why, fps = H._verify_sources(MR, SM90)
    for k, v in sorted(fps.items()):
        print("T0 fingerprint %-46s %s" % (k, v))
    check(ok, f"T0 fingerprints: {why}")

    rep = H.install_now(MR, SM90, fast=True, meter_every=0)
    check(rep["fast"], f"install_now: {rep}")
    check(getattr(SM90.FlashInferMLASparseSM90Builder._kv_lens_host, "_glm53_hostloop", False), "builder not patched")

    # T1: randomized schedule, several seeds
    tot_fast = tot = 0
    tf_all, to_all = [], []
    for sd in range(args.seed, args.seed + 3):
        r = run_sim(MR, SM90, args.steps, sd, label=f"seed{sd}")
        tot_fast += r["n_fast"]
        tot += r["n_checked"]
        tf_all += r["t_fast"]
        to_all += r["t_orig"]
        print(f"T1 seed {sd}: {r['n_checked']} steps checked, fast path {r['n_fast']}, stats {r['stats']}")
        check(r["enabled"], f"T1 seed {sd}: fast path switched off: {r['reason']}")
        check(r["stats"]["fallback_mismatch_ctx"] == 0 and r["stats"]["fallback_bound"] == 0,
              f"T1 seed {sd}: unexpected fallbacks {r['stats']}")
    check(tot_fast >= 0.95 * tot, f"T1: fast path used {tot_fast}/{tot}")
    print("T1 host time of the fast _kv_lens_host with a 30 ms drafter queued behind the snapshot: "
          "med %.3f ms p90 %.3f max %.3f (n=%d)" % (q(tf_all, .5), q(tf_all, .9), q(tf_all, 1), len(tf_all)))
    print("T1 host time of production's _kv_lens_host on the same metadata (waits for the drafter): "
          "med %.1f ms p10 %.1f (n=%d)" % (q(to_all, .5), q(to_all, .1), len(to_all)))
    check(q(tf_all, .9) < 5.0, "T1: fast path waited for the drafter")

    # N1: mutation — forget the fresh-slot bookkeeping -> wrong host seq_lens must be caught by verification
    r = run_sim(MR, SM90, 200, 11, drop_fresh=True, verify=10**9, label="N1")
    print(f"N1 drop fresh bookkeeping: stats {r['stats']} enabled={r['enabled']} reason={str(r['reason'])[:120]}")
    check(r["stats"]["mismatch"] >= 1 and not r["enabled"], "N1: mutation not caught by verification")

    # N2 / N3: metadata of another buffer, missing snapshot -> production path
    H.ST.__init__()
    H.ST.enabled = True
    H.ST.verify_first = 0
    for k in H.STATS:
        H.STATS[k] = 0
    ctx = H._Ctx(seq_lens_ptr=12345, num_reqs=1, idx_np=np.array([0]), qsl_np=np.array([0, 5]),
                 nc_upper=np.array([100]), fresh=np.array([False]))
    cam = types.SimpleNamespace(num_reqs=1, seq_lens=torch.tensor([105], dtype=torch.int32, device="cuda"),
                                query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32))
    check(H.exact_seq_lens(ctx, cam) is None and H.STATS["fallback_mismatch_ctx"] == 1, "N2: foreign metadata used")
    ctx.seq_lens_ptr = int(cam.seq_lens.data_ptr())
    H.ST.snap_valid = False
    check(H.exact_seq_lens(ctx, cam) is None and H.STATS["fallback_no_snapshot"] == 1, "N3: missing snapshot used")
    ctx.fresh = np.array([True])
    got = H.exact_seq_lens(ctx, cam)
    check(got is not None and got.tolist() == [105], f"N3b: fresh-only batch needs no snapshot, got {got}")
    print("N2/N3 stats", H.STATS)

    print("RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
