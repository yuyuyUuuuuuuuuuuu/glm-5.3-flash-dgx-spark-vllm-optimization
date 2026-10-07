"""[dec-hostloop review] GLM53_DEC_HOSTLOOP under a TRULY asynchronous host loop (no per-step host sync at all):
unlike tests/test_hostloop.py (which also runs production's syncing _kv_lens_host every step and so drains the stream
each step), here the host never waits except inside the fast path's own snapshot-event wait. Each step:
  prepare (real MRv2 methods) -> fast _kv_lens_host (host lens) -> D2D copy of the device seq_lens into a GPU history
  row (what production's .cpu() would have returned) -> "target" sleep -> sampling/post_update (+snapshot) ->
  "drafter" sleep.
At the end one sync, then every step's host lens must equal the lens computed (by production's own code) from the
device history row. Also checks the host never blocked for the drafter (max host time of the fast call).
Run: tests/hostloop_gpu.sh python3 tests/review_hostloop_async.py [--steps 600] [--seeds 20,21,22]
"""
import argparse
import sys
import time
import types

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests")
import random  # noqa: E402

import torch  # noqa: E402

import glm53_hostloop as H  # noqa: E402
from test_hostloop import Sim, make_cam, sleep_calibrate  # noqa: E402


def async_sample(sim, ib):
    """Sim.sample without its synchronizing pageable torch.tensor(..., device=cuda) copies."""
    real = torch.tensor
    def pinned(data, dtype=None, device=None):
        t = real(data, dtype=dtype).pin_memory()
        return t.to(device, non_blocking=True) if device is not None else t
    torch.tensor = pinned
    try:
        return sim.sample(ib)
    finally:
        torch.tensor = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", default="20,21,22")
    ap.add_argument("--target-ms", type=float, default=3.0)
    ap.add_argument("--drafter-ms", type=float, default=2.0)
    a = ap.parse_args()
    import importlib
    MR = importlib.import_module(H.MR_MODULE)
    SM90 = importlib.import_module(H.SM90_MODULE)
    rep = H.install_now(MR, SM90, fast=True, meter_every=0)
    assert rep["fast"], rep
    B = SM90.FlashInferMLASparseSM90Builder
    fast_fn = B._kv_lens_host
    orig_fn = fast_fn._glm53_orig
    bself = types.SimpleNamespace(_async_scheduling=True, _index_topk=2048, _index_kpool=4)
    cyc = sleep_calibrate()
    fails = 0
    for sd in [int(x) for x in a.seeds.split(",")]:
        H.ST.__init__()
        H.ST.enabled = True
        H.ST.verify_first = 0
        H.ST.verify_every = 0
        for k in H.STATS:
            H.STATS[k] = 0
        dev = torch.device("cuda")
        sim = Sim(MR, dev, random.Random(sd))
        R = sim.R
        hist = torch.zeros(a.steps, R.max_num_reqs, dtype=torch.int32, device=dev)
        recs = []
        host_t = []
        torch.cuda.synchronize()
        for step in range(a.steps):
            so, K, _ = sim.schedule()
            if so.total_num_scheduled_tokens == 0:
                continue
            R.finish_requests(so)
            R.add_requests(so)
            R.update_requests(so)
            brs, _ = R.gather_batch_req_state(so, False)
            desc, full = sim.batch_desc(so, brs)
            ib = R.prepare_inputs(so, brs, desc)
            cam = make_cam(ib, full)
            before = H.STATS["fast"]
            t0 = time.perf_counter()
            rows_f, lens_f = fast_fn(bself, cam)
            host_t.append((time.perf_counter() - t0) * 1e3)
            used = H.STATS["fast"] > before
            nr = int(cam.num_reqs)
            hist[step, :nr].copy_(cam.seq_lens[:nr], non_blocking=True)   # device truth, stream-ordered
            recs.append((step, nr, cam, rows_f, lens_f.clone(), used))
            H.ST.ctx = None
            torch.cuda._sleep(int(a.target_ms * cyc))                   # "target"
            sampled, ns, nrj = async_sample(sim, ib)                      # pinned, non_blocking: no host sync
            R.postprocess_sampled(ib.idx_mapping, sampled, ns, nrj, ib.query_start_loc)
            torch.cuda._sleep(int(a.drafter_ms * cyc))                  # "drafter", queued behind the snapshot
        torch.cuda.synchronize()
        hist_cpu = hist.cpu()
        bad = 0
        nfast = 0
        for step, nr, cam, rows_f, lens_f, used in recs:
            nfast += used
            view = H._CamView(cam, hist_cpu[step, :nr].clone())
            rows_o, lens_o = orig_fn(bself, view)                       # production's code on the device values
            if rows_o != rows_f or not torch.equal(lens_o, lens_f):
                bad += 1
                if bad <= 3:
                    print(f"  seed {sd} step {step}: MISMATCH fast={lens_f.tolist()[:6]} dev={lens_o.tolist()[:6]}")
        fails += bad
        ht = sorted(host_t)
        print(f"seed {sd}: {len(recs)} steps, fast {nfast}, mismatches {bad}, stats {dict(H.STATS)}; host time of the "
              f"fast call med {ht[len(ht) // 2]:.3f} ms p99 {ht[99 * len(ht) // 100]:.3f} max {ht[-1]:.3f} "
              f"(target {a.target_ms} + drafter {a.drafter_ms} ms per step)")
    print("RESULT:", "PASS" if fails == 0 else f"FAIL ({fails} mismatching steps)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
