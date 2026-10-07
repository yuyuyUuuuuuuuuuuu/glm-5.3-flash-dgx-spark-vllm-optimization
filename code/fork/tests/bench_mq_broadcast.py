"""[dec-hostloop] bound the per-step input latency of rank 1 (item 2 of the stream).

Multi-node TP=2 with the mp executor (v1/executor/multiproc_executor.py): the head's EngineCore enqueues every
execute_model / sample_tokens call into ONE MessageQueue (distributed/device_communicators/shm_broadcast.py) with
n_reader=2, n_local_reader=1: rank 0 (same node) reads it from a shared-memory ring buffer (SpinCondition reader,
production GLM53_SPINWAIT_MS=16 -> busy_loop_s=0.016, then zmq poll), rank 1 (nodeB, headless) from a zmq XPUB/SUB
TCP socket (MessageQueue.recv: socket.poll then recv_multipart, no spinning). This bench runs the production image's
MessageQueue on nodeC with both readers local processes, rank 1's socket over TCP loopback (so the numbers are a LOWER
bound for rank 1: nodeA->nodeB adds the RoCE/TCP wire and the kernel network stack), and measures enqueue ->
dequeue-return latency for a real decode SchedulerOutput ("execute_model" tuple like collective_rpc), for
  back : execute_model + sample_tokens enqueued together every 70 ms (what the engine does every decode step);
  cold : a single execute_model every 70 ms.
Both readers are then past the 16 ms spin window, i.e. they wake from a blocking zmq poll (cold CPU).
Run (no GPU needed): docker run --rm --network none -v <repo>:/w -w /w --entrypoint python3 <image> tests/bench_mq_broadcast.py
"""
import multiprocessing as mp
import os
import pickle
import statistics
import sys
import time


def reader(handle, rank, out_q, n):
    from vllm.distributed.device_communicators import shm_broadcast as SB
    SB.SpinCondition.__init__.__defaults__ = (0.016,)          # production GLM53_SPINWAIT_MS=16
    q = SB.MessageQueue.create_from_handle(handle, rank)
    q.wait_until_ready()
    got = []
    for _ in range(n):
        obj = q.dequeue(indefinite=True)
        got.append((obj[-1], time.monotonic_ns()))
    out_q.put((rank, got))


def make_so():
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    so = SchedulerOutput.make_empty()
    rid = "chatcmpl-9f1e2d3c4b5a69788796a5b4c3d2e1f0-0"
    cr = CachedRequestData.make_empty()
    cr.req_ids = [rid]
    cr.resumed_req_ids = set()
    cr.new_token_ids = [[]]
    cr.all_token_ids = {}
    cr.new_block_ids = [None]
    cr.num_computed_tokens = [123456]
    cr.num_output_tokens = [789]
    so.scheduled_cached_reqs = cr
    so.num_scheduled_tokens = {rid: 8}
    so.total_num_scheduled_tokens = 8
    so.scheduled_spec_decode_tokens = {rid: [-1] * 7}
    so.num_common_prefix_blocks = [0, 0, 0, 0]
    so.num_spec_tokens_to_schedule = 7
    return so


def main():
    from vllm.distributed.device_communicators import shm_broadcast as SB
    n_back, n_cold = 200, 60
    n = 2 * n_back + n_cold
    wq = SB.MessageQueue(2, 1, connect_ip="127.0.0.1", max_chunk_bytes=1024 * 1024 * 10)
    h = wq.export_handle()
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=reader, args=(h, r, out_q, n)) for r in (0, 1)]
    for p in procs:
        p.start()
    wq.wait_until_ready()
    so = make_so()
    print("payload: pickled execute_model(SchedulerOutput) %d bytes" % len(pickle.dumps(("execute_model", (so,), {},
                                                                                           None))))
    sent = {}
    time.sleep(0.5)
    k = 0
    for i in range(n_back):                                   # execute_model + sample_tokens back to back, 70 ms apart
        for meth in ("execute_model", "sample_tokens"):
            t = time.monotonic_ns()
            sent[k] = ("back", t)
            wq.enqueue((meth, (so,) if meth == "execute_model" else (None,), {}, None, k))
            k += 1
        time.sleep(0.070)
    for i in range(n_cold):
        time.sleep(0.070)
        t = time.monotonic_ns()
        sent[k] = ("cold", t)
        wq.enqueue(("execute_model", (so,), {}, None, k))
        k += 1
    res = {}
    for _ in procs:
        r, got = out_q.get(timeout=120)
        res[r] = got
    for p in procs:
        p.join(timeout=10)
    for kind in ("back", "cold"):
        for r in (0, 1):
            lat = [(t - sent[i][1]) / 1e3 for i, t in res[r] if sent[i][0] == kind and i >= 40]   # skip warm-up
            lat.sort()
            print("%-4s rank %d (%s): enqueue->dequeue us med %7.1f p10 %7.1f p90 %7.1f max %7.1f (n=%d)" % (
                kind, r, "shm ring, local" if r == 0 else "zmq TCP (loopback)", statistics.median(lat),
                lat[len(lat) // 10], lat[9 * len(lat) // 10], lat[-1], len(lat)))
        d = [(t1 - t0) / 1e3 for (i0, t0), (i1, t1) in zip(res[0], res[1])
             if sent[i0][0] == kind and i0 == i1 and i0 >= 40]
        d.sort()
        print("     rank1 - rank0 arrival us: med %7.1f p90 %7.1f" % (statistics.median(d), d[9 * len(d) // 10]))


if __name__ == "__main__":
    sys.path.insert(0, "/w")
    main()
