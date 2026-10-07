"""Adversarial review 2 (mhcsp): bound NCCL reduce-scatter + all-gather vs all-reduce on nodeC's one GPU.

nodeC has one GB10 and nodeA/nodeB are out of bounds, so the RoCE wire cannot be measured here. What CAN be measured
is NCCL's own algorithm/pipeline overhead of the SP pair vs the all-reduce it replaces, on the NET transport code
path (proxy thread + host-staged buffers, the same mechanics as production's no-GDR RoCE ring), by making two ranks
on the same GPU look like two hosts: NCCL_HOSTID differs per rank (so NCCL's same-host duplicate-GPU check does not
fire and P2P/SHM are not selected), NCCL_IB_DISABLE=1, socket transport over loopback. The absolute bandwidth is the
loopback socket's, NOT RoCE's; only the RATIO (RS+AG)/AR at production shapes is the evidence.

Shapes: [T, 4096] bf16, T in (13824, 6912, 2048, 1024); AR of [T,H]; RS of [T,H] -> [T/2,H]; AG [T/2,H] -> [T,H].
NCCL_MIN/MAX_NCHANNELS=8 like production. Also checks bitwise: RS(x)+AG == AR(x) on both ranks.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc_sp/review2_nccl_loopback.py
"""
from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 4096
TS = (13824, 6912, 2048, 1024)
WARM, ITERS = 3, 9


def _worker(rank: int, port: int, q) -> None:
    os.environ.update({
        "NCCL_HOSTID": f"mhcsp-review-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_NCHANNELS": os.environ.get("NCH", "8"), "NCCL_MAX_NCHANNELS": os.environ.get("NCH", "8"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
    })
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            device_id=torch.device("cuda:0"))
    out = {}
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    for T in TS:
        x = (torch.randn((T, H), device="cuda", generator=g, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        half = torch.empty((T // 2, H), device="cuda", dtype=torch.bfloat16)
        full = torch.empty((T, H), device="cuda", dtype=torch.bfloat16)

        def ar():
            y = x.clone()
            dist.all_reduce(y)
            return y

        def rs():
            dist.reduce_scatter_tensor(half, x)
            return half

        def ag():
            dist.all_gather_into_tensor(full, half)
            return full

        res = {}
        for name, fn in (("ar", ar), ("rs", rs), ("ag", ag), ("rs+ag", lambda: (rs(), ag())[1])):
            ts = []
            for i in range(WARM + ITERS):
                dist.barrier()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                if i >= WARM:
                    ts.append(a.elapsed_time(b))
            res[name] = statistics.median(ts)
        # bitwise: the SP pair reproduces the all-reduce
        y = ar()
        dist.reduce_scatter_tensor(half, x)
        dist.all_gather_into_tensor(full, half)
        res["bitwise"] = bool(torch.equal(y, full))
        out[T] = res
    q.put((rank, out))
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, port, q)) for r in (0, 1)]
    for p in ps:
        p.start()
    got = {}
    import queue
    import time
    t0 = time.time()
    while len(got) < 2:
        try:
            r, o = q.get(timeout=10)
            got[r] = o
        except queue.Empty:
            dead = [p.exitcode for p in ps if p.exitcode not in (None, 0)]
            if dead or time.time() - t0 > 1200:
                for p in ps:
                    p.kill()
                print(f"FAIL: worker exit codes {[p.exitcode for p in ps]} (or 1200 s timeout)")
                return 2
    for p in ps:
        p.join(timeout=120)
    ok = True
    print(f"NCCL {torch.cuda.nccl.version()}  transport=NET/socket loopback (2 'hosts' on one GB10), "
          f"channels={os.environ.get('NCH', '8')}; times are ms, medians of {ITERS}")
    print(f"{'T':>6} {'MB(AR)':>7} {'AR':>8} {'RS':>8} {'AG':>8} {'RS+AG':>8} {'(RS+AG)/AR':>11} {'sumsep/AR':>10} bitwise")
    for T in TS:
        a, b = got[0][T], got[1][T]
        m = {k: max(a[k], b[k]) for k in ("ar", "rs", "ag", "rs+ag")}
        bw = a["bitwise"] and b["bitwise"]
        ok &= bw
        print(f"{T:>6} {T * H * 2 / 1e6:>7.1f} {m['ar']:>8.2f} {m['rs']:>8.2f} {m['ag']:>8.2f} {m['rs+ag']:>8.2f} "
              f"{m['rs+ag'] / m['ar']:>11.3f} {(m['rs'] + m['ag']) / m['ar']:>10.3f} {bw}")
    print("ALL OK" if ok else "FAIL (bitwise)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
