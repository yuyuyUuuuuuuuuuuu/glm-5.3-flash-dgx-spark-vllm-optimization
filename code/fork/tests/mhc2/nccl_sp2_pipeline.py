"""mhc2: the GLM53_MHC_SP2 pipeline with REAL NCCL on real CUDA streams (nodeC, one GB10, two processes).

nodeC has one GPU and nodeA/nodeB are out of bounds, so - like tests/mhc_sp/review2_nccl_loopback.py - two ranks are
made to look like two hosts (different NCCL_HOSTID, IB/P2P/SHM off, socket transport over loopback). The collectives
are vLLM's own PyNcclCommunicator calls (the ones patch_mhc_sp2.py's helpers make), on a side stream, and the helper
code is exec'd verbatim out of the SP2-patched model.py.

A stack of L layers x 2 phases, per phase at the production shard shapes (T = 13,824, H = 4,096, 4 streams):
    serial  (r16x SP): RS(partial) -> production mhc_fused_post_pre_tilelang on the contiguous half -> AG
    pipelined (SP2)  : _sp2_reduce_scatter_begin -> _sp2_fused_post_pre_gather (k=2 interleaved sub-chunks)
followed by a cross-token stand-in for attention/MLP (fp32 cumulative sum over the tokens -> a rank partial), so a
wrong token order or a missing wait is a wrong answer.

Checks / measurements:
 A. correctness, both ranks computing: after L layers the gathered layer input of every phase (and the final one) is
    BITWISE equal between serial and pipelined on both ranks.
 B. timing, rank 1 as an always-ready peer (its mHC replaced by nothing, so the shared GPU runs one rank's kernels):
    ms per phase serial vs pipelined; plus the contention probe: one mHC sub-chunk alone vs with an NCCL RS of the
    other half in flight on the side stream.
 The loopback socket is ~9x slower than production's RoCE pair (review2: AR 113 MB = 48.6 ms vs 5.4 ms), so B's
 absolute numbers are NOT production's; the production gain is modelled from B's contention factor in docs/MHC_SP2.md.

Run: GPU_RUN_SHM=2g flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc2/nccl_sp2_pipeline.py
"""
from __future__ import annotations

import os
import statistics
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
T = int(os.environ.get("SP2_T", "13824"))
H, NS = 4096, 4
L = int(os.environ.get("SP2_LAYERS", "4"))
ITERS = int(os.environ.get("SP2_ITERS", "5"))


def _patched_model_src() -> str:
    import shutil
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, "overlay"))
    import patch_mhc_sp as P
    import patch_mhc_sp2 as P2
    d = tempfile.mkdtemp(prefix="sp2nccl-")
    mp_ = os.path.join(d, "model.py")
    shutil.copy(os.path.join(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"),
                             "vllm/models/glm5next/nvidia/model.py"), mp_)
    src = P.prepare(open(mp_).read(), P.MODEL_HUNKS)
    src = P2.prepare(src)
    shutil.rmtree(d, ignore_errors=True)
    return src


def _helpers(src: str, comm, rank: int):
    a = src.index("MHC_SP2_K = 2")
    b = src.index("from vllm.multimodal import MULTIMODAL_REGISTRY")
    from vllm.platforms import current_platform
    import logging
    ns = {"torch": torch, "MHC_SP_ACTIVE": True, "current_platform": current_platform,
          "logger": logging.getLogger("sp2"), "sp_shard": None, "sp_all_gather": None}
    exec(compile(src[a:b], "<sp2 helpers>", "exec"), ns)
    B = -(-T // 4)
    ns["_SP2"].update(k=2, B=B, N=2, r=rank, T=T, comm=comm, pend=None)
    return ns


def _worker(rank: int, port: int, q) -> None:
    os.environ.update({
        "NCCL_HOSTID": f"mhc2-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
        "NCCL_DEBUG": "WARN",
    })
    for k, e in (("NCCL_NSOCKS_PERTHREAD", "NSOCKS"), ("NCCL_SOCKET_NTHREADS", "NTHR")):
        if os.environ.get(e):
            os.environ[k] = os.environ[e]
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(dist.group.WORLD, torch.device("cuda:0"))
    sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
    from bench_mhc_halving import make
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    ns = _helpers(_patched_model_src(), comm, rank)
    B, S = ns["_SP2"]["B"], 2 * ns["_SP2"]["B"]
    assert 4 * B == T, "this harness uses T % 4 == 0"
    w = make(T, seed=5)                                   # the full-T state, identical on both ranks
    layer = types.SimpleNamespace(rms_norm_eps=1e-5, hc_eps=1e-6, mhc_post_mult_value=2.0, mhc_sinkhorn_iterations=20)
    norm = types.SimpleNamespace(weight=types.SimpleNamespace(data=w["norm"]), variance_epsilon=1e-5)
    g = torch.Generator(device="cuda").manual_seed(77 + rank)
    part0 = (torch.randn(T, H, generator=g, device="cuda") * 0.5).bfloat16()   # this rank's partial
    main = torch.cuda.current_stream()

    def standin(gathered):                                # cross-token, rank partial (sums to a full value)
        return ((torch.cumsum(gathered.float(), 0) * 1e-3 + gathered.float()) * 0.5).bfloat16()

    def shard_serial(t):
        return t[rank * S:(rank + 1) * S].contiguous()

    def shard_pipe(t):
        return torch.cat([t[rank * B:(rank + 1) * B], t[(rank + 2) * B:(rank + 3) * B]], 0)

    light = {"on": False}

    def run_serial(collect):
        res, post, comb = shard_serial(w["res"]), shard_serial(w["post"]), shard_serial(w["comb"])
        part, outs = part0, []
        for _ in range(2 * L):
            xs = torch.empty(S, H, dtype=torch.bfloat16, device="cuda")
            comm.reduce_scatter(xs, part, stream=main)
            if light["on"]:
                li = xs
            else:
                res, post, comb, li = mhc_fused_post_pre_tilelang(
                    xs, res, post, comb, w["fn"], w["scale"], w["base"], 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1,
                    w["norm"], 1e-5)
            gat = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
            comm.all_gather(gat, li.contiguous(), stream=main)
            if collect:
                outs.append(gat.clone())
            part = standin(gat)
        return outs

    def run_pipe(collect):
        res, post, comb = shard_pipe(w["res"]), shard_pipe(w["post"]), shard_pipe(w["comb"])
        part, outs = part0, []
        for _ in range(2 * L):
            x = ns["_sp2_reduce_scatter_begin"](part, False)
            if light["on"]:
                ns["_sp2_sync"]()
                gat = ns["_sp2_gather"](x, T)
            else:
                res, post, comb, gat = ns["_sp2_fused_post_pre_gather"](layer, x, res, post, comb, w["fn"],
                                                                       w["scale"], w["base"], norm, T)
            if collect:
                outs.append(gat.clone())
            part = standin(gat)
        return outs

    report = {}
    def say(m):
        print(f"[rank{rank}] {m}", flush=True)
    # ---- A. correctness (both ranks compute)
    say("A serial")
    ser = run_serial(True)
    torch.cuda.synchronize()
    say("A pipelined")
    pip = run_pipe(True)
    torch.cuda.synchronize()
    say(f"A done: bitwise={report['bitwise'] if 'bitwise' in report else None}")
    report["bitwise"] = all(torch.equal(a, b) for a, b in zip(ser, pip)) and len(ser) == len(pip) == 2 * L
    report["maxabs"] = max((a.float() - b.float()).abs().max().item() for a, b in zip(ser, pip))
    say(f"A result: serial vs pipelined bitwise={report['bitwise']} maxabs={report['maxabs']:.3e}")

    # ---- B. timing: rank 1 is an always-ready peer
    light["on"] = rank == 1

    def timed(fn):
        ts = []
        for i in range(ITERS + 1):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn(False)
            b.record()
            torch.cuda.synchronize()
            if i:
                ts.append(a.elapsed_time(b) / (2 * L))
        return statistics.median(ts)

    say("B serial")
    report["serial_ms_per_phase"] = timed(run_serial)
    say(f"B serial ms/phase {report['serial_ms_per_phase']:.2f}")
    say("B pipelined")
    report["pipe_ms_per_phase"] = timed(run_pipe)
    say(f"B pipelined ms/phase {report['pipe_ms_per_phase']:.2f}")
    say("B ops")

    # standalone costs on this rank (both ranks issue the same collectives in lockstep)
    part = part0
    xs = torch.empty(S, H, dtype=torch.bfloat16, device="cuda")
    gat = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")

    def t_op(fn, n=ITERS):
        ts = []
        for _ in range(n):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    report["rs_ms"] = t_op(lambda: comm.reduce_scatter(xs, part, stream=main))
    report["ag_ms"] = t_op(lambda: comm.all_gather(gat, xs, stream=main))
    say(f"RS {report['rs_ms']:.2f} ms AG {report['ag_ms']:.2f} ms")
    if rank == 0:
        res, post, comb = shard_pipe(w["res"]), shard_pipe(w["post"]), shard_pipe(w["comb"])
        x_sub, r_sub, p_sub, c_sub = xs[:B], res[:B], post[:B], comb[:B]
        mh = lambda: mhc_fused_post_pre_tilelang(x_sub, r_sub, p_sub, c_sub, w["fn"], w["scale"], w["base"],
                                                 1e-5, 1e-6, 1e-6, 2.0, 20, 1, 1, w["norm"], 1e-5)
        report["mhc_sub_ms_alone"] = t_op(mh)
    else:
        t_op(lambda: None)                                # same barrier count as rank 0
        report["mhc_sub_ms_alone"] = 0.0
    # contention: rank 0 runs one mHC sub-chunk while an RS of a 2B-row slice is in flight on the side stream
    cs = torch.cuda.Stream(priority=-1)
    half_in = part[: 2 * B].contiguous()
    half_out = torch.empty(B, H, dtype=torch.bfloat16, device="cuda")

    def overlapped():
        cs.wait_stream(main)
        with torch.cuda.stream(cs):
            comm.reduce_scatter(half_out, half_in, stream=cs)
        if rank == 0:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            mh()
            e1.record()
            main.wait_stream(cs)
            return e0, e1
        main.wait_stream(cs)
        return None

    ts_m, ts_tot = [], []
    for _ in range(ITERS):
        dist.barrier()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        r = overlapped()
        b.record()
        torch.cuda.synchronize()
        ts_tot.append(a.elapsed_time(b))
        if r is not None:
            ts_m.append(r[0].elapsed_time(r[1]))
    report["rs_half_ms_alone"] = t_op(lambda: comm.reduce_scatter(half_out, half_in, stream=main))
    report["mhc_sub_ms_with_rs"] = statistics.median(ts_m) if ts_m else 0.0
    report["mhc_sub_plus_rs_overlapped_ms"] = statistics.median(ts_tot)
    q.put((rank, report))
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    import queue
    import socket
    import time
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, port, q)) for r in (0, 1)]
    for p in ps:
        p.start()
    got, t0 = {}, time.time()
    while len(got) < 2:
        try:
            r, o = q.get(timeout=10)
            got[r] = o
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in ps) or time.time() - t0 > 3600:
                for p in ps:
                    p.kill()
                print(f"FAIL: worker exit codes {[p.exitcode for p in ps]} (or timeout)")
                return 2
    for p in ps:
        p.join(timeout=120)
    a, b = got[0], got[1]
    print(f"T={T} H={H} layers={L} (x2 phases), k=2 sub-chunks of {T // 4} rows; NCCL socket loopback "
          f"NSOCKS={os.environ.get('NSOCKS', 'default')} NTHR={os.environ.get('NTHR', 'default')}")
    print(f"A. serial (r16x SP) vs pipelined (SP2), both ranks computing: bitwise rank0={a['bitwise']} "
          f"rank1={b['bitwise']} (maxabs {max(a['maxabs'], b['maxabs']):.3e})")
    print(f"B. rank 0 (rank 1 = always-ready peer):  RS(56.6 MB) {a['rs_ms']:.2f} ms  AG {a['ag_ms']:.2f} ms  "
          f"RS of a 2B-row slice {a['rs_half_ms_alone']:.2f} ms  mHC sub-chunk alone {a['mhc_sub_ms_alone']:.3f} ms")
    print(f"   per phase: serial {a['serial_ms_per_phase']:.2f} ms, pipelined {a['pipe_ms_per_phase']:.2f} ms "
          f"(saving {a['serial_ms_per_phase'] - a['pipe_ms_per_phase']:.2f} ms/phase)")
    print(f"   contention: mHC sub-chunk {a['mhc_sub_ms_alone']:.3f} ms alone -> {a['mhc_sub_ms_with_rs']:.3f} ms with "
          f"an RS slice in flight (x{a['mhc_sub_ms_with_rs'] / max(a['mhc_sub_ms_alone'], 1e-9):.3f}); "
          f"both together {a['mhc_sub_plus_rs_overlapped_ms']:.2f} ms vs serial sum "
          f"{a['mhc_sub_ms_alone'] + a['rs_half_ms_alone']:.2f} ms")
    ok = a["bitwise"] and b["bitwise"]
    print("ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
