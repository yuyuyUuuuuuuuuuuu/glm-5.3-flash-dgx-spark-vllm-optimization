"""GLM53_DEC_AR1SHOT adversarial review tests (nodeC, two vLLM TP ranks = two processes on the one GB10, NCCL net
transport over loopback sockets, production's NCCL channel env NCCL_MIN/MAX_NCHANNELS=8, NCCL_CUMEM_ENABLE=0).
The plugin is installed BEFORE the process group exists (production order), so the agreement runs at communicator
construction and the first eligible all-reduce is inside a capture.

G  verify mode: the mismatch counter must count every compared element of every replay and eager call (production's
   verify hour reads it). Expected compared = replays x elements per replay + eager elements.
H  production's host-loop mix: a target-like graph (chained one-shot all-reduces, a side-stream all-reduce like the
   shared expert on the aux stream, a > MAX_KB all-reduce on the ring) and a drafter-like graph (own pools by default;
   AR1ADV_POOL=1 shares one pool, which is only valid when graphs replay in capture order - with this harness's order it
   fails identically with production's all-reduce in the graphs, AR1ADV_GRAPH_MODE=off: a harness control), replayed back to back WITHOUT a host sync while eager one-shot all-reduces (the target embedding) and an
   eager 1 MiB all-gather (lm_head logits) are issued on the same communicator between the replays; 24 iterations with
   M changing per iteration (partial acceptance) -> every output bit-identical to production's all-reduce (mode off).
K  NaN / inf payloads (informational: NCCL's __hadd2 vs torch's add may return different NaN payloads; a NaN hidden
   state is garbage either way, but verify mode would count it as 'differing').
Usage: GPU_RUN_SHM=2g tests/gpu_run.sh python3 tests/decode6/test_ar1shot_adv.py [verify-only]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
from test_ar1shot import H, bits, mk_pair  # noqa: E402

NCCL_ENV = {"NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo", "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1",
            "NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0", "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8",
            "NCCL_DEBUG": "WARN"}


def _init(rank, port, mode):
    os.environ.update(NCCL_ENV)
    os.environ.update({"NCCL_HOSTID": f"ar1adv-host{rank}", "GLM53_DEC_AR1SHOT": mode})
    torch.cuda.set_device(0)
    import glm53_ar1shot as A
    A.plugin_install()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.distributed.parallel_state import set_custom_all_reduce
    set_custom_all_reduce(False)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(2, 1)
    return A


def worker_g(rank, port, q):
    """G: verify-mode counter, first eligible call inside a capture."""
    out = []
    try:
        A = _init(rank, port, "verify")
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce as ar
        tp = get_tp_group()
        Ms = (5, 8, 1, 3)
        xs = {M: torch.zeros(M, H, dtype=torch.bfloat16, device="cuda") for M in Ms}
        g = torch.cuda.CUDAGraph()
        with tp.graph_capture() as gc:
            with torch.cuda.graph(g, stream=gc.stream):
                ys = {M: ar(xs[M] * 1) for M in Ms}
        per_replay = sum(M * H for M in Ms)
        R = 5
        for it in range(R):
            for M in Ms:
                a, b = mk_pair(M * H, 900 * it + M)
                xs[M].copy_((a if rank == 0 else b).view(M, H))
            g.replay()
        torch.cuda.synchronize()
        eager = 0
        for M in (2, 6, 64):
            a, b = mk_pair(M * H, 77 + M)
            ar((a if rank == 0 else b).view(M, H).contiguous())
            eager += M * H
        g.replay()                                   # one more replay after the eager calls
        torch.cuda.synchronize()
        expect = (R + 1) * per_replay + eager
        m = A.ST.mism.tolist() if A.ST.mism is not None else [None, None]
        ok = m[1] == expect and m[0] == 0
        out.append(("G", ok, f"rank {rank}: verify counter compared={m[1]} differing={m[0]} expected compared={expect} "
                             f"({R + 1} replays x {per_replay} + eager {eager}); counters {A.summary()}"))
        q.put((rank, out, None))
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, out, traceback.format_exc()))


def worker_hk(rank, port, q):
    """H: host-loop mix (no host sync between replays and eager collectives); K: NaN payloads."""
    out = []
    try:
        A = _init(rank, port, "1")
        gmode = os.environ.get("AR1ADV_GRAPH_MODE", "on")
        share_pool = os.environ.get("AR1ADV_POOL", "0") == "1"   # 1: graphs share one pool but replay out of capture order = a harness error (fails with production's all-reduce too)
        use_aux = os.environ.get("AR1ADV_AUX", "1") == "1"
        from vllm.distributed import (get_tp_group, tensor_model_parallel_all_gather as ag,
                                      tensor_model_parallel_all_reduce as ar)
        tp = get_tp_group()
        Mt = (5, 8, 6, 3)                       # target graphs, one per verify-block size
        Md = (1, 2)
        aux = torch.cuda.Stream()
        xt = {M: torch.zeros(M, H, dtype=torch.bfloat16, device="cuda") for M in Mt}
        xd = {M: torch.zeros(M, H, dtype=torch.bfloat16, device="cuda") for M in Md}
        big = {M: torch.zeros(80 * M, H, dtype=torch.bfloat16, device="cuda") for M in Mt}   # 3.2-5 MiB -> ring
        gt, yt, gd, yd = {}, {}, {}, {}
        pool = None
        A.ST.mode = gmode
        with tp.graph_capture() as gc:
            for M in Mt:
                gt[M] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gt[M], stream=gc.stream, pool=pool):
                    h = xt[M] * 1
                    for _ in range(6):                          # chained layers: o_proj AR -> MoE AR
                        h = ar(h * 1)
                        if use_aux:
                            ev = torch.cuda.Event()
                            ev.record()
                            with torch.cuda.stream(aux):            # shared expert on the aux stream
                                aux.wait_event(ev)
                                s = ar(h * 2)
                            torch.cuda.current_stream().wait_stream(aux)
                        else:
                            s = ar(h * 2)
                        h = ar(h * 1) + s
                    yb = ar(big[M] * 1)
                    yt[M] = (h, yb)
                pool = gt[M].pool() if share_pool else None
            for M in Md:
                gd[M] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gd[M], stream=gc.stream, pool=pool):
                    h = xd[M] * 1
                    for _ in range(4):
                        h = ar(h * 1)
                    yd[M] = h
        A.ST.mode = "on"
        c_cap = A.summary()

        def ref_target(x, bg):
            A.ST.mode = "off"
            h = x * 1
            for _ in range(6):
                h = ar(h * 1)
                s = ar(h * 2)
                h = ar(h * 1) + s
            r = (h, ar(bg * 1))
            A.ST.mode = "on"
            return r

        def ref_draft(x):
            A.ST.mode = "off"
            h = x * 1
            for _ in range(4):
                h = ar(h * 1)
            A.ST.mode = "on"
            return h

        ok = True
        nbad = 0
        it_log = []
        for it in range(24):
            Mt_i = Mt[it % len(Mt)]
            Md_i = Md[it % len(Md)]
            a, b = mk_pair(Mt_i * H, 5000 + it)
            xt[Mt_i].copy_((a if rank == 0 else b).view(Mt_i, H))
            big[Mt_i].copy_(torch.randn(80 * Mt_i, H, device="cuda", generator=torch.Generator("cuda").manual_seed(it + 31 * rank)).to(torch.bfloat16) * 0.01)
            a, b = mk_pair(Md_i * H, 6000 + it)
            xd[Md_i].copy_((a if rank == 0 else b).view(Md_i, H))
            ea, eb = mk_pair(Mt_i * H, 7000 + it)
            e_in = (ea if rank == 0 else eb).view(Mt_i, H).contiguous()
            lg = torch.randn(Mt_i, 512 * 128 // 1, device="cuda", generator=torch.Generator("cuda").manual_seed(it + 7 * rank)).to(torch.bfloat16)  # [M, 65536] bf16
            # host loop: drafter graph, eager embedding AR, target graph, eager lm_head AG - no host sync in between
            gd[Md_i].replay()
            e_out = ar(e_in)                                   # eager one-shot while the drafter graph is queued
            gt[Mt_i].replay()
            l_out = ag(lg, -1)                                 # eager all-gather while the target graph is queued
            torch.cuda.synchronize()
            # references (production all-reduce, eager), from the same inputs
            rh, rb = ref_target(xt[Mt_i], big[Mt_i])
            rd = ref_draft(xd[Md_i])
            A.ST.mode = "off"
            re_ = ar(e_in)
            rl = ag(lg, -1)
            A.ST.mode = "on"
            torch.cuda.synchronize()
            parts = {"h": (yt[Mt_i][0], rh), "big": (yt[Mt_i][1], rb), "draft": (yd[Md_i], rd), "emb": (e_out, re_),
                     "lm_ag": (l_out, rl)}
            bad = {k: int((bits(u) != bits(v)).sum()) for k, (u, v) in parts.items()}
            bad = {k: v for k, v in bad.items() if v}
            eq = not bad
            if not eq:
                nbad += 1
                it_log.append((it, Mt_i, Md_i, bad))
            ok &= eq
        msg = (f"rank {rank}: [graph mode {gmode}, shared pool {share_pool}, aux {use_aux}] {24} host-loop iterations (target M {Mt}, drafter M {Md}, aux-stream AR, ring AR > MAX_KB, "
               f"eager one-shot AR + eager 1 MiB AG between un-synced replays): mismatching iterations {nbad} {it_log}; "
               f"captured at capture time {c_cap['captured']} fallback {c_cap['fallback']}; counters {A.summary()}")
        out.append(("H", ok and c_cap["fallback"] == 0, msg[:1500]))
        # K: NaN / inf payloads
        n = 4096
        a = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        a[0::4] = float("inf"); b[0::4] = float("-inf")
        a[1::4] = float("nan"); b[1::4] = 1.0
        a[2::4] = 1.0; b[2::4] = float("nan")
        a[3::4] = float("inf"); b[3::4] = float("inf")
        x = (a if rank == 0 else b).view(1, n).contiguous()
        A.ST.mode = "off"
        r0 = ar(x)
        A.ST.mode = "on"
        r1 = ar(x)
        torch.cuda.synchronize()
        d = (bits(r0) != bits(r1))
        u0 = sorted(set(hex(v & 0xFFFF) for v in bits(r0).tolist()))
        u1 = sorted(set(hex(v & 0xFFFF) for v in bits(r1).tolist()))
        out.append(("K", True, f"rank {rank} (info): NaN/inf payloads: {int(d.sum())} of {n} bit patterns differ; "
                               f"NCCL {u0} vs one-shot {u1}; isnan equal: {torch.equal(r0.isnan(), r1.isnan())}"))
        q.put((rank, out, None))
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, out, traceback.format_exc()))


def run(target):
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=target, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = [q.get(timeout=3600) for _ in ps]
    for p in ps:
        p.join(60)
    allok = True
    for rank, out, err in sorted(res, key=lambda t: t[0]):
        for item in out:
            allok &= bool(item[1])
            print(f"{item[0]} {'OK ' if item[1] else 'FAIL'} {item[2]}", flush=True)
        if err:
            allok = False
            print(f"rank {rank} raised:\n{err}", flush=True)
    return allok


def main():
    ok = True
    if "h-only" not in sys.argv[1:]:
        ok &= run(worker_g)
    if "verify-only" not in sys.argv[1:]:
        ok &= run(worker_hk)
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
