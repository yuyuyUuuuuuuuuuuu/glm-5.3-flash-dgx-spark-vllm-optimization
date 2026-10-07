"""GLM53_DEC_AR1SHOT in production's ORDER (nodeC, two vLLM TP ranks = two processes on the one GB10, NCCL net transport
over loopback sockets, as tests/decode6/test_ar1shot.py):

E  the plugin is installed BEFORE the process group exists (integrate.plugin_register runs before init_device), so the
   TP-wide readiness agreement must happen inside CudaCommunicator.__init__ (eager, collective, before any capture);
   then the FIRST eligible all-reduce of the process is inside a CUDA-graph capture under vLLM's own
   GroupCoordinator.graph_capture context (production: full-graph capture of target + drafter, no eager decode-size
   all-reduce before it). Expected: agreement at init on both ranks for the TP group (and the 2-rank world group),
   every captured decode-size all-reduce served one-shot (fallback 0), a 2 MiB all-reduce inside the same capture on
   production's ring; replays with new inputs bit-identical to production's all-reduce (mode off, eager).
F  partial-acceptance-like mixed sizes in one graph: M = 1..8 rows (the verify block shrinks/grows with K and the
   number of requests: batch 1-2 -> 5..16 rows), fp32 and fp16 tensors too; replays equal.
Usage: GPU_RUN_SHM=2g tests/gpu_run.sh python3 tests/decode6/test_ar1shot_init.py
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


def worker(rank, port, q):
    os.environ.update({"NCCL_HOSTID": f"ar1i-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                       "NCCL_DEBUG": "WARN", "GLM53_DEC_AR1SHOT": "1"})
    out = []
    try:
        torch.cuda.set_device(0)
        import glm53_ar1shot as A
        A.plugin_install()                                   # as integrate.plugin_register: before the groups exist
        ok_e = A.ST.mode == "on"
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ensure_model_parallel_initialized(2, 1)
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
        tp = get_tp_group()
        dc = tp.device_communicator
        c_init = dict(A.COUNTERS)
        agreed_tp = A.ST.agreed.get(id(dc))
        ok_e &= agreed_tp is True and c_init.get("agree_at_init", 0) >= 1 and c_init["eager"] == 0 \
            and c_init["captured"] == 0
        msgs = [f"after init: agreed(tp)={agreed_tp} agree_at_init={c_init.get('agree_at_init', 0)} "
                f"eager={c_init['eager']} captured={c_init['captured']}"]
        # capture FIRST (no eager decode-size all-reduce before), inside vLLM's graph_capture context
        Ms = (5, 8, 1, 3, 6, 7, 2, 4)
        xs = {M: torch.zeros(M, H, dtype=torch.bfloat16, device="cuda") for M in Ms}
        x32 = torch.zeros(5, H, dtype=torch.float32, device="cuda")
        x16 = torch.zeros(5, H, dtype=torch.float16, device="cuda")
        big = torch.zeros(256, H, dtype=torch.bfloat16, device="cuda")      # 2 MiB > 512 KiB: production's ring
        gph = torch.cuda.CUDAGraph()
        with tp.graph_capture() as gc:
            with torch.cuda.graph(gph, stream=gc.stream):
                ys = {}
                for M in Ms:
                    y = tensor_model_parallel_all_reduce(xs[M] * 1)
                    ys[M] = tensor_model_parallel_all_reduce(y * 1)     # chained: the 2nd consumes the 1st
                y32 = tensor_model_parallel_all_reduce(x32 * 1)
                y16 = tensor_model_parallel_all_reduce(x16 * 1)
                ybig = tensor_model_parallel_all_reduce(big * 1)
        c_cap = dict(A.COUNTERS)
        ncap = c_cap["captured"] - c_init["captured"]
        ok_e &= ncap == 2 * len(Ms) + 2 and c_cap["fallback"] == 0
        msgs.append(f"capture: {ncap} one-shot calls captured (expect {2 * len(Ms) + 2}), fallback {c_cap['fallback']}")
        ok_f = True
        for it in range(6):
            for M in Ms:
                a, b = mk_pair(M * H, 300 * it + M)
                xs[M].copy_((a if rank == 0 else b).view(M, H))
            a, b = mk_pair(5 * H, 7000 + it, torch.float32)
            x32.copy_((a if rank == 0 else b).view(5, H))
            a, b = mk_pair(5 * H, 8000 + it, torch.float16)
            x16.copy_((a if rank == 0 else b).view(5, H))
            big.copy_(torch.randn(256, H, device="cuda").to(torch.bfloat16))
            gph.replay()
            torch.cuda.synchronize()
            A.ST.mode = "off"
            for M in Ms:
                r = tensor_model_parallel_all_reduce(tensor_model_parallel_all_reduce(xs[M] * 1) * 1)
                ok_f &= torch.equal(bits(r), bits(ys[M]))
            ok_f &= torch.equal(bits(tensor_model_parallel_all_reduce(x32 * 1)), bits(y32))
            ok_f &= torch.equal(bits(tensor_model_parallel_all_reduce(x16 * 1)), bits(y16))
            ok_f &= torch.equal(bits(tensor_model_parallel_all_reduce(big * 1)), bits(ybig))
            A.ST.mode = "on"
            torch.cuda.synchronize()
        out.append(("E", ok_e, f"rank {rank}: " + "; ".join(msgs)))
        out.append(("F", ok_f, f"rank {rank}: 6 replays x ({len(Ms)} bf16 sizes chained x2 + fp32 + fp16 + 2 MiB ring) "
                               f"== production's all-reduce bitwise: {ok_f}; counters {A.summary()}"))
        q.put((rank, out, None))
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, out, traceback.format_exc()))


def main():
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join(60)
    allok = True
    for rank, out, err in sorted(res, key=lambda t: t[0]):
        for item in out:
            allok &= bool(item[1])
            print(f"{item[0]} {'OK ' if item[1] else 'FAIL'} {item[2]}")
        if err:
            allok = False
            print(f"rank {rank} raised:\n{err}")
    print("ALL OK" if allok else "FAILED")


if __name__ == "__main__":
    main()
