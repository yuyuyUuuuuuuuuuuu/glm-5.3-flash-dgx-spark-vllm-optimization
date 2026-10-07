"""decode6: the GPU idle gap in front of every graph-captured NCCL kernel (R15 trace: p50 3.9 us before each of the
101 in-graph AllReduce kernels = 0.52 ms/step, vs 0.1 us before every other kernel) - where it comes from and which
NCCL settings remove it. nodeC: two vLLM TP ranks = two processes on the one GB10, NCCL net transport over loopback
sockets (NCCL_HOSTID per rank, as tests/decode6/test_ar1shot.py). The transport is TCP, so the collective's own
duration is meaningless here; the gap BEFORE the NCCL kernel (and after it) is a property of the launch / graph
topology, measured with torch.profiler inside a replayed CUDA graph of
    20 x [bf16 elementwise (main stream) -> collective (pynccl, the production decode path) -> elementwise]
Variants are passed as env per run: python3 mb_nccl_gap.py "NCCL_MEM_SYNC_DOMAIN=0" ...
Usage: GPU_RUN_SHM=2g tests/gpu_run.sh python3 tests/decode6/mb_nccl_gap.py [VAR=VAL,VAR2=VAL2 ...]
(each argument is one variant; "base" = no extra env)
"""
import os
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

H = 4096


def worker(rank, port, q, extra, kind):
    os.environ.update({"NCCL_HOSTID": f"gap-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                       "NCCL_DEBUG": "WARN"})
    os.environ.update(extra)
    try:
        torch.cuda.set_device(0)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ensure_model_parallel_initialized(2, 1)
        from vllm.distributed import get_tp_group
        pc = get_tp_group().device_communicator.pynccl_comm
        M = 5
        x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
        g2 = torch.empty(2, M * H, device="cuda", dtype=torch.bfloat16)

        def body():
            y = x
            for _ in range(20):
                y = y * 1.0001
                if kind == "ar":
                    y = pc.all_reduce(y)
                else:
                    pc.all_gather(g2, y.view(-1))
                    y = torch.add(g2[0], g2[1]).view(M, H)
                y = y + 0.5
            return y

        for _ in range(3):
            body()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            body()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=s):
                body()
        torch.cuda.synchronize()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
        d = tempfile.mkdtemp()
        f = os.path.join(d, "t.json")
        prof.export_chrome_trace(f)
        import json
        ev = json.load(open(f))["traceEvents"]
        k = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
        pre, post, dur, small_gap = [], [], [], []
        for i in range(1, len(k) - 1):
            g = k[i]["ts"] - (k[i - 1]["ts"] + k[i - 1]["dur"])
            if "nccl" in k[i]["name"]:
                pre.append(g)
                dur.append(k[i]["dur"])
                post.append(k[i + 1]["ts"] - (k[i]["ts"] + k[i]["dur"]))
            elif "nccl" not in k[i - 1]["name"]:
                small_gap.append(g)
        med = statistics.median
        q.put((rank, f"rank {rank} {kind} {extra or 'base'}: gap before nccl p50 {med(pre):.2f} mean "
                     f"{statistics.mean(pre):.2f} us | after nccl p50 {med(post):.2f} | between non-nccl kernels p50 "
                     f"{med(small_gap):.2f} | nccl kernels {len(pre)} ({k[[i for i in range(len(k)) if 'nccl' in k[i]['name']][0]]['name'][:48]})"))
    except Exception as exc:  # noqa: BLE001
        import traceback
        q.put((rank, f"rank {rank} FAILED {exc!r}\n{traceback.format_exc()}"))


def run(extra, kind, port):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, port, q, extra, kind)) for r in range(2)]
    for p in ps:
        p.start()
    res = sorted(q.get(timeout=600) for _ in ps)
    for p in ps:
        p.join(timeout=60)
    for _, m in res:
        print(m, flush=True)


if __name__ == "__main__":
    variants = sys.argv[1:] or ["base"]
    port = 29600
    for v in variants:
        extra = {} if v == "base" else dict(kv.split("=", 1) for kv in v.split(","))
        for kind in ("ar", "ag"):
            port += 1
            run(extra, kind, port)
    print("done")
